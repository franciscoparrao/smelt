//! Principal Component Analysis as a Transformer.

use super::Transformer;
use crate::{Result, SmeltError};
use ndarray::Array2;

/// PCA dimensionality reduction via eigendecomposition of the covariance matrix.
///
/// # Examples
///
/// ```
/// use smelt_ml::preprocess::{Transformer, PCA};
/// use ndarray::array;
///
/// let mut pca = PCA::new(2);
/// let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
/// let reduced = pca.fit_transform(&data).unwrap();
/// assert_eq!(reduced.ncols(), 2);
/// ```
#[derive(Clone)]
pub struct PCA {
    n_components: usize,
    means: Option<Vec<f64>>,
    components: Option<Array2<f64>>, // n_components × n_features
}

impl PCA {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            means: None,
            components: None,
        }
    }
}

impl Transformer for PCA {
    fn id(&self) -> &str {
        "pca"
    }

    fn fit(&mut self, features: &Array2<f64>) -> Result<()> {
        let n = features.nrows() as f64;
        let p = features.ncols();
        let nc = self.n_components.min(p);

        // Compute means
        let means: Vec<f64> = (0..p).map(|j| features.column(j).sum() / n).collect();

        // Center data
        let mut centered = features.clone();
        for i in 0..features.nrows() {
            for j in 0..p {
                centered[[i, j]] -= means[j];
            }
        }

        // Covariance matrix (p × p)
        let cov = centered.t().dot(&centered);
        // Scale by 1/n (not 1/(n-1) — matching sklearn default)
        let cov = cov / n;

        // Power iteration to find top nc eigenvectors
        let components = power_iteration_deflation(&cov, nc);

        self.means = Some(means);
        self.components = Some(components);
        Ok(())
    }

    fn transform(&self, features: &Array2<f64>) -> Result<Array2<f64>> {
        let means = self.means.as_ref().ok_or(SmeltError::NotTrained)?;
        let components = self.components.as_ref().ok_or(SmeltError::NotTrained)?;

        if features.ncols() != means.len() {
            return Err(SmeltError::DimensionMismatch {
                expected: means.len(),
                got: features.ncols(),
            });
        }

        // Center and project
        let mut centered = features.clone();
        for i in 0..features.nrows() {
            for j in 0..features.ncols() {
                centered[[i, j]] -= means[j];
            }
        }

        Ok(centered.dot(&components.t()))
    }

    fn transform_names(&self, _names: &[String]) -> Result<Vec<String>> {
        let nc = self
            .components
            .as_ref()
            .map_or(self.n_components, |c| c.nrows());
        Ok((0..nc).map(|i| format!("PC{}", i + 1)).collect())
    }

    fn clone_box(&self) -> Box<dyn Transformer> {
        Box::new(self.clone())
    }
}

/// Find top-k eigenvectors via power iteration with deflation.
fn power_iteration_deflation(matrix: &Array2<f64>, k: usize) -> Array2<f64> {
    let p = matrix.nrows();
    let mut mat = matrix.clone();
    let mut eigenvectors = Array2::zeros((k, p));

    for component in 0..k {
        // Power iteration for dominant eigenvector
        let mut v = ndarray::Array1::from_elem(p, 1.0 / (p as f64).sqrt());

        for _ in 0..200 {
            let mv = mat.dot(&v);
            let norm = mv.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if norm < 1e-15 {
                break;
            }
            let new_v = &mv / norm;

            // Check convergence
            let diff: f64 = v
                .iter()
                .zip(new_v.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>();
            v = new_v;
            if diff < 1e-12 {
                break;
            }
        }

        eigenvectors.row_mut(component).assign(&v);

        // Deflate: remove this component from the matrix
        let eigenvalue = v.dot(&mat.dot(&v));
        let outer = {
            let mut o = Array2::zeros((p, p));
            for i in 0..p {
                for j in 0..p {
                    o[[i, j]] = eigenvalue * v[i] * v[j];
                }
            }
            o
        };
        mat = mat - outer;
    }

    eigenvectors
}
