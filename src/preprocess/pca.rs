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
    /// Create a PCA transformer that reduces to `n_components` dimensions.
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

        // Covariance matrix (p × p), biased/population scaling (1/n, not
        // sklearn's 1/(n-1)). This only rescales the eigenVALUES by a
        // constant factor -- eigenvectors (and hence `transform`'s output)
        // are identical either way -- so it doesn't affect this transformer,
        // which never reports explained variance.
        let cov = centered.t().dot(&centered);
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

/// Finds a unit vector orthogonal to `eigenvectors[0..component]` via
/// Gram-Schmidt on the standard basis, trying each `e_i` in turn until one
/// isn't (nearly) in the span of the already-found components.
fn orthogonal_complement_vector(
    eigenvectors: &Array2<f64>,
    component: usize,
    p: usize,
) -> ndarray::Array1<f64> {
    for candidate in 0..p {
        let mut e = ndarray::Array1::zeros(p);
        e[candidate] = 1.0;
        for prev in 0..component {
            let proj = eigenvectors.row(prev).dot(&e);
            for i in 0..p {
                e[i] -= proj * eigenvectors[[prev, i]];
            }
        }
        let norm = e.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm > 1e-6 {
            return e / norm;
        }
    }
    // Unreachable for component < p (the standard basis spans R^p, so some
    // e_i must have a nonzero component orthogonal to `component < p`
    // already-found unit vectors) -- fall back to e_0 defensively.
    let mut e = ndarray::Array1::zeros(p);
    e[0] = 1.0;
    e
}

/// Find top-k eigenvectors via power iteration with deflation.
fn power_iteration_deflation(matrix: &Array2<f64>, k: usize) -> Array2<f64> {
    use rand::Rng;
    use rand::SeedableRng;

    let p = matrix.nrows();
    let mut mat = matrix.clone();
    let mut eigenvectors = Array2::zeros((k, p));
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let orig_norm = matrix.iter().map(|&x| x * x).sum::<f64>().sqrt().max(1e-300);

    for component in 0..k {
        // If deflation has already removed (numerically) all of the
        // remaining variance -- e.g. the covariance matrix was exactly
        // rank-deficient to begin with, such as two perfectly
        // anticorrelated features -- there's no meaningful "dominant
        // direction" left to find: the residual matrix is pure floating-
        // point noise, and running power iteration on it would normalize
        // that noise into an arbitrary (not necessarily orthogonal, not
        // reproducible) direction. Complete the basis deterministically via
        // Gram-Schmidt instead of iterating on noise.
        let mat_norm = mat.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let v = if mat_norm / orig_norm < 1e-10 {
            orthogonal_complement_vector(&eigenvectors, component, p)
        } else {
            // Random start vector, not a fixed one (e.g. the constant
            // all-ones vector this used to start from): a fixed start can
            // itself BE an eigenvector of `mat` for some inputs (e.g. two
            // standardized, anticorrelated features have the constant
            // vector as an eigenvector), converging power iteration in a
            // single step to whichever eigenvector the fixed start happens
            // to align with -- not necessarily the dominant one, since the
            // convergence check only verifies a fixed point, not dominance.
            // A random start overlaps the true dominant eigenvector's
            // direction with probability 1.
            let mut v = ndarray::Array1::from_shape_fn(p, |_| rng.random::<f64>() - 0.5);
            let norm0 = v.iter().map(|&x| x * x).sum::<f64>().sqrt().max(1e-15);
            v /= norm0;

            for _ in 0..200 {
                let mut mv = mat.dot(&v);
                // Re-orthogonalize against previously extracted components
                // each iteration, on top of deflation, to counter numerical
                // drift that can otherwise accumulate across successive
                // deflations.
                for prev in 0..component {
                    let proj = eigenvectors.row(prev).dot(&mv);
                    for i in 0..p {
                        mv[i] -= proj * eigenvectors[[prev, i]];
                    }
                }
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
            v
        };

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

#[cfg(test)]
mod tests {
    use super::*;

    /// Two eigenvectors point in the "same direction" if they match up to
    /// the sign ambiguity every eigenvector has (`v` and `-v` are both
    /// valid).
    fn same_direction(a: &[f64], b: &[f64], tol: f64) -> bool {
        let same = a.iter().zip(b).all(|(&x, &y)| (x - y).abs() < tol);
        let opposite = a.iter().zip(b).all(|(&x, &y)| (x + y).abs() < tol);
        same || opposite
    }

    /// Golden test against `sklearn.decomposition.PCA` (1.8.0) on a fixed,
    /// exactly-reproducible-in-Rust dataset: two features built from the
    /// SAME sin/cos formula in both languages (`sin(0.1i)` and its near
    /// negation `-sin(0.1i) + 0.05*cos(0.37i)`), avoiding any dependence on
    /// numpy's specific PRNG.
    #[test]
    fn golden_matches_sklearn_on_near_anticorrelated_features() {
        let n = 100;
        let mut feats = Vec::with_capacity(n * 2);
        for i in 0..n {
            let x1 = (i as f64 * 0.1).sin();
            let x2 = -x1 + 0.05 * (i as f64 * 0.37).cos();
            feats.push(x1);
            feats.push(x2);
        }
        let data = Array2::from_shape_vec((n, 2), feats).unwrap();

        let mut pca = PCA::new(2);
        pca.fit(&data).unwrap();
        let components = pca.components.as_ref().unwrap();

        // sklearn 1.8.0 components_ on this exact dataset.
        let expected_pc1 = [-0.7066938203811517, 0.7075195009574594];
        let expected_pc2 = [0.7075195009574594, 0.7066938203811517];

        assert!(
            same_direction(&components.row(0).to_vec(), &expected_pc1, 1e-4),
            "PC1 {:?} should match sklearn's dominant eigenvector {:?} (up to sign)",
            components.row(0),
            expected_pc1
        );
        assert!(
            same_direction(&components.row(1).to_vec(), &expected_pc2, 1e-4),
            "PC2 {:?} should match sklearn's second eigenvector {:?} (up to sign)",
            components.row(1),
            expected_pc2
        );

        // Projected coordinates (up to a possible sign flip per axis) should
        // also match sklearn's transform() on the same data.
        let projected = pca.transform(&data).unwrap();
        let expected_row0 = [0.29965671987715764, 0.03574682420209543];
        let expected_row50 = [1.6536408374109541, 0.032818189860606035];
        for (row_idx, expected) in [(0, expected_row0), (50, expected_row50)] {
            for col in 0..2 {
                let got = projected[[row_idx, col]];
                let exp = expected[col];
                assert!(
                    (got - exp).abs() < 1e-3 || (got + exp).abs() < 1e-3,
                    "row {row_idx} col {col}: got {got}, expected ±{exp}"
                );
            }
        }
    }

    /// Regression test for the exact failure shape in the audit (M12/N12):
    /// with two EXACTLY anticorrelated features (`x2 = -x1`), the constant
    /// vector `[1,1]/sqrt(2)` is an *exact* eigenvector of the covariance
    /// matrix -- but of the zero-variance direction, not the dominant one
    /// (`mat.dot([1,1]/sqrt2) = [0,0]` exactly). The old deterministic
    /// constant-vector power-iteration start hit the `norm < 1e-15` early
    /// return on its very first multiply and returned that same zero-variance
    /// direction as "PC1", in silence, with zero iterations of actual power
    /// iteration performed. A random start (this fix) has probability 1 of
    /// overlapping the true dominant eigenvector's direction instead.
    #[test]
    fn regression_exact_anticorrelation_finds_dominant_not_zero_variance_direction() {
        let n = 100;
        let mut feats = Vec::with_capacity(n * 2);
        for i in 0..n {
            let x1 = (i as f64 * 0.1).sin();
            feats.push(x1);
            feats.push(-x1); // exact anticorrelation, no noise
        }
        let data = Array2::from_shape_vec((n, 2), feats).unwrap();

        let mut pca = PCA::new(2);
        pca.fit(&data).unwrap();
        let components = pca.components.as_ref().unwrap();

        // sklearn 1.8.0: PC1 = [1,-1]/sqrt(2) (100% of variance), PC2 =
        // [1,1]/sqrt(2) (0% of variance -- the direction the old bug
        // returned as PC1).
        let expected_pc1 = [0.7071067811865475, -0.7071067811865475];
        assert!(
            same_direction(&components.row(0).to_vec(), &expected_pc1, 1e-6),
            "PC1 {:?} should be the dominant [1,-1]/sqrt(2) direction (100% of variance), \
             not the zero-variance constant direction the historical bug converged to",
            components.row(0)
        );

        let projected = pca.transform(&data).unwrap();
        // All variance is on PC1; PC2's projection must be ~0 for every row.
        for row in 0..n {
            assert!(
                projected[[row, 1]].abs() < 1e-6,
                "PC2 projection should be ~0 (zero-variance direction), got {} at row {row}",
                projected[[row, 1]]
            );
        }
        let expected_pc1_row0 = -0.2637140272360043;
        let got0 = projected[[0, 0]];
        assert!(
            (got0 - expected_pc1_row0).abs() < 1e-6 || (got0 + expected_pc1_row0).abs() < 1e-6,
            "PC1 projection row 0: got {got0}, expected ±{expected_pc1_row0}"
        );
    }
}
