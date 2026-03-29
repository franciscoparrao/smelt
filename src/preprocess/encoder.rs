//! One-hot encoding for categorical features.

use ndarray::Array2;
use crate::{SmeltError, Result};
use super::Transformer;

/// Encodes specified columns as binary indicator columns.
///
/// Each encoded column with K unique values produces K binary columns.
/// Non-encoded columns pass through unchanged.
///
/// # Examples
///
/// ```
/// use smelt_ml::preprocess::{Transformer, OneHotEncoder};
/// use ndarray::array;
///
/// // Column 0 has categories {0, 1, 2}
/// let mut enc = OneHotEncoder::new(vec![0]);
/// let data = array![[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]];
/// let encoded = enc.fit_transform(&data).unwrap();
/// assert_eq!(encoded.ncols(), 4); // 3 binary + 1 passthrough
/// ```
#[derive(Clone)]
pub struct OneHotEncoder {
    columns: Vec<usize>,
    categories: Option<Vec<(usize, Vec<f64>)>>,
    n_features_in: Option<usize>,
}

impl OneHotEncoder {
    pub fn new(columns: Vec<usize>) -> Self {
        Self { columns, categories: None, n_features_in: None }
    }
}

impl Transformer for OneHotEncoder {
    fn id(&self) -> &str { "one_hot_encoder" }

    fn fit(&mut self, features: &Array2<f64>) -> Result<()> {
        self.n_features_in = Some(features.ncols());
        let mut categories = Vec::new();

        for &col in &self.columns {
            if col >= features.ncols() {
                return Err(SmeltError::InvalidParameter(
                    format!("column index {} out of bounds ({})", col, features.ncols()),
                ));
            }
            let mut unique: Vec<f64> = features.column(col).iter().copied().collect();
            unique.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            unique.dedup();
            categories.push((col, unique));
        }

        self.categories = Some(categories);
        Ok(())
    }

    fn transform(&self, features: &Array2<f64>) -> Result<Array2<f64>> {
        let categories = self.categories.as_ref().ok_or(SmeltError::NotTrained)?;
        let n_in = self.n_features_in.unwrap();
        if features.ncols() != n_in {
            return Err(SmeltError::DimensionMismatch {
                expected: n_in,
                got: features.ncols(),
            });
        }

        // Compute output column count
        let n_out: usize = (0..n_in)
            .map(|j| {
                if let Some((_, cats)) = categories.iter().find(|(c, _)| *c == j) {
                    cats.len()
                } else {
                    1
                }
            })
            .sum();

        let nrows = features.nrows();
        let mut result = Array2::zeros((nrows, n_out));
        let mut out_col = 0;

        for j in 0..n_in {
            if let Some((_, cats)) = categories.iter().find(|(c, _)| *c == j) {
                // One-hot encode this column
                for i in 0..nrows {
                    let val = features[[i, j]];
                    if let Some(pos) = cats.iter().position(|&c| (c - val).abs() < f64::EPSILON) {
                        result[[i, out_col + pos]] = 1.0;
                    }
                    // Unseen category: all zeros (default)
                }
                out_col += cats.len();
            } else {
                // Pass through
                for i in 0..nrows {
                    result[[i, out_col]] = features[[i, j]];
                }
                out_col += 1;
            }
        }

        Ok(result)
    }

    fn transform_names(&self, names: &[String]) -> Result<Vec<String>> {
        let categories = self.categories.as_ref().ok_or(SmeltError::NotTrained)?;
        let mut result = Vec::new();

        for (j, name) in names.iter().enumerate() {
            if let Some((_, cats)) = categories.iter().find(|(c, _)| *c == j) {
                for cat in cats {
                    // Format integer-like values cleanly
                    if *cat == cat.trunc() {
                        result.push(format!("{}_{}", name, *cat as i64));
                    } else {
                        result.push(format!("{}_{}", name, cat));
                    }
                }
            } else {
                result.push(name.clone());
            }
        }

        Ok(result)
    }

    fn clone_box(&self) -> Box<dyn Transformer> { Box::new(self.clone()) }
}
