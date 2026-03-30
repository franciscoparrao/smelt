//! Input validation utilities for tasks and predictions.

use ndarray::Array2;
use crate::{SmeltError, Result};

/// Check that a feature matrix contains no NaN values.
/// Returns an error with the location of the first NaN found.
pub fn check_no_nan(features: &Array2<f64>) -> Result<()> {
    for i in 0..features.nrows() {
        for j in 0..features.ncols() {
            if features[[i, j]].is_nan() {
                return Err(SmeltError::InvalidParameter(
                    format!("NaN found at row {}, column {}", i, j)
                ));
            }
        }
    }
    Ok(())
}

/// Check that prediction features have the expected number of columns.
pub fn check_n_features(features: &Array2<f64>, expected: usize) -> Result<()> {
    if features.ncols() != expected {
        return Err(SmeltError::DimensionMismatch {
            expected,
            got: features.ncols(),
        });
    }
    Ok(())
}

/// Check that a feature matrix is non-empty.
pub fn check_non_empty(features: &Array2<f64>) -> Result<()> {
    if features.nrows() == 0 {
        return Err(SmeltError::EmptyDataset);
    }
    Ok(())
}
