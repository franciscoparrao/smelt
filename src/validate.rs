//! Input validation utilities for tasks and predictions.

use crate::{Result, SmeltError};
use ndarray::Array2;

/// Check that a feature matrix contains no NaN values.
/// Returns an error with the location of the first NaN found.
pub fn check_no_nan(features: &Array2<f64>) -> Result<()> {
    for i in 0..features.nrows() {
        for j in 0..features.ncols() {
            if features[[i, j]].is_nan() {
                return Err(SmeltError::InvalidParameter(format!(
                    "NaN found at row {}, column {}",
                    i, j
                )));
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

/// Check that every coordinate pair is finite (no NaN/±inf).
///
/// The spatial learners compute pairwise distances from these: a single
/// non-finite coordinate poisons every distance it touches, which either
/// spreads NaN through all predictions (kriging weights) or breaks the
/// total order `slice::sort` requires (GeoXGBoost's neighbour ranking —
/// a panic on Rust ≥ 1.81). Features get this guard via [`check_no_nan`];
/// coordinates need their own.
pub fn check_coords_finite(coords: &[(f64, f64)]) -> Result<()> {
    for (i, &(x, y)) in coords.iter().enumerate() {
        if !x.is_finite() || !y.is_finite() {
            return Err(SmeltError::InvalidParameter(format!(
                "non-finite coordinate at index {i}: ({x}, {y}) — every sample needs a finite georeference"
            )));
        }
    }
    Ok(())
}
