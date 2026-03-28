//! Feature scaling transformers: StandardScaler and MinMaxScaler.

use ndarray::Array2;
use crate::{SmeltError, Result};
use super::Transformer;

/// Standardizes features to zero mean and unit variance.
///
/// Per column: `x_scaled = (x - mean) / std`.
///
/// # Examples
///
/// ```
/// use smelt::preprocess::{Transformer, StandardScaler};
/// use ndarray::array;
///
/// let mut scaler = StandardScaler::new();
/// let data = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
/// let scaled = scaler.fit_transform(&data).unwrap();
/// ```
#[derive(Clone)]
pub struct StandardScaler {
    means: Option<Vec<f64>>,
    stds: Option<Vec<f64>>,
}

impl StandardScaler {
    pub fn new() -> Self { Self { means: None, stds: None } }
}

impl Default for StandardScaler {
    fn default() -> Self { Self::new() }
}

impl Transformer for StandardScaler {
    fn id(&self) -> &str { "standard_scaler" }

    fn fit(&mut self, features: &Array2<f64>) -> Result<()> {
        let n = features.nrows() as f64;
        let ncols = features.ncols();
        let mut means = vec![0.0; ncols];
        let mut stds = vec![0.0; ncols];

        for j in 0..ncols {
            let col = features.column(j);
            let mean = col.sum() / n;
            let variance = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
            means[j] = mean;
            stds[j] = if variance > 0.0 { variance.sqrt() } else { 1.0 };
        }

        self.means = Some(means);
        self.stds = Some(stds);
        Ok(())
    }

    fn transform(&self, features: &Array2<f64>) -> Result<Array2<f64>> {
        let means = self.means.as_ref().ok_or(SmeltError::NotTrained)?;
        let stds = self.stds.as_ref().ok_or(SmeltError::NotTrained)?;
        if features.ncols() != means.len() {
            return Err(SmeltError::DimensionMismatch {
                expected: means.len(),
                got: features.ncols(),
            });
        }
        let mut result = features.clone();
        for j in 0..features.ncols() {
            for i in 0..features.nrows() {
                result[[i, j]] = (features[[i, j]] - means[j]) / stds[j];
            }
        }
        Ok(result)
    }

    fn clone_box(&self) -> Box<dyn Transformer> { Box::new(self.clone()) }
}

/// Scales features to [0, 1] range.
///
/// Per column: `x_scaled = (x - min) / (max - min)`.
///
/// # Examples
///
/// ```
/// use smelt::preprocess::{Transformer, MinMaxScaler};
/// use ndarray::array;
///
/// let mut scaler = MinMaxScaler::new();
/// let data = array![[1.0], [5.0], [10.0]];
/// let scaled = scaler.fit_transform(&data).unwrap();
/// // scaled: [[0.0], [0.444], [1.0]]
/// ```
#[derive(Clone)]
pub struct MinMaxScaler {
    mins: Option<Vec<f64>>,
    ranges: Option<Vec<f64>>,
}

impl MinMaxScaler {
    pub fn new() -> Self { Self { mins: None, ranges: None } }
}

impl Default for MinMaxScaler {
    fn default() -> Self { Self::new() }
}

impl Transformer for MinMaxScaler {
    fn id(&self) -> &str { "min_max_scaler" }

    fn fit(&mut self, features: &Array2<f64>) -> Result<()> {
        let ncols = features.ncols();
        let mut mins = vec![0.0; ncols];
        let mut ranges = vec![0.0; ncols];

        for j in 0..ncols {
            let col = features.column(j);
            let min = col.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            mins[j] = min;
            ranges[j] = if (max - min).abs() > f64::EPSILON { max - min } else { 1.0 };
        }

        self.mins = Some(mins);
        self.ranges = Some(ranges);
        Ok(())
    }

    fn transform(&self, features: &Array2<f64>) -> Result<Array2<f64>> {
        let mins = self.mins.as_ref().ok_or(SmeltError::NotTrained)?;
        let ranges = self.ranges.as_ref().ok_or(SmeltError::NotTrained)?;
        if features.ncols() != mins.len() {
            return Err(SmeltError::DimensionMismatch {
                expected: mins.len(),
                got: features.ncols(),
            });
        }
        let mut result = features.clone();
        for j in 0..features.ncols() {
            for i in 0..features.nrows() {
                result[[i, j]] = (features[[i, j]] - mins[j]) / ranges[j];
            }
        }
        Ok(result)
    }

    fn clone_box(&self) -> Box<dyn Transformer> { Box::new(self.clone()) }
}
