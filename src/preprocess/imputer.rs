//! Missing value imputation.

use ndarray::Array2;
use crate::{SmeltError, Result};
use super::Transformer;

/// Strategy for filling missing values.
#[derive(Clone)]
pub enum ImputeStrategy {
    /// Replace NaN with column mean.
    Mean,
    /// Replace NaN with column median.
    Median,
    /// Replace NaN with a fixed constant.
    Constant(f64),
}

/// Fills NaN values using a specified strategy.
///
/// # Examples
///
/// ```
/// use smelt::preprocess::{Transformer, Imputer, ImputeStrategy};
/// use ndarray::array;
///
/// let mut imp = Imputer::new(ImputeStrategy::Mean);
/// let data = array![[1.0, f64::NAN], [3.0, 4.0]];
/// let filled = imp.fit_transform(&data).unwrap();
/// // NaN replaced with column mean (4.0)
/// ```
#[derive(Clone)]
pub struct Imputer {
    strategy: ImputeStrategy,
    fill_values: Option<Vec<f64>>,
}

impl Imputer {
    pub fn new(strategy: ImputeStrategy) -> Self {
        Self { strategy, fill_values: None }
    }

    pub fn mean() -> Self { Self::new(ImputeStrategy::Mean) }
    pub fn median() -> Self { Self::new(ImputeStrategy::Median) }
    pub fn constant(value: f64) -> Self { Self::new(ImputeStrategy::Constant(value)) }
}

fn compute_median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 0 {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    } else {
        values[n / 2]
    }
}

impl Transformer for Imputer {
    fn id(&self) -> &str { "imputer" }

    fn fit(&mut self, features: &Array2<f64>) -> Result<()> {
        let ncols = features.ncols();
        let mut fill_values = vec![0.0; ncols];

        for j in 0..ncols {
            let col = features.column(j);
            let non_nan: Vec<f64> = col.iter().copied().filter(|x| !x.is_nan()).collect();

            fill_values[j] = match &self.strategy {
                ImputeStrategy::Mean => {
                    if non_nan.is_empty() {
                        0.0
                    } else {
                        non_nan.iter().sum::<f64>() / non_nan.len() as f64
                    }
                }
                ImputeStrategy::Median => {
                    let mut vals = non_nan;
                    compute_median(&mut vals)
                }
                ImputeStrategy::Constant(v) => *v,
            };
        }

        self.fill_values = Some(fill_values);
        Ok(())
    }

    fn transform(&self, features: &Array2<f64>) -> Result<Array2<f64>> {
        let fill = self.fill_values.as_ref().ok_or(SmeltError::NotTrained)?;
        if features.ncols() != fill.len() {
            return Err(SmeltError::DimensionMismatch {
                expected: fill.len(),
                got: features.ncols(),
            });
        }
        let mut result = features.clone();
        for j in 0..features.ncols() {
            for i in 0..features.nrows() {
                if result[[i, j]].is_nan() {
                    result[[i, j]] = fill[j];
                }
            }
        }
        Ok(result)
    }

    fn clone_box(&self) -> Box<dyn Transformer> { Box::new(self.clone()) }
}
