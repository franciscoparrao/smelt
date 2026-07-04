//! Linear Regression (OLS) via normal equation.
//!
//! Solves w = (X'X)^{-1} X'y using Gaussian elimination with partial pivoting.

use crate::learner::{Learner, TrainedModel};
use crate::prediction::Prediction;
use crate::task::{RegressionTask, Task};
use crate::{Result, SmeltError};
use ndarray::{Array1, Array2};

/// Ordinary Least Squares linear regression.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use smelt_ml::learner::LinearRegression;
/// use ndarray::array;
///
/// let features = array![[1.0], [2.0], [3.0], [4.0]];
/// let target = vec![2.0, 4.0, 6.0, 8.0]; // y = 2x
/// let task = RegressionTask::new("linear", features, target).unwrap();
///
/// let mut lr = LinearRegression::default();
/// let model = lr.train_regress(&task).unwrap();
/// ```
pub struct LinearRegression;

impl Default for LinearRegression {
    fn default() -> Self {
        Self
    }
}

impl LinearRegression {
    /// Creates a new OLS linear regression learner (no hyperparameters to configure).
    pub fn new() -> Self {
        Self
    }
}

/// Solve Ax = b using Gaussian elimination with partial pivoting.
fn solve(a: &Array2<f64>, b: &Array1<f64>) -> Option<Array1<f64>> {
    let n = a.nrows();
    // Build augmented matrix [A | b]
    let mut aug = Array2::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Forward elimination
    for col in 0..n {
        // Partial pivoting
        let mut max_row = col;
        let mut max_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            let val = aug[[row, col]].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            return None; // singular
        }
        // Swap rows
        if max_row != col {
            for j in 0..=n {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }
        // Eliminate below
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / aug[[col, col]];
            for j in col..=n {
                aug[[row, j]] -= factor * aug[[col, j]];
            }
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        x[i] = aug[[i, n]];
        for j in (i + 1)..n {
            x[i] -= aug[[i, j]] * x[j];
        }
        x[i] /= aug[[i, i]];
    }
    Some(x)
}

// --- Trained model ---

use serde::{Deserialize, Serialize};

/// Trained OLS model: fitted weights (including bias) plus feature names.
#[derive(Serialize, Deserialize)]
pub struct TrainedLinearRegression {
    pub(crate) weights: Array1<f64>,
    pub(crate) feature_names: Vec<String>,
}

impl TrainedModel for TrainedLinearRegression {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        crate::validate::check_n_features(features, self.feature_names.len())?;
        let n_features = features.ncols();
        let predicted: Vec<f64> = features
            .rows()
            .into_iter()
            .map(|row| {
                let mut val = self.weights[n_features]; // bias
                for j in 0..n_features {
                    val += row[j] * self.weights[j];
                }
                val
            })
            .collect();
        Ok(Prediction::regression(predicted))
    }

    fn feature_importance(&self) -> Option<Vec<(String, f64)>> {
        let n = self.feature_names.len();
        let total: f64 = self.weights.iter().take(n).map(|w| w.abs()).sum();
        if total == 0.0 {
            return None;
        }
        Some(
            self.feature_names
                .iter()
                .enumerate()
                .map(|(i, name)| (name.clone(), self.weights[i].abs() / total))
                .collect(),
        )
    }
}

// --- Learner impl ---

impl Learner for LinearRegression {
    fn id(&self) -> &str {
        "linear_regression"
    }

    fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
        crate::validate::check_no_nan(task.features())?;
        let x = task.features();
        let y = task.target();
        let n = x.nrows();
        let p = x.ncols();

        // Augment X with bias column: [X | 1]
        let mut x_aug = Array2::zeros((n, p + 1));
        for i in 0..n {
            for j in 0..p {
                x_aug[[i, j]] = x[[i, j]];
            }
            x_aug[[i, p]] = 1.0;
        }

        // Normal equation: (X'X)w = X'y
        let xtx = x_aug.t().dot(&x_aug);
        let y_arr = Array1::from_vec(y.to_vec());
        let xty = x_aug.t().dot(&y_arr);

        let weights = solve(&xtx, &xty)
            .ok_or_else(|| SmeltError::NumericalError("Singular matrix in normal equation".into()))?;

        Ok(Box::new(TrainedLinearRegression {
            weights,
            feature_names: task.feature_names().to_vec(),
        }))
    }
}
