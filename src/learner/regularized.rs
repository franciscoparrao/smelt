//! Regularized regression: Ridge (L2), Lasso (L1), Elastic Net (L1+L2).

use ndarray::{Array1, Array2};
use serde::{Serialize, Deserialize};
use crate::task::{ClassificationTask, RegressionTask, Task};
use crate::learner::{Learner, TrainedModel};
use crate::prediction::Prediction;
use crate::{SmeltError, Result};

// --- Shared trained model ---

#[derive(Serialize, Deserialize)]
pub struct TrainedRegularizedRegression {
    pub(crate) weights: Array1<f64>,
    pub(crate) feature_names: Vec<String>,
    pub(crate) learner_id: String,
}

impl TrainedModel for TrainedRegularizedRegression {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        let n_features = features.ncols();
        let predicted: Vec<f64> = features.rows().into_iter()
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
        if total == 0.0 { return None; }
        Some(self.feature_names.iter().enumerate()
            .map(|(i, name)| (name.clone(), self.weights[i].abs() / total)).collect())
    }
}

// --- Ridge Regression ---

/// Ridge Regression (L2 regularization).
///
/// Solves: min ||Xw - y||² + alpha * ||w||²
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use ndarray::array;
///
/// let features = array![[1.0], [2.0], [3.0], [4.0]];
/// let target = vec![2.0, 4.0, 6.0, 8.0];
/// let task = RegressionTask::new("ridge", features, target).unwrap();
///
/// let mut ridge = Ridge::new(1.0);
/// let model = ridge.train_regress(&task).unwrap();
/// ```
pub struct Ridge {
    alpha: f64,
}

impl Ridge {
    pub fn new(alpha: f64) -> Self { Self { alpha } }
}

impl Default for Ridge {
    fn default() -> Self { Self { alpha: 1.0 } }
}

/// Solve Ax = b using Gaussian elimination with partial pivoting.
fn solve(a: &Array2<f64>, b: &Array1<f64>) -> Option<Array1<f64>> {
    let n = a.nrows();
    let mut aug = Array2::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n { aug[[i, j]] = a[[i, j]]; }
        aug[[i, n]] = b[i];
    }
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            let val = aug[[row, col]].abs();
            if val > max_val { max_val = val; max_row = row; }
        }
        if max_val < 1e-12 { return None; }
        if max_row != col {
            for j in 0..=n { let tmp = aug[[col, j]]; aug[[col, j]] = aug[[max_row, j]]; aug[[max_row, j]] = tmp; }
        }
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / aug[[col, col]];
            for j in col..=n { aug[[row, j]] -= factor * aug[[col, j]]; }
        }
    }
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        x[i] = aug[[i, n]];
        for j in (i + 1)..n { x[i] -= aug[[i, j]] * x[j]; }
        x[i] /= aug[[i, i]];
    }
    Some(x)
}

impl Learner for Ridge {
    fn id(&self) -> &str { "ridge" }

    fn train_classif(&mut self, _: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        Err(SmeltError::Other("Ridge does not support classification".into()))
    }

    fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
        let x = task.features();
        let y = task.target();
        let n = x.nrows();
        let p = x.ncols();

        // Augment with bias column
        let mut x_aug = Array2::zeros((n, p + 1));
        for i in 0..n {
            for j in 0..p { x_aug[[i, j]] = x[[i, j]]; }
            x_aug[[i, p]] = 1.0;
        }

        // (X'X + alpha*I)w = X'y  (don't penalize bias)
        let mut xtx = x_aug.t().dot(&x_aug);
        for j in 0..p {
            xtx[[j, j]] += self.alpha;
        }
        let y_arr = Array1::from_vec(y.to_vec());
        let xty = x_aug.t().dot(&y_arr);

        let weights = solve(&xtx, &xty)
            .ok_or_else(|| SmeltError::Other("Singular matrix in Ridge".into()))?;

        Ok(Box::new(TrainedRegularizedRegression {
            weights, feature_names: task.feature_names().to_vec(),
            learner_id: "ridge".into(),
        }))
    }
}

// --- Lasso Regression ---

/// Lasso Regression (L1 regularization) via coordinate descent.
///
/// Solves: min (1/2n) ||Xw - y||² + alpha * ||w||₁
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use ndarray::array;
///
/// let features = array![[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]];
/// let target = vec![2.0, 4.0, 6.0, 8.0];
/// let task = RegressionTask::new("lasso", features, target).unwrap();
///
/// let mut lasso = Lasso::new(0.01);
/// let model = lasso.train_regress(&task).unwrap();
/// ```
pub struct Lasso {
    alpha: f64,
    max_iter: usize,
    tol: f64,
}

impl Lasso {
    pub fn new(alpha: f64) -> Self { Self { alpha, max_iter: 1000, tol: 1e-6 } }
    pub fn with_max_iter(mut self, n: usize) -> Self { self.max_iter = n; self }
    pub fn with_tol(mut self, tol: f64) -> Self { self.tol = tol; self }
}

impl Default for Lasso {
    fn default() -> Self { Self::new(1.0) }
}

fn soft_threshold(x: f64, lambda: f64) -> f64 {
    if x > lambda { x - lambda }
    else if x < -lambda { x + lambda }
    else { 0.0 }
}

fn coordinate_descent(
    x: &Array2<f64>, y: &[f64], alpha: f64, l1_ratio: f64,
    max_iter: usize, tol: f64,
) -> Array1<f64> {
    let n = x.nrows();
    let p = x.ncols(); // includes bias column
    let n_f = n as f64;
    let mut w = Array1::zeros(p);

    for _ in 0..max_iter {
        let mut max_change = 0.0f64;

        for j in 0..p {
            let old_w = w[j];
            let mut residual_sum = 0.0;
            let mut col_sq_sum = 0.0;

            for i in 0..n {
                let pred: f64 = (0..p).map(|k| x[[i, k]] * w[k]).sum();
                let residual = y[i] - pred + x[[i, j]] * w[j];
                residual_sum += x[[i, j]] * residual;
                col_sq_sum += x[[i, j]] * x[[i, j]];
            }

            if col_sq_sum < 1e-12 { continue; }

            let rho = residual_sum / n_f;
            // Don't penalize bias (last column)
            w[j] = if j < p - 1 {
                soft_threshold(rho, alpha * l1_ratio) / (col_sq_sum / n_f + alpha * (1.0 - l1_ratio))
            } else {
                rho / (col_sq_sum / n_f)
            };

            max_change = max_change.max((w[j] - old_w).abs());
        }

        if max_change < tol { break; }
    }

    w
}

impl Learner for Lasso {
    fn id(&self) -> &str { "lasso" }

    fn train_classif(&mut self, _: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        Err(SmeltError::Other("Lasso does not support classification".into()))
    }

    fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
        let x = task.features();
        let y = task.target();
        let n = x.nrows();
        let p = x.ncols();

        let mut x_aug = Array2::zeros((n, p + 1));
        for i in 0..n {
            for j in 0..p { x_aug[[i, j]] = x[[i, j]]; }
            x_aug[[i, p]] = 1.0;
        }

        let weights = coordinate_descent(&x_aug, y, self.alpha, 1.0, self.max_iter, self.tol);

        Ok(Box::new(TrainedRegularizedRegression {
            weights, feature_names: task.feature_names().to_vec(),
            learner_id: "lasso".into(),
        }))
    }
}

// --- Elastic Net ---

/// Elastic Net regression (L1 + L2 regularization) via coordinate descent.
///
/// Solves: min (1/2n) ||Xw - y||² + alpha * (l1_ratio * ||w||₁ + (1-l1_ratio)/2 * ||w||²)
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use ndarray::array;
///
/// let features = array![[1.0], [2.0], [3.0], [4.0]];
/// let target = vec![2.0, 4.0, 6.0, 8.0];
/// let task = RegressionTask::new("enet", features, target).unwrap();
///
/// let mut enet = ElasticNet::new(0.01, 0.5);
/// let model = enet.train_regress(&task).unwrap();
/// ```
pub struct ElasticNet {
    alpha: f64,
    l1_ratio: f64,
    max_iter: usize,
    tol: f64,
}

impl ElasticNet {
    pub fn new(alpha: f64, l1_ratio: f64) -> Self {
        Self { alpha, l1_ratio, max_iter: 1000, tol: 1e-6 }
    }
    pub fn with_max_iter(mut self, n: usize) -> Self { self.max_iter = n; self }
    pub fn with_tol(mut self, tol: f64) -> Self { self.tol = tol; self }
}

impl Default for ElasticNet {
    fn default() -> Self { Self::new(1.0, 0.5) }
}

impl Learner for ElasticNet {
    fn id(&self) -> &str { "elastic_net" }

    fn train_classif(&mut self, _: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        Err(SmeltError::Other("ElasticNet does not support classification".into()))
    }

    fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
        let x = task.features();
        let y = task.target();
        let n = x.nrows();
        let p = x.ncols();

        let mut x_aug = Array2::zeros((n, p + 1));
        for i in 0..n {
            for j in 0..p { x_aug[[i, j]] = x[[i, j]]; }
            x_aug[[i, p]] = 1.0;
        }

        let weights = coordinate_descent(&x_aug, y, self.alpha, self.l1_ratio, self.max_iter, self.tol);

        Ok(Box::new(TrainedRegularizedRegression {
            weights, feature_names: task.feature_names().to_vec(),
            learner_id: "elastic_net".into(),
        }))
    }
}
