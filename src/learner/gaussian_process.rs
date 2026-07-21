//! Gaussian Process regression with predictive standard errors.
//!
//! A non-parametric Bayesian regressor: it places a Gaussian prior over
//! functions (an RBF/squared-exponential kernel here) and conditions on the
//! training data to get a posterior that is itself Gaussian at every test
//! point — so each prediction comes with a **standard error**, not just a
//! point estimate. That predictive `se` is the whole reason to reach for a GP
//! over the crate's other regressors, and it is exposed beyond the `Learner`
//! trait via [`TrainedGaussianProcess::predict_std`] /
//! [`TrainedGaussianProcess::predict_with_std`] (the same "concrete type
//! carries more than the trait" shape as `TrainedGeoXGBoost::predict_spatial`).
//!
//! Fitting is the exact GP posterior: Cholesky-factorize `K = k(X,X) + α·I`,
//! solve `α_vec = K⁻¹ y` once, then a test point `x*` has mean `k*ᵀ α_vec` and
//! variance `k(x*,x*) − k*ᵀ K⁻¹ k*`. `O(n³)` to fit and `O(n²)` memory (the
//! kernel matrix), so GPs are for small/medium datasets — as everywhere they
//! are used.
//!
//! Kernel hyperparameters (`length_scale`, `signal_variance`, noise `alpha`)
//! are **fixed**, not optimized: marginal-likelihood hyperparameter tuning is
//! a deliberate future extension (it needs the log-ML gradient), not done
//! here. Reference: Rasmussen & Williams (2006), *Gaussian Processes for
//! Machine Learning*, Algorithm 2.1.

use crate::learner::{Learner, LearnerProperties, TrainedModel};
use crate::prediction::Prediction;
use crate::task::{RegressionTask, Task};
use crate::{Result, SmeltError};
use ndarray::{Array1, Array2, ArrayView1};

/// Gaussian Process regressor with an RBF (squared-exponential) kernel.
///
/// `k(x, x') = signal_variance · exp(−‖x − x'‖² / (2·length_scale²))`, with a
/// noise term `alpha` added to the kernel matrix diagonal. See the module
/// docs for the fitting math and the `O(n³)` cost note.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use smelt_ml::learner::GaussianProcess;
/// use ndarray::array;
///
/// let x = array![[0.0], [1.0], [2.0], [3.0], [4.0]];
/// let y = vec![0.0, 1.0, 4.0, 9.0, 16.0];
/// let task = RegressionTask::new("gp", x.clone(), y).unwrap();
///
/// let mut gp = GaussianProcess::new().with_length_scale(1.5);
/// let model = gp.train_regress(&task).unwrap();
/// let pred = model.predict(&x).unwrap();
/// // In-sample predictions track the (near-noise-free) training targets.
/// if let Prediction::Regression { predicted, .. } = pred {
///     assert!((predicted[2] - 4.0).abs() < 0.5);
/// }
/// ```
pub struct GaussianProcess {
    length_scale: f64,
    signal_variance: f64,
    alpha: f64,
}

impl Default for GaussianProcess {
    fn default() -> Self {
        Self {
            length_scale: 1.0,
            signal_variance: 1.0,
            alpha: 1e-10,
        }
    }
}

impl GaussianProcess {
    /// Create a GP with default kernel hyperparameters (length scale 1, unit
    /// signal variance, noise `1e-10` — matching scikit-learn's near-noise-free
    /// default).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the RBF length scale: larger means smoother functions that vary
    /// over longer distances.
    pub fn with_length_scale(mut self, length_scale: f64) -> Self {
        self.length_scale = length_scale;
        self
    }

    /// Set the signal variance (kernel amplitude / prior variance of the
    /// function values).
    pub fn with_signal_variance(mut self, signal_variance: f64) -> Self {
        self.signal_variance = signal_variance;
        self
    }

    /// Set the observation noise variance added to the kernel diagonal.
    /// Larger values regularize the fit and keep the kernel matrix
    /// well-conditioned; too small can make the Cholesky factorization fail on
    /// near-duplicate inputs.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    fn check_params(&self) -> Result<()> {
        if !(self.length_scale > 0.0 && self.length_scale.is_finite()) {
            return Err(SmeltError::InvalidParameter(format!(
                "GaussianProcess length_scale must be positive and finite, got {}",
                self.length_scale
            )));
        }
        if !(self.signal_variance > 0.0 && self.signal_variance.is_finite()) {
            return Err(SmeltError::InvalidParameter(format!(
                "GaussianProcess signal_variance must be positive and finite, got {}",
                self.signal_variance
            )));
        }
        if !(self.alpha >= 0.0 && self.alpha.is_finite()) {
            return Err(SmeltError::InvalidParameter(format!(
                "GaussianProcess alpha (noise) must be non-negative and finite, got {}",
                self.alpha
            )));
        }
        Ok(())
    }

    /// Fit and return the **concrete** [`TrainedGaussianProcess`], which
    /// carries `predict_std`/`predict_with_std` beyond the `TrainedModel`
    /// trait. `Learner::train_regress` just boxes this (same split as
    /// `KrigingHybrid::fit` / `DeepForest::fit`).
    pub fn fit(&self, task: &RegressionTask) -> Result<TrainedGaussianProcess> {
        self.check_params()?;
        crate::validate::check_no_nan(task.features())?;
        crate::validate::check_no_weights(task.weights(), self.id())?;

        let features = task.features();
        let target = task.target();
        let n = features.nrows();
        if n == 0 {
            return Err(SmeltError::InvalidParameter(
                "GaussianProcess requires at least one training sample".into(),
            ));
        }

        // K = k(X, X) + alpha·I.
        let mut k = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in i..n {
                let val = rbf(
                    features.row(i),
                    features.row(j),
                    self.length_scale,
                    self.signal_variance,
                );
                k[[i, j]] = val;
                k[[j, i]] = val;
            }
            k[[i, i]] += self.alpha;
        }

        let l_chol = cholesky(&k).ok_or_else(|| {
            SmeltError::InvalidParameter(
                "GaussianProcess kernel matrix is not positive definite; increase alpha (noise) \
                 or the length scale"
                    .into(),
            )
        })?;

        // alpha_vec = K⁻¹ y  via  L z = y,  Lᵀ alpha_vec = z.
        let y = Array1::from_vec(target.to_vec());
        let z = forward_substitution(&l_chol, &y);
        let alpha_vec = back_substitution_lt(&l_chol, &z);

        Ok(TrainedGaussianProcess {
            x_train: features.clone(),
            alpha_vec,
            l_chol,
            length_scale: self.length_scale,
            signal_variance: self.signal_variance,
            n_features: task.n_features(),
        })
    }
}

/// RBF kernel value between two rows.
fn rbf(a: ArrayView1<f64>, b: ArrayView1<f64>, length_scale: f64, signal_variance: f64) -> f64 {
    let sq: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    signal_variance * (-sq / (2.0 * length_scale * length_scale)).exp()
}

/// Cholesky factorization of a symmetric positive-definite matrix: returns the
/// lower-triangular `L` with `A = L Lᵀ`, or `None` if `A` is not (numerically)
/// positive definite. Hand-rolled per this crate's per-module numeric-routine
/// convention (cf. `regularized.rs`, `kriging_hybrid.rs`, `survival/cox.rs`).
fn cholesky(a: &Array2<f64>) -> Option<Array2<f64>> {
    let n = a.nrows();
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for p in 0..j {
                sum -= l[[i, p]] * l[[j, p]];
            }
            if i == j {
                if sum <= 0.0 {
                    return None;
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    Some(l)
}

/// Forward substitution: solve `L x = b` for lower-triangular `L`.
fn forward_substitution(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = b.len();
    let mut x = Array1::zeros(n);
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[[i, j]] * x[j];
        }
        x[i] = sum / l[[i, i]];
    }
    x
}

/// Back substitution: solve `Lᵀ x = b`, using the lower-triangular `L`
/// (its transpose is upper-triangular).
fn back_substitution_lt(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = b.len();
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= l[[j, i]] * x[j]; // Lᵀ[i][j] = L[j][i]
        }
        x[i] = sum / l[[i, i]];
    }
    x
}

impl Learner for GaussianProcess {
    fn id(&self) -> &str {
        "gaussian_process"
    }

    fn properties(&self) -> LearnerProperties {
        LearnerProperties::regressor()
    }

    fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
        Ok(Box::new(self.fit(task)?))
    }
}

/// A fitted [`GaussianProcess`]: the training inputs, the solved dual weights
/// `K⁻¹ y`, and the Cholesky factor of `K`, retained so predictions carry
/// posterior means **and** standard errors.
pub struct TrainedGaussianProcess {
    x_train: Array2<f64>,
    alpha_vec: Array1<f64>,
    l_chol: Array2<f64>,
    length_scale: f64,
    signal_variance: f64,
    n_features: usize,
}

impl TrainedGaussianProcess {
    /// Posterior mean and standard deviation for every row of `features`.
    ///
    /// The standard deviation is `sqrt(k(x*,x*) − k*ᵀ K⁻¹ k*)` — the posterior
    /// uncertainty of the latent function (matching scikit-learn's
    /// `predict(return_std=True)`, which likewise does not add the observation
    /// noise back in). Negative variances from round-off are clamped to zero.
    pub fn predict_with_std(&self, features: &Array2<f64>) -> Result<(Vec<f64>, Vec<f64>)> {
        crate::validate::check_n_features(features, self.n_features)?;
        let n_train = self.x_train.nrows();
        let mut means = Vec::with_capacity(features.nrows());
        let mut stds = Vec::with_capacity(features.nrows());

        for row in features.rows() {
            // k* : kernel between this test point and every training point.
            let mut k_star = Array1::<f64>::zeros(n_train);
            for (j, train_row) in self.x_train.rows().into_iter().enumerate() {
                k_star[j] = rbf(row, train_row, self.length_scale, self.signal_variance);
            }
            means.push(k_star.dot(&self.alpha_vec));

            // var = k(x*,x*) − ‖L⁻¹ k*‖²  (k(x*,x*) = signal_variance for RBF).
            let v = forward_substitution(&self.l_chol, &k_star);
            let var = self.signal_variance - v.dot(&v);
            stds.push(var.max(0.0).sqrt());
        }
        Ok((means, stds))
    }

    /// Posterior standard deviations only (predictive `se`) for each row.
    pub fn predict_std(&self, features: &Array2<f64>) -> Result<Vec<f64>> {
        Ok(self.predict_with_std(features)?.1)
    }
}

impl TrainedModel for TrainedGaussianProcess {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        let (means, _) = self.predict_with_std(features)?;
        Ok(Prediction::regression(means))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Fixed dataset shared with the scikit-learn golden generator.
    fn golden_data() -> (Array2<f64>, Vec<f64>) {
        let x = array![
            [0.0, 0.0],
            [1.0, 0.5],
            [2.0, 1.0],
            [0.5, 2.0],
            [1.5, 1.5],
            [3.0, 0.0],
            [2.5, 2.5],
            [0.0, 3.0]
        ];
        let y = vec![1.0, 2.1, 3.2, 1.8, 2.9, 3.5, 4.1, 1.2];
        (x, y)
    }

    /// Golden test against scikit-learn 1.8.0
    /// `GaussianProcessRegressor(kernel=RBF(1.3), alpha=1e-2, optimizer=None,
    /// normalize_y=False)`: posterior means AND standard deviations at fresh
    /// test points must match to 1e-6.
    #[test]
    fn matches_sklearn_gp_golden() {
        let (x, y) = golden_data();
        let task = RegressionTask::new("gp", x, y).unwrap();
        let gp = GaussianProcess::new()
            .with_length_scale(1.3)
            .with_alpha(1e-2);
        let trained = gp.fit(&task).unwrap();

        let xtest = array![[0.2, 0.3], [2.0, 2.0], [3.0, 3.0], [1.0, 1.0]];
        let (means, stds) = trained.predict_with_std(&xtest).unwrap();

        let exp_mean = [
            1.269366793822,
            3.710432658716,
            3.471980793667,
            2.178255994815,
        ];
        let exp_std = [
            0.155930511291,
            0.165287792882,
            0.417518233886,
            0.147239827378,
        ];
        for (got, exp) in means.iter().zip(&exp_mean) {
            assert!((got - exp).abs() < 1e-6, "mean {got} vs {exp}");
        }
        for (got, exp) in stds.iter().zip(&exp_std) {
            assert!((got - exp).abs() < 1e-6, "std {got} vs {exp}");
        }
    }

    /// Predictive std must be small near training points and grow as test
    /// points move away from the data — the defining behavior of a GP's `se`.
    #[test]
    fn uncertainty_grows_away_from_data() {
        let x = array![[0.0], [1.0], [2.0], [3.0], [4.0]];
        let y = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let task = RegressionTask::new("gp", x, y).unwrap();
        let gp = GaussianProcess::new()
            .with_length_scale(1.0)
            .with_alpha(1e-6);
        let trained = gp.fit(&task).unwrap();

        let near = array![[2.0]]; // on a training point
        let far = array![[10.0]]; // far outside the training range
        let std_near = trained.predict_std(&near).unwrap()[0];
        let std_far = trained.predict_std(&far).unwrap()[0];
        assert!(std_near < 0.05, "std at a training point should be tiny");
        assert!(
            std_far > std_near + 0.5,
            "std should grow far from data: near={std_near}, far={std_far}"
        );
        // Far from data the posterior mean reverts toward the zero prior mean.
        let mean_far = match trained.predict(&far).unwrap() {
            Prediction::Regression { predicted, .. } => predicted[0],
            _ => unreachable!(),
        };
        assert!(mean_far.abs() < 0.5);
    }

    #[test]
    fn rejects_bad_hyperparameters_and_weights() {
        let (x, y) = golden_data();
        let task = RegressionTask::new("gp", x.clone(), y.clone()).unwrap();
        assert!(
            GaussianProcess::new()
                .with_length_scale(0.0)
                .train_regress(&task)
                .is_err()
        );
        assert!(
            GaussianProcess::new()
                .with_alpha(-1.0)
                .train_regress(&task)
                .is_err()
        );
        let weighted = RegressionTask::new("gp", x, y)
            .unwrap()
            .with_weights(vec![1.0; 8]);
        assert!(
            GaussianProcess::new().fit(&weighted).is_err(),
            "GP declares no weight support, so a weighted task must be rejected"
        );
    }
}
