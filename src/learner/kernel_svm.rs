//! Kernel Support Vector Machine (C-SVC) via Sequential Minimal Optimization.
//!
//! The dual soft-margin SVM: maximize
//! `Σ αᵢ − ½ ΣΣ αᵢαⱼ yᵢyⱼ K(xᵢ,xⱼ)` subject to `0 ≤ αᵢ ≤ C` and
//! `Σ αᵢyᵢ = 0`, then classify with `f(x) = Σ αᵢyᵢ K(xᵢ,x) − b`. The kernel
//! `K` lets the linear max-margin separator live in a high-dimensional
//! feature space without ever forming it — the "kernel trick" — so unlike the
//! crate's SGD-hinge [`crate::learner::LinearSVM`], this learns genuinely
//! non-linear boundaries (RBF / polynomial).
//!
//! The quadratic program is solved with **SMO** (Platt, 1998): repeatedly pick
//! a pair of multipliers violating the KKT conditions and optimize them
//! analytically (the smallest sub-problem that still respects the linear
//! constraint), which needs no external QP solver. Multiclass is one-vs-rest.
//!
//! `O(n²)` kernel memory and roughly `O(n²)`–`O(n³)` time, so — like every SVM
//! — this is for small/medium datasets. Reference: Platt, J. (1998).
//! Sequential Minimal Optimization. MSR-TR-98-14.

use crate::learner::{Learner, LearnerProperties, TrainedModel};
use crate::prediction::Prediction;
use crate::task::{ClassificationTask, Task};
use crate::{Result, SmeltError};
use ndarray::{Array2, ArrayView1};

/// Kernel function for [`KernelSVM`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Kernel {
    /// Linear kernel `⟨x, x'⟩` — equivalent to a linear SVM but solved in the
    /// dual.
    Linear,
    /// Polynomial kernel `(gamma·⟨x, x'⟩ + coef0)^degree`.
    Poly {
        /// Polynomial degree.
        degree: u32,
        /// Independent term `coef0`.
        coef0: f64,
    },
    /// Radial basis function (Gaussian) kernel `exp(−gamma·‖x − x'‖²)`.
    Rbf,
}

impl Kernel {
    fn eval(&self, a: ArrayView1<f64>, b: ArrayView1<f64>, gamma: f64) -> f64 {
        match self {
            Kernel::Linear => a.dot(&b),
            Kernel::Poly { degree, coef0 } => (gamma * a.dot(&b) + coef0).powi(*degree as i32),
            Kernel::Rbf => {
                let sq: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
                (-gamma * sq).exp()
            }
        }
    }
}

/// Kernel Support Vector Classifier (C-SVC), classification only.
///
/// `C` is the soft-margin penalty (larger = fewer margin violations, higher
/// variance). `gamma` scales the RBF/poly kernel; if left unset it defaults to
/// `1 / n_features` at fit time (scikit-learn's `gamma="auto"`). Multiclass
/// targets are handled one-vs-rest.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use smelt_ml::learner::{KernelSVM, Kernel};
/// use ndarray::array;
///
/// // Two clearly separated blobs.
/// let x = array![
///     [0.0, 0.0], [0.3, 0.2], [0.1, 0.4],
///     [3.0, 3.0], [3.2, 2.8], [2.9, 3.1]
/// ];
/// let y = vec![0, 0, 0, 1, 1, 1];
/// let task = ClassificationTask::new("svm", x.clone(), y).unwrap();
///
/// let mut svm = KernelSVM::new().with_kernel(Kernel::Rbf).with_gamma(0.5);
/// let model = svm.train_classif(&task).unwrap();
/// let pred = model.predict(&x).unwrap();
/// if let Prediction::Classification { predicted, .. } = pred {
///     assert_eq!(predicted, vec![0, 0, 0, 1, 1, 1]);
/// }
/// ```
pub struct KernelSVM {
    c: f64,
    kernel: Kernel,
    gamma: Option<f64>,
    tol: f64,
    max_iter: usize,
}

impl Default for KernelSVM {
    fn default() -> Self {
        Self {
            c: 1.0,
            kernel: Kernel::Rbf,
            gamma: None,
            tol: 1e-4,
            max_iter: 10_000,
        }
    }
}

impl KernelSVM {
    /// Create a C-SVC with defaults (`C = 1`, RBF kernel, `gamma = auto`,
    /// tolerance `1e-4`).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the soft-margin penalty `C` (must be positive).
    pub fn with_c(mut self, c: f64) -> Self {
        self.c = c;
        self
    }

    /// Set the kernel function.
    pub fn with_kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set the kernel coefficient `gamma` (RBF/poly). Unset ⇒ `1 / n_features`.
    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.gamma = Some(gamma);
        self
    }

    /// Set the KKT tolerance for the SMO stopping condition.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the maximum number of SMO outer iterations (a safety bound).
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Fit and return the **concrete** [`TrainedKernelSVM`], which carries
    /// [`TrainedKernelSVM::decision_function`] beyond the `TrainedModel`
    /// trait. `Learner::train_classif` just boxes this (same split as
    /// `GaussianProcess::fit` / `KrigingHybrid::fit`).
    pub fn fit(&self, task: &ClassificationTask) -> Result<TrainedKernelSVM> {
        crate::validate::check_no_nan(task.features())?;
        crate::validate::check_no_weights(task.weights(), self.id())?;
        if !(self.c > 0.0 && self.c.is_finite()) {
            return Err(SmeltError::InvalidParameter(format!(
                "KernelSVM C must be positive and finite, got {}",
                self.c
            )));
        }
        let features = task.features();
        let n = features.nrows();
        let n_classes = task.n_classes();
        if n == 0 {
            return Err(SmeltError::InvalidParameter(
                "KernelSVM requires at least one training sample".into(),
            ));
        }
        if n_classes < 2 {
            return Err(SmeltError::InvalidParameter(
                "KernelSVM requires at least two classes".into(),
            ));
        }
        let gamma = self.gamma.unwrap_or(1.0 / task.n_features() as f64);
        if !(gamma > 0.0 && gamma.is_finite()) {
            return Err(SmeltError::InvalidParameter(format!(
                "KernelSVM gamma must be positive and finite, got {gamma}"
            )));
        }
        let target = task.target();

        let models = if n_classes == 2 {
            // Single binary model: class 1 → +1, class 0 → −1.
            let bin_y: Vec<f64> = target
                .iter()
                .map(|&t| if t == 1 { 1.0 } else { -1.0 })
                .collect();
            vec![fit_binary(
                features,
                &bin_y,
                self.kernel,
                gamma,
                self.c,
                self.tol,
                self.max_iter,
            )]
        } else {
            // One-vs-rest: model c scores class c (+1) against the rest (−1).
            (0..n_classes)
                .map(|cls| {
                    let bin_y: Vec<f64> = target
                        .iter()
                        .map(|&t| if t == cls { 1.0 } else { -1.0 })
                        .collect();
                    fit_binary(
                        features,
                        &bin_y,
                        self.kernel,
                        gamma,
                        self.c,
                        self.tol,
                        self.max_iter,
                    )
                })
                .collect()
        };

        Ok(TrainedKernelSVM {
            models,
            kernel: self.kernel,
            gamma,
            n_classes,
            n_features: task.n_features(),
        })
    }
}

/// One trained binary (`±1`) SVM: the support vectors, their signed
/// coefficients `αᵢyᵢ`, and the bias `b`. `decision(x) = Σ coefᵢ K(svᵢ,x) − b`.
struct BinarySVM {
    sv: Array2<f64>,
    sv_coef: Vec<f64>,
    bias: f64,
}

impl BinarySVM {
    fn decision(&self, x: ArrayView1<f64>, kernel: Kernel, gamma: f64) -> f64 {
        let mut sum = 0.0;
        for (i, sv_row) in self.sv.rows().into_iter().enumerate() {
            sum += self.sv_coef[i] * kernel.eval(sv_row, x, gamma);
        }
        sum - self.bias
    }
}

/// Solve the C-SVC dual for `±1` labels `y` given the precomputed kernel
/// matrix `k`, via Platt's SMO. Returns the multipliers `α` and the bias `b`
/// (decision uses `Σ αᵢyᵢ K − b`).
fn smo_solve(k: &Array2<f64>, y: &[f64], c: f64, tol: f64, max_iter: usize) -> (Vec<f64>, f64) {
    let n = y.len();
    let mut alpha = vec![0.0f64; n];
    let mut b = 0.0f64;
    let eps = 1e-8;

    // Output u_i = Σ_j α_j y_j K[j][i] − b.
    let output = |alpha: &[f64], b: f64, i: usize| -> f64 {
        let mut s = 0.0;
        for j in 0..n {
            if alpha[j] != 0.0 {
                s += alpha[j] * y[j] * k[[j, i]];
            }
        }
        s - b
    };

    // Analytic 2-variable optimization step; returns whether it changed α/b.
    let take_step = |alpha: &mut [f64], b: &mut f64, i1: usize, i2: usize| -> bool {
        if i1 == i2 {
            return false;
        }
        let (a1, a2) = (alpha[i1], alpha[i2]);
        let (y1, y2) = (y[i1], y[i2]);
        let e1 = output(alpha, *b, i1) - y1;
        let e2 = output(alpha, *b, i2) - y2;
        let s = y1 * y2;

        let (l, h) = if (y1 - y2).abs() > 0.5 {
            ((a2 - a1).max(0.0), c.min(c + a2 - a1))
        } else {
            ((a2 + a1 - c).max(0.0), c.min(a2 + a1))
        };
        if (h - l).abs() < eps {
            return false;
        }

        let k11 = k[[i1, i1]];
        let k12 = k[[i1, i2]];
        let k22 = k[[i2, i2]];
        let eta = k11 + k22 - 2.0 * k12;

        let mut a2_new = if eta > 0.0 {
            (a2 + y2 * (e1 - e2) / eta).clamp(l, h)
        } else {
            // Non-positive curvature (duplicate/degenerate points): evaluate
            // the dual objective at both endpoints and take the better.
            let f1 = y1 * (e1 + *b) - a1 * k11 - s * a2 * k12;
            let f2 = y2 * (e2 + *b) - s * a1 * k12 - a2 * k22;
            let l1 = a1 + s * (a2 - l);
            let h1 = a1 + s * (a2 - h);
            let lobj =
                l1 * f1 + l * f2 + 0.5 * l1 * l1 * k11 + 0.5 * l * l * k22 + s * l * l1 * k12;
            let hobj =
                h1 * f1 + h * f2 + 0.5 * h1 * h1 * k11 + 0.5 * h * h * k22 + s * h * h1 * k12;
            if lobj < hobj - eps {
                l
            } else if lobj > hobj + eps {
                h
            } else {
                a2
            }
        };
        if (a2_new - a2).abs() < eps * (a2_new + a2 + eps) {
            return false;
        }
        a2_new = a2_new.clamp(0.0, c);
        let a1_new = a1 + s * (a2 - a2_new);

        // Update bias to re-satisfy the KKT conditions at the changed points.
        let b1 = *b + e1 + y1 * (a1_new - a1) * k11 + y2 * (a2_new - a2) * k12;
        let b2 = *b + e2 + y1 * (a1_new - a1) * k12 + y2 * (a2_new - a2) * k22;
        *b = if a1_new > eps && a1_new < c - eps {
            b1
        } else if a2_new > eps && a2_new < c - eps {
            b2
        } else {
            0.5 * (b1 + b2)
        };
        alpha[i1] = a1_new;
        alpha[i2] = a2_new;
        true
    };

    // Examine one example; if it violates KKT, find a partner and step.
    let examine = |alpha: &mut [f64], b: &mut f64, i2: usize| -> bool {
        let y2 = y[i2];
        let a2 = alpha[i2];
        let e2 = output(alpha, *b, i2) - y2;
        let r2 = e2 * y2;
        if !((r2 < -tol && a2 < c) || (r2 > tol && a2 > 0.0)) {
            return false;
        }
        let nonbound: Vec<usize> = (0..n)
            .filter(|&i| alpha[i] > eps && alpha[i] < c - eps)
            .collect();
        // Heuristic 1: pick i1 maximizing |E1 − E2|.
        if nonbound.len() > 1 {
            let mut best = usize::MAX;
            let mut best_delta = -1.0;
            for &i1 in &nonbound {
                let d = (output(alpha, *b, i1) - y[i1] - e2).abs();
                if d > best_delta {
                    best_delta = d;
                    best = i1;
                }
            }
            if best != usize::MAX && take_step(alpha, b, best, i2) {
                return true;
            }
        }
        // Heuristic 2: all non-bound, then all examples (deterministic order).
        for &i1 in &nonbound {
            if take_step(alpha, b, i1, i2) {
                return true;
            }
        }
        for i1 in 0..n {
            if take_step(alpha, b, i1, i2) {
                return true;
            }
        }
        false
    };

    let mut examine_all = true;
    let mut iters = 0;
    while iters < max_iter {
        iters += 1;
        let mut num_changed = 0;
        if examine_all {
            for i in 0..n {
                if examine(&mut alpha, &mut b, i) {
                    num_changed += 1;
                }
            }
        } else {
            for i in 0..n {
                if alpha[i] > eps && alpha[i] < c - eps && examine(&mut alpha, &mut b, i) {
                    num_changed += 1;
                }
            }
        }
        if examine_all {
            examine_all = false;
        } else if num_changed == 0 {
            break;
        } else {
            examine_all = true;
        }
    }
    (alpha, b)
}

/// Train one binary `±1` SVM (`bin_y` entries are `+1`/`−1`) on `features`.
fn fit_binary(
    features: &Array2<f64>,
    bin_y: &[f64],
    kernel: Kernel,
    gamma: f64,
    c: f64,
    tol: f64,
    max_iter: usize,
) -> BinarySVM {
    let n = features.nrows();
    let mut k = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in i..n {
            let val = kernel.eval(features.row(i), features.row(j), gamma);
            k[[i, j]] = val;
            k[[j, i]] = val;
        }
    }
    let (alpha, bias) = smo_solve(&k, bin_y, c, tol, max_iter);

    // Keep only support vectors (α > 0).
    let sv_idx: Vec<usize> = (0..n).filter(|&i| alpha[i] > 1e-8).collect();
    let n_sv = sv_idx.len();
    let n_feat = features.ncols();
    let mut sv = Array2::<f64>::zeros((n_sv, n_feat));
    let mut sv_coef = Vec::with_capacity(n_sv);
    for (row, &i) in sv_idx.iter().enumerate() {
        sv.row_mut(row).assign(&features.row(i));
        sv_coef.push(alpha[i] * bin_y[i]);
    }
    BinarySVM { sv, sv_coef, bias }
}

impl Learner for KernelSVM {
    fn id(&self) -> &str {
        "kernel_svm"
    }

    fn properties(&self) -> LearnerProperties {
        LearnerProperties::classifier()
    }

    fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        Ok(Box::new(self.fit(task)?))
    }
}

/// A fitted [`KernelSVM`]: one binary model (2 classes) or `n_classes`
/// one-vs-rest models, plus the kernel needed to score new points.
pub struct TrainedKernelSVM {
    models: Vec<BinarySVM>,
    kernel: Kernel,
    gamma: f64,
    n_classes: usize,
    n_features: usize,
}

impl TrainedKernelSVM {
    /// Raw decision value(s) for one sample: a single signed margin for the
    /// binary case, or one one-vs-rest score per class for multiclass.
    pub fn decision_function(&self, row: ArrayView1<f64>) -> Vec<f64> {
        self.models
            .iter()
            .map(|m| m.decision(row, self.kernel, self.gamma))
            .collect()
    }
}

impl TrainedModel for TrainedKernelSVM {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        crate::validate::check_n_features(features, self.n_features)?;
        let predicted: Vec<usize> = features
            .rows()
            .into_iter()
            .map(|row| {
                if self.n_classes == 2 {
                    // decision > 0 → class 1, else class 0.
                    if self.models[0].decision(row, self.kernel, self.gamma) > 0.0 {
                        1
                    } else {
                        0
                    }
                } else {
                    // argmax one-vs-rest score.
                    self.decision_function(row)
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(i, _)| i)
                        .unwrap_or(0)
                }
            })
            .collect();
        Ok(Prediction::Classification {
            predicted,
            truth: None,
            probabilities: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn golden_binary() -> (Array2<f64>, Vec<usize>) {
        let x = array![
            [0.0, 0.0],
            [0.4, 0.2],
            [0.2, 0.5],
            [0.6, 0.1],
            [0.1, 0.6],
            [2.0, 2.0],
            [2.4, 1.8],
            [1.8, 2.3],
            [2.2, 1.9],
            [1.9, 2.4],
            [0.3, 0.3],
            [2.1, 2.1],
            [0.5, 0.5],
            [1.7, 1.9]
        ];
        let y = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1];
        (x, y)
    }

    /// Golden test against scikit-learn 1.8.0 `SVC(C=1, kernel="rbf",
    /// gamma=0.5, tol=1e-4)`: the decision-function values on fresh points
    /// must match (both solve the same convex QP, so the decision function is
    /// unique), and predictions must match exactly.
    #[test]
    fn matches_sklearn_svc_rbf_golden() {
        let (x, y) = golden_binary();
        let task = ClassificationTask::new("svm", x, y).unwrap();
        let svm = KernelSVM::new()
            .with_kernel(Kernel::Rbf)
            .with_gamma(0.5)
            .with_c(1.0)
            .with_tol(1e-4);
        let trained = svm.fit(&task).unwrap();

        let xt = array![[0.3, 0.4], [2.0, 2.1], [1.0, 1.0], [0.7, 0.2], [1.8, 2.0]];
        let exp_dec = [
            -1.0709903096,
            1.0829726116,
            -0.2918542214,
            -0.9726765016,
            1.0533297936,
        ];
        let exp_pred = [0usize, 1, 0, 0, 1];

        for (i, row) in xt.rows().into_iter().enumerate() {
            let dec = trained.decision_function(row)[0];
            assert!(
                (dec - exp_dec[i]).abs() < 1e-2,
                "decision {i}: {dec} vs {}",
                exp_dec[i]
            );
        }
        if let Prediction::Classification { predicted, .. } = trained.predict(&xt).unwrap() {
            assert_eq!(predicted, exp_pred.to_vec());
        }
    }

    /// A non-linearly separable XOR-like problem: a linear SVM cannot solve
    /// it, but the RBF kernel achieves perfect in-sample accuracy.
    #[test]
    fn rbf_solves_nonlinear_xor() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [1.0, 1.0],
            [0.9, 0.9],
            [0.0, 1.0],
            [0.1, 0.9],
            [1.0, 0.0],
            [0.9, 0.1]
        ];
        let y = vec![0, 0, 0, 0, 1, 1, 1, 1]; // XOR: class 1 iff exactly one coord high
        let task = ClassificationTask::new("xor", x.clone(), y.clone()).unwrap();
        let mut svm = KernelSVM::new().with_kernel(Kernel::Rbf).with_gamma(2.0);
        let model = svm.train_classif(&task).unwrap();
        if let Prediction::Classification { predicted, .. } = model.predict(&x).unwrap() {
            assert_eq!(predicted, y, "RBF SVM should solve XOR perfectly");
        }
    }

    /// One-vs-rest multiclass on three separated blobs.
    #[test]
    fn multiclass_ovr_separates_three_blobs() {
        let x = array![
            [0.0, 0.0],
            [0.2, 0.1],
            [5.0, 5.0],
            [5.2, 4.8],
            [0.0, 5.0],
            [0.1, 5.2]
        ];
        let y = vec![0, 0, 1, 1, 2, 2];
        let task = ClassificationTask::new("multi", x.clone(), y.clone()).unwrap();
        let mut svm = KernelSVM::new().with_kernel(Kernel::Rbf).with_gamma(0.5);
        let model = svm.train_classif(&task).unwrap();
        if let Prediction::Classification { predicted, .. } = model.predict(&x).unwrap() {
            assert_eq!(predicted, y);
        }
    }

    /// The linear kernel separates a linearly separable problem.
    #[test]
    fn linear_kernel_separates_linear_problem() {
        let x = array![
            [0.0, 0.0],
            [1.0, 0.5],
            [0.5, 1.0],
            [4.0, 4.0],
            [5.0, 4.5],
            [4.5, 5.0]
        ];
        let y = vec![0, 0, 0, 1, 1, 1];
        let task = ClassificationTask::new("lin", x.clone(), y.clone()).unwrap();
        let mut svm = KernelSVM::new().with_kernel(Kernel::Linear);
        let model = svm.train_classif(&task).unwrap();
        if let Prediction::Classification { predicted, .. } = model.predict(&x).unwrap() {
            assert_eq!(predicted, y);
        }
    }

    #[test]
    fn rejects_bad_params_and_weights() {
        let (x, y) = golden_binary();
        let task = ClassificationTask::new("svm", x.clone(), y.clone()).unwrap();
        assert!(KernelSVM::new().with_c(0.0).fit(&task).is_err());
        assert!(KernelSVM::new().with_gamma(-1.0).fit(&task).is_err());
        let weighted = ClassificationTask::new("svm", x, y)
            .unwrap()
            .with_weights(vec![1.0; 14]);
        assert!(
            KernelSVM::new().fit(&weighted).is_err(),
            "KernelSVM declares no weight support, so a weighted task must be rejected"
        );
    }
}
