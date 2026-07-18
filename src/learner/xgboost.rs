//! XGBoost: eXtreme Gradient Boosting with histogram-based splitting.
//!
//! Core algorithm with Newton boosting, L1/L2 regularization, gamma min gain,
//! histogram-based + auto exact greedy, NaN handling, row/col subsampling,
//! parallel split finding, early stopping, zero-copy prediction, in-place partitioning.

use crate::{Result, SmeltError};
use crate::learner::math::{sigmoid, softmax};
use crate::learner::{Learner, TrainedModel};
use crate::prediction::Prediction;
use crate::task::{ClassificationTask, RegressionTask, Task};
use ndarray::{Array2, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::eval::{EarlyStopper, EvalSet, EvalTarget, validate_eval_classif, validate_eval_regress};
use super::hist_pool::HistPool;
use super::histogram::{HistBins, NAN_BIN, accumulate_histogram, best_categorical_split, best_numeric_split};

/// Regression objective: defines the per-sample gradient/hessian the trees
/// fit, the initial prediction, and the output transform.
///
/// Classification training is unaffected (it always uses log-loss /
/// softmax); setting an objective only changes `train_regress`.
#[derive(Clone)]
pub enum Objective {
    /// Squared error (the default): g = p - y, h = 1.
    SquaredError,
    /// Huber loss as gradient clipping: g = clamp(p - y, ±delta), h = 1.
    /// Robust to outliers in the target.
    Huber {
        /// Clipping threshold beyond which residuals are treated as outliers.
        delta: f64,
    },
    /// Poisson regression with log link: the model fits f = log(μ),
    /// g = exp(f) - y, h = exp(f); predictions are exp-transformed back to
    /// the response scale. Targets must be non-negative counts/rates.
    Poisson,
    /// Custom objective: `f(prediction, target) -> (gradient, hessian)` on
    /// the raw score. The hessian must be positive for stable Newton steps.
    Custom(std::sync::Arc<dyn Fn(f64, f64) -> (f64, f64) + Send + Sync>),
}

impl Objective {
    /// Per-sample gradient and hessian at prediction `p` for target `y`.
    #[inline]
    fn grad_hess(&self, p: f64, y: f64) -> (f64, f64) {
        match self {
            Objective::SquaredError => (p - y, 1.0),
            Objective::Huber { delta } => ((p - y).clamp(-delta, *delta), 1.0),
            Objective::Poisson => {
                let mu = p.min(30.0).exp(); // cap to avoid overflow
                (mu - y, mu.max(1e-15))
            }
            Objective::Custom(f) => f(p, y),
        }
    }

    /// Initial raw score given the (weighted) target mean.
    #[inline]
    fn initial_score(&self, target_mean: f64) -> f64 {
        match self {
            Objective::Poisson => target_mean.max(1e-15).ln(),
            _ => target_mean,
        }
    }

    /// Monitoring loss for early stopping: the mean per-sample objective
    /// value (MSE for squared error/custom, Huber loss for Huber, Poisson
    /// NLL for Poisson). Huber must monitor its own loss, not MSE: MSE is
    /// dominated by exactly the large-residual outliers the Huber objective
    /// is meant to be insensitive to, so an MSE monitor could stop (or
    /// refuse to stop) on outlier noise the model isn't even fitting.
    #[inline]
    fn monitor_loss(&self, p: f64, y: f64) -> f64 {
        match self {
            Objective::Poisson => p.min(30.0).exp() - y * p,
            Objective::Huber { delta } => {
                let r = (p - y).abs();
                if r <= *delta {
                    0.5 * r * r
                } else {
                    delta * (r - 0.5 * delta)
                }
            }
            _ => (p - y).powi(2),
        }
    }

    fn transform(&self) -> PredTransform {
        match self {
            Objective::Poisson => PredTransform::Exp,
            _ => PredTransform::Identity,
        }
    }
}

/// Output transform applied to raw regression scores at prediction time.
#[derive(Default, Clone, Copy, Serialize, Deserialize)]
pub(crate) enum PredTransform {
    #[default]
    Identity,
    Exp,
}

impl PredTransform {
    #[inline]
    fn apply(&self, v: f64) -> f64 {
        match self {
            PredTransform::Identity => v,
            PredTransform::Exp => v.min(30.0).exp(),
        }
    }
}

/// XGBoost learner (eXtreme Gradient Boosting).
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use ndarray::array;
///
/// let features = array![
///     [0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [0.0, 0.2],
///     [1.0, 1.0], [1.1, 0.9], [0.9, 1.1], [1.0, 0.9]
/// ];
/// let target = vec![0, 0, 0, 0, 1, 1, 1, 1];
/// let task = ClassificationTask::new("xgb", features, target).unwrap();
///
/// let mut xgb = XGBoost::new()
///     .with_n_estimators(50)
///     .with_max_depth(3)
///     .with_learning_rate(0.3);
/// let model = xgb.train_classif(&task).unwrap();
/// ```
pub struct XGBoost {
    n_estimators: usize,
    learning_rate: f64,
    max_depth: usize,
    lambda: f64,
    alpha: f64,
    gamma: f64,
    subsample: f64,
    colsample_bytree: f64,
    min_child_weight: f64,
    n_bins: usize,
    early_stopping_rounds: usize,
    seed: u64,
    /// Optional per-sample weights (regression). When set, each sample's gradient
    /// and hessian are scaled by its weight — the standard way to fit a weighted
    /// objective. Used by GeoXGBoost to apply the bi-square spatial kernel.
    sample_weight: Option<Vec<f64>>,
    /// Optional held-out set for early stopping. When set, `early_stopping_rounds`
    /// monitors loss on this set instead of the training set. Training loss under
    /// boosting is (near-)monotonically decreasing, so without a validation set
    /// early stopping rarely plateaus and almost never actually fires — it isn't
    /// a substitute for evaluating on unseen data.
    eval_set: EvalSet,
    /// Optional per-feature monotone constraints (+1 increasing, -1 decreasing,
    /// 0 unconstrained). Splits violating the direction are rejected and leaf
    /// weights are clamped to propagated bounds, as in official XGBoost.
    monotone_constraints: Option<Vec<i8>>,
    /// Regression objective (gradient/hessian definition). Only affects
    /// `train_regress`; classification always uses log-loss/softmax.
    objective: Objective,
}

impl Default for XGBoost {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.3,
            max_depth: 6,
            lambda: 1.0,
            alpha: 0.0,
            gamma: 0.0,
            subsample: 1.0,
            colsample_bytree: 1.0,
            min_child_weight: 1.0,
            n_bins: 256,
            early_stopping_rounds: 0,
            seed: 42,
            sample_weight: None,
            eval_set: None,
            monotone_constraints: None,
            objective: Objective::SquaredError,
        }
    }
}

impl XGBoost {
    /// Creates an `XGBoost` learner with default hyperparameters.
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets the number of boosting rounds (trees to fit).
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }
    /// Sets the shrinkage applied to each tree's contribution.
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }
    /// Sets the maximum depth of each tree.
    pub fn with_max_depth(mut self, d: usize) -> Self {
        self.max_depth = d;
        self
    }
    /// Sets the L2 regularization strength on leaf weights.
    pub fn with_lambda(mut self, l: f64) -> Self {
        self.lambda = l;
        self
    }
    /// Sets the L1 regularization strength on leaf weights.
    pub fn with_alpha(mut self, a: f64) -> Self {
        self.alpha = a;
        self
    }
    /// Sets the minimum loss reduction required to make a further split
    /// (the gamma / complexity penalty per split).
    pub fn with_gamma(mut self, g: f64) -> Self {
        self.gamma = g;
        self
    }
    /// Sets the fraction of rows randomly subsampled for each tree.
    pub fn with_subsample(mut self, s: f64) -> Self {
        self.subsample = s;
        self
    }
    /// Sets the fraction of columns randomly sampled for each tree.
    pub fn with_colsample_bytree(mut self, c: f64) -> Self {
        self.colsample_bytree = c;
        self
    }
    /// Sets the minimum sum of Hessian (instance weight) required in a leaf
    /// for a split to be considered.
    pub fn with_min_child_weight(mut self, w: f64) -> Self {
        self.min_child_weight = w;
        self
    }
    /// Stop after `n` rounds without improvement of the monitored loss
    /// (held-out loss when an eval set is provided, training loss otherwise).
    pub fn with_early_stopping_rounds(mut self, n: usize) -> Self {
        self.early_stopping_rounds = n;
        self
    }
    /// Sets the RNG seed controlling row/column subsampling.
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }
    /// Set per-sample weights for (weighted) regression. Length must match the
    /// number of training samples; each gradient/hessian is scaled by its weight.
    pub fn with_sample_weights(mut self, w: Vec<f64>) -> Self {
        self.sample_weight = Some(w);
        self
    }
    /// Per-feature monotone constraints: +1 forces the prediction to be
    /// non-decreasing in that feature, -1 non-increasing, 0 unconstrained.
    /// Length must match the number of features; applies to numeric features
    /// (categorical splits ignore it).
    pub fn with_monotone_constraints(mut self, c: Vec<i8>) -> Self {
        self.monotone_constraints = Some(c);
        self
    }
    /// Regression objective: `Objective::SquaredError` (default),
    /// `Huber { delta }`, `Poisson`, or `Custom(f)` where
    /// `f(prediction, target) -> (gradient, hessian)`. Only affects
    /// `train_regress`.
    ///
    /// Note on early stopping with `Custom`: a custom objective supplies
    /// only gradients/hessians, not a loss value, so `early_stopping_rounds`
    /// monitors MSE computed on the *raw* (untransformed) score as a proxy.
    /// If your custom loss ranks models differently from squared error
    /// (e.g. an asymmetric or robust loss), early stopping may stop on the
    /// wrong round for it — consider disabling early stopping or validating
    /// externally. `Huber`/`Poisson` monitor their own loss.
    pub fn with_objective(mut self, o: Objective) -> Self {
        self.objective = o;
        self
    }
    /// Set a held-out set that `early_stopping_rounds` evaluates on, for
    /// regression. Without this, early stopping monitors training loss, which
    /// rarely plateaus under boosting and so rarely actually fires.
    pub fn with_eval_set_regress(mut self, features: Array2<f64>, target: Vec<f64>) -> Self {
        self.eval_set = Some((features, EvalTarget::Regression(target)));
        self
    }
    /// Set a held-out set that `early_stopping_rounds` evaluates on, for
    /// classification (binary or multiclass). Without this, early stopping
    /// monitors training loss, which rarely plateaus under boosting and so
    /// rarely actually fires.
    pub fn with_eval_set_classif(mut self, features: Array2<f64>, target: Vec<usize>) -> Self {
        self.eval_set = Some((features, EvalTarget::Classification(target)));
        self
    }
}

// ── Histogram binning (NaN-aware, column-major, u8 packed) ──────────
//
type FeatureBins = HistBins;

// ── XGBoost tree node ───────────────────────────────────────────────

/// Internal tree node: a leaf with a fitted value, or a split on a feature
/// (numeric threshold or categorical membership).
#[derive(Clone, Serialize, Deserialize)]
pub enum XGBNode {
    /// Terminal node holding the leaf's fitted output value.
    Leaf {
        /// The leaf's fitted output value.
        weight: f64,
    },
    /// Numeric split: rows with `feature < threshold` go left, others right.
    Split {
        /// Index of the feature being split on.
        feature: usize,
        /// Threshold value separating left and right children.
        threshold: f64,
        /// Whether NaN values in `feature` route to the left child.
        nan_goes_left: bool,
        /// Left child, taken when the row's value is below `threshold`.
        left: Box<XGBNode>,
        /// Right child, taken when the row's value is at or above `threshold`.
        right: Box<XGBNode>,
    },
    /// Categorical split: the listed category codes go left; every other
    /// code — including categories unseen during training — goes right.
    CatSplit {
        /// Index of the feature being split on.
        feature: usize,
        /// Sorted category codes routed left.
        left_cats: Vec<u16>,
        /// Whether NaN values in `feature` route to the left child.
        nan_goes_left: bool,
        /// Left child, taken when the row's category code is in `left_cats`.
        left: Box<XGBNode>,
        /// Right child, taken for all other category codes.
        right: Box<XGBNode>,
    },
}

impl XGBNode {
    #[inline]
    fn predict_one(&self, sample: ArrayView1<f64>) -> f64 {
        match self {
            XGBNode::Leaf { weight } => *weight,
            XGBNode::Split {
                feature,
                threshold,
                nan_goes_left,
                left,
                right,
            } => {
                let val = sample[*feature];
                if val.is_nan() {
                    if *nan_goes_left {
                        left.predict_one(sample)
                    } else {
                        right.predict_one(sample)
                    }
                } else if val < *threshold {
                    left.predict_one(sample)
                } else {
                    right.predict_one(sample)
                }
            }
            XGBNode::CatSplit {
                feature,
                left_cats,
                nan_goes_left,
                left,
                right,
            } => {
                let val = sample[*feature];
                let goes_left = if val.is_nan() {
                    *nan_goes_left
                } else {
                    left_cats.binary_search(&(val as u16)).is_ok()
                };
                if goes_left {
                    left.predict_one(sample)
                } else {
                    right.predict_one(sample)
                }
            }
        }
    }
}

// ── Best split info ─────────────────────────────────────────────────

struct BestSplit {
    feature: usize,
    threshold: f64,
    gain: f64,
    nan_goes_left: bool,
    split_bin: usize,
    /// `Some(sorted category codes going left)` for categorical splits.
    left_cats: Option<Vec<u16>>,
}

impl BestSplit {
    /// Does bin `b` (never `NAN_BIN`) route left under this split?
    #[inline]
    fn bin_goes_left(&self, b: u8) -> bool {
        match &self.left_cats {
            Some(cats) => cats.binary_search(&(b as u16)).is_ok(),
            None => (b as usize) <= self.split_bin,
        }
    }

    /// Build the split node for this best split with the given children.
    fn into_node(self, left: XGBNode, right: XGBNode) -> XGBNode {
        match self.left_cats {
            Some(left_cats) => XGBNode::CatSplit {
                feature: self.feature,
                left_cats,
                nan_goes_left: self.nan_goes_left,
                left: Box::new(left),
                right: Box::new(right),
            },
            None => XGBNode::Split {
                feature: self.feature,
                threshold: self.threshold,
                nan_goes_left: self.nan_goes_left,
                left: Box::new(left),
                right: Box::new(right),
            },
        }
    }
}

// ── XGBoost tree builder ────────────────────────────────────────────

struct XGBTreeBuilder<'a> {
    features: &'a Array2<f64>,
    bins: &'a FeatureBins,
    grads: &'a [f64],
    hess: &'a [f64],
    lambda: f64,
    alpha: f64,
    gamma: f64,
    max_depth: usize,
    min_child_weight: f64,
    col_indices: Vec<usize>,
    use_exact: bool,
    feature_importances: Vec<f64>,
    pool: HistPool,
    /// Per-feature monotone constraints (+1 increasing, -1 decreasing,
    /// 0 none). Empty slice = unconstrained. Applies to numeric features;
    /// categorical splits ignore it (monotonicity is meaningless on codes).
    constraints: &'a [i8],
}

impl<'a> XGBTreeBuilder<'a> {
    #[inline]
    fn leaf_weight_gh(&self, g: f64, h: f64) -> f64 {
        let gt = if g > self.alpha {
            g - self.alpha
        } else if g < -self.alpha {
            g + self.alpha
        } else {
            0.0
        };
        -gt / (h + self.lambda)
    }

    /// Leaf weight clamped to the node's monotone bounds.
    #[inline]
    fn bounded_leaf(&self, g: f64, h: f64, lo: f64, hi: f64) -> f64 {
        self.leaf_weight_gh(g, h).clamp(lo, hi)
    }

    #[inline]
    fn split_gain(&self, gl: f64, hl: f64, gr: f64, hr: f64) -> f64 {
        0.5 * (gl * gl / (hl + self.lambda) + gr * gr / (hr + self.lambda)
            - (gl + gr) * (gl + gr) / (hl + hr + self.lambda))
            - self.gamma
    }

    /// True when splitting `feat` with these child stats would violate its
    /// monotone constraint (standard XGBoost check: compare child weights).
    #[inline]
    fn violates_monotone(&self, feat: usize, gl: f64, hl: f64, gr: f64, hr: f64) -> bool {
        let c = self.constraints.get(feat).copied().unwrap_or(0);
        if c == 0 {
            return false;
        }
        let wl = self.leaf_weight_gh(gl, hl);
        let wr = self.leaf_weight_gh(gr, hr);
        if c > 0 { wl > wr } else { wl < wr }
    }

    /// Bounds for the children of a node split on `feat` given the child
    /// gradient/hessian sums: unconstrained features pass the parent bounds
    /// through; constrained features cap the increasing side at the midpoint
    /// of the (clamped) child weights, as in official XGBoost.
    fn child_bounds_gh(
        &self,
        feat: usize,
        lo: f64,
        hi: f64,
        lg: f64,
        lh: f64,
        rg: f64,
        rh: f64,
    ) -> ((f64, f64), (f64, f64)) {
        let c = self.constraints.get(feat).copied().unwrap_or(0);
        if c == 0 {
            return ((lo, hi), (lo, hi));
        }
        let wl = self.bounded_leaf(lg, lh, lo, hi);
        let wr = self.bounded_leaf(rg, rh, lo, hi);
        let mid = (wl + wr) / 2.0;
        if c > 0 {
            ((lo, hi.min(mid)), (lo.max(mid), hi))
        } else {
            ((lo.max(mid), hi), (lo, hi.min(mid)))
        }
    }

    fn build(
        &mut self,
        indices: &mut Vec<usize>,
        start: usize,
        end: usize,
        depth: usize,
        lo: f64,
        hi: f64,
    ) -> XGBNode {
        if self.use_exact {
            return self.build_exact(indices, start, end, depth, lo, hi);
        }
        self.build_hist_sub(indices, start, end, depth, false, lo, hi)
    }

    /// Build with histogram subtraction.
    /// `hist_ready`: pool[depth] already populated from subtraction (skip scan).
    #[allow(clippy::too_many_arguments)]
    fn build_hist_sub(
        &mut self,
        indices: &mut Vec<usize>,
        start: usize,
        end: usize,
        depth: usize,
        hist_ready: bool,
        lo: f64,
        hi: f64,
    ) -> XGBNode {
        let n = end - start;
        let h_sum: f64 = indices[start..end].iter().map(|&i| self.hess[i]).sum();
        if depth >= self.max_depth || n <= 1 || h_sum < self.min_child_weight {
            let g_sum: f64 = indices[start..end].iter().map(|&i| self.grads[i]).sum();
            return XGBNode::Leaf {
                weight: self.bounded_leaf(g_sum, h_sum, lo, hi),
            };
        }

        let best = if hist_ready {
            // Subtracted histogram already in pool — just find best split
            let r = self.pool.find_best(
                depth,
                &self.col_indices,
                self.bins,
                self.min_child_weight,
                self.lambda,
                self.alpha,
                self.gamma,
                self.constraints,
            );
            r.map(|(feat, thr, gain, nl, sb, lc)| BestSplit {
                feature: feat,
                threshold: thr,
                gain,
                nan_goes_left: nl,
                split_bin: sb,
                left_cats: lc,
            })
        } else {
            // Original par_iter scan+find (fast!) — also capture histograms
            let (split, hists) = self.find_best_histogram_saving(&indices[start..end]);
            self.pool.store_hists(depth, &hists);
            split
        };

        let best = match best {
            Some(b) if b.gain > 0.0 => b,
            _ => {
                let g_sum: f64 = indices[start..end].iter().map(|&i| self.grads[i]).sum();
                return XGBNode::Leaf {
                    weight: self.bounded_leaf(g_sum, h_sum, lo, hi),
                };
            }
        };
        let feat = best.feature;

        self.feature_importances[feat] += best.gain;

        // Partition
        let (mut left_end, mut i) = (start, start);
        while i < end {
            let b = self.bins.get_bin(feat, indices[i]);
            let goes_left = if b == NAN_BIN {
                best.nan_goes_left
            } else {
                best.bin_goes_left(b)
            };
            if goes_left {
                indices.swap(left_end, i);
                left_end += 1;
            }
            i += 1;
        }

        if left_end == start || left_end == end {
            let g_sum: f64 = indices[start..end].iter().map(|&i| self.grads[i]).sum();
            return XGBNode::Leaf {
                weight: self.bounded_leaf(g_sum, h_sum, lo, hi),
            };
        }

        let constrained = self.constraints.get(feat).copied().unwrap_or(0) != 0;
        let children_are_leaves = depth + 1 >= self.max_depth;

        // Child g/h sums are only needed for leaf weights or monotone bounds —
        // skip the four O(n) passes on the common unconstrained inner node.
        let ((llo, lhi), (rlo, rhi), gh) = if constrained || children_are_leaves {
            let lg: f64 = indices[start..left_end].iter().map(|&i| self.grads[i]).sum();
            let lh: f64 = indices[start..left_end].iter().map(|&i| self.hess[i]).sum();
            let rg: f64 = indices[left_end..end].iter().map(|&i| self.grads[i]).sum();
            let rh: f64 = indices[left_end..end].iter().map(|&i| self.hess[i]).sum();
            let (lb, rb) = self.child_bounds_gh(feat, lo, hi, lg, lh, rg, rh);
            (lb, rb, Some((lg, lh, rg, rh)))
        } else {
            ((lo, hi), (lo, hi), None)
        };

        if children_are_leaves {
            // Children are leaves — skip histogram work
            let (lg, lh, rg, rh) = gh.expect("gh computed when children are leaves");
            let left = XGBNode::Leaf {
                weight: self.bounded_leaf(lg, lh, llo, lhi),
            };
            let right = XGBNode::Leaf {
                weight: self.bounded_leaf(rg, rh, rlo, rhi),
            };
            return best.into_node(left, right);
        }

        let left_count = left_end - start;
        let right_count = end - left_end;

        // Subtraction: scan+save SMALLER, process, subtract → LARGER (skip scan).
        // Only valid if the smaller side actually scans and stores its histogram —
        // if it hits the early-leaf return in build_hist_sub (n<=1 or h_sum <
        // min_child_weight; depth>=max_depth is impossible here since
        // children_are_leaves was already handled above), pool[depth+1] is never
        // written and subtract_in_place would produce garbage from a stale level.
        // Detect that case ahead of time and fall back to an explicit scan for
        // the larger sibling too.
        if left_count <= right_count {
            let left_h_sum: f64 = indices[start..left_end].iter().map(|&i| self.hess[i]).sum();
            let left_is_trivial = left_count <= 1 || left_h_sum < self.min_child_weight;
            let left = self.build_hist_sub(indices, start, left_end, depth + 1, false, llo, lhi);
            let right = if left_is_trivial {
                self.build_hist_sub(indices, left_end, end, depth + 1, false, rlo, rhi)
            } else {
                self.pool.subtract_in_place(depth, depth + 1);
                self.build_hist_sub(indices, left_end, end, depth + 1, true, rlo, rhi)
            };
            best.into_node(left, right)
        } else {
            let right_h_sum: f64 = indices[left_end..end].iter().map(|&i| self.hess[i]).sum();
            let right_is_trivial = right_count <= 1 || right_h_sum < self.min_child_weight;
            let right = self.build_hist_sub(indices, left_end, end, depth + 1, false, rlo, rhi);
            let left = if right_is_trivial {
                self.build_hist_sub(indices, start, left_end, depth + 1, false, llo, lhi)
            } else {
                self.pool.subtract_in_place(depth, depth + 1);
                self.build_hist_sub(indices, start, left_end, depth + 1, true, llo, lhi)
            };
            best.into_node(left, right)
        }
    }

    /// Exact greedy build.
    fn build_exact(
        &mut self,
        indices: &mut Vec<usize>,
        start: usize,
        end: usize,
        depth: usize,
        lo: f64,
        hi: f64,
    ) -> XGBNode {
        let n = end - start;
        let h_sum: f64 = indices[start..end].iter().map(|&i| self.hess[i]).sum();
        if depth >= self.max_depth || n <= 1 || h_sum < self.min_child_weight {
            let g_sum: f64 = indices[start..end].iter().map(|&i| self.grads[i]).sum();
            return XGBNode::Leaf {
                weight: self.bounded_leaf(g_sum, h_sum, lo, hi),
            };
        }
        let best = match self.find_best_exact(&indices[start..end]) {
            Some(b) if b.gain > 0.0 => b,
            _ => {
                let g_sum: f64 = indices[start..end].iter().map(|&i| self.grads[i]).sum();
                return XGBNode::Leaf {
                    weight: self.bounded_leaf(g_sum, h_sum, lo, hi),
                };
            }
        };
        self.feature_importances[best.feature] += best.gain;
        let feat = best.feature;
        let (mut left_end, mut i) = (start, start);
        while i < end {
            let v = self.features[[indices[i], feat]];
            let goes_left = if v.is_nan() {
                best.nan_goes_left
            } else {
                match &best.left_cats {
                    Some(cats) => cats.binary_search(&(v as u16)).is_ok(),
                    None => v < best.threshold,
                }
            };
            if goes_left {
                indices.swap(left_end, i);
                left_end += 1;
            }
            i += 1;
        }
        if left_end == start || left_end == end {
            let g_sum: f64 = indices[start..end].iter().map(|&i| self.grads[i]).sum();
            return XGBNode::Leaf {
                weight: self.bounded_leaf(g_sum, h_sum, lo, hi),
            };
        }
        let ((llo, lhi), (rlo, rhi)) =
            if self.constraints.get(feat).copied().unwrap_or(0) != 0 {
                let lg: f64 = indices[start..left_end].iter().map(|&i| self.grads[i]).sum();
                let lh: f64 = indices[start..left_end].iter().map(|&i| self.hess[i]).sum();
                let rg: f64 = indices[left_end..end].iter().map(|&i| self.grads[i]).sum();
                let rh: f64 = indices[left_end..end].iter().map(|&i| self.hess[i]).sum();
                self.child_bounds_gh(feat, lo, hi, lg, lh, rg, rh)
            } else {
                ((lo, hi), (lo, hi))
            };
        let left = self.build_exact(indices, start, left_end, depth + 1, llo, lhi);
        let right = self.build_exact(indices, left_end, end, depth + 1, rlo, rhi);
        best.into_node(left, right)
    }

    /// Histogram split finding — parallel per feature, also returns per-feature
    /// histogram data for the pool (used for parent/child subtraction).
    fn find_best_histogram_saving(
        &self,
        node_indices: &[usize],
    ) -> (Option<BestSplit>, Vec<(Vec<f64>, Vec<f64>, f64, f64)>) {
        let results: Vec<(Option<BestSplit>, (Vec<f64>, Vec<f64>, f64, f64))> = self
            .col_indices
            .par_iter()
            .map(|&feat| {
                let (bin_g, bin_h, nan_g, nan_h) =
                    accumulate_histogram(self.bins, feat, self.grads, self.hess, None, node_indices);

                if self.bins.cat[feat].is_some() {
                    let split = best_categorical_split(
                        &bin_g,
                        &bin_h,
                        nan_g,
                        nan_h,
                        self.min_child_weight,
                        |gl, hl, gr, hr| self.split_gain(gl, hl, gr, hr),
                    )
                    .map(|(left_cats, gain, nan_left)| BestSplit {
                        feature: feat,
                        threshold: f64::NAN,
                        gain,
                        nan_goes_left: nan_left,
                        split_bin: 0,
                        left_cats: Some(left_cats),
                    });
                    return (split, (bin_g, bin_h, nan_g, nan_h));
                }

                let best = best_numeric_split(
                    &bin_g,
                    &bin_h,
                    nan_g,
                    nan_h,
                    self.min_child_weight,
                    |gl, hl, gr, hr| {
                        if self.violates_monotone(feat, gl, hl, gr, hr) {
                            None
                        } else {
                            Some(self.split_gain(gl, hl, gr, hr))
                        }
                    },
                );

                let split = best.map(|(bin, gain, nan_left)| BestSplit {
                    feature: feat,
                    threshold: self.bins.bin_threshold(feat, bin),
                    gain,
                    nan_goes_left: nan_left,
                    split_bin: bin,
                    left_cats: None,
                });
                (split, (bin_g, bin_h, nan_g, nan_h))
            })
            .collect();

        let mut hists = Vec::with_capacity(results.len());
        let mut best_split: Option<BestSplit> = None;
        for (split, hist) in results {
            hists.push(hist);
            if let Some(s) = split
                && best_split.as_ref().is_none_or(|b| s.gain > b.gain)
            {
                best_split = Some(s);
            }
        }
        (best_split, hists)
    }

    /// Exact greedy split — parallel per feature.
    fn find_best_exact(&self, node_indices: &[usize]) -> Option<BestSplit> {
        let features = self.features;
        let results: Vec<Option<BestSplit>> = self
            .col_indices
            .par_iter()
            .map(|&feat| {
                if let Some(nc) = self.bins.cat[feat] {
                    // Categorical: aggregate g/h by code and Fisher-scan.
                    let mut bin_g = vec![0.0; nc];
                    let mut bin_h = vec![0.0; nc];
                    let (mut nan_g, mut nan_h) = (0.0, 0.0);
                    for &i in node_indices {
                        let v = features[[i, feat]];
                        if v.is_nan() {
                            nan_g += self.grads[i];
                            nan_h += self.hess[i];
                        } else {
                            let c = (v as usize).min(nc - 1);
                            bin_g[c] += self.grads[i];
                            bin_h[c] += self.hess[i];
                        }
                    }
                    return best_categorical_split(
                        &bin_g,
                        &bin_h,
                        nan_g,
                        nan_h,
                        self.min_child_weight,
                        |gl, hl, gr, hr| self.split_gain(gl, hl, gr, hr),
                    )
                    .map(|(left_cats, gain, nan_left)| BestSplit {
                        feature: feat,
                        threshold: f64::NAN,
                        gain,
                        nan_goes_left: nan_left,
                        split_bin: 0,
                        left_cats: Some(left_cats),
                    });
                }

                let mut sorted: Vec<usize> = node_indices
                    .iter()
                    .filter(|&&i| !features[[i, feat]].is_nan())
                    .copied()
                    .collect();
                sorted.sort_by(|&a, &b| {
                    features[[a, feat]]
                        .partial_cmp(&features[[b, feat]])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                let nan_g: f64 = node_indices
                    .iter()
                    .filter(|&&i| features[[i, feat]].is_nan())
                    .map(|&i| self.grads[i])
                    .sum();
                let nan_h: f64 = node_indices
                    .iter()
                    .filter(|&&i| features[[i, feat]].is_nan())
                    .map(|&i| self.hess[i])
                    .sum();

                if sorted.len() <= 1 {
                    return None;
                }
                let total_g: f64 = node_indices.iter().map(|&i| self.grads[i]).sum();
                let total_h: f64 = node_indices.iter().map(|&i| self.hess[i]).sum();

                let mut best_gain = 0.0;
                let mut best: Option<(f64, f64, bool)> = None;
                let (mut gl, mut hl) = (0.0, 0.0);

                for i in 0..(sorted.len() - 1) {
                    gl += self.grads[sorted[i]];
                    hl += self.hess[sorted[i]];
                    if (features[[sorted[i + 1], feat]] - features[[sorted[i], feat]]).abs()
                        < f64::EPSILON
                    {
                        continue;
                    }
                    let (gr, hr) = (total_g - gl, total_h - hl);
                    if hl >= self.min_child_weight
                        && hr >= self.min_child_weight
                        && !self.violates_monotone(feat, gl, hl, gr, hr)
                    {
                        let gain = self.split_gain(gl, hl, gr, hr);
                        if gain > best_gain {
                            best_gain = gain;
                            let t = (features[[sorted[i], feat]] + features[[sorted[i + 1], feat]])
                                / 2.0;
                            best = Some((t, gain, false));
                        }
                    }
                    if nan_h > 0.0 {
                        let (gln, hln) = (gl + nan_g, hl + nan_h);
                        let (grn, hrn) = (total_g - gln, total_h - hln);
                        if hln >= self.min_child_weight
                            && hrn >= self.min_child_weight
                            && !self.violates_monotone(feat, gln, hln, grn, hrn)
                        {
                            let gain = self.split_gain(gln, hln, grn, hrn);
                            if gain > best_gain {
                                best_gain = gain;
                                let t = (features[[sorted[i], feat]]
                                    + features[[sorted[i + 1], feat]])
                                    / 2.0;
                                best = Some((t, gain, true));
                            }
                        }
                    }
                }

                best.map(|(threshold, gain, nan_left)| BestSplit {
                    feature: feat,
                    threshold,
                    gain,
                    nan_goes_left: nan_left,
                    split_bin: 0,
                    left_cats: None,
                })
            })
            .collect();

        results.into_iter().flatten().max_by(|a, b| {
            a.gain
                .partial_cmp(&b.gain)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

// ── Trained model ───────────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub(crate) enum XGBMode {
    Regression,
    BinaryClassif,
    MultiClassif { n_classes: usize },
}

/// A trained XGBoost model, ready to predict.
#[derive(Clone, Serialize, Deserialize)]
pub struct TrainedXGBoost {
    pub(crate) trees: Vec<XGBNode>,
    pub(crate) initial: Vec<f64>,
    pub(crate) learning_rate: f64,
    pub(crate) mode: XGBMode,
    pub(crate) feature_names: Vec<String>,
    pub(crate) feature_importances: Vec<f64>,
    /// Output transform for regression scores (e.g. exp for Poisson).
    /// `serde(default)` = Identity, so pre-existing serialized models load.
    #[serde(default)]
    pub(crate) transform: PredTransform,
}

impl TrainedModel for TrainedXGBoost {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        crate::validate::check_n_features(features, self.feature_names.len())?;
        match &self.mode {
            XGBMode::Regression => {
                let predicted: Vec<f64> = (0..features.nrows())
                    .into_par_iter()
                    .map(|i| {
                        let row = features.row(i);
                        let mut v = self.initial[0];
                        for t in &self.trees {
                            v += self.learning_rate * t.predict_one(row);
                        }
                        self.transform.apply(v)
                    })
                    .collect();
                Ok(Prediction::regression(predicted))
            }
            XGBMode::BinaryClassif => {
                let results: Vec<(usize, Vec<f64>)> = (0..features.nrows())
                    .into_par_iter()
                    .map(|i| {
                        let row = features.row(i);
                        let mut f = self.initial[0];
                        for t in &self.trees {
                            f += self.learning_rate * t.predict_one(row);
                        }
                        let p = sigmoid(f);
                        (if p >= 0.5 { 1 } else { 0 }, vec![1.0 - p, p])
                    })
                    .collect();
                let mut predicted = Vec::with_capacity(results.len());
                let mut probabilities = Vec::with_capacity(results.len());
                for (pred, prob) in results {
                    predicted.push(pred);
                    probabilities.push(prob);
                }
                Ok(Prediction::Classification {
                    predicted,
                    truth: None,
                    probabilities: Some(probabilities),
                })
            }
            XGBMode::MultiClassif { n_classes } => {
                let k = *n_classes;
                let n_iters = self.trees.len() / k;
                let results: Vec<(usize, Vec<f64>)> = (0..features.nrows())
                    .into_par_iter()
                    .map(|i| {
                        let row = features.row(i);
                        let mut scores = self.initial.clone();
                        for iter in 0..n_iters {
                            for c in 0..k {
                                scores[c] +=
                                    self.learning_rate * self.trees[iter * k + c].predict_one(row);
                            }
                        }
                        let probs = softmax(&scores);
                        let pred = probs
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                            .unwrap()
                            .0;
                        (pred, probs)
                    })
                    .collect();
                let mut predicted = Vec::with_capacity(results.len());
                let mut probabilities = Vec::with_capacity(results.len());
                for (pred, probs) in results {
                    predicted.push(pred);
                    probabilities.push(probs);
                }
                Ok(Prediction::Classification {
                    predicted,
                    truth: None,
                    probabilities: Some(probabilities),
                })
            }
        }
    }

    fn feature_importance(&self) -> Option<Vec<(String, f64)>> {
        let total: f64 = self.feature_importances.iter().sum();
        if total == 0.0 {
            return None;
        }
        Some(
            self.feature_names
                .iter()
                .zip(&self.feature_importances)
                .map(|(n, &i)| (n.clone(), i / total))
                .collect(),
        )
    }

    fn to_serializable(&self) -> Option<crate::serialize::SerializableModel> {
        Some(crate::serialize::SerializableModel::XGBoost(self.clone()))
    }
}

// ── Learner ─────────────────────────────────────────────────────────

impl XGBoost {
    fn sample(
        &self,
        rng: &mut StdRng,
        n_samples: usize,
        n_features: usize,
    ) -> (Vec<usize>, Vec<usize>) {
        let rows = if self.subsample < 1.0 {
            let k = (n_samples as f64 * self.subsample).ceil() as usize;
            let mut v: Vec<usize> = (0..n_samples).collect();
            v.shuffle(rng);
            v.truncate(k);
            v
        } else {
            (0..n_samples).collect()
        };
        let cols = if self.colsample_bytree < 1.0 {
            let k = (n_features as f64 * self.colsample_bytree).ceil().max(1.0) as usize;
            let mut v: Vec<usize> = (0..n_features).collect();
            v.shuffle(rng);
            v.truncate(k);
            v.sort();
            v
        } else {
            (0..n_features).collect()
        };
        (rows, cols)
    }

    fn build_one_tree(
        &self,
        features: &Array2<f64>,
        bins: &FeatureBins,
        grads: &[f64],
        hess: &[f64],
        indices: &mut Vec<usize>,
        col_indices: Vec<usize>,
        n_features: usize,
    ) -> (XGBNode, Vec<f64>) {
        let n = indices.len();
        let use_exact = n <= self.n_bins;
        let max_bins = if use_exact {
            1
        } else {
            bins.boundaries
                .iter()
                .map(|b| b.len() + 1)
                .max()
                .unwrap_or(256)
        };
        let n_col = col_indices.len();
        let mut builder = XGBTreeBuilder {
            features,
            bins,
            grads,
            hess,
            lambda: self.lambda,
            alpha: self.alpha,
            gamma: self.gamma,
            max_depth: self.max_depth,
            min_child_weight: self.min_child_weight,
            col_indices,
            use_exact,
            feature_importances: vec![0.0; n_features],
            pool: HistPool::new(self.max_depth, n_col, max_bins),
            constraints: self.monotone_constraints.as_deref().unwrap_or(&[]),
        };
        let tree = builder.build(indices, 0, n, 0, f64::NEG_INFINITY, f64::INFINITY);
        (tree, builder.feature_importances)
    }

    /// Validate `monotone_constraints` against the task's feature count and
    /// values (+1/-1/0 only).
    fn validate_constraints(&self, n_features: usize) -> Result<()> {
        if let Some(c) = &self.monotone_constraints {
            if c.len() != n_features {
                return Err(SmeltError::DimensionMismatch {
                    expected: n_features,
                    got: c.len(),
                });
            }
            if c.iter().any(|&v| !(-1..=1).contains(&v)) {
                return Err(SmeltError::InvalidParameter(
                    "monotone constraints must be -1, 0, or +1".into(),
                ));
            }
        }
        Ok(())
    }

}

impl Learner for XGBoost {
    fn id(&self) -> &str {
        "xgboost"
    }

    fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
        let features = task.features();
        let target = task.target();
        let (ns, nf) = (task.n_samples(), task.n_features());
        if let Some(w) = &self.sample_weight
            && w.len() != ns {
                return Err(SmeltError::DimensionMismatch {
                    expected: ns,
                    got: w.len(),
                });
            }
        let eval = validate_eval_regress(&self.eval_set, nf)?;
        self.validate_constraints(nf)?;
        let bins = HistBins::build_typed(features, self.n_bins, task.feature_types());
        // Weighted mean as the initial prediction when sample weights are set;
        // the objective maps it to its raw-score space (e.g. log for Poisson).
        let target_mean = match &self.sample_weight {
            Some(w) => {
                let wsum: f64 = w.iter().sum();
                if wsum > 0.0 {
                    target.iter().zip(w).map(|(y, wi)| y * wi).sum::<f64>() / wsum
                } else {
                    target.iter().sum::<f64>() / ns as f64
                }
            }
            None => target.iter().sum::<f64>() / ns as f64,
        };
        let initial = self.objective.initial_score(target_mean);
        let mut preds = vec![initial; ns];
        let mut eval_preds = eval.map(|(ef, _)| vec![initial; ef.nrows()]);
        let mut trees = Vec::with_capacity(self.n_estimators);
        let mut imp = vec![0.0; nf];
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut stopper = EarlyStopper::new(self.early_stopping_rounds);

        for round in 0..self.n_estimators {
            let (mut grads, mut hess): (Vec<f64>, Vec<f64>) = preds
                .iter()
                .zip(target)
                .map(|(&p, &y)| self.objective.grad_hess(p, y))
                .unzip();
            if let Some(w) = &self.sample_weight {
                // Scale gradient and hessian by the sample weight: this fits the
                // weighted least-squares objective (XGBoost's standard mechanism).
                for i in 0..ns {
                    grads[i] *= w[i];
                    hess[i] *= w[i];
                }
            }
            let (mut idx, cols) = self.sample(&mut rng, ns, nf);
            let (tree, fi) =
                self.build_one_tree(features, &bins, &grads, &hess, &mut idx, cols, nf);
            for i in 0..ns {
                preds[i] += self.learning_rate * tree.predict_one(features.row(i));
            }
            if let (Some(ep), Some((ef, _))) = (&mut eval_preds, eval) {
                for i in 0..ef.nrows() {
                    ep[i] += self.learning_rate * tree.predict_one(ef.row(i));
                }
            }
            for (j, v) in fi.iter().enumerate() {
                imp[j] += v;
            }
            trees.push(tree);

            if stopper.is_active() {
                // Evaluate on the held-out set when one was provided —
                // training loss is (near-)monotonically decreasing under
                // boosting and rarely plateaus, so it's a poor early-stopping
                // signal on its own.
                let loss = if let (Some(ep), Some((_, et))) = (&eval_preds, eval) {
                    ep.iter()
                        .zip(et)
                        .map(|(&p, &y)| self.objective.monitor_loss(p, y))
                        .sum::<f64>()
                        / ep.len() as f64
                } else {
                    match &self.sample_weight {
                        Some(w) => {
                            let wsum: f64 = w.iter().sum::<f64>().max(1e-12);
                            preds
                                .iter()
                                .zip(target)
                                .zip(w)
                                .map(|((&p, &y), wi)| wi * self.objective.monitor_loss(p, y))
                                .sum::<f64>()
                                / wsum
                        }
                        None => {
                            preds
                                .iter()
                                .zip(target)
                                .map(|(&p, &y)| self.objective.monitor_loss(p, y))
                                .sum::<f64>()
                                / ns as f64
                        }
                    }
                };
                if let Some(best_n) = stopper.update(loss, round + 1) {
                    trees.truncate(best_n);
                    break;
                }
            }
        }
        Ok(Box::new(TrainedXGBoost {
            trees,
            initial: vec![initial],
            learning_rate: self.learning_rate,
            mode: XGBMode::Regression,
            feature_names: task.feature_names().to_vec(),
            feature_importances: imp,
            transform: self.objective.transform(),
        }))
    }

    fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        if task.n_classes() == 2 {
            self.train_binary(task)
        } else {
            self.train_multiclass(task)
        }
    }
}

impl XGBoost {
    fn train_binary(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        let features = task.features();
        let target = task.target();
        let (ns, nf) = (task.n_samples(), task.n_features());
        if let Some(w) = &self.sample_weight
            && w.len() != ns {
                return Err(SmeltError::DimensionMismatch {
                    expected: ns,
                    got: w.len(),
                });
            }
        let eval = validate_eval_classif(&self.eval_set, nf)?;
        self.validate_constraints(nf)?;
        let bins = HistBins::build_typed(features, self.n_bins, task.feature_types());
        // Weighted positive-class fraction as the initial log-odds when
        // sample weights are set (same objective as XGBoost's base_score).
        let p_pos = match &self.sample_weight {
            Some(w) => {
                let wsum: f64 = w.iter().sum();
                if wsum > 0.0 {
                    target
                        .iter()
                        .zip(w)
                        .map(|(&t, wi)| if t == 1 { *wi } else { 0.0 })
                        .sum::<f64>()
                        / wsum
                } else {
                    target.iter().filter(|&&t| t == 1).count() as f64 / ns as f64
                }
            }
            None => target.iter().filter(|&&t| t == 1).count() as f64 / ns as f64,
        };
        let initial = (p_pos / (1.0 - p_pos).max(1e-15)).ln();
        let mut fv = vec![initial; ns];
        let mut eval_fv = eval.map(|(ef, _)| vec![initial; ef.nrows()]);
        let mut trees = Vec::with_capacity(self.n_estimators);
        let mut imp = vec![0.0; nf];
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut stopper = EarlyStopper::new(self.early_stopping_rounds);

        for round in 0..self.n_estimators {
            let mut grads: Vec<f64> = (0..ns).map(|i| sigmoid(fv[i]) - target[i] as f64).collect();
            let mut hess: Vec<f64> = (0..ns)
                .map(|i| {
                    let p = sigmoid(fv[i]);
                    p * (1.0 - p).max(1e-15)
                })
                .collect();
            if let Some(w) = &self.sample_weight {
                // Scale gradient and hessian by the sample weight, matching
                // the weighted-least-squares mechanism used in train_regress.
                for i in 0..ns {
                    grads[i] *= w[i];
                    hess[i] *= w[i];
                }
            }
            let (mut idx, cols) = self.sample(&mut rng, ns, nf);
            let (tree, fi) =
                self.build_one_tree(features, &bins, &grads, &hess, &mut idx, cols, nf);
            for i in 0..ns {
                fv[i] += self.learning_rate * tree.predict_one(features.row(i));
            }
            if let (Some(efv), Some((ef, _))) = (&mut eval_fv, eval) {
                for i in 0..ef.nrows() {
                    efv[i] += self.learning_rate * tree.predict_one(ef.row(i));
                }
            }
            for (j, v) in fi.iter().enumerate() {
                imp[j] += v;
            }
            trees.push(tree);

            if stopper.is_active() {
                let eps = 1e-15;
                // Evaluate on the held-out set when one was provided —
                // training loss rarely plateaus under boosting.
                let loss = if let (Some(efv), Some((_, et))) = (&eval_fv, eval) {
                    et.iter()
                        .zip(efv)
                        .map(|(&y, &f)| {
                            let p = sigmoid(f).max(eps).min(1.0 - eps);
                            let y = y as f64;
                            -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
                        })
                        .sum::<f64>()
                        / efv.len() as f64
                } else {
                    let per_point = |i: usize| {
                        let p = sigmoid(fv[i]).max(eps).min(1.0 - eps);
                        let y = target[i] as f64;
                        -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
                    };
                    match &self.sample_weight {
                        Some(w) => {
                            let wsum: f64 = w.iter().sum::<f64>().max(1e-12);
                            (0..ns).map(|i| w[i] * per_point(i)).sum::<f64>() / wsum
                        }
                        None => (0..ns).map(per_point).sum::<f64>() / ns as f64,
                    }
                };
                if let Some(best_n) = stopper.update(loss, round + 1) {
                    trees.truncate(best_n);
                    break;
                }
            }
        }
        Ok(Box::new(TrainedXGBoost {
            trees,
            initial: vec![initial],
            learning_rate: self.learning_rate,
            mode: XGBMode::BinaryClassif,
            feature_names: task.feature_names().to_vec(),
            feature_importances: imp,
            transform: PredTransform::Identity,
        }))
    }

    fn train_multiclass(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        let features = task.features();
        let target = task.target();
        let (ns, nf, nc) = (task.n_samples(), task.n_features(), task.n_classes());
        if let Some(w) = &self.sample_weight
            && w.len() != ns {
                return Err(SmeltError::DimensionMismatch {
                    expected: ns,
                    got: w.len(),
                });
            }
        let eval = validate_eval_classif(&self.eval_set, nf)?;
        self.validate_constraints(nf)?;
        let bins = HistBins::build_typed(features, self.n_bins, task.feature_types());
        // Weighted per-class frequency as the initial log-prior when sample
        // weights are set.
        let initial: Vec<f64> = match &self.sample_weight {
            Some(w) => {
                let mut wc = vec![0.0; nc];
                for (&t, &wi) in target.iter().zip(w) {
                    wc[t] += wi;
                }
                let wsum: f64 = w.iter().sum::<f64>().max(1e-15);
                wc.iter().map(|&c| (c / wsum).max(1e-15).ln()).collect()
            }
            None => {
                let mut cc = vec![0usize; nc];
                for &t in target {
                    cc[t] += 1;
                }
                cc.iter()
                    .map(|&c| ((c as f64 / ns as f64).max(1e-15)).ln())
                    .collect()
            }
        };
        let mut fv: Vec<Vec<f64>> = (0..ns).map(|_| initial.clone()).collect();
        let mut eval_fv: Option<Vec<Vec<f64>>> =
            eval.map(|(ef, _)| (0..ef.nrows()).map(|_| initial.clone()).collect());
        let mut trees = Vec::with_capacity(self.n_estimators * nc);
        let mut imp = vec![0.0; nf];
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut stopper = EarlyStopper::new(self.early_stopping_rounds);

        for round in 0..self.n_estimators {
            let probs: Vec<Vec<f64>> = fv.iter().map(|f| softmax(f)).collect();
            let (idx_base, cols) = self.sample(&mut rng, ns, nf);
            for c in 0..nc {
                let mut grads: Vec<f64> = (0..ns)
                    .map(|i| probs[i][c] - if target[i] == c { 1.0 } else { 0.0 })
                    .collect();
                let mut hess: Vec<f64> = (0..ns)
                    .map(|i| (probs[i][c] * (1.0 - probs[i][c])).max(1e-15))
                    .collect();
                if let Some(w) = &self.sample_weight {
                    for i in 0..ns {
                        grads[i] *= w[i];
                        hess[i] *= w[i];
                    }
                }
                let mut idx = idx_base.clone();
                let (tree, fi) =
                    self.build_one_tree(features, &bins, &grads, &hess, &mut idx, cols.clone(), nf);
                for i in 0..ns {
                    fv[i][c] += self.learning_rate * tree.predict_one(features.row(i));
                }
                if let (Some(efv), Some((ef, _))) = (&mut eval_fv, eval) {
                    for i in 0..ef.nrows() {
                        efv[i][c] += self.learning_rate * tree.predict_one(ef.row(i));
                    }
                }
                for (j, v) in fi.iter().enumerate() {
                    imp[j] += v;
                }
                trees.push(tree);
            }
            if stopper.is_active() {
                let eps = 1e-15;
                // Evaluate on the held-out set when one was provided —
                // training loss rarely plateaus under boosting.
                let loss: f64 = if let (Some(efv), Some((_, et))) = (&eval_fv, eval) {
                    let epn: Vec<Vec<f64>> = efv.iter().map(|f| softmax(f)).collect();
                    (0..et.len())
                        .map(|i| -epn[i][et[i]].max(eps).ln())
                        .sum::<f64>()
                        / et.len() as f64
                } else {
                    let pn: Vec<Vec<f64>> = fv.iter().map(|f| softmax(f)).collect();
                    let per_point = |i: usize| -pn[i][target[i]].max(eps).ln();
                    match &self.sample_weight {
                        Some(w) => {
                            let wsum: f64 = w.iter().sum::<f64>().max(1e-12);
                            (0..ns).map(|i| w[i] * per_point(i)).sum::<f64>() / wsum
                        }
                        None => (0..ns).map(per_point).sum::<f64>() / ns as f64,
                    }
                };
                if let Some(best_n) = stopper.update(loss, (round + 1) * nc) {
                    trees.truncate(best_n);
                    break;
                }
            }
        }
        Ok(Box::new(TrainedXGBoost {
            trees,
            initial,
            learning_rate: self.learning_rate,
            mode: XGBMode::MultiClassif { n_classes: nc },
            feature_names: task.feature_names().to_vec(),
            feature_importances: imp,
            transform: PredTransform::Identity,
        }))
    }
}

#[cfg(test)]
mod cat_tests {
    use super::*;
    use ndarray::Array2;

    /// y depends on the parity of a 7-code categorical feature: a single
    /// numeric threshold can never separate {0,2,4,6} from {1,3,5}, but one
    /// native categorical split can. With 3 depth-1 stumps the numeric model
    /// must stay far from the target while the categorical one nails it.
    fn parity_task(n: usize, categorical: bool) -> (RegressionTask, Array2<f64>, Vec<f64>) {
        let mut features = Array2::<f64>::zeros((n, 1));
        let mut target = vec![0.0; n];
        for i in 0..n {
            let code = (i % 7) as f64;
            features[[i, 0]] = code;
            target[i] = ((i % 7) % 2) as f64 * 10.0;
        }
        let task = RegressionTask::new("parity", features.clone(), target.clone()).unwrap();
        let task = if categorical {
            task.with_categorical_features(&[0]).unwrap()
        } else {
            task
        };
        (task, features, target)
    }

    fn rmse_of(model: &dyn TrainedModel, features: &Array2<f64>, target: &[f64]) -> f64 {
        let Prediction::Regression { predicted, .. } = model.predict(features).unwrap() else {
            unreachable!()
        };
        (predicted
            .iter()
            .zip(target)
            .map(|(p, y)| (p - y).powi(2))
            .sum::<f64>()
            / target.len() as f64)
            .sqrt()
    }

    fn stumps() -> XGBoost {
        XGBoost::new()
            .with_n_estimators(3)
            .with_max_depth(1)
            .with_learning_rate(1.0)
            .with_lambda(1e-6)
    }

    /// Histogram mode: n > n_bins.
    #[test]
    fn categorical_split_beats_numeric_threshold_histogram_mode() {
        let n = 700;
        let (cat_task, features, target) = parity_task(n, true);
        let (num_task, _, _) = parity_task(n, false);

        let m_cat = stumps().train_regress(&cat_task).unwrap();
        let m_num = stumps().train_regress(&num_task).unwrap();
        let (rc, rn) = (rmse_of(&*m_cat, &features, &target), rmse_of(&*m_num, &features, &target));
        assert!(rc < 1.0, "categorical split should fit parity exactly, got RMSE={rc}");
        assert!(
            rn > 2.0,
            "numeric thresholds cannot fit parity with 3 stumps, got RMSE={rn}"
        );
    }

    /// Exact mode: n <= n_bins routes through find_best_exact.
    #[test]
    fn categorical_split_works_in_exact_mode() {
        let n = 140;
        let (cat_task, features, target) = parity_task(n, true);
        let m_cat = stumps().train_regress(&cat_task).unwrap();
        let rc = rmse_of(&*m_cat, &features, &target);
        assert!(rc < 1.0, "categorical split should fit parity in exact mode, got RMSE={rc}");
    }

    /// Unseen categories at prediction time route right (with the
    /// not-explicitly-listed categories) and never panic.
    #[test]
    fn unseen_category_routes_safely() {
        let (cat_task, ..) = parity_task(700, true);
        let m = stumps().train_regress(&cat_task).unwrap();
        let unseen = Array2::from_shape_vec((2, 1), vec![99.0, f64::NAN]).unwrap();
        let Prediction::Regression { predicted, .. } = m.predict(&unseen).unwrap() else {
            unreachable!()
        };
        assert!(predicted.iter().all(|p| p.is_finite()));
        assert!(
            (-1.0..=11.0).contains(&predicted[0]),
            "unseen category prediction should stay in target range, got {}",
            predicted[0]
        );
    }

    /// Classification also uses the typed bins (multiclass exercises the
    /// shared tree builder through a different train path).
    #[test]
    fn categorical_split_multiclass() {
        let n = 600;
        let mut features = Array2::<f64>::zeros((n, 1));
        let mut target = vec![0usize; n];
        for i in 0..n {
            features[[i, 0]] = (i % 6) as f64;
            target[i] = (i % 6) % 3; // class = code mod 3, non-ordinal in code space
        }
        let task = ClassificationTask::new("cat_mc", features.clone(), target.clone())
            .unwrap()
            .with_categorical_features(&[0])
            .unwrap();
        let mut m = XGBoost::new()
            .with_n_estimators(10)
            .with_max_depth(2)
            .with_learning_rate(0.5);
        let model = m.train_classif(&task).unwrap();
        let Prediction::Classification { predicted, .. } = model.predict(&features).unwrap()
        else {
            unreachable!()
        };
        let acc = predicted
            .iter()
            .zip(&target)
            .filter(|(p, t)| p == t)
            .count() as f64
            / n as f64;
        assert!(acc > 0.95, "categorical multiclass should be near-perfect, got acc={acc}");
    }
}

#[cfg(test)]
mod objective_tests {
    use super::*;
    use ndarray::Array2;

    fn predict_vec(model: &dyn TrainedModel, features: &Array2<f64>) -> Vec<f64> {
        let Prediction::Regression { predicted, .. } = model.predict(features).unwrap() else {
            unreachable!()
        };
        predicted
    }

    /// Huber must be far less influenced by a few extreme target outliers
    /// than squared error: on the clean portion of the data its error stays
    /// much smaller.
    #[test]
    fn huber_is_robust_to_outliers() {
        let n = 300;
        let mut features = Array2::<f64>::zeros((n, 1));
        let mut target = vec![0.0; n];
        for i in 0..n {
            let x = i as f64 / 3.0;
            features[[i, 0]] = x;
            // Every 25th point is a wild outlier.
            target[i] = if i % 25 == 0 { 1_000.0 } else { x };
        }
        let task = RegressionTask::new("outliers", features.clone(), target.clone()).unwrap();

        // Enough rounds for Huber to converge despite its clipped (≤ delta)
        // per-round gradient magnitude.
        let config = || {
            XGBoost::new()
                .with_n_estimators(300)
                .with_max_depth(3)
                .with_learning_rate(0.3)
        };
        let m_sq = config().train_regress(&task).unwrap();
        let m_hu = config()
            .with_objective(Objective::Huber { delta: 5.0 })
            .train_regress(&task)
            .unwrap();

        let clean_rmse = |preds: &[f64]| {
            let pairs: Vec<(f64, f64)> = preds
                .iter()
                .zip(&target)
                .enumerate()
                .filter(|(i, _)| i % 25 != 0)
                .map(|(_, (&p, &y))| (p, y))
                .collect();
            (pairs.iter().map(|(p, y)| (p - y).powi(2)).sum::<f64>() / pairs.len() as f64).sqrt()
        };
        let rmse_sq = clean_rmse(&predict_vec(&*m_sq, &features));
        let rmse_hu = clean_rmse(&predict_vec(&*m_hu, &features));
        assert!(
            rmse_hu < rmse_sq * 0.5,
            "Huber should resist outliers: huber clean RMSE={rmse_hu:.2}, \
             squared-error clean RMSE={rmse_sq:.2}"
        );
    }

    /// 4th-audit LOW: early stopping under `Objective::Huber` used to
    /// monitor plain MSE, which is dominated by exactly the large-residual
    /// outliers Huber is designed to be insensitive to. The monitor must be
    /// the Huber loss itself: quadratic inside delta, linear beyond it.
    #[test]
    fn huber_monitor_loss_is_huber_not_mse() {
        let obj = Objective::Huber { delta: 1.0 };
        // Inside delta: 0.5 * r^2.
        assert!((obj.monitor_loss(0.5, 0.0) - 0.125).abs() < 1e-12);
        // Beyond delta: delta * (r - delta/2), NOT r^2.
        assert!((obj.monitor_loss(3.0, 0.0) - 2.5).abs() < 1e-12);
        // Linear growth in the tail: one extra unit of residual adds
        // exactly delta, where MSE would add r-dependent (2r+1) amounts.
        let step = obj.monitor_loss(10.0, 0.0) - obj.monitor_loss(9.0, 0.0);
        assert!((step - 1.0).abs() < 1e-12, "tail must grow linearly, got step {step}");
    }

    /// Poisson objective fits count data on the log scale and returns
    /// exp-transformed (strictly positive) predictions.
    #[test]
    fn poisson_predictions_are_positive_and_fit_counts() {
        let n = 400;
        let mut features = Array2::<f64>::zeros((n, 1));
        let mut target = vec![0.0; n];
        for i in 0..n {
            let x = i as f64 / n as f64 * 10.0;
            features[[i, 0]] = x;
            target[i] = (0.5 * x).exp().round(); // deterministic "counts"
        }
        let task = RegressionTask::new("counts", features.clone(), target.clone()).unwrap();
        let mut m = XGBoost::new()
            .with_n_estimators(100)
            .with_max_depth(3)
            .with_learning_rate(0.3)
            .with_objective(Objective::Poisson);
        let preds = predict_vec(&*m.train_regress(&task).unwrap(), &features);

        assert!(preds.iter().all(|&p| p > 0.0), "Poisson predictions must be positive");
        // Relative error on the larger counts should be small.
        let rel_err: f64 = preds
            .iter()
            .zip(&target)
            .filter(|(_, y)| **y >= 5.0)
            .map(|(&p, &y)| ((p - y) / y).abs())
            .sum::<f64>()
            / target.iter().filter(|&&y| y >= 5.0).count() as f64;
        assert!(rel_err < 0.2, "mean relative error should be small, got {rel_err:.3}");
    }

    /// A custom objective encoding squared error must reproduce the built-in
    /// squared-error model (same gradients → same trees → same predictions).
    #[test]
    fn custom_objective_matches_equivalent_builtin() {
        let n = 300;
        let mut features = Array2::<f64>::zeros((n, 1));
        let mut target = vec![0.0; n];
        for i in 0..n {
            features[[i, 0]] = i as f64;
            target[i] = (i as f64 * 0.7).sin() * 10.0 + i as f64 * 0.1;
        }
        let task = RegressionTask::new("custom", features.clone(), target).unwrap();

        let config = || {
            XGBoost::new()
                .with_n_estimators(30)
                .with_max_depth(3)
                .with_learning_rate(0.3)
        };
        let m_builtin = config().train_regress(&task).unwrap();
        let m_custom = config()
            .with_objective(Objective::Custom(std::sync::Arc::new(|p, y| (p - y, 1.0))))
            .train_regress(&task)
            .unwrap();

        let (a, b) = (
            predict_vec(&*m_builtin, &features),
            predict_vec(&*m_custom, &features),
        );
        let max_diff = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_diff < 1e-9,
            "custom squared-error must match the builtin exactly, max diff = {max_diff}"
        );
    }
}

#[cfg(test)]
mod monotone_tests {
    use super::*;
    use ndarray::Array2;

    /// y grows with x except for a strong dip in the middle. Unconstrained
    /// boosting fits the dip (predictions decrease somewhere); a +1 monotone
    /// constraint must yield non-decreasing predictions over the whole range.
    fn dip_task(n: usize) -> (RegressionTask, Array2<f64>) {
        let mut features = Array2::<f64>::zeros((n, 1));
        let mut target = vec![0.0; n];
        for i in 0..n {
            let x = i as f64 / n as f64 * 100.0;
            features[[i, 0]] = x;
            target[i] = if (40.0..60.0).contains(&x) { x - 30.0 } else { x };
        }
        let task = RegressionTask::new("dip", features.clone(), target).unwrap();
        (task, features)
    }

    fn predictions_sorted_by_x(model: &dyn TrainedModel, features: &Array2<f64>) -> Vec<f64> {
        let Prediction::Regression { predicted, .. } = model.predict(features).unwrap() else {
            unreachable!()
        };
        predicted // features are already sorted by x in dip_task
    }

    fn max_decrease(preds: &[f64]) -> f64 {
        preds
            .windows(2)
            .map(|w| w[0] - w[1])
            .fold(0.0f64, f64::max)
    }

    #[test]
    fn increasing_constraint_enforces_monotone_predictions_histogram_mode() {
        let (task, features) = dip_task(700); // > n_bins → histogram path
        let config = || {
            XGBoost::new()
                .with_n_estimators(50)
                .with_max_depth(4)
                .with_learning_rate(0.3)
        };

        let m_free = config().train_regress(&task).unwrap();
        let free = predictions_sorted_by_x(&*m_free, &features);
        assert!(
            max_decrease(&free) > 1.0,
            "unconstrained model should fit the dip (otherwise this test proves nothing), \
             max decrease = {}",
            max_decrease(&free)
        );

        let m_mono = config()
            .with_monotone_constraints(vec![1])
            .train_regress(&task)
            .unwrap();
        let mono = predictions_sorted_by_x(&*m_mono, &features);
        assert!(
            max_decrease(&mono) < 1e-9,
            "+1-constrained predictions must be non-decreasing in x, max decrease = {}",
            max_decrease(&mono)
        );
    }

    #[test]
    fn increasing_constraint_enforces_monotone_predictions_exact_mode() {
        let (task, features) = dip_task(150); // <= n_bins → exact path
        let mut m = XGBoost::new()
            .with_n_estimators(50)
            .with_max_depth(4)
            .with_learning_rate(0.3)
            .with_monotone_constraints(vec![1]);
        let mono = predictions_sorted_by_x(&*m.train_regress(&task).unwrap(), &features);
        assert!(
            max_decrease(&mono) < 1e-9,
            "+1-constrained predictions must be non-decreasing in exact mode, max decrease = {}",
            max_decrease(&mono)
        );
    }

    #[test]
    fn decreasing_constraint_enforces_non_increasing_predictions() {
        // Mirror of the dip task: y falls with x except a bump.
        let n = 700;
        let mut features = Array2::<f64>::zeros((n, 1));
        let mut target = vec![0.0; n];
        for i in 0..n {
            let x = i as f64 / n as f64 * 100.0;
            features[[i, 0]] = x;
            target[i] = if (40.0..60.0).contains(&x) { 130.0 - x } else { 100.0 - x };
        }
        let task = RegressionTask::new("bump", features.clone(), target).unwrap();
        let mut m = XGBoost::new()
            .with_n_estimators(50)
            .with_max_depth(4)
            .with_learning_rate(0.3)
            .with_monotone_constraints(vec![-1]);
        let preds = predictions_sorted_by_x(&*m.train_regress(&task).unwrap(), &features);
        let max_increase = preds.windows(2).map(|w| w[1] - w[0]).fold(0.0f64, f64::max);
        assert!(
            max_increase < 1e-9,
            "-1-constrained predictions must be non-increasing, max increase = {max_increase}"
        );
    }

    #[test]
    fn invalid_constraints_are_rejected() {
        let (task, _) = dip_task(50);
        // Wrong length.
        let mut wrong_len = XGBoost::new().with_monotone_constraints(vec![1, 0]);
        assert!(wrong_len.train_regress(&task).is_err());
        // Out-of-range value.
        let mut bad_val = XGBoost::new().with_monotone_constraints(vec![2]);
        assert!(bad_val.train_regress(&task).is_err());
    }
}

#[cfg(test)]
mod weight_tests {
    use super::*;
    use ndarray::Array2;

    /// With a constant feature (no split possible) the model collapses to the
    /// (weighted) mean of the targets. Heavily weighting the y=10 group must pull
    /// the prediction far above the unweighted midpoint of 5.
    #[test]
    fn sample_weights_shift_prediction_toward_heavy_group() {
        let n = 20;
        let features = Array2::<f64>::zeros((n, 1));
        let target: Vec<f64> = (0..n).map(|i| if i < n / 2 { 0.0 } else { 10.0 }).collect();

        let task = RegressionTask::new("w", features.clone(), target.clone()).unwrap();
        let mut unweighted = XGBoost::new().with_n_estimators(50).with_learning_rate(0.3);
        let m0 = unweighted.train_regress(&task).unwrap();
        let p0 = match m0.predict(&features).unwrap() {
            Prediction::Regression { predicted, .. } => predicted[0],
            _ => unreachable!(),
        };
        assert!((4.0..=6.0).contains(&p0), "unweighted should be ~5, got {p0}");

        // Weight the y=10 group 9x heavier -> weighted mean = 9.
        let w: Vec<f64> = (0..n).map(|i| if i < n / 2 { 1.0 } else { 9.0 }).collect();
        let mut weighted = XGBoost::new()
            .with_n_estimators(50)
            .with_learning_rate(0.3)
            .with_sample_weights(w);
        let m1 = weighted.train_regress(&task).unwrap();
        let p1 = match m1.predict(&features).unwrap() {
            Prediction::Regression { predicted, .. } => predicted[0],
            _ => unreachable!(),
        };
        assert!(p1 > 7.0, "weighted prediction should be pulled toward 9, got {p1}");
    }

    #[test]
    fn sample_weights_length_mismatch_errors() {
        let features = Array2::<f64>::zeros((10, 1));
        let task = RegressionTask::new("w", features, vec![0.0; 10]).unwrap();
        let mut m = XGBoost::new().with_sample_weights(vec![1.0; 5]);
        assert!(m.train_regress(&task).is_err());
    }

    /// Regression test for the histogram binning boundary bug (src/learner/histogram.rs):
    /// a binary feature that perfectly determines the target used to be unsplittable
    /// once n > n_bins forced histogram mode (the two values collapsed into one bin),
    /// so the model collapsed to predicting the global mean.
    #[test]
    fn binary_feature_is_splittable_in_histogram_mode() {
        let n = 600; // > default n_bins (256) forces histogram (non-exact) mode
        let mut features = Array2::<f64>::zeros((n, 1));
        let mut target = vec![0.0; n];
        for i in 0..n {
            let bit = (i % 2) as f64;
            features[[i, 0]] = bit;
            target[i] = bit * 10.0;
        }
        let task = RegressionTask::new("binary", features.clone(), target.clone()).unwrap();
        let mut model = XGBoost::new().with_n_estimators(20).with_learning_rate(0.5);
        let trained = model.train_regress(&task).unwrap();
        let pred = match trained.predict(&features).unwrap() {
            Prediction::Regression { predicted, .. } => predicted,
            _ => unreachable!(),
        };
        let rmse = (pred
            .iter()
            .zip(&target)
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / n as f64)
            .sqrt();
        assert!(rmse < 1.0, "binary feature should be perfectly splittable, got RMSE={rmse}");
    }

    /// Regression test for the stale-histogram bug in `build_hist_sub`
    /// (histogram-subtraction trick): when the smaller child of a split hits
    /// the early-leaf return (n<=1 or h_sum < min_child_weight) it never
    /// scans/stores its histogram, so `subtract_in_place` used to compute the
    /// larger sibling's histogram from stale pool data left over from an
    /// unrelated level, collapsing whole subtrees to a single leaf.
    ///
    /// n=600 forces histogram mode; a single extreme-outlier row is isolated
    /// by the root split as a 1-sample child, leaving the other 599 rows (a
    /// clean step function) to be built from the (previously) corrupted
    /// sibling histogram.
    #[test]
    fn outlier_isolated_as_singleton_child_does_not_corrupt_sibling_histogram() {
        let n = 600;
        let mut feats = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);
        for i in 0..n - 1 {
            let x = i as f64;
            feats.push(x);
            target.push(if x >= 300.0 { 10.0 } else { 0.0 });
        }
        feats.push(1e9);
        target.push(1e6);

        let features = Array2::from_shape_vec((n, 1), feats.clone()).unwrap();
        let task = RegressionTask::new("stale", features.clone(), target.clone()).unwrap();
        let mut model = XGBoost::new()
            .with_n_estimators(1)
            .with_max_depth(3)
            .with_learning_rate(1.0)
            .with_lambda(1e-9);
        let trained = model.train_regress(&task).unwrap();
        let pred = match trained.predict(&features).unwrap() {
            Prediction::Regression { predicted, .. } => predicted,
            _ => unreachable!(),
        };

        // RMSE on the 599 clean rows only — the outlier's own leaf may be
        // arbitrarily off (that's expected: it's a singleton leaf).
        let sse: f64 = pred[..n - 1]
            .iter()
            .zip(&target[..n - 1])
            .map(|(p, t)| (p - t).powi(2))
            .sum();
        let rmse = (sse / (n - 1) as f64).sqrt();
        assert!(
            rmse < 1.0,
            "clean rows should still fit the step despite the outlier sibling, got RMSE={rmse}"
        );

        let mut distinct: Vec<f64> = pred[..n - 1].to_vec();
        distinct.sort_by(|a, b| a.partial_cmp(b).unwrap());
        distinct.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
        assert!(
            distinct.len() > 1,
            "clean rows collapsed to a single predicted value (stale-histogram bug): {distinct:?}"
        );
    }

    /// Regression test: `with_sample_weights` was documented as
    /// "regression"-only and silently ignored in train_classif/train_binary/
    /// train_multiclass -- a user who set weights and trained a classifier
    /// got an unweighted model with no error or warning.
    #[test]
    fn sample_weights_shift_binary_classification_toward_heavy_group() {
        let n = 20;
        let features = Array2::<f64>::zeros((n, 1)); // constant feature: no split possible
        let target: Vec<usize> = (0..n).map(|i| if i < n / 2 { 0 } else { 1 }).collect();
        let task = ClassificationTask::new("wc", features.clone(), target).unwrap();

        let mut unweighted = XGBoost::new().with_n_estimators(30).with_learning_rate(0.3);
        let m0 = unweighted.train_classif(&task).unwrap();
        let p0 = match m0.predict(&features).unwrap() {
            Prediction::Classification { probabilities: Some(p), .. } => p[0][1],
            _ => unreachable!(),
        };
        assert!((0.3..=0.7).contains(&p0), "unweighted P(class=1) should be ~0.5, got {p0}");

        // Weight the class=1 group 9x heavier -> weighted positive fraction = 0.9.
        let w: Vec<f64> = (0..n).map(|i| if i < n / 2 { 1.0 } else { 9.0 }).collect();
        let mut weighted = XGBoost::new()
            .with_n_estimators(30)
            .with_learning_rate(0.3)
            .with_sample_weights(w);
        let m1 = weighted.train_classif(&task).unwrap();
        let p1 = match m1.predict(&features).unwrap() {
            Prediction::Classification { probabilities: Some(p), .. } => p[0][1],
            _ => unreachable!(),
        };
        assert!(p1 > 0.75, "weighted P(class=1) should be pulled toward 0.9, got {p1}");
    }

    #[test]
    fn sample_weights_shift_multiclass_toward_heavy_group() {
        let n = 30;
        let features = Array2::<f64>::zeros((n, 1)); // constant feature: no split possible
        let target: Vec<usize> = (0..n).map(|i| i % 3).collect(); // classes 0,1,2 evenly
        let task = ClassificationTask::new("wmc", features.clone(), target.clone()).unwrap();

        let mut unweighted = XGBoost::new().with_n_estimators(30).with_learning_rate(0.3);
        let m0 = unweighted.train_classif(&task).unwrap();
        let p0 = match m0.predict(&features).unwrap() {
            Prediction::Classification { probabilities: Some(p), .. } => p[0][2],
            _ => unreachable!(),
        };
        assert!((0.15..=0.5).contains(&p0), "unweighted P(class=2) should be ~1/3, got {p0}");

        // Weight class=2 examples 9x heavier -> should dominate the posterior.
        let w: Vec<f64> = target.iter().map(|&t| if t == 2 { 9.0 } else { 1.0 }).collect();
        let mut weighted = XGBoost::new()
            .with_n_estimators(30)
            .with_learning_rate(0.3)
            .with_sample_weights(w);
        let m1 = weighted.train_classif(&task).unwrap();
        let p1 = match m1.predict(&features).unwrap() {
            Prediction::Classification { probabilities: Some(p), .. } => p[0][2],
            _ => unreachable!(),
        };
        assert!(p1 > 0.6, "weighted P(class=2) should dominate, got {p1}");
    }

    #[test]
    fn sample_weights_classif_length_mismatch_errors() {
        let features = Array2::<f64>::zeros((10, 1));
        let target = vec![0usize, 1, 0, 1, 0, 1, 0, 1, 0, 1];
        let task = ClassificationTask::new("wc_mismatch", features, target).unwrap();
        let mut m = XGBoost::new().with_sample_weights(vec![1.0; 5]);
        assert!(m.train_classif(&task).is_err());
    }

    /// Regression test: early stopping used to monitor training loss, which
    /// is (near-)monotonically decreasing under boosting and so almost never
    /// plateaus -- with a deliberately overfittable configuration (many
    /// estimators, high depth, tiny lambda, small noisy dataset), the
    /// train-loss-driven variant should overfit and generalize poorly, while
    /// the eval-set-driven variant should stop once held-out loss stops
    /// improving and generalize markedly better.
    #[test]
    fn eval_set_early_stopping_generalizes_better_than_train_loss_early_stopping() {
        use rand::Rng;

        let n = 40;
        let make = |seed: u64| {
            let mut r = StdRng::seed_from_u64(seed);
            let mut feats = Vec::with_capacity(n);
            let mut target = Vec::with_capacity(n);
            for i in 0..n {
                let x = i as f64;
                feats.push(x);
                target.push(x * 0.1 + r.random::<f64>() * 8.0); // weak signal, heavy noise
            }
            (Array2::from_shape_vec((n, 1), feats).unwrap(), target)
        };
        let (tr_feat, tr_tgt) = make(100);
        let (va_feat, va_tgt) = make(200);

        let base = || {
            XGBoost::new()
                .with_n_estimators(500)
                .with_max_depth(10)
                .with_learning_rate(0.5)
                .with_lambda(0.0001)
                .with_early_stopping_rounds(3)
        };
        let task = RegressionTask::new("es", tr_feat.clone(), tr_tgt.clone()).unwrap();

        let rmse = |model: &dyn TrainedModel, feat: &Array2<f64>, tgt: &[f64]| {
            let Prediction::Regression { predicted, .. } = model.predict(feat).unwrap() else {
                unreachable!()
            };
            (predicted.iter().zip(tgt).map(|(p, y)| (p - y).powi(2)).sum::<f64>()
                / tgt.len() as f64)
                .sqrt()
        };

        let mut train_loss_stopped = base();
        let m_train = train_loss_stopped.train_regress(&task).unwrap();
        let train_loss_val_rmse = rmse(&*m_train, &va_feat, &va_tgt);

        let mut eval_set_stopped = base().with_eval_set_regress(va_feat.clone(), va_tgt.clone());
        let m_eval = eval_set_stopped.train_regress(&task).unwrap();
        let eval_set_val_rmse = rmse(&*m_eval, &va_feat, &va_tgt);

        assert!(
            eval_set_val_rmse < train_loss_val_rmse * 0.9,
            "eval-set early stopping (val RMSE={eval_set_val_rmse:.3}) should generalize \
             markedly better than train-loss early stopping (val RMSE={train_loss_val_rmse:.3}) \
             on this deliberately overfittable configuration"
        );
    }

    #[test]
    fn eval_set_regress_rejects_dimension_mismatch() {
        let features = Array2::<f64>::zeros((10, 2));
        let target = vec![0.0; 10];
        let task = RegressionTask::new("es_dim", features, target).unwrap();

        // Wrong number of eval features (1 col instead of 2).
        let mut wrong_cols = XGBoost::new()
            .with_early_stopping_rounds(2)
            .with_eval_set_regress(Array2::<f64>::zeros((5, 1)), vec![0.0; 5]);
        assert!(wrong_cols.train_regress(&task).is_err());

        // Wrong eval target length.
        let mut wrong_len = XGBoost::new()
            .with_early_stopping_rounds(2)
            .with_eval_set_regress(Array2::<f64>::zeros((5, 2)), vec![0.0; 3]);
        assert!(wrong_len.train_regress(&task).is_err());
    }

    #[test]
    fn eval_set_task_type_mismatch_is_rejected() {
        let features = Array2::<f64>::zeros((10, 1));
        let target = vec![0.0; 10];
        let task = RegressionTask::new("es_type", features, target).unwrap();

        // A classification eval_set on a regression task must error, not be
        // silently ignored.
        let mut mismatched = XGBoost::new()
            .with_early_stopping_rounds(2)
            .with_eval_set_classif(Array2::<f64>::zeros((5, 1)), vec![0usize; 5]);
        assert!(mismatched.train_regress(&task).is_err());
    }
}
