//! LightGBM-inspired leaf-wise gradient boosting with GOSS.
//!
//! This implements two key ideas from the LightGBM paper:
//! - **Gradient-based One-Side Sampling (GOSS)**: keeps top gradients, samples rest
//! - **Leaf-wise (best-first) tree growth**: always splits the leaf with highest gain
//! - Histogram-based splits with NaN handling
//!
//! **Not implemented**: Exclusive Feature Bundling (EFB), GPU training,
//! distributed computation. This implementation does not match the official
//! library's performance; it is included for API completeness and the
//! leaf-wise growth strategy.
//!
//! Reference: Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting
//! Decision Tree. NeurIPS.

use crate::Result;
use crate::learner::math::{sigmoid, softmax};
use crate::learner::{Learner, LearnerProperties, TrainedModel};
use crate::prediction::Prediction;
use crate::task::{ClassificationTask, RegressionTask, Task};
use ndarray::{Array2, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::eval::{EarlyStopper, EvalSet, EvalTarget, validate_eval_classif, validate_eval_regress};
use super::histogram::{HistBins, NAN_BIN, accumulate_histogram, best_categorical_split, best_numeric_split};

/// LightGBM-inspired leaf-wise GBM with GOSS sampling.
///
/// Implements leaf-wise growth and gradient-based one-side sampling from
/// Ke et al. (2017). Does not include Exclusive Feature Bundling (EFB)
/// or weighted GOSS histograms.
///
/// # Sample weights
///
/// Per-sample weights attached via `Task::with_weights` scale each sample's
/// gradient and hessian *before* any accumulation (histograms, split gains,
/// leaf values, the initial score, and the train-loss early-stopping
/// monitor). GOSS therefore operates on the already-weighted gradients —
/// its top-|gradient| selection sees `|w_i * g_i|` — matching the official
/// implementation, and its own amplification weights multiply *on top of*
/// the sample weights in the histogram accumulation. The same compounding
/// applies to `with_subsample` row bagging (itself a documented divergence
/// from the official library, which forbids bagging+GOSS): bagging selects
/// rows uniformly regardless of weight; dropped rows simply contribute
/// nothing that tree. The eval-set early-stopping metric is NOT weighted —
/// weights belong to training rows; an eval row has no weight.
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
/// let task = ClassificationTask::new("lgbm", features, target).unwrap();
///
/// let mut lgbm = LightGBM::new()
///     .with_n_estimators(50)
///     .with_num_leaves(31)
///     .with_learning_rate(0.1);
/// let model = lgbm.train_classif(&task).unwrap();
/// ```
pub struct LightGBM {
    n_estimators: usize,
    learning_rate: f64,
    num_leaves: usize,        // max leaves per tree (leaf-wise control)
    max_depth: Option<usize>, // optional depth limit
    lambda: f64,
    min_child_weight: f64,
    /// GOSS: fraction of top gradients to keep
    top_rate: f64,
    /// GOSS: fraction of remaining to sample
    other_rate: f64,
    n_bins: usize,
    subsample: f64,
    colsample_bytree: f64,
    seed: u64,
    early_stopping_rounds: usize,
    /// Optional held-out set for early stopping — see `EvalSet` docs; training
    /// loss under boosting rarely plateaus, so the eval set is what makes
    /// `early_stopping_rounds` actually fire.
    eval_set: EvalSet,
}

impl Default for LightGBM {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.1,
            num_leaves: 31,
            max_depth: None,
            lambda: 0.0,
            min_child_weight: 1.0,
            // GOSS is OPT-IN, matching the official implementation (whose
            // default is plain GBDT; GOSS requires boosting=goss). top_rate
            // 1.0 / other_rate 0.0 degenerate goss_sample to "keep every row
            // at weight 1". Earlier releases (<3.0) defaulted to the paper's
            // 0.2/0.1, silently training every tree on ~30% of the rows.
            top_rate: 1.0,
            other_rate: 0.0,
            n_bins: 255,
            subsample: 1.0,
            colsample_bytree: 1.0,
            seed: 42,
            early_stopping_rounds: 0,
            eval_set: None,
        }
    }
}

impl LightGBM {
    /// Creates a `LightGBM` learner with default hyperparameters.
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
    /// Sets the maximum number of leaves per tree, the primary control on
    /// leaf-wise (best-first) growth.
    pub fn with_num_leaves(mut self, n: usize) -> Self {
        self.num_leaves = n;
        self
    }
    /// Sets an optional maximum tree depth, in addition to `num_leaves`.
    pub fn with_max_depth(mut self, d: usize) -> Self {
        self.max_depth = Some(d);
        self
    }
    /// Sets the L2 regularization strength on leaf weights.
    pub fn with_lambda(mut self, l: f64) -> Self {
        self.lambda = l;
        self
    }
    /// Sets the minimum sum of Hessian (instance weight) required in a leaf
    /// for a split to be considered.
    pub fn with_min_child_weight(mut self, w: f64) -> Self {
        self.min_child_weight = w;
        self
    }
    /// Sets the GOSS fraction of top-gradient samples always kept.
    /// Setting this below 1.0 enables GOSS; the paper's values are
    /// `top_rate=0.2, other_rate=0.1` (Ke et al. 2017).
    pub fn with_top_rate(mut self, r: f64) -> Self {
        self.top_rate = r;
        self
    }
    /// Sets the GOSS fraction of the remaining (small-gradient) samples
    /// randomly sampled.
    pub fn with_other_rate(mut self, r: f64) -> Self {
        self.other_rate = r;
        self
    }
    /// Sets the fraction of rows randomly subsampled for each tree, applied
    /// *before* GOSS sampling: GOSS's own top/other-rate selection then runs
    /// on this row population instead of the full training set. NOTE: this
    /// composition is a deliberate divergence from the official
    /// implementation, which makes bagging and GOSS mutually exclusive
    /// ("Cannot use bagging in GOSS"); here they compound. With the default
    /// `1.0` this is a no-op.
    pub fn with_subsample(mut self, s: f64) -> Self {
        self.subsample = s;
        self
    }
    /// Sets the fraction of columns randomly sampled for each tree.
    pub fn with_colsample_bytree(mut self, c: f64) -> Self {
        self.colsample_bytree = c;
        self
    }
    /// Sets the RNG seed controlling GOSS sampling and column/row subsampling.
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }
    /// Stop after `n` rounds without improvement of the monitored loss
    /// (held-out loss when an eval set is provided, training loss otherwise).
    pub fn with_early_stopping_rounds(mut self, n: usize) -> Self {
        self.early_stopping_rounds = n;
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
    /// classification (binary or multiclass).
    pub fn with_eval_set_classif(mut self, features: Array2<f64>, target: Vec<usize>) -> Self {
        self.eval_set = Some((features, EvalTarget::Classification(target)));
        self
    }
}

type Bins = HistBins;

// ── Tree node ───────────────────────────────────────────────────────

/// Internal tree node: a leaf with a fitted value, or a split on a feature
/// (numeric threshold or categorical membership).
#[derive(Clone, Serialize, Deserialize)]
pub enum LGBNode {
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
        nan_left: bool,
        /// Left child, taken when the row's value is below `threshold`.
        left: Box<LGBNode>,
        /// Right child, taken when the row's value is at or above `threshold`.
        right: Box<LGBNode>,
    },
    /// Categorical split: the listed category codes go left; every other
    /// code — including categories unseen during training — goes right.
    CatSplit {
        /// Index of the feature being split on.
        feature: usize,
        /// Sorted category codes routed left.
        left_cats: Vec<u16>,
        /// Whether NaN values in `feature` route to the left child.
        nan_left: bool,
        /// Left child, taken when the row's category code is in `left_cats`.
        left: Box<LGBNode>,
        /// Right child, taken for all other category codes.
        right: Box<LGBNode>,
    },
}

impl LGBNode {
    #[inline]
    fn predict_one(&self, row: ArrayView1<f64>) -> f64 {
        match self {
            LGBNode::Leaf { weight } => *weight,
            LGBNode::Split {
                feature,
                threshold,
                nan_left,
                left,
                right,
            } => {
                let v = row[*feature];
                if v.is_nan() {
                    if *nan_left {
                        left.predict_one(row)
                    } else {
                        right.predict_one(row)
                    }
                } else if v < *threshold {
                    left.predict_one(row)
                } else {
                    right.predict_one(row)
                }
            }
            LGBNode::CatSplit {
                feature,
                left_cats,
                nan_left,
                left,
                right,
            } => {
                let v = row[*feature];
                let goes_left = if v.is_nan() {
                    *nan_left
                } else {
                    left_cats.binary_search(&(v as u16)).is_ok()
                };
                if goes_left {
                    left.predict_one(row)
                } else {
                    right.predict_one(row)
                }
            }
        }
    }
}

// ── GOSS sampling ───────────────────────────────────────────────────

/// GOSS sampling restricted to `candidates` (a subset of `0..grads.len()`,
/// typically the output of [`LightGBM::sample_rows`] row bagging -- the
/// full `0..grads.len()` range when `subsample=1.0`). `top_rate`/
/// `other_rate` apply to `candidates.len()`, not the full training set, so
/// row bagging and GOSS compound here -- unlike the official
/// implementation, which forbids the combination ("Cannot use bagging in
/// GOSS"); a documented, deliberate divergence.
fn goss_sample(
    grads: &[f64],
    _hess: &[f64],
    candidates: &[usize],
    top_rate: f64,
    other_rate: f64,
    rng: &mut StdRng,
) -> (Vec<usize>, Vec<f64>) {
    // Degenerate "keep everything" configuration -- notably the plain-GBDT
    // DEFAULT (top_rate=1.0, other_rate=0.0): every candidate is a "top"
    // sample with weight 1.0 and nothing is subsampled, so the
    // sort-by-|gradient| below is a pure O(n log n) waste per tree (5th
    // audit, LOW-A). Skip it. RNG state is untouched either way (the
    // general path's shuffle of the empty `rest` draws nothing), so
    // seeded runs are unaffected.
    if top_rate >= 1.0 && other_rate <= 0.0 {
        let mut weights = vec![0.0; grads.len()];
        for &i in candidates {
            weights[i] = 1.0;
        }
        return (candidates.to_vec(), weights);
    }

    let n = candidates.len();
    let mut sorted: Vec<usize> = candidates.to_vec();
    sorted.sort_by(|&a, &b| {
        grads[b]
            .abs()
            .partial_cmp(&grads[a].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let top_n = (n as f64 * top_rate).ceil() as usize;
    let other_n = (n as f64 * other_rate).ceil() as usize;

    let top_indices = &sorted[..top_n.min(n)];
    let rest = &sorted[top_n.min(n)..];

    let mut sampled_rest: Vec<usize> = rest.to_vec();
    sampled_rest.shuffle(rng);
    sampled_rest.truncate(other_n);

    // Amplification factor for the randomly-sampled "small gradient" subset,
    // so its contribution to the sum of gradients/hessians matches, in
    // expectation, what the full "other" population would have contributed
    // (Ke et al. 2017, Algorithm 2) -- this is GOSS's central correction;
    // without it, the sampled sum is systematically biased low. Computed
    // from the actual counts (|rest| / |sampled|, the exact finite-sample
    // correction) rather than the paper's asymptotic (1-a)/b: the old form
    // clamped the denominator at 0.01, silently under-amplifying any
    // opted-in other_rate below that, and drifted from the true factor by
    // ceil-rounding whenever n*rate wasn't integral.
    let amplify = if sampled_rest.is_empty() {
        1.0
    } else {
        rest.len() as f64 / sampled_rest.len() as f64
    };

    let mut selected: Vec<usize> = top_indices.to_vec();
    selected.extend_from_slice(&sampled_rest);

    // Scatter weights back to original sample indices (0..grads.len()):
    // build_recursive and find_best_split_hist look up `weights[i]` by
    // original sample id, not by position within `selected`/`candidates`.
    let mut weights = vec![0.0; grads.len()];
    for &i in top_indices {
        weights[i] = 1.0;
    }
    for &i in &sampled_rest {
        weights[i] = amplify;
    }

    (selected, weights)
}

// ── Leaf-wise tree building ─────────────────────────────────────────

struct LeafCandidate {
    gain: f64,
    feature: usize,
    threshold: f64,
    nan_left: bool,
    split_bin: usize,
    /// `Some(sorted category codes going left)` for categorical splits.
    left_cats: Option<Vec<u16>>,
}

impl LeafCandidate {
    /// Does bin `b` (never `NAN_BIN`) route left under this split?
    #[inline]
    fn bin_goes_left(&self, b: u8) -> bool {
        match &self.left_cats {
            Some(cats) => cats.binary_search(&(b as u16)).is_ok(),
            None => (b as usize) <= self.split_bin,
        }
    }
}

/// Per-feature cached histogram: (bin_g, bin_h, nan_g, nan_h)
type LeafHist = Vec<(Vec<f64>, Vec<f64>, f64, f64)>;

/// Build histogram for a set of indices (parallel over features).
fn build_leaf_hist(
    bins: &Bins,
    grads: &[f64],
    hess: &[f64],
    weights: &[f64],
    indices: &[usize],
    col_indices: &[usize],
) -> LeafHist {
    col_indices
        .par_iter()
        .map(|&feat| accumulate_histogram(bins, feat, grads, hess, Some(weights), indices))
        .collect()
}

/// Denominator floor for `split_gain`/`leaf_weight`: guards `lambda=0`
/// (the default) combined with `min_child_weight=0` -- or a genuinely
/// zero-hessian loss -- from dividing by exactly zero and propagating
/// NaN/Inf into the tree (audit issue M6). A no-op under the actual
/// defaults, where `min_child_weight=1.0` already keeps every accepted
/// split's `hl`/`hr` >= 1.0.
const GAIN_DENOM_EPS: f64 = 1e-12;

/// Second-order (Newton) split gain, shared by the numeric and categorical
/// split search below (previously duplicated inline at each call site).
#[inline]
fn split_gain(gl: f64, hl: f64, gr: f64, hr: f64, lambda: f64) -> f64 {
    0.5 * (gl * gl / (hl + lambda).max(GAIN_DENOM_EPS)
        + gr * gr / (hr + lambda).max(GAIN_DENOM_EPS)
        - (gl + gr) * (gl + gr) / (hl + hr + lambda).max(GAIN_DENOM_EPS))
}

/// Find best split from a cached histogram (no scanning).
fn find_best_from_cache(
    bins: &Bins,
    cached: &LeafHist,
    col_indices: &[usize],
    lambda: f64,
    min_child_weight: f64,
) -> Option<LeafCandidate> {
    let results: Vec<Option<LeafCandidate>> = col_indices
        .par_iter()
        .enumerate()
        .map(|(fi, &feat)| {
            let (bin_g, bin_h, nan_g, nan_h) = &cached[fi];

            if bins.cat[feat].is_some() {
                return best_categorical_split(
                    bin_g,
                    bin_h,
                    *nan_g,
                    *nan_h,
                    min_child_weight,
                    |gl, hl, gr, hr| split_gain(gl, hl, gr, hr, lambda),
                )
                .map(|(left_cats, gain, nan_left)| LeafCandidate {
                    gain,
                    feature: feat,
                    threshold: f64::NAN,
                    nan_left,
                    split_bin: 0,
                    left_cats: Some(left_cats),
                });
            }

            let best = best_numeric_split(
                bin_g,
                bin_h,
                *nan_g,
                *nan_h,
                min_child_weight,
                |gl, hl, gr, hr| Some(split_gain(gl, hl, gr, hr, lambda)),
            );
            best.map(|(bin, gain, nan_left)| LeafCandidate {
                gain,
                feature: feat,
                threshold: bins.boundaries[feat][bin],
                nan_left,
                split_bin: bin,
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

/// Subtract: result[fi] = parent[fi] - child[fi]
fn subtract_leaf_hists(parent: &LeafHist, child: &LeafHist) -> LeafHist {
    parent
        .iter()
        .zip(child)
        .map(|((pg, ph, png, pnh), (cg, ch, cng, cnh))| {
            let bg: Vec<f64> = pg.iter().zip(cg).map(|(p, c)| p - c).collect();
            let bh: Vec<f64> = ph.iter().zip(ch).map(|(p, c)| p - c).collect();
            (bg, bh, png - cng, pnh - cnh)
        })
        .collect()
}

/// A node in the tree being grown leaf-wise. `Leaf` nodes are candidates for
/// splitting (tracked in `active_leaves` below, indexed by position in the
/// arena); once split, a `Leaf` slot is replaced in-place by a `Split` slot
/// pointing at two freshly-appended `Leaf` children.
enum ArenaNode {
    Leaf {
        indices: Vec<usize>,
        depth: usize,
        hist: LeafHist,
    },
    Split {
        feature: usize,
        threshold: f64,
        nan_left: bool,
        left_cats: Option<Vec<u16>>,
        left: usize,
        right: usize,
    },
}

/// Leaf-wise (best-first) tree growth (Ke et al. 2017): at each step, split
/// whichever *leaf* (not whichever depth level) has the highest gain, using
/// cached per-leaf histograms with subtraction (scan the smaller child,
/// derive the larger one as parent-minus-smaller) rather than rescanning
/// from scratch. Grows until `num_leaves` leaves exist or no leaf has a
/// positive-gain split left.
fn build_leaf_wise_tree(
    bins: &Bins,
    grads: &[f64],
    hess: &[f64],
    weights: &[f64],
    indices: Vec<usize>,
    col_indices: &[usize],
    num_leaves: usize,
    max_depth: Option<usize>,
    lambda: f64,
    min_child_weight: f64,
    feature_importances: &mut [f64],
) -> LGBNode {
    let root_hist = build_leaf_hist(bins, grads, hess, weights, &indices, col_indices);
    let mut arena: Vec<ArenaNode> = vec![ArenaNode::Leaf { indices, depth: 0, hist: root_hist }];
    let mut active_leaves: Vec<usize> = vec![0];
    let mut leaf_count = 1usize;

    while leaf_count < num_leaves {
        let mut best: Option<(usize, LeafCandidate)> = None;

        for &li in &active_leaves {
            let ArenaNode::Leaf { indices: leaf_indices, depth, hist } = &arena[li] else {
                unreachable!("active_leaves only ever references Leaf arena slots")
            };
            if let Some(md) = max_depth
                && *depth >= md
            {
                continue;
            }
            if leaf_indices.len() < 2 {
                continue;
            }
            let h_sum: f64 = leaf_indices.iter().map(|&i| hess[i] * weights[i]).sum();
            if h_sum < min_child_weight {
                continue;
            }
            if let Some(cand) = find_best_from_cache(bins, hist, col_indices, lambda, min_child_weight)
            {
                let is_better = match &best {
                    None => true,
                    Some((_, prev)) => cand.gain > prev.gain,
                };
                if is_better {
                    best = Some((li, cand));
                }
            }
        }

        let Some((li, split)) = best else { break };

        let placeholder = ArenaNode::Split {
            feature: split.feature,
            threshold: split.threshold,
            nan_left: split.nan_left,
            left_cats: split.left_cats.clone(),
            left: usize::MAX,
            right: usize::MAX,
        };
        let (leaf_indices, depth, parent_hist) = match std::mem::replace(&mut arena[li], placeholder)
        {
            ArenaNode::Leaf { indices, depth, hist } => (indices, depth, hist),
            ArenaNode::Split { .. } => unreachable!(),
        };

        let mut left_idx = Vec::new();
        let mut right_idx = Vec::new();
        for &idx in &leaf_indices {
            let b = bins.get_bin(split.feature, idx);
            let goes_left = if b == NAN_BIN {
                split.nan_left
            } else {
                split.bin_goes_left(b)
            };
            if goes_left {
                left_idx.push(idx);
            } else {
                right_idx.push(idx);
            }
        }

        if left_idx.is_empty() || right_idx.is_empty() {
            // Degenerate split (shouldn't happen given find_best_from_cache
            // only proposes splits with points on both sides, but guard
            // anyway): revert the slot to a leaf and stop trying to split it.
            arena[li] = ArenaNode::Leaf { indices: leaf_indices, depth, hist: parent_hist };
            active_leaves.retain(|&x| x != li);
            continue;
        }

        // Credit importance only once the split is definitely applied --
        // crediting before the degenerate check above charged the feature
        // for a split that was then reverted.
        feature_importances[split.feature] += split.gain;

        // Histogram subtraction: scan the smaller child, derive the larger
        // one as parent-minus-smaller.
        let (smaller_idx, larger_idx, smaller_is_left) = if left_idx.len() <= right_idx.len() {
            (&left_idx, &right_idx, true)
        } else {
            (&right_idx, &left_idx, false)
        };
        let _ = larger_idx; // only used to decide which side got the subtraction
        let smaller_hist = build_leaf_hist(bins, grads, hess, weights, smaller_idx, col_indices);
        let larger_hist = subtract_leaf_hists(&parent_hist, &smaller_hist);
        let (left_hist, right_hist) = if smaller_is_left {
            (smaller_hist, larger_hist)
        } else {
            (larger_hist, smaller_hist)
        };

        let left_arena_idx = arena.len();
        arena.push(ArenaNode::Leaf { indices: left_idx, depth: depth + 1, hist: left_hist });
        let right_arena_idx = arena.len();
        arena.push(ArenaNode::Leaf { indices: right_idx, depth: depth + 1, hist: right_hist });
        if let ArenaNode::Split { left, right, .. } = &mut arena[li] {
            *left = left_arena_idx;
            *right = right_arena_idx;
        }

        active_leaves.retain(|&x| x != li);
        active_leaves.push(left_arena_idx);
        active_leaves.push(right_arena_idx);
        leaf_count += 1; // one leaf became a split; two leaves replace it: net +1
    }

    fn to_lgb_node(
        arena: &mut [Option<ArenaNode>],
        idx: usize,
        grads: &[f64],
        hess: &[f64],
        weights: &[f64],
        lambda: f64,
    ) -> LGBNode {
        match arena[idx].take().expect("each arena slot is converted exactly once") {
            ArenaNode::Leaf { indices, .. } => LGBNode::Leaf {
                weight: leaf_weight(grads, hess, weights, &indices, lambda),
            },
            ArenaNode::Split { feature, threshold, nan_left, left_cats, left, right } => {
                let l = Box::new(to_lgb_node(arena, left, grads, hess, weights, lambda));
                let r = Box::new(to_lgb_node(arena, right, grads, hess, weights, lambda));
                match left_cats {
                    Some(left_cats) => LGBNode::CatSplit {
                        feature,
                        left_cats,
                        nan_left,
                        left: l,
                        right: r,
                    },
                    None => LGBNode::Split {
                        feature,
                        threshold,
                        nan_left,
                        left: l,
                        right: r,
                    },
                }
            }
        }
    }
    let mut arena_opt: Vec<Option<ArenaNode>> = arena.into_iter().map(Some).collect();
    to_lgb_node(&mut arena_opt, 0, grads, hess, weights, lambda)
}

fn leaf_weight(
    grads: &[f64],
    hess: &[f64],
    weights: &[f64],
    indices: &[usize],
    lambda: f64,
) -> f64 {
    let g: f64 = indices.iter().map(|&i| grads[i] * weights[i]).sum();
    let h: f64 = indices.iter().map(|&i| hess[i] * weights[i]).sum();
    -g / (h + lambda).max(GAIN_DENOM_EPS)
}

// ── Trained model ───────────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub(crate) enum LGBMode {
    Regression,
    BinaryClassif,
    MultiClassif { n_classes: usize },
}

/// A trained LightGBM model, ready to predict.
#[derive(Clone, Serialize, Deserialize)]
pub struct TrainedLightGBM {
    pub(crate) trees: Vec<LGBNode>,
    pub(crate) initial: Vec<f64>,
    pub(crate) learning_rate: f64,
    pub(crate) mode: LGBMode,
    pub(crate) feature_names: Vec<String>,
    pub(crate) feature_importances: Vec<f64>,
}

impl TrainedModel for TrainedLightGBM {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        crate::validate::check_n_features(features, self.feature_names.len())?;
        match &self.mode {
            LGBMode::Regression => {
                let predicted: Vec<f64> = (0..features.nrows())
                    .into_par_iter()
                    .map(|i| {
                        let r = features.row(i);
                        let mut v = self.initial[0];
                        for t in &self.trees {
                            v += self.learning_rate * t.predict_one(r);
                        }
                        v
                    })
                    .collect();
                Ok(Prediction::regression(predicted))
            }
            LGBMode::BinaryClassif => {
                let results: Vec<(usize, Vec<f64>)> = (0..features.nrows())
                    .into_par_iter()
                    .map(|i| {
                        let r = features.row(i);
                        let mut f = self.initial[0];
                        for t in &self.trees {
                            f += self.learning_rate * t.predict_one(r);
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
            LGBMode::MultiClassif { n_classes } => {
                let k = *n_classes;
                let ni = self.trees.len() / k;
                let results: Vec<(usize, Vec<f64>)> = (0..features.nrows())
                    .into_par_iter()
                    .map(|i| {
                        let r = features.row(i);
                        let mut scores = self.initial.clone();
                        for iter in 0..ni {
                            for c in 0..k {
                                scores[c] +=
                                    self.learning_rate * self.trees[iter * k + c].predict_one(r);
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
        Some(crate::serialize::SerializableModel::LightGBM(self.clone()))
    }
}

// ── Learner ─────────────────────────────────────────────────────────

impl LightGBM {
    fn sample_cols(&self, rng: &mut StdRng, nf: usize) -> Vec<usize> {
        if self.colsample_bytree < 1.0 {
            let k = (nf as f64 * self.colsample_bytree).ceil().max(1.0) as usize;
            let mut v: Vec<usize> = (0..nf).collect();
            v.shuffle(rng);
            v.truncate(k);
            v.sort();
            v
        } else {
            (0..nf).collect()
        }
    }

    /// Row bagging: the candidate population `goss_sample` then does its own
    /// top/other-rate selection over. `subsample=1.0` (the default) returns
    /// all `ns` rows, so GOSS's behavior is unchanged from before this
    /// method existed.
    fn sample_rows(&self, rng: &mut StdRng, ns: usize) -> Vec<usize> {
        if self.subsample < 1.0 {
            let k = (ns as f64 * self.subsample).ceil().max(1.0) as usize;
            let mut v: Vec<usize> = (0..ns).collect();
            v.shuffle(rng);
            v.truncate(k);
            v.sort_unstable();
            v
        } else {
            (0..ns).collect()
        }
    }
}

impl Learner for LightGBM {
    fn id(&self) -> &str {
        "lightgbm"
    }

    fn properties(&self) -> LearnerProperties {
        LearnerProperties::classifier_regressor()
            .with_weights()
            .with_proba()
            .with_nan()
            .with_categorical()
            .with_feature_importance()
            .with_serializable()
    }

    fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
        let features = task.features();
        let target = task.target();
        let (ns, nf) = (task.n_samples(), task.n_features());
        let sw = task.weights();
        let eval = validate_eval_regress(&self.eval_set, nf)?;
        let bins = HistBins::build_typed(features, self.n_bins, task.feature_types());
        // Weighted mean as the initial prediction when sample weights are set
        // (wsum > 0 is guaranteed by Task::with_weights validation).
        let initial = match sw {
            Some(w) => {
                let wsum: f64 = w.iter().sum();
                target.iter().zip(w).map(|(y, wi)| y * wi).sum::<f64>() / wsum
            }
            None => target.iter().sum::<f64>() / ns as f64,
        };
        let mut preds = vec![initial; ns];
        let mut eval_preds = eval.map(|(ef, _)| vec![initial; ef.nrows()]);
        let mut trees = Vec::with_capacity(self.n_estimators);
        let mut imp = vec![0.0; nf];
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut stopper = EarlyStopper::new(self.early_stopping_rounds);

        for _ in 0..self.n_estimators {
            let mut grads: Vec<f64> = preds.iter().zip(target).map(|(p, y)| p - y).collect();
            let mut hess = vec![1.0; ns];
            if let Some(w) = sw {
                // Weights scale gradient AND hessian before any accumulation;
                // GOSS below then samples on the weighted |gradient|.
                for i in 0..ns {
                    grads[i] *= w[i];
                    hess[i] *= w[i];
                }
            }

            let bag = self.sample_rows(&mut rng, ns);
            let (selected, weights) =
                goss_sample(&grads, &hess, &bag, self.top_rate, self.other_rate, &mut rng);
            let cols = self.sample_cols(&mut rng, nf);

            let tree = build_leaf_wise_tree(
                &bins,
                &grads,
                &hess,
                &weights,
                selected,
                &cols,
                self.num_leaves,
                self.max_depth,
                self.lambda,
                self.min_child_weight,
                &mut imp,
            );

            for i in 0..ns {
                preds[i] += self.learning_rate * tree.predict_one(features.row(i));
            }
            if let (Some(ep), Some((ef, _))) = (&mut eval_preds, eval) {
                for i in 0..ef.nrows() {
                    ep[i] += self.learning_rate * tree.predict_one(ef.row(i));
                }
            }
            trees.push(tree);

            if stopper.is_active() {
                // MSE on the held-out set when provided (unweighted: weights
                // are a training-row property), else (weighted) MSE on train.
                let loss = if let (Some(ep), Some((_, et))) = (&eval_preds, eval) {
                    ep.iter().zip(et).map(|(p, y)| (p - y).powi(2)).sum::<f64>() / ep.len() as f64
                } else {
                    match sw {
                        Some(w) => {
                            let wsum: f64 = w.iter().sum::<f64>().max(1e-12);
                            preds
                                .iter()
                                .zip(target)
                                .zip(w)
                                .map(|((p, y), wi)| wi * (p - y).powi(2))
                                .sum::<f64>()
                                / wsum
                        }
                        None => {
                            preds.iter().zip(target).map(|(p, y)| (p - y).powi(2)).sum::<f64>()
                                / ns as f64
                        }
                    }
                };
                if let Some(best_n) = stopper.update(loss, trees.len()) {
                    trees.truncate(best_n);
                    break;
                }
            }
        }

        Ok(Box::new(TrainedLightGBM {
            trees,
            initial: vec![initial],
            learning_rate: self.learning_rate,
            mode: LGBMode::Regression,
            feature_names: task.feature_names().to_vec(),
            feature_importances: imp,
        }))
    }

    fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        let nc = task.n_classes();
        if nc == 2 {
            self.train_binary(task)
        } else {
            self.train_multiclass(task)
        }
    }
}

impl LightGBM {
    fn train_binary(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        let features = task.features();
        let target = task.target();
        let (ns, nf) = (task.n_samples(), task.n_features());
        let sw = task.weights();
        let eval = validate_eval_classif(&self.eval_set, nf)?;
        let bins = HistBins::build_typed(features, self.n_bins, task.feature_types());
        // Weighted positive-class fraction as the initial log-odds when
        // sample weights are set.
        let p_pos = match sw {
            Some(w) => {
                let wsum: f64 = w.iter().sum();
                target
                    .iter()
                    .zip(w)
                    .map(|(&t, wi)| if t == 1 { *wi } else { 0.0 })
                    .sum::<f64>()
                    / wsum
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

        for _ in 0..self.n_estimators {
            let mut grads: Vec<f64> = (0..ns).map(|i| sigmoid(fv[i]) - target[i] as f64).collect();
            let mut hess: Vec<f64> = (0..ns)
                .map(|i| {
                    let p = sigmoid(fv[i]);
                    p * (1.0 - p).max(1e-15)
                })
                .collect();
            if let Some(w) = sw {
                // Weights scale gradient AND hessian before any accumulation;
                // GOSS below then samples on the weighted |gradient|.
                for i in 0..ns {
                    grads[i] *= w[i];
                    hess[i] *= w[i];
                }
            }
            let bag = self.sample_rows(&mut rng, ns);
            let (selected, weights) =
                goss_sample(&grads, &hess, &bag, self.top_rate, self.other_rate, &mut rng);
            let cols = self.sample_cols(&mut rng, nf);

            let tree = build_leaf_wise_tree(
                &bins,
                &grads,
                &hess,
                &weights,
                selected,
                &cols,
                self.num_leaves,
                self.max_depth,
                self.lambda,
                self.min_child_weight,
                &mut imp,
            );

            for i in 0..ns {
                fv[i] += self.learning_rate * tree.predict_one(features.row(i));
            }
            if let (Some(efv), Some((ef, _))) = (&mut eval_fv, eval) {
                for i in 0..ef.nrows() {
                    efv[i] += self.learning_rate * tree.predict_one(ef.row(i));
                }
            }
            trees.push(tree);

            if stopper.is_active() {
                let eps = 1e-15;
                let logloss = |f: f64, y: usize| {
                    let p = sigmoid(f).clamp(eps, 1.0 - eps);
                    let y = y as f64;
                    -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
                };
                // Eval-set metric unweighted (weights are a training-row
                // property); train-loss fallback weighted to track the
                // weighted objective actually being fit.
                let loss = if let (Some(efv), Some((_, et))) = (&eval_fv, eval) {
                    efv.iter().zip(et).map(|(&f, &y)| logloss(f, y)).sum::<f64>()
                        / efv.len() as f64
                } else {
                    match sw {
                        Some(w) => {
                            let wsum: f64 = w.iter().sum::<f64>().max(1e-12);
                            fv.iter()
                                .zip(target)
                                .zip(w)
                                .map(|((&f, &y), wi)| wi * logloss(f, y))
                                .sum::<f64>()
                                / wsum
                        }
                        None => {
                            fv.iter().zip(target).map(|(&f, &y)| logloss(f, y)).sum::<f64>()
                                / ns as f64
                        }
                    }
                };
                if let Some(best_n) = stopper.update(loss, trees.len()) {
                    trees.truncate(best_n);
                    break;
                }
            }
        }

        Ok(Box::new(TrainedLightGBM {
            trees,
            initial: vec![initial],
            learning_rate: self.learning_rate,
            mode: LGBMode::BinaryClassif,
            feature_names: task.feature_names().to_vec(),
            feature_importances: imp,
        }))
    }

    fn train_multiclass(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        let features = task.features();
        let target = task.target();
        let (ns, nf, nc) = (task.n_samples(), task.n_features(), task.n_classes());
        let sw = task.weights();
        let eval = validate_eval_classif(&self.eval_set, nf)?;
        let bins = HistBins::build_typed(features, self.n_bins, task.feature_types());
        // Weighted per-class frequency as the initial log-prior when sample
        // weights are set.
        let initial: Vec<f64> = match sw {
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

        for _ in 0..self.n_estimators {
            let probs: Vec<Vec<f64>> = fv.iter().map(|f| softmax(f)).collect();
            let cols = self.sample_cols(&mut rng, nf);
            let bag = self.sample_rows(&mut rng, ns);
            for c in 0..nc {
                let mut grads: Vec<f64> = (0..ns)
                    .map(|i| probs[i][c] - if target[i] == c { 1.0 } else { 0.0 })
                    .collect();
                let mut hess: Vec<f64> = (0..ns)
                    .map(|i| (probs[i][c] * (1.0 - probs[i][c])).max(1e-15))
                    .collect();
                if let Some(w) = sw {
                    // Weights scale gradient AND hessian before accumulation;
                    // GOSS below samples on the weighted |gradient|.
                    for i in 0..ns {
                        grads[i] *= w[i];
                        hess[i] *= w[i];
                    }
                }
                let (selected, weights) =
                    goss_sample(&grads, &hess, &bag, self.top_rate, self.other_rate, &mut rng);
                let tree = build_leaf_wise_tree(
                    &bins,
                    &grads,
                    &hess,
                    &weights,
                    selected,
                    &cols,
                    self.num_leaves,
                    self.max_depth,
                    self.lambda,
                    self.min_child_weight,
                    &mut imp,
                );
                for i in 0..ns {
                    fv[i][c] += self.learning_rate * tree.predict_one(features.row(i));
                }
                if let (Some(efv), Some((ef, _))) = (&mut eval_fv, eval) {
                    for i in 0..ef.nrows() {
                        efv[i][c] += self.learning_rate * tree.predict_one(ef.row(i));
                    }
                }
                trees.push(tree);
            }

            if stopper.is_active() {
                let eps = 1e-15;
                // Eval-set metric unweighted; weighted train-loss fallback.
                let loss = if let (Some(efv), Some((_, et))) = (&eval_fv, eval) {
                    let ep: Vec<Vec<f64>> = efv.iter().map(|f| softmax(f)).collect();
                    (0..et.len()).map(|i| -ep[i][et[i]].max(eps).ln()).sum::<f64>()
                        / et.len() as f64
                } else {
                    let pn: Vec<Vec<f64>> = fv.iter().map(|f| softmax(f)).collect();
                    let per_point = |i: usize| -pn[i][target[i]].max(eps).ln();
                    match sw {
                        Some(w) => {
                            let wsum: f64 = w.iter().sum::<f64>().max(1e-12);
                            (0..ns).map(|i| w[i] * per_point(i)).sum::<f64>() / wsum
                        }
                        None => (0..ns).map(per_point).sum::<f64>() / ns as f64,
                    }
                };
                if let Some(best_n) = stopper.update(loss, trees.len()) {
                    trees.truncate(best_n);
                    break;
                }
            }
        }

        Ok(Box::new(TrainedLightGBM {
            trees,
            initial,
            learning_rate: self.learning_rate,
            mode: LGBMode::MultiClassif { n_classes: nc },
            feature_names: task.feature_names().to_vec(),
            feature_importances: imp,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for M6 (docs/auditoria_motor_2026-07-05.md, Fase F):
    /// `leaf_weight` and `split_gain` divide by `h + lambda` with no floor.
    /// `lambda=0.0` is the default, and a genuinely (or near-)zero-hessian
    /// leaf/child is reachable whenever a user sets `min_child_weight=0.0`
    /// (its own default is 1.0, which normally keeps this from triggering) --
    /// without a clamp this produced `-g/0.0` = +-infinity or NaN instead of
    /// a finite (if extreme) leaf weight.
    #[test]
    fn leaf_weight_and_split_gain_stay_finite_at_zero_hessian_and_lambda() {
        let lw = leaf_weight(&[1.0], &[0.0], &[1.0], &[0], 0.0);
        assert!(lw.is_finite(), "leaf_weight with h=0, lambda=0 must not be NaN/Inf, got {lw}");

        let g = split_gain(1.0, 0.0, -1.0, 0.0, 0.0);
        assert!(g.is_finite(), "split_gain with hl=hr=0, lambda=0 must not be NaN/Inf, got {g}");
    }

    /// Regression test for the histogram binning boundary bug (src/learner/histogram.rs):
    /// LightGBM always uses histogram splits, so a binary feature that perfectly
    /// determines the target used to be unsplittable (both values collapsed into
    /// bin 0), making the model predict the global mean regardless of n.
    #[test]
    fn binary_feature_is_splittable() {
        let n = 600;
        let mut features = Array2::<f64>::zeros((n, 1));
        let mut target = vec![0.0; n];
        for i in 0..n {
            let bit = (i % 2) as f64;
            features[[i, 0]] = bit;
            target[i] = bit * 10.0;
        }
        let task = RegressionTask::new("binary", features.clone(), target.clone()).unwrap();
        let mut model = LightGBM::new().with_n_estimators(20).with_learning_rate(0.5);
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

    /// Regression test (5th audit, LOW-A): the plain-GBDT default
    /// (top_rate=1.0, other_rate=0.0) paid goss_sample's O(n log n)
    /// sort-by-|gradient| for nothing on every tree. The shortcut must be
    /// behaviorally invisible:
    /// 1. the default and explicitly-spelled GOSS 1.0/0.0 train
    ///    bit-identically (both are the same degenerate configuration and
    ///    take the shortcut);
    /// 2. `(top_rate=1.0, other_rate=0.5)` still takes the OLD general path
    ///    (sorts, then samples from an empty "rest", ending with the same
    ///    all-candidates selection and all-1.0 weights the pre-shortcut
    ///    default produced). Its trained predictions agree with the
    ///    shortcut's to floating-point reassociation only: the selection
    ///    set, weights, and RNG stream are identical (pinned exactly by
    ///    `goss_sample_shortcut_preserves_selection_weights_and_rng`), but
    ///    histogram/leaf sums accumulate in |gradient|-sorted vs natural
    ///    index order, so individual predictions may differ at ulp level —
    ///    the same class of divergence the 4th audit accepted for the M-3
    ///    incremental sweep ("paridad de calidad, diferencias a nivel de
    ///    ulp").
    #[test]
    fn gbdt_default_shortcut_matches_the_sorting_path_bit_for_bit() {
        let n = 200;
        let features = Array2::from_shape_fn((n, 2), |(i, j)| {
            ((i as f64) * (0.37 + 0.21 * j as f64)).sin() * 3.0
        });
        let target: Vec<f64> = (0..n)
            .map(|i| features[[i, 0]] * 2.0 + features[[i, 1]] + (i as f64 * 0.71).cos())
            .collect();
        let task = RegressionTask::new("goss_shortcut", features.clone(), target).unwrap();

        let train = |lgbm: &mut LightGBM| -> Vec<f64> {
            let model = lgbm.train_regress(&task).unwrap();
            let Prediction::Regression { predicted, .. } = model.predict(&features).unwrap() else {
                panic!("expected regression");
            };
            predicted
        };

        let mut default_cfg = LightGBM::new().with_n_estimators(20);
        let default_preds = train(&mut default_cfg);

        let mut explicit_goss = LightGBM::new()
            .with_n_estimators(20)
            .with_top_rate(1.0)
            .with_other_rate(0.0);
        assert_eq!(
            default_preds,
            train(&mut explicit_goss),
            "default and explicit GOSS 1.0/0.0 must be the same configuration"
        );

        let mut sorting_path = LightGBM::new()
            .with_n_estimators(20)
            .with_top_rate(1.0)
            .with_other_rate(0.5); // rest is empty, so nothing is actually sampled
        let sorting_preds = train(&mut sorting_path);
        for (i, (a, b)) in default_preds.iter().zip(&sorting_preds).enumerate() {
            assert!(
                (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0),
                "sample {i}: shortcut ({a}) vs sorting path ({b}) may differ only by \
                 floating-point reassociation, not materially"
            );
        }
    }

    /// Unit-level pin for the shortcut: same selection SET, identical
    /// weights vector, and untouched RNG stream versus the general path in
    /// the degenerate keep-everything configuration.
    #[test]
    fn goss_sample_shortcut_preserves_selection_weights_and_rng() {
        use rand::Rng;
        let n = 50;
        let grads: Vec<f64> = (0..n).map(|i| ((i * 37) % 11) as f64 - 5.0).collect();
        let hess = vec![1.0; n];
        let candidates: Vec<usize> = (0..n).filter(|i| i % 3 != 0).collect();

        let mut rng_shortcut = StdRng::seed_from_u64(9);
        let (sel_shortcut, w_shortcut) =
            goss_sample(&grads, &hess, &candidates, 1.0, 0.0, &mut rng_shortcut);

        let mut rng_general = StdRng::seed_from_u64(9);
        let (sel_general, w_general) =
            goss_sample(&grads, &hess, &candidates, 1.0, 0.5, &mut rng_general);

        assert_eq!(sel_shortcut, candidates, "shortcut keeps every candidate");
        let mut sorted_general = sel_general;
        sorted_general.sort_unstable();
        assert_eq!(sorted_general, candidates, "general path also keeps every candidate");
        assert_eq!(w_shortcut, w_general, "weights must be identical (1.0 on candidates)");
        assert_eq!(
            rng_shortcut.random::<u64>(),
            rng_general.random::<u64>(),
            "neither path may consume RNG state in the degenerate configuration"
        );
    }

    /// Regression test for GOSS's central correction (Ke et al. 2017,
    /// Algorithm 2): previously, all 3 call sites discarded the amplification
    /// weights goss_sample computed and passed `vec![1.0; ns]` instead, so
    /// the sampled "small gradient" subset's contribution to the
    /// gradient/hessian sum was never amplified back up -- systematically
    /// biasing the sum low by roughly `1 - (top_rate + other_rate)`.
    ///
    /// With hess=1 everywhere, the true total hessian mass is exactly `n`.
    /// The weighted sum over GOSS's selection should approximate that in
    /// expectation (top points contribute directly with weight 1; the
    /// randomly-sampled "other" points are scaled by `amplify` so their
    /// expected contribution reconstructs the full "other" population's
    /// mass). The old bug would give top_n + other_n ≈ n*(top_rate+other_rate),
    /// badly biased low.
    #[test]
    fn goss_sample_amplification_gives_unbiased_weighted_sum() {
        let n = 1000;
        let hess = vec![1.0; n];
        let mut grads = vec![0.0; n];
        for g in grads.iter_mut().take(100) {
            *g = 10.0; // clear top-gradient subset
        }
        for g in grads.iter_mut().skip(100) {
            *g = 0.1; // small-gradient tail
        }

        let top_rate = 0.2;
        let other_rate = 0.1;
        let trials = 200;
        let all: Vec<usize> = (0..n).collect();
        let mut total = 0.0;
        for t in 0..trials {
            let mut rng = StdRng::seed_from_u64(t as u64);
            let (selected, weights) =
                goss_sample(&grads, &hess, &all, top_rate, other_rate, &mut rng);
            total += selected.iter().map(|&i| weights[i] * hess[i]).sum::<f64>();
        }
        let avg = total / trials as f64;

        assert!(
            (avg - n as f64).abs() < n as f64 * 0.15,
            "average weighted hessian sum ({avg:.1}) should approximate the true \
             total ({n}) -- the unweighted (buggy) estimate would be ~{:.0}",
            n as f64 * (top_rate + other_rate)
        );
    }

    /// 4th-audit LOW: the amplification factor used to be the asymptotic
    /// (1-a)/b with the denominator clamped at 0.01, so an opted-in
    /// other_rate below the clamp silently under-amplified the sampled
    /// tail. It is now the exact finite-sample correction |rest|/|sampled|.
    #[test]
    fn goss_sample_amplifies_exactly_below_the_old_clamp() {
        let n = 1000;
        let hess = vec![1.0; n];
        let mut grads = vec![0.1; n];
        for g in grads.iter_mut().take(100) {
            *g = 10.0; // unmistakably the top 10%
        }
        let all: Vec<usize> = (0..n).collect();
        // top_n = 100, rest = 900, other_n = ceil(1000*0.005) = 5:
        // exact amplification 900/5 = 180. The old clamped form gave
        // (1-0.1)/max(0.005, 0.01) = 90 -- half the correct weight.
        let mut rng = StdRng::seed_from_u64(0);
        let (selected, weights) = goss_sample(&grads, &hess, &all, 0.1, 0.005, &mut rng);
        let sampled_tail: Vec<usize> = selected.iter().copied().filter(|&i| i >= 100).collect();
        assert_eq!(sampled_tail.len(), 5);
        for &i in &sampled_tail {
            assert!(
                (weights[i] - 180.0).abs() < 1e-9,
                "tail weight must be |rest|/|sampled| = 180, got {}",
                weights[i]
            );
        }
    }

    #[test]
    fn goss_sample_top_gradients_always_selected_with_unit_weight() {
        let n = 200;
        let hess = vec![1.0; n];
        let mut grads = vec![0.1; n];
        for g in grads.iter_mut().take(20) {
            *g = 100.0; // unmistakably the top 20
        }
        let mut rng = StdRng::seed_from_u64(0);
        let all: Vec<usize> = (0..n).collect();
        let (selected, weights) = goss_sample(&grads, &hess, &all, 0.1, 0.1, &mut rng);

        for i in 0..20 {
            assert!(selected.contains(&i), "top-gradient sample {i} should always be selected");
            assert_eq!(weights[i], 1.0, "top-gradient samples must have weight 1.0");
        }
    }

    /// Regression test for the HIGH finding: `subsample` was accepted by
    /// the builder and stored, but never consulted anywhere in training --
    /// confirmed by a probe showing `subsample=1.0` and `subsample=0.05`
    /// produced bit-for-bit identical predictions. With row bagging wired
    /// into `goss_sample`'s candidate pool, a small `subsample` must change
    /// the fitted model.
    #[test]
    fn subsample_actually_changes_the_fitted_model() {
        use rand::Rng as _;
        let mut rng = StdRng::seed_from_u64(7);
        let n = 500;
        let mut feats = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);
        for _ in 0..n {
            let x: f64 = rng.random::<f64>() * 10.0;
            feats.push(x);
            target.push(2.0 * x + rng.random::<f64>());
        }
        let features = Array2::from_shape_vec((n, 1), feats).unwrap();
        let task = RegressionTask::new("lgb_subsample", features.clone(), target).unwrap();

        let mut full = LightGBM::new().with_n_estimators(20).with_subsample(1.0);
        let mut bagged = LightGBM::new().with_n_estimators(20).with_subsample(0.05);

        let pred_full = full.train_regress(&task).unwrap().predict(&features).unwrap();
        let pred_bagged = bagged.train_regress(&task).unwrap().predict(&features).unwrap();

        let (Prediction::Regression { predicted: p_full, .. }, Prediction::Regression { predicted: p_bagged, .. }) =
            (pred_full, pred_bagged)
        else {
            panic!("expected regression predictions");
        };
        let max_abs_diff = p_full
            .iter()
            .zip(&p_bagged)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            max_abs_diff > 1e-6,
            "subsample=0.05 should produce different predictions than subsample=1.0, \
             but max |diff| = {max_abs_diff} (subsample is being silently ignored)"
        );
    }

    /// y depends on the parity of a 7-code categorical feature: one numeric
    /// threshold cannot separate {0,2,4,6} from {1,3,5}, one native
    /// categorical split can. Trees restricted to a single split
    /// (num_leaves=2) discriminate the two mechanisms.
    #[test]
    fn categorical_split_beats_numeric_threshold() {
        let n = 700;
        let mut features = Array2::<f64>::zeros((n, 1));
        let mut target = vec![0.0; n];
        for i in 0..n {
            features[[i, 0]] = (i % 7) as f64;
            target[i] = ((i % 7) % 2) as f64 * 10.0;
        }
        let num_task = RegressionTask::new("parity", features.clone(), target.clone()).unwrap();
        let cat_task = RegressionTask::new("parity", features.clone(), target.clone())
            .unwrap()
            .with_categorical_features(&[0])
            .unwrap();

        let stumps = || {
            LightGBM::new()
                .with_n_estimators(3)
                .with_num_leaves(2)
                .with_learning_rate(1.0)
                .with_top_rate(1.0)
                .with_other_rate(0.0)
        };
        let rmse = |m: &dyn TrainedModel| {
            let Prediction::Regression { predicted, .. } = m.predict(&features).unwrap() else {
                unreachable!()
            };
            (predicted.iter().zip(&target).map(|(p, y)| (p - y).powi(2)).sum::<f64>()
                / n as f64)
                .sqrt()
        };

        let m_cat = stumps().train_regress(&cat_task).unwrap();
        let m_num = stumps().train_regress(&num_task).unwrap();
        assert!(rmse(&*m_cat) < 1.0, "categorical split should fit parity, got {}", rmse(&*m_cat));
        assert!(
            rmse(&*m_num) > 2.0,
            "numeric thresholds cannot fit parity with 3 single-split trees, got {}",
            rmse(&*m_num)
        );

        // Unseen category and NaN at prediction time route safely.
        let unseen = Array2::from_shape_vec((2, 1), vec![42.0, f64::NAN]).unwrap();
        let Prediction::Regression { predicted, .. } = m_cat.predict(&unseen).unwrap() else {
            unreachable!()
        };
        assert!(predicted.iter().all(|p| p.is_finite()));
    }

    /// Eval-set early stopping must stop at the best held-out round on an
    /// overfittable config, generalizing better than train-loss monitoring
    /// (which almost never fires under boosting). GOSS is disabled
    /// (top_rate=1, other_rate=0) so the run is deterministic.
    #[test]
    fn eval_set_early_stopping_generalizes_better() {
        use rand::Rng;
        let n = 40;
        let make = |seed: u64| {
            let mut r = StdRng::seed_from_u64(seed);
            let mut feats = Vec::with_capacity(n);
            let mut target = Vec::with_capacity(n);
            for i in 0..n {
                feats.push(i as f64);
                target.push(i as f64 * 0.1 + r.random::<f64>() * 8.0);
            }
            (Array2::from_shape_vec((n, 1), feats).unwrap(), target)
        };
        let (tr_f, tr_t) = make(1);
        let (va_f, va_t) = make(2);
        let task = RegressionTask::new("es", tr_f, tr_t).unwrap();

        let rmse = |m: &dyn TrainedModel| {
            let Prediction::Regression { predicted, .. } = m.predict(&va_f).unwrap() else {
                panic!("expected regression")
            };
            (predicted.iter().zip(&va_t).map(|(p, y)| (p - y).powi(2)).sum::<f64>()
                / va_t.len() as f64)
                .sqrt()
        };

        let base = || {
            LightGBM::new()
                .with_n_estimators(500)
                .with_num_leaves(31)
                .with_learning_rate(0.5)
                .with_top_rate(1.0)
                .with_other_rate(0.0)
                .with_early_stopping_rounds(3)
        };
        let m_train = base().train_regress(&task).unwrap();
        let m_eval = base()
            .with_eval_set_regress(va_f.clone(), va_t.clone())
            .train_regress(&task)
            .unwrap();
        assert!(
            rmse(&*m_eval) < rmse(&*m_train),
            "eval-set early stopping should generalize better: eval={:.3} train-loss={:.3}",
            rmse(&*m_eval),
            rmse(&*m_train)
        );
    }

    #[test]
    fn eval_set_task_type_mismatch_is_rejected() {
        let features = Array2::<f64>::zeros((10, 1));
        let task = RegressionTask::new("es_type", features, vec![0.0; 10]).unwrap();
        let mut mismatched = LightGBM::new()
            .with_early_stopping_rounds(2)
            .with_eval_set_classif(Array2::<f64>::zeros((5, 1)), vec![0usize; 5]);
        assert!(mismatched.train_regress(&task).is_err());
    }

    fn count_leaves(node: &LGBNode) -> usize {
        match node {
            LGBNode::Leaf { .. } => 1,
            LGBNode::Split { left, right, .. } | LGBNode::CatSplit { left, right, .. } => {
                count_leaves(left) + count_leaves(right)
            }
        }
    }

    /// Regression test for the leaf-wise growth that was silently discarded:
    /// `build_leaf_wise_tree` used to perform the real best-first search
    /// (cached histograms, subtraction, picking the globally best-gain leaf
    /// each round) but then threw the result away and rebuilt the tree from
    /// scratch via depth-first `build_recursive` on the flattened leaf
    /// indices -- a structurally different algorithm. The fix keeps the
    /// actual arena of splits built during the search and converts it
    /// directly into an `LGBNode` tree, so asking for N leaves must produce
    /// exactly N leaves (not "however many depth-first-with-a-counter
    /// happens to produce").
    #[test]
    fn leaf_wise_growth_produces_exactly_the_requested_leaf_count() {
        let n = 200;
        let nf = 3;
        let mut feats = Vec::with_capacity(n * nf);
        let mut grads = Vec::with_capacity(n);
        for i in 0..n {
            let x0 = (i % 20) as f64;
            let x1 = ((i * 7) % 13) as f64;
            let x2 = ((i * 3) % 17) as f64;
            feats.extend_from_slice(&[x0, x1, x2]);
            grads.push(x0 * 2.0 - x1 + x2 * 0.5 - 10.0);
        }
        let features = Array2::from_shape_vec((n, nf), feats).unwrap();
        let bins = HistBins::build(&features, 32);
        let hess = vec![1.0; n];
        let weights = vec![1.0; n];
        let cols: Vec<usize> = (0..nf).collect();

        for &num_leaves in &[2usize, 4, 8, 16] {
            let mut imp_local = vec![0.0; nf];
            let tree = build_leaf_wise_tree(
                &bins,
                &grads,
                &hess,
                &weights,
                (0..n).collect(),
                &cols,
                num_leaves,
                None,
                1.0,
                1.0,
                &mut imp_local,
            );
            let leaves = count_leaves(&tree);
            assert_eq!(
                leaves, num_leaves,
                "requesting {num_leaves} leaves should produce exactly that many, got {leaves}"
            );
        }
    }
}
