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
use crate::learner::{Learner, TrainedModel};
use crate::prediction::Prediction;
use crate::task::{ClassificationTask, RegressionTask, Task};
use ndarray::{Array2, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::histogram::{HistBins, NAN_BIN};

/// LightGBM-inspired leaf-wise GBM with GOSS sampling.
///
/// Implements leaf-wise growth and gradient-based one-side sampling from
/// Ke et al. (2017). Does not include Exclusive Feature Bundling (EFB)
/// or weighted GOSS histograms.
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
            top_rate: 0.2,
            other_rate: 0.1,
            n_bins: 255,
            subsample: 1.0,
            colsample_bytree: 1.0,
            seed: 42,
        }
    }
}

impl LightGBM {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }
    pub fn with_num_leaves(mut self, n: usize) -> Self {
        self.num_leaves = n;
        self
    }
    pub fn with_max_depth(mut self, d: usize) -> Self {
        self.max_depth = Some(d);
        self
    }
    pub fn with_lambda(mut self, l: f64) -> Self {
        self.lambda = l;
        self
    }
    pub fn with_min_child_weight(mut self, w: f64) -> Self {
        self.min_child_weight = w;
        self
    }
    pub fn with_top_rate(mut self, r: f64) -> Self {
        self.top_rate = r;
        self
    }
    pub fn with_other_rate(mut self, r: f64) -> Self {
        self.other_rate = r;
        self
    }
    pub fn with_subsample(mut self, s: f64) -> Self {
        self.subsample = s;
        self
    }
    pub fn with_colsample_bytree(mut self, c: f64) -> Self {
        self.colsample_bytree = c;
        self
    }
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }
}

type Bins = HistBins;

// ── Tree node ───────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
pub enum LGBNode {
    Leaf {
        weight: f64,
    },
    Split {
        feature: usize,
        threshold: f64,
        nan_left: bool,
        left: Box<LGBNode>,
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
        }
    }
}

// ── GOSS sampling ───────────────────────────────────────────────────

fn goss_sample(
    grads: &[f64],
    _hess: &[f64],
    top_rate: f64,
    other_rate: f64,
    rng: &mut StdRng,
) -> (Vec<usize>, Vec<f64>) {
    let n = grads.len();
    let mut sorted: Vec<usize> = (0..n).collect();
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
    // without it, the sampled sum is systematically biased low.
    let amplify = if other_n > 0 {
        (1.0 - top_rate) / other_rate.max(0.01)
    } else {
        1.0
    };

    let mut selected: Vec<usize> = top_indices.to_vec();
    selected.extend_from_slice(&sampled_rest);

    // Scatter weights back to original sample indices (0..n): build_recursive
    // and find_best_split_hist look up `weights[i]` by original sample id,
    // not by position within `selected`.
    let mut weights = vec![0.0; n];
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
        .map(|&feat| {
            let nb = bins.boundaries[feat].len();
            let mut bg = vec![0.0; nb];
            let mut bh = vec![0.0; nb];
            let mut ng = 0.0;
            let mut nh = 0.0;
            for &idx in indices {
                let b = bins.get_bin(feat, idx);
                let w = weights[idx];
                if b == NAN_BIN {
                    ng += grads[idx] * w;
                    nh += hess[idx] * w;
                } else {
                    bg[b as usize] += grads[idx] * w;
                    bh[b as usize] += hess[idx] * w;
                }
            }
            (bg, bh, ng, nh)
        })
        .collect()
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
            let nb = bin_g.len();
            let total_g: f64 = bin_g.iter().sum::<f64>() + nan_g;
            let total_h: f64 = bin_h.iter().sum::<f64>() + nan_h;
            let mut best_gain = 0.0;
            let mut best: Option<(usize, f64, bool)> = None;

            let (mut gl, mut hl) = (0.0, 0.0);
            for bin in 0..nb.saturating_sub(1) {
                gl += bin_g[bin];
                hl += bin_h[bin];
                let (gr, hr) = (total_g - gl, total_h - hl);
                if hl < min_child_weight || hr < min_child_weight {
                    continue;
                }
                let gain = 0.5
                    * (gl * gl / (hl + lambda) + gr * gr / (hr + lambda)
                        - total_g * total_g / (total_h + lambda));
                if gain > best_gain {
                    best_gain = gain;
                    best = Some((bin, gain, false));
                }
            }
            if *nan_h > 0.0 {
                let (mut gl, mut hl) = (*nan_g, *nan_h);
                for bin in 0..nb.saturating_sub(1) {
                    gl += bin_g[bin];
                    hl += bin_h[bin];
                    let (gr, hr) = (total_g - gl, total_h - hl);
                    if hl < min_child_weight || hr < min_child_weight {
                        continue;
                    }
                    let gain = 0.5
                        * (gl * gl / (hl + lambda) + gr * gr / (hr + lambda)
                            - total_g * total_g / (total_h + lambda));
                    if gain > best_gain {
                        best_gain = gain;
                        best = Some((bin, gain, true));
                    }
                }
            }
            best.map(|(bin, gain, nan_left)| LeafCandidate {
                gain,
                feature: feat,
                threshold: bins.boundaries[feat][bin],
                nan_left,
                split_bin: bin,
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
    feature_importances: &mut Vec<f64>,
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
        feature_importances[split.feature] += split.gain;

        let placeholder = ArenaNode::Split {
            feature: split.feature,
            threshold: split.threshold,
            nan_left: split.nan_left,
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
                (b as usize) <= split.split_bin
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
            ArenaNode::Split { feature, threshold, nan_left, left, right } => LGBNode::Split {
                feature,
                threshold,
                nan_left,
                left: Box::new(to_lgb_node(arena, left, grads, hess, weights, lambda)),
                right: Box::new(to_lgb_node(arena, right, grads, hess, weights, lambda)),
            },
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
    -g / (h + lambda)
}

// ── Trained model ───────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
pub(crate) enum LGBMode {
    Regression,
    BinaryClassif,
    MultiClassif { n_classes: usize },
}

#[derive(Serialize, Deserialize)]
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
                let predicted: Vec<f64> = features
                    .rows()
                    .into_iter()
                    .map(|r| {
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
                let mut predicted = Vec::with_capacity(features.nrows());
                let mut probabilities = Vec::with_capacity(features.nrows());
                for r in features.rows() {
                    let mut f = self.initial[0];
                    for t in &self.trees {
                        f += self.learning_rate * t.predict_one(r);
                    }
                    let p = sigmoid(f);
                    predicted.push(if p >= 0.5 { 1 } else { 0 });
                    probabilities.push(vec![1.0 - p, p]);
                }
                Ok(Prediction::Classification {
                    predicted,
                    truth: None,
                    probabilities: Some(probabilities),
                })
            }
            LGBMode::MultiClassif { n_classes } => {
                let k = *n_classes;
                let mut predicted = Vec::with_capacity(features.nrows());
                let mut probabilities = Vec::with_capacity(features.nrows());
                for r in features.rows() {
                    let mut scores = self.initial.clone();
                    let ni = self.trees.len() / k;
                    for i in 0..ni {
                        for c in 0..k {
                            scores[c] += self.learning_rate * self.trees[i * k + c].predict_one(r);
                        }
                    }
                    let probs = softmax(&scores);
                    let pred = probs
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap()
                        .0;
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
}

impl Learner for LightGBM {
    fn id(&self) -> &str {
        "lightgbm"
    }

    fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
        let features = task.features();
        let target = task.target();
        let (ns, nf) = (task.n_samples(), task.n_features());
        let bins = HistBins::build(features, self.n_bins);
        let initial = target.iter().sum::<f64>() / ns as f64;
        let mut preds = vec![initial; ns];
        let mut trees = Vec::with_capacity(self.n_estimators);
        let mut imp = vec![0.0; nf];
        let mut rng = StdRng::seed_from_u64(self.seed);

        for _ in 0..self.n_estimators {
            let grads: Vec<f64> = preds.iter().zip(target).map(|(p, y)| p - y).collect();
            let hess = vec![1.0; ns];

            let (selected, weights) =
                goss_sample(&grads, &hess, self.top_rate, self.other_rate, &mut rng);
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
            trees.push(tree);
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
        let bins = HistBins::build(features, self.n_bins);
        let p_pos = target.iter().filter(|&&t| t == 1).count() as f64 / ns as f64;
        let initial = (p_pos / (1.0 - p_pos).max(1e-15)).ln();
        let mut fv = vec![initial; ns];
        let mut trees = Vec::with_capacity(self.n_estimators);
        let mut imp = vec![0.0; nf];
        let mut rng = StdRng::seed_from_u64(self.seed);

        for _ in 0..self.n_estimators {
            let grads: Vec<f64> = (0..ns).map(|i| sigmoid(fv[i]) - target[i] as f64).collect();
            let hess: Vec<f64> = (0..ns)
                .map(|i| {
                    let p = sigmoid(fv[i]);
                    p * (1.0 - p).max(1e-15)
                })
                .collect();
            let (selected, weights) =
                goss_sample(&grads, &hess, self.top_rate, self.other_rate, &mut rng);
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
            trees.push(tree);
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
        let bins = HistBins::build(features, self.n_bins);
        let mut cc = vec![0usize; nc];
        for &t in target {
            cc[t] += 1;
        }
        let initial: Vec<f64> = cc
            .iter()
            .map(|&c| ((c as f64 / ns as f64).max(1e-15)).ln())
            .collect();
        let mut fv: Vec<Vec<f64>> = (0..ns).map(|_| initial.clone()).collect();
        let mut trees = Vec::with_capacity(self.n_estimators * nc);
        let mut imp = vec![0.0; nf];
        let mut rng = StdRng::seed_from_u64(self.seed);

        for _ in 0..self.n_estimators {
            let probs: Vec<Vec<f64>> = fv.iter().map(|f| softmax(f)).collect();
            let cols = self.sample_cols(&mut rng, nf);
            for c in 0..nc {
                let grads: Vec<f64> = (0..ns)
                    .map(|i| probs[i][c] - if target[i] == c { 1.0 } else { 0.0 })
                    .collect();
                let hess: Vec<f64> = (0..ns)
                    .map(|i| (probs[i][c] * (1.0 - probs[i][c])).max(1e-15))
                    .collect();
                let (selected, weights) =
                    goss_sample(&grads, &hess, self.top_rate, self.other_rate, &mut rng);
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
                trees.push(tree);
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
        let mut total = 0.0;
        for t in 0..trials {
            let mut rng = StdRng::seed_from_u64(t as u64);
            let (selected, weights) = goss_sample(&grads, &hess, top_rate, other_rate, &mut rng);
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

    #[test]
    fn goss_sample_top_gradients_always_selected_with_unit_weight() {
        let n = 200;
        let hess = vec![1.0; n];
        let mut grads = vec![0.1; n];
        for g in grads.iter_mut().take(20) {
            *g = 100.0; // unmistakably the top 20
        }
        let mut rng = StdRng::seed_from_u64(0);
        let (selected, weights) = goss_sample(&grads, &hess, 0.1, 0.1, &mut rng);

        for i in 0..20 {
            assert!(selected.contains(&i), "top-gradient sample {i} should always be selected");
            assert_eq!(weights[i], 1.0, "top-gradient samples must have weight 1.0");
        }
    }

    fn count_leaves(node: &LGBNode) -> usize {
        match node {
            LGBNode::Leaf { .. } => 1,
            LGBNode::Split { left, right, .. } => count_leaves(left) + count_leaves(right),
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
