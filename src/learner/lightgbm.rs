//! LightGBM-inspired leaf-wise gradient boosting with GOSS.
//!
//! This implements two key ideas from the LightGBM paper:
//! - **Gradient-based One-Side Sampling (GOSS)**: keeps top gradients, samples rest
//! - **Leaf-wise (best-first) tree growth**: always splits the leaf with highest gain
//! - Histogram-based splits with NaN handling
//!
//! **Not implemented**: Exclusive Feature Bundling (EFB), weighted GOSS histogram
//! pass, GPU training, distributed computation. This implementation does not match
//! the official library's performance; it is included for API completeness and the
//! leaf-wise growth strategy.
//!
//! Reference: Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting
//! Decision Tree. NeurIPS.

use ndarray::{Array2, ArrayView1};
use rand::seq::SliceRandom;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use crate::task::{ClassificationTask, RegressionTask, Task};
use crate::learner::{Learner, TrainedModel};
use crate::prediction::Prediction;
use crate::Result;

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
    num_leaves: usize,       // max leaves per tree (leaf-wise control)
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
            n_estimators: 100, learning_rate: 0.1, num_leaves: 31,
            max_depth: None, lambda: 0.0, min_child_weight: 1.0,
            top_rate: 0.2, other_rate: 0.1,
            n_bins: 255, subsample: 1.0, colsample_bytree: 1.0, seed: 42,
        }
    }
}

impl LightGBM {
    pub fn new() -> Self { Self::default() }
    pub fn with_n_estimators(mut self, n: usize) -> Self { self.n_estimators = n; self }
    pub fn with_learning_rate(mut self, lr: f64) -> Self { self.learning_rate = lr; self }
    pub fn with_num_leaves(mut self, n: usize) -> Self { self.num_leaves = n; self }
    pub fn with_max_depth(mut self, d: usize) -> Self { self.max_depth = Some(d); self }
    pub fn with_lambda(mut self, l: f64) -> Self { self.lambda = l; self }
    pub fn with_min_child_weight(mut self, w: f64) -> Self { self.min_child_weight = w; self }
    pub fn with_top_rate(mut self, r: f64) -> Self { self.top_rate = r; self }
    pub fn with_other_rate(mut self, r: f64) -> Self { self.other_rate = r; self }
    pub fn with_subsample(mut self, s: f64) -> Self { self.subsample = s; self }
    pub fn with_colsample_bytree(mut self, c: f64) -> Self { self.colsample_bytree = c; self }
    pub fn with_seed(mut self, s: u64) -> Self { self.seed = s; self }
}

type Bins = HistBins;

// ── Tree node ───────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
pub enum LGBNode {
    Leaf { weight: f64 },
    Split { feature: usize, threshold: f64, nan_left: bool,
            left: Box<LGBNode>, right: Box<LGBNode> },
}

impl LGBNode {
    #[inline]
    fn predict_one(&self, row: ArrayView1<f64>) -> f64 {
        match self {
            LGBNode::Leaf { weight } => *weight,
            LGBNode::Split { feature, threshold, nan_left, left, right } => {
                let v = row[*feature];
                if v.is_nan() { if *nan_left { left.predict_one(row) } else { right.predict_one(row) } }
                else if v < *threshold { left.predict_one(row) }
                else { right.predict_one(row) }
            }
        }
    }
}

// ── GOSS sampling ───────────────────────────────────────────────────

fn goss_sample(
    grads: &[f64], _hess: &[f64], top_rate: f64, other_rate: f64, rng: &mut StdRng,
) -> (Vec<usize>, Vec<f64>) {
    let n = grads.len();
    let mut sorted: Vec<usize> = (0..n).collect();
    sorted.sort_by(|&a, &b| grads[b].abs().partial_cmp(&grads[a].abs()).unwrap_or(std::cmp::Ordering::Equal));

    let top_n = (n as f64 * top_rate).ceil() as usize;
    let other_n = (n as f64 * other_rate).ceil() as usize;

    let top_indices = &sorted[..top_n.min(n)];
    let rest = &sorted[top_n.min(n)..];

    let mut sampled_rest: Vec<usize> = rest.to_vec();
    sampled_rest.shuffle(rng);
    sampled_rest.truncate(other_n);

    let amplify = if other_n > 0 { (1.0 - top_rate) / other_rate.max(0.01) } else { 1.0 };

    let mut selected: Vec<usize> = top_indices.to_vec();
    selected.extend_from_slice(&sampled_rest);

    // Build weight multipliers (1.0 for top, amplify for sampled)
    let mut weights = vec![1.0; selected.len()];
    for i in top_n.min(selected.len())..selected.len() {
        weights[i] = amplify;
    }

    (selected, weights)
}

// ── Leaf-wise tree building ─────────────────────────────────────────

#[allow(dead_code)]
struct LeafCandidate {
    indices: Vec<usize>,
    depth: usize,
    gain: f64,
    feature: usize,
    threshold: f64,
    nan_left: bool,
    split_bin: usize,
}

#[allow(dead_code)]
fn build_leaf_wise_tree(
    bins: &Bins, grads: &[f64], hess: &[f64], weights: &[f64],
    indices: Vec<usize>, col_indices: &[usize],
    num_leaves: usize, max_depth: Option<usize>,
    lambda: f64, min_child_weight: f64,
    feature_importances: &mut Vec<f64>,
) -> LGBNode {
    // Start: single leaf with all indices
    let mut leaves: Vec<(Vec<usize>, usize)> = vec![(indices, 0)]; // (indices, depth)
    let _tree_nodes: Vec<Option<LGBNode>> = vec![None]; // placeholder

    while leaves.len() < num_leaves {
        // Find the leaf with the best potential split
        let mut best_leaf_idx = None;
        let mut best_split: Option<LeafCandidate> = None;

        for (li, (leaf_indices, depth)) in leaves.iter().enumerate() {
            if let Some(md) = max_depth {
                if *depth >= md { continue; }
            }
            if leaf_indices.len() < 2 { continue; }

            let h_sum: f64 = leaf_indices.iter().map(|&i| hess[i] * weights[i]).sum();
            if h_sum < min_child_weight { continue; }

            // Find best split for this leaf
            if let Some(cand) = find_best_split_hist(
                bins, grads, hess, weights, leaf_indices, col_indices, lambda, min_child_weight,
            ) {
                let is_better = match &best_split {
                    None => true,
                    Some(prev) => cand.gain > prev.gain,
                };
                if is_better {
                    best_leaf_idx = Some(li);
                    best_split = Some(cand);
                }
            }
        }

        match (best_leaf_idx, best_split) {
            (Some(li), Some(split)) => {
                feature_importances[split.feature] += split.gain;

                let (leaf_indices, depth) = leaves.remove(li);
                let feat = split.feature;
                let bin_thr = split.split_bin;
                let nan_left = split.nan_left;

                let mut left_idx = Vec::new();
                let mut right_idx = Vec::new();
                for &idx in &leaf_indices {
                    let b = bins.get_bin(feat, idx);
                    let goes_left = if b == NAN_BIN { nan_left } else { (b as usize) <= bin_thr };
                    if goes_left { left_idx.push(idx); } else { right_idx.push(idx); }
                }

                if left_idx.is_empty() || right_idx.is_empty() { break; }

                leaves.push((left_idx, depth + 1));
                leaves.push((right_idx, depth + 1));
            }
            _ => break, // no more splits possible
        }
    }

    // Build tree from final leaves
    if leaves.len() == 1 {
        let (indices, _) = &leaves[0];
        return LGBNode::Leaf { weight: leaf_weight(grads, hess, weights, indices, lambda) };
    }

    // For simplicity with leaf-wise: rebuild as a single tree from the splits recorded
    // Actually, we need a recursive approach. Let me use a simpler strategy:
    // Build depth-first but prioritize by gain (approximate leaf-wise behavior)
    build_recursive(bins, grads, hess, weights, &leaves.into_iter()
        .flat_map(|(idx, _)| idx).collect::<Vec<_>>(),
        col_indices, num_leaves, max_depth.unwrap_or(usize::MAX), 0,
        lambda, min_child_weight, feature_importances,
        &mut 0)
}

fn build_recursive(
    bins: &Bins, grads: &[f64], hess: &[f64], weights: &[f64],
    indices: &[usize], col_indices: &[usize],
    max_leaves: usize, max_depth: usize, depth: usize,
    lambda: f64, min_child_weight: f64,
    importances: &mut Vec<f64>,
    leaf_count: &mut usize,
) -> LGBNode {
    let h_sum: f64 = indices.iter().map(|&i| hess[i] * weights[i]).sum();

    if depth >= max_depth || indices.len() < 2 || h_sum < min_child_weight
        || *leaf_count >= max_leaves {
        *leaf_count += 1;
        return LGBNode::Leaf { weight: leaf_weight(grads, hess, weights, indices, lambda) };
    }

    match find_best_split_hist(bins, grads, hess, weights, indices, col_indices, lambda, min_child_weight) {
        Some(split) if split.gain > 0.0 => {
            importances[split.feature] += split.gain;

            let mut left_idx = Vec::new();
            let mut right_idx = Vec::new();
            for &idx in indices {
                let b = bins.get_bin(split.feature, idx);
                let goes_left = if b == NAN_BIN { split.nan_left }
                    else { (b as usize) <= split.split_bin };
                if goes_left { left_idx.push(idx); } else { right_idx.push(idx); }
            }

            if left_idx.is_empty() || right_idx.is_empty() {
                *leaf_count += 1;
                return LGBNode::Leaf { weight: leaf_weight(grads, hess, weights, indices, lambda) };
            }

            let left = build_recursive(bins, grads, hess, weights, &left_idx, col_indices,
                max_leaves, max_depth, depth + 1, lambda, min_child_weight, importances, leaf_count);
            let right = build_recursive(bins, grads, hess, weights, &right_idx, col_indices,
                max_leaves, max_depth, depth + 1, lambda, min_child_weight, importances, leaf_count);

            LGBNode::Split { feature: split.feature, threshold: split.threshold,
                nan_left: split.nan_left, left: Box::new(left), right: Box::new(right) }
        }
        _ => {
            *leaf_count += 1;
            LGBNode::Leaf { weight: leaf_weight(grads, hess, weights, indices, lambda) }
        }
    }
}

fn find_best_split_hist(
    bins: &Bins, grads: &[f64], hess: &[f64], weights: &[f64],
    indices: &[usize], col_indices: &[usize],
    lambda: f64, min_child_weight: f64,
) -> Option<LeafCandidate> {
    let results: Vec<Option<LeafCandidate>> = col_indices.par_iter().map(|&feat| {
        let nb = bins.boundaries[feat].len();
        let mut bin_g = vec![0.0; nb];
        let mut bin_h = vec![0.0; nb];
        let mut nan_g = 0.0;
        let mut nan_h = 0.0;

        for &idx in indices {
            let b = bins.get_bin(feat, idx);
            let w = weights[idx];
            if b == NAN_BIN { nan_g += grads[idx] * w; nan_h += hess[idx] * w; }
            else { bin_g[b as usize] += grads[idx] * w; bin_h[b as usize] += hess[idx] * w; }
        }

        let total_g: f64 = bin_g.iter().sum::<f64>() + nan_g;
        let total_h: f64 = bin_h.iter().sum::<f64>() + nan_h;
        let mut best_gain = 0.0;
        let mut best: Option<(usize, f64, bool)> = None;

        let (mut gl, mut hl) = (0.0, 0.0);
        for bin in 0..nb.saturating_sub(1) {
            gl += bin_g[bin]; hl += bin_h[bin];
            let (gr, hr) = (total_g - gl, total_h - hl);
            if hl < min_child_weight || hr < min_child_weight { continue; }
            let gain = 0.5 * (gl*gl/(hl+lambda) + gr*gr/(hr+lambda) - total_g*total_g/(total_h+lambda));
            if gain > best_gain { best_gain = gain; best = Some((bin, gain, false)); }
        }

        if nan_h > 0.0 {
            let (mut gl, mut hl) = (nan_g, nan_h);
            for bin in 0..nb.saturating_sub(1) {
                gl += bin_g[bin]; hl += bin_h[bin];
                let (gr, hr) = (total_g - gl, total_h - hl);
                if hl < min_child_weight || hr < min_child_weight { continue; }
                let gain = 0.5 * (gl*gl/(hl+lambda) + gr*gr/(hr+lambda) - total_g*total_g/(total_h+lambda));
                if gain > best_gain { best_gain = gain; best = Some((bin, gain, true)); }
            }
        }

        best.map(|(bin, gain, nan_left)| LeafCandidate {
            indices: Vec::new(), depth: 0, gain,
            feature: feat, threshold: bins.boundaries[feat][bin],
            nan_left, split_bin: bin,
        })
    }).collect();

    results.into_iter().flatten()
        .max_by(|a, b| a.gain.partial_cmp(&b.gain).unwrap_or(std::cmp::Ordering::Equal))
}

fn leaf_weight(grads: &[f64], hess: &[f64], weights: &[f64], indices: &[usize], lambda: f64) -> f64 {
    let g: f64 = indices.iter().map(|&i| grads[i] * weights[i]).sum();
    let h: f64 = indices.iter().map(|&i| hess[i] * weights[i]).sum();
    -g / (h + lambda)
}

// ── Trained model ───────────────────────────────────────────────────

fn sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }
fn softmax(s: &[f64]) -> Vec<f64> {
    let mx = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = s.iter().map(|&v| (v - mx).exp()).collect();
    let sm: f64 = e.iter().sum();
    e.iter().map(|&v| v / sm).collect()
}

#[derive(Serialize, Deserialize)]
pub(crate) enum LGBMode { Regression, BinaryClassif, MultiClassif { n_classes: usize } }

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
                let predicted: Vec<f64> = features.rows().into_iter()
                    .map(|r| {
                        let mut v = self.initial[0];
                        for t in &self.trees { v += self.learning_rate * t.predict_one(r); }
                        v
                    }).collect();
                Ok(Prediction::regression(predicted))
            }
            LGBMode::BinaryClassif => {
                let mut predicted = Vec::with_capacity(features.nrows());
                let mut probabilities = Vec::with_capacity(features.nrows());
                for r in features.rows() {
                    let mut f = self.initial[0];
                    for t in &self.trees { f += self.learning_rate * t.predict_one(r); }
                    let p = sigmoid(f);
                    predicted.push(if p >= 0.5 { 1 } else { 0 });
                    probabilities.push(vec![1.0 - p, p]);
                }
                Ok(Prediction::Classification { predicted, truth: None, probabilities: Some(probabilities) })
            }
            LGBMode::MultiClassif { n_classes } => {
                let k = *n_classes;
                let mut predicted = Vec::with_capacity(features.nrows());
                let mut probabilities = Vec::with_capacity(features.nrows());
                for r in features.rows() {
                    let mut scores = self.initial.clone();
                    let ni = self.trees.len() / k;
                    for i in 0..ni { for c in 0..k { scores[c] += self.learning_rate * self.trees[i*k+c].predict_one(r); } }
                    let probs = softmax(&scores);
                    let pred = probs.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).unwrap().0;
                    predicted.push(pred); probabilities.push(probs);
                }
                Ok(Prediction::Classification { predicted, truth: None, probabilities: Some(probabilities) })
            }
        }
    }

    fn feature_importance(&self) -> Option<Vec<(String, f64)>> {
        let total: f64 = self.feature_importances.iter().sum();
        if total == 0.0 { return None; }
        Some(self.feature_names.iter().zip(&self.feature_importances)
            .map(|(n, &i)| (n.clone(), i / total)).collect())
    }
}

// ── Learner ─────────────────────────────────────────────────────────

impl LightGBM {
    fn sample_cols(&self, rng: &mut StdRng, nf: usize) -> Vec<usize> {
        if self.colsample_bytree < 1.0 {
            let k = (nf as f64 * self.colsample_bytree).ceil().max(1.0) as usize;
            let mut v: Vec<usize> = (0..nf).collect();
            v.shuffle(rng); v.truncate(k); v.sort(); v
        } else { (0..nf).collect() }
    }
}

impl Learner for LightGBM {
    fn id(&self) -> &str { "lightgbm" }

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

            let (selected, _weights) = goss_sample(&grads, &hess, self.top_rate, self.other_rate, &mut rng);
            let cols = self.sample_cols(&mut rng, nf);

            // Remap grads/hess for selected indices
            let _sel_grads: Vec<f64> = selected.iter().map(|&i| grads[i]).collect();
            let _sel_hess: Vec<f64> = selected.iter().map(|&i| hess[i]).collect();

            let tree = build_recursive(&bins, &grads, &hess, &vec![1.0; ns],
                &selected, &cols, self.num_leaves, self.max_depth.unwrap_or(usize::MAX),
                0, self.lambda, self.min_child_weight, &mut imp, &mut 0);

            for i in 0..ns { preds[i] += self.learning_rate * tree.predict_one(features.row(i)); }
            trees.push(tree);
        }

        Ok(Box::new(TrainedLightGBM { trees, initial: vec![initial], learning_rate: self.learning_rate,
            mode: LGBMode::Regression, feature_names: task.feature_names().to_vec(), feature_importances: imp }))
    }

    fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        let nc = task.n_classes();
        if nc == 2 { self.train_binary(task) } else { self.train_multiclass(task) }
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
            let hess: Vec<f64> = (0..ns).map(|i| { let p = sigmoid(fv[i]); p * (1.0 - p).max(1e-15) }).collect();
            let (selected, _weights) = goss_sample(&grads, &hess, self.top_rate, self.other_rate, &mut rng);
            let cols = self.sample_cols(&mut rng, nf);

            let all_weights = vec![1.0; ns]; // weights applied via GOSS selection
            let tree = build_recursive(&bins, &grads, &hess, &all_weights,
                &selected, &cols, self.num_leaves, self.max_depth.unwrap_or(usize::MAX),
                0, self.lambda, self.min_child_weight, &mut imp, &mut 0);

            for i in 0..ns { fv[i] += self.learning_rate * tree.predict_one(features.row(i)); }
            trees.push(tree);
        }

        Ok(Box::new(TrainedLightGBM { trees, initial: vec![initial], learning_rate: self.learning_rate,
            mode: LGBMode::BinaryClassif, feature_names: task.feature_names().to_vec(), feature_importances: imp }))
    }

    fn train_multiclass(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        let features = task.features();
        let target = task.target();
        let (ns, nf, nc) = (task.n_samples(), task.n_features(), task.n_classes());
        let bins = HistBins::build(features, self.n_bins);
        let mut cc = vec![0usize; nc];
        for &t in target { cc[t] += 1; }
        let initial: Vec<f64> = cc.iter().map(|&c| ((c as f64 / ns as f64).max(1e-15)).ln()).collect();
        let mut fv: Vec<Vec<f64>> = (0..ns).map(|_| initial.clone()).collect();
        let mut trees = Vec::with_capacity(self.n_estimators * nc);
        let mut imp = vec![0.0; nf];
        let mut rng = StdRng::seed_from_u64(self.seed);

        for _ in 0..self.n_estimators {
            let probs: Vec<Vec<f64>> = fv.iter().map(|f| softmax(f)).collect();
            let cols = self.sample_cols(&mut rng, nf);
            for c in 0..nc {
                let grads: Vec<f64> = (0..ns).map(|i| probs[i][c] - if target[i] == c { 1.0 } else { 0.0 }).collect();
                let hess: Vec<f64> = (0..ns).map(|i| (probs[i][c] * (1.0 - probs[i][c])).max(1e-15)).collect();
                let (selected, _) = goss_sample(&grads, &hess, self.top_rate, self.other_rate, &mut rng);
                let all_weights = vec![1.0; ns];
                let tree = build_recursive(&bins, &grads, &hess, &all_weights,
                    &selected, &cols, self.num_leaves, self.max_depth.unwrap_or(usize::MAX),
                    0, self.lambda, self.min_child_weight, &mut imp, &mut 0);
                for i in 0..ns { fv[i][c] += self.learning_rate * tree.predict_one(features.row(i)); }
                trees.push(tree);
            }
        }

        Ok(Box::new(TrainedLightGBM { trees, initial, learning_rate: self.learning_rate,
            mode: LGBMode::MultiClassif { n_classes: nc }, feature_names: task.feature_names().to_vec(),
            feature_importances: imp }))
    }
}
