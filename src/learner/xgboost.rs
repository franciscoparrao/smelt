//! XGBoost: eXtreme Gradient Boosting with histogram-based splitting.
//!
//! Core algorithm with Newton boosting, L1/L2 regularization, gamma min gain,
//! histogram-based + auto exact greedy, NaN handling, row/col subsampling,
//! parallel split finding, early stopping, zero-copy prediction, in-place partitioning.

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
use super::hist_pool::HistPool;

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
}

impl Default for XGBoost {
    fn default() -> Self {
        Self {
            n_estimators: 100, learning_rate: 0.3, max_depth: 6,
            lambda: 1.0, alpha: 0.0, gamma: 0.0,
            subsample: 1.0, colsample_bytree: 1.0, min_child_weight: 1.0,
            n_bins: 256, early_stopping_rounds: 0, seed: 42,
        }
    }
}

impl XGBoost {
    pub fn new() -> Self { Self::default() }
    pub fn with_n_estimators(mut self, n: usize) -> Self { self.n_estimators = n; self }
    pub fn with_learning_rate(mut self, lr: f64) -> Self { self.learning_rate = lr; self }
    pub fn with_max_depth(mut self, d: usize) -> Self { self.max_depth = d; self }
    pub fn with_lambda(mut self, l: f64) -> Self { self.lambda = l; self }
    pub fn with_alpha(mut self, a: f64) -> Self { self.alpha = a; self }
    pub fn with_gamma(mut self, g: f64) -> Self { self.gamma = g; self }
    pub fn with_subsample(mut self, s: f64) -> Self { self.subsample = s; self }
    pub fn with_colsample_bytree(mut self, c: f64) -> Self { self.colsample_bytree = c; self }
    pub fn with_min_child_weight(mut self, w: f64) -> Self { self.min_child_weight = w; self }
    pub fn with_early_stopping_rounds(mut self, n: usize) -> Self { self.early_stopping_rounds = n; self }
    pub fn with_seed(mut self, s: u64) -> Self { self.seed = s; self }
}

// ── Histogram binning (NaN-aware, column-major, u8 packed) ──────────
//
type FeatureBins = HistBins;

// ── XGBoost tree node ───────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
pub enum XGBNode {
    Leaf { weight: f64 },
    Split { feature: usize, threshold: f64, nan_goes_left: bool,
            left: Box<XGBNode>, right: Box<XGBNode> },
}

impl XGBNode {
    #[inline]
    fn predict_one(&self, sample: ArrayView1<f64>) -> f64 {
        match self {
            XGBNode::Leaf { weight } => *weight,
            XGBNode::Split { feature, threshold, nan_goes_left, left, right } => {
                let val = sample[*feature];
                if val.is_nan() {
                    if *nan_goes_left { left.predict_one(sample) } else { right.predict_one(sample) }
                } else if val < *threshold {
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
}

impl<'a> XGBTreeBuilder<'a> {
    #[inline]
    fn leaf_weight_gh(&self, g: f64, h: f64) -> f64 {
        let gt = if g > self.alpha { g - self.alpha }
            else if g < -self.alpha { g + self.alpha }
            else { 0.0 };
        -gt / (h + self.lambda)
    }

    #[inline]
    fn split_gain(&self, gl: f64, hl: f64, gr: f64, hr: f64) -> f64 {
        0.5 * (gl * gl / (hl + self.lambda) + gr * gr / (hr + self.lambda)
            - (gl + gr) * (gl + gr) / (hl + hr + self.lambda)) - self.gamma
    }

    fn build(&mut self, indices: &mut Vec<usize>, start: usize, end: usize, depth: usize) -> XGBNode {
        if self.use_exact {
            return self.build_exact(indices, start, end, depth);
        }
        self.build_hist_sub(indices, start, end, depth, false)
    }

    /// Build with histogram subtraction.
    /// `hist_ready`: pool[depth] already populated from subtraction (skip scan).
    fn build_hist_sub(&mut self, indices: &mut Vec<usize>, start: usize, end: usize,
                      depth: usize, hist_ready: bool) -> XGBNode {
        let n = end - start;
        let h_sum: f64 = indices[start..end].iter().map(|&i| self.hess[i]).sum();
        if depth >= self.max_depth || n <= 1 || h_sum < self.min_child_weight {
            let g_sum: f64 = indices[start..end].iter().map(|&i| self.grads[i]).sum();
            return XGBNode::Leaf { weight: self.leaf_weight_gh(g_sum, h_sum) };
        }

        let best = if hist_ready {
            // Subtracted histogram already in pool — just find best split
            let r = self.pool.find_best(depth, &self.col_indices, self.bins,
                self.min_child_weight, self.lambda, self.alpha, self.gamma);
            r.map(|(feat, thr, gain, nl, sb)| BestSplit {
                feature: feat, threshold: thr, gain, nan_goes_left: nl, split_bin: sb,
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
                return XGBNode::Leaf { weight: self.leaf_weight_gh(g_sum, h_sum) };
            }
        };
        let (feat, threshold, gain, nan_goes_left, split_bin) =
            (best.feature, best.threshold, best.gain, best.nan_goes_left, best.split_bin);

        self.feature_importances[feat] += gain;

        // Partition
        let (mut left_end, mut i) = (start, start);
        while i < end {
            let b = self.bins.get_bin(feat, indices[i]);
            let goes_left = if b == NAN_BIN { nan_goes_left } else { (b as usize) <= split_bin };
            if goes_left { indices.swap(left_end, i); left_end += 1; }
            i += 1;
        }

        if left_end == start || left_end == end {
            let g_sum: f64 = indices[start..end].iter().map(|&i| self.grads[i]).sum();
            return XGBNode::Leaf { weight: self.leaf_weight_gh(g_sum, h_sum) };
        }

        if depth + 1 >= self.max_depth {
            // Children are leaves — skip histogram work
            let lg: f64 = indices[start..left_end].iter().map(|&i| self.grads[i]).sum();
            let lh: f64 = indices[start..left_end].iter().map(|&i| self.hess[i]).sum();
            let rg: f64 = indices[left_end..end].iter().map(|&i| self.grads[i]).sum();
            let rh: f64 = indices[left_end..end].iter().map(|&i| self.hess[i]).sum();
            let left = XGBNode::Leaf { weight: self.leaf_weight_gh(lg, lh) };
            let right = XGBNode::Leaf { weight: self.leaf_weight_gh(rg, rh) };
            return XGBNode::Split { feature: feat, threshold, nan_goes_left,
                left: Box::new(left), right: Box::new(right) };
        }

        let left_count = left_end - start;
        let right_count = end - left_end;

        // Subtraction: scan+save SMALLER, process, subtract → LARGER (skip scan)
        if left_count <= right_count {
            let left = self.build_hist_sub(indices, start, left_end, depth + 1, false);
            self.pool.subtract_in_place(depth, depth + 1);
            let right = self.build_hist_sub(indices, left_end, end, depth + 1, true);
            XGBNode::Split { feature: feat, threshold, nan_goes_left,
                left: Box::new(left), right: Box::new(right) }
        } else {
            let right = self.build_hist_sub(indices, left_end, end, depth + 1, false);
            self.pool.subtract_in_place(depth, depth + 1);
            let left = self.build_hist_sub(indices, start, left_end, depth + 1, true);
            XGBNode::Split { feature: feat, threshold, nan_goes_left,
                left: Box::new(left), right: Box::new(right) }
        }
    }

    /// Exact greedy build (unchanged).
    fn build_exact(&mut self, indices: &mut Vec<usize>, start: usize, end: usize, depth: usize) -> XGBNode {
        let n = end - start;
        let h_sum: f64 = indices[start..end].iter().map(|&i| self.hess[i]).sum();
        if depth >= self.max_depth || n <= 1 || h_sum < self.min_child_weight {
            let g_sum: f64 = indices[start..end].iter().map(|&i| self.grads[i]).sum();
            return XGBNode::Leaf { weight: self.leaf_weight_gh(g_sum, h_sum) };
        }
        let best = match self.find_best_exact(&indices[start..end]) {
            Some(b) if b.gain > 0.0 => b,
            _ => {
                let g_sum: f64 = indices[start..end].iter().map(|&i| self.grads[i]).sum();
                return XGBNode::Leaf { weight: self.leaf_weight_gh(g_sum, h_sum) };
            }
        };
        self.feature_importances[best.feature] += best.gain;
        let feat = best.feature;
        let (mut left_end, mut i) = (start, start);
        let (thr, nan_left) = (best.threshold, best.nan_goes_left);
        while i < end {
            let v = self.features[[indices[i], feat]];
            let goes_left = if v.is_nan() { nan_left } else { v < thr };
            if goes_left { indices.swap(left_end, i); left_end += 1; }
            i += 1;
        }
        if left_end == start || left_end == end {
            let g_sum: f64 = indices[start..end].iter().map(|&i| self.grads[i]).sum();
            return XGBNode::Leaf { weight: self.leaf_weight_gh(g_sum, h_sum) };
        }
        let left = self.build_exact(indices, start, left_end, depth + 1);
        let right = self.build_exact(indices, left_end, end, depth + 1);
        XGBNode::Split { feature: feat, threshold: best.threshold,
            nan_goes_left: best.nan_goes_left,
            left: Box::new(left), right: Box::new(right) }
    }

    /// Like find_best_histogram but also returns per-feature histogram data for the pool.
    fn find_best_histogram_saving(&self, node_indices: &[usize])
        -> (Option<BestSplit>, Vec<(Vec<f64>, Vec<f64>, f64, f64)>)
    {
        let results: Vec<(Option<BestSplit>, (Vec<f64>, Vec<f64>, f64, f64))> =
            self.col_indices.par_iter().map(|&feat| {
            let nb = self.bins.n_bins(feat);
            let mut bin_g = vec![0.0; nb];
            let mut bin_h = vec![0.0; nb];
            let mut nan_g = 0.0;
            let mut nan_h = 0.0;

            for &idx in node_indices {
                let b = self.bins.get_bin(feat, idx);
                if b == NAN_BIN { nan_g += self.grads[idx]; nan_h += self.hess[idx]; }
                else { bin_g[b as usize] += self.grads[idx]; bin_h[b as usize] += self.hess[idx]; }
            }

            let total_g: f64 = bin_g.iter().sum::<f64>() + nan_g;
            let total_h: f64 = bin_h.iter().sum::<f64>() + nan_h;
            let mut best_gain = 0.0;
            let mut best: Option<(usize, f64, bool)> = None;

            let (mut gl, mut hl) = (0.0, 0.0);
            for bin in 0..nb.saturating_sub(1) {
                gl += bin_g[bin]; hl += bin_h[bin];
                let (gr, hr) = (total_g - gl, total_h - hl);
                if hl < self.min_child_weight || hr < self.min_child_weight { continue; }
                let gain = self.split_gain(gl, hl, gr, hr);
                if gain > best_gain { best_gain = gain; best = Some((bin, gain, false)); }
            }
            if nan_h > 0.0 {
                let (mut gl, mut hl) = (nan_g, nan_h);
                for bin in 0..nb.saturating_sub(1) {
                    gl += bin_g[bin]; hl += bin_h[bin];
                    let (gr, hr) = (total_g - gl, total_h - hl);
                    if hl < self.min_child_weight || hr < self.min_child_weight { continue; }
                    let gain = self.split_gain(gl, hl, gr, hr);
                    if gain > best_gain { best_gain = gain; best = Some((bin, gain, true)); }
                }
            }

            let split = best.map(|(bin, gain, nan_left)| BestSplit {
                feature: feat, threshold: self.bins.bin_threshold(feat, bin),
                gain, nan_goes_left: nan_left, split_bin: bin,
            });
            (split, (bin_g, bin_h, nan_g, nan_h))
        }).collect();

        let mut hists = Vec::with_capacity(results.len());
        let mut best_split: Option<BestSplit> = None;
        for (split, hist) in results {
            hists.push(hist);
            if let Some(s) = split {
                if best_split.as_ref().map_or(true, |b| s.gain > b.gain) {
                    best_split = Some(s);
                }
            }
        }
        (best_split, hists)
    }

    /// Histogram split finding — parallel per feature, flat Vec<f64> for cache locality.
    #[allow(dead_code)]
    fn find_best_histogram(&self, node_indices: &[usize]) -> Option<BestSplit> {
        let results: Vec<Option<BestSplit>> = self.col_indices.par_iter().map(|&feat| {
            let nb = self.bins.n_bins(feat);
            let mut bin_g = vec![0.0; nb];
            let mut bin_h = vec![0.0; nb];
            let mut nan_g = 0.0;
            let mut nan_h = 0.0;

            for &idx in node_indices {
                let b = self.bins.get_bin(feat, idx);
                if b == NAN_BIN { nan_g += self.grads[idx]; nan_h += self.hess[idx]; }
                else { bin_g[b as usize] += self.grads[idx]; bin_h[b as usize] += self.hess[idx]; }
            }

            let total_g: f64 = bin_g.iter().sum::<f64>() + nan_g;
            let total_h: f64 = bin_h.iter().sum::<f64>() + nan_h;
            let mut best_gain = 0.0;
            let mut best: Option<(usize, f64, bool)> = None;

            // NaN right
            let (mut gl, mut hl) = (0.0, 0.0);
            for bin in 0..nb.saturating_sub(1) {
                gl += bin_g[bin]; hl += bin_h[bin];
                let (gr, hr) = (total_g - gl, total_h - hl);
                if hl < self.min_child_weight || hr < self.min_child_weight { continue; }
                let gain = self.split_gain(gl, hl, gr, hr);
                if gain > best_gain { best_gain = gain; best = Some((bin, gain, false)); }
            }
            // NaN left
            if nan_h > 0.0 {
                let (mut gl, mut hl) = (nan_g, nan_h);
                for bin in 0..nb.saturating_sub(1) {
                    gl += bin_g[bin]; hl += bin_h[bin];
                    let (gr, hr) = (total_g - gl, total_h - hl);
                    if hl < self.min_child_weight || hr < self.min_child_weight { continue; }
                    let gain = self.split_gain(gl, hl, gr, hr);
                    if gain > best_gain { best_gain = gain; best = Some((bin, gain, true)); }
                }
            }

            best.map(|(bin, gain, nan_left)| BestSplit {
                feature: feat, threshold: self.bins.bin_threshold(feat, bin),
                gain, nan_goes_left: nan_left, split_bin: bin,
            })
        }).collect();

        results.into_iter().flatten()
            .max_by(|a, b| a.gain.partial_cmp(&b.gain).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Exact greedy split — parallel per feature.
    fn find_best_exact(&self, node_indices: &[usize]) -> Option<BestSplit> {
        let features = self.features;
        let results: Vec<Option<BestSplit>> = self.col_indices.par_iter().map(|&feat| {
            let mut sorted: Vec<usize> = node_indices.iter()
                .filter(|&&i| !features[[i, feat]].is_nan()).copied().collect();
            sorted.sort_by(|&a, &b| features[[a, feat]].partial_cmp(&features[[b, feat]])
                .unwrap_or(std::cmp::Ordering::Equal));

            let nan_g: f64 = node_indices.iter().filter(|&&i| features[[i, feat]].is_nan())
                .map(|&i| self.grads[i]).sum();
            let nan_h: f64 = node_indices.iter().filter(|&&i| features[[i, feat]].is_nan())
                .map(|&i| self.hess[i]).sum();

            if sorted.len() <= 1 { return None; }
            let total_g: f64 = node_indices.iter().map(|&i| self.grads[i]).sum();
            let total_h: f64 = node_indices.iter().map(|&i| self.hess[i]).sum();

            let mut best_gain = 0.0;
            let mut best: Option<(f64, f64, bool)> = None;
            let (mut gl, mut hl) = (0.0, 0.0);

            for i in 0..(sorted.len() - 1) {
                gl += self.grads[sorted[i]]; hl += self.hess[sorted[i]];
                if (features[[sorted[i+1], feat]] - features[[sorted[i], feat]]).abs() < f64::EPSILON { continue; }
                let (gr, hr) = (total_g - gl, total_h - hl);
                if hl >= self.min_child_weight && hr >= self.min_child_weight {
                    let gain = self.split_gain(gl, hl, gr, hr);
                    if gain > best_gain {
                        best_gain = gain;
                        let t = (features[[sorted[i], feat]] + features[[sorted[i+1], feat]]) / 2.0;
                        best = Some((t, gain, false));
                    }
                }
                if nan_h > 0.0 {
                    let (gln, hln) = (gl + nan_g, hl + nan_h);
                    let (grn, hrn) = (total_g - gln, total_h - hln);
                    if hln >= self.min_child_weight && hrn >= self.min_child_weight {
                        let gain = self.split_gain(gln, hln, grn, hrn);
                        if gain > best_gain {
                            best_gain = gain;
                            let t = (features[[sorted[i], feat]] + features[[sorted[i+1], feat]]) / 2.0;
                            best = Some((t, gain, true));
                        }
                    }
                }
            }

            best.map(|(threshold, gain, nan_left)| BestSplit {
                feature: feat, threshold, gain, nan_goes_left: nan_left, split_bin: 0,
            })
        }).collect();

        results.into_iter().flatten()
            .max_by(|a, b| a.gain.partial_cmp(&b.gain).unwrap_or(std::cmp::Ordering::Equal))
    }
}

// ── Trained model ───────────────────────────────────────────────────

fn sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }

fn softmax(scores: &[f64]) -> Vec<f64> {
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|&s| (s - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

#[derive(Serialize, Deserialize)]
pub(crate) enum XGBMode { Regression, BinaryClassif, MultiClassif { n_classes: usize } }

#[derive(Serialize, Deserialize)]
pub struct TrainedXGBoost {
    pub(crate) trees: Vec<XGBNode>,
    pub(crate) initial: Vec<f64>,
    pub(crate) learning_rate: f64,
    pub(crate) mode: XGBMode,
    pub(crate) feature_names: Vec<String>,
    pub(crate) feature_importances: Vec<f64>,
}

impl TrainedModel for TrainedXGBoost {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        crate::validate::check_n_features(features, self.feature_names.len())?;
        match &self.mode {
            XGBMode::Regression => {
                let predicted: Vec<f64> = features.rows().into_iter()
                    .map(|row| {
                        let mut v = self.initial[0];
                        for t in &self.trees { v += self.learning_rate * t.predict_one(row); }
                        v
                    }).collect();
                Ok(Prediction::regression(predicted))
            }
            XGBMode::BinaryClassif => {
                let mut predicted = Vec::with_capacity(features.nrows());
                let mut probabilities = Vec::with_capacity(features.nrows());
                for row in features.rows() {
                    let mut f = self.initial[0];
                    for t in &self.trees { f += self.learning_rate * t.predict_one(row); }
                    let p = sigmoid(f);
                    predicted.push(if p >= 0.5 { 1 } else { 0 });
                    probabilities.push(vec![1.0 - p, p]);
                }
                Ok(Prediction::Classification { predicted, truth: None, probabilities: Some(probabilities) })
            }
            XGBMode::MultiClassif { n_classes } => {
                let k = *n_classes;
                let mut predicted = Vec::with_capacity(features.nrows());
                let mut probabilities = Vec::with_capacity(features.nrows());
                for row in features.rows() {
                    let mut scores = self.initial.clone();
                    let n_iters = self.trees.len() / k;
                    for iter in 0..n_iters {
                        for c in 0..k { scores[c] += self.learning_rate * self.trees[iter * k + c].predict_one(row); }
                    }
                    let probs = softmax(&scores);
                    let pred = probs.iter().enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).unwrap().0;
                    predicted.push(pred);
                    probabilities.push(probs);
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

impl XGBoost {
    fn sample(&self, rng: &mut StdRng, n_samples: usize, n_features: usize) -> (Vec<usize>, Vec<usize>) {
        let rows = if self.subsample < 1.0 {
            let k = (n_samples as f64 * self.subsample).ceil() as usize;
            let mut v: Vec<usize> = (0..n_samples).collect();
            v.shuffle(rng); v.truncate(k); v
        } else { (0..n_samples).collect() };
        let cols = if self.colsample_bytree < 1.0 {
            let k = (n_features as f64 * self.colsample_bytree).ceil().max(1.0) as usize;
            let mut v: Vec<usize> = (0..n_features).collect();
            v.shuffle(rng); v.truncate(k); v.sort(); v
        } else { (0..n_features).collect() };
        (rows, cols)
    }

    fn build_one_tree(&self, features: &Array2<f64>, bins: &FeatureBins,
                      grads: &[f64], hess: &[f64],
                      indices: &mut Vec<usize>, col_indices: Vec<usize>,
                      n_features: usize) -> (XGBNode, Vec<f64>) {
        let n = indices.len();
        let use_exact = n <= self.n_bins;
        let max_bins = if use_exact { 1 } else {
            bins.boundaries.iter().map(|b| b.len() + 1).max().unwrap_or(256)
        };
        let n_col = col_indices.len();
        let mut builder = XGBTreeBuilder {
            features, bins, grads, hess,
            lambda: self.lambda, alpha: self.alpha, gamma: self.gamma,
            max_depth: self.max_depth, min_child_weight: self.min_child_weight,
            col_indices, use_exact,
            feature_importances: vec![0.0; n_features],
            pool: HistPool::new(self.max_depth, n_col, max_bins),
        };
        let tree = builder.build(indices, 0, n, 0);
        (tree, builder.feature_importances)
    }
}

impl Learner for XGBoost {
    fn id(&self) -> &str { "xgboost" }

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
        let (mut best_loss, mut no_improve, mut best_n) = (f64::INFINITY, 0usize, 0usize);

        for round in 0..self.n_estimators {
            let grads: Vec<f64> = preds.iter().zip(target).map(|(p, y)| p - y).collect();
            let hess = vec![1.0; ns];
            let (mut idx, cols) = self.sample(&mut rng, ns, nf);
            let (tree, fi) = self.build_one_tree(features, &bins, &grads, &hess, &mut idx, cols, nf);
            for i in 0..ns { preds[i] += self.learning_rate * tree.predict_one(features.row(i)); }
            for (j, v) in fi.iter().enumerate() { imp[j] += v; }
            trees.push(tree);

            if self.early_stopping_rounds > 0 {
                let loss = preds.iter().zip(target).map(|(p, y)| (p - y).powi(2)).sum::<f64>() / ns as f64;
                if loss < best_loss - 1e-10 { best_loss = loss; best_n = round + 1; no_improve = 0; }
                else { no_improve += 1; if no_improve >= self.early_stopping_rounds { trees.truncate(best_n); break; } }
            }
        }
        Ok(Box::new(TrainedXGBoost { trees, initial: vec![initial], learning_rate: self.learning_rate,
            mode: XGBMode::Regression, feature_names: task.feature_names().to_vec(), feature_importances: imp }))
    }

    fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        if task.n_classes() == 2 { self.train_binary(task) } else { self.train_multiclass(task) }
    }
}

impl XGBoost {
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
        let (mut best_loss, mut no_improve, mut best_n) = (f64::INFINITY, 0usize, 0usize);

        for round in 0..self.n_estimators {
            let grads: Vec<f64> = (0..ns).map(|i| sigmoid(fv[i]) - target[i] as f64).collect();
            let hess: Vec<f64> = (0..ns).map(|i| { let p = sigmoid(fv[i]); p * (1.0 - p).max(1e-15) }).collect();
            let (mut idx, cols) = self.sample(&mut rng, ns, nf);
            let (tree, fi) = self.build_one_tree(features, &bins, &grads, &hess, &mut idx, cols, nf);
            for i in 0..ns { fv[i] += self.learning_rate * tree.predict_one(features.row(i)); }
            for (j, v) in fi.iter().enumerate() { imp[j] += v; }
            trees.push(tree);

            if self.early_stopping_rounds > 0 {
                let eps = 1e-15;
                let loss: f64 = (0..ns).map(|i| {
                    let p = sigmoid(fv[i]).max(eps).min(1.0 - eps);
                    let y = target[i] as f64;
                    -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
                }).sum::<f64>() / ns as f64;
                if loss < best_loss - 1e-10 { best_loss = loss; best_n = round + 1; no_improve = 0; }
                else { no_improve += 1; if no_improve >= self.early_stopping_rounds { trees.truncate(best_n); break; } }
            }
        }
        Ok(Box::new(TrainedXGBoost { trees, initial: vec![initial], learning_rate: self.learning_rate,
            mode: XGBMode::BinaryClassif, feature_names: task.feature_names().to_vec(), feature_importances: imp }))
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
        let (mut best_loss, mut no_improve, mut best_n) = (f64::INFINITY, 0usize, 0usize);

        for round in 0..self.n_estimators {
            let probs: Vec<Vec<f64>> = fv.iter().map(|f| softmax(f)).collect();
            let (idx_base, cols) = self.sample(&mut rng, ns, nf);
            for c in 0..nc {
                let grads: Vec<f64> = (0..ns).map(|i| probs[i][c] - if target[i] == c { 1.0 } else { 0.0 }).collect();
                let hess: Vec<f64> = (0..ns).map(|i| (probs[i][c] * (1.0 - probs[i][c])).max(1e-15)).collect();
                let mut idx = idx_base.clone();
                let (tree, fi) = self.build_one_tree(features, &bins, &grads, &hess, &mut idx, cols.clone(), nf);
                for i in 0..ns { fv[i][c] += self.learning_rate * tree.predict_one(features.row(i)); }
                for (j, v) in fi.iter().enumerate() { imp[j] += v; }
                trees.push(tree);
            }
            if self.early_stopping_rounds > 0 {
                let eps = 1e-15;
                let pn: Vec<Vec<f64>> = fv.iter().map(|f| softmax(f)).collect();
                let loss: f64 = (0..ns).map(|i| -pn[i][target[i]].max(eps).ln()).sum::<f64>() / ns as f64;
                if loss < best_loss - 1e-10 { best_loss = loss; best_n = (round + 1) * nc; no_improve = 0; }
                else { no_improve += 1; if no_improve >= self.early_stopping_rounds { trees.truncate(best_n); break; } }
            }
        }
        Ok(Box::new(TrainedXGBoost { trees, initial, learning_rate: self.learning_rate,
            mode: XGBMode::MultiClassif { n_classes: nc }, feature_names: task.feature_names().to_vec(),
            feature_importances: imp }))
    }
}
