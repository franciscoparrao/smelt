//! CatBoost-inspired symmetric gradient boosting with ordered target statistics.
//!
//! This implements two key innovations from the CatBoost paper:
//! - **Ordered Target Statistics** for categorical features (avoids target leakage
//!   via permutation-based encoding with Bayesian prior)
//! - **Oblivious (symmetric) trees**: same split at each depth level
//! - Newton boosting with L2 regularization
//!
//! **Not implemented**: full O(n²) ordered boosting (per-sample model approximations),
//! GPU training, distributed computation, interaction/monotone constraints.
//! This is a CatBoost-inspired symmetric GBM, not a feature-complete reimplementation.
//!
//! Reference: Prokhorenkova, L. et al. (2018). CatBoost: unbiased boosting
//! with categorical features. NeurIPS.

use super::histogram::HistBins;
use crate::Result;
use crate::learner::{Learner, TrainedModel};
use crate::prediction::Prediction;
use crate::task::{ClassificationTask, RegressionTask, Task};
use ndarray::{Array2, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// CatBoost-inspired symmetric GBM with ordered target statistics.
///
/// Implements oblivious trees and permutation-based target encoding from
/// Prokhorenkova et al. (2018). Does not include full ordered boosting
/// (O(n²) per-sample models), GPU support, or distributed training.
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
/// let task = ClassificationTask::new("cat", features, target).unwrap();
///
/// let mut cb = CatBoost::new()
///     .with_n_estimators(50)
///     .with_depth(4);
/// let model = cb.train_classif(&task).unwrap();
/// ```
pub struct CatBoost {
    n_estimators: usize,
    learning_rate: f64,
    depth: usize, // oblivious tree depth
    lambda: f64,  // L2 regularization
    /// Indices of categorical features (will use target statistics encoding).
    cat_features: Vec<usize>,
    /// Prior for target statistics smoothing.
    prior_strength: f64,
    seed: u64,
}

impl Default for CatBoost {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.1,
            depth: 6,
            lambda: 3.0,
            cat_features: Vec::new(),
            prior_strength: 1.0,
            seed: 42,
        }
    }
}

impl CatBoost {
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
    pub fn with_depth(mut self, d: usize) -> Self {
        self.depth = d;
        self
    }
    pub fn with_lambda(mut self, l: f64) -> Self {
        self.lambda = l;
        self
    }
    pub fn with_cat_features(mut self, cats: Vec<usize>) -> Self {
        self.cat_features = cats;
        self
    }
    pub fn with_prior_strength(mut self, p: f64) -> Self {
        self.prior_strength = p;
        self
    }
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }
}

// ── Ordered Target Statistics ───────────────────────────────────────

/// Encode categorical features using ordered target statistics.
/// For each sample i in the random permutation, the encoding uses only
/// targets from samples appearing before i in the permutation.
fn ordered_target_encode(
    features: &Array2<f64>,
    target: &[f64],
    cat_features: &[usize],
    prior: f64,
    prior_strength: f64,
    rng: &mut StdRng,
) -> Array2<f64> {
    let n = features.nrows();
    let mut encoded = features.clone();

    if cat_features.is_empty() {
        return encoded;
    }

    // Random permutation
    let mut perm: Vec<usize> = (0..n).collect();
    perm.shuffle(rng);

    for &cat_col in cat_features {
        // For each sample in permutation order, compute target statistic
        // using only previous samples with the same category value
        let mut sum_by_cat: HashMap<i64, f64> = HashMap::new();
        let mut count_by_cat: HashMap<i64, usize> = HashMap::new();

        for &idx in &perm {
            let cat_val = features[[idx, cat_col]] as i64; // discretize

            let count = *count_by_cat.get(&cat_val).unwrap_or(&0);
            let sum = *sum_by_cat.get(&cat_val).unwrap_or(&0.0);

            // Ordered target statistic (Eq from paper):
            // x_encoded = (sum_prev + prior_strength * prior) / (count_prev + prior_strength)
            let encoding = (sum + prior_strength * prior) / (count as f64 + prior_strength);
            encoded[[idx, cat_col]] = encoding;

            // Update running statistics (after encoding this sample)
            *sum_by_cat.entry(cat_val).or_insert(0.0) += target[idx];
            *count_by_cat.entry(cat_val).or_insert(0) += 1;
        }
    }

    encoded
}

// ── Oblivious Tree ──────────────────────────────────────────────────

/// Oblivious (symmetric) tree: same split at each depth level.
/// All nodes at the same depth use the same (feature, threshold).
/// Total leaves = 2^depth.
#[derive(Serialize, Deserialize)]
pub struct ObliviousTree {
    /// One split per depth level: (feature_index, threshold).
    splits: Vec<(usize, f64)>,
    /// Leaf weights: 2^depth values.
    leaf_weights: Vec<f64>,
}

impl ObliviousTree {
    fn predict_one(&self, row: ArrayView1<f64>) -> f64 {
        let mut leaf_idx = 0usize;
        for (level, &(feat, threshold)) in self.splits.iter().enumerate() {
            if row[feat] > threshold {
                leaf_idx |= 1 << level;
            }
        }
        self.leaf_weights[leaf_idx]
    }
}

type CBBins = HistBins;

fn build_oblivious_tree(
    bins: &CBBins,
    grads: &[f64],
    hess: &[f64],
    indices: &[usize],
    depth: usize,
    n_features: usize,
    lambda: f64,
) -> ObliviousTree {
    let n_leaves = 1 << depth;
    let mut splits = Vec::with_capacity(depth);
    let mut partitions: Vec<Vec<usize>> = vec![indices.to_vec()];

    // Histogram cache: per partition, per feature → (bin_g, bin_h)
    // cache[partition_idx][feature_idx] = (Vec<f64>, Vec<f64>)
    type PartHist = Vec<Vec<(Vec<f64>, Vec<f64>)>>;

    // Build initial cache: scan root partition for all features
    let initial_cache: Vec<(Vec<f64>, Vec<f64>)> = (0..n_features)
        .into_par_iter()
        .map(|feat| {
            let nb = bins.boundaries[feat].len();
            let mut bg = vec![0.0; nb];
            let mut bh = vec![0.0; nb];
            for &idx in indices {
                let b = bins.get_bin(feat, idx) as usize;
                bg[b] += grads[idx];
                bh[b] += hess[idx];
            }
            (bg, bh)
        })
        .collect();
    let mut cache: PartHist = vec![initial_cache];

    for _level in 0..depth {
        let mut best_gain = f64::NEG_INFINITY;
        let mut best_feat = 0;
        let mut best_bin = 0;

        // Find best split from CACHED histograms (no scanning!)
        let results: Vec<(usize, usize, f64)> = (0..n_features)
            .into_par_iter()
            .map(|feat| {
                let nb = bins.boundaries[feat].len();
                let mut best_local_gain = f64::NEG_INFINITY;
                let mut best_local_bin = 0;

                // Prefix sums from cached histograms
                let prefix: Vec<(Vec<f64>, Vec<f64>, f64, f64)> = cache
                    .iter()
                    .map(|part_cache| {
                        let (bg, bh) = &part_cache[feat];
                        let mut pg = vec![0.0; nb + 1];
                        let mut ph = vec![0.0; nb + 1];
                        for b in 0..nb {
                            pg[b + 1] = pg[b] + bg[b];
                            ph[b + 1] = ph[b] + bh[b];
                        }
                        let tg = pg[nb];
                        let th = ph[nb];
                        (pg, ph, tg, th)
                    })
                    .collect();

                for bin in 0..nb.saturating_sub(1) {
                    let mut total_gain = 0.0;
                    for (pg, ph, tg, th) in &prefix {
                        let gl = pg[bin + 1];
                        let hl = ph[bin + 1];
                        let gr = tg - gl;
                        let hr = th - hl;
                        if hl > 0.0 && hr > 0.0 {
                            total_gain += gl * gl / (hl + lambda) + gr * gr / (hr + lambda)
                                - tg * tg / (th + lambda);
                        }
                    }
                    if total_gain > best_local_gain {
                        best_local_gain = total_gain;
                        best_local_bin = bin;
                    }
                }
                (feat, best_local_bin, best_local_gain)
            })
            .collect();

        for (feat, bin, gain) in results {
            if gain > best_gain {
                best_gain = gain;
                best_feat = feat;
                best_bin = bin;
            }
        }

        let threshold = bins.boundaries[best_feat][best_bin];
        splits.push((best_feat, threshold));

        // Split all partitions + update histogram cache via subtraction
        let mut new_partitions = Vec::with_capacity(partitions.len() * 2);
        let mut new_cache: PartHist = Vec::with_capacity(partitions.len() * 2);

        for (pi, partition) in partitions.iter().enumerate() {
            let mut left = Vec::new();
            let mut right = Vec::new();
            for &idx in partition {
                if (bins.get_bin(best_feat, idx) as usize) <= best_bin {
                    left.push(idx);
                } else {
                    right.push(idx);
                }
            }

            // Histogram subtraction: scan smaller child, subtract for larger
            let parent_hists = &cache[pi];
            let (smaller, larger_is_right) = if left.len() <= right.len() {
                (&left, true)
            } else {
                (&right, false)
            };

            // Scan smaller child for all features (parallel)
            let smaller_hists: Vec<(Vec<f64>, Vec<f64>)> = (0..n_features)
                .into_par_iter()
                .map(|feat| {
                    let nb = bins.boundaries[feat].len();
                    let mut bg = vec![0.0; nb];
                    let mut bh = vec![0.0; nb];
                    for &idx in smaller.iter() {
                        let b = bins.get_bin(feat, idx) as usize;
                        bg[b] += grads[idx];
                        bh[b] += hess[idx];
                    }
                    (bg, bh)
                })
                .collect();

            // Subtract for larger child: parent - smaller
            let larger_hists: Vec<(Vec<f64>, Vec<f64>)> = (0..n_features)
                .map(|feat| {
                    let (pg, ph) = &parent_hists[feat];
                    let (sg, sh) = &smaller_hists[feat];
                    let bg: Vec<f64> = pg.iter().zip(sg).map(|(p, s)| p - s).collect();
                    let bh: Vec<f64> = ph.iter().zip(sh).map(|(p, s)| p - s).collect();
                    (bg, bh)
                })
                .collect();

            if larger_is_right {
                new_cache.push(smaller_hists); // left = smaller
                new_cache.push(larger_hists); // right = larger (subtracted)
            } else {
                new_cache.push(larger_hists); // left = larger (subtracted)
                new_cache.push(smaller_hists); // right = smaller
            }

            new_partitions.push(left);
            new_partitions.push(right);
        }
        partitions = new_partitions;
        cache = new_cache;
    }

    let mut leaf_weights = vec![0.0; n_leaves];
    for (leaf_idx, partition) in partitions.iter().enumerate() {
        let g: f64 = partition.iter().map(|&i| grads[i]).sum();
        let h: f64 = partition.iter().map(|&i| hess[i]).sum();
        leaf_weights[leaf_idx] = if h + lambda > 0.0 {
            -g / (h + lambda)
        } else {
            0.0
        };
    }

    ObliviousTree {
        splits,
        leaf_weights,
    }
}

// ── Trained model ───────────────────────────────────────────────────

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
fn softmax(s: &[f64]) -> Vec<f64> {
    let mx = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = s.iter().map(|&v| (v - mx).exp()).collect();
    let sm: f64 = e.iter().sum();
    e.iter().map(|&v| v / sm).collect()
}

#[derive(Serialize, Deserialize)]
pub(crate) enum CBMode {
    Regression,
    BinaryClassif,
    MultiClassif { n_classes: usize },
}

#[derive(Serialize, Deserialize)]
pub struct TrainedCatBoost {
    pub(crate) trees: Vec<ObliviousTree>,
    pub(crate) initial: Vec<f64>,
    pub(crate) learning_rate: f64,
    pub(crate) mode: CBMode,
    pub(crate) feature_names: Vec<String>,
    pub(crate) cat_features: Vec<usize>,
    pub(crate) cat_encodings: HashMap<usize, HashMap<i64, f64>>,
}

impl TrainedModel for TrainedCatBoost {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        crate::validate::check_n_features(features, self.feature_names.len())?;

        // Apply categorical encoding (using final training statistics)
        let mut encoded = features.clone();
        for (&col, encodings) in &self.cat_encodings {
            for i in 0..features.nrows() {
                let val = features[[i, col]] as i64;
                if let Some(&enc) = encodings.get(&val) {
                    encoded[[i, col]] = enc;
                }
            }
        }

        match &self.mode {
            CBMode::Regression => {
                let predicted: Vec<f64> = encoded
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
            CBMode::BinaryClassif => {
                let mut predicted = Vec::with_capacity(features.nrows());
                let mut probabilities = Vec::with_capacity(features.nrows());
                for r in encoded.rows() {
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
            CBMode::MultiClassif { n_classes } => {
                let k = *n_classes;
                let mut predicted = Vec::with_capacity(features.nrows());
                let mut probabilities = Vec::with_capacity(features.nrows());
                for r in encoded.rows() {
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
}

// ── Learner ─────────────────────────────────────────────────────────

impl Learner for CatBoost {
    fn id(&self) -> &str {
        "catboost"
    }

    fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
        let features = task.features();
        let target = task.target();
        let (ns, nf) = (task.n_samples(), task.n_features());
        let mut rng = StdRng::seed_from_u64(self.seed);

        let prior = target.iter().sum::<f64>() / ns as f64;
        let encoded = ordered_target_encode(
            features,
            target,
            &self.cat_features,
            prior,
            self.prior_strength,
            &mut rng,
        );

        let initial = prior;
        let mut preds = vec![initial; ns];
        let mut trees = Vec::with_capacity(self.n_estimators);
        let indices: Vec<usize> = (0..ns).collect();
        let bins = HistBins::build(&encoded, 64);

        for _ in 0..self.n_estimators {
            let grads: Vec<f64> = preds.iter().zip(target).map(|(p, y)| p - y).collect();
            let hess = vec![1.0; ns];
            let tree =
                build_oblivious_tree(&bins, &grads, &hess, &indices, self.depth, nf, self.lambda);
            for i in 0..ns {
                preds[i] += self.learning_rate * tree.predict_one(encoded.row(i));
            }
            trees.push(tree);
        }

        // Build final encoding map for prediction
        let cat_encodings = build_final_encodings(
            features,
            target,
            &self.cat_features,
            prior,
            self.prior_strength,
        );

        Ok(Box::new(TrainedCatBoost {
            trees,
            initial: vec![initial],
            learning_rate: self.learning_rate,
            mode: CBMode::Regression,
            feature_names: task.feature_names().to_vec(),
            cat_features: self.cat_features.clone(),
            cat_encodings,
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

impl CatBoost {
    fn train_binary(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        let features = task.features();
        let target = task.target();
        let (ns, nf) = (task.n_samples(), task.n_features());
        let mut rng = StdRng::seed_from_u64(self.seed);

        let target_f64: Vec<f64> = target.iter().map(|&t| t as f64).collect();
        let prior = target_f64.iter().sum::<f64>() / ns as f64;
        let encoded = ordered_target_encode(
            features,
            &target_f64,
            &self.cat_features,
            prior,
            self.prior_strength,
            &mut rng,
        );

        let p_pos = target.iter().filter(|&&t| t == 1).count() as f64 / ns as f64;
        let initial = (p_pos / (1.0 - p_pos).max(1e-15)).ln();
        let mut fv = vec![initial; ns];
        let mut trees = Vec::with_capacity(self.n_estimators);
        let indices: Vec<usize> = (0..ns).collect();
        let bins = HistBins::build(&encoded, 64);

        for _ in 0..self.n_estimators {
            let grads: Vec<f64> = (0..ns).map(|i| sigmoid(fv[i]) - target[i] as f64).collect();
            let hess: Vec<f64> = (0..ns)
                .map(|i| {
                    let p = sigmoid(fv[i]);
                    p * (1.0 - p).max(1e-15)
                })
                .collect();
            let tree =
                build_oblivious_tree(&bins, &grads, &hess, &indices, self.depth, nf, self.lambda);
            for i in 0..ns {
                fv[i] += self.learning_rate * tree.predict_one(encoded.row(i));
            }
            trees.push(tree);
        }

        let cat_encodings = build_final_encodings(
            features,
            &target_f64,
            &self.cat_features,
            prior,
            self.prior_strength,
        );

        Ok(Box::new(TrainedCatBoost {
            trees,
            initial: vec![initial],
            learning_rate: self.learning_rate,
            mode: CBMode::BinaryClassif,
            feature_names: task.feature_names().to_vec(),
            cat_features: self.cat_features.clone(),
            cat_encodings,
        }))
    }

    fn train_multiclass(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        let features = task.features();
        let target = task.target();
        let (ns, nf, nc) = (task.n_samples(), task.n_features(), task.n_classes());
        let mut rng = StdRng::seed_from_u64(self.seed);

        let target_f64: Vec<f64> = target.iter().map(|&t| t as f64).collect();
        let prior = target_f64.iter().sum::<f64>() / ns as f64;
        let encoded = ordered_target_encode(
            features,
            &target_f64,
            &self.cat_features,
            prior,
            self.prior_strength,
            &mut rng,
        );

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
        let indices: Vec<usize> = (0..ns).collect();
        let bins = HistBins::build(&encoded, 64);

        for _ in 0..self.n_estimators {
            let probs: Vec<Vec<f64>> = fv.iter().map(|f| softmax(f)).collect();
            for c in 0..nc {
                let grads: Vec<f64> = (0..ns)
                    .map(|i| probs[i][c] - if target[i] == c { 1.0 } else { 0.0 })
                    .collect();
                let hess: Vec<f64> = (0..ns)
                    .map(|i| (probs[i][c] * (1.0 - probs[i][c])).max(1e-15))
                    .collect();
                let tree = build_oblivious_tree(
                    &bins,
                    &grads,
                    &hess,
                    &indices,
                    self.depth,
                    nf,
                    self.lambda,
                );
                for i in 0..ns {
                    fv[i][c] += self.learning_rate * tree.predict_one(encoded.row(i));
                }
                trees.push(tree);
            }
        }

        let cat_encodings = build_final_encodings(
            features,
            &target_f64,
            &self.cat_features,
            prior,
            self.prior_strength,
        );

        Ok(Box::new(TrainedCatBoost {
            trees,
            initial,
            learning_rate: self.learning_rate,
            mode: CBMode::MultiClassif { n_classes: nc },
            feature_names: task.feature_names().to_vec(),
            cat_features: self.cat_features.clone(),
            cat_encodings,
        }))
    }
}

/// Build final target encoding map for prediction-time categorical handling.
fn build_final_encodings(
    features: &Array2<f64>,
    target: &[f64],
    cat_features: &[usize],
    prior: f64,
    prior_strength: f64,
) -> HashMap<usize, HashMap<i64, f64>> {
    let mut result = HashMap::new();
    for &col in cat_features {
        let mut sum_by_cat: HashMap<i64, f64> = HashMap::new();
        let mut count_by_cat: HashMap<i64, usize> = HashMap::new();
        for (i, &t) in target.iter().enumerate() {
            let cat_val = features[[i, col]] as i64;
            *sum_by_cat.entry(cat_val).or_insert(0.0) += t;
            *count_by_cat.entry(cat_val).or_insert(0) += 1;
        }
        let mut encodings = HashMap::new();
        for (&cat_val, &sum) in &sum_by_cat {
            let count = count_by_cat[&cat_val];
            let enc = (sum + prior_strength * prior) / (count as f64 + prior_strength);
            encodings.insert(cat_val, enc);
        }
        result.insert(col, encodings);
    }
    result
}
