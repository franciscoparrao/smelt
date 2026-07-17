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

use super::eval::{EarlyStopper, EvalSet, EvalTarget, validate_eval_classif, validate_eval_regress};
use super::histogram::{HistBins, NAN_BIN};
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
    /// When empty, the task's own `FeatureType::Categorical` columns are used.
    cat_features: Vec<usize>,
    /// Prior for target statistics smoothing.
    prior_strength: f64,
    /// Histogram bins per feature -- see `with_max_bins`.
    max_bins: usize,
    seed: u64,
    early_stopping_rounds: usize,
    /// Optional held-out set for early stopping — see `EvalSet` docs.
    eval_set: EvalSet,
}

impl Default for CatBoost {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.1,
            depth: 6,
            lambda: 1.0,
            cat_features: Vec::new(),
            prior_strength: 1.0,
            max_bins: 64,
            seed: 42,
            early_stopping_rounds: 0,
            eval_set: None,
        }
    }
}

impl CatBoost {
    /// Creates a `CatBoost` learner with default hyperparameters.
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets the number of boosting rounds (oblivious trees to fit).
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }
    /// Sets the shrinkage applied to each tree's contribution.
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }
    /// Sets the depth of each oblivious (symmetric) tree.
    pub fn with_depth(mut self, d: usize) -> Self {
        self.depth = d;
        self
    }
    /// Sets the L2 regularization strength on leaf weights.
    pub fn with_lambda(mut self, l: f64) -> Self {
        self.lambda = l;
        self
    }
    /// Sets the indices of categorical features to encode with ordered
    /// target statistics instead of treating them as numeric.
    pub fn with_cat_features(mut self, cats: Vec<usize>) -> Self {
        self.cat_features = cats;
        self
    }
    /// Sets the Bayesian prior strength used to smooth categorical target
    /// statistics (larger values shrink encodings toward the global mean).
    pub fn with_prior_strength(mut self, p: f64) -> Self {
        self.prior_strength = p;
        self
    }
    /// Sets the number of histogram bins used to discretize each feature
    /// for split search. Default is 64 -- a deliberate divergence from the
    /// official CatBoost's `border_count` default of 254, trading split
    /// resolution for speed; raise it toward 254 when fine-grained numeric
    /// thresholds matter more than training time.
    pub fn with_max_bins(mut self, b: usize) -> Self {
        self.max_bins = b;
        self
    }
    /// Sets the RNG seed controlling the target-statistics permutation order.
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

    /// Categorical columns to target-encode: the explicitly configured ones,
    /// or the task's own categorical metadata when none were configured.
    fn effective_cat_features(&self, task: &dyn Task) -> Vec<usize> {
        if self.cat_features.is_empty() {
            task.categorical_features()
        } else {
            self.cat_features.clone()
        }
    }

    /// Encode eval-set features with the final training statistics (the same
    /// map prediction uses): seen categories → their statistic, unseen → prior,
    /// NaN stays NaN.
    fn encode_eval(
        eval_features: &Array2<f64>,
        cat_encodings: &HashMap<usize, HashMap<i64, f64>>,
        prior: f64,
    ) -> Array2<f64> {
        let mut encoded = eval_features.clone();
        for (&col, encodings) in cat_encodings {
            for i in 0..encoded.nrows() {
                let raw = eval_features[[i, col]];
                if raw.is_nan() {
                    continue;
                }
                encoded[[i, col]] = encodings.get(&(raw as i64)).copied().unwrap_or(prior);
            }
        }
        encoded
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
            let raw = features[[idx, cat_col]];
            if raw.is_nan() {
                // Missing category stays NaN — the NaN-aware binning and
                // splits handle it. Casting NaN as i64 would silently merge
                // missing values into category 0.
                continue;
            }
            let cat_val = raw as i64; // discretize

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
#[derive(Clone, Serialize, Deserialize)]
pub struct ObliviousTree {
    /// One split per depth level: (feature_index, threshold).
    splits: Vec<(usize, f64)>,
    /// Per-level direction for NaN values (true = left). Learned during
    /// training like XGBoost's default direction. `#[serde(default)]` so
    /// models serialized before this field existed still load; missing
    /// entries mean left, matching the old `NaN >= t == false` routing.
    #[serde(default)]
    nan_left: Vec<bool>,
    /// Leaf weights: 2^depth values.
    leaf_weights: Vec<f64>,
    /// Per-level split gain, index-aligned with `splits`. `#[serde(default)]`
    /// so models serialized before this field existed still load; when empty,
    /// `feature_importance` falls back to counting splits (weight 1/level).
    #[serde(default)]
    gains: Vec<f64>,
}

impl ObliviousTree {
    fn predict_one(&self, row: ArrayView1<f64>) -> f64 {
        // Training assembles partitions via `new_pi = 2*old_pi + right_bit`, so
        // pi's high bit = earliest level, LSB = latest level. leaf_idx must use
        // the same bit ordering (level 0 → most-significant bit) or the weight
        // lookup reads an unrelated leaf for asymmetric paths, causing
        // catastrophic divergence on small training sets.
        //
        // `>=` (not `>`): samples whose value equals `bounds[best_bin]` are
        // routed RIGHT during training (their bin index > best_bin), so prediction
        // must route them RIGHT too.
        let mut leaf_idx = 0usize;
        let d = self.splits.len();
        for (level, &(feat, threshold)) in self.splits.iter().enumerate() {
            let v = row[feat];
            let goes_right = if v.is_nan() {
                !self.nan_left.get(level).copied().unwrap_or(true)
            } else {
                v >= threshold
            };
            if goes_right {
                leaf_idx |= 1 << (d - 1 - level);
            }
        }
        self.leaf_weights[leaf_idx]
    }
}

type CBBins = HistBins;

/// Per-partition per-feature histogram: (bin_g, bin_h, nan_g, nan_h).
/// NaN gradient/hessian mass is tracked separately so split gains account for
/// missing values (they used to be excluded from the gain but still routed
/// left, giving inconsistent statistics — audit issue M2).
///
/// Bins accumulate in `f32` rather than `f64` (item 16d): this is CatBoost's
/// one histogram hot loop (`scan_partition_hists`, below), profiled at ~45%
/// of total training time — the largest fraction of the 3 boosting engines,
/// and the only one where halving the per-bin footprint (fits twice as many
/// bins per cache line, matching the official implementations) clearly
/// justifies the numerical-drift risk. The gain formula in
/// `build_oblivious_tree` widens back to `f64` before dividing, so only the
/// per-bin sum itself loses precision, not the split comparison.
type FeatHist = (Vec<f32>, Vec<f32>, f32, f32);

fn scan_partition_hists(
    bins: &CBBins,
    grads: &[f64],
    hess: &[f64],
    indices: &[usize],
    n_features: usize,
) -> Vec<FeatHist> {
    (0..n_features)
        .into_par_iter()
        .map(|feat| {
            let nb = bins.boundaries[feat].len();
            let mut bg = vec![0.0f32; nb];
            let mut bh = vec![0.0f32; nb];
            let mut ng = 0.0f32;
            let mut nh = 0.0f32;
            for &idx in indices {
                let b = bins.get_bin(feat, idx);
                if b == NAN_BIN {
                    ng += grads[idx] as f32;
                    nh += hess[idx] as f32;
                } else {
                    bg[b as usize] += grads[idx] as f32;
                    bh[b as usize] += hess[idx] as f32;
                }
            }
            (bg, bh, ng, nh)
        })
        .collect()
}

fn build_oblivious_tree(
    bins: &CBBins,
    grads: &[f64],
    hess: &[f64],
    indices: &[usize],
    depth: usize,
    n_features: usize,
    lambda: f64,
) -> ObliviousTree {
    let mut splits = Vec::with_capacity(depth);
    let mut nan_lefts = Vec::with_capacity(depth);
    let mut gains = Vec::with_capacity(depth);
    let mut partitions: Vec<Vec<usize>> = vec![indices.to_vec()];

    // Histogram cache: cache[partition_idx][feature_idx] = FeatHist
    type PartHist = Vec<Vec<FeatHist>>;

    let mut cache: PartHist = vec![scan_partition_hists(bins, grads, hess, indices, n_features)];

    for _level in 0..depth {
        let mut best_gain = f64::NEG_INFINITY;
        let mut best_feat = 0;
        let mut best_bin = 0;
        let mut best_nan_left = true;

        // Find best split from CACHED histograms (no scanning!). For each
        // candidate bin, evaluate NaN-goes-left vs NaN-goes-right (the
        // direction is shared across partitions, like the split itself —
        // oblivious property).
        let results: Vec<(usize, usize, f64, bool)> = (0..n_features)
            .into_par_iter()
            .map(|feat| {
                let nb = bins.boundaries[feat].len();
                let mut best_local_gain = f64::NEG_INFINITY;
                let mut best_local_bin = 0;
                let mut best_local_nan_left = true;

                // Prefix sums from cached histograms, plus NaN mass and
                // NaN-inclusive totals per partition.
                let prefix: Vec<(Vec<f64>, Vec<f64>, f64, f64, f64, f64)> = cache
                    .iter()
                    .map(|part_cache| {
                        let (bg, bh, ng, nh) = &part_cache[feat];
                        // Widen f32 bin sums to f64 here so the gain formula
                        // below (division, squaring) runs at full precision —
                        // only the per-bin accumulation in scan_partition_hists
                        // is f32.
                        let (ng, nh) = (*ng as f64, *nh as f64);
                        let mut pg = vec![0.0; nb + 1];
                        let mut ph = vec![0.0; nb + 1];
                        for b in 0..nb {
                            pg[b + 1] = pg[b] + bg[b] as f64;
                            ph[b + 1] = ph[b] + bh[b] as f64;
                        }
                        let tg = pg[nb] + ng;
                        let th = ph[nb] + nh;
                        (pg, ph, tg, th, ng, nh)
                    })
                    .collect();

                for bin in 0..nb.saturating_sub(1) {
                    for nan_left in [false, true] {
                        let mut total_gain = 0.0;
                        for (pg, ph, tg, th, ng, nh) in &prefix {
                            let (mut gl, mut hl) = (pg[bin + 1], ph[bin + 1]);
                            if nan_left {
                                gl += ng;
                                hl += nh;
                            }
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
                            best_local_nan_left = nan_left;
                        }
                    }
                }
                (feat, best_local_bin, best_local_gain, best_local_nan_left)
            })
            .collect();

        for (feat, bin, gain, nan_left) in results {
            if gain > best_gain {
                best_gain = gain;
                best_feat = feat;
                best_bin = bin;
                best_nan_left = nan_left;
            }
        }

        // Because the split is shared across every current leaf (the
        // oblivious/symmetric-tree property), a level's best available gain
        // can be negative even though each individual leaf's greedy gain
        // would be checked separately in a non-oblivious tree: forcing that
        // split would make the loss worse on net. Stop growing rather than
        // accept it -- the tree simply ends up shallower than `depth`.
        if best_gain <= 0.0 {
            break;
        }

        let threshold = bins.boundaries[best_feat][best_bin];
        splits.push((best_feat, threshold));
        nan_lefts.push(best_nan_left);
        gains.push(best_gain);

        // Split all partitions + update histogram cache via subtraction
        let mut new_partitions = Vec::with_capacity(partitions.len() * 2);
        let mut new_cache: PartHist = Vec::with_capacity(partitions.len() * 2);

        for (pi, partition) in partitions.iter().enumerate() {
            let mut left = Vec::new();
            let mut right = Vec::new();
            for &idx in partition {
                let b = bins.get_bin(best_feat, idx);
                let goes_left = if b == NAN_BIN {
                    best_nan_left
                } else {
                    (b as usize) <= best_bin
                };
                if goes_left {
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

            let smaller_hists = scan_partition_hists(bins, grads, hess, smaller, n_features);

            // Subtract for larger child: parent - smaller
            let larger_hists: Vec<FeatHist> = (0..n_features)
                .map(|feat| {
                    let (pg, ph, png, pnh) = &parent_hists[feat];
                    let (sg, sh, sng, snh) = &smaller_hists[feat];
                    let bg: Vec<f32> = pg.iter().zip(sg).map(|(p, s)| p - s).collect();
                    let bh: Vec<f32> = ph.iter().zip(sh).map(|(p, s)| p - s).collect();
                    (bg, bh, png - sng, pnh - snh)
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

    // Not necessarily `n_leaves` (2^depth): a level breaks out of the loop
    // above and stops growing early if its best available gain is <= 0, so
    // `partitions.len()` is the true final leaf count.
    let mut leaf_weights = vec![0.0; partitions.len()];
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
        nan_left: nan_lefts,
        leaf_weights,
        gains,
    }
}

// ── Trained model ───────────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub(crate) enum CBMode {
    Regression,
    BinaryClassif,
    MultiClassif { n_classes: usize },
}

/// A trained CatBoost model, ready to predict.
#[derive(Clone, Serialize, Deserialize)]
pub struct TrainedCatBoost {
    pub(crate) trees: Vec<ObliviousTree>,
    pub(crate) initial: Vec<f64>,
    pub(crate) learning_rate: f64,
    pub(crate) mode: CBMode,
    pub(crate) feature_names: Vec<String>,
    pub(crate) cat_features: Vec<usize>,
    /// One target-statistic encoding map per class output. Regression and
    /// binary classification have a single output, so this has exactly one
    /// entry. Multiclass has `n_classes` entries — each computed from that
    /// class's own one-vs-rest binary indicator target (audit issue M4: a
    /// single encoding computed from the raw class *index* as if it were a
    /// continuous/ordinal target conflated unrelated nominal classes into
    /// one meaningless "average class index" statistic).
    #[serde(with = "cat_encodings_serde", default)]
    pub(crate) cat_encodings: Vec<HashMap<usize, HashMap<i64, f64>>>,
    /// Target prior used as the fallback encoding for categories never seen
    /// during training (audit issue M3: they used to pass through as raw codes
    /// into thresholds living in target-statistic space), index-aligned with
    /// `cat_encodings`. `serde(default)` so models serialized before this
    /// field existed still load (empty keeps the pre-M4 neutral fallback of
    /// 0.0 for every class; retrain to get the real priors).
    #[serde(default)]
    pub(crate) prior: Vec<f64>,
}

/// serde adapter: (de)serializes `cat_encodings` as sorted vecs of pairs
/// instead of nested JSON objects. Both map levels have integer keys
/// (`usize` column index, `i64` category code), which JSON stores as
/// strings; `SerializableModel`'s internally-tagged enum buffers the
/// payload through serde's `Content` representation, which cannot turn a
/// string key back into an integer on any deserialization path — so a
/// CatBoost model trained with `cat_features` saved fine but could never
/// be loaded (same bug class as HoeffdingTree's `feature_stats`). Models
/// without categorical features serialize an empty vec and are unaffected.
mod cat_encodings_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::HashMap;

    type Encodings = Vec<HashMap<usize, HashMap<i64, f64>>>;

    pub fn serialize<S: Serializer>(v: &Encodings, s: S) -> Result<S::Ok, S::Error> {
        let as_pairs: Vec<Vec<(usize, Vec<(i64, f64)>)>> = v
            .iter()
            .map(|by_col| {
                let mut cols: Vec<(usize, Vec<(i64, f64)>)> = by_col
                    .iter()
                    .map(|(&col, by_cat)| {
                        let mut cats: Vec<(i64, f64)> =
                            by_cat.iter().map(|(&c, &enc)| (c, enc)).collect();
                        cats.sort_by_key(|(c, _)| *c);
                        (col, cats)
                    })
                    .collect();
                cols.sort_by_key(|(col, _)| *col);
                cols
            })
            .collect();
        as_pairs.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Encodings, D::Error> {
        let as_pairs = Vec::<Vec<(usize, Vec<(i64, f64)>)>>::deserialize(d)?;
        Ok(as_pairs
            .into_iter()
            .map(|cols| {
                cols.into_iter()
                    .map(|(col, cats)| (col, cats.into_iter().collect()))
                    .collect()
            })
            .collect())
    }
}

impl TrainedCatBoost {
    /// Apply the `c`-th output's categorical encoding (using its final
    /// training statistics) to raw features. NaN stays NaN (missing category
    /// → NaN-aware split routing); unseen categories fall back to that
    /// output's training prior. `c` is always 0 for Regression/BinaryClassif
    /// (a single output); MultiClassif has one independently-fit encoding
    /// per class (see `cat_encodings`'s doc comment).
    fn encode_for_output(&self, features: &Array2<f64>, c: usize) -> Array2<f64> {
        let mut encoded = features.clone();
        let Some(encodings_by_col) = self.cat_encodings.get(c) else {
            return encoded; // pre-M4 model with no stored encodings at all
        };
        let prior = self.prior.get(c).copied().unwrap_or(0.0);
        for (&col, encodings) in encodings_by_col {
            for i in 0..features.nrows() {
                let raw = features[[i, col]];
                if raw.is_nan() {
                    continue;
                }
                encoded[[i, col]] = encodings.get(&(raw as i64)).copied().unwrap_or(prior);
            }
        }
        encoded
    }
}

impl TrainedModel for TrainedCatBoost {
    /// Gain-based importance summed over every level of every oblivious
    /// tree, normalized to sum to 1 (same convention as XGBoost/LightGBM in
    /// this crate). Models serialized before per-level gains existed load
    /// with empty `gains` and fall back to split counting (weight 1/level).
    fn feature_importance(&self) -> Option<Vec<(String, f64)>> {
        let mut importances = vec![0.0; self.feature_names.len()];
        for tree in &self.trees {
            for (level, &(feat, _)) in tree.splits.iter().enumerate() {
                importances[feat] += tree.gains.get(level).copied().unwrap_or(1.0);
            }
        }
        let total: f64 = importances.iter().sum();
        if total == 0.0 {
            return None;
        }
        Some(
            self.feature_names
                .iter()
                .zip(&importances)
                .map(|(n, &i)| (n.clone(), i / total))
                .collect(),
        )
    }

    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        crate::validate::check_n_features(features, self.feature_names.len())?;

        match &self.mode {
            CBMode::Regression => {
                let encoded = self.encode_for_output(features, 0);
                let predicted: Vec<f64> = (0..encoded.nrows())
                    .into_par_iter()
                    .map(|i| {
                        let r = encoded.row(i);
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
                let encoded = self.encode_for_output(features, 0);
                let results: Vec<(usize, Vec<f64>)> = (0..encoded.nrows())
                    .into_par_iter()
                    .map(|i| {
                        let r = encoded.row(i);
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
            CBMode::MultiClassif { n_classes } => {
                let k = *n_classes;
                let ni = self.trees.len() / k;
                // One independently-encoded feature matrix per class output
                // (audit issue M4) — each class's trees were trained on
                // categorical stats derived from that class's own
                // one-vs-rest indicator target, not a single shared
                // "average class index" encoding.
                let encoded_by_class: Vec<Array2<f64>> =
                    (0..k).map(|c| self.encode_for_output(features, c)).collect();
                let n_rows = features.nrows();
                let results: Vec<(usize, Vec<f64>)> = (0..n_rows)
                    .into_par_iter()
                    .map(|i| {
                        let mut scores = self.initial.clone();
                        for iter in 0..ni {
                            for c in 0..k {
                                let r = encoded_by_class[c].row(i);
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

    fn to_serializable(&self) -> Option<crate::serialize::SerializableModel> {
        Some(crate::serialize::SerializableModel::CatBoost(self.clone()))
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
        let eval = validate_eval_regress(&self.eval_set, nf)?;
        let mut rng = StdRng::seed_from_u64(self.seed);
        let cat_features = self.effective_cat_features(task);

        let prior = target.iter().sum::<f64>() / ns as f64;
        let encoded = ordered_target_encode(
            features,
            target,
            &cat_features,
            prior,
            self.prior_strength,
            &mut rng,
        );

        // Final encoding map: used at prediction time, and to encode the eval
        // set during training (it only depends on the data, not the model).
        let cat_encodings =
            build_final_encodings(features, target, &cat_features, prior, self.prior_strength);
        let eval_encoded = eval.map(|(ef, _)| Self::encode_eval(ef, &cat_encodings, prior));

        let initial = prior;
        let mut preds = vec![initial; ns];
        let mut eval_preds = eval.map(|(ef, _)| vec![initial; ef.nrows()]);
        let mut trees = Vec::with_capacity(self.n_estimators);
        let indices: Vec<usize> = (0..ns).collect();
        let bins = HistBins::build(&encoded, self.max_bins);
        let mut stopper = EarlyStopper::new(self.early_stopping_rounds);

        for _ in 0..self.n_estimators {
            let grads: Vec<f64> = preds.iter().zip(target).map(|(p, y)| p - y).collect();
            let hess = vec![1.0; ns];
            let tree =
                build_oblivious_tree(&bins, &grads, &hess, &indices, self.depth, nf, self.lambda);
            for i in 0..ns {
                preds[i] += self.learning_rate * tree.predict_one(encoded.row(i));
            }
            if let (Some(ep), Some(ee)) = (&mut eval_preds, &eval_encoded) {
                for i in 0..ee.nrows() {
                    ep[i] += self.learning_rate * tree.predict_one(ee.row(i));
                }
            }
            trees.push(tree);

            if stopper.is_active() {
                let loss = if let (Some(ep), Some((_, et))) = (&eval_preds, eval) {
                    ep.iter().zip(et).map(|(p, y)| (p - y).powi(2)).sum::<f64>() / ep.len() as f64
                } else {
                    preds.iter().zip(target).map(|(p, y)| (p - y).powi(2)).sum::<f64>() / ns as f64
                };
                if let Some(best_n) = stopper.update(loss, trees.len()) {
                    trees.truncate(best_n);
                    break;
                }
            }
        }

        Ok(Box::new(TrainedCatBoost {
            trees,
            initial: vec![initial],
            learning_rate: self.learning_rate,
            mode: CBMode::Regression,
            feature_names: task.feature_names().to_vec(),
            cat_features,
            cat_encodings: vec![cat_encodings],
            prior: vec![prior],
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
        let eval = validate_eval_classif(&self.eval_set, nf)?;
        let mut rng = StdRng::seed_from_u64(self.seed);
        let cat_features = self.effective_cat_features(task);

        let target_f64: Vec<f64> = target.iter().map(|&t| t as f64).collect();
        let prior = target_f64.iter().sum::<f64>() / ns as f64;
        let encoded = ordered_target_encode(
            features,
            &target_f64,
            &cat_features,
            prior,
            self.prior_strength,
            &mut rng,
        );

        let cat_encodings = build_final_encodings(
            features,
            &target_f64,
            &cat_features,
            prior,
            self.prior_strength,
        );
        let eval_encoded = eval.map(|(ef, _)| Self::encode_eval(ef, &cat_encodings, prior));

        let p_pos = target.iter().filter(|&&t| t == 1).count() as f64 / ns as f64;
        let initial = (p_pos / (1.0 - p_pos).max(1e-15)).ln();
        let mut fv = vec![initial; ns];
        let mut eval_fv = eval.map(|(ef, _)| vec![initial; ef.nrows()]);
        let mut trees = Vec::with_capacity(self.n_estimators);
        let indices: Vec<usize> = (0..ns).collect();
        let bins = HistBins::build(&encoded, self.max_bins);
        let mut stopper = EarlyStopper::new(self.early_stopping_rounds);

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
            if let (Some(efv), Some(ee)) = (&mut eval_fv, &eval_encoded) {
                for i in 0..ee.nrows() {
                    efv[i] += self.learning_rate * tree.predict_one(ee.row(i));
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
                let loss = if let (Some(efv), Some((_, et))) = (&eval_fv, eval) {
                    efv.iter().zip(et).map(|(&f, &y)| logloss(f, y)).sum::<f64>()
                        / efv.len() as f64
                } else {
                    fv.iter().zip(target).map(|(&f, &y)| logloss(f, y)).sum::<f64>() / ns as f64
                };
                if let Some(best_n) = stopper.update(loss, trees.len()) {
                    trees.truncate(best_n);
                    break;
                }
            }
        }

        Ok(Box::new(TrainedCatBoost {
            trees,
            initial: vec![initial],
            learning_rate: self.learning_rate,
            mode: CBMode::BinaryClassif,
            feature_names: task.feature_names().to_vec(),
            cat_features,
            cat_encodings: vec![cat_encodings],
            prior: vec![prior],
        }))
    }

    fn train_multiclass(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        let features = task.features();
        let target = task.target();
        let (ns, nf, nc) = (task.n_samples(), task.n_features(), task.n_classes());
        let eval = validate_eval_classif(&self.eval_set, nf)?;
        let mut rng = StdRng::seed_from_u64(self.seed);
        let cat_features = self.effective_cat_features(task);

        // One target-statistic encoding per class (audit issue M4): the raw
        // class index (0, 1, 2, ...) has no meaning as a continuous target
        // for nominal classes, so a single encoding computed from it (as
        // this used to do) produced a meaningless "average class index"
        // statistic instead of per-class target information. Each class c
        // gets its own ordered target encoding from a one-vs-rest binary
        // indicator (1{class == c}), its own final encoding map (used at
        // predict time), and its own histogram bins — matching how that
        // class's gradients/hessians are actually computed below.
        let one_vs_rest: Vec<Vec<f64>> = (0..nc)
            .map(|c| target.iter().map(|&t| if t == c { 1.0 } else { 0.0 }).collect())
            .collect();
        let priors: Vec<f64> = one_vs_rest
            .iter()
            .map(|ind| ind.iter().sum::<f64>() / ns as f64)
            .collect();
        let encoded_by_class: Vec<Array2<f64>> = (0..nc)
            .map(|c| {
                ordered_target_encode(
                    features,
                    &one_vs_rest[c],
                    &cat_features,
                    priors[c],
                    self.prior_strength,
                    &mut rng,
                )
            })
            .collect();
        let cat_encodings: Vec<HashMap<usize, HashMap<i64, f64>>> = (0..nc)
            .map(|c| {
                build_final_encodings(
                    features,
                    &one_vs_rest[c],
                    &cat_features,
                    priors[c],
                    self.prior_strength,
                )
            })
            .collect();
        let eval_encoded_by_class: Option<Vec<Array2<f64>>> = eval.map(|(ef, _)| {
            (0..nc)
                .map(|c| Self::encode_eval(ef, &cat_encodings[c], priors[c]))
                .collect()
        });

        let mut cc = vec![0usize; nc];
        for &t in target {
            cc[t] += 1;
        }
        let initial: Vec<f64> = cc
            .iter()
            .map(|&c| ((c as f64 / ns as f64).max(1e-15)).ln())
            .collect();
        let mut fv: Vec<Vec<f64>> = (0..ns).map(|_| initial.clone()).collect();
        let mut eval_fv: Option<Vec<Vec<f64>>> =
            eval.map(|(ef, _)| (0..ef.nrows()).map(|_| initial.clone()).collect());
        let mut trees = Vec::with_capacity(self.n_estimators * nc);
        let indices: Vec<usize> = (0..ns).collect();
        let bins_by_class: Vec<HistBins> =
            encoded_by_class.iter().map(|enc| HistBins::build(enc, self.max_bins)).collect();
        let mut stopper = EarlyStopper::new(self.early_stopping_rounds);

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
                    &bins_by_class[c],
                    &grads,
                    &hess,
                    &indices,
                    self.depth,
                    nf,
                    self.lambda,
                );
                for i in 0..ns {
                    fv[i][c] += self.learning_rate * tree.predict_one(encoded_by_class[c].row(i));
                }
                if let (Some(efv), Some(eebc)) = (&mut eval_fv, &eval_encoded_by_class) {
                    let ee = &eebc[c];
                    for i in 0..ee.nrows() {
                        efv[i][c] += self.learning_rate * tree.predict_one(ee.row(i));
                    }
                }
                trees.push(tree);
            }

            if stopper.is_active() {
                let eps = 1e-15;
                let loss = if let (Some(efv), Some((_, et))) = (&eval_fv, eval) {
                    let ep: Vec<Vec<f64>> = efv.iter().map(|f| softmax(f)).collect();
                    (0..et.len()).map(|i| -ep[i][et[i]].max(eps).ln()).sum::<f64>()
                        / et.len() as f64
                } else {
                    let pn: Vec<Vec<f64>> = fv.iter().map(|f| softmax(f)).collect();
                    (0..ns).map(|i| -pn[i][target[i]].max(eps).ln()).sum::<f64>() / ns as f64
                };
                if let Some(best_n) = stopper.update(loss, trees.len()) {
                    trees.truncate(best_n);
                    break;
                }
            }
        }

        Ok(Box::new(TrainedCatBoost {
            trees,
            initial,
            learning_rate: self.learning_rate,
            mode: CBMode::MultiClassif { n_classes: nc },
            feature_names: task.feature_names().to_vec(),
            cat_features,
            cat_encodings,
            prior: priors,
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
            let raw = features[[i, col]];
            if raw.is_nan() {
                continue; // missing category: no statistic to accumulate
            }
            let cat_val = raw as i64;
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// 4th-audit LOW: 64 bins were hardcoded at every HistBins::build site,
    /// with no builder and no doc of the divergence from the official 254
    /// (`border_count`). `with_max_bins` must be wired through to training
    /// at both extremes.
    #[test]
    fn with_max_bins_is_wired_through_training() {
        let n = 80;
        let features = Array2::from_shape_fn((n, 2), |(i, j)| (i as f64) + (j as f64) * 0.5);
        let target: Vec<usize> = (0..n).map(|i| usize::from(i >= n / 2)).collect();
        let task = ClassificationTask::new("bins", features.clone(), target.clone()).unwrap();

        for bins in [2, 254] {
            let model = CatBoost::new()
                .with_n_estimators(30)
                .with_depth(3)
                .with_max_bins(bins)
                .train_classif(&task)
                .unwrap();
            let Prediction::Classification { predicted, .. } = model.predict(&features).unwrap()
            else {
                panic!("expected classification prediction");
            };
            let acc = predicted
                .iter()
                .zip(&target)
                .filter(|(p, t)| p == t)
                .count() as f64
                / n as f64;
            assert!(
                acc > 0.9,
                "max_bins={bins}: separable data should fit well, got acc={acc}"
            );
        }
    }

    /// Regression test for the leaf-index bit-order bug and boundary-value routing bug.
    /// On small datasets with large-magnitude features, these two bugs caused
    /// predictions to diverge to 1e8+ while y remained in a normal range.
    /// Ref: team feedback 2026-04-20 (CatBoost SLOO RMSE=2.3e8).
    #[test]
    fn small_n_large_scale_does_not_diverge() {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(42);
        let n = 56;
        let p = 8;
        let mut features = Array2::zeros((n, p));
        for i in 0..n {
            // UTM-scale coords + mixed feature scales
            features[[i, 0]] = 6_500_000.0 + rng.random::<f64>() * 200_000.0;
            features[[i, 1]] = 300_000.0 + rng.random::<f64>() * 200_000.0;
            features[[i, 2]] = 100.0 + rng.random::<f64>() * 3900.0;
            for j in 3..p {
                features[[i, j]] = rng.random::<f64>() * 100.0;
            }
        }
        // target roughly in [0, 100]
        let target: Vec<f64> = (0..n)
            .map(|i| (0.01 * features[[i, 2]] + rng.random::<f64>() * 5.0 + 20.0).clamp(0.37, 93.62))
            .collect();

        let task = RegressionTask::new("cat_diverge", features.clone(), target.clone()).unwrap();
        let mut cb = CatBoost::new()
            .with_n_estimators(261)
            .with_depth(5)
            .with_learning_rate(0.28);
        let model = cb.train_regress(&task).unwrap();
        let pred = model.predict(&features).unwrap();
        let Prediction::Regression { predicted, .. } = pred else {
            panic!("expected regression")
        };
        let max_abs = predicted.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        let y_max = target.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        assert!(
            max_abs < 10.0 * y_max,
            "CatBoost diverged: max_pred={max_abs}, y_max={y_max}"
        );
    }

    /// Regression test for the histogram binning boundary bug (src/learner/histogram.rs):
    /// CatBoost always uses histogram splits (64 bins hardcoded), so any feature with
    /// <=64 unique values used to lose its top-2 values to the same bin. A binary
    /// feature perfectly determining the target must remain splittable.
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
        let mut model = CatBoost::new().with_n_estimators(20).with_learning_rate(0.5);
        let trained = model.train_regress(&task).unwrap();
        let pred = trained.predict(&features).unwrap();
        let Prediction::Regression { predicted, .. } = pred else {
            panic!("expected regression")
        };
        let rmse = (predicted
            .iter()
            .zip(&target)
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / n as f64)
            .sqrt();
        assert!(rmse < 1.0, "binary feature should be perfectly splittable, got RMSE={rmse}");
    }

    /// Regression test for M5 (docs/auditoria_motor_2026-07-05.md, Fase F):
    /// `build_oblivious_tree` used to always split `depth` times regardless
    /// of gain, even when the best available split at some level made the
    /// loss strictly worse (gain <= 0) -- a real possibility for oblivious
    /// trees specifically, since the split is forced across every current
    /// leaf simultaneously rather than chosen greedily per leaf. With
    /// perfectly uninformative gradients (every sample has gradient 0), no
    /// split can improve on gain 0, so the tree must stay a single leaf
    /// (`splits.len() == 0`) instead of forcing `depth` pointless splits.
    #[test]
    fn zero_gain_stops_growth_instead_of_forcing_a_split() {
        let n = 200;
        let features = Array2::<f64>::from_shape_fn((n, 3), |(i, j)| ((i * 7 + j * 3) % 11) as f64);
        let grads = vec![0.0; n];
        let hess = vec![1.0; n];
        let indices: Vec<usize> = (0..n).collect();
        let bins = HistBins::build(&features, 64);

        let tree = build_oblivious_tree(&bins, &grads, &hess, &indices, 6, 3, 1.0);
        assert_eq!(
            tree.splits.len(),
            0,
            "zero-gradient data has no beneficial split; tree should stay a single leaf"
        );
        assert_eq!(tree.leaf_weights.len(), 1);
    }

    /// Regression test for audit issue M2: NaN gradient/hessian mass used to be
    /// excluded from split gains but still routed left, so a target pattern
    /// carried *only* by missingness was invisible to split finding (zero gain
    /// everywhere → root-only trees → global-mean predictions). With NaN-aware
    /// gains and a learned per-level direction, the model must separate the
    /// NaN group from the rest.
    #[test]
    fn missingness_pattern_is_splittable() {
        let n = 400;
        let mut features = Array2::<f64>::zeros((n, 2));
        let mut target = vec![0.0; n];
        for i in 0..n {
            if i % 2 == 0 {
                features[[i, 0]] = f64::NAN;
                target[i] = 10.0;
            } else {
                features[[i, 0]] = (i % 7) as f64;
                target[i] = 0.0;
            }
            features[[i, 1]] = (i % 5) as f64; // uninformative noise column
        }
        let task = RegressionTask::new("nan_split", features.clone(), target.clone()).unwrap();
        let mut cb = CatBoost::new().with_n_estimators(30).with_depth(3).with_learning_rate(0.3);
        let model = cb.train_regress(&task).unwrap();
        let Prediction::Regression { predicted, .. } = model.predict(&features).unwrap() else {
            panic!("expected regression")
        };
        let rmse = (predicted
            .iter()
            .zip(&target)
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / n as f64)
            .sqrt();
        assert!(
            rmse < 2.0,
            "NaN-carried signal should be splittable (old behavior: RMSE=5 from \
             predicting the global mean), got RMSE={rmse}"
        );
    }

    /// Regression test for audit issue M3: unseen categories at prediction time
    /// used to pass through as raw integer codes into thresholds living in
    /// target-statistic space (~[0,1] here), silently routing them to an
    /// arbitrary side. They must now fall back to the training prior.
    #[test]
    fn unseen_category_falls_back_to_prior() {
        let n = 200;
        let mut features = Array2::<f64>::zeros((n, 1));
        let mut target = vec![0.0; n];
        for i in 0..n {
            let cat = (i % 2) as f64;
            features[[i, 0]] = cat;
            target[i] = cat * 10.0; // cat 0 → 0, cat 1 → 10; prior = 5
        }
        let task = RegressionTask::new("unseen_cat", features, target).unwrap();
        let mut cb = CatBoost::new()
            .with_n_estimators(50)
            .with_depth(3)
            .with_cat_features(vec![0]);
        let model = cb.train_regress(&task).unwrap();

        // Category 99 was never seen: its encoding must be the prior (5.0),
        // so the prediction must land strictly between the two class means,
        // not snap to either extreme.
        let unseen = Array2::from_shape_vec((1, 1), vec![99.0]).unwrap();
        let Prediction::Regression { predicted, .. } = model.predict(&unseen).unwrap() else {
            panic!("expected regression")
        };
        assert!(
            (2.0..=8.0).contains(&predicted[0]),
            "unseen category should predict near the prior (5.0), got {}",
            predicted[0]
        );
    }

    /// The ordered/final target encoders must skip NaN cells: casting NaN to
    /// i64 yields 0 in Rust, which used to silently merge missing values into
    /// category 0's statistics.
    #[test]
    fn target_encoding_skips_nan() {
        let features =
            Array2::from_shape_vec((4, 1), vec![0.0, f64::NAN, 1.0, 0.0]).unwrap();
        let target = vec![1.0, 100.0, 2.0, 3.0];

        let mut rng = StdRng::seed_from_u64(0);
        let encoded = ordered_target_encode(&features, &target, &[0], 2.0, 1.0, &mut rng);
        assert!(encoded[[1, 0]].is_nan(), "NaN cell must stay NaN after encoding");

        let finals = build_final_encodings(&features, &target, &[0], 2.0, 1.0);
        let enc0 = finals[&0][&0]; // category 0: targets 1 and 3, prior 2, strength 1
        assert!(
            (enc0 - (1.0 + 3.0 + 2.0) / 3.0).abs() < 1e-12,
            "category 0 statistic must not include the NaN row's target (100), got {enc0}"
        );
    }

    /// Regression test for M4 (docs/auditoria_motor_2026-07-05.md, Fase F):
    /// `train_multiclass` used to encode categorical features once, from the
    /// raw class *index* treated as a continuous target -- meaningless for
    /// nominal classes. This constructs a categorical feature where that
    /// collapses two genuinely distinguishable categories onto the same
    /// encoded value: category 0 is a 50/50 mix of class 0 and class 2
    /// (average index (0+2)/2 = 1.0), category 1 is 100% class 1 (average
    /// index 1.0) -- under the old shared encoding both categories land at
    /// ~1.0 and the tree has no way to split on this feature at all, even
    /// though it's perfectly informative once encoded per-class
    /// (one-vs-rest: category 1 is 100% positive for class 1 and 0% for
    /// class 0/2; category 0 is 50% positive for class 0 and class 2, 0%
    /// for class 1). With the per-class fix, class 1 must be near-perfectly
    /// separable on this feature alone.
    #[test]
    fn multiclass_categorical_uses_per_class_target_stats() {
        let n = 300;
        let mut features = Array2::<f64>::zeros((n, 2));
        let mut target = vec![0usize; n];
        for i in 0..n {
            if i < 150 {
                features[[i, 0]] = 0.0; // category 0: mixed class 0/2
                target[i] = if i % 2 == 0 { 0 } else { 2 };
            } else {
                features[[i, 0]] = 1.0; // category 1: pure class 1
                target[i] = 1;
            }
            features[[i, 1]] = ((i * 37) % 17) as f64; // uninformative noise feature
        }
        let task = ClassificationTask::new("multiclass_cat", features.clone(), target.clone())
            .unwrap();
        let mut cb = CatBoost::new()
            .with_n_estimators(50)
            .with_depth(3)
            .with_cat_features(vec![0]);
        let model = cb.train_classif(&task).unwrap();
        let pred = model.predict(&features).unwrap();
        let Prediction::Classification { predicted, .. } = pred else {
            panic!("expected classification")
        };

        // Class 1 (pure category 1) must be recovered essentially perfectly;
        // the old shared "average class index" encoding could not separate
        // it from category 0 at all on this feature.
        let class1_recall = (150..n).filter(|&i| predicted[i] == 1).count() as f64 / 150.0;
        assert!(
            class1_recall > 0.95,
            "class 1 (pure category) should be near-perfectly recovered via per-class \
             target stats, got recall={class1_recall}"
        );
    }

    /// CatBoost picks up categorical columns declared on the Task when none
    /// were configured explicitly (item 14: FeatureType metadata).
    #[test]
    fn task_categorical_metadata_is_used_automatically() {
        let features = Array2::from_shape_vec((4, 2), vec![0.0, 1.0, 1.0, 2.0, 0.0, 3.0, 1.0, 4.0])
            .unwrap();
        let task = RegressionTask::new("meta", features, vec![1.0, 2.0, 3.0, 4.0])
            .unwrap()
            .with_categorical_features(&[0])
            .unwrap();

        let cb = CatBoost::new();
        assert_eq!(cb.effective_cat_features(&task), vec![0]);

        // Explicit configuration wins over task metadata.
        let cb = CatBoost::new().with_cat_features(vec![1]);
        assert_eq!(cb.effective_cat_features(&task), vec![1]);
    }

    /// Eval-set early stopping must truncate to the best round on held-out
    /// loss instead of running all estimators on an overfittable config.
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
            CatBoost::new()
                .with_n_estimators(500)
                .with_depth(6)
                .with_learning_rate(0.5)
                .with_lambda(0.0001)
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
}
