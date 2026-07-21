//! Hoeffding Tree (VFDT): streaming/online decision tree.
//!
//! Learns incrementally from a stream of samples. Uses the Hoeffding bound
//! to decide when to split: guarantees with probability 1-δ that the chosen
//! split is within ε of the best possible split after seeing n samples.
//!
//! Split condition: G_best - G_second >= sqrt(R² · ln(1/δ) / (2n))
//!
//! Reference: Domingos, P. & Hulten, G. (2000). Mining high-speed data streams.
//! KDD, 71-80.

use crate::Result;
use crate::learner::{Learner, LearnerProperties, TrainedModel};
use crate::prediction::Prediction;
use crate::task::{ClassificationTask, Task};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Hoeffding Tree for online/streaming classification.
///
/// Unlike batch learners, HoeffdingTree learns one sample at a time via
/// `partial_fit()`. It can also be used as a standard Learner via `train_classif()`.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use ndarray::array;
///
/// let mut ht = HoeffdingTree::new()
///     .with_grace_period(10)
///     .with_delta(1e-5);
///
/// // Online learning: feed samples one at a time
/// let features = array![[0.0, 0.0], [0.1, 0.1], [1.0, 1.0], [1.1, 0.9]];
/// let labels = vec![0, 0, 1, 1];
///
/// for i in 0..features.nrows() {
///     ht.partial_fit(&features.row(i).to_vec(), labels[i], 2);
/// }
///
/// // Or batch: standard Learner interface
/// let task = ClassificationTask::new("ht", features, labels).unwrap();
/// let mut ht2 = HoeffdingTree::new();
/// let model = ht2.train_classif(&task).unwrap();
/// ```
#[derive(Clone, Serialize, Deserialize)]
pub struct HoeffdingTree {
    /// Confidence parameter: lower = more conservative splits.
    delta: f64,
    /// Minimum samples before considering a split.
    grace_period: usize,
    /// Maximum tree depth (None = unlimited).
    max_depth: Option<usize>,
    /// Internal tree root.
    root: Option<HNode>,
    n_classes: usize,
    n_features: usize,
    /// Restricts split candidates to these feature indices when set (audit
    /// issue N6: used by [`AdaptiveRandomForest`] to give each member tree
    /// its own random feature subspace, the way `RandomForest`/`ExtraTrees`
    /// already do via `MaxFeatures` -- a plain `HoeffdingTree` considers
    /// every feature, matching VFDT's original (non-ensemble) design.
    /// `serde(default)` so models serialized before this field existed
    /// still load (`None` keeps the old "every feature" behavior).
    #[serde(default)]
    feature_subset: Option<Vec<usize>>,
}

impl Default for HoeffdingTree {
    fn default() -> Self {
        Self {
            delta: 1e-7,
            grace_period: 200,
            max_depth: None,
            root: None,
            n_classes: 0,
            n_features: 0,
            feature_subset: None,
        }
    }
}

impl HoeffdingTree {
    /// Creates a Hoeffding tree with delta 1e-7, grace period 200, and unlimited depth.
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets the confidence parameter delta (lower = more conservative splits).
    pub fn with_delta(mut self, d: f64) -> Self {
        self.delta = d;
        self
    }
    /// Sets the minimum number of samples seen at a leaf before a split is considered.
    pub fn with_grace_period(mut self, g: usize) -> Self {
        self.grace_period = g;
        self
    }
    /// Sets the maximum tree depth.
    pub fn with_max_depth(mut self, d: usize) -> Self {
        self.max_depth = Some(d);
        self
    }

    /// Restricts split candidates to `subset` (audit issue N6): lets an
    /// ensemble like [`AdaptiveRandomForest`] give this tree its own random
    /// feature subspace, instead of every tree considering every feature.
    pub fn with_feature_subset(mut self, subset: Vec<usize>) -> Self {
        self.feature_subset = Some(subset);
        self
    }

    /// Online update: train on a single sample.
    pub fn partial_fit(&mut self, features: &[f64], label: usize, n_classes: usize) {
        self.n_features = features.len();
        self.n_classes = self.n_classes.max(n_classes);

        if self.root.is_none() {
            self.root = Some(HNode::new_leaf(self.n_classes, 0));
        }

        let delta = self.delta;
        let grace = self.grace_period;
        let max_depth = self.max_depth;
        let n_classes_local = self.n_classes;
        let feature_subset = self.feature_subset.as_deref();

        if let Some(root) = &mut self.root {
            Self::update_node(
                root,
                features,
                label,
                delta,
                grace,
                max_depth,
                n_classes_local,
                feature_subset,
            );
        }
    }

    /// Predicts from the tree's current (possibly still-streaming) state,
    /// without requiring [`Learner::train_classif`]. Returns `None` before
    /// the first `partial_fit` call. This is what lets an ensemble like
    /// `AdaptiveRandomForest` get live per-sample votes from trees that are
    /// still being updated.
    pub fn predict_one(&self, features: &[f64]) -> Option<(usize, Vec<f64>)> {
        self.root.as_ref().map(|r| r.predict_class(features))
    }

    fn update_node(
        node: &mut HNode,
        features: &[f64],
        label: usize,
        delta: f64,
        grace: usize,
        max_depth: Option<usize>,
        n_classes: usize,
        feature_subset: Option<&[usize]>,
    ) {
        match node {
            HNode::Leaf {
                counts,
                n_seen,
                depth,
                feature_stats,
            } => {
                // Update class counts
                if label < counts.len() {
                    counts[label] += 1;
                }
                *n_seen += 1;

                // Update per-feature statistics -- restricted to
                // `feature_subset` when set (audit issue N6), so
                // `find_best_split` below (which only ever sees features
                // present in `feature_stats`) only considers this tree's
                // own random feature subspace.
                for (j, &val) in features.iter().enumerate() {
                    if feature_subset.is_some_and(|s| !s.contains(&j)) {
                        continue;
                    }
                    let stats = feature_stats
                        .entry(j)
                        .or_insert_with(|| FeatureStats::new(n_classes));
                    stats.update(val, label);
                }

                // Check if we should split
                if *n_seen >= grace && *n_seen % grace == 0 {
                    if max_depth.is_some_and(|d| *depth >= d) {
                        return;
                    }

                    let (best_feat, best_gain, second_gain) =
                        find_best_split(feature_stats, counts, *n_seen, n_classes);

                    // Hoeffding bound. R is the range of the information-gain
                    // random variable being bounded: max entropy for
                    // n_classes outcomes. `entropy`/`entropy_weighted` below
                    // use natural log (nats), so R must too (audit issue N4:
                    // this used `log2(n_classes)` -- bits -- while the gain
                    // itself is in nats, making epsilon ~1/ln(2) ~= 1.44x too
                    // large and needlessly delaying splits).
                    let r = (n_classes as f64).ln();
                    let epsilon = (r * r * (1.0 / delta).ln() / (2.0 * *n_seen as f64)).sqrt();

                    // Tie-breaking (Domingos & Hulten 2000): once epsilon is
                    // small enough that two close-but-distinct candidates may
                    // never separate, force a split on the current best
                    // rather than wait forever. This must still require
                    // best_gain > 0 (audit issue N5): without that guard, a
                    // leaf that has simply seen enough samples for epsilon to
                    // fall below the tie threshold will force a split even
                    // when every feature has zero information gain (pure
                    // noise), growing the tree without bound purely from
                    // sample count once n_seen is large enough (~80k/leaf
                    // with this crate's default delta).
                    if best_gain > 0.0 && (best_gain - second_gain > epsilon || epsilon < 0.01) {
                        // Split!
                        let threshold = feature_stats
                            .get(&best_feat)
                            .map(|s| s.best_threshold(n_classes))
                            .unwrap_or(0.0);

                        let left = HNode::new_leaf(n_classes, *depth + 1);
                        let right = HNode::new_leaf(n_classes, *depth + 1);

                        *node = HNode::Split {
                            feature: best_feat,
                            threshold,
                            left: Box::new(left),
                            right: Box::new(right),
                        };

                        // Route current sample
                        Self::update_node(
                            node,
                            features,
                            label,
                            delta,
                            grace,
                            max_depth,
                            n_classes,
                            feature_subset,
                        );
                    }
                }
            }
            HNode::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                if features[*feature] <= *threshold {
                    Self::update_node(
                        left,
                        features,
                        label,
                        delta,
                        grace,
                        max_depth,
                        n_classes,
                        feature_subset,
                    );
                } else {
                    Self::update_node(
                        right,
                        features,
                        label,
                        delta,
                        grace,
                        max_depth,
                        n_classes,
                        feature_subset,
                    );
                }
            }
        }
    }
}

// ── Hoeffding Tree Node ─────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
enum HNode {
    Leaf {
        counts: Vec<usize>, // class counts
        n_seen: usize,
        depth: usize,
        #[serde(with = "stats_map_serde")]
        feature_stats: HashMap<usize, FeatureStats>,
    },
    Split {
        feature: usize,
        threshold: f64,
        left: Box<HNode>,
        right: Box<HNode>,
    },
}

/// serde adapter: (de)serializes `feature_stats` as a sorted vec of pairs
/// instead of a JSON object. JSON object keys are strings, and
/// `SerializableModel`'s internally-tagged enum makes serde buffer the
/// payload through its `Content` representation, which cannot convert a
/// string key back into a `usize` on ANY deserialization path — so every
/// saved HoeffdingTree/AdaptiveRandomForest file was unloadable
/// (`invalid type: string "0", expected usize`). Pairs sidestep string
/// keys entirely; sorting keeps the on-disk output deterministic.
mod stats_map_serde {
    use super::FeatureStats;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::HashMap;

    pub fn serialize<S: Serializer>(
        m: &HashMap<usize, FeatureStats>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        let mut pairs: Vec<(&usize, &FeatureStats)> = m.iter().collect();
        pairs.sort_by_key(|(k, _)| **k);
        pairs.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<HashMap<usize, FeatureStats>, D::Error> {
        Ok(Vec::<(usize, FeatureStats)>::deserialize(d)?
            .into_iter()
            .collect())
    }
}

impl HNode {
    fn new_leaf(n_classes: usize, depth: usize) -> Self {
        HNode::Leaf {
            counts: vec![0; n_classes],
            n_seen: 0,
            depth,
            feature_stats: HashMap::new(),
        }
    }

    fn predict_class(&self, features: &[f64]) -> (usize, Vec<f64>) {
        match self {
            HNode::Leaf { counts, n_seen, .. } => {
                let total = *n_seen as f64;
                let probs: Vec<f64> = if total > 0.0 {
                    counts.iter().map(|&c| c as f64 / total).collect()
                } else {
                    vec![1.0 / counts.len() as f64; counts.len()]
                };
                let pred = counts
                    .iter()
                    .enumerate()
                    .max_by_key(|&(_, &c)| c)
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                (pred, probs)
            }
            HNode::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                if features[*feature] <= *threshold {
                    left.predict_class(features)
                } else {
                    right.predict_class(features)
                }
            }
        }
    }
}

// ── Feature statistics for split finding ────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
struct FeatureStats {
    /// For each class, store (sum, sum_sq, count) of feature values.
    class_stats: Vec<(f64, f64, usize)>, // (sum, sum_sq, count) per class
    /// Histogram bins for split threshold selection.
    min_val: f64,
    max_val: f64,
}

impl FeatureStats {
    fn new(n_classes: usize) -> Self {
        Self {
            class_stats: vec![(0.0, 0.0, 0); n_classes],
            min_val: f64::INFINITY,
            max_val: f64::NEG_INFINITY,
        }
    }

    fn update(&mut self, val: f64, label: usize) {
        if label < self.class_stats.len() {
            self.class_stats[label].0 += val;
            self.class_stats[label].1 += val * val;
            self.class_stats[label].2 += 1;
        }
        self.min_val = self.min_val.min(val);
        self.max_val = self.max_val.max(val);
    }

    fn best_threshold(&self, _n_classes: usize) -> f64 {
        // Use the midpoint weighted by class means
        let mut total_sum = 0.0;
        let mut total_count = 0;
        for &(sum, _, count) in &self.class_stats {
            total_sum += sum;
            total_count += count;
        }
        if total_count > 0 {
            total_sum / total_count as f64
        } else {
            (self.min_val + self.max_val) / 2.0
        }
    }
}

fn entropy(counts: &[usize], total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let t = total as f64;
    counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / t;
            -p * p.ln()
        })
        .sum()
}

/// Entropy from possibly-fractional (Gaussian-estimated) class weights.
fn entropy_weighted(weights: &[f64], total: f64) -> f64 {
    if total <= 0.0 {
        return 0.0;
    }
    weights
        .iter()
        .filter(|&&w| w > 1e-12)
        .map(|&w| {
            let p = w / total;
            -p * p.ln()
        })
        .sum()
}

/// Standard normal CDF via the Abramowitz & Stegun 7.1.26 `erf`
/// approximation (accurate to ~1.5e-7). There's no `f64::erf` in stable
/// Rust and no numerics crate in this workspace, so this is hand-rolled
/// rather than adding a dependency for one function -- same "hand-roll
/// small numerics" precedent as `CsrMatrix` in `src/sparse.rs`.
fn normal_cdf(x: f64, mean: f64, std: f64) -> f64 {
    if std <= 1e-9 {
        // Degenerate (near-zero variance): treat as a step function at the mean.
        return if x < mean { 0.0 } else { 1.0 };
    }
    let z = (x - mean) / (std * std::f64::consts::SQRT_2);
    let sign = if z < 0.0 { -1.0 } else { 1.0 };
    let z = z.abs();
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let t = 1.0 / (1.0 + p * z);
    let erf = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-z * z).exp();
    0.5 * (1.0 + sign * erf)
}

fn find_best_split(
    feature_stats: &HashMap<usize, FeatureStats>,
    parent_counts: &[usize],
    n_total: usize,
    n_classes: usize,
) -> (usize, f64, f64) {
    let parent_ent = entropy(parent_counts, n_total);
    let mut gains: Vec<(usize, f64)> = Vec::new();

    for (&feat, stats) in feature_stats {
        let threshold = stats.best_threshold(n_classes);

        // Estimate left/right counts from each class's running Gaussian
        // (mean, variance from `sum`/`sum_sq`/`count`) via the normal CDF at
        // `threshold`, rather than comparing a single per-class mean point
        // against the threshold: the mean-point comparison assigns an
        // entire class to one side or the other regardless of how much its
        // distribution actually overlaps the other class's, which makes a
        // pure-noise feature look just as "perfectly separating" as a
        // genuinely predictive one (both classes' means almost never land
        // exactly on the same side of a threshold, so gain reads as ~perfect
        // for every feature) -- starving the Hoeffding-bound gain-difference
        // test, which can then never clear its confidence bar.
        let mut left_counts = vec![0.0f64; n_classes];
        let mut right_counts = vec![0.0f64; n_classes];

        for (c, &(sum, sum_sq, count)) in stats.class_stats.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let count_f = count as f64;
            let mean = sum / count_f;
            let var = (sum_sq / count_f - mean * mean).max(0.0);
            let std = var.sqrt();
            let left_frac = normal_cdf(threshold, mean, std);
            left_counts[c] = count_f * left_frac;
            right_counts[c] = count_f * (1.0 - left_frac);
        }

        let n_left: f64 = left_counts.iter().sum();
        let n_right: f64 = right_counts.iter().sum();

        if n_left < 1e-6 || n_right < 1e-6 {
            continue;
        }

        let left_ent = entropy_weighted(&left_counts, n_left);
        let right_ent = entropy_weighted(&right_counts, n_right);

        let gain = parent_ent
            - (n_left / n_total as f64) * left_ent
            - (n_right / n_total as f64) * right_ent;

        gains.push((feat, gain));
    }

    // Tie-break equal gains by feature index: `gains` is built by iterating
    // a HashMap, so without this the winner among tied features follows the
    // map's per-process iteration order and the grown tree can differ
    // between runs on identical input.
    gains.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });

    let best_feat = gains.first().map(|&(f, _)| f).unwrap_or(0);
    let best_gain = gains.first().map(|&(_, g)| g).unwrap_or(0.0);
    let second_gain = gains.get(1).map(|&(_, g)| g).unwrap_or(0.0);

    (best_feat, best_gain, second_gain)
}

// ── Learner implementation ──────────────────────────────────────────

impl Learner for HoeffdingTree {
    fn id(&self) -> &str {
        "hoeffding_tree"
    }

    fn properties(&self) -> LearnerProperties {
        LearnerProperties::classifier()
            .with_proba()
            .with_serializable()
    }

    fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        crate::validate::check_no_weights(task.weights(), "HoeffdingTree")?;
        crate::validate::check_no_nan(task.features())?;
        let features = task.features();
        let target = task.target();
        let n_classes = task.n_classes();

        // Reset tree
        self.root = None;
        self.n_classes = n_classes;
        self.n_features = task.n_features();

        // Feed all samples as a stream
        for i in 0..task.n_samples() {
            let row: Vec<f64> = features.row(i).to_vec();
            self.partial_fit(&row, target[i], n_classes);
        }

        // Extract trained tree for prediction
        let root = self
            .root
            .take()
            .unwrap_or_else(|| HNode::new_leaf(n_classes, 0));

        Ok(Box::new(TrainedHoeffdingTree {
            root,
            n_features: self.n_features,
            n_classes,
        }))
    }
}

/// A trained (batch-constructed) Hoeffding tree snapshot.
#[derive(Clone, Serialize, Deserialize)]
pub struct TrainedHoeffdingTree {
    root: HNode,
    n_features: usize,
    #[allow(dead_code)]
    n_classes: usize,
}

impl TrainedModel for TrainedHoeffdingTree {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        crate::validate::check_n_features(features, self.n_features)?;

        let mut predicted = Vec::with_capacity(features.nrows());
        let mut probabilities = Vec::with_capacity(features.nrows());

        for row in features.rows() {
            let row_vec: Vec<f64> = row.to_vec();
            let (pred, probs) = self.root.predict_class(&row_vec);
            predicted.push(pred);
            probabilities.push(probs);
        }

        Ok(Prediction::Classification {
            predicted,
            truth: None,
            probabilities: Some(probabilities),
        })
    }

    fn to_serializable(&self) -> Option<crate::serialize::SerializableModel> {
        Some(crate::serialize::SerializableModel::HoeffdingTree(
            self.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// 4th-audit LOW: with tied gains, `find_best_split` used to inherit the
    /// winner from the HashMap's per-process iteration order. Two features
    /// with identical statistics must deterministically resolve to the lower
    /// feature index, whatever order the map yields them in.
    #[test]
    fn find_best_split_breaks_gain_ties_by_feature_index() {
        // Class 0 clustered near 0.0, class 1 near 1.0 -- identical stats
        // for both features, so their gains tie exactly.
        let mut stats = FeatureStats::new(2);
        for v in [0.0, 0.1, 0.2] {
            stats.update(v, 0);
        }
        for v in [1.0, 1.1, 1.2] {
            stats.update(v, 1);
        }
        let mut feature_stats: HashMap<usize, FeatureStats> = HashMap::new();
        // Insert the higher index first to make insertion order adversarial.
        feature_stats.insert(7, stats.clone());
        feature_stats.insert(2, stats);

        for _ in 0..20 {
            let (best_feat, best_gain, _) = find_best_split(&feature_stats, &[3, 3], 6, 2);
            assert!(
                best_gain > 0.0,
                "separated classes must yield positive gain"
            );
            assert_eq!(
                best_feat, 2,
                "tied gains must resolve to the lower feature index"
            );
        }
    }

    /// Regression test for `find_best_split`'s split-quality estimation: it
    /// must actually distinguish a genuinely predictive feature from pure
    /// noise. The previous implementation compared each class's mean
    /// feature value against a single threshold as an all-or-nothing
    /// assignment -- since two classes' means are almost never on the exact
    /// same side of a threshold, this made every feature (including pure
    /// noise) look like a "perfect" split, so the Hoeffding-bound
    /// gain-difference test could never clear its confidence bar and the
    /// tree never split at all.
    #[test]
    fn partial_fit_learns_a_single_feature_threshold_rule() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut tree = HoeffdingTree::new().with_grace_period(50);

        for _ in 0..2000 {
            let x0 = rng.random::<f64>();
            let x1 = rng.random::<f64>(); // pure noise, independent of label
            let y = if x0 > 0.5 { 1 } else { 0 };
            tree.partial_fit(&[x0, x1], y, 2);
        }

        let mut correct = 0;
        let n_eval = 500;
        for _ in 0..n_eval {
            let x0 = rng.random::<f64>();
            let x1 = rng.random::<f64>();
            let y = if x0 > 0.5 { 1 } else { 0 };
            if let Some((pred, _)) = tree.predict_one(&[x0, x1])
                && pred == y
            {
                correct += 1;
            }
        }
        let acc = correct as f64 / n_eval as f64;
        assert!(
            acc > 0.85,
            "HoeffdingTree should learn this trivial single-feature rule well, got accuracy {acc}"
        );
    }

    #[test]
    fn predict_one_is_none_before_any_partial_fit() {
        let tree = HoeffdingTree::new();
        assert!(tree.predict_one(&[0.5, 0.5]).is_none());
    }

    #[test]
    fn train_classif_matches_partial_fit_accuracy() {
        let mut rng = StdRng::seed_from_u64(1);
        let mut feats = Vec::new();
        let mut target = Vec::new();
        for _ in 0..1000 {
            let x0 = rng.random::<f64>();
            feats.push(x0);
            target.push(if x0 > 0.5 { 1usize } else { 0 });
        }
        let features = Array2::from_shape_vec((1000, 1), feats).unwrap();
        let task = ClassificationTask::new("ht", features.clone(), target).unwrap();

        let mut tree = HoeffdingTree::new().with_grace_period(50);
        let model = tree.train_classif(&task).unwrap();
        let pred = model.predict(&features).unwrap();
        let Prediction::Classification { predicted, .. } = pred else {
            panic!("expected classification");
        };
        let correct = predicted
            .iter()
            .zip(task.target())
            .filter(|(p, t)| *p == *t)
            .count();
        let acc = correct as f64 / predicted.len() as f64;
        assert!(
            acc > 0.85,
            "batch train_classif should fit this simple rule well, got {acc}"
        );
    }

    fn count_nodes(node: &HNode) -> usize {
        match node {
            HNode::Leaf { .. } => 1,
            HNode::Split { left, right, .. } => 1 + count_nodes(left) + count_nodes(right),
        }
    }

    /// Regression test for N5 (docs/auditoria_motor_2026-07-05.md, Fase F):
    /// the tie-breaking fallback (`epsilon < 0.01`) used to force a split
    /// even when `best_gain == 0.0` (every feature carries zero information
    /// about the label). Once a leaf sees enough samples for delta's
    /// default epsilon to fall under 0.01 (~80k, per the audit's estimate),
    /// this grew the tree without bound. Constant (zero-variance) features
    /// give a *deterministic* gain of exactly 0.0 (both classes' running
    /// Gaussians are identical, not just "small" from finite-sample noise
    /// the way genuinely random features would be) -- with 150k samples,
    /// well past that threshold, the tree must stay a single leaf.
    #[test]
    fn zero_variance_features_do_not_force_splits_once_epsilon_is_small() {
        let mut rng = StdRng::seed_from_u64(7);
        let mut tree = HoeffdingTree::new().with_grace_period(200);

        for _ in 0..150_000 {
            let y = rng.random_range(0..2); // label independent of every feature
            tree.partial_fit(&[5.0, -3.0], y, 2); // constant features: zero gain, exactly
        }

        let n_nodes = tree.root.as_ref().map(count_nodes).unwrap_or(1);
        assert_eq!(
            n_nodes, 1,
            "zero-information (constant) features should never split, even \
             with enough samples for epsilon < 0.01, got {n_nodes} nodes"
        );
    }

    #[test]
    fn normal_cdf_matches_known_values() {
        // Standard normal: CDF(0) = 0.5, CDF(mean) = 0.5 regardless of std.
        assert!((normal_cdf(0.0, 0.0, 1.0) - 0.5).abs() < 1e-6);
        assert!((normal_cdf(5.0, 5.0, 2.0) - 0.5).abs() < 1e-6);
        // CDF(mean + 1.96*std) ~ 0.975 for a standard normal.
        assert!((normal_cdf(1.96, 0.0, 1.0) - 0.975).abs() < 1e-3);
        // Degenerate (zero variance): step function at the mean.
        assert_eq!(normal_cdf(0.4, 0.5, 0.0), 0.0);
        assert_eq!(normal_cdf(0.6, 0.5, 0.0), 1.0);
    }
}
