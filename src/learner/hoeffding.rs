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

use crate::learner::{Learner, TrainedModel};
use crate::prediction::Prediction;
use crate::Result;
use crate::task::{ClassificationTask, Task};
use ndarray::Array2;
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

        if let Some(root) = &mut self.root {
            Self::update_node(
                root,
                features,
                label,
                delta,
                grace,
                max_depth,
                n_classes_local,
            );
        }
    }

    fn update_node(
        node: &mut HNode,
        features: &[f64],
        label: usize,
        delta: f64,
        grace: usize,
        max_depth: Option<usize>,
        n_classes: usize,
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

                // Update per-feature statistics
                for (j, &val) in features.iter().enumerate() {
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

                    // Hoeffding bound
                    let r = (n_classes as f64).log2(); // range of information gain
                    let epsilon = (r * r * (1.0 / delta).ln() / (2.0 * *n_seen as f64)).sqrt();

                    if best_gain - second_gain > epsilon || epsilon < 0.01 {
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
                            node, features, label, delta, grace, max_depth, n_classes,
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
                    Self::update_node(left, features, label, delta, grace, max_depth, n_classes);
                } else {
                    Self::update_node(right, features, label, delta, grace, max_depth, n_classes);
                }
            }
        }
    }
}

// ── Hoeffding Tree Node ─────────────────────────────────────────────

enum HNode {
    Leaf {
        counts: Vec<usize>, // class counts
        n_seen: usize,
        depth: usize,
        feature_stats: HashMap<usize, FeatureStats>,
    },
    Split {
        feature: usize,
        threshold: f64,
        left: Box<HNode>,
        right: Box<HNode>,
    },
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

        // Estimate left/right counts using class means
        let mut left_counts = vec![0usize; n_classes];
        let mut right_counts = vec![0usize; n_classes];

        for (c, &(sum, _, count)) in stats.class_stats.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let mean = sum / count as f64;
            if mean <= threshold {
                left_counts[c] = count;
            } else {
                right_counts[c] = count;
            }
        }

        let n_left: usize = left_counts.iter().sum();
        let n_right: usize = right_counts.iter().sum();

        if n_left == 0 || n_right == 0 {
            continue;
        }

        let left_ent = entropy(&left_counts, n_left);
        let right_ent = entropy(&right_counts, n_right);

        let gain = parent_ent
            - (n_left as f64 / n_total as f64) * left_ent
            - (n_right as f64 / n_total as f64) * right_ent;

        gains.push((feat, gain));
    }

    gains.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

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

    fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
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

struct TrainedHoeffdingTree {
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
}
