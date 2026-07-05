//! Shared tree-building infrastructure for DecisionTree, RandomForest, and GradientBoosting.

pub mod decision_tree;
pub mod extra_trees;
pub mod gradient_boosting;
pub mod random_forest;

use ndarray::{ArrayView1, ArrayView2};
use rand::Rng;
use rand::seq::index::sample;

// --- Core types ---

use serde::{Deserialize, Serialize};

/// A node in a decision tree: either a leaf prediction or an internal split.
#[derive(Serialize, Deserialize)]
pub enum Node {
    /// A terminal node holding the prediction for samples that reach it.
    Leaf(LeafValue),
    /// An internal node that routes samples to `left` or `right` based on `feature`/`threshold`.
    Split {
        /// Index of the feature this split tests.
        feature: usize,
        /// Threshold value; samples with `feature <= threshold` go left, others go right.
        threshold: f64,
        /// Subtree for samples with `feature <= threshold`.
        left: Box<Node>,
        /// Subtree for samples with `feature > threshold`.
        right: Box<Node>,
    },
}

/// The prediction stored at a leaf node.
#[derive(Serialize, Deserialize)]
pub enum LeafValue {
    /// Classification leaf: predicted class index and per-class probability vector.
    Class(usize, Vec<f64>),
    /// Regression leaf: predicted value.
    Value(f64),
}

impl Node {
    pub(crate) fn predict_one(&self, sample: ArrayView1<f64>) -> &LeafValue {
        match self {
            Node::Leaf(value) => value,
            Node::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                if sample[*feature] <= *threshold {
                    left.predict_one(sample)
                } else {
                    right.predict_one(sample)
                }
            }
        }
    }
}

/// How many candidate features [`RandomForest`](super::RandomForest)/
/// [`ExtraTrees`](super::ExtraTrees) consider at each split.
///
/// `Auto` resolves differently by task: `sqrt(n_features)` for
/// classification (Breiman's original RF guidance, and scikit-learn's
/// `RandomForestClassifier`/`ExtraTreesClassifier` default), but *all*
/// features for regression (scikit-learn's `RandomForestRegressor`/
/// `ExtraTreesRegressor` default). Regression tasks are hurt more by
/// aggressive feature subsampling than classification: when a small number
/// of features carries most of the signal (common in wide regression
/// tables), restricting every split to a random `sqrt(p)`-sized subset can
/// mean many splits never see the informative features at all, degrading
/// every tree in the ensemble rather than just adding beneficial diversity.
/// Confirmed empirically: applying the classification-style `sqrt(p)`
/// default to a 48-feature regression benchmark (OpenML `pol`) more than
/// doubled RMSE versus scikit-learn's all-features default.
#[derive(Clone, Copy, Debug)]
pub(crate) enum MaxFeatures {
    /// Task-appropriate default -- see the type's own docs.
    Auto,
    /// Force the `sqrt(n_features)` heuristic regardless of task.
    Sqrt,
    /// Explicit fraction of `n_features` in `(0.0, 1.0]`.
    Fraction(f64),
}

impl MaxFeatures {
    /// Resolves to the number of candidate features per split, or `None`
    /// for "consider all features" (no subsampling).
    pub(crate) fn resolve(&self, n_features: usize, is_classif: bool) -> Option<usize> {
        match self {
            MaxFeatures::Auto => {
                if is_classif {
                    Some(sqrt_heuristic(n_features))
                } else {
                    None
                }
            }
            MaxFeatures::Sqrt => Some(sqrt_heuristic(n_features)),
            MaxFeatures::Fraction(f) => Some((n_features as f64 * f).ceil().max(1.0) as usize),
        }
    }
}

fn sqrt_heuristic(n_features: usize) -> usize {
    (n_features as f64).sqrt().ceil() as usize
}

// --- Tree builder ---

pub(crate) struct TreeBuilder {
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    /// If Some, only consider these features at each split (for Random Forest).
    /// If None, consider all features.
    max_features: Option<usize>,
    /// If true, pick a random threshold per feature instead of the optimal one (Extra Trees).
    random_splits: bool,
    pub(crate) feature_importances: Vec<f64>,
    n_features: usize,
}

impl TreeBuilder {
    pub(crate) fn new(
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        max_features: Option<usize>,
        n_features: usize,
    ) -> Self {
        Self {
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_features,
            random_splits: false,
            feature_importances: vec![0.0; n_features],
            n_features,
        }
    }

    pub(crate) fn with_random_splits(mut self, random: bool) -> Self {
        self.random_splits = random;
        self
    }

    fn candidate_features(&self, rng: &mut impl Rng) -> Vec<usize> {
        match self.max_features {
            None => (0..self.n_features).collect(),
            Some(k) if k >= self.n_features => (0..self.n_features).collect(),
            Some(k) => sample(rng, self.n_features, k).into_vec(),
        }
    }

    pub(crate) fn build_classifier(
        &mut self,
        features: &ArrayView2<f64>,
        target: &[usize],
        indices: &[usize],
        n_classes: usize,
        depth: usize,
        rng: &mut impl Rng,
    ) -> Node {
        if indices.len() < self.min_samples_split
            || self.max_depth.is_some_and(|d| depth >= d)
            || all_same(indices, |i| target[i])
        {
            return Node::Leaf(classification_leaf(target, indices, n_classes));
        }

        let candidates = self.candidate_features(rng);
        if let Some((feat, threshold, left_idx, right_idx, gain)) =
            self.best_split_classif(features, target, indices, n_classes, &candidates, rng)
        {
            if left_idx.len() < self.min_samples_leaf || right_idx.len() < self.min_samples_leaf {
                return Node::Leaf(classification_leaf(target, indices, n_classes));
            }

            self.feature_importances[feat] += gain * indices.len() as f64;
            let left =
                self.build_classifier(features, target, &left_idx, n_classes, depth + 1, rng);
            let right =
                self.build_classifier(features, target, &right_idx, n_classes, depth + 1, rng);

            Node::Split {
                feature: feat,
                threshold,
                left: Box::new(left),
                right: Box::new(right),
            }
        } else {
            Node::Leaf(classification_leaf(target, indices, n_classes))
        }
    }

    pub(crate) fn build_regressor(
        &mut self,
        features: &ArrayView2<f64>,
        target: &[f64],
        indices: &[usize],
        depth: usize,
        rng: &mut impl Rng,
    ) -> Node {
        if indices.len() < self.min_samples_split
            || self.max_depth.is_some_and(|d| depth >= d)
            || all_same(indices, |i| target[i].to_bits())
        {
            return Node::Leaf(regression_leaf(target, indices));
        }

        let candidates = self.candidate_features(rng);
        if let Some((feat, threshold, left_idx, right_idx, gain)) =
            self.best_split_regress(features, target, indices, &candidates, rng)
        {
            if left_idx.len() < self.min_samples_leaf || right_idx.len() < self.min_samples_leaf {
                return Node::Leaf(regression_leaf(target, indices));
            }

            self.feature_importances[feat] += gain * indices.len() as f64;
            let left = self.build_regressor(features, target, &left_idx, depth + 1, rng);
            let right = self.build_regressor(features, target, &right_idx, depth + 1, rng);

            Node::Split {
                feature: feat,
                threshold,
                left: Box::new(left),
                right: Box::new(right),
            }
        } else {
            Node::Leaf(regression_leaf(target, indices))
        }
    }

    fn best_split_classif(
        &self,
        features: &ArrayView2<f64>,
        target: &[usize],
        indices: &[usize],
        n_classes: usize,
        candidate_features: &[usize],
        rng: &mut impl Rng,
    ) -> Option<(usize, f64, Vec<usize>, Vec<usize>, f64)> {
        let parent_gini = gini(target, indices, n_classes);
        let n = indices.len() as f64;
        let mut best_gain = 0.0;
        let mut best = None;

        for &feat in candidate_features {
            let mut sorted: Vec<usize> = indices.to_vec();
            sorted.sort_by(|&a, &b| {
                features[[a, feat]]
                    .partial_cmp(&features[[b, feat]])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let split_points: Vec<usize> = if self.random_splits {
                // Extra Trees: one random threshold per feature
                let min_val = features[[sorted[0], feat]];
                let max_val = features[[sorted[sorted.len() - 1], feat]];
                if (max_val - min_val).abs() < f64::EPSILON {
                    continue;
                }
                let threshold = rng.random_range(min_val..max_val);
                match sorted
                    .iter()
                    .position(|&idx| features[[idx, feat]] > threshold)
                {
                    Some(pos) if pos > 0 => vec![pos],
                    _ => continue,
                }
            } else {
                // Standard: all valid split points
                (1..sorted.len())
                    .filter(|&i| {
                        (features[[sorted[i], feat]] - features[[sorted[i - 1], feat]]).abs()
                            >= f64::EPSILON
                    })
                    .collect()
            };

            for i in split_points {
                let left_idx = &sorted[..i];
                let right_idx = &sorted[i..];

                let gain = parent_gini
                    - (left_idx.len() as f64 / n) * gini(target, left_idx, n_classes)
                    - (right_idx.len() as f64 / n) * gini(target, right_idx, n_classes);

                if gain > best_gain {
                    best_gain = gain;
                    let threshold =
                        (features[[sorted[i - 1], feat]] + features[[sorted[i], feat]]) / 2.0;
                    best = Some((feat, threshold, left_idx.to_vec(), right_idx.to_vec(), gain));
                }
            }
        }

        best
    }

    fn best_split_regress(
        &self,
        features: &ArrayView2<f64>,
        target: &[f64],
        indices: &[usize],
        candidate_features: &[usize],
        rng: &mut impl Rng,
    ) -> Option<(usize, f64, Vec<usize>, Vec<usize>, f64)> {
        let parent_mse = mse(target, indices);
        let n = indices.len() as f64;
        let mut best_gain = 0.0;
        let mut best = None;

        for &feat in candidate_features {
            let mut sorted: Vec<usize> = indices.to_vec();
            sorted.sort_by(|&a, &b| {
                features[[a, feat]]
                    .partial_cmp(&features[[b, feat]])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let split_points: Vec<usize> = if self.random_splits {
                let min_val = features[[sorted[0], feat]];
                let max_val = features[[sorted[sorted.len() - 1], feat]];
                if (max_val - min_val).abs() < f64::EPSILON {
                    continue;
                }
                let threshold = rng.random_range(min_val..max_val);
                match sorted
                    .iter()
                    .position(|&idx| features[[idx, feat]] > threshold)
                {
                    Some(pos) if pos > 0 => vec![pos],
                    _ => continue,
                }
            } else {
                (1..sorted.len())
                    .filter(|&i| {
                        (features[[sorted[i], feat]] - features[[sorted[i - 1], feat]]).abs()
                            >= f64::EPSILON
                    })
                    .collect()
            };

            for i in split_points {
                let left_idx = &sorted[..i];
                let right_idx = &sorted[i..];

                let gain = parent_mse
                    - (left_idx.len() as f64 / n) * mse(target, left_idx)
                    - (right_idx.len() as f64 / n) * mse(target, right_idx);

                if gain > best_gain {
                    best_gain = gain;
                    let threshold =
                        (features[[sorted[i - 1], feat]] + features[[sorted[i], feat]]) / 2.0;
                    best = Some((feat, threshold, left_idx.to_vec(), right_idx.to_vec(), gain));
                }
            }
        }

        best
    }
}

// --- Free functions ---

pub(crate) fn all_same<T: Eq>(indices: &[usize], val: impl Fn(usize) -> T) -> bool {
    let first = val(indices[0]);
    indices.iter().all(|&i| val(i) == first)
}

pub(crate) fn gini(target: &[usize], indices: &[usize], n_classes: usize) -> f64 {
    let mut counts = vec![0usize; n_classes];
    for &i in indices {
        counts[target[i]] += 1;
    }
    let total = indices.len() as f64;
    1.0 - counts
        .iter()
        .map(|&c| (c as f64 / total).powi(2))
        .sum::<f64>()
}

pub(crate) fn mse(target: &[f64], indices: &[usize]) -> f64 {
    let mean = indices.iter().map(|&i| target[i]).sum::<f64>() / indices.len() as f64;
    indices
        .iter()
        .map(|&i| (target[i] - mean).powi(2))
        .sum::<f64>()
        / indices.len() as f64
}

pub(crate) fn classification_leaf(
    target: &[usize],
    indices: &[usize],
    n_classes: usize,
) -> LeafValue {
    let mut counts = vec![0usize; n_classes];
    for &i in indices {
        counts[target[i]] += 1;
    }
    let total = indices.len() as f64;
    let probs: Vec<f64> = counts.iter().map(|&c| c as f64 / total).collect();
    let predicted = counts
        .iter()
        .enumerate()
        .max_by_key(|&(_, &c)| c)
        .unwrap()
        .0;
    LeafValue::Class(predicted, probs)
}

pub(crate) fn regression_leaf(target: &[f64], indices: &[usize]) -> LeafValue {
    let mean = indices.iter().map(|&i| target[i]).sum::<f64>() / indices.len() as f64;
    LeafValue::Value(mean)
}
