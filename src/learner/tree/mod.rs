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
#[derive(Clone, Serialize, Deserialize)]
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
#[derive(Clone, Serialize, Deserialize)]
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
    /// Explicit fraction of `n_features`, intended to be in `(0.0, 1.0]`.
    /// `f <= 0.0` falls back to the `sqrt(n_features)` heuristic (see
    /// [`MaxFeatures::resolve`]) rather than degenerating to a single
    /// feature per split.
    Fraction(f64),
}

impl MaxFeatures {
    /// Resolves to the number of candidate features per split, or `None`
    /// for "consider all features" (no subsampling).
    ///
    /// `Fraction(f)` with `f <= 0.0` resolves to the `sqrt(n_features)`
    /// heuristic, not "0 features" (which `.max(1.0)` would otherwise turn
    /// into "1 feature per split"). This preserves the sentinel meaning
    /// `with_max_features_fraction(0.0)` had before `MaxFeatures` existed
    /// as an enum with its own explicit `Sqrt` variant -- silently
    /// reinterpreting it as "1 feature" instead would change any existing
    /// caller's model without a compile error or a runtime one.
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
            MaxFeatures::Fraction(f) if *f <= 0.0 => Some(sqrt_heuristic(n_features)),
            MaxFeatures::Fraction(f) => {
                Some((n_features as f64 * f.min(1.0)).ceil().max(1.0) as usize)
            }
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

            if self.random_splits {
                // Extra Trees: one random threshold per feature -- already a
                // single candidate, so the O(n) gini(left)/gini(right) scan
                // below isn't the O(n²) hot path (that's the "all valid split
                // points" branch, rewritten below to sweep incrementally).
                let min_val = features[[sorted[0], feat]];
                let max_val = features[[sorted[sorted.len() - 1], feat]];
                if (max_val - min_val).abs() < f64::EPSILON {
                    continue;
                }
                let threshold = rng.random_range(min_val..max_val);
                let Some(pos @ 1..) = sorted
                    .iter()
                    .position(|&idx| features[[idx, feat]] > threshold)
                else {
                    continue;
                };

                let left_idx = &sorted[..pos];
                let right_idx = &sorted[pos..];
                let gain = parent_gini
                    - (left_idx.len() as f64 / n) * gini(target, left_idx, n_classes)
                    - (right_idx.len() as f64 / n) * gini(target, right_idx, n_classes);

                if gain > best_gain {
                    best_gain = gain;
                    let threshold_mid =
                        (features[[sorted[pos - 1], feat]] + features[[sorted[pos], feat]]) / 2.0;
                    best = Some((feat, threshold_mid, left_idx.to_vec(), right_idx.to_vec(), gain));
                }
                continue;
            }

            // Standard: sweep left-to-right maintaining running per-class
            // counts, so gini(left)/gini(right) at each candidate threshold
            // is O(n_classes) instead of a full O(n) rescan of left_idx/
            // right_idx -- the previous code recomputed both from scratch at
            // every one of up to n-1 thresholds, making this loop O(n²) per
            // feature/node (audit: "TreeBuilder O(n²)"). Moving one sample
            // from right to left per step and updating counts incrementally
            // gives the exact same counts (and hence the exact same gini
            // values) as the old from-scratch computation, just without
            // rescanning.
            let mut left_counts = vec![0usize; n_classes];
            let mut right_counts = vec![0usize; n_classes];
            for &idx in &sorted {
                right_counts[target[idx]] += 1;
            }

            for i in 1..sorted.len() {
                let moved_class = target[sorted[i - 1]];
                left_counts[moved_class] += 1;
                right_counts[moved_class] -= 1;

                if (features[[sorted[i], feat]] - features[[sorted[i - 1], feat]]).abs()
                    < f64::EPSILON
                {
                    continue;
                }

                let n_left = i as f64;
                let n_right = n - n_left;
                let gain = parent_gini
                    - (n_left / n) * gini_from_counts(&left_counts, n_left)
                    - (n_right / n) * gini_from_counts(&right_counts, n_right);

                if gain > best_gain {
                    best_gain = gain;
                    let threshold =
                        (features[[sorted[i - 1], feat]] + features[[sorted[i], feat]]) / 2.0;
                    best = Some((
                        feat,
                        threshold,
                        sorted[..i].to_vec(),
                        sorted[i..].to_vec(),
                        gain,
                    ));
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
        // Center the running sums on the node mean: variance is
        // shift-invariant, but accumulating E[y²]−E[y]² on raw targets with
        // a large additive offset (UTM coordinates ~1e6-1e7, timestamps)
        // cancels catastrophically — eps·offset² reaches the magnitude of
        // the true variance and split gains become rounding noise.
        let shift = indices.iter().map(|&i| target[i]).sum::<f64>() / n;
        let mut best_gain = 0.0;
        let mut best = None;

        for &feat in candidate_features {
            let mut sorted: Vec<usize> = indices.to_vec();
            sorted.sort_by(|&a, &b| {
                features[[a, feat]]
                    .partial_cmp(&features[[b, feat]])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            if self.random_splits {
                let min_val = features[[sorted[0], feat]];
                let max_val = features[[sorted[sorted.len() - 1], feat]];
                if (max_val - min_val).abs() < f64::EPSILON {
                    continue;
                }
                let threshold = rng.random_range(min_val..max_val);
                let Some(pos @ 1..) = sorted
                    .iter()
                    .position(|&idx| features[[idx, feat]] > threshold)
                else {
                    continue;
                };

                let left_idx = &sorted[..pos];
                let right_idx = &sorted[pos..];
                let gain = parent_mse
                    - (left_idx.len() as f64 / n) * mse(target, left_idx)
                    - (right_idx.len() as f64 / n) * mse(target, right_idx);

                if gain > best_gain {
                    best_gain = gain;
                    let threshold_mid =
                        (features[[sorted[pos - 1], feat]] + features[[sorted[pos], feat]]) / 2.0;
                    best = Some((feat, threshold_mid, left_idx.to_vec(), right_idx.to_vec(), gain));
                }
                continue;
            }

            // Standard: sweep left-to-right maintaining running sum/sum-of-
            // squares, so mse(left)/mse(right) at each threshold is O(1)
            // instead of an O(n) rescan -- same fix as best_split_classif's
            // running counts, adapted to a continuous target (sum/sum_sq ->
            // variance is the standard CART/scikit-learn incremental MSE
            // formula, not specific to this crate).
            let mut left_sum = 0.0;
            let mut left_sq = 0.0;
            let mut right_sum = 0.0;
            let mut right_sq = 0.0;
            for &idx in &sorted {
                let y = target[idx] - shift;
                right_sum += y;
                right_sq += y * y;
            }

            for i in 1..sorted.len() {
                let y = target[sorted[i - 1]] - shift;
                left_sum += y;
                left_sq += y * y;
                right_sum -= y;
                right_sq -= y * y;

                if (features[[sorted[i], feat]] - features[[sorted[i - 1], feat]]).abs()
                    < f64::EPSILON
                {
                    continue;
                }

                let n_left = i as f64;
                let n_right = n - n_left;
                let gain = parent_mse
                    - (n_left / n) * mse_from_sums(left_sum, left_sq, n_left)
                    - (n_right / n) * mse_from_sums(right_sum, right_sq, n_right);

                if gain > best_gain {
                    best_gain = gain;
                    let threshold =
                        (features[[sorted[i - 1], feat]] + features[[sorted[i], feat]]) / 2.0;
                    best = Some((
                        feat,
                        threshold,
                        sorted[..i].to_vec(),
                        sorted[i..].to_vec(),
                        gain,
                    ));
                }
            }
        }

        best
    }
}

// --- Weighted tree building (Fase B1 of per-sample weights) ---
//
// Separate `*_weighted` twins of `build_classifier`/`build_regressor` and
// their sweeps, dispatched once per training call on `task.weights()`. The
// unweighted functions above are deliberately left byte-for-byte untouched:
// the hot path carries no implicit-1.0 multiplications, no weight
// allocations, and no per-sample branches (the sweep rewrite of 6c8f720
// stays exactly as it was). The weighted twins follow the sklearn
// convention:
//
// - weights enter the impurity (weighted Gini counts / weighted MSE sums)
//   and the leaf values (weighted vote / weighted mean),
// - `min_samples_split`/`min_samples_leaf` still count ROWS, not total
//   weight (sklearn's `min_samples_*` semantics; a
//   `min_weight_fraction_leaf`-style knob would be a separate parameter),
// - a weight of exactly 0.0 means "sample excluded": callers filter
//   zero-weight rows out of the root index set (see
//   [`retain_positive_weight`]), which makes weight-0 bit-identical to
//   deleting the row — including the candidate split thresholds, which a
//   merely-zero-contribution row would otherwise still perturb.
//
// The weighted MSE sweep keeps the HIGH-1 lesson: sums are accumulated on
// targets centered on the node's (now weighted) mean, so large additive
// target offsets (UTM northings, timestamps) don't cancel catastrophically.
// Products are ordered `w * y` and `w * (y * y)` — scaling the centered
// value once per row — so integer weights reproduce repeated addition's
// rounding (`fl(3y) = fl(y+y+y)`), which is what makes the
// "integer weight k ≡ row duplicated k times" oracle exact.

impl TreeBuilder {
    pub(crate) fn build_classifier_weighted(
        &mut self,
        features: &ArrayView2<f64>,
        target: &[usize],
        weights: &[f64],
        indices: &[usize],
        n_classes: usize,
        depth: usize,
        rng: &mut impl Rng,
    ) -> Node {
        if indices.len() < self.min_samples_split
            || self.max_depth.is_some_and(|d| depth >= d)
            || all_same(indices, |i| target[i])
        {
            return Node::Leaf(classification_leaf_weighted(
                target, weights, indices, n_classes,
            ));
        }

        let candidates = self.candidate_features(rng);
        if let Some((feat, threshold, left_idx, right_idx, gain)) = self
            .best_split_classif_weighted(
                features, target, weights, indices, n_classes, &candidates, rng,
            )
        {
            if left_idx.len() < self.min_samples_leaf || right_idx.len() < self.min_samples_leaf {
                return Node::Leaf(classification_leaf_weighted(
                    target, weights, indices, n_classes,
                ));
            }

            // Importance credit scales by the node's total weight — the
            // weighted analogue of `gain * indices.len()` (with all-ones
            // weights the two are bit-identical).
            let node_weight: f64 = indices.iter().map(|&i| weights[i]).sum();
            self.feature_importances[feat] += gain * node_weight;
            let left = self.build_classifier_weighted(
                features, target, weights, &left_idx, n_classes, depth + 1, rng,
            );
            let right = self.build_classifier_weighted(
                features, target, weights, &right_idx, n_classes, depth + 1, rng,
            );

            Node::Split {
                feature: feat,
                threshold,
                left: Box::new(left),
                right: Box::new(right),
            }
        } else {
            Node::Leaf(classification_leaf_weighted(
                target, weights, indices, n_classes,
            ))
        }
    }

    pub(crate) fn build_regressor_weighted(
        &mut self,
        features: &ArrayView2<f64>,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        depth: usize,
        rng: &mut impl Rng,
    ) -> Node {
        if indices.len() < self.min_samples_split
            || self.max_depth.is_some_and(|d| depth >= d)
            || all_same(indices, |i| target[i].to_bits())
        {
            return Node::Leaf(regression_leaf_weighted(target, weights, indices));
        }

        let candidates = self.candidate_features(rng);
        if let Some((feat, threshold, left_idx, right_idx, gain)) =
            self.best_split_regress_weighted(features, target, weights, indices, &candidates, rng)
        {
            if left_idx.len() < self.min_samples_leaf || right_idx.len() < self.min_samples_leaf {
                return Node::Leaf(regression_leaf_weighted(target, weights, indices));
            }

            let node_weight: f64 = indices.iter().map(|&i| weights[i]).sum();
            self.feature_importances[feat] += gain * node_weight;
            let left =
                self.build_regressor_weighted(features, target, weights, &left_idx, depth + 1, rng);
            let right = self
                .build_regressor_weighted(features, target, weights, &right_idx, depth + 1, rng);

            Node::Split {
                feature: feat,
                threshold,
                left: Box::new(left),
                right: Box::new(right),
            }
        } else {
            Node::Leaf(regression_leaf_weighted(target, weights, indices))
        }
    }

    /// Weighted twin of [`TreeBuilder::best_split_classif`]: identical sweep
    /// structure, with per-class weight sums where the unweighted sweep has
    /// integer counts. With all-ones weights every accumulated quantity is
    /// the exact same f64 value the unweighted sweep computes (integer sums
    /// are exact in f64), so the two return bit-identical splits.
    fn best_split_classif_weighted(
        &self,
        features: &ArrayView2<f64>,
        target: &[usize],
        weights: &[f64],
        indices: &[usize],
        n_classes: usize,
        candidate_features: &[usize],
        rng: &mut impl Rng,
    ) -> Option<(usize, f64, Vec<usize>, Vec<usize>, f64)> {
        let parent_gini = gini_weighted(target, weights, indices, n_classes);
        let n: f64 = indices.iter().map(|&i| weights[i]).sum();
        let mut best_gain = 0.0;
        let mut best = None;

        for &feat in candidate_features {
            let mut sorted: Vec<usize> = indices.to_vec();
            sorted.sort_by(|&a, &b| {
                features[[a, feat]]
                    .partial_cmp(&features[[b, feat]])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            if self.random_splits {
                let min_val = features[[sorted[0], feat]];
                let max_val = features[[sorted[sorted.len() - 1], feat]];
                if (max_val - min_val).abs() < f64::EPSILON {
                    continue;
                }
                let threshold = rng.random_range(min_val..max_val);
                let Some(pos @ 1..) = sorted
                    .iter()
                    .position(|&idx| features[[idx, feat]] > threshold)
                else {
                    continue;
                };

                let left_idx = &sorted[..pos];
                let right_idx = &sorted[pos..];
                let w_left: f64 = left_idx.iter().map(|&i| weights[i]).sum();
                let w_right = n - w_left;
                let gain = parent_gini
                    - (w_left / n) * gini_weighted(target, weights, left_idx, n_classes)
                    - (w_right / n) * gini_weighted(target, weights, right_idx, n_classes);

                if gain > best_gain {
                    best_gain = gain;
                    let threshold_mid =
                        (features[[sorted[pos - 1], feat]] + features[[sorted[pos], feat]]) / 2.0;
                    best = Some((feat, threshold_mid, left_idx.to_vec(), right_idx.to_vec(), gain));
                }
                continue;
            }

            let mut left_counts = vec![0.0f64; n_classes];
            let mut right_counts = vec![0.0f64; n_classes];
            for &idx in &sorted {
                right_counts[target[idx]] += weights[idx];
            }

            let mut w_left = 0.0;
            for i in 1..sorted.len() {
                let moved = sorted[i - 1];
                let w = weights[moved];
                left_counts[target[moved]] += w;
                right_counts[target[moved]] -= w;
                w_left += w;

                if (features[[sorted[i], feat]] - features[[sorted[i - 1], feat]]).abs()
                    < f64::EPSILON
                {
                    continue;
                }

                let w_right = n - w_left;
                let gain = parent_gini
                    - (w_left / n) * gini_from_weighted_counts(&left_counts, w_left)
                    - (w_right / n) * gini_from_weighted_counts(&right_counts, w_right);

                if gain > best_gain {
                    best_gain = gain;
                    let threshold =
                        (features[[sorted[i - 1], feat]] + features[[sorted[i], feat]]) / 2.0;
                    best = Some((
                        feat,
                        threshold,
                        sorted[..i].to_vec(),
                        sorted[i..].to_vec(),
                        gain,
                    ));
                }
            }
        }

        best
    }

    /// Weighted twin of [`TreeBuilder::best_split_regress`]. Keeps the
    /// centered accumulation (the shift is now the node's *weighted* mean),
    /// so the HIGH-1 offset-invariance property holds for weighted trees
    /// too.
    fn best_split_regress_weighted(
        &self,
        features: &ArrayView2<f64>,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        candidate_features: &[usize],
        rng: &mut impl Rng,
    ) -> Option<(usize, f64, Vec<usize>, Vec<usize>, f64)> {
        let parent_mse = wmse(target, weights, indices);
        let n: f64 = indices.iter().map(|&i| weights[i]).sum();
        let shift = indices.iter().map(|&i| weights[i] * target[i]).sum::<f64>() / n;
        let mut best_gain = 0.0;
        let mut best = None;

        for &feat in candidate_features {
            let mut sorted: Vec<usize> = indices.to_vec();
            sorted.sort_by(|&a, &b| {
                features[[a, feat]]
                    .partial_cmp(&features[[b, feat]])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            if self.random_splits {
                let min_val = features[[sorted[0], feat]];
                let max_val = features[[sorted[sorted.len() - 1], feat]];
                if (max_val - min_val).abs() < f64::EPSILON {
                    continue;
                }
                let threshold = rng.random_range(min_val..max_val);
                let Some(pos @ 1..) = sorted
                    .iter()
                    .position(|&idx| features[[idx, feat]] > threshold)
                else {
                    continue;
                };

                let left_idx = &sorted[..pos];
                let right_idx = &sorted[pos..];
                let w_left: f64 = left_idx.iter().map(|&i| weights[i]).sum();
                let w_right = n - w_left;
                let gain = parent_mse
                    - (w_left / n) * wmse(target, weights, left_idx)
                    - (w_right / n) * wmse(target, weights, right_idx);

                if gain > best_gain {
                    best_gain = gain;
                    let threshold_mid =
                        (features[[sorted[pos - 1], feat]] + features[[sorted[pos], feat]]) / 2.0;
                    best = Some((feat, threshold_mid, left_idx.to_vec(), right_idx.to_vec(), gain));
                }
                continue;
            }

            let mut left_sum = 0.0;
            let mut left_sq = 0.0;
            let mut right_sum = 0.0;
            let mut right_sq = 0.0;
            for &idx in &sorted {
                let y = target[idx] - shift;
                let w = weights[idx];
                right_sum += w * y;
                right_sq += w * (y * y);
            }

            let mut w_left = 0.0;
            for i in 1..sorted.len() {
                let moved = sorted[i - 1];
                let y = target[moved] - shift;
                let w = weights[moved];
                let wy = w * y;
                let wyy = w * (y * y);
                left_sum += wy;
                left_sq += wyy;
                right_sum -= wy;
                right_sq -= wyy;
                w_left += w;

                if (features[[sorted[i], feat]] - features[[sorted[i - 1], feat]]).abs()
                    < f64::EPSILON
                {
                    continue;
                }

                let w_right = n - w_left;
                let gain = parent_mse
                    - (w_left / n) * mse_from_sums(left_sum, left_sq, w_left)
                    - (w_right / n) * mse_from_sums(right_sum, right_sq, w_right);

                if gain > best_gain {
                    best_gain = gain;
                    let threshold =
                        (features[[sorted[i - 1], feat]] + features[[sorted[i], feat]]) / 2.0;
                    best = Some((
                        feat,
                        threshold,
                        sorted[..i].to_vec(),
                        sorted[i..].to_vec(),
                        gain,
                    ));
                }
            }
        }

        best
    }
}

/// Filters an index set down to rows with positive weight (weight 0.0 means
/// "sample excluded" — Fase A's validated semantics). If filtering empties
/// the set (a degenerate bootstrap/subsample that drew only excluded rows),
/// falls back to every positive-weight row of the training set so the tree
/// still trains on something honoring the weights, rather than panicking on
/// an empty node.
pub(crate) fn retain_positive_weight(
    indices: Vec<usize>,
    weights: &[f64],
    n_samples: usize,
) -> Vec<usize> {
    let filtered: Vec<usize> = indices.into_iter().filter(|&i| weights[i] > 0.0).collect();
    if !filtered.is_empty() {
        return filtered;
    }
    (0..n_samples).filter(|&i| weights[i] > 0.0).collect()
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

/// Gini impurity from pre-aggregated per-class counts, for `n` samples --
/// equivalent to `gini(target, indices, n_classes)` but O(n_classes) instead
/// of O(n), used by [`TreeBuilder::best_split_classif`]'s incremental sweep
/// (and by `ObliqueTreeBuilder`'s, which shares the same rescan-free sweep).
pub(crate) fn gini_from_counts(counts: &[usize], n: f64) -> f64 {
    if n <= 0.0 {
        return 0.0;
    }
    1.0 - counts
        .iter()
        .map(|&c| (c as f64 / n).powi(2))
        .sum::<f64>()
}

/// MSE (variance) from a running sum and sum-of-squares over `n` samples --
/// equivalent to `mse(target, indices)` but O(1) instead of O(n), used by
/// [`TreeBuilder::best_split_regress`]'s incremental sweep (and by the
/// ObliqueTree/QuantileForest sweeps, which share it). The standard
/// CART/scikit-learn formula (`E[y²] - E[y]²`); `.max(0.0)` guards against a
/// tiny negative value from floating-point cancellation when the true
/// variance is at or near zero. The caller must accumulate `sum`/`sum_sq`
/// over targets centered on the node mean: on raw values this formula
/// cancels catastrophically once `mean² · f64::EPSILON` approaches the true
/// variance (offsets ≳1e6 for unit-scale spread).
pub(crate) fn mse_from_sums(sum: f64, sum_sq: f64, n: f64) -> f64 {
    if n <= 0.0 {
        return 0.0;
    }
    let mean = sum / n;
    (sum_sq / n - mean * mean).max(0.0)
}

/// Weighted Gini impurity: per-class weight sums instead of counts. With
/// all-ones weights this is bit-identical to [`gini`] (integer sums are
/// exact in f64 and the final expression is the same).
pub(crate) fn gini_weighted(
    target: &[usize],
    weights: &[f64],
    indices: &[usize],
    n_classes: usize,
) -> f64 {
    let mut counts = vec![0.0f64; n_classes];
    let mut total = 0.0;
    for &i in indices {
        counts[target[i]] += weights[i];
        total += weights[i];
    }
    gini_from_weighted_counts(&counts, total)
}

/// [`gini_from_counts`] over pre-aggregated per-class *weight* sums, for
/// total weight `n` — the weighted sweep's O(n_classes) impurity.
pub(crate) fn gini_from_weighted_counts(counts: &[f64], n: f64) -> f64 {
    if n <= 0.0 {
        return 0.0;
    }
    1.0 - counts.iter().map(|&c| (c / n).powi(2)).sum::<f64>()
}

/// Weighted MSE (weighted variance) around the weighted mean — the weighted
/// twin of [`mse`], with the same accumulate-around-the-mean form (no
/// E[y²]−E[y]² on raw targets; see the HIGH-1 note on [`mse_from_sums`]).
pub(crate) fn wmse(target: &[f64], weights: &[f64], indices: &[usize]) -> f64 {
    let mut sw = 0.0;
    let mut swy = 0.0;
    for &i in indices {
        sw += weights[i];
        swy += weights[i] * target[i];
    }
    let mean = swy / sw;
    let mut acc = 0.0;
    for &i in indices {
        let d = target[i] - mean;
        acc += weights[i] * (d * d);
    }
    acc / sw
}

/// Weighted classification leaf: probabilities are per-class weight shares
/// and the predicted class is the weighted-majority vote. Tie-breaking
/// matches [`classification_leaf`]'s `max_by_key` (last maximum wins), so
/// all-ones weights produce a bit-identical leaf.
pub(crate) fn classification_leaf_weighted(
    target: &[usize],
    weights: &[f64],
    indices: &[usize],
    n_classes: usize,
) -> LeafValue {
    let mut counts = vec![0.0f64; n_classes];
    let mut total = 0.0;
    for &i in indices {
        counts[target[i]] += weights[i];
        total += weights[i];
    }
    let probs: Vec<f64> = counts.iter().map(|&c| c / total).collect();
    let predicted = counts
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap()
        .0;
    LeafValue::Class(predicted, probs)
}

/// Weighted regression leaf: the weighted mean of the raw targets.
pub(crate) fn regression_leaf_weighted(
    target: &[f64],
    weights: &[f64],
    indices: &[usize],
) -> LeafValue {
    let mut sw = 0.0;
    let mut swy = 0.0;
    for &i in indices {
        sw += weights[i];
        swy += weights[i] * target[i];
    }
    LeafValue::Value(swy / sw)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test: before this fix, `Fraction(0.0)` resolved to
    /// `(n_features as f64 * 0.0).ceil().max(1.0) = 1` -- a single feature
    /// per split -- silently changing the meaning `0.0` had before
    /// `MaxFeatures` existed as an enum, when it was a sentinel for "use
    /// the sqrt(n_features) heuristic". `Fraction(f<=0.0)` must resolve the
    /// same as `Sqrt`, not degenerate to 1 feature.
    #[test]
    fn fraction_zero_or_negative_falls_back_to_sqrt_heuristic() {
        let sqrt_resolved = MaxFeatures::Sqrt.resolve(48, false);
        assert_eq!(MaxFeatures::Fraction(0.0).resolve(48, false), sqrt_resolved);
        assert_eq!(MaxFeatures::Fraction(-1.0).resolve(48, false), sqrt_resolved);
        assert_eq!(sqrt_resolved, Some(7)); // ceil(sqrt(48)) = 7
    }

    /// A positive fraction still resolves proportionally, unaffected by the
    /// `f <= 0.0` special case.
    #[test]
    fn fraction_positive_resolves_proportionally() {
        assert_eq!(MaxFeatures::Fraction(0.5).resolve(10, false), Some(5));
        assert_eq!(MaxFeatures::Fraction(1.0).resolve(10, false), Some(10));
        // Values above 1.0 are clamped, not extrapolated past n_features.
        assert_eq!(MaxFeatures::Fraction(2.0).resolve(10, false), Some(10));
    }

    /// Regression test for the TreeBuilder O(n²) fix (audit
    /// "TreeBuilder O(n²)"): `gini_from_counts` (used by the incremental
    /// sweep) must equal the brute-force `gini()` for any partition, since
    /// they're both computing the exact same quantity from the same
    /// underlying per-class counts -- one incrementally, one by rescanning.
    #[test]
    fn gini_from_counts_matches_full_scan_gini() {
        let target = vec![0usize, 1, 2, 0, 1, 1, 2, 0, 0, 2];
        let n_classes = 3;
        // Every possible split point of the sorted-by-index sample, i.e.
        // every prefix/suffix pair -- exercises small and uneven counts.
        for split in 1..target.len() {
            let left: Vec<usize> = (0..split).collect();
            let mut counts = vec![0usize; n_classes];
            for &i in &left {
                counts[target[i]] += 1;
            }
            let expected = gini(&target, &left, n_classes);
            let actual = gini_from_counts(&counts, left.len() as f64);
            assert!(
                (expected - actual).abs() < 1e-12,
                "split={split}: full-scan gini={expected}, from-counts gini={actual}"
            );
        }
    }

    /// Same equivalence check as above, for the regression (MSE) path.
    #[test]
    fn mse_from_sums_matches_full_scan_mse() {
        let target = vec![1.0, 5.0, 3.0, 8.0, 2.0, 9.0, 4.0, 7.0, 6.0, 0.0];
        for split in 1..target.len() {
            let left: Vec<usize> = (0..split).collect();
            let (sum, sq) = left.iter().fold((0.0, 0.0), |(s, sq), &i| {
                (s + target[i], sq + target[i] * target[i])
            });
            let expected = mse(&target, &left);
            let actual = mse_from_sums(sum, sq, left.len() as f64);
            assert!(
                (expected - actual).abs() < 1e-9,
                "split={split}: full-scan mse={expected}, from-sums mse={actual}"
            );
        }
    }

    /// Golden test: a single feature with an obvious optimal split
    /// (threshold between x=2 and x=3 perfectly separates the two classes)
    /// must be found by the incrementally-swept `best_split_classif`, with
    /// perfect (gini=0) children.
    #[test]
    fn best_split_classif_finds_the_hand_computed_optimal_split() {
        use ndarray::array;
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let features = array![[1.0], [2.0], [2.0], [3.0], [4.0], [4.0]];
        let target = vec![0, 0, 0, 1, 1, 1];
        let indices: Vec<usize> = (0..6).collect();
        let builder = TreeBuilder::new(None, 2, 1, None, 1);
        let mut rng = StdRng::seed_from_u64(0);

        let (feat, threshold, left, right, gain) = builder
            .best_split_classif(&features.view(), &target, &indices, 2, &[0], &mut rng)
            .expect("an obviously separable dataset must find a split");

        assert_eq!(feat, 0);
        assert!(
            (threshold - 2.5).abs() < 1e-9,
            "expected the midpoint between x=2 and x=3, got {threshold}"
        );
        assert_eq!(left.len(), 3);
        assert_eq!(right.len(), 3);
        assert!(gain > 0.49, "a perfect split should recover ~all of the parent's gini: {gain}");
    }

    /// Same golden check for the regression (MSE) path: an obvious step
    /// function must be split exactly at the step.
    #[test]
    fn best_split_regress_finds_the_hand_computed_optimal_split() {
        use ndarray::array;
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let features = array![[1.0], [2.0], [2.0], [3.0], [4.0], [4.0]];
        let target = vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0];
        let indices: Vec<usize> = (0..6).collect();
        let builder = TreeBuilder::new(None, 2, 1, None, 1);
        let mut rng = StdRng::seed_from_u64(0);

        let (feat, threshold, left, right, gain) = builder
            .best_split_regress(&features.view(), &target, &indices, &[0], &mut rng)
            .expect("an obvious step function must find a split");

        assert_eq!(feat, 0);
        assert!((threshold - 2.5).abs() < 1e-9, "got {threshold}");
        assert_eq!(left.len(), 3);
        assert_eq!(right.len(), 3);
        // Parent MSE is variance of {0,0,0,10,10,10} = 25; a perfect split
        // (both children constant) should recover essentially all of it.
        assert!(gain > 24.9, "a perfect split should recover ~all of the parent's mse: {gain}");
    }

    /// Regression test for the catastrophic-cancellation follow-up to the
    /// O(n²) fix: the incremental sweep accumulates E[y²]−E[y]² and, on raw
    /// targets carrying a large additive offset (UTM northing ~7e6,
    /// timestamps ~1e9), eps·offset² swamps the true variance and split
    /// gains become rounding noise. The sums must therefore be centered on
    /// the node mean, making the found split independent of the offset.
    #[test]
    fn best_split_regress_is_invariant_to_large_target_offsets() {
        use ndarray::Array2;
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let n = 400;
        let features =
            Array2::from_shape_fn((n, 1), |(i, _)| i as f64 / n as f64 * 10.0);
        let base: Vec<f64> = (0..n)
            .map(|i| {
                let x = features[[i, 0]];
                // Step signal + deterministic pseudo-noise, unit scale.
                let step = if x < 5.0 { 0.0 } else { 4.0 };
                step + 0.3 * ((i as f64 * 12.9898).sin())
            })
            .collect();
        let indices: Vec<usize> = (0..n).collect();
        let builder = TreeBuilder::new(None, 2, 1, None, 1);

        for offset in [0.0, 1e6, 1e8] {
            let target: Vec<f64> = base.iter().map(|y| y + offset).collect();
            let mut rng = StdRng::seed_from_u64(0);
            let (feat, threshold, ..) = builder
                .best_split_regress(&features.view(), &target, &indices, &[0], &mut rng)
                .expect("a clear step function must find a split");
            assert_eq!(feat, 0);
            assert!(
                (threshold - 5.0).abs() < 0.1,
                "offset {offset:e}: the split must stay at the step (x≈5), got {threshold}"
            );
        }
    }
}
