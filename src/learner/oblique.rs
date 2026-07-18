//! Oblique Decision Trees and Forest (SPORF).
//!
//! Splits on linear combinations of features instead of single features.
//! Uses Sparse Projection Oblique Randomer Forest (SPORF) approach:
//! sparse random projections from {-1, +1} weights.
//!
//! Reference: Tomita et al. (2020) "Sparse Projection Oblique Randomer Forests"
//! JMLR 21(104):1-39.

use crate::Result;
use crate::learner::tree::{gini_from_counts, mse_from_sums};
use crate::learner::{Learner, TrainedModel};
use crate::prediction::Prediction;
use crate::task::{ClassificationTask, RegressionTask, Task};
use ndarray::{Array2, ArrayView1};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// ── Oblique tree node ───────────────────────────────────────────────

/// A projection: linear combination of features.
/// Stored as sparse (feature_index, weight) pairs.
#[derive(Clone, Serialize, Deserialize)]
struct Projection(Vec<(usize, f64)>);

impl Projection {
    /// Project a sample: dot product of weights with feature values.
    #[inline]
    fn project(&self, sample: ArrayView1<f64>) -> f64 {
        self.0.iter().map(|&(j, w)| sample[j] * w).sum()
    }
}

#[derive(Clone, Serialize, Deserialize)]
enum ObliqueNode {
    Leaf(ObliqueLeaf),
    Split {
        projection: Projection,
        threshold: f64,
        left: Box<ObliqueNode>,
        right: Box<ObliqueNode>,
    },
}

#[derive(Clone, Serialize, Deserialize)]
enum ObliqueLeaf {
    Class(usize, Vec<f64>),
    Value(f64),
}

impl ObliqueNode {
    fn predict_one(&self, sample: ArrayView1<f64>) -> &ObliqueLeaf {
        match self {
            ObliqueNode::Leaf(leaf) => leaf,
            ObliqueNode::Split {
                projection,
                threshold,
                left,
                right,
            } => {
                if projection.project(sample) <= *threshold {
                    left.predict_one(sample)
                } else {
                    right.predict_one(sample)
                }
            }
        }
    }
}

// ── Tree builder ────────────────────────────────────────────────────

struct ObliqueTreeBuilder {
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    n_projections: usize,     // projections to try per node
    features_per_proj: usize, // features per projection (sparsity)
    n_features: usize,
    feature_importances: Vec<f64>,
}

impl ObliqueTreeBuilder {
    /// Generate a sparse random projection from {-1, +1}.
    fn random_projection(&self, rng: &mut impl Rng) -> Projection {
        let k = self.features_per_proj.min(self.n_features);
        let mut indices: Vec<usize> = (0..self.n_features).collect();
        // Fisher-Yates partial shuffle for k indices
        for i in 0..k {
            let j = rng.random_range(i..self.n_features);
            indices.swap(i, j);
        }

        let weights: Vec<(usize, f64)> = indices[..k]
            .iter()
            .map(|&idx| {
                let w = if rng.random_range(0..2) == 0 {
                    -1.0
                } else {
                    1.0
                };
                (idx, w)
            })
            .collect();

        Projection(weights)
    }

    fn build_classifier(
        &mut self,
        features: &Array2<f64>,
        target: &[usize],
        indices: &[usize],
        n_classes: usize,
        depth: usize,
        rng: &mut impl Rng,
    ) -> ObliqueNode {
        if indices.len() < self.min_samples_split
            || self.max_depth.is_some_and(|d| depth >= d)
            || all_same_class(target, indices)
        {
            return ObliqueNode::Leaf(classification_leaf(target, indices, n_classes));
        }

        if let Some((proj, threshold, left_idx, right_idx, gain)) =
            self.best_oblique_split_classif(features, target, indices, n_classes, rng)
        {
            if left_idx.len() < self.min_samples_leaf || right_idx.len() < self.min_samples_leaf {
                return ObliqueNode::Leaf(classification_leaf(target, indices, n_classes));
            }

            // Update feature importances for all features in the projection
            for &(feat, _) in &proj.0 {
                self.feature_importances[feat] += gain * indices.len() as f64;
            }

            let left =
                self.build_classifier(features, target, &left_idx, n_classes, depth + 1, rng);
            let right =
                self.build_classifier(features, target, &right_idx, n_classes, depth + 1, rng);

            ObliqueNode::Split {
                projection: proj,
                threshold,
                left: Box::new(left),
                right: Box::new(right),
            }
        } else {
            ObliqueNode::Leaf(classification_leaf(target, indices, n_classes))
        }
    }

    fn build_regressor(
        &mut self,
        features: &Array2<f64>,
        target: &[f64],
        indices: &[usize],
        depth: usize,
        rng: &mut impl Rng,
    ) -> ObliqueNode {
        if indices.len() < self.min_samples_split
            || self.max_depth.is_some_and(|d| depth >= d)
            || all_same_value(target, indices)
        {
            return ObliqueNode::Leaf(regression_leaf(target, indices));
        }

        if let Some((proj, threshold, left_idx, right_idx, gain)) =
            self.best_oblique_split_regress(features, target, indices, rng)
        {
            if left_idx.len() < self.min_samples_leaf || right_idx.len() < self.min_samples_leaf {
                return ObliqueNode::Leaf(regression_leaf(target, indices));
            }

            for &(feat, _) in &proj.0 {
                self.feature_importances[feat] += gain * indices.len() as f64;
            }

            let left = self.build_regressor(features, target, &left_idx, depth + 1, rng);
            let right = self.build_regressor(features, target, &right_idx, depth + 1, rng);

            ObliqueNode::Split {
                projection: proj,
                threshold,
                left: Box::new(left),
                right: Box::new(right),
            }
        } else {
            ObliqueNode::Leaf(regression_leaf(target, indices))
        }
    }

    fn best_oblique_split_classif(
        &self,
        features: &Array2<f64>,
        target: &[usize],
        indices: &[usize],
        n_classes: usize,
        rng: &mut impl Rng,
    ) -> Option<(Projection, f64, Vec<usize>, Vec<usize>, f64)> {
        let parent_gini = gini(target, indices, n_classes);
        let n = indices.len() as f64;
        let mut best_gain = 0.0;
        let mut best = None;

        for _ in 0..self.n_projections {
            let proj = self.random_projection(rng);

            // Project all samples
            let mut projected: Vec<(usize, f64)> = indices
                .iter()
                .map(|&i| (i, proj.project(features.row(i))))
                .collect();
            projected.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Try all split points: sweep left-to-right maintaining running
            // per-class counts, so gini(left)/gini(right) at each candidate
            // threshold is O(n_classes) instead of a full O(n) rescan (which,
            // combined with materializing left/right Vecs per candidate, made
            // this loop O(n²) per projection — audit M-3, same fix as
            // TreeBuilder::best_split_classif). Left/right index Vecs are now
            // only materialized when a candidate improves on the best gain.
            let mut left_counts = vec![0usize; n_classes];
            let mut right_counts = vec![0usize; n_classes];
            for &(idx, _) in &projected {
                right_counts[target[idx]] += 1;
            }

            for s in 1..projected.len() {
                let moved_class = target[projected[s - 1].0];
                left_counts[moved_class] += 1;
                right_counts[moved_class] -= 1;

                if (projected[s].1 - projected[s - 1].1).abs() < f64::EPSILON {
                    continue;
                }

                let n_left = s as f64;
                let n_right = n - n_left;
                let gain = parent_gini
                    - (n_left / n) * gini_from_counts(&left_counts, n_left)
                    - (n_right / n) * gini_from_counts(&right_counts, n_right);

                if gain > best_gain {
                    best_gain = gain;
                    let threshold = (projected[s - 1].1 + projected[s].1) / 2.0;
                    let left_idx: Vec<usize> = projected[..s].iter().map(|(i, _)| *i).collect();
                    let right_idx: Vec<usize> = projected[s..].iter().map(|(i, _)| *i).collect();
                    best = Some((proj.clone(), threshold, left_idx, right_idx, gain));
                }
            }
        }

        best
    }

    fn best_oblique_split_regress(
        &self,
        features: &Array2<f64>,
        target: &[f64],
        indices: &[usize],
        rng: &mut impl Rng,
    ) -> Option<(Projection, f64, Vec<usize>, Vec<usize>, f64)> {
        let parent_mse = mse(target, indices);
        let n = indices.len() as f64;
        // Center the running sums on the node mean — same catastrophic-
        // cancellation guard as TreeBuilder::best_split_regress: E[y²]−E[y]²
        // on raw targets with a large additive offset (UTM coordinates,
        // timestamps) turns split gains into rounding noise.
        let shift = indices.iter().map(|&i| target[i]).sum::<f64>() / n;
        let mut best_gain = 0.0;
        let mut best = None;

        for _ in 0..self.n_projections {
            let proj = self.random_projection(rng);

            let mut projected: Vec<(usize, f64)> = indices
                .iter()
                .map(|&i| (i, proj.project(features.row(i))))
                .collect();
            projected.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Sweep with running sum/sum-of-squares so mse(left)/mse(right)
            // is O(1) per candidate instead of an O(n) rescan (audit M-3,
            // same fix as TreeBuilder::best_split_regress).
            let mut left_sum = 0.0;
            let mut left_sq = 0.0;
            let mut right_sum = 0.0;
            let mut right_sq = 0.0;
            for &(idx, _) in &projected {
                let y = target[idx] - shift;
                right_sum += y;
                right_sq += y * y;
            }

            for s in 1..projected.len() {
                let y = target[projected[s - 1].0] - shift;
                left_sum += y;
                left_sq += y * y;
                right_sum -= y;
                right_sq -= y * y;

                if (projected[s].1 - projected[s - 1].1).abs() < f64::EPSILON {
                    continue;
                }

                let n_left = s as f64;
                let n_right = n - n_left;
                let gain = parent_mse
                    - (n_left / n) * mse_from_sums(left_sum, left_sq, n_left)
                    - (n_right / n) * mse_from_sums(right_sum, right_sq, n_right);

                if gain > best_gain {
                    best_gain = gain;
                    let threshold = (projected[s - 1].1 + projected[s].1) / 2.0;
                    let left_idx: Vec<usize> = projected[..s].iter().map(|(i, _)| *i).collect();
                    let right_idx: Vec<usize> = projected[s..].iter().map(|(i, _)| *i).collect();
                    best = Some((proj.clone(), threshold, left_idx, right_idx, gain));
                }
            }
        }

        best
    }
}

// ── Helper functions ────────────────────────────────────────────────

fn all_same_class(target: &[usize], indices: &[usize]) -> bool {
    let first = target[indices[0]];
    indices.iter().all(|&i| target[i] == first)
}

fn all_same_value(target: &[f64], indices: &[usize]) -> bool {
    let first = target[indices[0]];
    indices
        .iter()
        .all(|&i| (target[i] - first).abs() < f64::EPSILON)
}

fn gini(target: &[usize], indices: &[usize], n_classes: usize) -> f64 {
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

fn mse(target: &[f64], indices: &[usize]) -> f64 {
    let mean = indices.iter().map(|&i| target[i]).sum::<f64>() / indices.len() as f64;
    indices
        .iter()
        .map(|&i| (target[i] - mean).powi(2))
        .sum::<f64>()
        / indices.len() as f64
}

fn classification_leaf(target: &[usize], indices: &[usize], n_classes: usize) -> ObliqueLeaf {
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
    ObliqueLeaf::Class(predicted, probs)
}

fn regression_leaf(target: &[f64], indices: &[usize]) -> ObliqueLeaf {
    let mean = indices.iter().map(|&i| target[i]).sum::<f64>() / indices.len() as f64;
    ObliqueLeaf::Value(mean)
}

// ── Trained models ──────────────────────────────────────────────────

/// A trained oblique (SPORF) decision tree.
#[derive(Clone, Serialize, Deserialize)]
pub struct TrainedObliqueTree {
    root: ObliqueNode,
    feature_names: Vec<String>,
    feature_importances: Vec<f64>,
    n_classes: Option<usize>,
    is_classifier: bool,
}

impl TrainedModel for TrainedObliqueTree {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        crate::validate::check_n_features(features, self.feature_names.len())?;
        predict_from_oblique_roots(
            &[&self.root],
            features,
            self.is_classifier,
            self.n_classes.unwrap_or(0),
        )
    }

    fn feature_importance(&self) -> Option<Vec<(String, f64)>> {
        normalize_importance(&self.feature_names, &self.feature_importances)
    }

    fn to_serializable(&self) -> Option<crate::serialize::SerializableModel> {
        Some(crate::serialize::SerializableModel::ObliqueTree(
            self.clone(),
        ))
    }
}

/// A trained oblique (SPORF) forest.
#[derive(Clone, Serialize, Deserialize)]
pub struct TrainedObliqueForest {
    trees: Vec<ObliqueNode>,
    feature_names: Vec<String>,
    feature_importances: Vec<f64>,
    n_classes: Option<usize>,
    is_classifier: bool,
}

impl TrainedModel for TrainedObliqueForest {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        crate::validate::check_n_features(features, self.feature_names.len())?;
        let refs: Vec<&ObliqueNode> = self.trees.iter().collect();
        predict_from_oblique_roots(
            &refs,
            features,
            self.is_classifier,
            self.n_classes.unwrap_or(0),
        )
    }

    fn feature_importance(&self) -> Option<Vec<(String, f64)>> {
        normalize_importance(&self.feature_names, &self.feature_importances)
    }

    fn to_serializable(&self) -> Option<crate::serialize::SerializableModel> {
        Some(crate::serialize::SerializableModel::ObliqueForest(
            self.clone(),
        ))
    }
}

fn predict_from_oblique_roots(
    trees: &[&ObliqueNode],
    features: &Array2<f64>,
    is_classifier: bool,
    n_classes: usize,
) -> Result<Prediction> {
    let n_trees = trees.len() as f64;

    if is_classifier {
        let mut predicted = Vec::with_capacity(features.nrows());
        let mut probabilities = Vec::with_capacity(features.nrows());

        for row in features.rows() {
            let mut avg_probs = vec![0.0; n_classes];
            for tree in trees {
                if let ObliqueLeaf::Class(_cls, probs) = tree.predict_one(row) {
                    for (j, p) in probs.iter().enumerate() {
                        avg_probs[j] += p;
                    }
                }
            }
            for p in &mut avg_probs {
                *p /= n_trees;
            }
            let cls = avg_probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap()
                .0;
            predicted.push(cls);
            probabilities.push(avg_probs);
        }
        Ok(Prediction::Classification {
            predicted,
            truth: None,
            probabilities: Some(probabilities),
        })
    } else {
        let predicted: Vec<f64> = features
            .rows()
            .into_iter()
            .map(|row| {
                trees
                    .iter()
                    .map(|t| match t.predict_one(row) {
                        ObliqueLeaf::Value(v) => *v,
                        _ => 0.0,
                    })
                    .sum::<f64>()
                    / n_trees
            })
            .collect();
        Ok(Prediction::regression(predicted))
    }
}

fn normalize_importance(names: &[String], importances: &[f64]) -> Option<Vec<(String, f64)>> {
    let total: f64 = importances.iter().sum();
    if total == 0.0 {
        return None;
    }
    Some(
        names
            .iter()
            .zip(importances)
            .map(|(n, &i)| (n.clone(), i / total))
            .collect(),
    )
}

// ── ObliqueTree learner ─────────────────────────────────────────────

/// Single oblique decision tree with sparse random projections.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use ndarray::array;
///
/// let features = array![[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]];
/// let target = vec![1, 1, 0, 0]; // XOR-like: x0+x1 determines class
/// let task = ClassificationTask::new("oblique", features, target).unwrap();
///
/// let mut tree = ObliqueTree::default();
/// let model = tree.train_classif(&task).unwrap();
/// ```
pub struct ObliqueTree {
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    n_projections: usize,
    features_per_proj: Option<usize>, // None = sqrt(n_features)
    seed: u64,
}

impl Default for ObliqueTree {
    fn default() -> Self {
        Self {
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            n_projections: 10,
            features_per_proj: None,
            seed: 42,
        }
    }
}

impl ObliqueTree {
    /// Creates a single oblique decision tree with default hyperparameters.
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets the maximum tree depth.
    pub fn with_max_depth(mut self, d: usize) -> Self {
        self.max_depth = Some(d);
        self
    }
    /// Sets the number of random sparse projections tried at each split.
    pub fn with_n_projections(mut self, n: usize) -> Self {
        self.n_projections = n;
        self
    }
    /// Sets the number of features combined in each sparse projection
    /// (defaults to `sqrt(n_features)` if unset).
    pub fn with_features_per_proj(mut self, k: usize) -> Self {
        self.features_per_proj = Some(k);
        self
    }
    /// Sets the RNG seed used for projection sampling.
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }
}

impl Learner for ObliqueTree {
    fn id(&self) -> &str {
        "oblique_tree"
    }

    fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        crate::validate::check_no_weights(task.weights(), "ObliqueTree")?;
        crate::validate::check_no_nan(task.features())?;
        let nf = task.n_features();
        let fpp = self
            .features_per_proj
            .unwrap_or((nf as f64).sqrt().ceil() as usize);
        let indices: Vec<usize> = (0..task.n_samples()).collect();
        let mut rng = StdRng::seed_from_u64(self.seed);

        let mut builder = ObliqueTreeBuilder {
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            n_projections: self.n_projections,
            features_per_proj: fpp,
            n_features: nf,
            feature_importances: vec![0.0; nf],
        };
        let root = builder.build_classifier(
            task.features(),
            task.target(),
            &indices,
            task.n_classes(),
            0,
            &mut rng,
        );

        Ok(Box::new(TrainedObliqueTree {
            root,
            feature_names: task.feature_names().to_vec(),
            feature_importances: builder.feature_importances,
            n_classes: Some(task.n_classes()),
            is_classifier: true,
        }))
    }

    fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
        crate::validate::check_no_weights(task.weights(), "ObliqueTree")?;
        crate::validate::check_no_nan(task.features())?;
        let nf = task.n_features();
        let fpp = self
            .features_per_proj
            .unwrap_or((nf as f64).sqrt().ceil() as usize);
        let indices: Vec<usize> = (0..task.n_samples()).collect();
        let mut rng = StdRng::seed_from_u64(self.seed);

        let mut builder = ObliqueTreeBuilder {
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            n_projections: self.n_projections,
            features_per_proj: fpp,
            n_features: nf,
            feature_importances: vec![0.0; nf],
        };
        let root = builder.build_regressor(task.features(), task.target(), &indices, 0, &mut rng);

        Ok(Box::new(TrainedObliqueTree {
            root,
            feature_names: task.feature_names().to_vec(),
            feature_importances: builder.feature_importances,
            n_classes: None,
            is_classifier: false,
        }))
    }
}

// ── ObliqueForest learner ───────────────────────────────────────────

/// Oblique Random Forest with Sparse Projections (SPORF).
///
/// Each tree uses random sparse linear combinations of features for
/// splitting instead of single features. Often outperforms standard RF.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use ndarray::array;
///
/// let features = array![[0.0, 0.0], [0.1, 0.1], [1.0, 1.0], [1.1, 0.9]];
/// let target = vec![0, 0, 1, 1];
/// let task = ClassificationTask::new("sporf", features, target).unwrap();
///
/// let mut forest = ObliqueForest::new().with_n_estimators(50).with_seed(42);
/// let model = forest.train_classif(&task).unwrap();
/// ```
pub struct ObliqueForest {
    n_estimators: usize,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    n_projections: usize,
    features_per_proj: Option<usize>,
    seed: u64,
}

impl Default for ObliqueForest {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            n_projections: 10,
            features_per_proj: None,
            seed: 42,
        }
    }
}

impl ObliqueForest {
    /// Creates an oblique random forest (SPORF) with default hyperparameters.
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets the number of trees in the forest.
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }
    /// Sets the maximum depth of each tree.
    pub fn with_max_depth(mut self, d: usize) -> Self {
        self.max_depth = Some(d);
        self
    }
    /// Sets the number of random sparse projections tried at each split.
    pub fn with_n_projections(mut self, n: usize) -> Self {
        self.n_projections = n;
        self
    }
    /// Sets the number of features combined in each sparse projection
    /// (defaults to `sqrt(n_features)` if unset).
    pub fn with_features_per_proj(mut self, k: usize) -> Self {
        self.features_per_proj = Some(k);
        self
    }
    /// Sets the RNG seed used for bootstrap sampling and projection sampling.
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }
}

impl Learner for ObliqueForest {
    fn id(&self) -> &str {
        "oblique_forest"
    }

    fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        crate::validate::check_no_weights(task.weights(), "ObliqueForest")?;
        crate::validate::check_no_nan(task.features())?;
        let nf = task.n_features();
        let ns = task.n_samples();
        let nc = task.n_classes();
        let fpp = self
            .features_per_proj
            .unwrap_or((nf as f64).sqrt().ceil() as usize);

        let results: Vec<(ObliqueNode, Vec<f64>)> = (0..self.n_estimators)
            .into_par_iter()
            .map(|i| {
                let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(i as u64));
                // Bootstrap sample
                let indices: Vec<usize> = (0..ns).map(|_| rng.random_range(0..ns)).collect();

                let mut builder = ObliqueTreeBuilder {
                    max_depth: self.max_depth,
                    min_samples_split: self.min_samples_split,
                    min_samples_leaf: self.min_samples_leaf,
                    n_projections: self.n_projections,
                    features_per_proj: fpp,
                    n_features: nf,
                    feature_importances: vec![0.0; nf],
                };
                let root = builder.build_classifier(
                    task.features(),
                    task.target(),
                    &indices,
                    nc,
                    0,
                    &mut rng,
                );
                (root, builder.feature_importances)
            })
            .collect();

        let mut total_imp = vec![0.0; nf];
        let mut trees = Vec::with_capacity(self.n_estimators);
        for (root, imp) in results {
            for (j, v) in imp.iter().enumerate() {
                total_imp[j] += v;
            }
            trees.push(root);
        }

        Ok(Box::new(TrainedObliqueForest {
            trees,
            feature_names: task.feature_names().to_vec(),
            feature_importances: total_imp,
            n_classes: Some(nc),
            is_classifier: true,
        }))
    }

    fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
        crate::validate::check_no_weights(task.weights(), "ObliqueForest")?;
        crate::validate::check_no_nan(task.features())?;
        let nf = task.n_features();
        let ns = task.n_samples();
        let fpp = self
            .features_per_proj
            .unwrap_or((nf as f64).sqrt().ceil() as usize);

        let results: Vec<(ObliqueNode, Vec<f64>)> = (0..self.n_estimators)
            .into_par_iter()
            .map(|i| {
                let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(i as u64));
                let indices: Vec<usize> = (0..ns).map(|_| rng.random_range(0..ns)).collect();

                let mut builder = ObliqueTreeBuilder {
                    max_depth: self.max_depth,
                    min_samples_split: self.min_samples_split,
                    min_samples_leaf: self.min_samples_leaf,
                    n_projections: self.n_projections,
                    features_per_proj: fpp,
                    n_features: nf,
                    feature_importances: vec![0.0; nf],
                };
                let root =
                    builder.build_regressor(task.features(), task.target(), &indices, 0, &mut rng);
                (root, builder.feature_importances)
            })
            .collect();

        let mut total_imp = vec![0.0; nf];
        let mut trees = Vec::with_capacity(self.n_estimators);
        for (root, imp) in results {
            for (j, v) in imp.iter().enumerate() {
                total_imp[j] += v;
            }
            trees.push(root);
        }

        Ok(Box::new(TrainedObliqueForest {
            trees,
            feature_names: task.feature_names().to_vec(),
            feature_importances: total_imp,
            n_classes: None,
            is_classifier: false,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test: `TrainedObliqueTree::predict` used to pass
    /// `feature_names.len()` (the feature count) as `n_classes` — with
    /// n_features != n_classes this either panics (index out of bounds when
    /// n_classes < n_features) or silently returns probability rows of the
    /// wrong width.
    #[test]
    fn predict_uses_n_classes_not_n_features_for_probability_width() {
        // 2 features, 3 classes: n_classes (3) > n_features (2).
        let n = 90;
        let mut feats = Vec::with_capacity(n * 2);
        let mut target = Vec::with_capacity(n);
        for i in 0..n {
            let c = i % 3;
            feats.push(c as f64 + (i as f64) * 1e-3);
            feats.push((c as f64) * 2.0);
            target.push(c);
        }
        let features = Array2::from_shape_vec((n, 2), feats).unwrap();
        let task = ClassificationTask::new("obl", features.clone(), target).unwrap();
        let mut t = ObliqueTree::new().with_max_depth(4);
        let model = t.train_classif(&task).unwrap();
        let Prediction::Classification {
            probabilities: Some(probs),
            ..
        } = model.predict(&features).unwrap()
        else {
            panic!("expected classification with probabilities");
        };
        assert_eq!(probs[0].len(), 3, "probability rows must be n_classes wide, not n_features wide");

        // 5 features, 2 classes: n_classes (2) < n_features (5) — the
        // silent-wrong-length failure mode (no panic, but wrong width).
        let n2 = 40;
        let mut f2 = Vec::new();
        let mut t2v = Vec::new();
        for i in 0..n2 {
            let c = i % 2;
            for j in 0..5 {
                f2.push(c as f64 * (j + 1) as f64 + i as f64 * 1e-3);
            }
            t2v.push(c);
        }
        let features2 = Array2::from_shape_vec((n2, 5), f2).unwrap();
        let task2 = ClassificationTask::new("obl2", features2.clone(), t2v).unwrap();
        let mut t2 = ObliqueTree::new().with_max_depth(3);
        let model2 = t2.train_classif(&task2).unwrap();
        let Prediction::Classification {
            probabilities: Some(probs2),
            ..
        } = model2.predict(&features2).unwrap()
        else {
            panic!("expected classification with probabilities");
        };
        assert_eq!(probs2[0].len(), 2, "probability rows must be n_classes wide, not n_features wide");
    }

    fn test_builder(n_features: usize) -> ObliqueTreeBuilder {
        ObliqueTreeBuilder {
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            n_projections: 4,
            features_per_proj: 1,
            n_features,
            feature_importances: vec![0.0; n_features],
        }
    }

    /// Golden test for the M-3 O(n²) fix (incremental sweep replacing the
    /// per-candidate gini rescan): with a single feature every projection is
    /// ±x, so the obviously separable step data must be split at ±2.5 (the
    /// midpoint between x=2 and x=3, negated when the projection weight is
    /// −1) with perfect (gini=0) children — same golden shape as
    /// `best_split_classif_finds_the_hand_computed_optimal_split` in
    /// `tree/mod.rs`.
    #[test]
    fn oblique_classif_sweep_finds_the_hand_computed_optimal_split() {
        use ndarray::array;
        use rand::rngs::StdRng;

        let features = array![[1.0], [2.0], [2.0], [3.0], [4.0], [4.0]];
        let target = vec![0usize, 0, 0, 1, 1, 1];
        let indices: Vec<usize> = (0..6).collect();
        let builder = test_builder(1);
        let mut rng = StdRng::seed_from_u64(0);

        let (proj, threshold, left, right, gain) = builder
            .best_oblique_split_classif(&features, &target, &indices, 2, &mut rng)
            .expect("an obviously separable dataset must find a split");

        let w = proj.0[0].1;
        assert!(
            (threshold - w * 2.5).abs() < 1e-9,
            "expected the (possibly negated) midpoint between x=2 and x=3, got {threshold} (w={w})"
        );
        assert_eq!(left.len(), 3);
        assert_eq!(right.len(), 3);
        assert!(gain > 0.49, "a perfect split should recover ~all of the parent's gini: {gain}");
    }

    /// Regression test for the M-3 regression-path sweep inheriting the
    /// HIGH-1 guard: accumulating E[y²]−E[y]² on raw targets with a large
    /// additive offset (UTM northing ~7e6, timestamps ~1e9) cancels
    /// catastrophically, so the sums must be centered on the node mean and
    /// the found split must be independent of the offset. The projection
    /// stream only consumes RNG on features, so with a fixed seed the same
    /// projections are tried for every offset and the thresholds must match
    /// exactly.
    #[test]
    fn oblique_regress_sweep_is_invariant_to_large_target_offsets() {
        use rand::rngs::StdRng;

        let n = 200;
        let features = Array2::from_shape_fn((n, 1), |(i, _)| i as f64 / n as f64 * 10.0);
        let base: Vec<f64> = (0..n)
            .map(|i| {
                let x = features[[i, 0]];
                let step = if x < 5.0 { 0.0 } else { 4.0 };
                step + 0.3 * ((i as f64 * 12.9898).sin())
            })
            .collect();
        let indices: Vec<usize> = (0..n).collect();
        let builder = test_builder(1);

        // The +x and −x projections find mirror-image splits with equal gain
        // (same partition, negated threshold), and which of the two exact
        // ties wins can flip with the offset's last-ulp noise — so compare
        // the offset-invariant quantities: |threshold| and the induced
        // partition (as an unordered pair of index sets).
        let mut splits = Vec::new();
        for offset in [0.0, 1e6, 1e8] {
            let target: Vec<f64> = base.iter().map(|y| y + offset).collect();
            let mut rng = StdRng::seed_from_u64(0);
            let (_, threshold, left, right, _) = builder
                .best_oblique_split_regress(&features, &target, &indices, &mut rng)
                .expect("an obvious step function must find a split");
            let mut left = left;
            let mut right = right;
            left.sort_unstable();
            right.sort_unstable();
            let mut partition = [left, right];
            partition.sort();
            splits.push((threshold.abs(), partition));
        }
        assert_eq!(
            splits[0], splits[1],
            "offset 1e6 changed the chosen split (|threshold| or partition)"
        );
        assert_eq!(
            splits[0], splits[2],
            "offset 1e8 changed the chosen split (|threshold| or partition)"
        );
    }
}
