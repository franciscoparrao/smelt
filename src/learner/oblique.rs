//! Oblique Decision Trees and Forest (SPORF).
//!
//! Splits on linear combinations of features instead of single features.
//! Uses Sparse Projection Oblique Randomer Forest (SPORF) approach:
//! sparse random projections from {-1, +1} weights.
//!
//! Reference: Tomita et al. (2020) "Sparse Projection Oblique Randomer Forests"
//! JMLR 21(104):1-39.

use crate::Result;
use crate::learner::{Learner, TrainedModel};
use crate::prediction::Prediction;
use crate::task::{ClassificationTask, RegressionTask, Task};
use ndarray::{Array2, ArrayView1};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;

// ── Oblique tree node ───────────────────────────────────────────────

/// A projection: linear combination of features.
/// Stored as sparse (feature_index, weight) pairs.
#[derive(Clone)]
struct Projection(Vec<(usize, f64)>);

impl Projection {
    /// Project a sample: dot product of weights with feature values.
    #[inline]
    fn project(&self, sample: ArrayView1<f64>) -> f64 {
        self.0.iter().map(|&(j, w)| sample[j] * w).sum()
    }
}

enum ObliqueNode {
    Leaf(ObliqueLeaf),
    Split {
        projection: Projection,
        threshold: f64,
        left: Box<ObliqueNode>,
        right: Box<ObliqueNode>,
    },
}

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

            // Try all split points
            for s in 1..projected.len() {
                if (projected[s].1 - projected[s - 1].1).abs() < f64::EPSILON {
                    continue;
                }

                let left_idx: Vec<usize> = projected[..s].iter().map(|(i, _)| *i).collect();
                let right_idx: Vec<usize> = projected[s..].iter().map(|(i, _)| *i).collect();

                let gain = parent_gini
                    - (left_idx.len() as f64 / n) * gini(target, &left_idx, n_classes)
                    - (right_idx.len() as f64 / n) * gini(target, &right_idx, n_classes);

                if gain > best_gain {
                    best_gain = gain;
                    let threshold = (projected[s - 1].1 + projected[s].1) / 2.0;
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
        let mut best_gain = 0.0;
        let mut best = None;

        for _ in 0..self.n_projections {
            let proj = self.random_projection(rng);

            let mut projected: Vec<(usize, f64)> = indices
                .iter()
                .map(|&i| (i, proj.project(features.row(i))))
                .collect();
            projected.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            for s in 1..projected.len() {
                if (projected[s].1 - projected[s - 1].1).abs() < f64::EPSILON {
                    continue;
                }

                let left_idx: Vec<usize> = projected[..s].iter().map(|(i, _)| *i).collect();
                let right_idx: Vec<usize> = projected[s..].iter().map(|(i, _)| *i).collect();

                let gain = parent_mse
                    - (left_idx.len() as f64 / n) * mse(target, &left_idx)
                    - (right_idx.len() as f64 / n) * mse(target, &right_idx);

                if gain > best_gain {
                    best_gain = gain;
                    let threshold = (projected[s - 1].1 + projected[s].1) / 2.0;
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

struct TrainedObliqueTree {
    root: ObliqueNode,
    feature_names: Vec<String>,
    feature_importances: Vec<f64>,
    is_classifier: bool,
}

impl TrainedModel for TrainedObliqueTree {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        crate::validate::check_n_features(features, self.feature_names.len())?;
        predict_from_oblique_roots(
            &[&self.root],
            features,
            self.is_classifier,
            self.feature_names.len(),
        )
    }

    fn feature_importance(&self) -> Option<Vec<(String, f64)>> {
        normalize_importance(&self.feature_names, &self.feature_importances)
    }
}

struct TrainedObliqueForest {
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
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_max_depth(mut self, d: usize) -> Self {
        self.max_depth = Some(d);
        self
    }
    pub fn with_n_projections(mut self, n: usize) -> Self {
        self.n_projections = n;
        self
    }
    pub fn with_features_per_proj(mut self, k: usize) -> Self {
        self.features_per_proj = Some(k);
        self
    }
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
            is_classifier: true,
        }))
    }

    fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
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
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }
    pub fn with_max_depth(mut self, d: usize) -> Self {
        self.max_depth = Some(d);
        self
    }
    pub fn with_n_projections(mut self, n: usize) -> Self {
        self.n_projections = n;
        self
    }
    pub fn with_features_per_proj(mut self, k: usize) -> Self {
        self.features_per_proj = Some(k);
        self
    }
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
