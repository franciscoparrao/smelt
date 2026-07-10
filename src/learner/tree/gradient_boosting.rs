//! Gradient Boosting: sequential ensemble of shallow regression trees.
//!
//! Regression uses MSE loss (residual fitting). Classification uses log-loss
//! with sigmoid (binary) or softmax (multiclass).

use super::{LeafValue, Node, TreeBuilder};
use crate::Result;
use crate::learner::math::{sigmoid, softmax};
use crate::learner::{Learner, TrainedModel};
use crate::prediction::Prediction;
use crate::task::{ClassificationTask, RegressionTask, Task};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

/// Gradient Boosting learner.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use ndarray::array;
///
/// let features = array![[1.0], [2.0], [3.0], [4.0]];
/// let target = vec![2.0, 4.0, 6.0, 8.0];
/// let task = RegressionTask::new("gb_demo", features, target).unwrap();
///
/// let mut gb = GradientBoosting::new()
///     .with_n_estimators(50)
///     .with_learning_rate(0.1);
/// let model = gb.train_regress(&task).unwrap();
/// ```
pub struct GradientBoosting {
    n_estimators: usize,
    learning_rate: f64,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    subsample: f64,
    seed: u64,
}

impl Default for GradientBoosting {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: Some(3),
            min_samples_split: 2,
            min_samples_leaf: 1,
            subsample: 1.0,
            seed: 42,
        }
    }
}

impl GradientBoosting {
    /// Creates a Gradient Boosting learner with default hyperparameters (100 trees, learning rate 0.1, max depth 3).
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the number of boosting stages (trees).
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }

    /// Sets the shrinkage factor applied to each tree's contribution.
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Sets the maximum depth of each individual tree.
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }

    /// Sets the minimum number of samples required to split an internal node.
    pub fn with_min_samples_split(mut self, n: usize) -> Self {
        self.min_samples_split = n;
        self
    }

    /// Sets the minimum number of samples required in each leaf.
    pub fn with_min_samples_leaf(mut self, n: usize) -> Self {
        self.min_samples_leaf = n;
        self
    }

    /// Sets the fraction of samples randomly drawn (without replacement) to fit each tree.
    pub fn with_subsample(mut self, ratio: f64) -> Self {
        self.subsample = ratio;
        self
    }

    /// Sets the RNG seed used for subsampling.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    fn subsample_indices(&self, n_samples: usize, rng: &mut StdRng) -> Vec<usize> {
        if self.subsample >= 1.0 {
            return (0..n_samples).collect();
        }
        let k = (n_samples as f64 * self.subsample).ceil() as usize;
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(rng);
        indices.truncate(k);
        indices
    }
}

// --- Internal mode ---

use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub(crate) enum GBMode {
    Regression,
    BinaryClassif,
    MultiClassif { n_classes: usize },
}

/// A trained Gradient Boosting ensemble, ready to predict.
#[derive(Clone, Serialize, Deserialize)]
pub struct TrainedGradientBoosting {
    pub(crate) initial: Vec<f64>,
    pub(crate) trees: Vec<Node>,
    pub(crate) learning_rate: f64,
    pub(crate) feature_names: Vec<String>,
    pub(crate) feature_importances: Vec<f64>,
    pub(crate) mode: GBMode,
}

impl TrainedModel for TrainedGradientBoosting {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        crate::validate::check_n_features(features, self.feature_names.len())?;
        match &self.mode {
            GBMode::Regression => {
                let predicted: Vec<f64> = features
                    .rows()
                    .into_iter()
                    .map(|row| {
                        let mut val = self.initial[0];
                        for tree in &self.trees {
                            if let LeafValue::Value(v) = tree.predict_one(row) {
                                val += self.learning_rate * v;
                            }
                        }
                        val
                    })
                    .collect();
                Ok(Prediction::regression(predicted))
            }
            GBMode::BinaryClassif => {
                let mut predicted = Vec::with_capacity(features.nrows());
                let mut probabilities = Vec::with_capacity(features.nrows());

                for row in features.rows() {
                    let mut f = self.initial[0];
                    for tree in &self.trees {
                        if let LeafValue::Value(v) = tree.predict_one(row) {
                            f += self.learning_rate * v;
                        }
                    }
                    let prob = sigmoid(f);
                    predicted.push(if prob >= 0.5 { 1 } else { 0 });
                    probabilities.push(vec![1.0 - prob, prob]);
                }

                Ok(Prediction::Classification {
                    predicted,
                    truth: None,
                    probabilities: Some(probabilities),
                })
            }
            GBMode::MultiClassif { n_classes } => {
                let k = *n_classes;
                let mut predicted = Vec::with_capacity(features.nrows());
                let mut probabilities = Vec::with_capacity(features.nrows());

                for row in features.rows() {
                    let mut scores = self.initial.clone();
                    // trees are stored flat: iteration i, class c => trees[i * k + c]
                    let n_iters = self.trees.len() / k;
                    for iter in 0..n_iters {
                        for c in 0..k {
                            if let LeafValue::Value(v) = self.trees[iter * k + c].predict_one(row) {
                                scores[c] += self.learning_rate * v;
                            }
                        }
                    }
                    let probs = softmax(&scores);
                    let pred_class = probs
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap()
                        .0;
                    predicted.push(pred_class);
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

    fn feature_importance(&self) -> Option<Vec<(String, f64)>> {
        let total: f64 = self.feature_importances.iter().sum();
        if total == 0.0 {
            return None;
        }
        Some(
            self.feature_names
                .iter()
                .zip(&self.feature_importances)
                .map(|(name, &imp)| (name.clone(), imp / total))
                .collect(),
        )
    }

    fn to_serializable(&self) -> Option<crate::serialize::SerializableModel> {
        Some(crate::serialize::SerializableModel::GradientBoosting(
            self.clone(),
        ))
    }
}

impl Learner for GradientBoosting {
    fn id(&self) -> &str {
        "gradient_boosting"
    }

    fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        crate::validate::check_no_nan(task.features())?;
        let n_classes = task.n_classes();
        if n_classes == 2 {
            self.train_binary(task)
        } else {
            self.train_multiclass(task)
        }
    }

    fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
        crate::validate::check_no_nan(task.features())?;
        let features = task.features();
        let target = task.target();
        let n_samples = task.n_samples();
        let n_features = task.n_features();

        let initial = target.iter().sum::<f64>() / n_samples as f64;
        let mut current_preds = vec![initial; n_samples];
        let mut trees = Vec::with_capacity(self.n_estimators);
        let mut total_importances = vec![0.0; n_features];
        let mut rng = StdRng::seed_from_u64(self.seed);

        for _ in 0..self.n_estimators {
            let residuals: Vec<f64> = target
                .iter()
                .zip(&current_preds)
                .map(|(y, p)| y - p)
                .collect();

            let indices = self.subsample_indices(n_samples, &mut rng);

            let mut builder = TreeBuilder::new(
                self.max_depth,
                self.min_samples_split,
                self.min_samples_leaf,
                None,
                n_features,
            );
            let root = builder.build_regressor(&features.view(), &residuals, &indices, 0, &mut rng);

            // Update predictions
            for i in 0..n_samples {
                if let LeafValue::Value(v) = root.predict_one(features.row(i)) {
                    current_preds[i] += self.learning_rate * v;
                }
            }

            for (j, imp) in builder.feature_importances.iter().enumerate() {
                total_importances[j] += imp;
            }
            trees.push(root);
        }

        Ok(Box::new(TrainedGradientBoosting {
            initial: vec![initial],
            trees,
            learning_rate: self.learning_rate,
            feature_names: task.feature_names().to_vec(),
            feature_importances: total_importances,
            mode: GBMode::Regression,
        }))
    }
}

/// Refits every leaf's value via one Newton-Raphson step
/// (`sum(grad) / sum(hess)` over the samples routed to that leaf), instead
/// of leaving the plain mean-of-residuals a squared-error-minimizing
/// `TreeBuilder` gives each leaf by construction (audit issue M7). The tree
/// *structure* (which splits separate the data well) is still chosen by the
/// SSE-based builder against the pseudo-residuals -- only the leaf *values*
/// are corrected here to account for the loss's true curvature (Friedman
/// 2001, sec. 4.6). For squared-error regression this is a no-op (hessian
/// is constant 1 everywhere, so the Newton step degenerates to the same
/// mean the tree already computed); it only changes behavior for the
/// classification losses (log-loss / softmax), whose hessian varies per
/// sample with the current prediction.
fn refit_leaf_newton(
    node: &mut Node,
    features: &Array2<f64>,
    indices: &[usize],
    grads: &[f64],
    hess: &[f64],
) {
    match node {
        Node::Leaf(LeafValue::Value(v)) => {
            let g: f64 = indices.iter().map(|&i| grads[i]).sum();
            let h: f64 = indices.iter().map(|&i| hess[i]).sum();
            if h > 1e-12 {
                *v = g / h;
            }
            // Else: leave the SSE-based mean-residual value in place. A
            // leaf with near-zero total hessian has no reliable Newton
            // step; TreeBuilder already enforces `min_samples_leaf`, so
            // this is only reachable via near-degenerate per-sample
            // hessians (e.g. predictions already extremely confident),
            // not the common path.
        }
        Node::Leaf(LeafValue::Class(..)) => {} // not produced by build_regressor
        Node::Split { feature, threshold, left, right } => {
            let (mut left_idx, mut right_idx) = (Vec::new(), Vec::new());
            for &i in indices {
                if features[[i, *feature]] <= *threshold {
                    left_idx.push(i);
                } else {
                    right_idx.push(i);
                }
            }
            refit_leaf_newton(left, features, &left_idx, grads, hess);
            refit_leaf_newton(right, features, &right_idx, grads, hess);
        }
    }
}

impl GradientBoosting {
    fn train_binary(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        let features = task.features();
        let target = task.target();
        let n_samples = task.n_samples();
        let n_features = task.n_features();

        // Initial log-odds
        let p_pos = target.iter().filter(|&&t| t == 1).count() as f64 / n_samples as f64;
        let initial = (p_pos / (1.0 - p_pos).max(1e-15)).ln();
        let mut current_f = vec![initial; n_samples];
        let mut trees = Vec::with_capacity(self.n_estimators);
        let mut total_importances = vec![0.0; n_features];
        let mut rng = StdRng::seed_from_u64(self.seed);

        for _ in 0..self.n_estimators {
            // Pseudo-residuals: y - sigmoid(F)
            let residuals: Vec<f64> = target
                .iter()
                .zip(&current_f)
                .map(|(&y, &f)| y as f64 - sigmoid(f))
                .collect();

            let indices = self.subsample_indices(n_samples, &mut rng);

            let mut builder = TreeBuilder::new(
                self.max_depth,
                self.min_samples_split,
                self.min_samples_leaf,
                None,
                n_features,
            );
            let mut root =
                builder.build_regressor(&features.view(), &residuals, &indices, 0, &mut rng);

            // Newton step (audit issue M7): correct each leaf's value using
            // the log-loss hessian p*(1-p) at the *current* ensemble
            // prediction, evaluated on the same subsample the tree was
            // built on.
            let hess: Vec<f64> = current_f
                .iter()
                .map(|&f| {
                    let p = sigmoid(f);
                    (p * (1.0 - p)).max(1e-15)
                })
                .collect();
            refit_leaf_newton(&mut root, features, &indices, &residuals, &hess);

            for i in 0..n_samples {
                if let LeafValue::Value(v) = root.predict_one(features.row(i)) {
                    current_f[i] += self.learning_rate * v;
                }
            }

            for (j, imp) in builder.feature_importances.iter().enumerate() {
                total_importances[j] += imp;
            }
            trees.push(root);
        }

        Ok(Box::new(TrainedGradientBoosting {
            initial: vec![initial],
            trees,
            learning_rate: self.learning_rate,
            feature_names: task.feature_names().to_vec(),
            feature_importances: total_importances,
            mode: GBMode::BinaryClassif,
        }))
    }

    fn train_multiclass(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        let features = task.features();
        let target = task.target();
        let n_samples = task.n_samples();
        let n_features = task.n_features();
        let n_classes = task.n_classes();

        // Initial log-proportions per class
        let mut class_counts = vec![0usize; n_classes];
        for &t in target {
            class_counts[t] += 1;
        }
        let initial: Vec<f64> = class_counts
            .iter()
            .map(|&c| ((c as f64 / n_samples as f64).max(1e-15)).ln())
            .collect();

        // Current raw scores: [sample][class]
        let mut current_f: Vec<Vec<f64>> = (0..n_samples).map(|_| initial.clone()).collect();

        let mut trees = Vec::with_capacity(self.n_estimators * n_classes);
        let mut total_importances = vec![0.0; n_features];
        let mut rng = StdRng::seed_from_u64(self.seed);

        for _ in 0..self.n_estimators {
            // Compute softmax probabilities for all samples
            let probs: Vec<Vec<f64>> = current_f.iter().map(|f| softmax(f)).collect();

            let indices = self.subsample_indices(n_samples, &mut rng);

            // One tree per class
            for c in 0..n_classes {
                // Pseudo-residuals: y_ic - p_ic
                let residuals: Vec<f64> = (0..n_samples)
                    .map(|i| {
                        let y_ic = if target[i] == c { 1.0 } else { 0.0 };
                        y_ic - probs[i][c]
                    })
                    .collect();

                let mut builder = TreeBuilder::new(
                    self.max_depth,
                    self.min_samples_split,
                    self.min_samples_leaf,
                    None,
                    n_features,
                );
                let mut root =
                    builder.build_regressor(&features.view(), &residuals, &indices, 0, &mut rng);

                // Newton step (audit issue M7): diagonal softmax hessian
                // p_ic*(1-p_ic), the same per-class approximation used
                // elsewhere for multiclass boosting (e.g. CatBoost's
                // `train_multiclass`).
                let hess: Vec<f64> =
                    probs.iter().map(|p| (p[c] * (1.0 - p[c])).max(1e-15)).collect();
                refit_leaf_newton(&mut root, features, &indices, &residuals, &hess);

                // Update scores for this class
                for i in 0..n_samples {
                    if let LeafValue::Value(v) = root.predict_one(features.row(i)) {
                        current_f[i][c] += self.learning_rate * v;
                    }
                }

                for (j, imp) in builder.feature_importances.iter().enumerate() {
                    total_importances[j] += imp;
                }
                trees.push(root);
            }
        }

        Ok(Box::new(TrainedGradientBoosting {
            initial,
            trees,
            learning_rate: self.learning_rate,
            feature_names: task.feature_names().to_vec(),
            feature_importances: total_importances,
            mode: GBMode::MultiClassif { n_classes },
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for M7 (docs/auditoria_motor_2026-07-05.md, Fase F):
    /// `refit_leaf_newton` must replace each leaf's SSE-mean value with
    /// `sum(grad) / sum(hess)` over the samples actually routed there --
    /// not just leave whatever the squared-error-minimizing `TreeBuilder`
    /// put there. A hand-built 2-leaf tree with very different per-sample
    /// hessians on each side makes the two computations clearly diverge.
    #[test]
    fn refit_leaf_newton_uses_gradient_over_hessian_per_leaf() {
        let features = Array2::from_shape_vec((4, 1), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let mut tree = Node::Split {
            feature: 0,
            threshold: 0.5,
            left: Box::new(Node::Leaf(LeafValue::Value(999.0))), // stand-in SSE value
            right: Box::new(Node::Leaf(LeafValue::Value(999.0))),
        };
        let indices = vec![0, 1, 2, 3];
        // Left leaf (samples 0,1): grads sum=3.0, hess sum=6.0 -> 0.5
        // Right leaf (samples 2,3): grads sum=1.0, hess sum=0.5 -> 2.0
        let grads = vec![1.0, 2.0, 0.5, 0.5];
        let hess = vec![2.0, 4.0, 0.25, 0.25];

        refit_leaf_newton(&mut tree, &features, &indices, &grads, &hess);

        let Node::Split { left, right, .. } = &tree else {
            panic!("expected split");
        };
        let Node::Leaf(LeafValue::Value(lv)) = **left else {
            panic!("expected leaf")
        };
        let Node::Leaf(LeafValue::Value(rv)) = **right else {
            panic!("expected leaf")
        };
        assert!((lv - 0.5).abs() < 1e-12, "left leaf should be sum(grad)/sum(hess)=0.5, got {lv}");
        assert!((rv - 2.0).abs() < 1e-12, "right leaf should be sum(grad)/sum(hess)=2.0, got {rv}");
    }

    /// A leaf whose routed samples have (near-)zero total hessian has no
    /// reliable Newton step; the pre-existing SSE-mean value must be left
    /// untouched rather than blowing up via division by ~0.
    #[test]
    fn refit_leaf_newton_leaves_zero_hessian_leaf_unchanged() {
        let features = Array2::from_shape_vec((2, 1), vec![0.0, 0.0]).unwrap();
        let mut tree = Node::Leaf(LeafValue::Value(0.42));
        refit_leaf_newton(&mut tree, &features, &[0, 1], &[1.0, 1.0], &[0.0, 0.0]);
        let Node::Leaf(LeafValue::Value(v)) = tree else {
            panic!("expected leaf")
        };
        assert!((v - 0.42).abs() < 1e-12, "zero-hessian leaf must keep its prior value, got {v}");
    }
}
