//! Deep Forest (gcForest): a cascade of forest layers that progressively
//! augments the original features with each layer's own out-of-fold class
//! probabilities, growing (and stopping) depth the way a neural network's
//! layers do -- but with forests as the "neurons" instead of trained
//! weight matrices, and no backpropagation.
//!
//! Zhou, Z.-H., & Feng, J. (2017). "Deep Forest: Towards An Alternative to
//! Deep Neural Networks." IJCAI 2017.
//!
//! Classification only, and scoped to the "cascade forest" half of the
//! paper (not the "multi-grained scanning" step, which targets structured
//! image/sequence inputs by design -- out of scope for the tabular data
//! this crate otherwise targets).
//!
//! # How a layer is built
//!
//! Each layer trains `2 * n_forests_per_type` forests (alternating
//! `RandomForest`/`ExtraTrees`, matching the paper's "two completely-random
//! tree forests + two random forests" convention) on the CURRENT layer's
//! input (original features, augmented with every previous layer's
//! out-of-fold probabilities). Each forest's contribution to that
//! augmentation is itself produced via internal k-fold cross-validation --
//! not the forest's own in-sample predictions -- so a forest that
//! overfits its own layer's input doesn't get to fabricate falsely
//! confident features for the next layer. The same k-fold's average
//! accuracy also decides whether the cascade keeps growing: after
//! [`DeepForest::with_early_stopping_rounds`] consecutive layers without
//! improvement, the cascade stops and is truncated back to its best-so-far
//! depth.

use crate::learner::tree::extra_trees::ExtraTrees;
use crate::learner::tree::random_forest::RandomForest;
use crate::learner::{Learner, LearnerProperties, TrainedModel};
use crate::prediction::Prediction;
use crate::resample::{CrossValidation, Resample};
use crate::task::{ClassificationTask, Task};
use crate::Result;
use ndarray::{s, Array2, Axis};

fn argmax_row(probs: &Array2<f64>, i: usize) -> usize {
    probs
        .row(i)
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(j, _)| j)
        .unwrap_or(0)
}

fn accuracy_from_probs(probs: &Array2<f64>, target: &[usize]) -> f64 {
    let correct = (0..target.len()).filter(|&i| argmax_row(probs, i) == target[i]).count();
    correct as f64 / target.len().max(1) as f64
}

fn average_arrays(arrays: &[Array2<f64>]) -> Array2<f64> {
    let mut sum = arrays[0].clone();
    for arr in &arrays[1..] {
        sum += arr;
    }
    sum / arrays.len() as f64
}

/// Horizontally concatenates `original` (n x p0) with each forest's
/// probability array (n x n_classes) in `probs_per_forest`, building the
/// next layer's augmented input.
fn concat_features(original: &Array2<f64>, probs_per_forest: &[Array2<f64>]) -> Array2<f64> {
    let n = original.nrows();
    let p0 = original.ncols();
    let extra_cols: usize = probs_per_forest.iter().map(|a| a.ncols()).sum();
    let mut out = Array2::zeros((n, p0 + extra_cols));
    out.slice_mut(s![.., 0..p0]).assign(original);
    let mut offset = p0;
    for arr in probs_per_forest {
        let w = arr.ncols();
        out.slice_mut(s![.., offset..offset + w]).assign(arr);
        offset += w;
    }
    out
}

fn new_forest(is_random_forest: bool, n_estimators: usize, max_depth: usize, seed: u64) -> Box<dyn Learner> {
    if is_random_forest {
        Box::new(
            RandomForest::new()
                .with_n_estimators(n_estimators)
                .with_max_depth(max_depth)
                .with_seed(seed),
        )
    } else {
        Box::new(
            ExtraTrees::new()
                .with_n_estimators(n_estimators)
                .with_max_depth(max_depth)
                .with_seed(seed),
        )
    }
}

fn probabilities_array(model: &dyn TrainedModel, features: &Array2<f64>, n_classes: usize) -> Result<Array2<f64>> {
    let n = features.nrows();
    let pred = model.predict(features)?;
    let mut out = Array2::zeros((n, n_classes));
    if let Prediction::Classification {
        probabilities: Some(probs),
        ..
    } = pred
    {
        for (i, row) in probs.iter().enumerate() {
            for (c, &p) in row.iter().enumerate() {
                if c < n_classes {
                    out[[i, c]] = p;
                }
            }
        }
    }
    Ok(out)
}

/// Deep Forest (gcForest) classifier: a cascade of forest layers.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use ndarray::array;
///
/// let features = array![
///     [0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [0.0, 0.2],
///     [1.0, 1.0], [1.1, 0.9], [0.9, 1.1], [1.0, 0.9],
///     [0.05, 0.15], [0.15, 0.05], [1.05, 0.95], [0.95, 1.05],
/// ];
/// let target = vec![0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1];
/// let task = ClassificationTask::new("gcforest", features, target).unwrap();
///
/// let mut df = DeepForest::new().with_n_estimators_per_forest(20).with_seed(1);
/// let model = df.train_classif(&task).unwrap();
/// ```
pub struct DeepForest {
    n_forests_per_type: usize,
    n_estimators_per_forest: usize,
    max_depth: usize,
    cv_folds: usize,
    max_layers: usize,
    early_stopping_rounds: usize,
    seed: u64,
}

impl Default for DeepForest {
    fn default() -> Self {
        Self::new()
    }
}

impl DeepForest {
    /// Creates a Deep Forest with the paper's default cascade shape: 2
    /// `RandomForest`s + 2 `ExtraTrees` per layer (100 trees each, depth
    /// 10), 3-fold internal CV, up to 10 layers with early stopping after 2
    /// consecutive layers without improvement.
    pub fn new() -> Self {
        Self {
            n_forests_per_type: 2,
            n_estimators_per_forest: 100,
            max_depth: 10,
            cv_folds: 3,
            max_layers: 10,
            early_stopping_rounds: 2,
            seed: 42,
        }
    }
    /// Sets how many forests of EACH type (`RandomForest`/`ExtraTrees`) each
    /// layer trains -- total forests per layer is twice this.
    pub fn with_n_forests_per_type(mut self, n: usize) -> Self {
        self.n_forests_per_type = n.max(1);
        self
    }
    /// Sets the number of trees in each member forest.
    pub fn with_n_estimators_per_forest(mut self, n: usize) -> Self {
        self.n_estimators_per_forest = n.max(1);
        self
    }
    /// Sets each member forest's maximum tree depth.
    pub fn with_max_depth(mut self, d: usize) -> Self {
        self.max_depth = d;
        self
    }
    /// Sets the number of folds used to produce each layer's honest
    /// out-of-fold class probabilities.
    pub fn with_cv_folds(mut self, k: usize) -> Self {
        self.cv_folds = k.max(2);
        self
    }
    /// Sets the maximum number of cascade layers.
    pub fn with_max_layers(mut self, n: usize) -> Self {
        self.max_layers = n.max(1);
        self
    }
    /// Sets how many consecutive layers without improvement in the
    /// out-of-fold accuracy trigger stopping (and truncating back to the
    /// best-so-far depth).
    pub fn with_early_stopping_rounds(mut self, n: usize) -> Self {
        self.early_stopping_rounds = n.max(1);
        self
    }
    /// Sets the RNG seed controlling both the internal CV splits and each
    /// member forest's own randomness.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

struct Layer {
    forests: Vec<Box<dyn TrainedModel>>,
}

impl DeepForest {
    /// Trains the cascade and returns the concrete [`TrainedDeepForest`]
    /// (rather than a boxed [`TrainedModel`]) -- lets callers inspect
    /// [`TrainedDeepForest::n_layers`] to confirm early stopping actually
    /// truncated the cascade, the same "concrete type beyond the trait"
    /// shape as `TrainedKrigingHybrid`/`TrainedGeoXGBoost`.
    pub fn fit(&mut self, task: &ClassificationTask) -> Result<TrainedDeepForest> {
        // Guard here (not in `Learner::train_classif`, which just boxes this)
        // so BOTH public entry points reject weighted tasks.
        crate::validate::check_no_weights(task.weights(), "DeepForest")?;
        crate::validate::check_no_nan(task.features())?;
        let original_features = task.features().clone();
        let target = task.target().to_vec();
        let n_classes = task.n_classes();
        let n_samples = task.n_samples();
        let n_forests = 2 * self.n_forests_per_type;

        let mut current_input = original_features.clone();
        let mut kept_layers: Vec<Layer> = Vec::new();
        let mut best_acc = f64::NEG_INFINITY;
        let mut best_layer_count = 0usize;
        let mut rounds_without_improvement = 0usize;

        for layer_idx in 0..self.max_layers {
            let cv = CrossValidation::new(self.cv_folds).with_seed(self.seed.wrapping_add(layer_idx as u64));
            let splits = cv.splits(n_samples)?;

            let mut oof_probs_per_forest: Vec<Array2<f64>> = Vec::with_capacity(n_forests);
            let mut trained_forests: Vec<Box<dyn TrainedModel>> = Vec::with_capacity(n_forests);

            for spec_idx in 0..n_forests {
                let is_random_forest = spec_idx % 2 == 0;
                let forest_seed = self
                    .seed
                    .wrapping_add(1000 * (layer_idx as u64 + 1))
                    .wrapping_add(spec_idx as u64);

                let mut oof = Array2::zeros((n_samples, n_classes));
                for (train_idx, test_idx) in &splits {
                    let train_features = current_input.select(Axis(0), train_idx);
                    let train_target: Vec<usize> = train_idx.iter().map(|&i| target[i]).collect();
                    // Propagate class_names (backlog M-9, closed in the 5th
                    // audit): rebuilding a fold task from scratch re-derives
                    // n_classes as max(label)+1, so a fold missing the
                    // highest class trained a narrower forest whose OOF
                    // probability rows dropped that class's column.
                    let fold_task = ClassificationTask::new("deep_forest_fold", train_features, train_target)?
                        .with_class_names(task.class_names().to_vec());

                    let mut fold_learner = new_forest(is_random_forest, self.n_estimators_per_forest, self.max_depth, forest_seed);
                    let fold_model = fold_learner.train_classif(&fold_task)?;

                    let test_features = current_input.select(Axis(0), test_idx);
                    let test_probs = probabilities_array(&*fold_model, &test_features, n_classes)?;
                    for (row_i, &sample_idx) in test_idx.iter().enumerate() {
                        for c in 0..n_classes {
                            oof[[sample_idx, c]] = test_probs[[row_i, c]];
                        }
                    }
                }
                oof_probs_per_forest.push(oof);

                // Retrain on the FULL current-layer input for deployment.
                let full_task = ClassificationTask::new("deep_forest_layer", current_input.clone(), target.clone())?
                    .with_class_names(task.class_names().to_vec());
                let mut full_learner = new_forest(is_random_forest, self.n_estimators_per_forest, self.max_depth, forest_seed);
                let full_model = full_learner.train_classif(&full_task)?;
                trained_forests.push(full_model);
            }

            let mean_probs = average_arrays(&oof_probs_per_forest);
            let layer_acc = accuracy_from_probs(&mean_probs, &target);

            kept_layers.push(Layer { forests: trained_forests });

            if layer_acc > best_acc + 1e-6 {
                best_acc = layer_acc;
                best_layer_count = kept_layers.len();
                rounds_without_improvement = 0;
            } else {
                rounds_without_improvement += 1;
                if rounds_without_improvement >= self.early_stopping_rounds {
                    break;
                }
            }

            current_input = concat_features(&original_features, &oof_probs_per_forest);
        }

        kept_layers.truncate(best_layer_count.max(1));

        Ok(TrainedDeepForest {
            layers: kept_layers,
            n_classes,
            n_original_features: original_features.ncols(),
        })
    }
}

impl Learner for DeepForest {
    fn id(&self) -> &str {
        "deep_forest"
    }

    fn properties(&self) -> LearnerProperties {
        LearnerProperties::classifier()
            .with_proba()
    }

    fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        Ok(Box::new(self.fit(task)?))
    }
}

/// A trained [`DeepForest`] cascade.
pub struct TrainedDeepForest {
    layers: Vec<Layer>,
    n_classes: usize,
    n_original_features: usize,
}

impl TrainedDeepForest {
    /// Number of cascade layers actually kept (after early-stopping
    /// truncation) -- useful for confirming the cascade stopped early
    /// rather than always growing to `max_layers`.
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }
}

impl TrainedModel for TrainedDeepForest {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        crate::validate::check_n_features(features, self.n_original_features)?;
        let mut current_input = features.clone();
        let mut last_layer_probs: Vec<Array2<f64>> = Vec::new();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let mut probs_per_forest = Vec::with_capacity(layer.forests.len());
            for forest in &layer.forests {
                probs_per_forest.push(probabilities_array(&**forest, &current_input, self.n_classes)?);
            }
            let is_last = layer_idx + 1 == self.layers.len();
            if is_last {
                last_layer_probs = probs_per_forest;
            } else {
                current_input = concat_features(features, &probs_per_forest);
            }
        }

        let mean_probs = average_arrays(&last_layer_probs);
        let n = mean_probs.nrows();
        let predicted: Vec<usize> = (0..n).map(|i| argmax_row(&mean_probs, i)).collect();
        let probabilities: Vec<Vec<f64>> = (0..n).map(|i| mean_probs.row(i).to_vec()).collect();

        Ok(Prediction::Classification {
            predicted,
            truth: None,
            probabilities: Some(probabilities),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn registered_id_matches() {
        assert_eq!(DeepForest::new().id(), "deep_forest");
    }

    #[test]
    fn fits_a_simple_classification_boundary() {
        let mut rng = StdRng::seed_from_u64(1);
        let n = 300;
        let mut feats = Vec::with_capacity(n * 2);
        let mut target = Vec::with_capacity(n);
        for _ in 0..n {
            let x0: f64 = rng.random();
            let x1: f64 = rng.random();
            feats.push(x0);
            feats.push(x1);
            target.push(if x0 + x1 > 1.0 { 1usize } else { 0 });
        }
        let features = Array2::from_shape_vec((n, 2), feats).unwrap();
        let task = ClassificationTask::new("df_simple", features.clone(), target.clone()).unwrap();

        let mut df = DeepForest::new()
            .with_n_estimators_per_forest(20)
            .with_max_layers(3)
            .with_seed(2);
        let model = df.train_classif(&task).unwrap();
        let pred = model.predict(&features).unwrap();
        let Prediction::Classification { predicted, .. } = pred else {
            panic!("expected classification");
        };
        let correct = predicted.iter().zip(&target).filter(|(p, t)| *p == *t).count();
        let acc = correct as f64 / n as f64;
        assert!(acc > 0.85, "should fit a simple boundary well, got acc={acc}");
    }

    /// Regression test (backlog M-9, closed in the 5th audit): fold tasks
    /// (and the full-layer retrain task) are rebuilt from scratch inside
    /// `fit`, which re-derives n_classes as max(label)+1 — without
    /// propagating the parent task's class_names, a CV fold whose training
    /// split lost the rare highest class trained a narrower forest whose
    /// OOF probability column for that class silently degenerated to zero.
    /// With the propagation, every internal task declares the full class
    /// set, so the cascade trains cleanly and every probability row —
    /// including through the multi-layer OOF-augmented path — has exactly
    /// the declared width, with the rare class recoverable at predict time.
    #[test]
    fn rare_class_with_class_names_keeps_full_probability_width() {
        // 3 declared classes; class 2 has a single sample, so with 3 CV
        // folds the fold holding it in TEST trains without class 2 at all —
        // exactly the narrowing scenario class_names propagation prevents.
        let features = array![
            [0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [0.0, 0.2], [0.1, 0.0],
            [5.0, 5.0], [5.1, 4.9], [4.9, 5.1], [5.0, 4.8], [5.1, 5.1],
            [10.0, 10.0],
        ];
        let target = vec![0usize, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2];
        let task = ClassificationTask::new("df_rare", features.clone(), target.clone())
            .unwrap()
            .with_class_names(vec!["a".into(), "b".into(), "c".into()]);
        assert_eq!(task.n_classes(), 3);

        let mut df = DeepForest::new()
            .with_n_estimators_per_forest(20)
            .with_max_layers(2)
            .with_seed(3);
        let model = df.fit(&task).unwrap();
        let pred = model.predict(&features).unwrap();
        let Prediction::Classification { predicted, probabilities: Some(probs), .. } = pred else {
            panic!("expected classification with probabilities");
        };
        for row in &probs {
            assert_eq!(row.len(), 3, "every probability row must span the declared 3 classes");
        }
        // The rare class is trivially separable (far from both clusters), so
        // the deployed cascade must actually recover it.
        assert_eq!(predicted[10], 2, "the rare class must be predictable, got {predicted:?}");
    }

    #[test]
    fn multiclass_works() {
        let features = array![
            [0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [0.0, 0.2],
            [5.0, 5.0], [5.1, 4.9], [4.9, 5.1], [5.0, 4.8],
            [10.0, 0.0], [10.1, 0.1], [9.9, -0.1], [10.0, 0.2],
        ];
        let target = vec![0usize, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
        let task = ClassificationTask::new("df_multi", features.clone(), target.clone()).unwrap();

        let mut df = DeepForest::new()
            .with_n_estimators_per_forest(20)
            .with_max_layers(3)
            .with_cv_folds(3)
            .with_seed(3);
        let model = df.train_classif(&task).unwrap();
        let pred = model.predict(&features).unwrap();
        let Prediction::Classification { predicted, probabilities, .. } = pred else {
            panic!("expected classification");
        };
        let correct = predicted.iter().zip(&target).filter(|(p, t)| *p == *t).count();
        assert!(correct as f64 / target.len() as f64 > 0.8, "should separate 3 well-separated clusters");

        for row in &probabilities.unwrap() {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "probabilities should sum to 1, got {sum}");
        }
    }

    /// Early stopping must actually truncate the cascade -- with a strict
    /// `early_stopping_rounds=1` on data simple enough that a single layer
    /// already does about as well as more layers would, the kept depth
    /// should be well short of `max_layers`.
    #[test]
    fn early_stopping_truncates_the_cascade() {
        let mut rng = StdRng::seed_from_u64(4);
        let n = 200;
        let mut feats = Vec::with_capacity(n * 2);
        let mut target = Vec::with_capacity(n);
        for _ in 0..n {
            let x0: f64 = rng.random();
            let x1: f64 = rng.random();
            feats.push(x0);
            feats.push(x1);
            target.push(if x0 > 0.5 { 1usize } else { 0 });
        }
        let features = Array2::from_shape_vec((n, 2), feats).unwrap();
        let task = ClassificationTask::new("df_early_stop", features, target).unwrap();

        let mut df = DeepForest::new()
            .with_n_estimators_per_forest(20)
            .with_max_layers(10)
            .with_early_stopping_rounds(1)
            .with_seed(5);
        let trained = df.fit(&task).unwrap();
        assert!(
            trained.n_layers() < 10,
            "with early_stopping_rounds=1 on data a single layer already handles well, \
             the cascade should stop well short of max_layers=10, got {}",
            trained.n_layers()
        );
    }

    #[test]
    fn rejects_wrong_feature_count_at_predict() {
        let features = array![[0.0, 0.0], [1.0, 1.0], [0.1, 0.1], [0.9, 0.9]];
        let target = vec![0usize, 1, 0, 1];
        let task = ClassificationTask::new("df_dim", features, target).unwrap();
        let mut df = DeepForest::new().with_n_estimators_per_forest(10).with_max_layers(2).with_seed(1);
        let model = df.train_classif(&task).unwrap();

        let wrong = array![[1.0, 2.0, 3.0]];
        assert!(model.predict(&wrong).is_err());
    }
}
