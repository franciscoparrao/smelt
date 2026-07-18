//! Generic bootstrap aggregating (bagging) wrapper for any Learner.

use crate::Result;
use crate::learner::{Learner, TrainedModel};
use crate::prediction::Prediction;
use crate::task::{ClassificationTask, RegressionTask, Task};
use ndarray::{Array2, Axis};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;

/// Generic bagging wrapper that trains multiple copies of a base learner
/// on bootstrap samples and aggregates their predictions.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use ndarray::array;
///
/// let features = array![[0.0, 0.0], [0.1, 0.1], [1.0, 1.0], [1.1, 0.9]];
/// let target = vec![0, 0, 1, 1];
/// let task = ClassificationTask::new("bag_demo", features, target).unwrap();
///
/// let mut bag = Bagging::new(|| Box::new(DecisionTree::default()))
///     .with_n_estimators(10)
///     .with_seed(42);
/// let model = bag.train_classif(&task).unwrap();
/// ```
pub struct Bagging {
    factory: Box<dyn Fn() -> Box<dyn Learner> + Send + Sync>,
    n_estimators: usize,
    seed: u64,
}

impl Bagging {
    /// Creates a bagging ensemble from a factory that produces fresh base
    /// learners, defaulting to 10 estimators and seed 42.
    pub fn new(factory: impl Fn() -> Box<dyn Learner> + Send + Sync + 'static) -> Self {
        Self {
            factory: Box::new(factory),
            n_estimators: 10,
            seed: 42,
        }
    }

    /// Sets the number of bootstrap samples (base learners) to train.
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }

    /// Sets the RNG seed used to draw bootstrap samples.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// `with_n_estimators(0)` used to be accepted and silently produced a
    /// model with zero members, whose aggregation degenerates to a constant
    /// (class 0 / 0.0) prediction (5th audit, LOW-D — the ensemble twin of
    /// the `max_depth=0` validation added in the 4th audit's LOW batch).
    /// The builder doesn't return `Result`, so the check runs at train time.
    fn check_n_estimators(&self) -> Result<()> {
        if self.n_estimators == 0 {
            return Err(crate::SmeltError::InvalidParameter(
                "bagging n_estimators must be at least 1 (got 0): an ensemble with zero \
                 members would silently predict a constant"
                    .into(),
            ));
        }
        Ok(())
    }
}

struct TrainedBagging {
    models: Vec<Box<dyn TrainedModel>>,
    n_classes: Option<usize>,
    is_classifier: bool,
}

impl TrainedModel for TrainedBagging {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        if self.is_classifier {
            self.predict_classif(features)
        } else {
            self.predict_regress(features)
        }
    }

    fn feature_importance(&self) -> Option<Vec<(String, f64)>> {
        // Collect importances from all models that provide them
        let all_imps: Vec<Vec<(String, f64)>> = self
            .models
            .iter()
            .filter_map(|m| m.feature_importance())
            .collect();
        if all_imps.is_empty() {
            return None;
        }
        let n_features = all_imps[0].len();
        let n_models = all_imps.len() as f64;
        let mut avg = vec![0.0; n_features];
        let mut names = Vec::new();
        for imp in &all_imps {
            if names.is_empty() {
                names = imp.iter().map(|(n, _)| n.clone()).collect();
            }
            for (j, (_, v)) in imp.iter().enumerate() {
                avg[j] += v;
            }
        }
        let total: f64 = avg.iter().sum();
        if total == 0.0 {
            return None;
        }
        Some(
            names
                .into_iter()
                .zip(avg)
                .map(|(name, imp)| (name, imp / n_models))
                .collect(),
        )
    }
}

impl TrainedBagging {
    fn predict_classif(&self, features: &Array2<f64>) -> Result<Prediction> {
        let n_classes = self.n_classes.unwrap();
        let preds: Vec<Prediction> = self
            .models
            .iter()
            .map(|m| m.predict(features))
            .collect::<Result<Vec<_>>>()?;

        // Try soft voting (probability averaging) first
        let has_probs = preds.iter().all(|p| {
            matches!(
                p,
                Prediction::Classification {
                    probabilities: Some(_),
                    ..
                }
            )
        });

        let n_samples = features.nrows();
        let n_models = preds.len() as f64;

        if has_probs {
            let mut predicted = Vec::with_capacity(n_samples);
            let mut probabilities = Vec::with_capacity(n_samples);

            for s in 0..n_samples {
                let mut avg = vec![0.0; n_classes];
                for pred in &preds {
                    if let Prediction::Classification {
                        probabilities: Some(probs),
                        ..
                    } = pred
                    {
                        for (j, p) in probs[s].iter().enumerate() {
                            avg[j] += p;
                        }
                    }
                }
                for p in &mut avg {
                    *p /= n_models;
                }
                let cls = avg
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap()
                    .0;
                predicted.push(cls);
                probabilities.push(avg);
            }

            Ok(Prediction::Classification {
                predicted,
                truth: None,
                probabilities: Some(probabilities),
            })
        } else {
            // Hard majority vote
            let mut predicted = Vec::with_capacity(n_samples);
            for s in 0..n_samples {
                let mut votes = vec![0usize; n_classes];
                for pred in &preds {
                    if let Prediction::Classification { predicted: p, .. } = pred {
                        votes[p[s]] += 1;
                    }
                }
                let cls = votes.iter().enumerate().max_by_key(|&(_, &c)| c).unwrap().0;
                predicted.push(cls);
            }
            Ok(Prediction::classification(predicted))
        }
    }

    fn predict_regress(&self, features: &Array2<f64>) -> Result<Prediction> {
        let preds: Vec<Prediction> = self
            .models
            .iter()
            .map(|m| m.predict(features))
            .collect::<Result<Vec<_>>>()?;

        let n_samples = features.nrows();
        let n_models = preds.len() as f64;
        let mut predicted = vec![0.0; n_samples];

        for pred in &preds {
            if let Prediction::Regression { predicted: p, .. } = pred {
                for (i, v) in p.iter().enumerate() {
                    predicted[i] += v;
                }
            }
        }
        for v in &mut predicted {
            *v /= n_models;
        }

        Ok(Prediction::regression(predicted))
    }
}

impl Learner for Bagging {
    fn id(&self) -> &str {
        "bagging"
    }

    fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        crate::validate::check_no_weights(task.weights(), "Bagging")?;
        self.check_n_estimators()?;
        let features = task.features();
        let target = task.target();
        let n_samples = task.n_samples();
        let n_classes = task.n_classes();
        let task_id = task.id().to_string();
        let factory = &self.factory;
        let seed = self.seed;

        let results: Vec<Result<Box<dyn TrainedModel>>> = (0..self.n_estimators)
            .into_par_iter()
            .map(|i| {
                let mut rng = StdRng::seed_from_u64(seed.wrapping_add(i as u64));
                let indices: Vec<usize> = (0..n_samples)
                    .map(|_| rng.random_range(0..n_samples))
                    .collect();

                let boot_features = features.select(Axis(0), &indices);
                let boot_target: Vec<usize> = indices.iter().map(|&idx| target[idx]).collect();
                // class_names forwarded so a bootstrap that lost the highest
                // class still yields full-width probability rows (same
                // propagation Stacking/DeepForest folds need; audit LOW).
                let boot_task = ClassificationTask::new(&task_id, boot_features, boot_target)?
                    .with_class_names(task.class_names().to_vec());

                let mut learner = factory();
                learner.train_classif(&boot_task)
            })
            .collect();

        let models: Vec<Box<dyn TrainedModel>> = results.into_iter().collect::<Result<Vec<_>>>()?;

        Ok(Box::new(TrainedBagging {
            models,
            n_classes: Some(n_classes),
            is_classifier: true,
        }))
    }

    fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
        crate::validate::check_no_weights(task.weights(), "Bagging")?;
        self.check_n_estimators()?;
        let features = task.features();
        let target = task.target();
        let n_samples = task.n_samples();
        let task_id = task.id().to_string();
        let factory = &self.factory;
        let seed = self.seed;

        let results: Vec<Result<Box<dyn TrainedModel>>> = (0..self.n_estimators)
            .into_par_iter()
            .map(|i| {
                let mut rng = StdRng::seed_from_u64(seed.wrapping_add(i as u64));
                let indices: Vec<usize> = (0..n_samples)
                    .map(|_| rng.random_range(0..n_samples))
                    .collect();

                let boot_features = features.select(Axis(0), &indices);
                let boot_target: Vec<f64> = indices.iter().map(|&idx| target[idx]).collect();
                let boot_task = RegressionTask::new(&task_id, boot_features, boot_target)?;

                let mut learner = factory();
                learner.train_regress(&boot_task)
            })
            .collect();

        let models: Vec<Box<dyn TrainedModel>> = results.into_iter().collect::<Result<Vec<_>>>()?;

        Ok(Box::new(TrainedBagging {
            models,
            n_classes: None,
            is_classifier: false,
        }))
    }
}
