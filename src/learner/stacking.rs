//! Stacking (Super Learner): meta-ensemble combining multiple learners.
//!
//! Level 0: trains K base learners with cross-validation.
//! Level 1: trains a meta-learner on out-of-fold predictions from level 0.

use crate::Result;
use crate::learner::{Learner, TrainedModel};
use crate::prediction::Prediction;
use crate::resample::{CrossValidation, Resample};
use crate::task::{ClassificationTask, RegressionTask, Task};
use ndarray::{Array2, Axis};

/// Stacking ensemble (Super Learner).
///
/// Combines predictions from multiple heterogeneous learners using a
/// meta-learner trained on out-of-fold predictions.
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
/// let task = ClassificationTask::new("stack", features, target).unwrap();
///
/// let mut stack = Stacking::new(
///     vec![
///         Box::new(|| Box::new(DecisionTree::default()) as Box<dyn Learner>),
///         Box::new(|| Box::new(KNearestNeighbors::new(3)) as Box<dyn Learner>),
///     ],
///     || Box::new(LogisticRegression::new().with_max_iter(500)),
/// );
/// let model = stack.train_classif(&task).unwrap();
/// ```
pub struct Stacking {
    base_factories: Vec<Box<dyn Fn() -> Box<dyn Learner> + Send + Sync>>,
    meta_factory: Box<dyn Fn() -> Box<dyn Learner> + Send + Sync>,
    cv_folds: usize,
    cv_seed: u64,
}

impl Stacking {
    pub fn new(
        base_factories: Vec<Box<dyn Fn() -> Box<dyn Learner> + Send + Sync>>,
        meta_factory: impl Fn() -> Box<dyn Learner> + Send + Sync + 'static,
    ) -> Self {
        Self {
            base_factories,
            meta_factory: Box::new(meta_factory),
            cv_folds: 5,
            cv_seed: 42,
        }
    }

    pub fn with_cv_folds(mut self, folds: usize) -> Self {
        self.cv_folds = folds;
        self
    }
    pub fn with_cv_seed(mut self, seed: u64) -> Self {
        self.cv_seed = seed;
        self
    }
}

struct TrainedStacking {
    base_models: Vec<Box<dyn TrainedModel>>,
    meta_model: Box<dyn TrainedModel>,
    n_base: usize,
    is_classifier: bool,
    n_classes: Option<usize>,
}

impl TrainedModel for TrainedStacking {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        // Generate level-1 features from base models
        let meta_features = self.build_meta_features(features)?;
        self.meta_model.predict(&meta_features)
    }
}

impl TrainedStacking {
    fn build_meta_features(&self, features: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = features.nrows();

        if self.is_classifier {
            let nc = self.n_classes.unwrap_or(2);
            let n_cols = self.n_base * nc;
            let mut meta = Array2::zeros((n_samples, n_cols));

            for (m, model) in self.base_models.iter().enumerate() {
                let pred = model.predict(features)?;
                if let Prediction::Classification {
                    probabilities: Some(probs),
                    ..
                } = &pred
                {
                    for i in 0..n_samples {
                        for c in 0..nc {
                            meta[[i, m * nc + c]] = probs[i][c];
                        }
                    }
                }
            }
            Ok(meta)
        } else {
            let mut meta = Array2::zeros((n_samples, self.n_base));
            for (m, model) in self.base_models.iter().enumerate() {
                let pred = model.predict(features)?;
                if let Prediction::Regression { predicted, .. } = &pred {
                    for i in 0..n_samples {
                        meta[[i, m]] = predicted[i];
                    }
                }
            }
            Ok(meta)
        }
    }
}

impl Learner for Stacking {
    fn id(&self) -> &str {
        "stacking"
    }

    fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        let features = task.features();
        let target = task.target();
        let n_samples = task.n_samples();
        let n_base = self.base_factories.len();
        let n_classes = task.n_classes();

        let cv = CrossValidation::new(self.cv_folds).with_seed(self.cv_seed);
        let splits = cv.splits(n_samples)?;

        // Build out-of-fold predictions (level-1 features)
        let n_meta_cols = n_base * n_classes;
        let mut oof_meta = Array2::zeros((n_samples, n_meta_cols));

        for (m, factory) in self.base_factories.iter().enumerate() {
            for (train_idx, test_idx) in &splits {
                let train_features = features.select(Axis(0), train_idx);
                let train_target: Vec<usize> = train_idx.iter().map(|&i| target[i]).collect();
                let train_task = ClassificationTask::new(task.id(), train_features, train_target)?;

                let mut learner = factory();
                let model = learner.train_classif(&train_task)?;
                let test_features = features.select(Axis(0), test_idx);
                let pred = model.predict(&test_features)?;

                if let Prediction::Classification {
                    probabilities: Some(probs),
                    ..
                } = &pred
                {
                    for (j, &idx) in test_idx.iter().enumerate() {
                        for c in 0..n_classes {
                            oof_meta[[idx, m * n_classes + c]] = probs[j][c];
                        }
                    }
                }
            }
        }

        // Train meta-learner on OOF predictions
        let meta_task = ClassificationTask::new("meta", oof_meta, target.to_vec())?;
        let mut meta_learner = (self.meta_factory)();
        let meta_model = meta_learner.train_classif(&meta_task)?;

        // Retrain base models on full data
        let mut base_models: Vec<Box<dyn TrainedModel>> = Vec::with_capacity(n_base);
        for factory in &self.base_factories {
            let mut learner = factory();
            base_models.push(learner.train_classif(task)?);
        }

        Ok(Box::new(TrainedStacking {
            base_models,
            meta_model,
            n_base,
            is_classifier: true,
            n_classes: Some(n_classes),
        }))
    }

    fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
        let features = task.features();
        let target = task.target();
        let n_samples = task.n_samples();
        let n_base = self.base_factories.len();

        let cv = CrossValidation::new(self.cv_folds).with_seed(self.cv_seed);
        let splits = cv.splits(n_samples)?;

        let mut oof_meta = Array2::zeros((n_samples, n_base));

        for (m, factory) in self.base_factories.iter().enumerate() {
            for (train_idx, test_idx) in &splits {
                let train_features = features.select(Axis(0), train_idx);
                let train_target: Vec<f64> = train_idx.iter().map(|&i| target[i]).collect();
                let train_task = RegressionTask::new(task.id(), train_features, train_target)?;

                let mut learner = factory();
                let model = learner.train_regress(&train_task)?;
                let test_features = features.select(Axis(0), test_idx);
                let pred = model.predict(&test_features)?;

                if let Prediction::Regression { predicted, .. } = &pred {
                    for (j, &idx) in test_idx.iter().enumerate() {
                        oof_meta[[idx, m]] = predicted[j];
                    }
                }
            }
        }

        let meta_task = RegressionTask::new("meta", oof_meta, target.to_vec())?;
        let mut meta_learner = (self.meta_factory)();
        let meta_model = meta_learner.train_regress(&meta_task)?;

        let mut base_models: Vec<Box<dyn TrainedModel>> = Vec::with_capacity(n_base);
        for factory in &self.base_factories {
            let mut learner = factory();
            base_models.push(learner.train_regress(task)?);
        }

        Ok(Box::new(TrainedStacking {
            base_models,
            meta_model,
            n_base,
            is_classifier: false,
            n_classes: None,
        }))
    }
}
