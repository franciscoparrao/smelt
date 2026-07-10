//! Pipeline: chains transformers with a learner into a single Learner.

use super::{Resampler, Transformer};
use crate::learner::{Learner, TrainedModel};
use crate::prediction::Prediction;
use crate::task::{ClassificationTask, RegressionTask, Task};
use crate::{Result, SmeltError};
use ndarray::Array2;

/// Chains zero or more Transformers followed by a Learner.
///
/// Pipeline implements `Learner`, so it integrates with the benchmark/resample
/// system. Transformers are fitted only on training data, preventing data leakage.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use ndarray::array;
///
/// let features = array![[1.0, 100.0], [2.0, 200.0], [3.0, 300.0], [4.0, 400.0]];
/// let target = vec![0, 0, 1, 1];
/// let task = ClassificationTask::new("pipe", features, target).unwrap();
///
/// let mut pipe = Pipeline::new(
///     vec![Box::new(StandardScaler::new())],
///     Box::new(DecisionTree::default()),
/// );
/// let model = pipe.train_classif(&task).unwrap();
/// ```
///
/// A resampler (SMOTE/ADASYN) can be attached via [`Pipeline::with_resampler`]
/// to rebalance the training set before any transformer runs -- see that
/// method's docs for why it's a separate stage from `transformers` (audit
/// issue M18).
pub struct Pipeline {
    resampler: Option<Box<dyn Resampler>>,
    transformers: Vec<Box<dyn Transformer>>,
    learner: Box<dyn Learner>,
    id: String,
}

impl Pipeline {
    /// Create a pipeline that applies `transformers` in order, then trains
    /// `learner` on the transformed data.
    pub fn new(transformers: Vec<Box<dyn Transformer>>, learner: Box<dyn Learner>) -> Self {
        let id = Self::build_id(None, &transformers, learner.id());
        Self {
            resampler: None,
            transformers,
            learner,
            id,
        }
    }

    /// Attach a resampler (e.g. [`super::Smote`]/[`super::Adasyn`]) applied
    /// once at the start of `train_classif`, before any transformer --
    /// unlike a `Transformer`, it never runs at predict time (there's
    /// nothing to rebalance in held-out data) and its output isn't stored
    /// in the trained model. `train_regress` returns an error if a
    /// resampler is set: SMOTE/ADASYN rebalance discrete class counts, and
    /// there's no regression equivalent.
    pub fn with_resampler(mut self, resampler: Box<dyn Resampler>) -> Self {
        self.id = Self::build_id(Some(resampler.id()), &self.transformers, self.learner.id());
        self.resampler = Some(resampler);
        self
    }

    fn build_id(
        resampler_id: Option<&str>,
        transformers: &[Box<dyn Transformer>],
        learner_id: &str,
    ) -> String {
        let mut stages: Vec<&str> = resampler_id.into_iter().collect();
        stages.extend(transformers.iter().map(|t| t.id()));
        if stages.is_empty() {
            format!("pipeline({})", learner_id)
        } else {
            format!("pipeline({}+{})", stages.join("+"), learner_id)
        }
    }
}

struct TrainedPipeline {
    transformers: Vec<Box<dyn Transformer>>,
    model: Box<dyn TrainedModel>,
}

impl TrainedModel for TrainedPipeline {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        let mut transformed = features.clone();
        for transformer in &self.transformers {
            transformed = transformer.transform(&transformed)?;
        }
        self.model.predict(&transformed)
    }

    fn feature_importance(&self) -> Option<Vec<(String, f64)>> {
        self.model.feature_importance()
    }
}

impl Learner for Pipeline {
    fn id(&self) -> &str {
        &self.id
    }

    fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        let resampled;
        let task = if let Some(resampler) = &self.resampler {
            resampled = resampler.resample(task)?;
            &resampled
        } else {
            task
        };

        let mut features = task.features().clone();
        let mut names = task.feature_names().to_vec();
        // Pass target as f64 for supervised filters
        let target_f64: Vec<f64> = task.target().iter().map(|&t| t as f64).collect();

        for transformer in &mut self.transformers {
            features = transformer.fit_transform_supervised(&features, &target_f64)?;
            names = transformer.transform_names(&names)?;
        }

        // Propagate class_names: rebuilding the task from scratch would
        // re-derive n_classes as max(label)+1, silently narrowing the
        // probability rows whenever this pipeline's training split lost the
        // highest class -- exactly the width mismatch Stacking/DES defend
        // against by forwarding class_names to every fold (a base learner
        // that is itself a Pipeline used to destroy that propagation and
        // panic downstream).
        let transformed_task =
            ClassificationTask::new(task.id(), features, task.target().to_vec())?
                .with_feature_names(names)?
                .with_class_names(task.class_names().to_vec());

        let model = self.learner.train_classif(&transformed_task)?;

        let fitted: Vec<Box<dyn Transformer>> =
            self.transformers.iter().map(|t| t.clone_box()).collect();

        Ok(Box::new(TrainedPipeline {
            transformers: fitted,
            model,
        }))
    }

    fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
        if self.resampler.is_some() {
            return Err(SmeltError::InvalidParameter(
                "Pipeline's resampler (SMOTE/ADASYN) only applies to classification tasks -- \
                 this pipeline was configured via with_resampler(...) but train_regress was called"
                    .into(),
            ));
        }

        let mut features = task.features().clone();
        let mut names = task.feature_names().to_vec();
        let target_f64 = task.target();

        for transformer in &mut self.transformers {
            features = transformer.fit_transform_supervised(&features, target_f64)?;
            names = transformer.transform_names(&names)?;
        }

        let transformed_task = RegressionTask::new(task.id(), features, task.target().to_vec())?
            .with_feature_names(names)?;

        let model = self.learner.train_regress(&transformed_task)?;

        let fitted: Vec<Box<dyn Transformer>> =
            self.transformers.iter().map(|t| t.clone_box()).collect();

        Ok(Box::new(TrainedPipeline {
            transformers: fitted,
            model,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::knn::KNearestNeighbors;
    use crate::preprocess::Smote;
    use ndarray::Array2;

    /// 20 majority-class points, 2 minority-class points far away.
    fn imbalanced_task() -> ClassificationTask {
        let mut rows = Vec::new();
        let mut target = Vec::new();
        for i in 0..20 {
            rows.push([i as f64 * 0.01, i as f64 * 0.01]);
            target.push(0);
        }
        rows.push([5.0, 5.0]);
        rows.push([5.1, 4.9]);
        target.push(1);
        target.push(1);
        let n = rows.len();
        let flat: Vec<f64> = rows.iter().flatten().copied().collect();
        let features = Array2::from_shape_vec((n, 2), flat).unwrap();
        ClassificationTask::new("imbalanced", features, target).unwrap()
    }

    #[test]
    fn resampler_runs_before_transformers_and_grows_training_set() {
        let task = imbalanced_task();
        let mut pipe = Pipeline::new(vec![], Box::new(KNearestNeighbors::new(1)))
            .with_resampler(Box::new(Smote::new().with_k_neighbors(1).with_seed(42)));

        // Exercise train_classif's resample step directly via the trait,
        // matching what Pipeline does internally, to confirm the class
        // counts actually grew (the behavior a Transformer could never
        // provide, since it can't change sample count or target).
        let resampler = Smote::new().with_k_neighbors(1).with_seed(42);
        let balanced = resampler.resample(&task).unwrap();
        let n_minority_before = task.target().iter().filter(|&&t| t == 1).count();
        let n_minority_after = balanced.target().iter().filter(|&&t| t == 1).count();
        assert!(
            n_minority_after > n_minority_before,
            "resample should have synthesized additional minority samples: before={n_minority_before}, after={n_minority_after}"
        );

        // Pipeline itself must train successfully end-to-end with the
        // resampler attached.
        let model = pipe.train_classif(&task).unwrap();
        let pred = model.predict(task.features()).unwrap();
        let n = match pred {
            crate::prediction::Prediction::Classification { predicted, .. } => predicted.len(),
            _ => unreachable!(),
        };
        assert_eq!(
            n,
            task.n_samples(),
            "predict on the ORIGINAL (pre-resample) features must return exactly \
             one prediction per input row -- the resampler must not leak into predict"
        );
    }

    #[test]
    fn train_regress_rejects_a_configured_resampler() {
        use crate::learner::linear_regression::LinearRegression;
        use crate::task::RegressionTask;

        let features = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let target = vec![1.0, 2.0, 3.0, 4.0];
        let task = RegressionTask::new("r", features, target).unwrap();

        let mut pipe = Pipeline::new(vec![], Box::new(LinearRegression::new()))
            .with_resampler(Box::new(Smote::new()));

        match pipe.train_regress(&task) {
            Err(err) => assert!(
                format!("{err}").contains("only applies to classification"),
                "error should explain why a resampler can't be used for regression: {err}"
            ),
            Ok(_) => panic!("expected an error: a resampler was configured for a regression task"),
        }
    }

    #[test]
    fn id_includes_the_resampler_stage() {
        let pipe = Pipeline::new(vec![], Box::new(KNearestNeighbors::new(1)))
            .with_resampler(Box::new(Smote::new()));
        assert!(pipe.id().contains("smote"), "id={}", pipe.id());
    }
}
