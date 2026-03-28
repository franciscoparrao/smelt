//! Pipeline: chains transformers with a learner into a single Learner.

use ndarray::Array2;
use crate::task::{ClassificationTask, RegressionTask, Task};
use crate::learner::{Learner, TrainedModel};
use crate::prediction::Prediction;
use crate::Result;
use super::Transformer;

/// Chains zero or more Transformers followed by a Learner.
///
/// Pipeline implements `Learner`, so it integrates with the benchmark/resample
/// system. Transformers are fitted only on training data, preventing data leakage.
///
/// # Examples
///
/// ```
/// use smelt::prelude::*;
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
pub struct Pipeline {
    transformers: Vec<Box<dyn Transformer>>,
    learner: Box<dyn Learner>,
    id: String,
}

impl Pipeline {
    pub fn new(
        transformers: Vec<Box<dyn Transformer>>,
        learner: Box<dyn Learner>,
    ) -> Self {
        let id = Self::build_id(&transformers, learner.id());
        Self { transformers, learner, id }
    }

    fn build_id(transformers: &[Box<dyn Transformer>], learner_id: &str) -> String {
        if transformers.is_empty() {
            format!("pipeline({})", learner_id)
        } else {
            let t_ids: Vec<&str> = transformers.iter().map(|t| t.id()).collect();
            format!("pipeline({}+{})", t_ids.join("+"), learner_id)
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
    fn id(&self) -> &str { &self.id }

    fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        let mut features = task.features().clone();
        let mut names = task.feature_names().to_vec();

        for transformer in &mut self.transformers {
            features = transformer.fit_transform(&features)?;
            names = transformer.transform_names(&names)?;
        }

        let transformed_task = ClassificationTask::new(
            task.id(), features, task.target().to_vec(),
        )?.with_feature_names(names)?;

        let model = self.learner.train_classif(&transformed_task)?;

        let fitted: Vec<Box<dyn Transformer>> = self.transformers
            .iter()
            .map(|t| t.clone_box())
            .collect();

        Ok(Box::new(TrainedPipeline {
            transformers: fitted,
            model,
        }))
    }

    fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
        let mut features = task.features().clone();
        let mut names = task.feature_names().to_vec();

        for transformer in &mut self.transformers {
            features = transformer.fit_transform(&features)?;
            names = transformer.transform_names(&names)?;
        }

        let transformed_task = RegressionTask::new(
            task.id(), features, task.target().to_vec(),
        )?.with_feature_names(names)?;

        let model = self.learner.train_regress(&transformed_task)?;

        let fitted: Vec<Box<dyn Transformer>> = self.transformers
            .iter()
            .map(|t| t.clone_box())
            .collect();

        Ok(Box::new(TrainedPipeline {
            transformers: fitted,
            model,
        }))
    }
}
