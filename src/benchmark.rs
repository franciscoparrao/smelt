//! Benchmark pipeline: resample + measure loop.
//!
//! Runs a learner on resampled data and evaluates with multiple measures.

use crate::Result;
use crate::learner::Learner;
use crate::measure::Measure;
use crate::resample::Resample;
use crate::task::{ClassificationTask, RegressionTask, Task};
use ndarray::Axis;

/// Results from a resampling benchmark run.
#[derive(Debug)]
pub struct BenchmarkResult {
    /// Learner identifier.
    pub learner_id: String,
    /// Measure identifiers.
    pub measure_ids: Vec<String>,
    /// Scores per fold: `scores[fold][measure]`.
    pub scores: Vec<Vec<f64>>,
}

impl BenchmarkResult {
    /// Mean score for each measure across all folds.
    pub fn mean_scores(&self) -> Vec<f64> {
        let n_folds = self.scores.len() as f64;
        let n_measures = self.measure_ids.len();
        (0..n_measures)
            .map(|m| self.scores.iter().map(|fold| fold[m]).sum::<f64>() / n_folds)
            .collect()
    }
}

/// Run a classification learner through resampling and evaluate.
pub fn resample_classif(
    learner: &mut dyn Learner,
    task: &ClassificationTask,
    resampling: &dyn Resample,
    measures: &[&dyn Measure],
) -> Result<BenchmarkResult> {
    let splits = resampling.splits(task.n_samples())?;
    let features = task.features();
    let target = task.target();
    let mut scores = Vec::with_capacity(splits.len());

    for (train_idx, test_idx) in &splits {
        let train_features = features.select(Axis(0), train_idx);
        let train_target: Vec<usize> = train_idx.iter().map(|&i| target[i]).collect();
        let train_task = ClassificationTask::new(task.id(), train_features, train_target)?
            .with_feature_names(task.feature_names().to_vec())?
            .with_feature_types(task.feature_types().to_vec())?
            .with_class_names(task.class_names().to_vec());

        let model = learner.train_classif(&train_task)?;

        let test_features = features.select(Axis(0), test_idx);
        let test_target: Vec<usize> = test_idx.iter().map(|&i| target[i]).collect();
        let pred = model
            .predict(&test_features)?
            .with_truth_classif(test_target);

        let fold_scores: Result<Vec<f64>> = measures.iter().map(|m| m.score(&pred)).collect();
        scores.push(fold_scores?);
    }

    Ok(BenchmarkResult {
        learner_id: learner.id().to_string(),
        measure_ids: measures.iter().map(|m| m.id().to_string()).collect(),
        scores,
    })
}

/// Run a regression learner through resampling and evaluate.
pub fn resample_regress(
    learner: &mut dyn Learner,
    task: &RegressionTask,
    resampling: &dyn Resample,
    measures: &[&dyn Measure],
) -> Result<BenchmarkResult> {
    let splits = resampling.splits(task.n_samples())?;
    let features = task.features();
    let target = task.target();
    let mut scores = Vec::with_capacity(splits.len());

    for (train_idx, test_idx) in &splits {
        let train_features = features.select(Axis(0), train_idx);
        let train_target: Vec<f64> = train_idx.iter().map(|&i| target[i]).collect();
        let train_task = RegressionTask::new(task.id(), train_features, train_target)?
            .with_feature_names(task.feature_names().to_vec())?
            .with_feature_types(task.feature_types().to_vec())?;

        let model = learner.train_regress(&train_task)?;

        let test_features = features.select(Axis(0), test_idx);
        let test_target: Vec<f64> = test_idx.iter().map(|&i| target[i]).collect();
        let pred = model
            .predict(&test_features)?
            .with_truth_regress(test_target);

        let fold_scores: Result<Vec<f64>> = measures.iter().map(|m| m.score(&pred)).collect();
        scores.push(fold_scores?);
    }

    Ok(BenchmarkResult {
        learner_id: learner.id().to_string(),
        measure_ids: measures.iter().map(|m| m.id().to_string()).collect(),
        scores,
    })
}
