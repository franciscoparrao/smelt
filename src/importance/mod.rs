//! Model-agnostic feature importance via permutation.

use rand::seq::SliceRandom;
use rand::rngs::StdRng;
use rand::SeedableRng;
use crate::task::{ClassificationTask, RegressionTask, Task};
use crate::learner::TrainedModel;
use crate::measure::Measure;
use crate::Result;

/// Feature importance result for a single feature.
#[derive(Debug, Clone)]
pub struct FeatureImportance {
    /// Feature name.
    pub feature: String,
    /// Mean importance (performance drop when shuffled).
    pub importance: f64,
    /// Standard deviation across repeats.
    pub std_dev: f64,
}

/// Compute permutation importance for a classification model.
///
/// For each feature, shuffles it `n_repeats` times and measures the
/// performance drop relative to the baseline score. Higher importance
/// means the feature is more critical for predictions.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use smelt_ml::importance::permutation_importance_classif;
/// use ndarray::array;
///
/// let features = array![[0.0, 99.0], [0.1, 42.0], [1.0, 99.0], [1.1, 42.0]];
/// let target = vec![0, 0, 1, 1];
/// let task = ClassificationTask::new("imp", features, target).unwrap();
///
/// let mut tree = DecisionTree::default();
/// let model = tree.train_classif(&task).unwrap();
/// let imp = permutation_importance_classif(&*model, &task, &Accuracy, 5, 42).unwrap();
/// ```
pub fn permutation_importance_classif(
    model: &dyn TrainedModel,
    task: &ClassificationTask,
    measure: &dyn Measure,
    n_repeats: usize,
    seed: u64,
) -> Result<Vec<FeatureImportance>> {
    let features = task.features();
    let truth = task.target();

    // Baseline score
    let baseline_pred = model.predict(features)?
        .with_truth_classif(truth.to_vec());
    let baseline_score = measure.score(&baseline_pred)?;

    let names = task.feature_names().to_vec();
    let n_features = task.n_features();

    let results: Vec<Result<FeatureImportance>> = (0..n_features)
        .map(|j| {
            let mut scores = Vec::with_capacity(n_repeats);

            for r in 0..n_repeats {
                let mut shuffled = features.clone();
                let mut rng = StdRng::seed_from_u64(seed.wrapping_add((j * n_repeats + r) as u64));
                let mut col: Vec<f64> = shuffled.column(j).to_vec();
                col.shuffle(&mut rng);
                for (i, &val) in col.iter().enumerate() {
                    shuffled[[i, j]] = val;
                }

                let pred = model.predict(&shuffled)?
                    .with_truth_classif(truth.to_vec());
                scores.push(measure.score(&pred)?);
            }

            let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;
            let importance = if measure.maximize() {
                baseline_score - mean_score
            } else {
                mean_score - baseline_score
            };

            let variance = scores.iter()
                .map(|&s| {
                    let diff = if measure.maximize() {
                        baseline_score - s
                    } else {
                        s - baseline_score
                    };
                    (diff - importance).powi(2)
                })
                .sum::<f64>() / scores.len() as f64;

            Ok(FeatureImportance {
                feature: names[j].clone(),
                importance,
                std_dev: variance.sqrt(),
            })
        })
        .collect();

    let mut importances: Vec<FeatureImportance> = results.into_iter().collect::<Result<Vec<_>>>()?;
    importances.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap_or(std::cmp::Ordering::Equal));
    Ok(importances)
}

/// Compute permutation importance for a regression model.
pub fn permutation_importance_regress(
    model: &dyn TrainedModel,
    task: &RegressionTask,
    measure: &dyn Measure,
    n_repeats: usize,
    seed: u64,
) -> Result<Vec<FeatureImportance>> {
    let features = task.features();
    let truth = task.target();

    let baseline_pred = model.predict(features)?
        .with_truth_regress(truth.to_vec());
    let baseline_score = measure.score(&baseline_pred)?;

    let names = task.feature_names().to_vec();
    let n_features = task.n_features();

    let results: Vec<Result<FeatureImportance>> = (0..n_features)
        .map(|j| {
            let mut scores = Vec::with_capacity(n_repeats);

            for r in 0..n_repeats {
                let mut shuffled = features.clone();
                let mut rng = StdRng::seed_from_u64(seed.wrapping_add((j * n_repeats + r) as u64));
                let mut col: Vec<f64> = shuffled.column(j).to_vec();
                col.shuffle(&mut rng);
                for (i, &val) in col.iter().enumerate() {
                    shuffled[[i, j]] = val;
                }

                let pred = model.predict(&shuffled)?
                    .with_truth_regress(truth.to_vec());
                scores.push(measure.score(&pred)?);
            }

            let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;
            let importance = if measure.maximize() {
                baseline_score - mean_score
            } else {
                mean_score - baseline_score
            };

            let variance = scores.iter()
                .map(|&s| {
                    let diff = if measure.maximize() {
                        baseline_score - s
                    } else {
                        s - baseline_score
                    };
                    (diff - importance).powi(2)
                })
                .sum::<f64>() / scores.len() as f64;

            Ok(FeatureImportance {
                feature: names[j].clone(),
                importance,
                std_dev: variance.sqrt(),
            })
        })
        .collect();

    let mut importances: Vec<FeatureImportance> = results.into_iter().collect::<Result<Vec<_>>>()?;
    importances.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap_or(std::cmp::Ordering::Equal));
    Ok(importances)
}
