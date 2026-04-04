//! Benchmark design: compare multiple learners × tasks × resamplings.
//!
//! Equivalent to mlr3's `benchmark()` function.

use crate::Result;
use crate::benchmark;
use crate::learner::Learner;
use crate::measure::Measure;
use crate::resample::Resample;
use crate::task::{ClassificationTask, RegressionTask};

/// Result of a benchmark design experiment.
#[derive(Debug)]
pub struct BenchmarkDesign {
    /// Results per (learner, task) combination.
    pub entries: Vec<BenchmarkEntry>,
}

/// Single entry in a benchmark design.
#[derive(Debug)]
pub struct BenchmarkEntry {
    pub learner_id: String,
    pub task_id: String,
    pub measure_ids: Vec<String>,
    /// Mean scores per measure.
    pub mean_scores: Vec<f64>,
    /// Scores per fold per measure: `scores[fold][measure]`.
    pub fold_scores: Vec<Vec<f64>>,
}

impl BenchmarkDesign {
    /// Print a summary table.
    pub fn summary(&self) -> String {
        if self.entries.is_empty() {
            return String::from("(empty)");
        }

        let measures = &self.entries[0].measure_ids;
        let mut lines = Vec::new();

        // Header
        let mut header = format!("{:<20} {:<15}", "Learner", "Task");
        for m in measures {
            header.push_str(&format!(" {:>12}", m));
        }
        lines.push(header);
        lines.push("-".repeat(35 + measures.len() * 13));

        for entry in &self.entries {
            let mut line = format!("{:<20} {:<15}", entry.learner_id, entry.task_id);
            for &score in &entry.mean_scores {
                line.push_str(&format!(" {:>12.4}", score));
            }
            lines.push(line);
        }

        lines.join("\n")
    }
}

/// Run a benchmark comparing multiple learners on a classification task.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use smelt_ml::benchmark_design::benchmark_classif;
/// use ndarray::array;
///
/// let features = array![
///     [0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [0.0, 0.2],
///     [1.0, 1.0], [1.1, 0.9], [0.9, 1.1], [1.0, 0.9]
/// ];
/// let target = vec![0, 0, 0, 0, 1, 1, 1, 1];
/// let task = ClassificationTask::new("demo", features, target).unwrap();
///
/// let mut learners: Vec<Box<dyn Learner>> = vec![
///     Box::new(DecisionTree::default()),
///     Box::new(RandomForest::new().with_n_estimators(10).with_seed(42)),
/// ];
/// let cv = CrossValidation::new(2).with_seed(42);
/// let result = benchmark_classif(&mut learners, &[&task], &cv, &[&Accuracy]).unwrap();
/// println!("{}", result.summary());
/// ```
pub fn benchmark_classif(
    learners: &mut [Box<dyn Learner>],
    tasks: &[&ClassificationTask],
    resampling: &dyn Resample,
    measures: &[&dyn Measure],
) -> Result<BenchmarkDesign> {
    let mut entries = Vec::new();

    for task in tasks {
        for learner in learners.iter_mut() {
            let result = benchmark::resample_classif(&mut **learner, task, resampling, measures)?;
            let means = result.mean_scores();
            entries.push(BenchmarkEntry {
                learner_id: result.learner_id,
                task_id: task.id().to_string(),
                measure_ids: result.measure_ids,
                mean_scores: means,
                fold_scores: result.scores,
            });
        }
    }

    Ok(BenchmarkDesign { entries })
}

/// Run a benchmark comparing multiple learners on a regression task.
pub fn benchmark_regress(
    learners: &mut [Box<dyn Learner>],
    tasks: &[&RegressionTask],
    resampling: &dyn Resample,
    measures: &[&dyn Measure],
) -> Result<BenchmarkDesign> {
    let mut entries = Vec::new();

    for task in tasks {
        for learner in learners.iter_mut() {
            let result = benchmark::resample_regress(&mut **learner, task, resampling, measures)?;
            let means = result.mean_scores();
            entries.push(BenchmarkEntry {
                learner_id: result.learner_id,
                task_id: task.id().to_string(),
                measure_ids: result.measure_ids,
                mean_scores: means,
                fold_scores: result.scores,
            });
        }
    }

    Ok(BenchmarkDesign { entries })
}

use crate::task::Task;
