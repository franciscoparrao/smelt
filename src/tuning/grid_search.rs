//! Exhaustive grid search over hyperparameter combinations.

use super::{ParamGrid, ParamSet, TuneResult, cartesian_product};
use crate::Result;
use crate::benchmark;
use crate::learner::Learner;
use crate::measure::Measure;
use crate::resample::Resample;
use crate::task::{ClassificationTask, RegressionTask};
use rayon::prelude::*;

/// Exhaustive search over a grid of hyperparameter values.
///
/// Evaluates every combination of parameters using cross-validation
/// and returns the best configuration.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use smelt_ml::tuning::{GridSearch, ParamGrid, ParamValue};
/// use ndarray::array;
///
/// let features = array![
///     [0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [0.0, 0.2],
///     [1.0, 1.0], [1.1, 0.9], [0.9, 1.1], [1.0, 0.9]
/// ];
/// let target = vec![0, 0, 0, 0, 1, 1, 1, 1];
/// let task = ClassificationTask::new("tune", features, target).unwrap();
///
/// let mut grid = ParamGrid::new();
/// grid.insert("max_depth".into(), vec![ParamValue::Int(1), ParamValue::Int(3), ParamValue::Int(5)]);
///
/// let gs = GridSearch::new(
///     |params| Box::new(DecisionTree::new()
///         .with_max_depth(params["max_depth"].as_usize().unwrap())),
///     grid,
/// );
/// let cv = CrossValidation::new(2).with_seed(42);
/// let result = gs.tune_classif(&task, &cv, &Accuracy).unwrap();
/// ```
pub struct GridSearch {
    factory: Box<dyn Fn(&ParamSet) -> Box<dyn Learner> + Send + Sync>,
    param_grid: ParamGrid,
}

impl GridSearch {
    /// Create a grid search over `param_grid` using `factory` to build a
    /// learner from each parameter combination.
    pub fn new(
        factory: impl Fn(&ParamSet) -> Box<dyn Learner> + Send + Sync + 'static,
        param_grid: ParamGrid,
    ) -> Self {
        Self {
            factory: Box::new(factory),
            param_grid,
        }
    }

    /// Tune for classification. Returns the best hyperparameter configuration.
    pub fn tune_classif(
        &self,
        task: &ClassificationTask,
        resampling: &dyn Resample,
        measure: &dyn Measure,
    ) -> Result<TuneResult> {
        let combinations = cartesian_product(&self.param_grid);

        // Each combination builds and evaluates its own learner independently
        // (the factory is Send + Sync precisely to allow this) -- embarrassingly
        // parallel across combinations, which usually far outnumber CPU cores.
        let results: Result<Vec<(ParamSet, f64)>> = combinations
            .into_par_iter()
            .map(|params| {
                let mut learner = (self.factory)(&params);
                let bench =
                    benchmark::resample_classif(&mut *learner, task, resampling, &[measure])?;
                let mean_score = bench.mean_scores()[0];
                Ok((params, mean_score))
            })
            .collect();

        TuneResult::select_best(results?, measure.id().to_string(), measure.maximize())
    }

    /// Tune for regression. Returns the best hyperparameter configuration.
    pub fn tune_regress(
        &self,
        task: &RegressionTask,
        resampling: &dyn Resample,
        measure: &dyn Measure,
    ) -> Result<TuneResult> {
        let combinations = cartesian_product(&self.param_grid);

        let results: Result<Vec<(ParamSet, f64)>> = combinations
            .into_par_iter()
            .map(|params| {
                let mut learner = (self.factory)(&params);
                let bench =
                    benchmark::resample_regress(&mut *learner, task, resampling, &[measure])?;
                let mean_score = bench.mean_scores()[0];
                Ok((params, mean_score))
            })
            .collect();

        TuneResult::select_best(results?, measure.id().to_string(), measure.maximize())
    }
}
