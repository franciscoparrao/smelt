//! Exhaustive grid search over hyperparameter combinations.

use super::{Dependency, ParamGrid, ParamSet, TuneResult, cartesian_product_with_deps};
use crate::Result;
use crate::benchmark;
use crate::learner::Learner;
use crate::measure::Measure;
use crate::resample::Resample;
use crate::task::{ClassificationTask, RegressionTask};
use rayon::prelude::*;
use std::collections::HashSet;

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
    dependencies: Vec<Dependency>,
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
            dependencies: Vec::new(),
        }
    }

    /// Register a conditional [`Dependency`]: a child parameter that is only
    /// active (reaches the factory) when its parent's value satisfies the
    /// condition. Combinations that differ only in an inactive child collapse
    /// to a single trial, so a gated parameter never wastes evaluations on
    /// bit-identical models. Chainable to declare several.
    pub fn with_dependency(mut self, dep: Dependency) -> Self {
        self.dependencies.push(dep);
        self
    }

    /// Validate the registered dependencies against the grid's parameter
    /// names (shared by `tune_classif`/`tune_regress`).
    fn validate_deps(&self) -> Result<()> {
        let names: HashSet<&str> = self.param_grid.keys().map(String::as_str).collect();
        super::validate_dependencies(&names, &self.dependencies)
    }

    /// Tune for classification. Returns the best hyperparameter configuration.
    pub fn tune_classif(
        &self,
        task: &ClassificationTask,
        resampling: &dyn Resample,
        measure: &dyn Measure,
    ) -> Result<TuneResult> {
        self.validate_deps()?;
        let combinations = cartesian_product_with_deps(&self.param_grid, &self.dependencies);

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
        self.validate_deps()?;
        let combinations = cartesian_product_with_deps(&self.param_grid, &self.dependencies);

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
