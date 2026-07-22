//! Random search over hyperparameter distributions.

use super::{Dependency, ParamSet, ParamSpace, ParetoResult, TuneResult};
use crate::Result;
use crate::benchmark;
use crate::learner::Learner;
use crate::measure::Measure;
use crate::resample::Resample;
use crate::task::{ClassificationTask, RegressionTask};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;
use std::collections::HashSet;

/// Random search over hyperparameter distributions.
///
/// Samples `n_iter` random configurations and evaluates them using
/// cross-validation. Often more efficient than grid search for
/// high-dimensional parameter spaces.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use smelt_ml::tuning::{RandomSearch, ParamSpace, ParamDistribution, ParamValue};
/// use ndarray::array;
///
/// let features = array![
///     [0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [0.0, 0.2],
///     [1.0, 1.0], [1.1, 0.9], [0.9, 1.1], [1.0, 0.9]
/// ];
/// let target = vec![0, 0, 0, 0, 1, 1, 1, 1];
/// let task = ClassificationTask::new("tune", features, target).unwrap();
///
/// let mut space = ParamSpace::new();
/// space.insert("max_depth".into(), ParamDistribution::Choice(
///     vec![ParamValue::Int(1), ParamValue::Int(3), ParamValue::Int(5), ParamValue::Int(10)]
/// ));
///
/// let rs = RandomSearch::new(
///     |params| Box::new(DecisionTree::new()
///         .with_max_depth(params["max_depth"].as_usize().unwrap())),
///     space,
/// ).with_n_iter(5).with_seed(42);
///
/// let cv = CrossValidation::new(2).with_seed(42);
/// let result = rs.tune_classif(&task, &cv, &Accuracy).unwrap();
/// ```
pub struct RandomSearch {
    factory: Box<dyn Fn(&ParamSet) -> Box<dyn Learner> + Send + Sync>,
    param_space: ParamSpace,
    n_iter: usize,
    seed: u64,
    dependencies: Vec<Dependency>,
}

impl RandomSearch {
    /// Create a random search over `param_space` using `factory` to build a
    /// learner from a sampled parameter set.
    pub fn new(
        factory: impl Fn(&ParamSet) -> Box<dyn Learner> + Send + Sync + 'static,
        param_space: ParamSpace,
    ) -> Self {
        Self {
            factory: Box::new(factory),
            param_space,
            n_iter: 10,
            seed: 42,
            dependencies: Vec::new(),
        }
    }

    /// Set the number of random configurations to sample and evaluate.
    pub fn with_n_iter(mut self, n: usize) -> Self {
        self.n_iter = n;
        self
    }

    /// Set the RNG seed for reproducible configuration sampling.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Register a conditional [`Dependency`]: a child parameter that only
    /// reaches the factory when its parent's sampled value satisfies the
    /// condition. Sampling still draws the full space (so seeded runs stay
    /// reproducible), but an inactive child is pruned before the factory sees
    /// it — never a silent no-op. Chainable to declare several.
    pub fn with_dependency(mut self, dep: Dependency) -> Self {
        self.dependencies.push(dep);
        self
    }

    fn sample_params(&self, rng: &mut impl Rng) -> ParamSet {
        let mut params = super::sample_param_space(&self.param_space, rng);
        super::prune_inactive(&mut params, &self.dependencies);
        params
    }

    /// Validate the registered dependencies against the space's parameter
    /// names (shared by `tune_classif`/`tune_regress`).
    fn validate_deps(&self) -> Result<()> {
        let names: HashSet<&str> = self.param_space.keys().map(String::as_str).collect();
        super::validate_dependencies(&names, &self.dependencies)
    }

    /// Tune for classification.
    pub fn tune_classif(
        &self,
        task: &ClassificationTask,
        resampling: &dyn Resample,
        measure: &dyn Measure,
    ) -> Result<TuneResult> {
        super::validate_param_space(&self.param_space)?;
        self.validate_deps()?;
        // Sampling stays sequential (it's cheap and depends on shared &mut
        // rng state, so the sequence -- and hence reproducibility for a
        // given seed -- is unaffected); only the expensive train+evaluate
        // step per candidate runs in parallel.
        let mut rng = StdRng::seed_from_u64(self.seed);
        let param_sets: Vec<ParamSet> = (0..self.n_iter)
            .map(|_| self.sample_params(&mut rng))
            .collect();

        let results: Result<Vec<(ParamSet, f64)>> = param_sets
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

    /// Tune for regression.
    pub fn tune_regress(
        &self,
        task: &RegressionTask,
        resampling: &dyn Resample,
        measure: &dyn Measure,
    ) -> Result<TuneResult> {
        super::validate_param_space(&self.param_space)?;
        self.validate_deps()?;
        let mut rng = StdRng::seed_from_u64(self.seed);
        let param_sets: Vec<ParamSet> = (0..self.n_iter)
            .map(|_| self.sample_params(&mut rng))
            .collect();

        let results: Result<Vec<(ParamSet, f64)>> = param_sets
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

    /// Multi-objective classification tuning: evaluate every sampled
    /// configuration on **all** `measures` and return the Pareto front (the
    /// non-dominated trade-offs) instead of a single best. See [`ParetoResult`].
    pub fn tune_classif_multi(
        &self,
        task: &ClassificationTask,
        resampling: &dyn Resample,
        measures: &[&dyn Measure],
    ) -> Result<ParetoResult> {
        if measures.is_empty() {
            return Err(crate::SmeltError::InvalidParameter(
                "multi-objective tuning requires at least one measure".into(),
            ));
        }
        super::validate_param_space(&self.param_space)?;
        self.validate_deps()?;
        let mut rng = StdRng::seed_from_u64(self.seed);
        let param_sets: Vec<ParamSet> = (0..self.n_iter)
            .map(|_| self.sample_params(&mut rng))
            .collect();

        let results: Result<Vec<(ParamSet, Vec<f64>)>> = param_sets
            .into_par_iter()
            .map(|params| {
                let mut learner = (self.factory)(&params);
                let bench = benchmark::resample_classif(&mut *learner, task, resampling, measures)?;
                Ok((params, bench.mean_scores()))
            })
            .collect();

        let ids = measures.iter().map(|m| m.id().to_string()).collect();
        let maximize = measures.iter().map(|m| m.maximize()).collect();
        ParetoResult::from_results(results?, ids, maximize)
    }

    /// Multi-objective regression tuning — the regression counterpart of
    /// [`tune_classif_multi`](Self::tune_classif_multi).
    pub fn tune_regress_multi(
        &self,
        task: &RegressionTask,
        resampling: &dyn Resample,
        measures: &[&dyn Measure],
    ) -> Result<ParetoResult> {
        if measures.is_empty() {
            return Err(crate::SmeltError::InvalidParameter(
                "multi-objective tuning requires at least one measure".into(),
            ));
        }
        super::validate_param_space(&self.param_space)?;
        self.validate_deps()?;
        let mut rng = StdRng::seed_from_u64(self.seed);
        let param_sets: Vec<ParamSet> = (0..self.n_iter)
            .map(|_| self.sample_params(&mut rng))
            .collect();

        let results: Result<Vec<(ParamSet, Vec<f64>)>> = param_sets
            .into_par_iter()
            .map(|params| {
                let mut learner = (self.factory)(&params);
                let bench = benchmark::resample_regress(&mut *learner, task, resampling, measures)?;
                Ok((params, bench.mean_scores()))
            })
            .collect();

        let ids = measures.iter().map(|m| m.id().to_string()).collect();
        let maximize = measures.iter().map(|m| m.maximize()).collect();
        ParetoResult::from_results(results?, ids, maximize)
    }
}
