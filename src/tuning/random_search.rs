//! Random search over hyperparameter distributions.

use super::{ParamSet, ParamSpace, TuneResult};
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

    fn sample_params(&self, rng: &mut impl Rng) -> ParamSet {
        super::sample_param_space(&self.param_space, rng)
    }

    /// Tune for classification.
    pub fn tune_classif(
        &self,
        task: &ClassificationTask,
        resampling: &dyn Resample,
        measure: &dyn Measure,
    ) -> Result<TuneResult> {
        // Sampling stays sequential (it's cheap and depends on shared &mut
        // rng state, so the sequence -- and hence reproducibility for a
        // given seed -- is unaffected); only the expensive train+evaluate
        // step per candidate runs in parallel.
        let mut rng = StdRng::seed_from_u64(self.seed);
        let param_sets: Vec<ParamSet> = (0..self.n_iter).map(|_| self.sample_params(&mut rng)).collect();

        let results: Result<Vec<(ParamSet, f64)>> = param_sets
            .into_par_iter()
            .map(|params| {
                let mut learner = (self.factory)(&params);
                let bench = benchmark::resample_classif(&mut *learner, task, resampling, &[measure])?;
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
        let mut rng = StdRng::seed_from_u64(self.seed);
        let param_sets: Vec<ParamSet> = (0..self.n_iter).map(|_| self.sample_params(&mut rng)).collect();

        let results: Result<Vec<(ParamSet, f64)>> = param_sets
            .into_par_iter()
            .map(|params| {
                let mut learner = (self.factory)(&params);
                let bench = benchmark::resample_regress(&mut *learner, task, resampling, &[measure])?;
                let mean_score = bench.mean_scores()[0];
                Ok((params, mean_score))
            })
            .collect();

        TuneResult::select_best(results?, measure.id().to_string(), measure.maximize())
    }
}
