//! Hyperband: efficient hyperparameter tuning via successive halving.
//!
//! Evaluates many configurations with few resources (small CV folds),
//! discards the worst, and allocates more resources to the best.

use super::{ParamSet, ParamSpace, TuneResult};
use crate::Result;
use crate::benchmark;
use crate::learner::Learner;
use crate::measure::Measure;
use crate::resample::CrossValidation;
use crate::task::{ClassificationTask, RegressionTask};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;

/// Hyperband tuner for efficient hyperparameter optimization.
///
/// Uses successive halving: starts with many random configurations
/// evaluated on few CV folds, then progressively eliminates the worst
/// and increases the budget (more folds) for survivors.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use smelt_ml::tuning::{Hyperband, ParamSpace, ParamDistribution};
/// use ndarray::array;
///
/// let features = array![
///     [0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [0.0, 0.2],
///     [0.1, 0.0], [0.2, 0.1], [0.0, 0.1], [0.1, 0.2],
///     [1.0, 1.0], [1.1, 0.9], [0.9, 1.1], [1.0, 0.9],
///     [1.1, 1.0], [0.9, 1.0], [1.0, 1.1], [1.1, 1.1]
/// ];
/// let target = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
/// let task = ClassificationTask::new("hb", features, target).unwrap();
///
/// let mut space = ParamSpace::new();
/// space.insert("max_depth".into(), ParamDistribution::Uniform(1.0, 10.0));
///
/// let hb = Hyperband::new(
///     |params| Box::new(DecisionTree::new()
///         .with_max_depth(params["max_depth"].as_usize().unwrap())),
///     space,
/// ).with_max_folds(4).with_seed(42);
///
/// let result = hb.tune_classif(&task, &Accuracy).unwrap();
/// ```
pub struct Hyperband {
    factory: Box<dyn Fn(&ParamSet) -> Box<dyn Learner> + Send + Sync>,
    param_space: ParamSpace,
    /// Maximum number of CV folds (the "full budget").
    max_folds: usize,
    /// Reduction factor: eliminate 1/eta configs per round.
    eta: usize,
    seed: u64,
}

impl Hyperband {
    /// Create a Hyperband tuner over `param_space` using `factory` to build
    /// a learner from a sampled parameter set.
    pub fn new(
        factory: impl Fn(&ParamSet) -> Box<dyn Learner> + Send + Sync + 'static,
        param_space: ParamSpace,
    ) -> Self {
        Self {
            factory: Box::new(factory),
            param_space,
            max_folds: 5,
            eta: 3,
            seed: 42,
        }
    }

    /// Set the maximum CV fold budget given to surviving configurations.
    pub fn with_max_folds(mut self, n: usize) -> Self {
        self.max_folds = n;
        self
    }
    /// Set the halving rate: eliminate all but 1/`eta` of configurations per
    /// round.
    pub fn with_eta(mut self, e: usize) -> Self {
        self.eta = e;
        self
    }
    /// Set the RNG seed for reproducible configuration sampling.
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    fn sample_random(&self, rng: &mut StdRng) -> ParamSet {
        super::sample_param_space(&self.param_space, rng)
    }

    /// Tune for classification using successive halving.
    pub fn tune_classif(
        &self,
        task: &ClassificationTask,
        measure: &dyn Measure,
    ) -> Result<TuneResult> {
        let maximize = measure.maximize();
        let mut rng = StdRng::seed_from_u64(self.seed);

        let s_max = (self.max_folds as f64).log(self.eta as f64).floor() as usize;
        let mut all_results: Vec<(ParamSet, f64)> = Vec::new();

        // Successive halving brackets
        for s in (0..=s_max).rev() {
            let n_configs = ((self.eta.pow(s as u32)) as f64 * (s_max + 1) as f64 / (s + 1) as f64)
                .ceil() as usize;
            let min_folds =
                (self.max_folds as f64 / self.eta.pow(s as u32) as f64).max(2.0) as usize;

            // Sample initial configurations
            let mut configs: Vec<ParamSet> = (0..n_configs)
                .map(|_| self.sample_random(&mut rng))
                .collect();

            let mut folds = min_folds;

            // Successive halving within this bracket
            for _i in 0..=s {
                folds = folds.min(self.max_folds).max(2);

                // Evaluate each config with current budget -- independent
                // across configs (each builds its own learner), so this
                // round's evaluations run in parallel.
                let cv = CrossValidation::new(folds).with_seed(self.seed);
                let mut scored: Vec<(ParamSet, f64)> = configs
                    .par_iter()
                    .map(|params| {
                        let mut learner = (self.factory)(params);
                        let result = benchmark::resample_classif(&mut *learner, task, &cv, &[measure])?;
                        Ok::<_, crate::SmeltError>((params.clone(), result.mean_scores()[0]))
                    })
                    .collect::<Result<Vec<_>>>()?;

                all_results.extend(scored.iter().cloned());

                // Keep top 1/eta
                if maximize {
                    scored
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                } else {
                    scored
                        .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                }

                let keep = (configs.len() / self.eta).max(1);
                configs = scored.into_iter().take(keep).map(|(p, _)| p).collect();

                folds = (folds as f64 * self.eta as f64) as usize;
            }
        }

        TuneResult::select_best(all_results, measure.id().to_string(), maximize)
    }

    /// Tune for regression using successive halving.
    pub fn tune_regress(&self, task: &RegressionTask, measure: &dyn Measure) -> Result<TuneResult> {
        let maximize = measure.maximize();
        let mut rng = StdRng::seed_from_u64(self.seed);

        let s_max = (self.max_folds as f64).log(self.eta as f64).floor() as usize;
        let mut all_results: Vec<(ParamSet, f64)> = Vec::new();

        for s in (0..=s_max).rev() {
            let n_configs = ((self.eta.pow(s as u32)) as f64 * (s_max + 1) as f64 / (s + 1) as f64)
                .ceil() as usize;
            let min_folds =
                (self.max_folds as f64 / self.eta.pow(s as u32) as f64).max(2.0) as usize;

            let mut configs: Vec<ParamSet> = (0..n_configs)
                .map(|_| self.sample_random(&mut rng))
                .collect();

            let mut folds = min_folds;

            for _i in 0..=s {
                folds = folds.min(self.max_folds).max(2);
                let cv = CrossValidation::new(folds).with_seed(self.seed);
                let mut scored: Vec<(ParamSet, f64)> = configs
                    .par_iter()
                    .map(|params| {
                        let mut learner = (self.factory)(params);
                        let result = benchmark::resample_regress(&mut *learner, task, &cv, &[measure])?;
                        Ok::<_, crate::SmeltError>((params.clone(), result.mean_scores()[0]))
                    })
                    .collect::<Result<Vec<_>>>()?;

                all_results.extend(scored.iter().cloned());

                if maximize {
                    scored
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                } else {
                    scored
                        .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                }

                let keep = (configs.len() / self.eta).max(1);
                configs = scored.into_iter().take(keep).map(|(p, _)| p).collect();
                folds = (folds as f64 * self.eta as f64) as usize;
            }
        }

        TuneResult::select_best(all_results, measure.id().to_string(), maximize)
    }
}
