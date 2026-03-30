//! Hyperband: efficient hyperparameter tuning via successive halving.
//!
//! Evaluates many configurations with few resources (small CV folds),
//! discards the worst, and allocates more resources to the best.

use rand::rngs::StdRng;
use rand::SeedableRng;
use crate::task::{ClassificationTask, RegressionTask};
use crate::learner::Learner;
use crate::measure::Measure;
use crate::resample::CrossValidation;
use crate::benchmark;
use crate::Result;
use super::{ParamSet, ParamSpace, ParamDistribution, TuneResult};

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
///         .with_max_depth(params["max_depth"] as usize)),
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

    pub fn with_max_folds(mut self, n: usize) -> Self { self.max_folds = n; self }
    pub fn with_eta(mut self, e: usize) -> Self { self.eta = e; self }
    pub fn with_seed(mut self, s: u64) -> Self { self.seed = s; self }

    fn sample_random(&self, rng: &mut StdRng) -> ParamSet {
        use rand::Rng;
        let mut params = ParamSet::new();
        for (name, dist) in &self.param_space {
            let value = match dist {
                ParamDistribution::Uniform(lo, hi) => rng.random_range(*lo..=*hi),
                ParamDistribution::LogUniform(lo, hi) => {
                    let log_lo = lo.log10();
                    let log_hi = hi.log10();
                    10.0f64.powf(rng.random_range(log_lo..=log_hi))
                }
                ParamDistribution::Choice(vals) => vals[rng.random_range(0..vals.len())],
            };
            params.insert(name.clone(), value);
        }
        params
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
            let n_configs = ((self.eta.pow(s as u32)) as f64 * (s_max + 1) as f64
                / (s + 1) as f64).ceil() as usize;
            let min_folds = (self.max_folds as f64 / self.eta.pow(s as u32) as f64).max(2.0) as usize;

            // Sample initial configurations
            let mut configs: Vec<ParamSet> = (0..n_configs)
                .map(|_| self.sample_random(&mut rng))
                .collect();

            let mut folds = min_folds;

            // Successive halving within this bracket
            for _i in 0..=s {
                folds = folds.min(self.max_folds).max(2);

                // Evaluate each config with current budget
                let cv = CrossValidation::new(folds).with_seed(self.seed);
                let mut scored: Vec<(ParamSet, f64)> = Vec::new();

                for params in &configs {
                    let mut learner = (self.factory)(params);
                    let result = benchmark::resample_classif(&mut *learner, task, &cv, &[measure])?;
                    scored.push((params.clone(), result.mean_scores()[0]));
                }

                all_results.extend(scored.iter().cloned());

                // Keep top 1/eta
                if maximize {
                    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                } else {
                    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                }

                let keep = (configs.len() / self.eta).max(1);
                configs = scored.into_iter().take(keep).map(|(p, _)| p).collect();

                folds = (folds as f64 * self.eta as f64) as usize;
            }
        }

        Ok(TuneResult::select_best(all_results, measure.id().to_string(), maximize))
    }

    /// Tune for regression using successive halving.
    pub fn tune_regress(
        &self,
        task: &RegressionTask,
        measure: &dyn Measure,
    ) -> Result<TuneResult> {
        let maximize = measure.maximize();
        let mut rng = StdRng::seed_from_u64(self.seed);

        let s_max = (self.max_folds as f64).log(self.eta as f64).floor() as usize;
        let mut all_results: Vec<(ParamSet, f64)> = Vec::new();

        for s in (0..=s_max).rev() {
            let n_configs = ((self.eta.pow(s as u32)) as f64 * (s_max + 1) as f64
                / (s + 1) as f64).ceil() as usize;
            let min_folds = (self.max_folds as f64 / self.eta.pow(s as u32) as f64).max(2.0) as usize;

            let mut configs: Vec<ParamSet> = (0..n_configs)
                .map(|_| self.sample_random(&mut rng))
                .collect();

            let mut folds = min_folds;

            for _i in 0..=s {
                folds = folds.min(self.max_folds).max(2);
                let cv = CrossValidation::new(folds).with_seed(self.seed);
                let mut scored: Vec<(ParamSet, f64)> = Vec::new();

                for params in &configs {
                    let mut learner = (self.factory)(params);
                    let result = benchmark::resample_regress(&mut *learner, task, &cv, &[measure])?;
                    scored.push((params.clone(), result.mean_scores()[0]));
                }

                all_results.extend(scored.iter().cloned());

                if maximize {
                    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                } else {
                    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                }

                let keep = (configs.len() / self.eta).max(1);
                configs = scored.into_iter().take(keep).map(|(p, _)| p).collect();
                folds = (folds as f64 * self.eta as f64) as usize;
            }
        }

        Ok(TuneResult::select_best(all_results, measure.id().to_string(), maximize))
    }
}
