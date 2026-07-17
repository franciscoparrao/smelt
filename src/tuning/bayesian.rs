//! Bayesian Optimization via Tree-structured Parzen Estimator (TPE).
//!
//! More efficient than Grid/Random search for expensive evaluations.
//! Models the distribution of good vs bad hyperparameter configurations
//! and samples promising candidates via density ratio l(x)/g(x).

use super::{ParamDistribution, ParamSet, ParamSpace, ParamValue, TuneResult};
use crate::Result;
use crate::benchmark;
use crate::learner::Learner;
use crate::measure::Measure;
use crate::resample::Resample;
use crate::task::{ClassificationTask, RegressionTask};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Bayesian Optimization using Tree-structured Parzen Estimator (TPE).
///
/// Builds separate density models for "good" and "bad" hyperparameter
/// regions and selects candidates that maximize the density ratio l(x)/g(x),
/// approximating Expected Improvement.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use smelt_ml::tuning::{BayesianOptimizer, ParamSpace, ParamDistribution};
/// use ndarray::array;
///
/// let features = array![
///     [0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [0.0, 0.2],
///     [0.1, 0.0], [0.2, 0.1], [0.0, 0.1], [0.1, 0.2],
///     [1.0, 1.0], [1.1, 0.9], [0.9, 1.1], [1.0, 0.9],
///     [1.1, 1.0], [0.9, 1.0], [1.0, 1.1], [1.1, 1.1]
/// ];
/// let target = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
/// let task = ClassificationTask::new("bo", features, target).unwrap();
///
/// let mut space = ParamSpace::new();
/// space.insert("max_depth".into(), ParamDistribution::Uniform(1.0, 10.0));
///
/// let bo = BayesianOptimizer::new(
///     |params| Box::new(DecisionTree::new()
///         .with_max_depth(params["max_depth"].as_usize().unwrap())),
///     space,
/// ).with_n_iter(15).with_seed(42);
///
/// let cv = CrossValidation::new(3).with_seed(42);
/// let result = bo.tune_classif(&task, &cv, &Accuracy).unwrap();
/// ```
pub struct BayesianOptimizer {
    factory: Box<dyn Fn(&ParamSet) -> Box<dyn Learner> + Send + Sync>,
    param_space: ParamSpace,
    n_iter: usize,
    n_initial: usize,
    gamma: f64,
    n_candidates: usize,
    seed: u64,
}

impl BayesianOptimizer {
    /// Create a TPE-based Bayesian optimizer over `param_space` using
    /// `factory` to build a learner from a sampled parameter set.
    pub fn new(
        factory: impl Fn(&ParamSet) -> Box<dyn Learner> + Send + Sync + 'static,
        param_space: ParamSpace,
    ) -> Self {
        Self {
            factory: Box::new(factory),
            param_space,
            n_iter: 30,
            n_initial: 5,
            gamma: 0.25,
            n_candidates: 24,
            seed: 42,
        }
    }

    /// Set the total number of optimization iterations.
    pub fn with_n_iter(mut self, n: usize) -> Self {
        self.n_iter = n;
        self
    }
    /// Set the number of initial random iterations before TPE-guided
    /// sampling begins.
    pub fn with_n_initial(mut self, n: usize) -> Self {
        self.n_initial = n;
        self
    }
    /// Set the quantile (0-1) splitting evaluated configurations into the
    /// "good" (top `gamma`) and "bad" density models.
    pub fn with_gamma(mut self, g: f64) -> Self {
        self.gamma = g;
        self
    }
    /// Set how many candidates are sampled from the "good" density model
    /// each iteration, picking the one with the best l(x)/g(x) ratio.
    pub fn with_n_candidates(mut self, n: usize) -> Self {
        self.n_candidates = n;
        self
    }
    /// Set the RNG seed for reproducible sampling.
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Tune for classification.
    ///
    /// Unlike `GridSearch`/`RandomSearch`/`Hyperband`, this loop is NOT
    /// parallelized: TPE is inherently sequential -- `sample_tpe` builds its
    /// proposal from the density of ALL previous iterations' scores, so
    /// iteration `i` cannot start before iteration `i-1`'s result is known.
    /// Evaluating in parallel would require a genuinely different batch/
    /// parallel-BO algorithm (propose a batch from the current history,
    /// evaluate it concurrently, then update history once per batch), a
    /// different design decision rather than "just connect existing
    /// parallelism", so it's left sequential here.
    pub fn tune_classif(
        &self,
        task: &ClassificationTask,
        resampling: &dyn Resample,
        measure: &dyn Measure,
    ) -> Result<TuneResult> {
        super::validate_param_space(&self.param_space)?;
        let maximize = measure.maximize();
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut history: Vec<(ParamSet, f64)> = Vec::with_capacity(self.n_iter);

        for i in 0..self.n_iter {
            let params = if i < self.n_initial || history.len() < 4 {
                self.sample_random(&mut rng)
            } else {
                self.sample_tpe(&history, maximize, &mut rng)
            };

            let mut learner = (self.factory)(&params);
            let bench = benchmark::resample_classif(&mut *learner, task, resampling, &[measure])?;
            let score = bench.mean_scores()[0];
            history.push((params, score));
        }

        TuneResult::select_best(history, measure.id().to_string(), maximize)
    }

    /// Tune for regression.
    pub fn tune_regress(
        &self,
        task: &RegressionTask,
        resampling: &dyn Resample,
        measure: &dyn Measure,
    ) -> Result<TuneResult> {
        super::validate_param_space(&self.param_space)?;
        let maximize = measure.maximize();
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut history: Vec<(ParamSet, f64)> = Vec::with_capacity(self.n_iter);

        for i in 0..self.n_iter {
            let params = if i < self.n_initial || history.len() < 4 {
                self.sample_random(&mut rng)
            } else {
                self.sample_tpe(&history, maximize, &mut rng)
            };

            let mut learner = (self.factory)(&params);
            let bench = benchmark::resample_regress(&mut *learner, task, resampling, &[measure])?;
            let score = bench.mean_scores()[0];
            history.push((params, score));
        }

        TuneResult::select_best(history, measure.id().to_string(), maximize)
    }

    /// Sample a random parameter set from the space.
    fn sample_random(&self, rng: &mut StdRng) -> ParamSet {
        super::sample_param_space(&self.param_space, rng)
    }

    /// Sample using TPE: split history into good/bad, build KDE, maximize l/g.
    fn sample_tpe(
        &self,
        history: &[(ParamSet, f64)],
        maximize: bool,
        rng: &mut StdRng,
    ) -> ParamSet {
        // Sort observations by score
        let mut sorted: Vec<(usize, f64)> = history
            .iter()
            .enumerate()
            .map(|(i, (_, s))| (i, *s))
            .collect();
        if maximize {
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        // Split into good (top gamma%) and bad (rest)
        let n_good = (sorted.len() as f64 * self.gamma).ceil().max(1.0) as usize;
        let good_indices: Vec<usize> = sorted[..n_good].iter().map(|(i, _)| *i).collect();
        let bad_indices: Vec<usize> = sorted[n_good..].iter().map(|(i, _)| *i).collect();

        // For each parameter, collect good and bad values. Sorted so the
        // KDE candidate draws consume the RNG in a process-independent
        // order (same reproducibility fix as sample_param_space).
        let mut param_names: Vec<String> = self.param_space.keys().cloned().collect();
        param_names.sort();

        // Sample n_candidates from l(x), pick the one with best l(x)/g(x)
        let mut best_params = self.sample_random(rng);
        let mut best_ratio = f64::NEG_INFINITY;

        for _ in 0..self.n_candidates {
            let candidate = self.sample_from_good(history, &good_indices, &param_names, rng);
            let log_l = self.log_density(history, &good_indices, &candidate, &param_names);
            let log_g = self.log_density(history, &bad_indices, &candidate, &param_names);
            let ratio = log_l - log_g;

            if ratio > best_ratio {
                best_ratio = ratio;
                best_params = candidate;
            }
        }

        best_params
    }

    /// Sample a candidate from the "good" distribution l(x).
    fn sample_from_good(
        &self,
        history: &[(ParamSet, f64)],
        good_indices: &[usize],
        param_names: &[String],
        rng: &mut StdRng,
    ) -> ParamSet {
        // Pick a random good observation and perturb it
        let idx = good_indices[rng.random_range(0..good_indices.len())];
        let base = &history[idx].0;

        let mut params = ParamSet::new();
        for name in param_names {
            let dist = &self.param_space[name];

            let value = match dist {
                ParamDistribution::Uniform(lo, hi) => {
                    let good_vals = numeric_values(history, good_indices, name);
                    let bw = bandwidth(&good_vals);
                    let base_v = base[name]
                        .as_f64()
                        .expect("Uniform-distributed parameter must be numeric");
                    let perturbed = base_v + rng.random_range(-bw..bw);
                    ParamValue::Float(perturbed.clamp(*lo, *hi))
                }
                ParamDistribution::LogUniform(lo, hi) => {
                    let good_vals = numeric_values(history, good_indices, name);
                    let log_vals: Vec<f64> = good_vals.iter().map(|v| v.ln()).collect();
                    let bw = bandwidth(&log_vals);
                    let log_base = base[name]
                        .as_f64()
                        .expect("LogUniform-distributed parameter must be numeric")
                        .ln();
                    let perturbed = (log_base + rng.random_range(-bw..bw)).exp();
                    ParamValue::Float(perturbed.clamp(*lo, *hi))
                }
                ParamDistribution::Choice(choices) => {
                    // Sample proportional to frequency in good set, with smoothing
                    let mut counts = vec![1.0; choices.len()]; // Laplace smoothing
                    for &i in good_indices {
                        if let Some(pos) = choices.iter().position(|c| *c == history[i].0[name]) {
                            counts[pos] += 1.0;
                        }
                    }
                    let total: f64 = counts.iter().sum();
                    let mut r = rng.random_range(0.0..total);
                    let mut selected = choices[0].clone();
                    for (i, &c) in counts.iter().enumerate() {
                        r -= c;
                        if r <= 0.0 {
                            selected = choices[i].clone();
                            break;
                        }
                    }
                    selected
                }
            };
            params.insert(name.clone(), value);
        }
        params
    }

    /// Log-density of a candidate under a KDE built from the given subset.
    fn log_density(
        &self,
        history: &[(ParamSet, f64)],
        indices: &[usize],
        candidate: &ParamSet,
        param_names: &[String],
    ) -> f64 {
        if indices.is_empty() {
            return f64::NEG_INFINITY;
        }

        let mut total_log_density = 0.0;

        for name in param_names {
            let dist = &self.param_space[name];

            let log_d = match dist {
                ParamDistribution::Uniform(lo, hi) => {
                    let vals = numeric_values(history, indices, name);
                    let x = candidate[name]
                        .as_f64()
                        .expect("Uniform-distributed parameter must be numeric");
                    let bw = bandwidth(&vals).max((*hi - *lo) * 0.01);
                    // Gaussian KDE
                    let n = vals.len() as f64;
                    let density: f64 = vals
                        .iter()
                        .map(|&v| gaussian_kernel((x - v) / bw))
                        .sum::<f64>()
                        / (n * bw);
                    (density.max(1e-300)).ln()
                }
                ParamDistribution::LogUniform(lo, hi) => {
                    let vals = numeric_values(history, indices, name);
                    let log_vals: Vec<f64> = vals.iter().map(|v| v.ln()).collect();
                    let log_x = candidate[name]
                        .as_f64()
                        .expect("LogUniform-distributed parameter must be numeric")
                        .ln();
                    let bw = bandwidth(&log_vals).max((hi.ln() - lo.ln()) * 0.01);
                    let n = log_vals.len() as f64;
                    let density: f64 = log_vals
                        .iter()
                        .map(|&v| gaussian_kernel((log_x - v) / bw))
                        .sum::<f64>()
                        / (n * bw);
                    (density.max(1e-300)).ln()
                }
                ParamDistribution::Choice(choices) => {
                    // Categorical: count frequency with smoothing
                    let mut counts = vec![1.0; choices.len()];
                    for &i in indices {
                        if let Some(pos) = choices.iter().position(|c| *c == history[i].0[name]) {
                            counts[pos] += 1.0;
                        }
                    }
                    let total: f64 = counts.iter().sum();
                    let x = &candidate[name];
                    let pos = choices.iter().position(|c| c == x).unwrap_or(0);
                    (counts[pos] / total).ln()
                }
            };

            total_log_density += log_d;
        }

        total_log_density
    }
}

// ── Helper functions ────────────────────────────────────────────────

/// Read a numeric parameter's value across a set of history indices —
/// shared by the `Uniform`/`LogUniform` arms of `sample_from_good` and
/// `log_density`, which only ever model continuous (non-`Choice`)
/// parameters and can assume every value is convertible via `as_f64`.
fn numeric_values(history: &[(ParamSet, f64)], indices: &[usize], name: &str) -> Vec<f64> {
    indices
        .iter()
        .map(|&i| {
            history[i].0[name]
                .as_f64()
                .expect("Uniform/LogUniform-distributed parameter must be numeric")
        })
        .collect()
}

/// Gaussian kernel (standard normal PDF).
#[inline]
fn gaussian_kernel(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Scott's rule bandwidth: h = σ * n^(-1/5).
fn bandwidth(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    if n <= 1.0 {
        return 1.0;
    }
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt().max(1e-10);
    std * n.powf(-0.2) // Scott's rule
}
