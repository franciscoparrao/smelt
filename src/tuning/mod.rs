//! Hyperparameter tuning: grid search, random search.

pub mod bayesian;
pub mod grid_search;
pub mod hyperband;
pub mod random_search;

pub use bayesian::BayesianOptimizer;
pub use grid_search::GridSearch;
pub use hyperband::Hyperband;
pub use random_search::RandomSearch;

use crate::{Result, SmeltError};
use rand::Rng;
use std::collections::HashMap;

/// A single hyperparameter value (audit issue M10: `ParamSet` used to be a
/// bare `HashMap<String, f64>`, forcing every hyperparameter through `f64` --
/// a factory closure read an integer via `params["max_depth"] as usize`, and
/// a string-valued hyperparameter like an `objective`/`variogram_model`
/// choice had no representation at all). `Float`/`Int`/`Bool` interconvert
/// via the `as_*` accessors (matching the old cast-based access pattern,
/// including `as_usize`'s truncation of a `Float` for behavioral parity with
/// existing seeded tuning runs); `Str` only accepts `as_str`.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamValue {
    /// A floating-point value (e.g. `learning_rate`).
    Float(f64),
    /// An integer value (e.g. `max_depth`, `n_estimators`).
    Int(i64),
    /// A boolean flag.
    Bool(bool),
    /// A string enum choice (e.g. `objective`, `variogram_model`).
    Str(String),
}

impl ParamValue {
    /// Read as `f64`. `Int`/`Bool` convert; `Str` errors.
    pub fn as_f64(&self) -> Result<f64> {
        match self {
            Self::Float(v) => Ok(*v),
            Self::Int(v) => Ok(*v as f64),
            Self::Bool(v) => Ok(if *v { 1.0 } else { 0.0 }),
            Self::Str(s) => Err(SmeltError::InvalidParameter(format!(
                "expected a numeric parameter value, got string \"{s}\""
            ))),
        }
    }

    /// Read as `usize`. A `Float` truncates the same way the old bare
    /// `params[key] as usize` cast did; `Int` rejects negative values
    /// instead of silently wrapping.
    pub fn as_usize(&self) -> Result<usize> {
        match self {
            Self::Int(v) => usize::try_from(*v).map_err(|_| {
                SmeltError::InvalidParameter(format!(
                    "parameter value {v} is negative, cannot convert to usize"
                ))
            }),
            Self::Float(v) => Ok(*v as usize),
            Self::Bool(v) => Ok(*v as usize),
            Self::Str(s) => Err(SmeltError::InvalidParameter(format!(
                "expected a numeric parameter value, got string \"{s}\""
            ))),
        }
    }

    /// Read as `i64`.
    pub fn as_i64(&self) -> Result<i64> {
        match self {
            Self::Int(v) => Ok(*v),
            Self::Float(v) => Ok(*v as i64),
            Self::Bool(v) => Ok(*v as i64),
            Self::Str(s) => Err(SmeltError::InvalidParameter(format!(
                "expected a numeric parameter value, got string \"{s}\""
            ))),
        }
    }

    /// Read as `bool`. `Int`/`Float` treat nonzero as `true`.
    pub fn as_bool(&self) -> Result<bool> {
        match self {
            Self::Bool(v) => Ok(*v),
            Self::Int(v) => Ok(*v != 0),
            Self::Float(v) => Ok(*v != 0.0),
            Self::Str(s) => Err(SmeltError::InvalidParameter(format!(
                "expected a boolean parameter value, got string \"{s}\""
            ))),
        }
    }

    /// Read as a string slice. Only valid for `Str`.
    pub fn as_str(&self) -> Result<&str> {
        match self {
            Self::Str(s) => Ok(s),
            other => Err(SmeltError::InvalidParameter(format!(
                "expected a string parameter value, got {other:?}"
            ))),
        }
    }
}

impl From<f64> for ParamValue {
    fn from(v: f64) -> Self {
        Self::Float(v)
    }
}
impl From<i64> for ParamValue {
    fn from(v: i64) -> Self {
        Self::Int(v)
    }
}
impl From<usize> for ParamValue {
    fn from(v: usize) -> Self {
        Self::Int(v as i64)
    }
}
impl From<bool> for ParamValue {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}
impl From<String> for ParamValue {
    fn from(v: String) -> Self {
        Self::Str(v)
    }
}
impl From<&str> for ParamValue {
    fn from(v: &str) -> Self {
        Self::Str(v.to_string())
    }
}

/// A single set of hyperparameter values.
pub type ParamSet = HashMap<String, ParamValue>;

/// A grid of hyperparameter values for exhaustive search.
pub type ParamGrid = HashMap<String, Vec<ParamValue>>;

/// Distribution for sampling hyperparameters. `Uniform`/`LogUniform` always
/// sample a `ParamValue::Float`; `Choice` can hold any mix of value types
/// (e.g. a string enum choice), sampled uniformly at random.
#[derive(Clone)]
pub enum ParamDistribution {
    /// Uniform distribution over [low, high].
    Uniform(f64, f64),
    /// Log-uniform distribution: 10^Uniform(log10(low), log10(high)).
    LogUniform(f64, f64),
    /// Choose from a fixed set of values.
    Choice(Vec<ParamValue>),
}

/// A space of hyperparameter distributions for random search.
pub type ParamSpace = HashMap<String, ParamDistribution>;

/// Sample one `ParamSet` from `space` — shared by `RandomSearch`,
/// `BayesianOptimizer` (initial/random rounds), and `Hyperband`, which
/// previously each duplicated this same match-on-`ParamDistribution` logic.
///
/// Keys are sorted before drawing (same convention as `cartesian_product`):
/// iterating the `HashMap` directly would assign the RNG's draws to
/// parameters in `RandomState` order, which differs per process -- so the
/// same seed produced different configurations across runs, breaking the
/// reproducibility `with_seed` promises.
pub(crate) fn sample_param_space(space: &ParamSpace, rng: &mut impl Rng) -> ParamSet {
    let mut keys: Vec<&String> = space.keys().collect();
    keys.sort();
    keys.into_iter()
        .map(|name| (name.clone(), sample_one(&space[name], rng)))
        .collect()
}

/// Sample a single value from one `ParamDistribution`.
pub(crate) fn sample_one(dist: &ParamDistribution, rng: &mut impl Rng) -> ParamValue {
    match dist {
        ParamDistribution::Uniform(lo, hi) => ParamValue::Float(rng.random_range(*lo..=*hi)),
        ParamDistribution::LogUniform(lo, hi) => {
            let log_lo = lo.log10();
            let log_hi = hi.log10();
            ParamValue::Float(10.0f64.powf(rng.random_range(log_lo..=log_hi)))
        }
        ParamDistribution::Choice(values) => values[rng.random_range(0..values.len())].clone(),
    }
}

/// Result of a tuning run.
#[derive(Debug)]
pub struct TuneResult {
    /// Best hyperparameter configuration found.
    pub best_params: ParamSet,
    /// Score of the best configuration.
    pub best_score: f64,
    /// All evaluated configurations with their scores.
    pub all_results: Vec<(ParamSet, f64)>,
    /// Measure used for evaluation.
    pub measure_id: String,
    /// Whether higher scores are better.
    pub maximize: bool,
}

impl TuneResult {
    pub(crate) fn select_best(
        results: Vec<(ParamSet, f64)>,
        measure_id: String,
        maximize: bool,
    ) -> Result<Self> {
        if results.is_empty() {
            return Err(SmeltError::InvalidParameter(
                "tuning produced no candidates to select from (n_iter=0 or an empty grid?)".into(),
            ));
        }
        let best_idx = if maximize {
            results
                .iter()
                .enumerate()
                .max_by(|a, b| {
                    a.1.1
                        .partial_cmp(&b.1.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .expect("checked non-empty above")
                .0
        } else {
            results
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    a.1.1
                        .partial_cmp(&b.1.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .expect("checked non-empty above")
                .0
        };

        Ok(Self {
            best_params: results[best_idx].0.clone(),
            best_score: results[best_idx].1,
            all_results: results,
            measure_id,
            maximize,
        })
    }
}

/// Generate the Cartesian product of all parameter values.
pub(crate) fn cartesian_product(grid: &ParamGrid) -> Vec<ParamSet> {
    let mut keys: Vec<&String> = grid.keys().collect();
    keys.sort(); // deterministic order
    let values: Vec<&Vec<ParamValue>> = keys.iter().map(|k| &grid[*k]).collect();

    if keys.is_empty() {
        return vec![ParamSet::new()];
    }

    let mut result = Vec::new();
    let mut indices = vec![0usize; keys.len()];

    loop {
        let mut params = ParamSet::new();
        for (i, key) in keys.iter().enumerate() {
            params.insert((*key).clone(), values[i][indices[i]].clone());
        }
        result.push(params);

        let mut carry = true;
        for i in (0..keys.len()).rev() {
            if carry {
                indices[i] += 1;
                if indices[i] >= values[i].len() {
                    indices[i] = 0;
                } else {
                    carry = false;
                }
            }
        }
        if carry {
            break;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// Regression test (4th audit, HIGH-6): the draw→parameter assignment
    /// must not depend on `HashMap` iteration order (per-process
    /// `RandomState`), or the same seed samples different configurations
    /// across runs. Two spaces with identical distributions but different
    /// insertion orders and capacities must sample identically.
    #[test]
    fn sample_param_space_is_independent_of_hashmap_iteration_order() {
        let names: Vec<String> = (0..10).map(|i| format!("param_{i}")).collect();

        let mut space_a = ParamSpace::new();
        for (i, n) in names.iter().enumerate() {
            space_a.insert(n.clone(), ParamDistribution::Uniform(0.0, (i + 1) as f64));
        }
        let mut space_b = ParamSpace::with_capacity(512);
        for (i, n) in names.iter().enumerate().rev() {
            space_b.insert(n.clone(), ParamDistribution::Uniform(0.0, (i + 1) as f64));
        }

        let sampled_a = sample_param_space(&space_a, &mut StdRng::seed_from_u64(42));
        let sampled_b = sample_param_space(&space_b, &mut StdRng::seed_from_u64(42));
        for n in &names {
            assert_eq!(
                sampled_a[n], sampled_b[n],
                "{n}: same seed must assign the same draw regardless of map layout"
            );
        }
    }
}
