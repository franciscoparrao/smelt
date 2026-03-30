//! Hyperparameter tuning: grid search, random search.

pub mod grid_search;
pub mod random_search;
pub mod bayesian;

pub use grid_search::GridSearch;
pub use random_search::RandomSearch;
pub use bayesian::BayesianOptimizer;

use std::collections::HashMap;

/// A single set of hyperparameter values.
pub type ParamSet = HashMap<String, f64>;

/// A grid of hyperparameter values for exhaustive search.
pub type ParamGrid = HashMap<String, Vec<f64>>;

/// Distribution for sampling hyperparameters.
#[derive(Clone)]
pub enum ParamDistribution {
    /// Uniform distribution over [low, high].
    Uniform(f64, f64),
    /// Log-uniform distribution: 10^Uniform(log10(low), log10(high)).
    LogUniform(f64, f64),
    /// Choose from a fixed set of values.
    Choice(Vec<f64>),
}

/// A space of hyperparameter distributions for random search.
pub type ParamSpace = HashMap<String, ParamDistribution>;

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
    ) -> Self {
        let best_idx = if maximize {
            results.iter().enumerate()
                .max_by(|a, b| a.1.1.partial_cmp(&b.1.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap().0
        } else {
            results.iter().enumerate()
                .min_by(|a, b| a.1.1.partial_cmp(&b.1.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap().0
        };

        Self {
            best_params: results[best_idx].0.clone(),
            best_score: results[best_idx].1,
            all_results: results,
            measure_id,
            maximize,
        }
    }
}

/// Generate the Cartesian product of all parameter values.
pub(crate) fn cartesian_product(grid: &ParamGrid) -> Vec<ParamSet> {
    let mut keys: Vec<&String> = grid.keys().collect();
    keys.sort(); // deterministic order
    let values: Vec<&Vec<f64>> = keys.iter().map(|k| &grid[*k]).collect();

    if keys.is_empty() {
        return vec![ParamSet::new()];
    }

    let mut result = Vec::new();
    let mut indices = vec![0usize; keys.len()];

    loop {
        let mut params = ParamSet::new();
        for (i, key) in keys.iter().enumerate() {
            params.insert((*key).clone(), values[i][indices[i]]);
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
