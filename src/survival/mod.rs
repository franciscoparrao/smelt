//! Survival Analysis: Random Survival Forest and C-index.
//!
//! For time-to-event data with right censoring.
//! RSF uses log-rank split criterion and Nelson-Aalen survival estimation.
//!
//! References:
//! - Ishwaran et al. (2008). Random survival forests. Annals of Applied Statistics.
//! - Hothorn et al. (2006). Survival ensembles. Biostatistics.

use ndarray::{Array2, ArrayView1};
use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use crate::Result;
use crate::SmeltError;

/// Survival observation: time and event indicator.
#[derive(Debug, Clone, Copy)]
pub struct SurvivalEvent {
    /// Observed time (time to event or censoring).
    pub time: f64,
    /// Event indicator: true = event occurred, false = censored.
    pub event: bool,
}

/// Predicted survival function for a single individual.
#[derive(Debug, Clone)]
pub struct SurvivalPrediction {
    /// Unique event times.
    pub times: Vec<f64>,
    /// Survival probability at each time: S(t) = P(T > t).
    pub survival: Vec<f64>,
    /// Cumulative hazard at each time: H(t).
    pub cumulative_hazard: Vec<f64>,
    /// Predicted risk score (higher = higher risk of event).
    pub risk_score: f64,
}

impl SurvivalPrediction {
    /// Predicted median survival time (time where S(t) = 0.5).
    pub fn median_survival(&self) -> Option<f64> {
        for (i, &s) in self.survival.iter().enumerate() {
            if s <= 0.5 { return Some(self.times[i]); }
        }
        None
    }

    /// Survival probability at a specific time.
    pub fn survival_at(&self, t: f64) -> f64 {
        let mut s = 1.0;
        for (i, &time) in self.times.iter().enumerate() {
            if time > t { break; }
            s = self.survival[i];
        }
        s
    }
}

/// Concordance index (C-index): measures discrimination of survival model.
/// C = P(risk_i > risk_j | T_i < T_j) for comparable pairs.
/// Range [0, 1], 0.5 = random, 1.0 = perfect.
pub fn concordance_index(
    predictions: &[SurvivalPrediction],
    events: &[SurvivalEvent],
) -> f64 {
    let n = events.len();
    let mut concordant = 0.0;
    let mut total = 0.0;

    for i in 0..n {
        if !events[i].event { continue; } // skip censored as reference
        for j in 0..n {
            if i == j { continue; }
            if events[j].time < events[i].time { continue; } // j must survive longer
            if events[j].time == events[i].time && !events[j].event { continue; }

            total += 1.0;
            if predictions[i].risk_score > predictions[j].risk_score {
                concordant += 1.0;
            } else if (predictions[i].risk_score - predictions[j].risk_score).abs() < 1e-10 {
                concordant += 0.5; // tie
            }
        }
    }

    if total > 0.0 { concordant / total } else { 0.5 }
}

// ── RSF Tree ────────────────────────────────────────────────────────

enum RSFNode {
    Leaf { hazards: Vec<(f64, f64)> }, // (time, cumulative_hazard) — Nelson-Aalen
    Split { feature: usize, threshold: f64,
            left: Box<RSFNode>, right: Box<RSFNode> },
}

impl RSFNode {
    fn find_leaf(&self, row: ArrayView1<f64>) -> &[(f64, f64)] {
        match self {
            RSFNode::Leaf { hazards } => hazards,
            RSFNode::Split { feature, threshold, left, right } => {
                if row[*feature] <= *threshold { left.find_leaf(row) }
                else { right.find_leaf(row) }
            }
        }
    }
}

/// Compute Nelson-Aalen cumulative hazard estimator.
fn nelson_aalen(events: &[SurvivalEvent], indices: &[usize]) -> Vec<(f64, f64)> {
    // Collect event times and counts
    let mut times: Vec<f64> = indices.iter()
        .filter(|&&i| events[i].event)
        .map(|&i| events[i].time)
        .collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    times.dedup();

    if times.is_empty() { return vec![(0.0, 0.0)]; }

    let mut result = Vec::with_capacity(times.len());
    let mut cum_hazard = 0.0;

    for &t in &times {
        let at_risk = indices.iter().filter(|&&i| events[i].time >= t).count() as f64;
        let n_events = indices.iter().filter(|&&i| events[i].time == t && events[i].event).count() as f64;

        if at_risk > 0.0 {
            cum_hazard += n_events / at_risk;
        }
        result.push((t, cum_hazard));
    }

    result
}

/// Log-rank split criterion for RSF.
fn log_rank_score(
    events: &[SurvivalEvent],
    left: &[usize],
    right: &[usize],
) -> f64 {
    let all: Vec<usize> = left.iter().chain(right.iter()).copied().collect();
    let mut times: Vec<f64> = all.iter().map(|&i| events[i].time).collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    times.dedup();

    let mut obs_left = 0.0;
    let mut exp_left = 0.0;
    let mut var = 0.0;

    for &t in &times {
        let d_left = left.iter().filter(|&&i| events[i].time == t && events[i].event).count() as f64;
        let d_right = right.iter().filter(|&&i| events[i].time == t && events[i].event).count() as f64;
        let n_left = left.iter().filter(|&&i| events[i].time >= t).count() as f64;
        let n_right = right.iter().filter(|&&i| events[i].time >= t).count() as f64;

        let d = d_left + d_right;
        let n = n_left + n_right;

        if n < 2.0 { continue; }

        obs_left += d_left;
        exp_left += n_left * d / n;

        if n > 1.0 {
            var += n_left * n_right * d * (n - d) / (n * n * (n - 1.0));
        }
    }

    if var > 0.0 { (obs_left - exp_left).powi(2) / var } else { 0.0 }
}

fn build_rsf_tree(
    features: &Array2<f64>,
    events: &[SurvivalEvent],
    indices: &[usize],
    max_depth: Option<usize>,
    min_node_size: usize,
    n_features: usize,
    depth: usize,
    rng: &mut impl Rng,
) -> RSFNode {
    let n = indices.len();

    if n < min_node_size * 2
        || max_depth.is_some_and(|d| depth >= d)
        || indices.iter().filter(|&&i| events[i].event).count() < 2
    {
        return RSFNode::Leaf { hazards: nelson_aalen(events, indices) };
    }

    let n_try = ((n_features as f64).sqrt().ceil() as usize).max(1);
    let mut feat_idx: Vec<usize> = (0..n_features).collect();
    for i in 0..n_try.min(n_features) {
        let j = rng.random_range(i..n_features);
        feat_idx.swap(i, j);
    }

    let mut best_score = 0.0;
    let mut best_split = None;

    for &feat in &feat_idx[..n_try.min(n_features)] {
        let mut sorted: Vec<usize> = indices.to_vec();
        sorted.sort_by(|&a, &b| features[[a, feat]].partial_cmp(&features[[b, feat]])
            .unwrap_or(std::cmp::Ordering::Equal));

        for s in min_node_size..(n.saturating_sub(min_node_size)) {
            if (features[[sorted[s], feat]] - features[[sorted[s-1], feat]]).abs() < f64::EPSILON {
                continue;
            }
            let left = &sorted[..s];
            let right = &sorted[s..];
            let score = log_rank_score(events, left, right);

            if score > best_score {
                best_score = score;
                let threshold = (features[[sorted[s-1], feat]] + features[[sorted[s], feat]]) / 2.0;
                best_split = Some((feat, threshold, left.to_vec(), right.to_vec()));
            }
        }
    }

    match best_split {
        Some((feat, threshold, left_idx, right_idx)) => {
            let left = build_rsf_tree(features, events, &left_idx, max_depth, min_node_size, n_features, depth+1, rng);
            let right = build_rsf_tree(features, events, &right_idx, max_depth, min_node_size, n_features, depth+1, rng);
            RSFNode::Split { feature: feat, threshold, left: Box::new(left), right: Box::new(right) }
        }
        None => RSFNode::Leaf { hazards: nelson_aalen(events, indices) },
    }
}

// ── Random Survival Forest ──────────────────────────────────────────

/// Random Survival Forest for time-to-event prediction.
///
/// # Examples
///
/// ```
/// use smelt_ml::survival::{RandomSurvivalForest, SurvivalEvent, concordance_index};
/// use ndarray::array;
///
/// let features = array![
///     [25.0, 0.0], [30.0, 1.0], [35.0, 0.0], [40.0, 1.0],
///     [50.0, 0.0], [55.0, 1.0], [60.0, 0.0], [65.0, 1.0],
/// ];
/// let events = vec![
///     SurvivalEvent { time: 10.0, event: true },
///     SurvivalEvent { time: 15.0, event: false },
///     SurvivalEvent { time: 8.0, event: true },
///     SurvivalEvent { time: 20.0, event: false },
///     SurvivalEvent { time: 5.0, event: true },
///     SurvivalEvent { time: 12.0, event: true },
///     SurvivalEvent { time: 3.0, event: true },
///     SurvivalEvent { time: 7.0, event: false },
/// ];
///
/// let rsf = RandomSurvivalForest::new().with_n_estimators(50).with_seed(42);
/// let predictions = rsf.fit_predict(&features, &events).unwrap();
/// let c_index = concordance_index(&predictions, &events);
/// ```
pub struct RandomSurvivalForest {
    n_estimators: usize,
    max_depth: Option<usize>,
    min_node_size: usize,
    seed: u64,
}

impl Default for RandomSurvivalForest {
    fn default() -> Self {
        Self { n_estimators: 100, max_depth: None, min_node_size: 6, seed: 42 }
    }
}

impl RandomSurvivalForest {
    pub fn new() -> Self { Self::default() }
    pub fn with_n_estimators(mut self, n: usize) -> Self { self.n_estimators = n; self }
    pub fn with_max_depth(mut self, d: usize) -> Self { self.max_depth = Some(d); self }
    pub fn with_min_node_size(mut self, n: usize) -> Self { self.min_node_size = n; self }
    pub fn with_seed(mut self, s: u64) -> Self { self.seed = s; self }

    /// Fit the forest and predict survival functions for each sample.
    pub fn fit_predict(
        &self,
        features: &Array2<f64>,
        events: &[SurvivalEvent],
    ) -> Result<Vec<SurvivalPrediction>> {
        let n = features.nrows();
        let nf = features.ncols();

        if events.len() != n {
            return Err(SmeltError::DimensionMismatch { expected: n, got: events.len() });
        }

        // Build trees
        let trees: Vec<RSFNode> = (0..self.n_estimators)
            .into_par_iter()
            .map(|i| {
                let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(i as u64));
                let indices: Vec<usize> = (0..n).map(|_| rng.random_range(0..n)).collect();
                build_rsf_tree(features, events, &indices, self.max_depth,
                    self.min_node_size, nf, 0, &mut rng)
            })
            .collect();

        // Predict: average cumulative hazard across trees
        let mut predictions = Vec::with_capacity(n);

        // Collect all unique event times
        let mut all_times: Vec<f64> = events.iter().filter(|e| e.event).map(|e| e.time).collect();
        all_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        all_times.dedup();

        for i in 0..n {
            let row = features.row(i);

            // Average cumulative hazard from all trees at each time point
            let mut cum_hazard_at_times = vec![0.0; all_times.len()];

            for tree in &trees {
                let leaf_hazards = tree.find_leaf(row);
                for (ti, &t) in all_times.iter().enumerate() {
                    // Find cumulative hazard at time t in this tree's leaf
                    let mut h = 0.0;
                    for &(ht, hv) in leaf_hazards {
                        if ht <= t { h = hv; } else { break; }
                    }
                    cum_hazard_at_times[ti] += h;
                }
            }

            let n_trees = self.n_estimators as f64;
            let cumulative_hazard: Vec<f64> = cum_hazard_at_times.iter().map(|&h| h / n_trees).collect();
            let survival: Vec<f64> = cumulative_hazard.iter().map(|&h| (-h).exp()).collect();
            let risk_score = cumulative_hazard.last().copied().unwrap_or(0.0);

            predictions.push(SurvivalPrediction {
                times: all_times.clone(),
                survival,
                cumulative_hazard,
                risk_score,
            });
        }

        Ok(predictions)
    }
}
