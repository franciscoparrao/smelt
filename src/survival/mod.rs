//! Survival Analysis: Random Survival Forest and C-index.
//!
//! For time-to-event data with right censoring.
//! RSF uses log-rank split criterion and Nelson-Aalen survival estimation.
//!
//! References:
//! - Ishwaran et al. (2008). Random survival forests. Annals of Applied Statistics.
//! - Hothorn et al. (2006). Survival ensembles. Biostatistics.

use crate::Result;
use crate::SmeltError;
use ndarray::{Array2, ArrayView1};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;

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
            if s <= 0.5 {
                return Some(self.times[i]);
            }
        }
        None
    }

    /// Survival probability at a specific time.
    pub fn survival_at(&self, t: f64) -> f64 {
        let mut s = 1.0;
        for (i, &time) in self.times.iter().enumerate() {
            if time > t {
                break;
            }
            s = self.survival[i];
        }
        s
    }
}

/// Concordance index (C-index): measures discrimination of survival model.
/// C = P(risk_i > risk_j | T_i < T_j) for comparable pairs.
/// Range [0, 1], 0.5 = random, 1.0 = perfect.
///
/// Each unordered pair `{i, j}` is visited exactly once (`i < j`). A
/// previous implementation looped `i, j` over all ordered pairs with `i`
/// restricted to events, which double-counted every pair where both `i` and
/// `j` had an event at the exact same time (nothing skipped either
/// direction), inflating `total` without inflating `concordant` to match —
/// dragging the score toward 0.5 whenever tied event times were common.
pub fn concordance_index(predictions: &[SurvivalPrediction], events: &[SurvivalEvent]) -> f64 {
    let n = events.len();
    let mut concordant = 0.0;
    let mut total = 0.0;

    for i in 0..n {
        for j in (i + 1)..n {
            let (ti, ei) = (events[i].time, events[i].event);
            let (tj, ej) = (events[j].time, events[j].event);

            if ti == tj {
                if !ei || !ej {
                    // Both censored, or one censored exactly at the other's
                    // event time: no way to establish which failed first.
                    continue;
                }
                // Both had the event at the exact same time: comparable,
                // but time alone gives no order to check risk against —
                // credit as a tie rather than picking a direction
                // arbitrarily by array index.
                total += 1.0;
                concordant += 0.5;
                continue;
            }

            let (earlier, later) = if ti < tj { (i, j) } else { (j, i) };
            if !events[earlier].event {
                continue; // the earlier observation must be an event, not a censor
            }

            total += 1.0;
            let diff = predictions[earlier].risk_score - predictions[later].risk_score;
            if diff.abs() < 1e-10 {
                concordant += 0.5;
            } else if diff > 0.0 {
                concordant += 1.0;
            }
        }
    }

    if total > 0.0 { concordant / total } else { 0.5 }
}

// ── RSF Tree ────────────────────────────────────────────────────────

enum RSFNode {
    Leaf {
        hazards: Vec<(f64, f64)>,
    }, // (time, cumulative_hazard) — Nelson-Aalen
    Split {
        feature: usize,
        threshold: f64,
        left: Box<RSFNode>,
        right: Box<RSFNode>,
    },
}

impl RSFNode {
    fn find_leaf(&self, row: ArrayView1<f64>) -> &[(f64, f64)] {
        match self {
            RSFNode::Leaf { hazards } => hazards,
            RSFNode::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                if row[*feature] <= *threshold {
                    left.find_leaf(row)
                } else {
                    right.find_leaf(row)
                }
            }
        }
    }
}

/// Compute Nelson-Aalen cumulative hazard estimator.
fn nelson_aalen(events: &[SurvivalEvent], indices: &[usize]) -> Vec<(f64, f64)> {
    // Collect event times and counts
    let mut times: Vec<f64> = indices
        .iter()
        .filter(|&&i| events[i].event)
        .map(|&i| events[i].time)
        .collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    times.dedup();

    if times.is_empty() {
        return vec![(0.0, 0.0)];
    }

    let mut result = Vec::with_capacity(times.len());
    let mut cum_hazard = 0.0;

    for &t in &times {
        let at_risk = indices.iter().filter(|&&i| events[i].time >= t).count() as f64;
        let n_events = indices
            .iter()
            .filter(|&&i| events[i].time == t && events[i].event)
            .count() as f64;

        if at_risk > 0.0 {
            cum_hazard += n_events / at_risk;
        }
        result.push((t, cum_hazard));
    }

    result
}

/// Log-rank split criterion for RSF.
fn log_rank_score(events: &[SurvivalEvent], left: &[usize], right: &[usize]) -> f64 {
    let all: Vec<usize> = left.iter().chain(right.iter()).copied().collect();
    let mut times: Vec<f64> = all.iter().map(|&i| events[i].time).collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    times.dedup();

    let mut obs_left = 0.0;
    let mut exp_left = 0.0;
    let mut var = 0.0;

    for &t in &times {
        let d_left = left
            .iter()
            .filter(|&&i| events[i].time == t && events[i].event)
            .count() as f64;
        let d_right = right
            .iter()
            .filter(|&&i| events[i].time == t && events[i].event)
            .count() as f64;
        let n_left = left.iter().filter(|&&i| events[i].time >= t).count() as f64;
        let n_right = right.iter().filter(|&&i| events[i].time >= t).count() as f64;

        let d = d_left + d_right;
        let n = n_left + n_right;

        if n < 2.0 {
            continue;
        }

        obs_left += d_left;
        exp_left += n_left * d / n;

        if n > 1.0 {
            var += n_left * n_right * d * (n - d) / (n * n * (n - 1.0));
        }
    }

    if var > 0.0 {
        (obs_left - exp_left).powi(2) / var
    } else {
        0.0
    }
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
        return RSFNode::Leaf {
            hazards: nelson_aalen(events, indices),
        };
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
        sorted.sort_by(|&a, &b| {
            features[[a, feat]]
                .partial_cmp(&features[[b, feat]])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for s in min_node_size..(n.saturating_sub(min_node_size)) {
            if (features[[sorted[s], feat]] - features[[sorted[s - 1], feat]]).abs() < f64::EPSILON
            {
                continue;
            }
            let left = &sorted[..s];
            let right = &sorted[s..];
            let score = log_rank_score(events, left, right);

            if score > best_score {
                best_score = score;
                let threshold =
                    (features[[sorted[s - 1], feat]] + features[[sorted[s], feat]]) / 2.0;
                best_split = Some((feat, threshold, left.to_vec(), right.to_vec()));
            }
        }
    }

    match best_split {
        Some((feat, threshold, left_idx, right_idx)) => {
            let left = build_rsf_tree(
                features,
                events,
                &left_idx,
                max_depth,
                min_node_size,
                n_features,
                depth + 1,
                rng,
            );
            let right = build_rsf_tree(
                features,
                events,
                &right_idx,
                max_depth,
                min_node_size,
                n_features,
                depth + 1,
                rng,
            );
            RSFNode::Split {
                feature: feat,
                threshold,
                left: Box::new(left),
                right: Box::new(right),
            }
        }
        None => RSFNode::Leaf {
            hazards: nelson_aalen(events, indices),
        },
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
        Self {
            n_estimators: 100,
            max_depth: None,
            min_node_size: 6,
            seed: 42,
        }
    }
}

impl RandomSurvivalForest {
    /// Creates a forest with default hyperparameters (100 trees, unbounded
    /// depth, min node size 6, seed 42).
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets the number of survival trees in the ensemble.
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }
    /// Sets the maximum depth of each survival tree.
    pub fn with_max_depth(mut self, d: usize) -> Self {
        self.max_depth = Some(d);
        self
    }
    /// Sets the minimum number of samples required in a leaf node.
    pub fn with_min_node_size(mut self, n: usize) -> Self {
        self.min_node_size = n;
        self
    }
    /// Sets the RNG seed used for bootstrap sampling across trees.
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Fit the forest, keeping it for prediction on new data.
    ///
    /// Also returns an out-of-bag (OOB) concordance index: each training
    /// sample's risk score is averaged only over the trees whose bootstrap
    /// sample excluded it, so the C-index reflects genuine generalization
    /// instead of the optimistic in-sample score `concordance_index` gives
    /// when applied to `fit_predict`'s output (every tree saw every
    /// training row it predicts on there).
    pub fn fit(
        &self,
        features: &Array2<f64>,
        events: &[SurvivalEvent],
    ) -> Result<(TrainedRandomSurvivalForest, f64)> {
        let n = features.nrows();
        let nf = features.ncols();

        if events.len() != n {
            return Err(SmeltError::DimensionMismatch {
                expected: n,
                got: events.len(),
            });
        }

        // Build trees, tracking each tree's in-bag indicator so OOB
        // predictions can be formed below.
        let tree_results: Vec<(RSFNode, Vec<bool>)> = (0..self.n_estimators)
            .into_par_iter()
            .map(|i| {
                let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(i as u64));
                let indices: Vec<usize> = (0..n).map(|_| rng.random_range(0..n)).collect();
                let mut in_bag = vec![false; n];
                for &idx in &indices {
                    in_bag[idx] = true;
                }
                let tree = build_rsf_tree(
                    features,
                    events,
                    &indices,
                    self.max_depth,
                    self.min_node_size,
                    nf,
                    0,
                    &mut rng,
                );
                (tree, in_bag)
            })
            .collect();

        let mut trees = Vec::with_capacity(self.n_estimators);
        let mut in_bags = Vec::with_capacity(self.n_estimators);
        for (tree, in_bag) in tree_results {
            trees.push(tree);
            in_bags.push(in_bag);
        }

        // Collect all unique event times -- the common grid every
        // prediction's survival/hazard curve is reported on.
        let mut event_times: Vec<f64> = events.iter().filter(|e| e.event).map(|e| e.time).collect();
        event_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        event_times.dedup();

        // OOB risk score per training sample: only trees where this sample
        // was NOT in the bootstrap draw contribute.
        let oob_predictions: Vec<SurvivalPrediction> = (0..n)
            .map(|i| {
                let row = features.row(i);
                let oob_trees = trees
                    .iter()
                    .zip(&in_bags)
                    .filter(|(_, in_bag)| !in_bag[i])
                    .map(|(tree, _)| tree);
                let (hazard_sum, n_trees) = sum_hazard_from_trees(oob_trees, row, &event_times);
                survival_prediction_from_hazard_sum(&event_times, hazard_sum, n_trees)
            })
            .collect();
        let oob_c_index = concordance_index(&oob_predictions, events);

        Ok((
            TrainedRandomSurvivalForest { trees, event_times },
            oob_c_index,
        ))
    }

    /// Fit the forest and predict survival functions for each training
    /// sample (in-sample: every tree that saw a sample during its bootstrap
    /// draw still contributes to that sample's prediction here). For an
    /// honest estimate of generalization, or to predict genuinely new data,
    /// use [`RandomSurvivalForest::fit`] instead.
    pub fn fit_predict(
        &self,
        features: &Array2<f64>,
        events: &[SurvivalEvent],
    ) -> Result<Vec<SurvivalPrediction>> {
        let (trained, _oob_c_index) = self.fit(features, events)?;
        Ok(trained.predict(features))
    }
}

/// Sums cumulative hazard across `trees` at each of `times`, for the leaf
/// `row` falls into in each tree. Returns `(sum_per_time, n_trees_summed)`.
fn sum_hazard_from_trees<'a>(
    trees: impl Iterator<Item = &'a RSFNode>,
    row: ArrayView1<f64>,
    times: &[f64],
) -> (Vec<f64>, usize) {
    let mut sum = vec![0.0; times.len()];
    let mut n_trees = 0usize;
    for tree in trees {
        n_trees += 1;
        let leaf_hazards = tree.find_leaf(row);
        for (ti, &t) in times.iter().enumerate() {
            let mut h = 0.0;
            for &(ht, hv) in leaf_hazards {
                if ht <= t {
                    h = hv;
                } else {
                    break;
                }
            }
            sum[ti] += h;
        }
    }
    (sum, n_trees)
}

/// Turns a per-time hazard sum (from [`sum_hazard_from_trees`]) into a full
/// [`SurvivalPrediction`] by averaging over however many trees contributed.
/// `n_trees == 0` (every tree happened to have this sample in-bag -- only
/// plausible with very few trees) falls back to a flat zero-hazard/100%-
/// survival curve rather than dividing by zero.
fn survival_prediction_from_hazard_sum(
    times: &[f64],
    hazard_sum: Vec<f64>,
    n_trees: usize,
) -> SurvivalPrediction {
    let denom = n_trees.max(1) as f64;
    let cumulative_hazard: Vec<f64> = hazard_sum.iter().map(|&h| h / denom).collect();
    let survival: Vec<f64> = cumulative_hazard.iter().map(|&h| (-h).exp()).collect();
    let risk_score = cumulative_hazard.last().copied().unwrap_or(0.0);
    SurvivalPrediction {
        times: times.to_vec(),
        survival,
        cumulative_hazard,
        risk_score,
    }
}

/// A fitted [`RandomSurvivalForest`], retained so it can predict on new
/// (not just training) data -- unlike `fit_predict`, which discarded the
/// forest after producing in-sample predictions.
pub struct TrainedRandomSurvivalForest {
    trees: Vec<RSFNode>,
    event_times: Vec<f64>,
}

impl TrainedRandomSurvivalForest {
    /// Predicts survival functions for `features` (new data, or the
    /// training data) by averaging cumulative hazard across every tree in
    /// the forest.
    pub fn predict(&self, features: &Array2<f64>) -> Vec<SurvivalPrediction> {
        (0..features.nrows())
            .map(|i| {
                let row = features.row(i);
                let (hazard_sum, n_trees) =
                    sum_hazard_from_trees(self.trees.iter(), row, &self.event_times);
                survival_prediction_from_hazard_sum(&self.event_times, hazard_sum, n_trees)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn synthetic_survival_data(n: usize, seed: u64) -> (Array2<f64>, Vec<SurvivalEvent>) {
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut features = Array2::zeros((n, 2));
        let mut events = Vec::with_capacity(n);
        for i in 0..n {
            let x0: f64 = rng.random_range(0.0..10.0);
            let x1: f64 = rng.random_range(0.0..10.0);
            features[[i, 0]] = x0;
            features[[i, 1]] = x1;
            // Higher x0 -> shorter survival time (higher risk).
            let base_time = 20.0 - x0 + rng.random_range(-1.0..1.0);
            let censor_time: f64 = rng.random_range(0.0..25.0);
            if base_time <= censor_time {
                events.push(SurvivalEvent {
                    time: base_time.max(0.1),
                    event: true,
                });
            } else {
                events.push(SurvivalEvent {
                    time: censor_time.max(0.1),
                    event: false,
                });
            }
        }
        (features, events)
    }

    /// Regression test for HIGH-15: previously only `fit_predict` existed,
    /// which discards the forest after producing in-sample predictions --
    /// there was no way to predict a genuinely new (held-out) sample. This
    /// checks `fit()` returns a model that can score new feature rows never
    /// seen during training.
    #[test]
    fn fit_returns_a_model_that_predicts_new_unseen_data() {
        let (features, events) = synthetic_survival_data(60, 7);
        let rsf = RandomSurvivalForest::new()
            .with_n_estimators(30)
            .with_seed(1);
        let (trained, _oob_c_index) = rsf.fit(&features, &events).unwrap();

        // New data, not part of the training set at all.
        let new_features = Array2::from_shape_vec((2, 2), vec![9.0, 1.0, 1.0, 9.0]).unwrap();
        let preds = trained.predict(&new_features);
        assert_eq!(preds.len(), 2);
        // High x0 (row 0) -> higher risk than low x0 (row 1), matching the
        // synthetic data-generating process (higher x0 -> shorter survival).
        assert!(
            preds[0].risk_score > preds[1].risk_score,
            "high-x0 new sample should have a higher predicted risk score than low-x0: {} vs {}",
            preds[0].risk_score,
            preds[1].risk_score
        );
    }

    /// Regression test for HIGH-15's second half: the C-index reported
    /// alongside `fit_predict` was in-sample (every tree saw every row it
    /// predicts on). The OOB C-index instead only aggregates a sample's
    /// prediction from trees that did NOT have it in their bootstrap draw,
    /// so it should be noticeably less optimistic than plugging
    /// `fit_predict`'s in-sample predictions into `concordance_index`
    /// directly -- if the forest can perfectly overfit small training data
    /// (spuriously separating essentially every pair), in-sample C-index
    /// tends toward 1.0 while the honest OOB estimate does not.
    #[test]
    fn oob_c_index_is_less_optimistic_than_in_sample_c_index() {
        let (features, events) = synthetic_survival_data(40, 11);
        let rsf = RandomSurvivalForest::new()
            .with_n_estimators(50)
            .with_min_node_size(1) // encourage overfitting on this small n
            .with_seed(2);

        let (trained, oob_c_index) = rsf.fit(&features, &events).unwrap();
        let in_sample_predictions = trained.predict(&features);
        let in_sample_c_index = concordance_index(&in_sample_predictions, &events);

        assert!(
            oob_c_index <= in_sample_c_index + 1e-9,
            "OOB C-index ({oob_c_index}) should not exceed the in-sample C-index \
             ({in_sample_c_index}) -- in-sample is optimistic by construction"
        );
        assert!(
            in_sample_c_index > oob_c_index,
            "with min_node_size=1 the forest should overfit enough that in-sample \
             ({in_sample_c_index}) is measurably higher than OOB ({oob_c_index})"
        );
    }

    /// `fit_predict` must keep its exact previous (in-sample, all-trees)
    /// behavior -- it's built on top of the new `fit`/`predict` split, and
    /// existing callers depend on its numeric output being unchanged.
    #[test]
    fn fit_predict_matches_fit_then_predict_on_training_data() {
        let (features, events) = synthetic_survival_data(30, 5);
        let rsf = RandomSurvivalForest::new()
            .with_n_estimators(20)
            .with_seed(9);

        let via_fit_predict = rsf.fit_predict(&features, &events).unwrap();
        let (trained, _) = rsf.fit(&features, &events).unwrap();
        let via_fit_then_predict = trained.predict(&features);

        assert_eq!(via_fit_predict.len(), via_fit_then_predict.len());
        for (a, b) in via_fit_predict.iter().zip(&via_fit_then_predict) {
            assert!(
                (a.risk_score - b.risk_score).abs() < 1e-9,
                "fit_predict's risk score ({}) should exactly match fit().predict() ({})",
                a.risk_score,
                b.risk_score
            );
        }
    }

    fn prediction_with_risk(risk_score: f64) -> SurvivalPrediction {
        SurvivalPrediction {
            times: vec![],
            survival: vec![],
            cumulative_hazard: vec![],
            risk_score,
        }
    }

    #[test]
    fn concordance_index_no_ties_perfect_discrimination() {
        let events = vec![
            SurvivalEvent {
                time: 1.0,
                event: true,
            },
            SurvivalEvent {
                time: 5.0,
                event: true,
            },
            SurvivalEvent {
                time: 10.0,
                event: true,
            },
            SurvivalEvent {
                time: 20.0,
                event: true,
            },
        ];
        // Risk strictly decreasing as time increases -> perfect discrimination.
        let predictions: Vec<_> = [4.0, 3.0, 2.0, 1.0]
            .into_iter()
            .map(prediction_with_risk)
            .collect();
        let c = concordance_index(&predictions, &events);
        assert!(
            (c - 1.0).abs() < 1e-12,
            "expected perfect C-index 1.0, got {c}"
        );
    }

    /// Regression test for N11 (docs/auditoria_motor_2026-07-05.md, Fase F):
    /// two subjects with the exact same event time (idx 2, 3) used to be
    /// visited in both `(i, j)` orders by the old "loop i over events, loop
    /// j over everyone" implementation -- neither direction's skip
    /// conditions fired for a tied-event/tied-event pair, so it counted
    /// twice (`total += 2`) instead of once. That double weight (relative
    /// to every other, correctly single-counted pair) drags the aggregate
    /// C-index toward 0.5 whenever tied event times are common.
    #[test]
    fn concordance_index_does_not_double_count_tied_event_times() {
        let events = vec![
            SurvivalEvent {
                time: 1.0,
                event: true,
            },
            SurvivalEvent {
                time: 10.0,
                event: true,
            },
            SurvivalEvent {
                time: 5.0,
                event: true,
            }, // tied with idx 3
            SurvivalEvent {
                time: 5.0,
                event: true,
            }, // tied with idx 2
        ];
        let predictions: Vec<_> = [1.0, 0.0, 0.9, 0.1]
            .into_iter()
            .map(prediction_with_risk)
            .collect();

        let c = concordance_index(&predictions, &events);
        // 6 unordered pairs total: 5 strictly concordant (risk fully
        // consistent with time order) + the tied-time pair (2, 3) credited
        // 0.5 (time alone can't order it) = 5.5 / 6. The old implementation
        // gave 6/7 ~= 0.857 on this same data (pair (2, 3) contributed
        // total += 2 instead of 1, since neither traversal direction was
        // skipped).
        assert!(
            (c - 5.5 / 6.0).abs() < 1e-12,
            "expected 5.5/6 = 0.91666..., got {c}"
        );
    }

    #[test]
    fn concordance_index_ignores_censored_ties_and_mixed_censoring() {
        // Two subjects censored at the same time contribute nothing (no
        // event to anchor an order on); a censored subject tied with an
        // event at the same time is also not comparable (unknown whether
        // the censored one would have failed before or after).
        let events = vec![
            SurvivalEvent {
                time: 5.0,
                event: false,
            },
            SurvivalEvent {
                time: 5.0,
                event: false,
            },
            SurvivalEvent {
                time: 5.0,
                event: true,
            },
        ];
        let predictions: Vec<_> = [0.5, 0.5, 0.5]
            .into_iter()
            .map(prediction_with_risk)
            .collect();
        // No comparable pairs at all -> falls back to the 0.5 default.
        assert_eq!(concordance_index(&predictions, &events), 0.5);
    }
}
