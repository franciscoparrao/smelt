//! Adaptive Random Forest (ARF): an ensemble of streaming Hoeffding trees
//! with online bagging and per-tree ADWIN concept-drift detection.
//!
//! Gomes, H.M. et al. (2017). "Adaptive random forests for evolving data
//! stream classification." Machine Learning, 106(9-10), 1469-1495.
//!
//! Each incoming sample is fed to every tree a Poisson(λ)-weighted number of
//! times (online bagging, Oza & Russell 2001). Two ADWIN (Bifet & Gavaldà,
//! 2007) detectors per tree monitor its running 0/1 prediction error: a
//! looser "warning" detector (larger delta) starts a background tree
//! training in parallel once triggered; a stricter "drift" detector (smaller
//! delta) swaps the background tree in to replace the (presumably now-stale)
//! foreground tree once triggered.

use crate::learner::hoeffding::HoeffdingTree;
use crate::learner::{Learner, LearnerProperties, TrainedModel};
use crate::prediction::Prediction;
use crate::task::{ClassificationTask, Task};
use crate::Result;
use ndarray::Array2;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// ADWIN (ADaptive WINdowing) concept-drift detector (Bifet & Gavaldà, 2007).
///
/// Maintains a window of recent values in `[0, 1]` (e.g. a 0/1 prediction-
/// error indicator) and, on each [`Adwin::add`], scans candidate cut points
/// for a statistically significant difference between the two sub-window
/// means (a Hoeffding bound on the means, corrected for testing every cut
/// point via a `1/n` union bound); the first qualifying cut drops the older
/// sub-window and reports a detected change, and the scan repeats on the
/// shrunk window until no cut qualifies.
///
/// This exact "scan every cut point" version is deliberately simpler than
/// the original paper's exponential-histogram bucket compression (which
/// needs O(log n) memory by merging same-size buckets): it costs O(window
/// length) per `add` instead of O(log n). That trade-off is bounded by
/// [`Adwin::with_max_window`] rather than solved algorithmically -- a
/// reasonable scope call given this detector runs twice per tree per
/// incoming sample in [`AdaptiveRandomForest`], not as a one-off analysis.
///
/// **Sensitivity floor** (audit issue N7): capping the window also caps how
/// small an error-rate change this can ever detect, once the window is full.
/// At the balanced cut (`n0 = n1 = max_window/2`), `epsilon = sqrt((1/n) *
/// ln(4n/delta))`; with the default `max_window = 200` and
/// [`AdaptiveRandomForest`]'s default warning delta (0.01), that's
/// `epsilon ~= 0.34` -- a change in the running error rate smaller than
/// ~34 percentage points, sustained across the whole window, is invisible
/// to this detector at its default settings, no matter how long it persists.
/// Smaller/gradual drifts need either a larger `max_window` (more memory and
/// per-`add` cost) or a larger `delta` (more false positives) to be caught.
#[derive(Debug, Clone)]
pub struct Adwin {
    delta: f64,
    max_window: usize,
    window: VecDeque<f64>,
}

impl Adwin {
    /// Creates an ADWIN detector with confidence parameter `delta` (smaller
    /// = more evidence required before reporting a change) and a default
    /// max window of 200.
    pub fn new(delta: f64) -> Self {
        Self {
            delta,
            max_window: 200,
            window: VecDeque::new(),
        }
    }

    /// Sets the maximum window length (oldest values are dropped once
    /// exceeded, independent of drift detection).
    pub fn with_max_window(mut self, n: usize) -> Self {
        self.max_window = n.max(10);
        self
    }

    /// Adds a new value (expected in `[0, 1]`) and reports whether a change
    /// was detected (and the window shrunk) as a result.
    pub fn add(&mut self, value: f64) -> bool {
        self.window.push_back(value);
        if self.window.len() > self.max_window {
            self.window.pop_front();
        }

        let mut shrunk = false;
        loop {
            let n = self.window.len();
            if n < 10 {
                break;
            }

            let mut prefix = Vec::with_capacity(n + 1);
            prefix.push(0.0);
            for &v in &self.window {
                prefix.push(prefix.last().unwrap() + v);
            }

            // Minimum sub-window size 5 on each side guards against noisy
            // near-edge splits triggering on too little evidence.
            let mut cut_found = None;
            for n0 in 5..=(n - 5) {
                let n1 = n - n0;
                let mean0 = prefix[n0] / n0 as f64;
                let mean1 = (prefix[n] - prefix[n0]) / n1 as f64;
                let m = 1.0 / (1.0 / n0 as f64 + 1.0 / n1 as f64);
                // delta/n: a union-bound correction for testing ~n cut
                // points exactly, not the paper's delta/ln(n) (calibrated
                // for its O(log n) bucket representation, which tests far
                // fewer candidate points).
                let delta_prime = self.delta / n as f64;
                let epsilon = ((1.0 / (2.0 * m)) * (4.0 / delta_prime).ln()).sqrt();
                if (mean0 - mean1).abs() > epsilon {
                    cut_found = Some(n0);
                    break;
                }
            }

            match cut_found {
                Some(cut) => {
                    for _ in 0..cut {
                        self.window.pop_front();
                    }
                    shrunk = true;
                }
                None => break,
            }
        }
        shrunk
    }

    /// The current window's mean.
    pub fn mean(&self) -> f64 {
        if self.window.is_empty() {
            0.0
        } else {
            self.window.iter().sum::<f64>() / self.window.len() as f64
        }
    }
}

/// Samples from Poisson(lambda) via Knuth's algorithm. Fine for the small
/// lambda (typically 1-10) used in online bagging; avoids pulling in
/// `rand_distr` for a single distribution, consistent with this crate's
/// preference for hand-rolling small numeric routines (see `src/sparse.rs`).
fn sample_poisson(rng: &mut impl Rng, lambda: f64) -> u32 {
    let l = (-lambda).exp();
    let mut k = 0u32;
    let mut p = 1.0;
    loop {
        k += 1;
        p *= rng.random::<f64>();
        if p <= l {
            break;
        }
    }
    k - 1
}

/// Builds a fresh `HoeffdingTree` with the given hyperparameters, restricted
/// to `feature_subset` (audit issue N6: each ensemble member gets its own
/// random feature subspace, like `RandomForest`/`ExtraTrees` -- otherwise
/// every tree in the "forest" would consider every feature at every split,
/// and diversity would come only from online bagging's resampling, not from
/// the feature-subspace diversity real random forests rely on too). A free
/// function (not a `&self` method) so it can be called from inside a loop
/// that already holds a mutable borrow of `AdaptiveRandomForest::trees`.
fn build_tree(
    split_confidence: f64,
    grace_period: usize,
    max_depth: Option<usize>,
    feature_subset: Vec<usize>,
) -> HoeffdingTree {
    let mut t = HoeffdingTree::new()
        .with_delta(split_confidence)
        .with_grace_period(grace_period)
        .with_feature_subset(feature_subset);
    if let Some(d) = max_depth {
        t = t.with_max_depth(d);
    }
    t
}

/// Draws `k` distinct feature indices out of `0..n_features` uniformly at
/// random (Fisher-Yates via `SliceRandom::shuffle`, then truncate).
fn random_feature_subset(rng: &mut StdRng, n_features: usize, k: usize) -> Vec<usize> {
    use rand::seq::SliceRandom;
    let mut all: Vec<usize> = (0..n_features).collect();
    all.shuffle(rng);
    all.truncate(k.min(n_features).max(1));
    all
}

/// Majority vote across a set of (possibly still-streaming) trees. Skips
/// trees that haven't seen a sample yet (`predict_one` returns `None`).
fn majority_vote<'a>(
    trees: impl Iterator<Item = &'a HoeffdingTree>,
    features: &[f64],
    n_classes: usize,
) -> Option<(usize, Vec<f64>)> {
    let mut votes = vec![0usize; n_classes.max(1)];
    let mut any_trained = false;
    for tree in trees {
        if let Some((pred, _)) = tree.predict_one(features) {
            any_trained = true;
            if pred < votes.len() {
                votes[pred] += 1;
            }
        }
    }
    if !any_trained {
        return None;
    }
    let total: usize = votes.iter().sum();
    let probs: Vec<f64> = if total > 0 {
        votes.iter().map(|&v| v as f64 / total as f64).collect()
    } else {
        vec![1.0 / votes.len() as f64; votes.len()]
    };
    let pred = votes
        .iter()
        .enumerate()
        .max_by_key(|&(_, &c)| c)
        .map(|(i, _)| i)
        .unwrap_or(0);
    Some((pred, probs))
}

struct BackgroundTree {
    tree: HoeffdingTree,
}

struct Member {
    tree: HoeffdingTree,
    warning: Adwin,
    drift: Adwin,
    background: Option<BackgroundTree>,
    /// This slot's random feature subspace (audit issue N6), fixed once at
    /// creation and reused for every background/replacement tree built for
    /// this slot -- a drift-triggered replacement restarts learning, not
    /// the slot's place in the ensemble's feature-subspace diversity.
    feature_subset: Vec<usize>,
}

/// Adaptive Random Forest: an ensemble of streaming Hoeffding trees with
/// online bagging and per-tree ADWIN concept-drift detection.
///
/// Unlike every other learner in this crate, which trains once on a static
/// batch and predicts forever after, `AdaptiveRandomForest` is meant for
/// data whose feature-target relationship can drift over time (continuous
/// sensor monitoring, evolving spatial time series): call [`Self::partial_fit`]
/// one sample at a time, or use [`Learner::train_classif`] to replay a
/// static `ClassificationTask` as a stream in row order (matching
/// [`HoeffdingTree::train_classif`]'s own convention).
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
///
/// let mut arf = AdaptiveRandomForest::new()
///     .with_n_trees(5)
///     .with_seed(7);
///
/// for i in 0..500 {
///     let x = (i % 10) as f64 / 10.0;
///     let y = if x > 0.5 { 1 } else { 0 };
///     arf.partial_fit(&[x], y, 2);
/// }
///
/// let (pred, _probs) = arf.predict_one(&[0.9]).unwrap();
/// assert_eq!(pred, 1);
/// ```
pub struct AdaptiveRandomForest {
    n_trees: usize,
    lambda: f64,
    delta_warning: f64,
    delta_drift: f64,
    split_confidence: f64,
    grace_period: usize,
    max_depth: Option<usize>,
    seed: u64,
    trees: Vec<Member>,
    n_classes: usize,
    rng: StdRng,
    total_drifts: usize,
}

impl Default for AdaptiveRandomForest {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveRandomForest {
    /// Creates an `AdaptiveRandomForest` with 10 trees, online-bagging
    /// lambda 6.0 (the ARF paper's default), warning delta 0.01, drift
    /// delta 0.001, and each member `HoeffdingTree` at its own defaults.
    pub fn new() -> Self {
        Self {
            n_trees: 10,
            lambda: 6.0,
            delta_warning: 0.01,
            delta_drift: 0.001,
            split_confidence: 1e-7,
            grace_period: 200,
            max_depth: None,
            seed: 42,
            trees: Vec::new(),
            n_classes: 0,
            rng: StdRng::seed_from_u64(42),
            total_drifts: 0,
        }
    }

    /// Sets the number of trees in the ensemble.
    pub fn with_n_trees(mut self, n: usize) -> Self {
        self.n_trees = n.max(1);
        self
    }
    /// Sets the online-bagging Poisson parameter (higher = each tree sees
    /// each sample more times on average).
    pub fn with_lambda(mut self, l: f64) -> Self {
        self.lambda = l;
        self
    }
    /// Sets the ADWIN delta for the warning detector (starts a background
    /// tree once triggered).
    pub fn with_delta_warning(mut self, d: f64) -> Self {
        self.delta_warning = d;
        self
    }
    /// Sets the ADWIN delta for the drift detector (replaces the tree once
    /// triggered) -- should be stricter (smaller) than the warning delta.
    pub fn with_delta_drift(mut self, d: f64) -> Self {
        self.delta_drift = d;
        self
    }
    /// Sets each member `HoeffdingTree`'s own split-confidence delta (not
    /// to be confused with the ADWIN deltas above).
    pub fn with_split_confidence(mut self, d: f64) -> Self {
        self.split_confidence = d;
        self
    }
    /// Sets each member `HoeffdingTree`'s grace period.
    pub fn with_grace_period(mut self, g: usize) -> Self {
        self.grace_period = g;
        self
    }
    /// Sets each member `HoeffdingTree`'s maximum depth.
    pub fn with_max_depth(mut self, d: usize) -> Self {
        self.max_depth = Some(d);
        self
    }
    /// Sets the RNG seed used for online-bagging Poisson draws.
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self.rng = StdRng::seed_from_u64(s);
        self
    }

    /// Number of drift-triggered tree replacements across the whole
    /// ensemble so far.
    pub fn n_drifts(&self) -> usize {
        self.total_drifts
    }

    fn new_member(&self, feature_subset: Vec<usize>) -> Member {
        Member {
            tree: build_tree(
                self.split_confidence,
                self.grace_period,
                self.max_depth,
                feature_subset.clone(),
            ),
            warning: Adwin::new(self.delta_warning),
            drift: Adwin::new(self.delta_drift),
            background: None,
            feature_subset,
        }
    }

    /// Online update: train on a single sample, adapting to concept drift.
    pub fn partial_fit(&mut self, features: &[f64], label: usize, n_classes: usize) {
        self.n_classes = self.n_classes.max(n_classes);
        if self.trees.is_empty() {
            // Random feature subspace per tree (audit issue N6), sized like
            // RandomForest/ExtraTrees' classification default: sqrt(p).
            // n_features is only known now, from this first sample -- like
            // HoeffdingTree itself, ARF has no batch-upfront shape.
            let n_features = features.len();
            let subset_size = (n_features as f64).sqrt().ceil() as usize;
            self.trees = (0..self.n_trees)
                .map(|_| {
                    let subset = random_feature_subset(&mut self.rng, n_features, subset_size);
                    self.new_member(subset)
                })
                .collect();
        }
        // Copied out before the loop: `self.trees.iter_mut()` below holds a
        // borrow of that one field, but building a replacement tree needs
        // `&self` as a whole (a `self.new_tree()` call inside the loop
        // would conflict) -- these locals let the loop body call the free
        // `build_tree` function instead.
        let n_classes_local = self.n_classes;
        let lambda = self.lambda;
        let split_confidence = self.split_confidence;
        let grace_period = self.grace_period;
        let max_depth = self.max_depth;
        let delta_warning = self.delta_warning;
        let delta_drift = self.delta_drift;
        let mut drifts_this_call = 0usize;

        for member in self.trees.iter_mut() {
            // Error against the tree's CURRENT state, before this sample
            // updates it -- a drift detector must monitor genuine
            // prediction error, not error against a sample the tree just
            // memorized.
            let error = match member.tree.predict_one(features) {
                Some((pred, _)) => {
                    if pred == label {
                        0.0
                    } else {
                        1.0
                    }
                }
                None => 0.0, // untrained tree: no evidence of error yet
            };

            let warned = member.warning.add(error);
            let drifted = member.drift.add(error);

            // Same Poisson draw shared by the foreground and background
            // tree: the background tree shadows the exact online-bagged
            // stream the foreground would see, not an independently
            // resampled one.
            let k = sample_poisson(&mut self.rng, lambda);
            for _ in 0..k {
                member.tree.partial_fit(features, label, n_classes_local);
            }
            if let Some(bg) = member.background.as_mut() {
                for _ in 0..k {
                    bg.tree.partial_fit(features, label, n_classes_local);
                }
            }

            if warned && member.background.is_none() {
                member.background = Some(BackgroundTree {
                    tree: build_tree(
                        split_confidence,
                        grace_period,
                        max_depth,
                        member.feature_subset.clone(),
                    ),
                });
            }

            if drifted {
                member.tree = match member.background.take() {
                    Some(bg) => bg.tree,
                    None => build_tree(
                        split_confidence,
                        grace_period,
                        max_depth,
                        member.feature_subset.clone(),
                    ),
                };
                member.warning = Adwin::new(delta_warning);
                member.drift = Adwin::new(delta_drift);
                drifts_this_call += 1;
            }
        }

        self.total_drifts += drifts_this_call;
    }

    /// Predicts from the ensemble's current (possibly still-streaming)
    /// state via majority vote over the foreground trees. Returns `None`
    /// before any tree has seen a sample.
    pub fn predict_one(&self, features: &[f64]) -> Option<(usize, Vec<f64>)> {
        majority_vote(self.trees.iter().map(|m| &m.tree), features, self.n_classes)
    }
}

impl Learner for AdaptiveRandomForest {
    fn id(&self) -> &str {
        "adaptive_random_forest"
    }

    fn properties(&self) -> LearnerProperties {
        LearnerProperties::classifier()
            .with_proba()
            .with_serializable()
    }

    fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        crate::validate::check_no_weights(task.weights(), "AdaptiveRandomForest")?;
        crate::validate::check_no_nan(task.features())?;
        let features = task.features();
        let target = task.target();
        let n_classes = task.n_classes();

        self.trees.clear();
        self.n_classes = 0;

        for i in 0..task.n_samples() {
            let row: Vec<f64> = features.row(i).to_vec();
            self.partial_fit(&row, target[i], n_classes);
        }

        Ok(Box::new(TrainedAdaptiveRandomForest {
            trees: std::mem::take(&mut self.trees).into_iter().map(|m| m.tree).collect(),
            n_features: task.n_features(),
            n_classes: self.n_classes,
        }))
    }
}

/// A trained Adaptive Random Forest ensemble of streaming Hoeffding trees.
#[derive(Clone, Serialize, Deserialize)]
pub struct TrainedAdaptiveRandomForest {
    trees: Vec<HoeffdingTree>,
    n_features: usize,
    n_classes: usize,
}

impl TrainedModel for TrainedAdaptiveRandomForest {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        crate::validate::check_n_features(features, self.n_features)?;

        let mut predicted = Vec::with_capacity(features.nrows());
        let mut probabilities = Vec::with_capacity(features.nrows());
        for row in features.rows() {
            let row_vec: Vec<f64> = row.to_vec();
            let (pred, probs) = majority_vote(self.trees.iter(), &row_vec, self.n_classes)
                .unwrap_or((0, vec![1.0 / self.n_classes.max(1) as f64; self.n_classes.max(1)]));
            predicted.push(pred);
            probabilities.push(probs);
        }

        Ok(Prediction::Classification {
            predicted,
            truth: None,
            probabilities: Some(probabilities),
        })
    }

    fn to_serializable(&self) -> Option<crate::serialize::SerializableModel> {
        Some(crate::serialize::SerializableModel::AdaptiveRandomForest(
            self.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn adwin_detects_injected_mean_shift() {
        let mut rng = StdRng::seed_from_u64(1);
        let mut adwin = Adwin::new(0.05);
        let mut detected_in_second_regime = false;

        for _ in 0..200 {
            let v = if rng.random::<f64>() < 0.05 { 1.0 } else { 0.0 };
            adwin.add(v);
        }
        for _ in 0..200 {
            let v = if rng.random::<f64>() < 0.6 { 1.0 } else { 0.0 };
            if adwin.add(v) {
                detected_in_second_regime = true;
            }
        }

        assert!(
            detected_in_second_regime,
            "ADWIN should detect the injected mean shift from 0.05 to 0.6"
        );
    }

    /// Regression test for N6 (docs/auditoria_motor_2026-07-05.md, Fase F):
    /// each tree in the ensemble must get its own random feature subspace
    /// (sized like RandomForest/ExtraTrees' sqrt(p) classification default),
    /// not consider every feature -- and different trees must actually end
    /// up with different subsets (the diversity this is meant to provide).
    #[test]
    fn arf_gives_each_tree_a_distinct_random_feature_subspace() {
        let mut rng = StdRng::seed_from_u64(3);
        let mut arf = AdaptiveRandomForest::new().with_n_trees(8).with_seed(5);
        let n_features = 10;
        for _ in 0..50 {
            let x: Vec<f64> = (0..n_features).map(|_| rng.random::<f64>()).collect();
            let y = rng.random_range(0..2);
            arf.partial_fit(&x, y, 2);
        }

        let expected_size = (n_features as f64).sqrt().ceil() as usize; // 4
        for member in &arf.trees {
            assert_eq!(member.feature_subset.len(), expected_size);
            assert!(member.feature_subset.iter().all(|&f| f < n_features));
            let mut sorted = member.feature_subset.clone();
            sorted.sort();
            sorted.dedup();
            assert_eq!(sorted.len(), expected_size, "feature subset must not contain duplicates");
        }

        let distinct_subsets: std::collections::HashSet<Vec<usize>> = arf
            .trees
            .iter()
            .map(|m| {
                let mut s = m.feature_subset.clone();
                s.sort();
                s
            })
            .collect();
        assert!(
            distinct_subsets.len() > 1,
            "with 8 trees and C(10,4)=210 possible subsets, at least two trees should differ, \
             got {} distinct subset(s)",
            distinct_subsets.len()
        );
    }

    /// Regression/golden test for N7's documented sensitivity floor: with
    /// the default `max_window=200` and a delta of 0.01 (this crate's
    /// default warning delta), the doc comment on `Adwin` derives a floor of
    /// ~0.34 at the balanced cut. A clearly-smaller sustained jump must not
    /// be detected; a clearly-larger one must be, once the new regime has
    /// had the full window length to take over.
    #[test]
    fn adwin_sensitivity_floor_matches_doc_derivation() {
        let detects_shift = |first: f64, second: f64| -> bool {
            let mut adwin = Adwin::new(0.01);
            for _ in 0..200 {
                adwin.add(first);
            }
            let mut detected = false;
            for _ in 0..200 {
                if adwin.add(second) {
                    detected = true;
                }
            }
            detected
        };

        assert!(
            !detects_shift(0.0, 0.15),
            "a 0.15 jump is well below the documented ~0.34 floor and should not be detected"
        );
        assert!(
            detects_shift(0.0, 0.6),
            "a 0.6 jump is well above the documented ~0.34 floor and should be detected"
        );
    }

    #[test]
    fn adwin_low_false_positive_rate_on_stationary_stream() {
        let mut total_drifts = 0usize;
        for seed in 0..5u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut adwin = Adwin::new(0.05);
            for _ in 0..500 {
                let v = if rng.random::<f64>() < 0.1 { 1.0 } else { 0.0 };
                if adwin.add(v) {
                    total_drifts += 1;
                }
            }
        }
        assert!(
            total_drifts <= 5,
            "expected few false-positive drifts on a stationary stream, got {total_drifts}"
        );
    }

    #[test]
    fn sample_poisson_mean_matches_lambda() {
        let mut rng = StdRng::seed_from_u64(3);
        let lambda = 6.0;
        let n = 20_000;
        let total: u64 = (0..n).map(|_| sample_poisson(&mut rng, lambda) as u64).sum();
        let mean = total as f64 / n as f64;
        assert!(
            (mean - lambda).abs() < 0.15,
            "empirical mean {mean} should be close to lambda {lambda}"
        );
    }

    #[test]
    fn registered_id_matches() {
        assert_eq!(AdaptiveRandomForest::new().id(), "adaptive_random_forest");
    }

    #[test]
    fn adapts_to_concept_drift_faster_than_plain_hoeffding_tree() {
        let mut rng = StdRng::seed_from_u64(42);
        let grace = 50;

        let mut arf = AdaptiveRandomForest::new()
            .with_n_trees(10)
            .with_grace_period(grace)
            .with_seed(11);
        let mut plain = HoeffdingTree::new().with_grace_period(grace);

        // Regime 1: label depends on x0; x1 is noise. Long enough to train
        // well past the grace period.
        for _ in 0..2000 {
            let x0 = rng.random::<f64>();
            let x1 = rng.random::<f64>();
            let y = if x0 > 0.5 { 1 } else { 0 };
            arf.partial_fit(&[x0, x1], y, 2);
            plain.partial_fit(&[x0, x1], y, 2);
        }

        // Regime 2: label now depends on x1; x0 is noise. Kept short --
        // just enough for ARF's warning -> background-train -> drift
        // -> replace cycle to complete.
        for _ in 0..(grace * 6) {
            let x0 = rng.random::<f64>();
            let x1 = rng.random::<f64>();
            let y = if x1 > 0.5 { 1 } else { 0 };
            arf.partial_fit(&[x0, x1], y, 2);
            plain.partial_fit(&[x0, x1], y, 2);
        }

        assert!(arf.n_drifts() >= 1, "expected at least one drift-triggered replacement");

        // Evaluate both on a larger, disjoint regime-2 holdout set.
        let mut arf_correct = 0usize;
        let mut plain_correct = 0usize;
        let n_eval = 500;
        for _ in 0..n_eval {
            let x0 = rng.random::<f64>();
            let x1 = rng.random::<f64>();
            let y = if x1 > 0.5 { 1 } else { 0 };

            if let Some((pred, _)) = arf.predict_one(&[x0, x1])
                && pred == y {
                    arf_correct += 1;
                }
            if let Some((pred, _)) = plain.predict_one(&[x0, x1])
                && pred == y {
                    plain_correct += 1;
                }
        }

        let arf_acc = arf_correct as f64 / n_eval as f64;
        let plain_acc = plain_correct as f64 / n_eval as f64;

        assert!(arf_acc > 0.85, "ARF regime-2 accuracy should be high after adapting, got {arf_acc}");
        assert!(
            plain_acc < 0.7,
            "plain HoeffdingTree should still be hindered by its stale root split, got {plain_acc}"
        );
    }

    #[test]
    fn no_worse_than_plain_hoeffding_tree_on_a_stationary_stream() {
        let mut rng = StdRng::seed_from_u64(5);
        let mut arf = AdaptiveRandomForest::new().with_n_trees(10).with_seed(9);
        let mut plain = HoeffdingTree::new();

        for _ in 0..3000 {
            let x0 = rng.random::<f64>();
            let x1 = rng.random::<f64>();
            let y = if x0 > 0.5 { 1 } else { 0 };
            arf.partial_fit(&[x0, x1], y, 2);
            plain.partial_fit(&[x0, x1], y, 2);
        }

        let mut arf_correct = 0usize;
        let mut plain_correct = 0usize;
        let n_eval = 500;
        for _ in 0..n_eval {
            let x0 = rng.random::<f64>();
            let x1 = rng.random::<f64>();
            let y = if x0 > 0.5 { 1 } else { 0 };
            if let Some((pred, _)) = arf.predict_one(&[x0, x1])
                && pred == y {
                    arf_correct += 1;
                }
            if let Some((pred, _)) = plain.predict_one(&[x0, x1])
                && pred == y {
                    plain_correct += 1;
                }
        }

        let arf_acc = arf_correct as f64 / n_eval as f64;
        let plain_acc = plain_correct as f64 / n_eval as f64;
        assert!(
            arf_acc >= plain_acc - 0.1,
            "ARF ({arf_acc}) shouldn't be meaningfully worse than a plain tree ({plain_acc}) with no drift"
        );
    }

    #[test]
    fn train_classif_batch_api_works() {
        let mut feats = Vec::new();
        let mut target = Vec::new();
        let mut rng = StdRng::seed_from_u64(2);
        for _ in 0..1000 {
            let x0 = rng.random::<f64>();
            feats.push(x0);
            target.push(if x0 > 0.5 { 1usize } else { 0 });
        }
        let features = Array2::from_shape_vec((1000, 1), feats).unwrap();
        let task = ClassificationTask::new("arf", features.clone(), target).unwrap();

        let mut arf = AdaptiveRandomForest::new().with_n_trees(5);
        let model = arf.train_classif(&task).unwrap();
        let pred = model.predict(&features).unwrap();
        let Prediction::Classification { predicted, .. } = pred else {
            panic!("expected classification");
        };
        let correct = predicted
            .iter()
            .zip(task.target())
            .filter(|(p, t)| *p == *t)
            .count();
        let acc = correct as f64 / predicted.len() as f64;
        assert!(acc > 0.85, "batch-trained ARF should fit this simple threshold rule well, got {acc}");
    }
}
