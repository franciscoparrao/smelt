//! AdaBoost (Adaptive Boosting) classifier via SAMME algorithm.

use crate::learner::{Learner, TrainedModel};
use crate::prediction::Prediction;
use crate::Result;
use crate::task::{ClassificationTask, Task};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// AdaBoost classifier using decision stumps (depth-1 trees).
///
/// Iteratively trains weak learners on weighted data, focusing on
/// previously misclassified samples. Uses the SAMME algorithm for
/// multiclass support.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use ndarray::array;
///
/// let features = array![[0.0], [0.5], [1.0], [1.5], [2.0], [2.5], [3.0], [3.5]];
/// let target = vec![0, 0, 0, 0, 1, 1, 1, 1];
/// let task = ClassificationTask::new("ada", features, target).unwrap();
///
/// let mut ada = AdaBoost::new().with_n_estimators(50);
/// let model = ada.train_classif(&task).unwrap();
/// ```
pub struct AdaBoost {
    n_estimators: usize,
    learning_rate: f64,
}

impl Default for AdaBoost {
    fn default() -> Self {
        Self {
            n_estimators: 50,
            learning_rate: 1.0,
        }
    }
}

impl AdaBoost {
    /// Creates an AdaBoost with 50 estimators and a learning rate of 1.0.
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets the number of boosting rounds (decision stumps to train).
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }
    /// Sets the learning rate scaling each stump's SAMME alpha weight.
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }
}

/// A trained AdaBoost (SAMME) ensemble, ready to predict.
#[derive(Clone, Serialize, Deserialize)]
pub struct TrainedAdaBoost {
    pub(crate) stumps: Vec<TrainedStump>,
    pub(crate) alphas: Vec<f64>,
    pub(crate) n_classes: usize,
    pub(crate) feature_names: Vec<String>,
}

/// A trained decision stump (depth-1 tree) stored as a simple split.
#[derive(Clone, Serialize, Deserialize)]
pub struct TrainedStump {
    feature: usize,
    threshold: f64,
    left_class: usize,
    right_class: usize,
}

impl TrainedStump {
    fn predict_one(&self, row: &[f64]) -> usize {
        if row[self.feature] <= self.threshold {
            self.left_class
        } else {
            self.right_class
        }
    }
}

/// Train a weighted decision stump — find the single best split.
fn train_stump(
    features: &Array2<f64>,
    target: &[usize],
    weights: &[f64],
    n_classes: usize,
) -> (TrainedStump, f64) {
    let n = features.nrows();
    let p = features.ncols();
    let mut best_err = f64::INFINITY;
    let mut best_stump = TrainedStump {
        feature: 0,
        threshold: 0.0,
        left_class: 0,
        right_class: 0,
    };

    // Per-class weighted totals over the whole node, computed once: the
    // sweep below derives right-side counts as total − left, so it never
    // subtracts from a running right-side accumulator.
    let mut total_counts = vec![0.0; n_classes];
    for idx in 0..n {
        total_counts[target[idx]] += weights[idx];
    }
    let total_weight: f64 = total_counts.iter().sum();

    for feat in 0..p {
        let mut sorted: Vec<usize> = (0..n).collect();
        sorted.sort_by(|&a, &b| {
            features[[a, feat]]
                .partial_cmp(&features[[b, feat]])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Sweep left-to-right maintaining running weighted per-class counts,
        // so the majority class and weighted error per candidate threshold
        // are O(n_classes) instead of three full O(n) rescans (audit M-3,
        // same fix as TreeBuilder::best_split_classif). The error of
        // predicting the majority class on each side is that side's total
        // weight minus its majority class's weight.
        let mut left_counts = vec![0.0; n_classes];
        let mut left_total = 0.0;

        for i in 1..n {
            let moved = sorted[i - 1];
            left_counts[target[moved]] += weights[moved];
            left_total += weights[moved];

            if (features[[sorted[i], feat]] - features[[sorted[i - 1], feat]]).abs() < f64::EPSILON
            {
                continue;
            }

            let threshold = (features[[sorted[i - 1], feat]] + features[[sorted[i], feat]]) / 2.0;

            // `>=` so ties pick the highest class index, matching the
            // last-max tie-breaking of the `Iterator::max_by` this replaces.
            let mut left_class = 0;
            let mut left_best = f64::NEG_INFINITY;
            let mut right_class = 0;
            let mut right_best = f64::NEG_INFINITY;
            for c in 0..n_classes {
                if left_counts[c] >= left_best {
                    left_best = left_counts[c];
                    left_class = c;
                }
                let right_c = total_counts[c] - left_counts[c];
                if right_c >= right_best {
                    right_best = right_c;
                    right_class = c;
                }
            }

            let err = (left_total - left_best) + ((total_weight - left_total) - right_best);

            if err < best_err {
                best_err = err;
                best_stump = TrainedStump {
                    feature: feat,
                    threshold,
                    left_class,
                    right_class,
                };
            }
        }
    }

    (best_stump, best_err)
}

impl TrainedModel for TrainedAdaBoost {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        crate::validate::check_n_features(features, self.feature_names.len())?;
        let mut predicted = Vec::with_capacity(features.nrows());
        let mut probabilities = Vec::with_capacity(features.nrows());

        for row in features.rows() {
            let row_slice: Vec<f64> = row.to_vec();
            let mut class_scores = vec![0.0; self.n_classes];
            for (stump, &alpha) in self.stumps.iter().zip(&self.alphas) {
                let pred = stump.predict_one(&row_slice);
                class_scores[pred] += alpha;
            }

            // Softmax for probabilities
            let max_s = class_scores
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let exp_sum: f64 = class_scores.iter().map(|&s| (s - max_s).exp()).sum();
            let probs: Vec<f64> = class_scores
                .iter()
                .map(|&s| (s - max_s).exp() / exp_sum)
                .collect();

            let pred_class = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap()
                .0;
            predicted.push(pred_class);
            probabilities.push(probs);
        }

        Ok(Prediction::Classification {
            predicted,
            truth: None,
            probabilities: Some(probabilities),
        })
    }

    fn to_serializable(&self) -> Option<crate::serialize::SerializableModel> {
        Some(crate::serialize::SerializableModel::AdaBoost(self.clone()))
    }
}

impl AdaBoost {
    /// Trains and returns the concrete [`TrainedAdaBoost`] directly (rather
    /// than through the boxed `Box<dyn TrainedModel>` [`Learner::train_classif`]
    /// returns), so callers -- in this module, its unit tests -- can inspect
    /// internals like the number of stumps actually trained.
    fn fit_classif(&mut self, task: &ClassificationTask) -> Result<TrainedAdaBoost> {
        crate::validate::check_no_nan(task.features())?;
        let features = task.features();
        let target = task.target();
        let n = task.n_samples();
        let n_classes = task.n_classes();

        let mut weights = vec![1.0 / n as f64; n];
        let mut stumps = Vec::with_capacity(self.n_estimators);
        let mut alphas = Vec::with_capacity(self.n_estimators);

        for _ in 0..self.n_estimators {
            let (stump, err) = train_stump(features, target, &weights, n_classes);

            // SAMME's "can't improve" condition is "no better than random
            // guessing among n_classes classes", i.e. `err >= 1 - 1/K`. The
            // previous `1.0 - 1e-10 / n_classes as f64` was off by many
            // orders of magnitude (evaluates to ~1.0 for any realistic K),
            // so this guard almost never actually fired, letting SAMME
            // accept stumps barely better than chance -- in a many-class
            // problem with `learning_rate < 1.0` those could still receive
            // a net-positive alpha (see below) and get amplified into the
            // ensemble's vote.
            if err >= 1.0 - 1.0 / n_classes as f64 {
                break;
            }

            // A perfect stump (err=0) leaves every sample weight unchanged
            // below (nothing was misclassified to reweight), so every
            // subsequent round would retrain the *identical* stump --
            // remember this so the loop can stop instead of wasting the
            // rest of `n_estimators` re-adding it.
            let perfect = err <= 0.0;
            let err = err.max(1e-10); // avoid log(0)

            // SAMME alpha (Zhu et al. 2009, "Multi-class AdaBoost",
            // Algorithm 1: `alpha = ln((1-err)/err) + ln(K-1)`). The
            // learning rate scales the *entire* alpha, not just the
            // error-ratio term -- multiplying only the first term (the
            // previous implementation) left `ln(n_classes - 1)` unscaled,
            // so with a small learning rate the class-count term alone
            // could keep `alpha > 0` for a stump this same round's guard
            // was supposed to reject as no-better-than-chance.
            let alpha =
                self.learning_rate * (((1.0 - err) / err).ln() + (n_classes as f64 - 1.0).ln());

            if alpha <= 0.0 {
                break;
            }

            // Update weights
            for i in 0..n {
                let row: Vec<f64> = features.row(i).to_vec();
                let pred = stump.predict_one(&row);
                if pred != target[i] {
                    weights[i] *= (alpha).exp();
                }
            }
            // Normalize
            let w_sum: f64 = weights.iter().sum();
            for w in &mut weights {
                *w /= w_sum;
            }

            stumps.push(stump);
            alphas.push(alpha);

            if perfect {
                break;
            }
        }

        Ok(TrainedAdaBoost {
            stumps,
            alphas,
            n_classes,
            feature_names: task.feature_names().to_vec(),
        })
    }
}

impl Learner for AdaBoost {
    fn id(&self) -> &str {
        "adaboost"
    }

    fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        Ok(Box::new(self.fit_classif(task)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Regression test: a stump with err=0 (perfectly separates the two
    /// classes) used to leave every sample weight unchanged (nothing
    /// misclassified to reweight), so every remaining round would retrain
    /// the *identical* stump. The trained ensemble should stop with a
    /// single stump once one achieves err=0, instead of wasting the rest
    /// of `n_estimators`.
    #[test]
    fn perfect_stump_stops_training_early() {
        let features = array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]];
        let target = vec![0usize, 0, 0, 0, 1, 1, 1, 1];
        let task = ClassificationTask::new("ada_perfect", features, target).unwrap();

        let mut ada = AdaBoost::new().with_n_estimators(50);
        let model = ada.fit_classif(&task).unwrap();
        assert_eq!(
            model.stumps.len(),
            1,
            "a perfectly-separating stump should stop training immediately"
        );
    }

    /// Regression test for the learning-rate scaling bug: SAMME's alpha is
    /// `ln((1-err)/err) + ln(K-1)`, and `learning_rate` must scale the
    /// *whole* expression (Zhu et al. 2009). The previous implementation
    /// multiplied only the first term, leaving `ln(K-1)` unscaled -- with
    /// `learning_rate=0.0` that bug would still produce `alpha = ln(K-1) >
    /// 0` (nonzero) for any stump better than a coin flip in a 3+ class
    /// problem, when it should produce exactly `alpha = 0.0` and reject
    /// every candidate immediately (`alpha <= 0.0` guard), leaving an
    /// empty ensemble.
    #[test]
    fn learning_rate_scales_the_full_samme_alpha() {
        // 3 classes in blocks -- a single split can separate at most 2 of
        // the 3 blocks perfectly, so `err > 0` and the log-ratio term is
        // exercised (not the degenerate err=0 early-stop path above).
        let features = array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]];
        let target = vec![0usize, 0, 0, 1, 1, 1, 2, 2, 2];
        let task = ClassificationTask::new("ada_lr_zero", features, target).unwrap();

        let mut ada = AdaBoost::new().with_n_estimators(5).with_learning_rate(0.0);
        let model = ada.fit_classif(&task).unwrap();
        assert!(
            model.stumps.is_empty(),
            "learning_rate=0.0 should scale the entire SAMME alpha to exactly 0.0 \
             and reject every stump immediately, got {} stumps",
            model.stumps.len()
        );
    }

    /// Regression test for the M-3 O(n²) fix: `train_stump`'s incremental
    /// sweep (running weighted per-class counts, right side derived as
    /// total − left) must pick the same stump, with the same weighted error,
    /// as the brute-force reference it replaced (recompute both sides' counts
    /// and rescan all samples for the error, at every candidate threshold).
    /// Non-uniform weights and 3 classes exercise the weighted-majority and
    /// error bookkeeping beyond what uniform binary data can.
    #[test]
    fn train_stump_matches_the_brute_force_reference() {
        let n = 60;
        let p = 3;
        let mut feats = Vec::with_capacity(n * p);
        let mut target = Vec::with_capacity(n);
        let mut weights = Vec::with_capacity(n);
        for i in 0..n {
            // Deterministic pseudo-random features on an integer grid, so
            // candidate thresholds are exact midpoints (x.5) and the old
            // code's `<= threshold` rescan partitions exactly like the
            // sweep's sorted prefix.
            for j in 0..p {
                feats.push(((i * 7 + j * 13) % 10) as f64);
            }
            // Class loosely tied to feature 1, with deliberate impurity.
            target.push(if i % 11 == 0 { 2 } else { ((i * 7 + 13) % 10) / 5 });
            weights.push(1.0 + 0.01 * ((i as f64 * 3.7).sin()));
        }
        let features = Array2::from_shape_vec((n, p), feats).unwrap();
        let n_classes = 3;

        let (stump, err) = train_stump(&features, &target, &weights, n_classes);

        // Brute-force reference: the exact pre-M-3 algorithm.
        let mut best_err = f64::INFINITY;
        let mut best = (0usize, 0.0f64, 0usize, 0usize);
        for feat in 0..p {
            let mut sorted: Vec<usize> = (0..n).collect();
            sorted.sort_by(|&a, &b| {
                features[[a, feat]]
                    .partial_cmp(&features[[b, feat]])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for i in 1..n {
                if (features[[sorted[i], feat]] - features[[sorted[i - 1], feat]]).abs()
                    < f64::EPSILON
                {
                    continue;
                }
                let threshold =
                    (features[[sorted[i - 1], feat]] + features[[sorted[i], feat]]) / 2.0;
                let mut left_counts = vec![0.0; n_classes];
                let mut right_counts = vec![0.0; n_classes];
                for &idx in &sorted[..i] {
                    left_counts[target[idx]] += weights[idx];
                }
                for &idx in &sorted[i..] {
                    right_counts[target[idx]] += weights[idx];
                }
                let argmax = |counts: &[f64]| {
                    counts
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap()
                        .0
                };
                let (lc, rc) = (argmax(&left_counts), argmax(&right_counts));
                let mut e = 0.0;
                for idx in 0..n {
                    let pred = if features[[idx, feat]] <= threshold { lc } else { rc };
                    if pred != target[idx] {
                        e += weights[idx];
                    }
                }
                if e < best_err {
                    best_err = e;
                    best = (feat, threshold, lc, rc);
                }
            }
        }

        assert!(
            (err - best_err).abs() < 1e-9,
            "sweep err={err}, brute-force err={best_err}"
        );
        let preds_sweep: Vec<usize> = (0..n)
            .map(|i| stump.predict_one(features.row(i).to_vec().as_slice()))
            .collect();
        let ref_stump = TrainedStump {
            feature: best.0,
            threshold: best.1,
            left_class: best.2,
            right_class: best.3,
        };
        let preds_ref: Vec<usize> = (0..n)
            .map(|i| ref_stump.predict_one(features.row(i).to_vec().as_slice()))
            .collect();
        assert_eq!(
            preds_sweep, preds_ref,
            "sweep stump (feat={}, thr={}) predicts differently from brute-force \
             stump (feat={}, thr={})",
            stump.feature, stump.threshold, best.0, best.1
        );
    }
}
