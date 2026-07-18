//! Decision-threshold tuning for binary classifiers: wraps a probabilistic
//! classifier and replaces the implicit 0.5 decision boundary with either a
//! user-supplied threshold or one tuned on held-out data to maximize a chosen
//! measure. The smelt analogue of scikit-learn's `TunedThresholdClassifierCV`
//! / mlr3's `PipeOpTuneThreshold`.
//!
//! The default "predict class 1 iff `P(y=1|x) >= 0.5`" rule is optimal only
//! for accuracy on balanced data with well-calibrated probabilities. Under
//! class imbalance, or when the operating metric is F1 / balanced accuracy /
//! an asymmetric cost, a different threshold does markedly better — a rare-
//! positive detector often needs a threshold well below 0.5 to reach usable
//! recall.
//!
//! # Relationship to `CostSensitiveClassifier`
//!
//! For a binary problem with a false-positive cost `c_fp` and false-negative
//! cost `c_fn`, the [`crate::learner::CostSensitiveClassifier`] Bayes rule
//! reduces to a fixed threshold: predict class 1 iff
//! `P(y=1|x) >= c_fp / (c_fp + c_fn)`. That is the same *shape* of decision
//! this wrapper makes — the difference is how the threshold is obtained:
//! `CostSensitiveClassifier` computes it in closed form from a cost matrix you
//! supply, whereas this wrapper *tunes* it empirically on a holdout to
//! optimize a metric you may not be able to express as a cost. Use the closed
//! form when you know your costs; tune when you only know the metric. They are
//! deliberately kept separate rather than unified.
//!
//! # Strategy
//!
//! In tuned mode, the base learner is trained on a fit-set, its positive-class
//! probabilities on a held-out calibration-set are swept over candidate
//! thresholds (each observed probability value), the best threshold under the
//! chosen [`Measure`] is kept, and the base is **refit on the whole training
//! set** — the same holdout + refit-on-all convention as
//! [`crate::learner::CalibratedClassifier`]. In fixed mode no split is needed:
//! the base is trained on everything and the given threshold is used as-is.
//!
//! # Registry
//!
//! Like the other factory-based wrappers, this needs a base-learner factory
//! with no sensible default, so it is **not** constructible via
//! [`crate::learner::registry::learner_from_id`].

use crate::Result;
use crate::SmeltError;
use crate::learner::{Learner, LearnerProperties, TrainedModel};
use crate::measure::{F1Score, Measure};
use crate::prediction::Prediction;
use crate::resample::{Holdout, Resample};
use crate::task::{ClassificationTask, Task};
use ndarray::{Array2, Axis};

/// How the decision threshold is obtained.
enum ThresholdStrategy {
    /// A fixed threshold supplied by the caller (no tuning, no split).
    Fixed(f64),
    /// A threshold tuned on a holdout to optimize this measure.
    Tuned(Box<dyn Measure>),
}

/// Binary decision-threshold wrapper around any probabilistic classifier.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use smelt_ml::learner::ThresholdedClassifier;
/// use ndarray::array;
///
/// let features = array![
///     [0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [0.0, 0.2],
///     [1.0, 1.0], [1.1, 0.9], [0.9, 1.1], [1.0, 0.9]
/// ];
/// let target = vec![0, 0, 0, 0, 1, 1, 1, 1];
/// let task = ClassificationTask::new("thr_demo", features, target).unwrap();
///
/// // Tuned mode (default measure: F1).
/// let mut tc = ThresholdedClassifier::new(|| Box::new(LogisticRegression::new())).with_seed(0);
/// let model = tc.train_classif(&task).unwrap();
///
/// // Fixed mode.
/// let mut fixed = ThresholdedClassifier::new(|| Box::new(LogisticRegression::new()))
///     .with_threshold(0.7);
/// let _ = fixed.train_classif(&task).unwrap();
/// ```
pub struct ThresholdedClassifier {
    factory: Box<dyn Fn() -> Box<dyn Learner> + Send + Sync>,
    strategy: ThresholdStrategy,
    calib_fraction: f64,
    seed: u64,
}

impl ThresholdedClassifier {
    /// Create a threshold wrapper from a base-learner factory. Defaults to
    /// *tuned* mode optimizing [`F1Score`], `calib_fraction = 0.3`,
    /// `seed = 42`. Switch to a fixed threshold with [`Self::with_threshold`]
    /// or a different tuning measure with [`Self::with_metric`].
    pub fn new(factory: impl Fn() -> Box<dyn Learner> + Send + Sync + 'static) -> Self {
        Self {
            factory: Box::new(factory),
            strategy: ThresholdStrategy::Tuned(Box::new(F1Score)),
            calib_fraction: 0.3,
            seed: 42,
        }
    }

    /// Use a fixed decision threshold `t` (predict class 1 iff
    /// `P(y=1|x) >= t`), skipping tuning and the internal holdout entirely.
    pub fn with_threshold(mut self, t: f64) -> Self {
        self.strategy = ThresholdStrategy::Fixed(t);
        self
    }

    /// Tune the threshold to maximize/minimize (per [`Measure::maximize`]) the
    /// given measure instead of the default F1.
    pub fn with_metric(mut self, measure: Box<dyn Measure>) -> Self {
        self.strategy = ThresholdStrategy::Tuned(measure);
        self
    }

    /// Fraction of the training data held out to tune the threshold (tuned
    /// mode only; ignored for a fixed threshold). Must be in `(0, 1)`;
    /// default `0.3`.
    pub fn with_calib_fraction(mut self, fraction: f64) -> Self {
        self.calib_fraction = fraction;
        self
    }

    /// RNG seed for the internal tuning holdout split (tuned mode only).
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    fn subset_task(task: &ClassificationTask, idx: &[usize]) -> Result<ClassificationTask> {
        let features = task.features().select(Axis(0), idx);
        let target: Vec<usize> = idx.iter().map(|&i| task.target()[i]).collect();
        Ok(ClassificationTask::new(task.id(), features, target)?
            .with_feature_names(task.feature_names().to_vec())?
            .with_feature_types(task.feature_types().to_vec())?
            .with_class_names(task.class_names().to_vec()))
    }

    /// Positive-class probabilities from a classification prediction, or a
    /// clear error if the base learner emits no probabilities.
    fn positive_probs(pred: &Prediction) -> Result<Vec<f64>> {
        match pred {
            Prediction::Classification {
                probabilities: Some(p),
                ..
            } => Ok(p.iter().map(|row| row.get(1).copied().unwrap_or(0.0)).collect()),
            Prediction::Classification {
                probabilities: None,
                ..
            } => Err(SmeltError::IncompatiblePrediction(
                "ThresholdedClassifier requires a base learner that produces probabilities \
                 (e.g. logistic_regression, random_forest, decision_tree, gaussian_nb)"
                    .into(),
            )),
            _ => Err(SmeltError::IncompatiblePrediction(
                "ThresholdedClassifier requires classification predictions".into(),
            )),
        }
    }

    /// Train and return the concrete [`TrainedThresholdedClassifier`] (which
    /// carries [`TrainedThresholdedClassifier::best_threshold`] beyond the
    /// [`TrainedModel`] trait — same "concrete carries more than the trait"
    /// shape as [`crate::learner::TrainedAutoTuner::best_params`]).
    pub fn fit_classif(&self, task: &ClassificationTask) -> Result<TrainedThresholdedClassifier> {
        crate::validate::check_no_weights(task.weights(), "ThresholdedClassifier")?;
        if task.n_classes() != 2 {
            return Err(SmeltError::InvalidParameter(format!(
                "ThresholdedClassifier is binary-only, but the task has {} classes",
                task.n_classes()
            )));
        }

        match &self.strategy {
            ThresholdStrategy::Fixed(t) => {
                let mut base = (self.factory)();
                let model = base.train_classif(task)?;
                // Confirm the base is probabilistic up front (a hard-label
                // base makes the threshold meaningless).
                let _ = Self::positive_probs(&model.predict(task.features())?)?;
                Ok(TrainedThresholdedClassifier {
                    base: model,
                    threshold: *t,
                })
            }
            ThresholdStrategy::Tuned(measure) => {
                if !(self.calib_fraction > 0.0 && self.calib_fraction < 1.0) {
                    return Err(SmeltError::InvalidParameter(format!(
                        "ThresholdedClassifier calib_fraction must be in (0, 1), got {}",
                        self.calib_fraction
                    )));
                }
                let n = task.n_samples();
                let splits = Holdout::new(1.0 - self.calib_fraction)
                    .with_seed(self.seed)
                    .splits(n)?;
                let (fit_idx, calib_idx) = &splits[0];
                if fit_idx.is_empty() || calib_idx.is_empty() {
                    return Err(SmeltError::InvalidParameter(format!(
                        "ThresholdedClassifier: the fit/tuning split left an empty side \
                         ({} fit, {} calib rows from {n} samples)",
                        fit_idx.len(),
                        calib_idx.len()
                    )));
                }

                let fit_task = Self::subset_task(task, fit_idx)?;
                let mut fit_base = (self.factory)();
                let fit_model = fit_base.train_classif(&fit_task)?;
                let calib_features = task.features().select(Axis(0), calib_idx);
                let calib_pred = fit_model.predict(&calib_features)?;
                let calib_p1 = Self::positive_probs(&calib_pred)?;
                let calib_labels: Vec<usize> =
                    calib_idx.iter().map(|&i| task.target()[i]).collect();

                let best = Self::tune_threshold(&calib_p1, &calib_labels, measure.as_ref())?;

                // Refit the base on ALL the training data.
                let mut final_base = (self.factory)();
                let final_model = final_base.train_classif(task)?;
                Ok(TrainedThresholdedClassifier {
                    base: final_model,
                    threshold: best,
                })
            }
        }
    }

    /// Sweep candidate thresholds (each observed positive-class probability,
    /// plus the two extremes that force all-1 / all-0) and return the one that
    /// optimizes `measure` on the calibration set.
    fn tune_threshold(p1: &[f64], labels: &[usize], measure: &dyn Measure) -> Result<f64> {
        let mut candidates: Vec<f64> = p1.to_vec();
        // 0.0 => predict all 1 (every prob >= 0). max+1 => predict all 0.
        candidates.push(0.0);
        let max_p = p1.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        candidates.push(if max_p.is_finite() { max_p + 1.0 } else { 1.0 });
        candidates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        candidates.dedup_by(|a, b| (*a - *b).abs() <= f64::EPSILON);

        let maximize = measure.maximize();
        let probs: Vec<Vec<f64>> = p1.iter().map(|&p| vec![1.0 - p, p]).collect();
        let mut best_t = 0.5_f64;
        let mut best_score = if maximize { f64::NEG_INFINITY } else { f64::INFINITY };
        for &t in &candidates {
            let predicted: Vec<usize> = p1.iter().map(|&p| usize::from(p >= t)).collect();
            let pred = Prediction::Classification {
                predicted,
                truth: Some(labels.to_vec()),
                probabilities: Some(probs.clone()),
            };
            let score = measure.score(&pred)?;
            let better = if maximize {
                score > best_score
            } else {
                score < best_score
            };
            if better {
                best_score = score;
                best_t = t;
            }
        }
        Ok(best_t)
    }
}

impl Learner for ThresholdedClassifier {
    fn id(&self) -> &str {
        "thresholded"
    }

    fn properties(&self) -> LearnerProperties {
        LearnerProperties::classifier().with_proba()
    }

    fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        Ok(Box::new(self.fit_classif(task)?))
    }
}

/// A trained [`ThresholdedClassifier`]: the base model plus the applied
/// decision threshold. `predict` returns the base model's probabilities
/// unchanged but labels each sample class 1 iff `P(y=1|x) >= threshold`.
pub struct TrainedThresholdedClassifier {
    base: Box<dyn TrainedModel>,
    threshold: f64,
}

impl TrainedThresholdedClassifier {
    /// The decision threshold this model applies (fixed, or the tuned winner).
    pub fn best_threshold(&self) -> f64 {
        self.threshold
    }
}

impl TrainedModel for TrainedThresholdedClassifier {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        let pred = self.base.predict(features)?;
        match pred {
            Prediction::Classification {
                truth,
                probabilities: Some(probs),
                ..
            } => {
                let predicted: Vec<usize> = probs
                    .iter()
                    .map(|row| usize::from(row.get(1).copied().unwrap_or(0.0) >= self.threshold))
                    .collect();
                Ok(Prediction::Classification {
                    predicted,
                    truth,
                    probabilities: Some(probs),
                })
            }
            Prediction::Classification {
                probabilities: None,
                ..
            } => Err(SmeltError::IncompatiblePrediction(
                "ThresholdedClassifier requires a base learner that produces probabilities".into(),
            )),
            _ => Err(SmeltError::IncompatiblePrediction(
                "ThresholdedClassifier requires classification predictions".into(),
            )),
        }
    }

    fn feature_importance(&self) -> Option<Vec<(String, f64)>> {
        self.base.feature_importance()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::LogisticRegression;
    use crate::measure::{BalancedAccuracy, F1Score};
    use ndarray::Array2;

    struct Rng(u64);
    impl Rng {
        fn next_f64(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// Heavily imbalanced binary data (~10% positives) with strong class
    /// overlap (close centers, wide spread). Because the rare positive class
    /// is hard to separate, a logistic model pulls most probabilities below
    /// 0.5 (toward the 10% prior), so the default 0.5 rule badly under-
    /// predicts positives — exactly where threshold tuning helps.
    fn imbalanced(n: usize, seed: u64) -> (Array2<f64>, Vec<usize>) {
        let mut rng = Rng(seed);
        let mut feats = Vec::with_capacity(n * 2);
        let mut target = Vec::with_capacity(n);
        for _ in 0..n {
            let class = if rng.next_f64() < 0.10 { 1usize } else { 0 };
            let center = if class == 1 { 1.0 } else { 0.0 };
            feats.push(center + (rng.next_f64() - 0.5) * 4.0);
            feats.push(center + (rng.next_f64() - 0.5) * 4.0);
            target.push(class);
        }
        (Array2::from_shape_vec((n, 2), feats).unwrap(), target)
    }

    fn f1_of(model: &dyn TrainedModel, x: &Array2<f64>, y: &[usize]) -> f64 {
        let pred = model.predict(x).unwrap().with_truth_classif(y.to_vec());
        F1Score.score(&pred).unwrap()
    }

    #[test]
    fn registered_id_and_best_threshold_exposed() {
        let tc = ThresholdedClassifier::new(|| Box::new(LogisticRegression::new()))
            .with_threshold(0.42);
        assert_eq!(tc.id(), "thresholded");
        let (x, y) = imbalanced(80, 1);
        let task = ClassificationTask::new("t", x, y).unwrap();
        let trained = tc.fit_classif(&task).unwrap();
        assert!((trained.best_threshold() - 0.42).abs() < 1e-12);
    }

    /// The core oracle: on imbalanced data, tuning the threshold for F1 beats
    /// the default 0.5 decision rule of the same base learner.
    #[test]
    fn tuned_threshold_beats_default_on_imbalanced_data() {
        let (xtr, ytr) = imbalanced(1200, 2);
        let (xte, yte) = imbalanced(800, 909);
        let task = ClassificationTask::new("t", xtr.clone(), ytr.clone()).unwrap();

        // Default 0.5 rule (plain base).
        let plain = LogisticRegression::new().train_classif(&task).unwrap();
        let plain_f1 = f1_of(&*plain, &xte, &yte);

        // Tuned-for-F1 wrapper.
        let tc = ThresholdedClassifier::new(|| Box::new(LogisticRegression::new())).with_seed(3);
        let tuned = tc.fit_classif(&task).unwrap();
        let tuned_f1 = f1_of(&tuned, &xte, &yte);

        assert!(
            tuned_f1 >= plain_f1,
            "tuned threshold should not lose to 0.5 on F1: tuned={tuned_f1:.4} \
             (t={:.3}) vs plain={plain_f1:.4}",
            tuned.best_threshold()
        );
        // On this genuinely imbalanced fixture it should strictly win.
        assert!(
            tuned_f1 > plain_f1 + 0.02,
            "expected a real F1 gain from threshold tuning: tuned={tuned_f1:.4} vs plain={plain_f1:.4}"
        );
    }

    /// Sanity: an extreme fixed threshold actually flips the whole prediction.
    #[test]
    fn extreme_fixed_thresholds_saturate_predictions() {
        let (x, y) = imbalanced(400, 4);
        let task = ClassificationTask::new("t", x.clone(), y).unwrap();

        let hi = ThresholdedClassifier::new(|| Box::new(LogisticRegression::new()))
            .with_threshold(0.99)
            .fit_classif(&task)
            .unwrap();
        // Threshold 0.0: every probability is >= 0, so all samples become
        // class 1 regardless of how confident the base model is.
        let lo = ThresholdedClassifier::new(|| Box::new(LogisticRegression::new()))
            .with_threshold(0.0)
            .fit_classif(&task)
            .unwrap();

        let count1 = |m: &TrainedThresholdedClassifier| match m.predict(&x).unwrap() {
            Prediction::Classification { predicted, .. } => {
                predicted.iter().filter(|&&p| p == 1).count()
            }
            _ => panic!(),
        };
        let n = x.nrows();
        assert!(count1(&hi) <= n / 20, "threshold 0.99 => almost all class 0");
        assert_eq!(count1(&lo), n, "threshold 0.0 => all class 1");
    }

    #[test]
    fn tuning_by_balanced_accuracy_also_works() {
        let (xtr, ytr) = imbalanced(700, 5);
        let (xte, yte) = imbalanced(400, 55);
        let task = ClassificationTask::new("t", xtr, ytr).unwrap();

        let plain = LogisticRegression::new().train_classif(&task).unwrap();
        let plain_bacc = BalancedAccuracy
            .score(&plain.predict(&xte).unwrap().with_truth_classif(yte.clone()))
            .unwrap();

        let tc = ThresholdedClassifier::new(|| Box::new(LogisticRegression::new()))
            .with_metric(Box::new(BalancedAccuracy))
            .with_seed(6);
        let tuned = tc.fit_classif(&task).unwrap();
        let tuned_bacc = BalancedAccuracy
            .score(&tuned.predict(&xte).unwrap().with_truth_classif(yte))
            .unwrap();
        assert!(
            tuned_bacc >= plain_bacc,
            "tuned balanced accuracy should not lose to 0.5: tuned={tuned_bacc:.4} vs plain={plain_bacc:.4}"
        );
    }

    #[test]
    fn rejects_multiclass_and_weights() {
        // Multiclass rejected.
        let x = Array2::from_shape_fn((30, 2), |(i, j)| (i + j) as f64);
        let y: Vec<usize> = (0..30).map(|i| i % 3).collect();
        let task = ClassificationTask::new("mc", x, y).unwrap();
        let tc = ThresholdedClassifier::new(|| Box::new(LogisticRegression::new()));
        assert!(tc.fit_classif(&task).is_err());

        // Weighted rejected.
        let (x2, y2) = imbalanced(40, 7);
        let wtask = ClassificationTask::new("w", x2, y2)
            .unwrap()
            .with_weights(vec![1.0; 40]);
        let tc2 = ThresholdedClassifier::new(|| Box::new(LogisticRegression::new()));
        let err = tc2.fit_classif(&wtask).map(|_| ()).unwrap_err();
        assert!(format!("{err}").contains("does not support sample weights"));
    }
}
