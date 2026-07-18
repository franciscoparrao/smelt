//! Probability calibration: wraps any probabilistic classifier and remaps its
//! predicted probabilities so they better match observed frequencies (a
//! prediction of "0.8" should be right ~80% of the time). The smelt analogue
//! of scikit-learn's `CalibratedClassifierCV` / mlr3's `PipeOpCalibration`.
//!
//! Platt, J. (1999). "Probabilistic Outputs for Support Vector Machines and
//! Comparisons to Regularized Likelihood Methods." (sigmoid / logistic
//! calibration.) Zadrozny, B. & Elkan, C. (2002). "Transforming Classifier
//! Scores into Accurate Multiclass Probability Estimates." (isotonic
//! calibration.)
//!
//! Many strong classifiers rank well but are *miscalibrated*: a deep decision
//! tree or a naive-Bayes model routinely reports probabilities far more
//! extreme (near 0/1) than the true class frequencies, so any downstream
//! decision that reads the probability itself (expected-cost thresholding,
//! risk scoring, conformal-style abstention) is distorted. Calibration fixes
//! the *values* without touching the *ranking* (Platt is strictly monotone, so
//! AUC is preserved).
//!
//! # Strategy: holdout split + refit-on-all ("prefit"-like)
//!
//! [`CalibratedClassifier`] splits the training data into a *fit-set* and a
//! *calibration-set* (a single [`Holdout`], `calib_fraction` of the rows held
//! out, seed-controlled). It trains the base learner on the fit-set, reads its
//! probabilities on the (unseen) calibration-set, fits the calibrator against
//! those, and then **refits the base learner on the whole training set** — the
//! model actually used at predict time. This matches scikit-learn's
//! `CalibratedClassifierCV(ensemble=False)`: one final base model trained on
//! all the data, with a single calibrator learned from held-out scores.
//!
//! The one deliberate simplification versus scikit-learn's default is the
//! split: scikit-learn uses k-fold `cross_val_predict` to gather the held-out
//! calibration scores (so every training row contributes an out-of-fold
//! score), whereas this uses a single holdout. Holdout is the minimum viable
//! version — cheaper, simpler, and unbiased for the calibrator — at the cost
//! of the calibrator seeing fewer points. A CV variant is a possible future
//! opt-in, deliberately not implemented here.
//!
//! # Multiclass
//!
//! Binary calibration fits one 1-D calibrator on the positive-class score.
//! Multiclass uses the standard one-vs-rest extension: one calibrator per
//! class (score of class `c` vs. the indicator `y == c`), then renormalize the
//! per-class calibrated values to sum to 1 (Zadrozny & Elkan 2002).
//!
//! # Registry
//!
//! Like `Bagging`/`Stacking`/`CostSensitiveClassifier`, this wrapper needs a
//! base-learner factory with no sensible default, so it is **not**
//! constructible via [`crate::learner::registry::learner_from_id`].

use crate::Result;
use crate::SmeltError;
use crate::learner::{Learner, LearnerProperties, TrainedModel};
use crate::prediction::Prediction;
use crate::resample::{Holdout, Resample};
use crate::task::{ClassificationTask, Task};
use ndarray::{Array2, Axis};

/// How predicted probabilities are remapped to calibrated ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationMethod {
    /// Platt scaling: a 1-D logistic `1/(1+exp(a·s + b))` fit by maximum
    /// likelihood on the held-out scores, with Platt's smoothed targets
    /// (`y+ = (N+ +1)/(N+ +2)`, `y- = 1/(N- +2)`) to avoid overfitting when
    /// the calibration scores are nearly separable. Strictly monotone, so it
    /// preserves the base model's ranking (and hence AUC).
    Platt,
    /// Isotonic regression: a free-form non-decreasing step function fit by
    /// the pool-adjacent-violators algorithm. More flexible than Platt (can
    /// correct non-sigmoidal miscalibration) but needs more calibration data
    /// and can overfit small sets. Non-*strictly* monotone (flat segments),
    /// so ranking is preserved only weakly.
    Isotonic,
}

/// A fitted 1-D calibrator mapping a raw score in `[0, 1]` to a calibrated
/// probability in `[0, 1]`.
enum Calibrator {
    /// Platt sigmoid parameters: `p = 1/(1 + exp(a·s + b))`.
    Platt { a: f64, b: f64 },
    /// Isotonic step function as interpolation nodes: `xs` strictly
    /// increasing, `ys` non-decreasing; linear interpolation between nodes,
    /// clamped to the endpoints outside `[xs[0], xs[last]]`.
    Isotonic { xs: Vec<f64>, ys: Vec<f64> },
}

impl Calibrator {
    /// Map a raw score to its calibrated probability.
    fn apply(&self, s: f64) -> f64 {
        match self {
            Calibrator::Platt { a, b } => {
                let f = a * s + b;
                // 1/(1+exp(f)), numerically stable.
                let p = if f >= 0.0 {
                    let e = (-f).exp();
                    e / (1.0 + e)
                } else {
                    1.0 / (1.0 + f.exp())
                };
                p.clamp(0.0, 1.0)
            }
            Calibrator::Isotonic { xs, ys } => interpolate(xs, ys, s).clamp(0.0, 1.0),
        }
    }
}

/// Fit Platt's sigmoid `p = 1/(1+exp(a·s + b))` by Newton's method on the
/// (convex) smoothed-target log-likelihood. The smoothing (`hi`/`lo` targets
/// instead of hard 0/1) keeps `a` finite even when the calibration scores are
/// perfectly separable — the whole point of Platt's 1999 refinement.
fn fit_platt(scores: &[f64], labels: &[bool]) -> Calibrator {
    let n_pos = labels.iter().filter(|&&l| l).count() as f64;
    let n_neg = labels.len() as f64 - n_pos;
    // Platt's smoothed targets.
    let hi = (n_pos + 1.0) / (n_pos + 2.0);
    let lo = 1.0 / (n_neg + 2.0);
    let targets: Vec<f64> = labels.iter().map(|&l| if l { hi } else { lo }).collect();

    let mut a = 0.0_f64;
    // Platt's initialisation for b (at a = 0, p = 1/(1+exp(b)) ≈ prior).
    let mut b = ((n_neg + 1.0) / (n_pos + 1.0)).ln();
    if !b.is_finite() {
        b = 0.0;
    }

    for _ in 0..100 {
        let (mut ga, mut gb) = (0.0_f64, 0.0_f64);
        let (mut haa, mut hab, mut hbb) = (0.0_f64, 0.0_f64, 0.0_f64);
        for (i, &s) in scores.iter().enumerate() {
            let f = a * s + b;
            let p = if f >= 0.0 {
                let e = (-f).exp();
                e / (1.0 + e)
            } else {
                1.0 / (1.0 + f.exp())
            };
            let d = targets[i] - p; // dL/df
            ga += d * s;
            gb += d;
            let w = p * (1.0 - p);
            haa += w * s * s;
            hab += w * s;
            hbb += w;
        }
        // Newton step: solve H·δ = grad, params -= δ. Diagonal damping keeps
        // the 2x2 solve non-singular as the weights w -> 0 near separation.
        haa += 1e-10;
        hbb += 1e-10;
        let det = haa * hbb - hab * hab;
        if det.abs() < 1e-300 {
            break;
        }
        let da = (hbb * ga - hab * gb) / det;
        let db = (haa * gb - hab * ga) / det;
        if !da.is_finite() || !db.is_finite() {
            break;
        }
        a -= da;
        b -= db;
        if ga.abs() + gb.abs() < 1e-9 {
            break;
        }
    }
    Calibrator::Platt { a, b }
}

/// Fit an isotonic (non-decreasing) regression of `labels` on `scores` via the
/// pool-adjacent-violators algorithm, returning interpolation nodes.
fn fit_isotonic(scores: &[f64], labels: &[bool]) -> Calibrator {
    let mut points: Vec<(f64, f64)> = scores
        .iter()
        .zip(labels)
        .map(|(&s, &l)| (s, if l { 1.0 } else { 0.0 }))
        .collect();
    points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // PAV: a stack of blocks (pooled mean, weight, count-of-points).
    let mut vals: Vec<f64> = Vec::new();
    let mut wts: Vec<f64> = Vec::new();
    let mut counts: Vec<usize> = Vec::new();
    for &(_, y) in &points {
        let mut cur_v = y;
        let mut cur_w = 1.0_f64;
        let mut cur_c = 1usize;
        while let Some(&last_v) = vals.last() {
            if last_v <= cur_v {
                break;
            }
            let lw = wts.pop().unwrap();
            let lc = counts.pop().unwrap();
            vals.pop();
            cur_v = (last_v * lw + cur_v * cur_w) / (lw + cur_w);
            cur_w += lw;
            cur_c += lc;
        }
        vals.push(cur_v);
        wts.push(cur_w);
        counts.push(cur_c);
    }

    // Expand block means back to per-point fitted values, then collapse to
    // interpolation nodes (dedup equal x, keeping the last — monotone — y).
    let mut fitted: Vec<f64> = Vec::with_capacity(points.len());
    for (v, c) in vals.iter().zip(&counts) {
        for _ in 0..*c {
            fitted.push(*v);
        }
    }
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    for (&(x, _), &fy) in points.iter().zip(&fitted) {
        if let Some(&last_x) = xs.last()
            && (x - last_x).abs() <= f64::EPSILON
        {
            *ys.last_mut().unwrap() = fy;
            continue;
        }
        xs.push(x);
        ys.push(fy);
    }
    if xs.is_empty() {
        // No calibration points at all: fall back to the base rate (0.5).
        xs.push(0.0);
        ys.push(0.5);
    }
    Calibrator::Isotonic { xs, ys }
}

/// Piecewise-linear interpolation over `(xs, ys)` (xs strictly increasing),
/// clamped to the endpoints outside the fitted range.
fn interpolate(xs: &[f64], ys: &[f64], q: f64) -> f64 {
    if xs.len() == 1 || q <= xs[0] {
        return ys[0];
    }
    let last = xs.len() - 1;
    if q >= xs[last] {
        return ys[last];
    }
    // Binary search for the bracket xs[k] <= q < xs[k+1].
    let k = match xs.binary_search_by(|v| v.partial_cmp(&q).unwrap_or(std::cmp::Ordering::Equal)) {
        Ok(i) => return ys[i],
        Err(i) => i - 1,
    };
    let (x0, x1) = (xs[k], xs[k + 1]);
    let (y0, y1) = (ys[k], ys[k + 1]);
    if (x1 - x0).abs() < f64::EPSILON {
        return y1;
    }
    y0 + (y1 - y0) * (q - x0) / (x1 - x0)
}

fn argmax(row: &[f64]) -> usize {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Probability-calibration wrapper around any probabilistic classifier.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use smelt_ml::learner::{CalibratedClassifier, CalibrationMethod};
/// use ndarray::array;
///
/// let features = array![
///     [0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [0.0, 0.2],
///     [1.0, 1.0], [1.1, 0.9], [0.9, 1.1], [1.0, 0.9]
/// ];
/// let target = vec![0, 0, 0, 0, 1, 1, 1, 1];
/// let task = ClassificationTask::new("calib_demo", features, target).unwrap();
///
/// let mut cc = CalibratedClassifier::new(
///     || Box::new(DecisionTree::default()),
///     CalibrationMethod::Platt,
/// ).with_seed(0);
/// let model = cc.train_classif(&task).unwrap();
/// ```
pub struct CalibratedClassifier {
    factory: Box<dyn Fn() -> Box<dyn Learner> + Send + Sync>,
    method: CalibrationMethod,
    calib_fraction: f64,
    seed: u64,
}

impl CalibratedClassifier {
    /// Create a calibration wrapper from a base-learner factory and the
    /// calibration method. Defaults: `calib_fraction = 0.3`, `seed = 42`.
    pub fn new(
        factory: impl Fn() -> Box<dyn Learner> + Send + Sync + 'static,
        method: CalibrationMethod,
    ) -> Self {
        Self {
            factory: Box::new(factory),
            method,
            calib_fraction: 0.3,
            seed: 42,
        }
    }

    /// Fraction of the training data held out as the calibration set (the rest
    /// is the fit-set the base learner is trained on before the base is refit
    /// on everything). Must be in `(0, 1)`; default `0.3`.
    pub fn with_calib_fraction(mut self, fraction: f64) -> Self {
        self.calib_fraction = fraction;
        self
    }

    /// RNG seed for the internal fit/calibration holdout split. Reproducible:
    /// two runs with the same seed produce the same calibrator.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Build a subset task on `idx`, propagating feature names/types and the
    /// full class-name set so a subset that happens to drop a class still
    /// reports the parent's `n_classes` (keeping probability rows full-width).
    fn subset_task(task: &ClassificationTask, idx: &[usize]) -> Result<ClassificationTask> {
        let features = task.features().select(Axis(0), idx);
        let target: Vec<usize> = idx.iter().map(|&i| task.target()[i]).collect();
        Ok(ClassificationTask::new(task.id(), features, target)?
            .with_feature_names(task.feature_names().to_vec())?
            .with_feature_types(task.feature_types().to_vec())?
            .with_class_names(task.class_names().to_vec()))
    }
}

impl Learner for CalibratedClassifier {
    fn id(&self) -> &str {
        "calibrated"
    }

    fn properties(&self) -> LearnerProperties {
        // Calibration exists precisely to make `supports_proba` meaningful.
        // Conservatively does NOT advertise feature importance (the base may
        // or may not provide it); the trained model still delegates it when
        // present.
        LearnerProperties::classifier().with_proba()
    }

    fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        crate::validate::check_no_weights(task.weights(), "CalibratedClassifier")?;
        if !(self.calib_fraction > 0.0 && self.calib_fraction < 1.0) {
            return Err(SmeltError::InvalidParameter(format!(
                "CalibratedClassifier calib_fraction must be in (0, 1), got {}",
                self.calib_fraction
            )));
        }
        let n = task.n_samples();
        let n_classes = task.n_classes();

        // Split into fit-set (trains the base) and calib-set (fits the
        // calibrator). Holdout's ratio is the *training* fraction.
        let splits = Holdout::new(1.0 - self.calib_fraction)
            .with_seed(self.seed)
            .splits(n)?;
        let (fit_idx, calib_idx) = &splits[0];
        if fit_idx.is_empty() || calib_idx.is_empty() {
            return Err(SmeltError::InvalidParameter(format!(
                "CalibratedClassifier: the fit/calibration split left an empty side \
                 ({} fit, {} calib rows from {n} samples); use more data or a different \
                 calib_fraction",
                fit_idx.len(),
                calib_idx.len()
            )));
        }

        // Train the base on the fit-set, read its calibration-set probabilities.
        let fit_task = Self::subset_task(task, fit_idx)?;
        let mut fit_base = (self.factory)();
        let fit_model = fit_base.train_classif(&fit_task)?;
        let calib_features = task.features().select(Axis(0), calib_idx);
        let calib_pred = fit_model.predict(&calib_features)?;
        let calib_probs = match calib_pred {
            Prediction::Classification {
                probabilities: Some(p),
                ..
            } => p,
            Prediction::Classification {
                probabilities: None,
                ..
            } => {
                return Err(SmeltError::IncompatiblePrediction(
                    "CalibratedClassifier requires a base learner that produces probabilities \
                     (e.g. logistic_regression, random_forest, decision_tree, gaussian_nb)"
                        .into(),
                ));
            }
            _ => {
                return Err(SmeltError::IncompatiblePrediction(
                    "CalibratedClassifier requires classification predictions".into(),
                ));
            }
        };
        let calib_labels: Vec<usize> = calib_idx.iter().map(|&i| task.target()[i]).collect();

        // Fit calibrator(s): binary -> one on the positive-class score;
        // multiclass -> one-vs-rest per class.
        let binary = n_classes == 2;
        let fit_one = |c: usize| -> Calibrator {
            let scores: Vec<f64> = calib_probs
                .iter()
                .map(|row| row.get(c).copied().unwrap_or(0.0))
                .collect();
            let labels: Vec<bool> = calib_labels.iter().map(|&y| y == c).collect();
            match self.method {
                CalibrationMethod::Platt => fit_platt(&scores, &labels),
                CalibrationMethod::Isotonic => fit_isotonic(&scores, &labels),
            }
        };
        let calibrators: Vec<Calibrator> = if binary {
            vec![fit_one(1)]
        } else {
            (0..n_classes).map(fit_one).collect()
        };

        // Refit the base on ALL the training data (the model used at predict
        // time), scikit-learn `ensemble=False` style.
        let mut final_base = (self.factory)();
        let final_model = final_base.train_classif(task)?;

        Ok(Box::new(TrainedCalibratedClassifier {
            base: final_model,
            calibrators,
            binary,
            n_classes,
        }))
    }
}

/// A trained [`CalibratedClassifier`]: the refit base model plus the fitted
/// calibrator(s). `predict` returns calibrated, renormalized probabilities and
/// their argmax.
pub struct TrainedCalibratedClassifier {
    base: Box<dyn TrainedModel>,
    calibrators: Vec<Calibrator>,
    binary: bool,
    n_classes: usize,
}

impl TrainedModel for TrainedCalibratedClassifier {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        let pred = self.base.predict(features)?;
        let (truth, probs) = match pred {
            Prediction::Classification {
                truth,
                probabilities: Some(probs),
                ..
            } => (truth, probs),
            Prediction::Classification {
                probabilities: None,
                ..
            } => {
                return Err(SmeltError::IncompatiblePrediction(
                    "CalibratedClassifier requires a base learner that produces probabilities"
                        .into(),
                ));
            }
            _ => {
                return Err(SmeltError::IncompatiblePrediction(
                    "CalibratedClassifier requires classification predictions".into(),
                ));
            }
        };

        let n = self.n_classes;
        let calibrated: Vec<Vec<f64>> = probs
            .iter()
            .map(|row| {
                if self.binary {
                    let s1 = row.get(1).copied().unwrap_or(0.0);
                    let c = self.calibrators[0].apply(s1);
                    vec![1.0 - c, c]
                } else {
                    let mut cal: Vec<f64> = (0..n)
                        .map(|c| self.calibrators[c].apply(row.get(c).copied().unwrap_or(0.0)))
                        .collect();
                    let sum: f64 = cal.iter().sum();
                    if sum > 0.0 {
                        for v in &mut cal {
                            *v /= sum;
                        }
                    } else {
                        cal = vec![1.0 / n as f64; n];
                    }
                    cal
                }
            })
            .collect();

        let predicted: Vec<usize> = calibrated.iter().map(|row| argmax(row)).collect();
        Ok(Prediction::Classification {
            predicted,
            truth,
            probabilities: Some(calibrated),
        })
    }

    fn feature_importance(&self) -> Option<Vec<(String, f64)>> {
        self.base.feature_importance()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::{DecisionTree, LogisticRegression};
    use crate::measure::{AucRoc, Brier, Measure};
    use ndarray::Array2;

    /// Deterministic xorshift pseudo-noise, so the fixtures are reproducible
    /// without a rand dependency in the test.
    struct Rng(u64);
    impl Rng {
        fn next_f64(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// Two overlapping Gaussians in 2-D: the Bayes-optimal classifier is far
    /// from perfect, so a deep tree is overconfident (miscalibrated) on unseen
    /// data — exactly the regime calibration is meant to fix.
    fn noisy_binary(n: usize, seed: u64) -> (Array2<f64>, Vec<usize>) {
        let mut rng = Rng(seed);
        let mut feats = Vec::with_capacity(n * 2);
        let mut target = Vec::with_capacity(n);
        for _ in 0..n {
            let class = if rng.next_f64() < 0.5 { 0usize } else { 1 };
            let center = if class == 0 { 0.0 } else { 1.2 };
            // Wide spread => heavy overlap between the two classes.
            let x0 = center + (rng.next_f64() - 0.5) * 4.0;
            let x1 = center + (rng.next_f64() - 0.5) * 4.0;
            feats.push(x0);
            feats.push(x1);
            target.push(class);
        }
        (Array2::from_shape_vec((n, 2), feats).unwrap(), target)
    }

    fn brier_on(model: &dyn TrainedModel, x: &Array2<f64>, y: &[usize]) -> f64 {
        let pred = model
            .predict(x)
            .unwrap()
            .with_truth_classif(y.to_vec());
        Brier.score(&pred).unwrap()
    }

    #[test]
    fn registered_id() {
        let cc = CalibratedClassifier::new(|| Box::new(DecisionTree::default()), CalibrationMethod::Platt);
        assert_eq!(cc.id(), "calibrated");
    }

    /// The core oracle: on a miscalibrated base (a deep tree over overlapping
    /// classes) evaluated on a held-out test set, BOTH Platt and isotonic
    /// calibration reduce the Brier score versus the raw base model.
    #[test]
    fn calibration_improves_brier_on_miscalibrated_tree() {
        let (xtr, ytr) = noisy_binary(600, 1);
        let (xte, yte) = noisy_binary(400, 999);
        let task = ClassificationTask::new("t", xtr.clone(), ytr.clone()).unwrap();

        let deep = || Box::new(DecisionTree::new().with_max_depth(20)) as Box<dyn Learner>;
        let raw_model = deep().train_classif(&task).unwrap();
        let raw_brier = brier_on(&*raw_model, &xte, &yte);

        for method in [CalibrationMethod::Platt, CalibrationMethod::Isotonic] {
            let mut cc = CalibratedClassifier::new(deep, method).with_seed(7);
            let cal_model = cc.train_classif(&task).unwrap();
            let cal_brier = brier_on(&*cal_model, &xte, &yte);
            assert!(
                cal_brier < raw_brier,
                "{method:?}: calibration should reduce Brier on a miscalibrated tree: \
                 cal={cal_brier:.4} vs raw={raw_brier:.4}"
            );
        }
    }

    /// Platt is strictly monotone, so calibrating a base model must preserve
    /// its ranking and therefore its AUC exactly (up to float noise).
    #[test]
    fn platt_preserves_auc() {
        let (xtr, ytr) = noisy_binary(500, 3);
        let (xte, yte) = noisy_binary(300, 42);
        let task = ClassificationTask::new("t", xtr, ytr).unwrap();

        let raw_model = LogisticRegression::new().train_classif(&task).unwrap();
        let raw_auc = AucRoc
            .score(&raw_model.predict(&xte).unwrap().with_truth_classif(yte.clone()))
            .unwrap();

        let mut cc = CalibratedClassifier::new(
            || Box::new(LogisticRegression::new()),
            CalibrationMethod::Platt,
        )
        .with_seed(5);
        let cal_model = cc.train_classif(&task).unwrap();
        let cal_auc = AucRoc
            .score(&cal_model.predict(&xte).unwrap().with_truth_classif(yte))
            .unwrap();

        assert!(
            (cal_auc - raw_auc).abs() < 1e-9,
            "Platt must preserve AUC (monotone map): cal={cal_auc} vs raw={raw_auc}"
        );
    }

    /// Calibrating an already well-calibrated model (logistic regression on
    /// well-separated data) must not blow up the Brier score.
    #[test]
    fn calibration_does_not_wreck_a_well_calibrated_model() {
        let (xtr, ytr) = noisy_binary(500, 11);
        let (xte, yte) = noisy_binary(300, 77);
        let task = ClassificationTask::new("t", xtr.clone(), ytr.clone()).unwrap();

        let raw_model = LogisticRegression::new().train_classif(&task).unwrap();
        let raw_brier = brier_on(&*raw_model, &xte, &yte);

        let mut cc = CalibratedClassifier::new(
            || Box::new(LogisticRegression::new()),
            CalibrationMethod::Platt,
        )
        .with_seed(9);
        let cal_model = cc.train_classif(&task).unwrap();
        let cal_brier = brier_on(&*cal_model, &xte, &yte);

        assert!(
            cal_brier <= raw_brier + 0.05,
            "calibration should not badly degrade an already-calibrated model: \
             cal={cal_brier:.4} vs raw={raw_brier:.4}"
        );
    }

    /// Multiclass one-vs-rest: probability rows come back full-width,
    /// normalized (sum to 1), and consistent with the argmax label.
    #[test]
    fn multiclass_probs_are_normalized_and_consistent() {
        let mut feats = Vec::new();
        let mut target = Vec::new();
        let mut rng = Rng(123);
        for _ in 0..300 {
            let class = (rng.next_f64() * 3.0) as usize % 3;
            let c = class as f64 * 2.0;
            feats.push(c + (rng.next_f64() - 0.5));
            feats.push(c + (rng.next_f64() - 0.5));
            target.push(class);
        }
        let x = Array2::from_shape_vec((300, 2), feats).unwrap();
        let task = ClassificationTask::new("mc", x.clone(), target).unwrap();

        let mut cc = CalibratedClassifier::new(
            || Box::new(DecisionTree::new().with_max_depth(10)),
            CalibrationMethod::Isotonic,
        )
        .with_seed(2);
        let model = cc.train_classif(&task).unwrap();
        let pred = model.predict(&x).unwrap();
        let Prediction::Classification {
            predicted,
            probabilities: Some(probs),
            ..
        } = pred
        else {
            panic!("expected classification with probabilities");
        };
        for (row, &lab) in probs.iter().zip(&predicted) {
            assert_eq!(row.len(), 3, "full-width probability rows");
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "rows must sum to 1, got {sum}");
            assert_eq!(argmax(row), lab, "argmax must equal the predicted label");
        }
    }

    #[test]
    fn rejects_non_probabilistic_base() {
        // A stub base learner that emits hard labels only (no probabilities):
        // calibration has nothing to calibrate and must error clearly.
        struct NoProba;
        struct NoProbaModel;
        impl TrainedModel for NoProbaModel {
            fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
                Ok(Prediction::Classification {
                    predicted: vec![0; features.nrows()],
                    truth: None,
                    probabilities: None,
                })
            }
        }
        impl Learner for NoProba {
            fn id(&self) -> &str {
                "no_proba"
            }
            fn properties(&self) -> LearnerProperties {
                LearnerProperties::classifier()
            }
            fn train_classif(
                &mut self,
                _t: &ClassificationTask,
            ) -> Result<Box<dyn TrainedModel>> {
                Ok(Box::new(NoProbaModel))
            }
        }

        let (x, y) = noisy_binary(60, 4);
        let task = ClassificationTask::new("t", x, y).unwrap();
        let mut cc = CalibratedClassifier::new(|| Box::new(NoProba), CalibrationMethod::Platt)
            .with_seed(1);
        assert!(cc.train_classif(&task).is_err());
    }

    #[test]
    fn rejects_bad_calib_fraction_and_weights() {
        let (x, y) = noisy_binary(40, 6);
        let task = ClassificationTask::new("t", x.clone(), y.clone()).unwrap();
        let mut bad = CalibratedClassifier::new(
            || Box::new(DecisionTree::default()),
            CalibrationMethod::Platt,
        )
        .with_calib_fraction(1.5);
        assert!(bad.train_classif(&task).is_err());

        let wtask = ClassificationTask::new("t", x, y)
            .unwrap()
            .with_weights(vec![1.0; 40]);
        let mut cc =
            CalibratedClassifier::new(|| Box::new(DecisionTree::default()), CalibrationMethod::Platt);
        let err = cc.train_classif(&wtask).map(|_| ()).unwrap_err();
        assert!(format!("{err}").contains("does not support sample weights"));
    }

    /// Same seed => identical calibrated probabilities (deterministic split).
    #[test]
    fn deterministic_under_fixed_seed() {
        let (x, y) = noisy_binary(200, 8);
        let task = ClassificationTask::new("t", x.clone(), y).unwrap();
        let run = || {
            let mut cc = CalibratedClassifier::new(
                || Box::new(DecisionTree::new().with_max_depth(8)),
                CalibrationMethod::Platt,
            )
            .with_seed(321);
            let m = cc.train_classif(&task).unwrap();
            match m.predict(&x).unwrap() {
                Prediction::Classification {
                    probabilities: Some(p),
                    ..
                } => p,
                _ => panic!(),
            }
        };
        assert_eq!(run(), run(), "same seed must give identical calibrated probs");
    }
}
