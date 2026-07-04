//! Measures: evaluate prediction quality.
//!
//! Classification: Accuracy, Precision, Recall, F1Score, LogLoss, AucRoc,
//! BalancedAccuracy, CohensKappa, Mcc, Brier.
//! Regression: Rmse, Mae, RSquared, Mape.

use crate::prediction::Prediction;
use crate::{Result, SmeltError};

/// Trait for evaluation metrics.
pub trait Measure: Send + Sync {
    /// Metric identifier (e.g., "classif.accuracy").
    fn id(&self) -> &str;
    /// Compute score. Higher is better for maximize=true, lower for maximize=false.
    fn score(&self, prediction: &Prediction) -> Result<f64>;
    /// Whether higher values are better.
    fn maximize(&self) -> bool;
}

/// Classification accuracy.
pub struct Accuracy;

impl Measure for Accuracy {
    fn id(&self) -> &str {
        "classif.accuracy"
    }
    fn maximize(&self) -> bool {
        true
    }

    fn score(&self, prediction: &Prediction) -> Result<f64> {
        match prediction {
            Prediction::Classification {
                predicted,
                truth: Some(truth),
                ..
            } => {
                let correct = predicted.iter().zip(truth).filter(|(p, t)| p == t).count();
                Ok(correct as f64 / predicted.len() as f64)
            }
            _ => Err(SmeltError::IncompatiblePrediction(
                "Accuracy requires classification prediction with truth".into(),
            )),
        }
    }
}

/// Root Mean Squared Error.
pub struct Rmse;

impl Measure for Rmse {
    fn id(&self) -> &str {
        "regr.rmse"
    }
    fn maximize(&self) -> bool {
        false
    }

    fn score(&self, prediction: &Prediction) -> Result<f64> {
        match prediction {
            Prediction::Regression {
                predicted,
                truth: Some(truth),
            } => {
                let mse: f64 = predicted
                    .iter()
                    .zip(truth)
                    .map(|(p, t)| (p - t).powi(2))
                    .sum::<f64>()
                    / predicted.len() as f64;
                Ok(mse.sqrt())
            }
            _ => Err(SmeltError::IncompatiblePrediction(
                "RMSE requires regression prediction with truth".into(),
            )),
        }
    }
}

/// Mean Absolute Error.
pub struct Mae;

impl Measure for Mae {
    fn id(&self) -> &str {
        "regr.mae"
    }
    fn maximize(&self) -> bool {
        false
    }

    fn score(&self, prediction: &Prediction) -> Result<f64> {
        match prediction {
            Prediction::Regression {
                predicted,
                truth: Some(truth),
            } => {
                let mae: f64 = predicted
                    .iter()
                    .zip(truth)
                    .map(|(p, t)| (p - t).abs())
                    .sum::<f64>()
                    / predicted.len() as f64;
                Ok(mae)
            }
            _ => Err(SmeltError::IncompatiblePrediction(
                "MAE requires regression prediction with truth".into(),
            )),
        }
    }
}

// ── Classification helpers ──────────────────────────────────────────

fn n_classes(predicted: &[usize], truth: &[usize]) -> usize {
    let max_pred = predicted.iter().max().copied().unwrap_or(0);
    let max_truth = truth.iter().max().copied().unwrap_or(0);
    max_pred.max(max_truth) + 1
}

/// Per-class true positives, false positives, false negatives.
fn class_counts(
    predicted: &[usize],
    truth: &[usize],
    n_classes: usize,
) -> Vec<(usize, usize, usize)> {
    let mut counts = vec![(0usize, 0usize, 0usize); n_classes]; // (tp, fp, fn)
    for (&p, &t) in predicted.iter().zip(truth) {
        if p == t {
            counts[p].0 += 1; // TP
        } else {
            counts[p].1 += 1; // FP for predicted class
            counts[t].2 += 1; // FN for true class
        }
    }
    counts
}

// ── Classification metrics ──────────────────────────────────────────

/// Macro-averaged precision.
pub struct Precision;

impl Measure for Precision {
    fn id(&self) -> &str {
        "classif.precision"
    }
    fn maximize(&self) -> bool {
        true
    }

    fn score(&self, prediction: &Prediction) -> Result<f64> {
        match prediction {
            Prediction::Classification {
                predicted,
                truth: Some(truth),
                ..
            } => {
                let nc = n_classes(predicted, truth);
                let counts = class_counts(predicted, truth, nc);
                // Macro average over ALL classes present in truth ∪ predicted
                // (not just the ones with a defined precision) -- a class the
                // model never predicts contributes 0, matching sklearn's
                // `zero_division=0` convention. Averaging over only the
                // "valid" (defined) classes instead (the old behavior)
                // silently drops the worst classes from the denominator,
                // inflating the score of degenerate classifiers (e.g. one
                // that always predicts the majority class).
                let sum: f64 = counts
                    .iter()
                    .map(|&(tp, fp, _)| if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 })
                    .sum();
                Ok(if nc > 0 { sum / nc as f64 } else { 0.0 })
            }
            _ => Err(SmeltError::IncompatiblePrediction(
                "Precision requires classification prediction with truth".into(),
            )),
        }
    }
}

/// Macro-averaged recall (sensitivity).
pub struct Recall;

impl Measure for Recall {
    fn id(&self) -> &str {
        "classif.recall"
    }
    fn maximize(&self) -> bool {
        true
    }

    fn score(&self, prediction: &Prediction) -> Result<f64> {
        match prediction {
            Prediction::Classification {
                predicted,
                truth: Some(truth),
                ..
            } => {
                let nc = n_classes(predicted, truth);
                let counts = class_counts(predicted, truth, nc);
                // See Precision::score for why the average is over all `nc`
                // classes, not just the ones with a defined recall.
                let sum: f64 = counts
                    .iter()
                    .map(|&(tp, _, fn_)| if tp + fn_ > 0 { tp as f64 / (tp + fn_) as f64 } else { 0.0 })
                    .sum();
                Ok(if nc > 0 { sum / nc as f64 } else { 0.0 })
            }
            _ => Err(SmeltError::IncompatiblePrediction(
                "Recall requires classification prediction with truth".into(),
            )),
        }
    }
}

/// Macro-averaged F1 score (harmonic mean of precision and recall).
pub struct F1Score;

impl Measure for F1Score {
    fn id(&self) -> &str {
        "classif.f1"
    }
    fn maximize(&self) -> bool {
        true
    }

    fn score(&self, prediction: &Prediction) -> Result<f64> {
        match prediction {
            Prediction::Classification {
                predicted,
                truth: Some(truth),
                ..
            } => {
                let nc = n_classes(predicted, truth);
                let counts = class_counts(predicted, truth, nc);
                // See Precision::score for why the average is over all `nc`
                // classes: a class with undefined precision AND recall (never
                // predicted and never true -- impossible here since `nc` only
                // spans observed labels, but a class that's never predicted
                // contributes F1=0) still counts in the denominator.
                let sum: f64 = counts
                    .iter()
                    .map(|&(tp, fp, fn_)| {
                        let prec = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
                        let rec = if tp + fn_ > 0 { tp as f64 / (tp + fn_) as f64 } else { 0.0 };
                        if prec + rec > 0.0 { 2.0 * prec * rec / (prec + rec) } else { 0.0 }
                    })
                    .sum();
                Ok(if nc > 0 { sum / nc as f64 } else { 0.0 })
            }
            _ => Err(SmeltError::IncompatiblePrediction(
                "F1 requires classification prediction with truth".into(),
            )),
        }
    }
}

/// Logarithmic loss. Requires probabilities.
pub struct LogLoss;

impl Measure for LogLoss {
    fn id(&self) -> &str {
        "classif.logloss"
    }
    fn maximize(&self) -> bool {
        false
    }

    fn score(&self, prediction: &Prediction) -> Result<f64> {
        match prediction {
            Prediction::Classification {
                truth: Some(truth),
                probabilities: Some(probs),
                ..
            } => {
                let eps = 1e-15;
                let n = truth.len() as f64;
                let loss: f64 = truth
                    .iter()
                    .zip(probs)
                    .map(|(&t, p)| {
                        let prob = p[t].max(eps).min(1.0 - eps);
                        -prob.ln()
                    })
                    .sum::<f64>()
                    / n;
                Ok(loss)
            }
            Prediction::Classification {
                probabilities: None,
                ..
            } => Err(SmeltError::IncompatiblePrediction("LogLoss requires probabilities".into())),
            _ => Err(SmeltError::IncompatiblePrediction(
                "LogLoss requires classification prediction with truth and probabilities".into(),
            )),
        }
    }
}

/// Area Under the ROC Curve. Requires probabilities.
///
/// For binary classification, uses the probability of class 1.
/// For multiclass, computes macro-averaged one-vs-rest AUC.
pub struct AucRoc;

impl Measure for AucRoc {
    fn id(&self) -> &str {
        "classif.auc"
    }
    fn maximize(&self) -> bool {
        true
    }

    fn score(&self, prediction: &Prediction) -> Result<f64> {
        match prediction {
            Prediction::Classification {
                truth: Some(truth),
                probabilities: Some(probs),
                ..
            } => {
                let nc = *truth.iter().max().unwrap_or(&0) + 1;
                if nc == 2 {
                    // Binary: AUC for class 1
                    let scores: Vec<f64> = probs.iter().map(|p| p[1]).collect();
                    let labels: Vec<bool> = truth.iter().map(|&t| t == 1).collect();
                    auc_binary(&scores, &labels)
                } else {
                    // Macro OVR AUC
                    let mut sum = 0.0;
                    let mut valid = 0;
                    for c in 0..nc {
                        let scores: Vec<f64> = probs.iter().map(|p| p[c]).collect();
                        let labels: Vec<bool> = truth.iter().map(|&t| t == c).collect();
                        if labels.iter().any(|&l| l) && labels.iter().any(|&l| !l) {
                            sum += auc_binary(&scores, &labels)?;
                            valid += 1;
                        }
                    }
                    Ok(if valid > 0 { sum / valid as f64 } else { 0.5 })
                }
            }
            Prediction::Classification {
                probabilities: None,
                ..
            } => Err(SmeltError::IncompatiblePrediction("AUC-ROC requires probabilities".into())),
            _ => Err(SmeltError::IncompatiblePrediction(
                "AUC-ROC requires classification prediction with truth and probabilities".into(),
            )),
        }
    }
}

/// Compute AUC for a binary problem using the trapezoidal rule.
fn auc_binary(scores: &[f64], labels: &[bool]) -> Result<f64> {
    let n = scores.len();
    let mut indexed: Vec<(f64, bool)> = scores.iter().zip(labels).map(|(&s, &l)| (s, l)).collect();
    // Sort descending by score
    indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let n_pos = labels.iter().filter(|&&l| l).count() as f64;
    let n_neg = n as f64 - n_pos;

    if n_pos == 0.0 || n_neg == 0.0 {
        return Ok(0.5); // undefined, return 0.5
    }

    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut auc = 0.0;
    let mut prev_fp = 0.0;
    let mut prev_tp = 0.0;

    let mut i = 0;
    while i < n {
        // Find all tied scores
        let score = indexed[i].0;
        let mut j = i;
        while j < n && (indexed[j].0 - score).abs() < f64::EPSILON {
            if indexed[j].1 {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
            j += 1;
        }
        // Trapezoidal area
        auc += (fp - prev_fp) * (tp + prev_tp) / 2.0;
        prev_fp = fp;
        prev_tp = tp;
        i = j;
    }

    Ok(auc / (n_pos * n_neg))
}

/// `n_classes x n_classes` confusion matrix: `matrix[true_class][predicted_class]`.
fn confusion_matrix(predicted: &[usize], truth: &[usize], nc: usize) -> Vec<Vec<u64>> {
    let mut m = vec![vec![0u64; nc]; nc];
    for (&p, &t) in predicted.iter().zip(truth) {
        m[t][p] += 1;
    }
    m
}

/// Balanced accuracy: macro-average of per-class recall (sensitivity).
///
/// Unlike plain [`Accuracy`], a classifier that always predicts the
/// majority class scores 1/n_classes here instead of the majority
/// class's prevalence — the right metric for imbalanced datasets.
pub struct BalancedAccuracy;

impl Measure for BalancedAccuracy {
    fn id(&self) -> &str {
        "classif.balanced_accuracy"
    }
    fn maximize(&self) -> bool {
        true
    }

    fn score(&self, prediction: &Prediction) -> Result<f64> {
        match prediction {
            Prediction::Classification {
                predicted,
                truth: Some(truth),
                ..
            } => {
                let nc = n_classes(predicted, truth);
                let counts = class_counts(predicted, truth, nc);
                let mut sum = 0.0;
                let mut valid = 0;
                for &(tp, _, fn_) in &counts {
                    if tp + fn_ > 0 {
                        sum += tp as f64 / (tp + fn_) as f64;
                        valid += 1;
                    }
                }
                Ok(if valid > 0 { sum / valid as f64 } else { 0.0 })
            }
            _ => Err(SmeltError::IncompatiblePrediction(
                "BalancedAccuracy requires classification prediction with truth".into(),
            )),
        }
    }
}

/// Cohen's Kappa: agreement between predictions and truth, corrected for
/// the agreement expected by chance given the observed class marginals.
/// 1 = perfect agreement, 0 = no better than chance, negative = worse
/// than chance.
pub struct CohensKappa;

impl Measure for CohensKappa {
    fn id(&self) -> &str {
        "classif.kappa"
    }
    fn maximize(&self) -> bool {
        true
    }

    fn score(&self, prediction: &Prediction) -> Result<f64> {
        match prediction {
            Prediction::Classification {
                predicted,
                truth: Some(truth),
                ..
            } => {
                let nc = n_classes(predicted, truth);
                let cm = confusion_matrix(predicted, truth, nc);
                let n = predicted.len() as f64;
                let po: f64 = (0..nc).map(|i| cm[i][i] as f64).sum::<f64>() / n;
                let row_sum: Vec<f64> = (0..nc).map(|i| cm[i].iter().sum::<u64>() as f64).collect();
                let col_sum: Vec<f64> = (0..nc)
                    .map(|j| (0..nc).map(|i| cm[i][j]).sum::<u64>() as f64)
                    .collect();
                let pe: f64 = (0..nc).map(|i| row_sum[i] * col_sum[i]).sum::<f64>() / (n * n);
                Ok(if (1.0 - pe).abs() > f64::EPSILON {
                    (po - pe) / (1.0 - pe)
                } else {
                    // pe == 1 only when every sample falls in one class for both
                    // predicted and truth marginals, i.e. trivially perfect agreement.
                    1.0
                })
            }
            _ => Err(SmeltError::IncompatiblePrediction(
                "CohensKappa requires classification prediction with truth".into(),
            )),
        }
    }
}

/// Matthews Correlation Coefficient, generalized to multiclass (Gorodkin,
/// 2004). Ranges `[-1, 1]`: 1 = perfect prediction, 0 = no better than
/// random, -1 = total disagreement. Unlike accuracy-based measures it
/// accounts for all four confusion-matrix quadrants at once, so it stays
/// informative under class imbalance.
pub struct Mcc;

impl Measure for Mcc {
    fn id(&self) -> &str {
        "classif.mcc"
    }
    fn maximize(&self) -> bool {
        true
    }

    fn score(&self, prediction: &Prediction) -> Result<f64> {
        match prediction {
            Prediction::Classification {
                predicted,
                truth: Some(truth),
                ..
            } => {
                let nc = n_classes(predicted, truth);
                let cm = confusion_matrix(predicted, truth, nc); // cm[true][pred]
                let n = predicted.len() as f64;
                let c: f64 = (0..nc).map(|i| cm[i][i] as f64).sum();
                // t_k = true occurrences of class k (row sum); p_k = predicted occurrences (col sum).
                let t: Vec<f64> = (0..nc).map(|i| cm[i].iter().sum::<u64>() as f64).collect();
                let p: Vec<f64> = (0..nc)
                    .map(|j| (0..nc).map(|i| cm[i][j]).sum::<u64>() as f64)
                    .collect();
                let sum_tp: f64 = t.iter().zip(&p).map(|(ti, pi)| ti * pi).sum();
                let sum_t2: f64 = t.iter().map(|ti| ti * ti).sum();
                let sum_p2: f64 = p.iter().map(|pi| pi * pi).sum();
                let numerator = c * n - sum_tp;
                let denominator = ((n * n - sum_p2) * (n * n - sum_t2)).sqrt();
                Ok(if denominator > f64::EPSILON {
                    numerator / denominator
                } else {
                    // A degenerate marginal (predictions or truth collapse to one
                    // class) makes the correlation undefined; mlr3/sklearn return 0.
                    0.0
                })
            }
            _ => Err(SmeltError::IncompatiblePrediction(
                "MCC requires classification prediction with truth".into(),
            )),
        }
    }
}

/// Brier score: mean squared error between predicted class probabilities
/// and the one-hot true label, summed over classes. Lower is better (0 =
/// perfect, matching sklearn's multiclass Brier score convention).
pub struct Brier;

impl Measure for Brier {
    fn id(&self) -> &str {
        "classif.brier"
    }
    fn maximize(&self) -> bool {
        false
    }

    fn score(&self, prediction: &Prediction) -> Result<f64> {
        match prediction {
            Prediction::Classification {
                truth: Some(truth),
                probabilities: Some(probs),
                ..
            } => {
                let n = truth.len() as f64;
                let sum: f64 = truth
                    .iter()
                    .zip(probs)
                    .map(|(&t, p)| {
                        p.iter()
                            .enumerate()
                            .map(|(c, &pc)| {
                                let indicator = if c == t { 1.0 } else { 0.0 };
                                (pc - indicator).powi(2)
                            })
                            .sum::<f64>()
                    })
                    .sum::<f64>()
                    / n;
                Ok(sum)
            }
            Prediction::Classification {
                probabilities: None,
                ..
            } => Err(SmeltError::IncompatiblePrediction("Brier requires probabilities".into())),
            _ => Err(SmeltError::IncompatiblePrediction(
                "Brier requires classification prediction with truth and probabilities".into(),
            )),
        }
    }
}

// ── Regression metrics ──────────────────────────────────────────────

/// Coefficient of determination (R²).
pub struct RSquared;

impl Measure for RSquared {
    fn id(&self) -> &str {
        "regr.rsq"
    }
    fn maximize(&self) -> bool {
        true
    }

    fn score(&self, prediction: &Prediction) -> Result<f64> {
        match prediction {
            Prediction::Regression {
                predicted,
                truth: Some(truth),
            } => {
                let mean = truth.iter().sum::<f64>() / truth.len() as f64;
                let ss_res: f64 = predicted
                    .iter()
                    .zip(truth)
                    .map(|(p, t)| (t - p).powi(2))
                    .sum();
                let ss_tot: f64 = truth.iter().map(|t| (t - mean).powi(2)).sum();
                Ok(if ss_tot > 0.0 {
                    1.0 - ss_res / ss_tot
                } else {
                    0.0
                })
            }
            _ => Err(SmeltError::IncompatiblePrediction(
                "R² requires regression prediction with truth".into(),
            )),
        }
    }
}

/// Mean Absolute Percentage Error.
pub struct Mape;

impl Measure for Mape {
    fn id(&self) -> &str {
        "regr.mape"
    }
    fn maximize(&self) -> bool {
        false
    }

    fn score(&self, prediction: &Prediction) -> Result<f64> {
        match prediction {
            Prediction::Regression {
                predicted,
                truth: Some(truth),
            } => {
                let n = truth.len() as f64;
                let mape: f64 = predicted
                    .iter()
                    .zip(truth)
                    .map(|(p, t)| {
                        if t.abs() > f64::EPSILON {
                            ((p - t) / t).abs()
                        } else {
                            0.0 // skip zero-valued actuals
                        }
                    })
                    .sum::<f64>()
                    / n;
                Ok(mape)
            }
            _ => Err(SmeltError::IncompatiblePrediction(
                "MAPE requires regression prediction with truth".into(),
            )),
        }
    }
}

/// Precision in Estimation of Heterogeneous Effects (Hill 2011; the
/// standard evaluation metric for CATE/treatment-effect estimators, see
/// e.g. Shalit, Johansson & Sontag 2017). `sqrt(mean squared error)`
/// between estimated and *true* `tau(x)`. Only computable when ground
/// truth is known -- i.e. synthetic benchmarks, never real observational
/// or experimental data (which never reveals both potential outcomes for
/// the same unit).
pub struct Pehe;

impl Measure for Pehe {
    fn id(&self) -> &str {
        "causal.pehe"
    }
    fn maximize(&self) -> bool {
        false
    }

    fn score(&self, prediction: &Prediction) -> Result<f64> {
        match prediction {
            Prediction::CausalEffect {
                estimated,
                true_effect: Some(truth),
            } => {
                let mse: f64 = estimated
                    .iter()
                    .zip(truth)
                    .map(|(e, t)| (e - t).powi(2))
                    .sum::<f64>()
                    / estimated.len() as f64;
                Ok(mse.sqrt())
            }
            _ => Err(SmeltError::IncompatiblePrediction(
                "PEHE requires a CausalEffect prediction with known ground truth".into(),
            )),
        }
    }
}

/// Absolute bias of the estimated ATE (mean of per-unit CATE estimates)
/// against the known true ATE. Companion to [`Pehe`] -- PEHE measures
/// per-unit precision, this measures aggregate bias; a method can have low
/// ATE bias while still having poor per-unit PEHE if errors cancel out.
pub struct AteBias;

impl Measure for AteBias {
    fn id(&self) -> &str {
        "causal.ate_bias"
    }
    fn maximize(&self) -> bool {
        false
    }

    fn score(&self, prediction: &Prediction) -> Result<f64> {
        match prediction {
            Prediction::CausalEffect {
                estimated,
                true_effect: Some(truth),
            } => {
                let est_ate = estimated.iter().sum::<f64>() / estimated.len() as f64;
                let true_ate = truth.iter().sum::<f64>() / truth.len() as f64;
                Ok((est_ate - true_ate).abs())
            }
            _ => Err(SmeltError::IncompatiblePrediction(
                "AteBias requires a CausalEffect prediction with known ground truth".into(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    //! Golden tests against independently-computed reference values (mostly
    //! scikit-learn 1.8.0 / `sklearn.metrics`, run in the project's
    //! `smelt-py/.venv`; noted otherwise where conventions differ). Before
    //! this module, `measure/mod.rs` had zero tests — the 2026-07-04 engine
    //! audit's own principle ("every CRITICAL statistical bug would have
    //! been caught by a golden test") applies here just as much as to the
    //! modules that already had fixes.
    use super::*;

    /// 3-class fixture (n=30), scores from `sklearn.metrics` with default
    /// settings (`average='macro'` for precision/recall/f1).
    fn multiclass_fixture() -> (Vec<usize>, Vec<usize>) {
        let truth = vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 1, 2, 0, 1,
            2,
        ];
        let predicted = vec![
            0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 1, 1, 1, 1, 0, 1, 1, 1, 2, 2, 1, 2, 2, 2, 0, 1, 2, 1, 1,
            2,
        ];
        (predicted, truth)
    }

    #[test]
    fn golden_classification_metrics_match_sklearn_reference() {
        let (predicted, truth) = multiclass_fixture();
        let pred = Prediction::classification_with_truth(predicted, truth);

        let cases: &[(&dyn Measure, f64)] = &[
            (&Accuracy, 0.8333333333333334),
            (&Precision, 0.8416666666666667),
            (&Recall, 0.8416666666666667),
            (&F1Score, 0.8371212121212123),
            (&BalancedAccuracy, 0.8416666666666667),
            (&Mcc, 0.7533783783783784),
            (&CohensKappa, 0.7483221476510067),
        ];
        for (measure, expected) in cases {
            let got = measure.score(&pred).unwrap();
            assert!(
                (got - expected).abs() < 1e-9,
                "{}: expected {expected}, got {got}",
                measure.id()
            );
        }
    }

    /// Regression test for [MEDIUM finding N10 in the 2026-07-04 audit]:
    /// macro precision/recall/F1 used to average only over classes with a
    /// *defined* score (`tp+fp>0` etc.), silently dropping classes a
    /// degenerate classifier never predicts from the denominator. A binary
    /// classifier that always predicts class 0 on perfectly balanced data
    /// used to score macro-precision 0.5 here vs sklearn's 0.25
    /// (`zero_division=0`) -- this asserts the sklearn-matching values.
    #[test]
    fn macro_precision_recall_f1_use_full_class_denominator_matching_sklearn_zero_division() {
        let truth: Vec<usize> = (0..20).map(|_| 0).chain((0..20).map(|_| 1)).collect();
        let predicted = vec![0usize; 40]; // always predicts class 0
        let pred = Prediction::classification_with_truth(predicted, truth);

        let prec = Precision.score(&pred).unwrap();
        let rec = Recall.score(&pred).unwrap();
        let f1 = F1Score.score(&pred).unwrap();
        assert!((prec - 0.25).abs() < 1e-9, "precision: expected 0.25 (sklearn zero_division=0), got {prec}");
        assert!((rec - 0.5).abs() < 1e-9, "recall: expected 0.5, got {rec}");
        assert!(
            (f1 - 0.3333333333333333).abs() < 1e-9,
            "f1: expected 0.3333... (sklearn zero_division=0), got {f1}"
        );
    }

    #[test]
    fn golden_regression_metrics_match_sklearn_reference() {
        let truth = vec![3.0, 5.0, 2.5, 7.0, 4.0, 6.5];
        let predicted = vec![2.8, 5.2, 2.0, 7.5, 4.5, 6.0];
        let pred = Prediction::regression_with_truth(predicted, truth);

        let rmse = Rmse.score(&pred).unwrap();
        let mae = Mae.score(&pred).unwrap();
        let r2 = RSquared.score(&pred).unwrap();
        let mape = Mape.score(&pred).unwrap();
        assert!((rmse - 0.42426406871192857).abs() < 1e-9, "rmse: got {rmse}");
        assert!((mae - 0.4000000000000001).abs() < 1e-9, "mae: got {mae}");
        assert!((r2 - 0.9358415841584158).abs() < 1e-9, "r2: got {r2}");
        assert!((mape - 0.09666971916971918).abs() < 1e-9, "mape: got {mape}");
    }

    /// AUC-ROC and LogLoss golden values match `sklearn.metrics.roc_auc_score`
    /// / `log_loss` exactly (both are standard, convention-free formulas).
    /// Brier does NOT: this crate sums the squared error over *all* classes
    /// (the historical multi-category Brier score definition), while
    /// `sklearn.metrics.brier_score_loss` only uses the positive-class
    /// probability for binary problems -- exactly half this crate's value
    /// for a 2-class prediction. The reference below is computed
    /// independently (plain numpy, not sklearn) to match this crate's own
    /// documented convention, not sklearn's binary-specific one.
    #[test]
    fn golden_auc_logloss_brier_probabilistic_reference() {
        let truth = vec![0, 0, 0, 1, 1, 1, 0, 1, 1, 0];
        let p1 = [0.1, 0.3, 0.6, 0.8, 0.4, 0.9, 0.55, 0.7, 0.2, 0.35];
        let predicted: Vec<usize> = p1.iter().map(|&p| if p >= 0.5 { 1 } else { 0 }).collect();
        let probabilities: Vec<Vec<f64>> = p1.iter().map(|&p| vec![1.0 - p, p]).collect();
        let pred = Prediction::Classification {
            predicted,
            truth: Some(truth),
            probabilities: Some(probabilities),
        };

        let auc = AucRoc.score(&pred).unwrap();
        let logloss = LogLoss.score(&pred).unwrap();
        let brier = Brier.score(&pred).unwrap();
        assert!((auc - 0.76).abs() < 1e-9, "auc: expected 0.76 (sklearn roc_auc_score), got {auc}");
        assert!(
            (logloss - 0.5818524458999963).abs() < 1e-9,
            "logloss: expected 0.5818524458999963 (sklearn log_loss), got {logloss}"
        );
        assert!(
            (brier - 0.4050000000000001).abs() < 1e-9,
            "brier: expected 0.405 (this crate's 2-class-sum convention, i.e. \
             2x sklearn's binary brier_score_loss of 0.2025), got {brier}"
        );
    }
}
