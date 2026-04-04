//! Measures: evaluate prediction quality.
//!
//! Classification: Accuracy, Precision, Recall, F1Score, LogLoss, AucRoc.
//! Regression: Rmse, Mae, RSquared, Mape.

use crate::prediction::Prediction;
use crate::{Result, SmeltError};

/// Trait for evaluation metrics.
pub trait Measure {
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
            _ => Err(SmeltError::Other(
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
            _ => Err(SmeltError::Other(
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
            _ => Err(SmeltError::Other(
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
                let mut sum = 0.0;
                let mut valid = 0;
                for &(tp, fp, _) in &counts {
                    if tp + fp > 0 {
                        sum += tp as f64 / (tp + fp) as f64;
                        valid += 1;
                    }
                }
                Ok(if valid > 0 { sum / valid as f64 } else { 0.0 })
            }
            _ => Err(SmeltError::Other(
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
            _ => Err(SmeltError::Other(
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
                let mut sum = 0.0;
                let mut valid = 0;
                for &(tp, fp, fn_) in &counts {
                    let prec = if tp + fp > 0 {
                        tp as f64 / (tp + fp) as f64
                    } else {
                        0.0
                    };
                    let rec = if tp + fn_ > 0 {
                        tp as f64 / (tp + fn_) as f64
                    } else {
                        0.0
                    };
                    if prec + rec > 0.0 {
                        sum += 2.0 * prec * rec / (prec + rec);
                        valid += 1;
                    }
                }
                Ok(if valid > 0 { sum / valid as f64 } else { 0.0 })
            }
            _ => Err(SmeltError::Other(
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
            } => Err(SmeltError::Other("LogLoss requires probabilities".into())),
            _ => Err(SmeltError::Other(
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
            } => Err(SmeltError::Other("AUC-ROC requires probabilities".into())),
            _ => Err(SmeltError::Other(
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
            _ => Err(SmeltError::Other(
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
            _ => Err(SmeltError::Other(
                "MAPE requires regression prediction with truth".into(),
            )),
        }
    }
}
