//! Predictions: output of trained models.

use serde::{Deserialize, Serialize};

/// Holds predictions and (optionally) ground truth for evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Prediction {
    /// Classification: predicted class labels + optional probabilities.
    Classification {
        /// Predicted class labels (encoded as indices).
        predicted: Vec<usize>,
        /// Ground-truth class labels, when available (e.g. on a test set).
        truth: Option<Vec<usize>>,
        /// Per-class predicted probabilities, one row per sample, when the learner supports it.
        probabilities: Option<Vec<Vec<f64>>>,
    },
    /// Regression: predicted continuous values.
    Regression {
        /// Predicted continuous target values.
        predicted: Vec<f64>,
        /// Ground-truth target values, when available (e.g. on a test set).
        truth: Option<Vec<f64>>,
    },
    /// Estimated per-unit treatment effect (CATE), from a causal
    /// meta-learner (`TLearner`/`SLearner`/`XLearner`/`RLearner`/
    /// `DrLearner`) or `CausalForest`. `true_effect` is only ever `Some`
    /// for synthetic benchmarks with a known ground-truth `tau(x)` --
    /// real data never has this, so it's what `Pehe`/`AteBias` require.
    CausalEffect {
        /// Estimated per-unit treatment effect (CATE).
        estimated: Vec<f64>,
        /// Known ground-truth `tau(x)`, only ever `Some` for synthetic benchmarks.
        true_effect: Option<Vec<f64>>,
    },
}

impl Prediction {
    /// Builds a classification prediction with no ground truth or probabilities attached.
    pub fn classification(predicted: Vec<usize>) -> Self {
        Self::Classification {
            predicted,
            truth: None,
            probabilities: None,
        }
    }

    /// Builds a classification prediction paired with its ground-truth labels.
    pub fn classification_with_truth(predicted: Vec<usize>, truth: Vec<usize>) -> Self {
        Self::Classification {
            predicted,
            truth: Some(truth),
            probabilities: None,
        }
    }

    /// Builds a regression prediction with no ground truth attached.
    pub fn regression(predicted: Vec<f64>) -> Self {
        Self::Regression {
            predicted,
            truth: None,
        }
    }

    /// Builds a regression prediction paired with its ground-truth values.
    pub fn regression_with_truth(predicted: Vec<f64>, truth: Vec<f64>) -> Self {
        Self::Regression {
            predicted,
            truth: Some(truth),
        }
    }

    /// Attaches ground-truth labels to a `Classification` prediction; no-op on other variants.
    pub fn with_truth_classif(self, truth: Vec<usize>) -> Self {
        match self {
            Self::Classification {
                predicted,
                probabilities,
                ..
            } => Self::Classification {
                predicted,
                truth: Some(truth),
                probabilities,
            },
            other => other,
        }
    }

    /// Attaches ground-truth values to a `Regression` prediction; no-op on other variants.
    pub fn with_truth_regress(self, truth: Vec<f64>) -> Self {
        match self {
            Self::Regression { predicted, .. } => Self::Regression {
                predicted,
                truth: Some(truth),
            },
            other => other,
        }
    }

    /// Builds a causal-effect prediction with no known ground-truth `tau(x)`.
    pub fn causal_effect(estimated: Vec<f64>) -> Self {
        Self::CausalEffect {
            estimated,
            true_effect: None,
        }
    }

    /// Builds a causal-effect prediction paired with a known ground-truth `tau(x)`.
    pub fn causal_effect_with_truth(estimated: Vec<f64>, true_effect: Vec<f64>) -> Self {
        Self::CausalEffect {
            estimated,
            true_effect: Some(true_effect),
        }
    }

    /// Attaches a ground-truth `tau(x)` to a `CausalEffect` prediction; no-op on other variants.
    pub fn with_truth_causal(self, true_effect: Vec<f64>) -> Self {
        match self {
            Self::CausalEffect { estimated, .. } => Self::CausalEffect {
                estimated,
                true_effect: Some(true_effect),
            },
            other => other,
        }
    }

    /// Number of samples covered by this prediction.
    pub fn n_samples(&self) -> usize {
        match self {
            Self::Classification { predicted, .. } => predicted.len(),
            Self::Regression { predicted, .. } => predicted.len(),
            Self::CausalEffect { estimated, .. } => estimated.len(),
        }
    }
}
