//! Predictions: output of trained models.

use serde::{Deserialize, Serialize};

/// Holds predictions and (optionally) ground truth for evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Prediction {
    /// Classification: predicted class labels + optional probabilities.
    Classification {
        predicted: Vec<usize>,
        truth: Option<Vec<usize>>,
        probabilities: Option<Vec<Vec<f64>>>,
    },
    /// Regression: predicted continuous values.
    Regression {
        predicted: Vec<f64>,
        truth: Option<Vec<f64>>,
    },
    /// Estimated per-unit treatment effect (CATE), from a causal
    /// meta-learner (`TLearner`/`SLearner`/`XLearner`/`RLearner`/
    /// `DrLearner`) or `CausalForest`. `true_effect` is only ever `Some`
    /// for synthetic benchmarks with a known ground-truth `tau(x)` --
    /// real data never has this, so it's what `Pehe`/`AteBias` require.
    CausalEffect {
        estimated: Vec<f64>,
        true_effect: Option<Vec<f64>>,
    },
}

impl Prediction {
    pub fn classification(predicted: Vec<usize>) -> Self {
        Self::Classification {
            predicted,
            truth: None,
            probabilities: None,
        }
    }

    pub fn classification_with_truth(predicted: Vec<usize>, truth: Vec<usize>) -> Self {
        Self::Classification {
            predicted,
            truth: Some(truth),
            probabilities: None,
        }
    }

    pub fn regression(predicted: Vec<f64>) -> Self {
        Self::Regression {
            predicted,
            truth: None,
        }
    }

    pub fn regression_with_truth(predicted: Vec<f64>, truth: Vec<f64>) -> Self {
        Self::Regression {
            predicted,
            truth: Some(truth),
        }
    }

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

    pub fn with_truth_regress(self, truth: Vec<f64>) -> Self {
        match self {
            Self::Regression { predicted, .. } => Self::Regression {
                predicted,
                truth: Some(truth),
            },
            other => other,
        }
    }

    pub fn causal_effect(estimated: Vec<f64>) -> Self {
        Self::CausalEffect {
            estimated,
            true_effect: None,
        }
    }

    pub fn causal_effect_with_truth(estimated: Vec<f64>, true_effect: Vec<f64>) -> Self {
        Self::CausalEffect {
            estimated,
            true_effect: Some(true_effect),
        }
    }

    pub fn with_truth_causal(self, true_effect: Vec<f64>) -> Self {
        match self {
            Self::CausalEffect { estimated, .. } => Self::CausalEffect {
                estimated,
                true_effect: Some(true_effect),
            },
            other => other,
        }
    }

    pub fn n_samples(&self) -> usize {
        match self {
            Self::Classification { predicted, .. } => predicted.len(),
            Self::Regression { predicted, .. } => predicted.len(),
            Self::CausalEffect { estimated, .. } => estimated.len(),
        }
    }
}
