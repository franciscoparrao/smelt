//! Multi-label classification: predict multiple labels per instance.
//!
//! Implements Classifier Chains (Read et al. 2011) — trains a sequence
//! of binary classifiers where each uses previous predictions as features.

use crate::learner::{Learner, TrainedModel};
use crate::prediction::Prediction;
use crate::task::ClassificationTask;
use crate::{Result, SmeltError};
use ndarray::Array2;

/// Multi-label prediction result.
#[derive(Debug, Clone)]
pub struct MultiLabelPrediction {
    /// Binary predictions per label: `labels[sample][label]` = 0 or 1.
    pub labels: Vec<Vec<usize>>,
    /// Number of samples.
    pub n_samples: usize,
    /// Number of labels.
    pub n_labels: usize,
}

/// Classifier Chain for multi-label classification.
///
/// Trains L binary classifiers sequentially. Each classifier j receives
/// the original features augmented with predictions from classifiers 1..j-1.
/// This captures label dependencies.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use smelt_ml::multilabel::ClassifierChain;
/// use ndarray::array;
///
/// let features = array![
///     [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0],
///     [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0],
/// ];
/// // 3 binary labels per sample
/// let labels = vec![
///     vec![1, 0, 1], vec![0, 1, 0], vec![1, 1, 1], vec![0, 0, 0],
///     vec![1, 0, 1], vec![0, 1, 0], vec![1, 1, 1], vec![0, 0, 0],
/// ];
///
/// let cc = ClassifierChain::new(
///     || Box::new(DecisionTree::default()),
/// );
/// let model = cc.fit(&features, &labels).unwrap();
/// let pred = model.predict(&features).unwrap();
/// assert_eq!(pred.n_labels, 3);
/// ```
pub struct ClassifierChain {
    factory: Box<dyn Fn() -> Box<dyn Learner> + Send + Sync>,
}

impl ClassifierChain {
    pub fn new(factory: impl Fn() -> Box<dyn Learner> + Send + Sync + 'static) -> Self {
        Self {
            factory: Box::new(factory),
        }
    }

    /// Fit the classifier chain.
    ///
    /// - `features`: n_samples × n_features
    /// - `labels`: Vec of n_samples, each Vec of n_labels (0 or 1)
    pub fn fit(
        &self,
        features: &Array2<f64>,
        labels: &[Vec<usize>],
    ) -> Result<TrainedClassifierChain> {
        let n_samples = features.nrows();
        let n_features = features.ncols();
        let n_labels = labels[0].len();

        if labels.len() != n_samples {
            return Err(SmeltError::DimensionMismatch {
                expected: n_samples,
                got: labels.len(),
            });
        }

        let mut models: Vec<Box<dyn TrainedModel>> = Vec::with_capacity(n_labels);

        for j in 0..n_labels {
            // Build augmented features: original + previous label predictions
            let n_aug = n_features + j;
            let mut aug_features = Array2::zeros((n_samples, n_aug));

            // Copy original features
            for i in 0..n_samples {
                for f in 0..n_features {
                    aug_features[[i, f]] = features[[i, f]];
                }
                // Add previous labels as features
                for prev in 0..j {
                    aug_features[[i, n_features + prev]] = labels[i][prev] as f64;
                }
            }

            // Binary target for this label
            let target: Vec<usize> = labels.iter().map(|l| l[j]).collect();

            let task = ClassificationTask::new(format!("label_{j}"), aug_features, target)?;

            let mut learner = (self.factory)();
            let model = learner.train_classif(&task)?;
            models.push(model);
        }

        Ok(TrainedClassifierChain {
            models,
            n_features,
            n_labels,
        })
    }
}

/// Trained classifier chain that can predict multiple labels.
pub struct TrainedClassifierChain {
    models: Vec<Box<dyn TrainedModel>>,
    n_features: usize,
    n_labels: usize,
}

impl TrainedClassifierChain {
    /// Predict multiple labels for new data.
    pub fn predict(&self, features: &Array2<f64>) -> Result<MultiLabelPrediction> {
        let n_samples = features.nrows();
        let mut all_preds: Vec<Vec<usize>> = vec![vec![0; self.n_labels]; n_samples];

        for j in 0..self.n_labels {
            // Build augmented features with previous predictions
            let n_aug = self.n_features + j;
            let mut aug_features = Array2::zeros((n_samples, n_aug));

            for i in 0..n_samples {
                for f in 0..self.n_features {
                    aug_features[[i, f]] = features[[i, f]];
                }
                for prev in 0..j {
                    aug_features[[i, self.n_features + prev]] = all_preds[i][prev] as f64;
                }
            }

            let pred = self.models[j].predict(&aug_features)?;
            if let Prediction::Classification { predicted, .. } = &pred {
                for (i, &p) in predicted.iter().enumerate() {
                    all_preds[i][j] = p;
                }
            }
        }

        Ok(MultiLabelPrediction {
            labels: all_preds,
            n_samples,
            n_labels: self.n_labels,
        })
    }

    /// Evaluate with subset accuracy (exact match ratio).
    pub fn subset_accuracy(&self, predictions: &MultiLabelPrediction, truth: &[Vec<usize>]) -> f64 {
        let n = predictions.n_samples;
        let correct = predictions
            .labels
            .iter()
            .zip(truth)
            .filter(|(pred, true_labels)| *pred == *true_labels)
            .count();
        correct as f64 / n as f64
    }

    /// Evaluate with Hamming score (1 - hamming loss).
    pub fn hamming_score(&self, predictions: &MultiLabelPrediction, truth: &[Vec<usize>]) -> f64 {
        let n = predictions.n_samples;
        let l = predictions.n_labels;
        let total = n * l;
        let correct: usize = predictions
            .labels
            .iter()
            .zip(truth)
            .map(|(pred, true_labels)| pred.iter().zip(true_labels).filter(|(p, t)| p == t).count())
            .sum();
        correct as f64 / total as f64
    }
}
