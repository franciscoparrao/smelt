//! Dynamic Ensemble Selection (DES): selects the best classifier(s) per instance.
//!
//! Unlike static stacking, DES evaluates which base classifiers are competent
//! for each specific test instance based on its local neighborhood.
//!
//! Implements KNORA-E (K Nearest Output Profiles — Eliminate):
//! selects classifiers that correctly classify ALL k-nearest neighbors.
//!
//! Reference: Ko, A. et al. (2008). From dynamic classifier selection to
//! dynamic ensemble selection. Pattern Recognition.

use crate::learner::{Learner, TrainedModel};
use crate::prediction::Prediction;
use crate::Result;
use crate::task::{ClassificationTask, Task};
use ndarray::Array2;

/// Dynamic Ensemble Selection (KNORA-E).
///
/// Trains multiple base learners, then for each test instance selects
/// only the classifiers that correctly predict its k-nearest validation neighbors.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use smelt_ml::learner::DynamicEnsemble;
/// use ndarray::array;
///
/// let features = array![
///     [0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [0.0, 0.2],
///     [1.0, 1.0], [1.1, 0.9], [0.9, 1.1], [1.0, 0.9]
/// ];
/// let target = vec![0, 0, 0, 0, 1, 1, 1, 1];
/// let task = ClassificationTask::new("des", features, target).unwrap();
///
/// let mut des = DynamicEnsemble::new(vec![
///     Box::new(|| Box::new(DecisionTree::default()) as Box<dyn Learner>),
///     Box::new(|| Box::new(KNearestNeighbors::new(3)) as Box<dyn Learner>),
///     Box::new(|| Box::new(GaussianNB::new()) as Box<dyn Learner>),
/// ]);
/// let model = des.train_classif(&task).unwrap();
/// ```
pub struct DynamicEnsemble {
    base_factories: Vec<Box<dyn Fn() -> Box<dyn Learner> + Send + Sync>>,
    k_neighbors: usize,
}

impl DynamicEnsemble {
    /// Creates a DES ensemble from the given base-learner factories,
    /// defaulting to 7 neighbors for the KNORA-E competence check.
    pub fn new(factories: Vec<Box<dyn Fn() -> Box<dyn Learner> + Send + Sync>>) -> Self {
        Self {
            base_factories: factories,
            k_neighbors: 7,
        }
    }

    /// Sets the number of nearest neighbors (k) used to assess each base
    /// classifier's local competence (KNORA-E).
    pub fn with_k_neighbors(mut self, k: usize) -> Self {
        self.k_neighbors = k;
        self
    }
}

struct TrainedDES {
    models: Vec<Box<dyn TrainedModel>>,
    val_features: Array2<f64>,
    val_targets: Vec<usize>,
    val_predictions: Vec<Vec<usize>>, // [model][sample] = predicted class
    n_classes: usize,
    k: usize,
}

fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

impl TrainedModel for TrainedDES {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        let mut predicted = Vec::with_capacity(features.nrows());
        let mut probabilities = Vec::with_capacity(features.nrows());

        for row in features.rows() {
            let row_vec: Vec<f64> = row.to_vec();

            // Find k nearest neighbors in validation set
            let mut dists: Vec<(usize, f64)> = (0..self.val_features.nrows())
                .map(|j| {
                    (
                        j,
                        euclidean_dist(&row_vec, &self.val_features.row(j).to_vec()),
                    )
                })
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let neighbors: Vec<usize> = dists.iter().take(self.k).map(|(j, _)| *j).collect();

            // KNORA-E: select models that correctly classify ALL neighbors
            let mut competent: Vec<usize> = Vec::new();
            for (m, preds) in self.val_predictions.iter().enumerate() {
                let all_correct = neighbors.iter().all(|&n| preds[n] == self.val_targets[n]);
                if all_correct {
                    competent.push(m);
                }
            }

            // Fallback: if no model is fully competent, use all models
            if competent.is_empty() {
                competent = (0..self.models.len()).collect();
            }

            // Aggregate predictions from competent models
            let mut votes = vec![0usize; self.n_classes];
            let single = Array2::from_shape_vec((1, features.ncols()), row_vec).unwrap();
            for &m in &competent {
                if let Ok(Prediction::Classification { predicted: p, .. }) =
                    &self.models[m].predict(&single)
                    && p[0] < votes.len()
                {
                    votes[p[0]] += 1;
                }
            }

            let pred_class = votes
                .iter()
                .enumerate()
                .max_by_key(|&(_, &v)| v)
                .map(|(i, _)| i)
                .unwrap_or(0);
            let total: f64 = votes.iter().sum::<usize>() as f64;
            let probs: Vec<f64> = votes
                .iter()
                .map(|&v| {
                    if total > 0.0 {
                        v as f64 / total
                    } else {
                        1.0 / self.n_classes as f64
                    }
                })
                .collect();

            predicted.push(pred_class);
            probabilities.push(probs);
        }

        Ok(Prediction::Classification {
            predicted,
            truth: None,
            probabilities: Some(probabilities),
        })
    }
}

impl Learner for DynamicEnsemble {
    fn id(&self) -> &str {
        "dynamic_ensemble"
    }

    fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        let features = task.features();
        let target = task.target();
        let n_classes = task.n_classes();

        // Train all base models on full data
        let mut models: Vec<Box<dyn TrainedModel>> = Vec::new();
        for factory in &self.base_factories {
            let mut learner = factory();
            models.push(learner.train_classif(task)?);
        }

        // Get predictions of each model on training data (used as validation)
        let mut val_predictions: Vec<Vec<usize>> = Vec::new();
        for model in &models {
            let pred = model.predict(features)?;
            if let Prediction::Classification { predicted, .. } = pred {
                val_predictions.push(predicted);
            }
        }

        Ok(Box::new(TrainedDES {
            models,
            val_features: features.clone(),
            val_targets: target.to_vec(),
            val_predictions,
            n_classes,
            k: self.k_neighbors.min(task.n_samples()),
        }))
    }
}
