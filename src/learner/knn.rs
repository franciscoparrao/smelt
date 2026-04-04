//! K-Nearest Neighbors learner for classification and regression.
//!
//! Uses Euclidean distance. Classification by majority vote, regression by mean.

use crate::Result;
use crate::learner::{Learner, TrainedModel};
use crate::prediction::Prediction;
use crate::task::{ClassificationTask, RegressionTask, Task};
use ndarray::{Array2, ArrayView1};

/// K-Nearest Neighbors learner.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use smelt_ml::learner::KNearestNeighbors;
/// use ndarray::array;
///
/// let features = array![[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]];
/// let target = vec![0, 1, 0, 1];
/// let task = ClassificationTask::new("knn_demo", features, target).unwrap();
///
/// let mut knn = KNearestNeighbors::new(3);
/// let model = knn.train_classif(&task).unwrap();
/// ```
pub struct KNearestNeighbors {
    k: usize,
}

impl Default for KNearestNeighbors {
    fn default() -> Self {
        Self { k: 5 }
    }
}

impl KNearestNeighbors {
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

fn euclidean_distance(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn k_nearest(train: &Array2<f64>, sample: ArrayView1<f64>, k: usize) -> Vec<usize> {
    let mut dists: Vec<(usize, f64)> = train
        .rows()
        .into_iter()
        .enumerate()
        .map(|(i, row)| (i, euclidean_distance(row, sample)))
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    dists.iter().take(k).map(|(i, _)| *i).collect()
}

// --- Trained models ---

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct TrainedKnnClassifier {
    pub(crate) features: Array2<f64>,
    pub(crate) target: Vec<usize>,
    pub(crate) n_classes: usize,
    pub(crate) k: usize,
}

impl TrainedModel for TrainedKnnClassifier {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        crate::validate::check_n_features(features, self.features.ncols())?;
        let k = self.k.min(self.features.nrows());
        let mut predicted = Vec::with_capacity(features.nrows());
        let mut probabilities = Vec::with_capacity(features.nrows());

        for row in features.rows() {
            let neighbors = k_nearest(&self.features, row, k);
            let mut counts = vec![0usize; self.n_classes];
            for &idx in &neighbors {
                counts[self.target[idx]] += 1;
            }
            let total = neighbors.len() as f64;
            let probs: Vec<f64> = counts.iter().map(|&c| c as f64 / total).collect();
            let pred_class = counts
                .iter()
                .enumerate()
                .max_by_key(|&(_, &c)| c)
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
}

#[derive(Serialize, Deserialize)]
pub struct TrainedKnnRegressor {
    pub(crate) features: Array2<f64>,
    pub(crate) target: Vec<f64>,
    pub(crate) k: usize,
}

impl TrainedModel for TrainedKnnRegressor {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        crate::validate::check_n_features(features, self.features.ncols())?;
        let k = self.k.min(self.features.nrows());
        let predicted: Vec<f64> = features
            .rows()
            .into_iter()
            .map(|row| {
                let neighbors = k_nearest(&self.features, row, k);
                neighbors.iter().map(|&i| self.target[i]).sum::<f64>() / neighbors.len() as f64
            })
            .collect();

        Ok(Prediction::regression(predicted))
    }
}

// --- Learner impl ---

impl Learner for KNearestNeighbors {
    fn id(&self) -> &str {
        "knn"
    }

    fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        Ok(Box::new(TrainedKnnClassifier {
            features: task.features().clone(),
            target: task.target().to_vec(),
            n_classes: task.n_classes(),
            k: self.k,
        }))
    }

    fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
        Ok(Box::new(TrainedKnnRegressor {
            features: task.features().clone(),
            target: task.target().to_vec(),
            k: self.k,
        }))
    }
}
