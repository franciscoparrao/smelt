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
use crate::task::{ClassificationTask, Task};
use crate::{Result, SmeltError};
use ndarray::{Array2, Axis};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

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
    dsel_fraction: f64,
    seed: u64,
}

impl DynamicEnsemble {
    /// Creates a DES ensemble from the given base-learner factories,
    /// defaulting to 7 neighbors for the KNORA-E competence check and a
    /// 30% held-out Dynamic Selection set (DSEL).
    pub fn new(factories: Vec<Box<dyn Fn() -> Box<dyn Learner> + Send + Sync>>) -> Self {
        Self {
            base_factories: factories,
            k_neighbors: 7,
            dsel_fraction: 0.3,
            seed: 42,
        }
    }

    /// Sets the number of nearest neighbors (k) used to assess each base
    /// classifier's local competence (KNORA-E).
    pub fn with_k_neighbors(mut self, k: usize) -> Self {
        self.k_neighbors = k;
        self
    }

    /// Sets the fraction of training data held out as the Dynamic Selection
    /// set (DSEL): base models are trained ONLY on the rest, and competence
    /// (which models correctly classify a query's k nearest neighbors) is
    /// evaluated ONLY on the DSEL. Evaluating competence on the same data a
    /// model was trained on -- the previous behavior -- makes an overfit
    /// base model look competent everywhere. Default `0.3`.
    pub fn with_dsel_fraction(mut self, fraction: f64) -> Self {
        self.dsel_fraction = fraction;
        self
    }

    /// Sets the RNG seed controlling the train/DSEL split.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Size of the held-out DSEL for `n_samples` total, given
    /// `dsel_fraction`: rounded, then clamped to `[1, n_samples - 1]` so
    /// both the training set and the DSEL always get at least one sample.
    /// Broken out as its own method (rather than inlined in
    /// `train_classif`) so it's directly unit-testable.
    fn dsel_split_size(&self, n_samples: usize) -> usize {
        (((n_samples as f64) * self.dsel_fraction).round() as usize).clamp(1, n_samples - 1)
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
        let n_samples = task.n_samples();

        if n_samples < 2 {
            return Err(SmeltError::InvalidParameter(
                "DynamicEnsemble requires at least 2 samples to split into a training set \
                 and a held-out DSEL"
                    .into(),
            ));
        }

        // Split off a held-out Dynamic Selection set (DSEL): base models are
        // trained ONLY on `train_idx`, and their competence -- along with
        // the k-NN neighborhood lookup at predict time -- is evaluated ONLY
        // on `dsel_idx`. The final deployed models are the same train-only
        // models (not retrained on the full data): retraining afterward
        // would make the competence estimates stale relative to what's
        // actually deployed.
        let mut indices: Vec<usize> = (0..n_samples).collect();
        let mut rng = StdRng::seed_from_u64(self.seed);
        indices.shuffle(&mut rng);
        let dsel_size = self.dsel_split_size(n_samples);
        let (dsel_idx, train_idx) = indices.split_at(dsel_size);

        let train_features = features.select(Axis(0), train_idx);
        let train_target: Vec<usize> = train_idx.iter().map(|&i| target[i]).collect();
        let train_task = ClassificationTask::new(task.id(), train_features, train_target)?
            .with_feature_names(task.feature_names().to_vec())?
            .with_feature_types(task.feature_types().to_vec())?
            .with_class_names(task.class_names().to_vec());

        let mut models: Vec<Box<dyn TrainedModel>> = Vec::new();
        for factory in &self.base_factories {
            let mut learner = factory();
            models.push(learner.train_classif(&train_task)?);
        }

        let dsel_features = features.select(Axis(0), dsel_idx);
        let dsel_target: Vec<usize> = dsel_idx.iter().map(|&i| target[i]).collect();

        // Held-out predictions used for the KNORA-E competence check.
        let mut val_predictions: Vec<Vec<usize>> = Vec::new();
        for model in &models {
            let pred = model.predict(&dsel_features)?;
            if let Prediction::Classification { predicted, .. } = pred {
                val_predictions.push(predicted);
            }
        }

        Ok(Box::new(TrainedDES {
            models,
            val_features: dsel_features,
            val_targets: dsel_target,
            val_predictions,
            n_classes,
            k: self.k_neighbors.min(dsel_idx.len()),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::tree::decision_tree::DecisionTree;
    use ndarray::array;

    fn one_learner() -> Vec<Box<dyn Fn() -> Box<dyn Learner> + Send + Sync>> {
        vec![Box::new(|| Box::new(DecisionTree::default()) as Box<dyn Learner>)]
    }

    /// Regression test for HIGH-13: competence used to be evaluated on the
    /// exact data the base models were trained on, which is equivalent to
    /// `dsel_fraction` effectively being 0 (no held-out data at all). This
    /// checks the split mechanism directly: a proper DSEL always gets at
    /// least 1 sample and the training set always gets at least 1 sample,
    /// scaling with `dsel_fraction` in between.
    #[test]
    fn dsel_split_size_respects_fraction_and_stays_within_bounds() {
        let des = DynamicEnsemble::new(one_learner());
        assert_eq!(des.dsel_split_size(10), 3, "default 0.3 fraction on n=10 should give dsel=3");

        let tiny_dsel = DynamicEnsemble::new(one_learner()).with_dsel_fraction(0.01);
        assert_eq!(tiny_dsel.dsel_split_size(10), 1, "dsel must never be empty");

        let huge_dsel = DynamicEnsemble::new(one_learner()).with_dsel_fraction(0.99);
        assert_eq!(huge_dsel.dsel_split_size(10), 9, "training set must never be empty (n-1 cap)");
    }

    #[test]
    fn train_classif_rejects_fewer_than_two_samples() {
        let features = array![[0.0, 0.0]];
        let target = vec![0];
        let task = ClassificationTask::new("des", features, target).unwrap();
        let mut des = DynamicEnsemble::new(one_learner());
        assert!(des.train_classif(&task).is_err());
    }

    /// End-to-end: base models are now trained on a strict subset of the
    /// task (train_idx only), never on the full data -- this exercises that
    /// path (previously `train_classif(task)` on the FULL task) still
    /// produces a working, reasonably accurate model on clearly separable
    /// data.
    #[test]
    fn train_classif_with_dsel_holdout_still_fits_separable_data() {
        let features = array![
            [0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [0.0, 0.2], [0.15, 0.05], [0.05, 0.18],
            [1.0, 1.0], [1.1, 0.9], [0.9, 1.1], [1.0, 0.9], [1.05, 0.95], [0.95, 1.08],
        ];
        let target = vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
        let task = ClassificationTask::new("des", features.clone(), target.clone()).unwrap();

        let mut des = DynamicEnsemble::new(vec![
            Box::new(|| Box::new(DecisionTree::default()) as Box<dyn Learner>),
        ])
        .with_dsel_fraction(0.3)
        .with_seed(3);
        let model = des.train_classif(&task).unwrap();
        let pred = model.predict(&features).unwrap();
        let Prediction::Classification { predicted, .. } = pred else {
            panic!("expected classification");
        };
        let correct = predicted.iter().zip(&target).filter(|(p, t)| *p == *t).count();
        assert!(
            correct as f64 / target.len() as f64 >= 0.8,
            "DES with a held-out DSEL should still fit clearly separable data well"
        );
    }
}
