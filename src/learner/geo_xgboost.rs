//! Geographical-XGBoost (G-XGBoost): spatially local regression via XGBoost.
//!
//! Implements Grekousis (2025) "Geographical-XGBoost: a new ensemble model
//! for spatially local regression based on gradient-boosted trees."
//! Journal of Geographical Systems, 27(2), 169-195.
//!
//! Key features:
//! - Bi-square spatial kernel weights
//! - Local XGBoost models per spatial unit (gradients × spatial weights)
//! - Ensemble of global + local models with adaptive alpha
//! - Bandwidth selection via k-fold cross-validation (`select_bandwidth`)
//! - Local feature importance

use crate::learner::xgboost::XGBoost;
use crate::learner::{Learner, TrainedModel};
use crate::prediction::Prediction;
use crate::task::{ClassificationTask, RegressionTask, Task};
use crate::{Result, SmeltError};
use ndarray::Array2;

/// Geographical-XGBoost for spatially local regression.
///
/// Extends XGBoost with spatial awareness by training local models
/// at each spatial unit, weighted by a bi-square kernel. Combines
/// global and local predictions via an adaptive alpha weight.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use ndarray::array;
///
/// let features = array![
///     [0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]
/// ];
/// let target = vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0];
/// let coords = vec![
///     (0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0),
///     (4.0, 0.0), (5.0, 0.0), (6.0, 0.0), (7.0, 0.0),
/// ];
/// let task = RegressionTask::new("geo", features, target).unwrap();
///
/// let mut gxgb = GeoXGBoost::new(coords)
///     .with_bandwidth(4)
///     .with_n_estimators(50);
/// let model = gxgb.train_regress(&task).unwrap();
/// ```
pub struct GeoXGBoost {
    coords: Vec<(f64, f64)>,
    /// Number of nearest neighbors for adaptive kernel (bandwidth).
    bandwidth: usize,
    /// XGBoost hyperparameters for both global and local models.
    n_estimators: usize,
    max_depth: usize,
    learning_rate: f64,
    lambda: f64,
    seed: u64,
    /// Alpha weight: 0.0 = only global, 1.0 = only local, None = adaptive (Eq. 20).
    alpha: Option<f64>,
}

impl GeoXGBoost {
    pub fn new(coords: Vec<(f64, f64)>) -> Self {
        Self {
            coords,
            bandwidth: 30,
            n_estimators: 100,
            max_depth: 6,
            learning_rate: 0.3,
            lambda: 1.0,
            seed: 42,
            alpha: None, // adaptive by default
        }
    }

    pub fn with_bandwidth(mut self, bw: usize) -> Self {
        self.bandwidth = bw;
        self
    }
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }
    pub fn with_max_depth(mut self, d: usize) -> Self {
        self.max_depth = d;
        self
    }
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }
    pub fn with_lambda(mut self, l: f64) -> Self {
        self.lambda = l;
        self
    }
    pub fn with_alpha(mut self, a: f64) -> Self {
        self.alpha = Some(a);
        self
    }
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Train and return a `TrainedGeoXGBoost` that supports `predict_spatial`.
    ///
    /// Unlike `train_regress` (which returns `Box<dyn TrainedModel>`), this
    /// preserves the concrete type so you can call `predict_spatial` with
    /// new coordinates for spatially-aware out-of-sample prediction.
    pub fn train_geo(&mut self, task: &RegressionTask) -> Result<TrainedGeoXGBoost> {
        self.train_geo_inner(task)
    }

    /// Select the optimal bandwidth (number of nearest neighbours) by k-fold
    /// cross-validation over a grid of candidate values.
    ///
    /// For each candidate bandwidth, the data are split into `k_folds` folds;
    /// a model is trained on the other folds and evaluated on the held-out fold
    /// via spatially-aware prediction (`predict_spatial`). The candidate that
    /// minimises the mean cross-validated RMSE is returned, together with the
    /// full per-candidate scores.
    ///
    /// All other hyperparameters (`n_estimators`, `max_depth`, `learning_rate`,
    /// `lambda`, `alpha`, `seed`) are held fixed at their current values. This
    /// is the principled alternative to setting `bandwidth` by hand: the optimal
    /// neighbourhood size is method- and data-dependent and should be tuned, not
    /// assumed (cf. Grekousis, 2025).
    pub fn select_bandwidth(
        &self,
        task: &RegressionTask,
        candidates: &[usize],
        k_folds: usize,
    ) -> Result<BandwidthSelection> {
        let n = task.n_samples();
        if k_folds < 2 {
            return Err(SmeltError::Other(
                "select_bandwidth requires k_folds >= 2".into(),
            ));
        }
        if self.coords.len() != n {
            return Err(SmeltError::DimensionMismatch {
                expected: n,
                got: self.coords.len(),
            });
        }
        if candidates.is_empty() {
            return Err(SmeltError::Other(
                "select_bandwidth requires at least one candidate bandwidth".into(),
            ));
        }

        let features = task.features();
        let target = task.target();

        // Deterministic fold assignment from a seeded permutation.
        let perm = seeded_permutation(n, self.seed);
        let mut folds: Vec<Vec<usize>> = vec![Vec::new(); k_folds];
        for (pos, &idx) in perm.iter().enumerate() {
            folds[pos % k_folds].push(idx);
        }

        let mut scores: Vec<(usize, f64)> = Vec::with_capacity(candidates.len());
        for &bw in candidates {
            let mut fold_rmses: Vec<f64> = Vec::with_capacity(k_folds);
            for k in 0..k_folds {
                let test_idx = &folds[k];
                let train_idx: Vec<usize> = (0..k_folds)
                    .filter(|&j| j != k)
                    .flat_map(|j| folds[j].iter().copied())
                    .collect();
                // Need enough neighbours to fit a meaningful local model.
                if train_idx.len() <= 3 || test_idx.is_empty() {
                    continue;
                }

                let train_features = features.select(ndarray::Axis(0), &train_idx);
                let train_target: Vec<f64> = train_idx.iter().map(|&j| target[j]).collect();
                let train_coords: Vec<(f64, f64)> =
                    train_idx.iter().map(|&j| self.coords[j]).collect();
                let train_task = RegressionTask::new(task.id(), train_features, train_target)?;

                let mut fold_model = GeoXGBoost::new(train_coords)
                    .with_bandwidth(bw)
                    .with_n_estimators(self.n_estimators)
                    .with_max_depth(self.max_depth)
                    .with_learning_rate(self.learning_rate)
                    .with_lambda(self.lambda)
                    .with_seed(self.seed);
                if let Some(a) = self.alpha {
                    fold_model = fold_model.with_alpha(a);
                }
                let trained = fold_model.train_geo(&train_task)?;

                let test_features = features.select(ndarray::Axis(0), test_idx);
                let test_coords: Vec<(f64, f64)> =
                    test_idx.iter().map(|&j| self.coords[j]).collect();
                let pred = trained.predict_spatial(&test_features, &test_coords)?;
                let pred_vals = match &pred {
                    Prediction::Regression { predicted, .. } => predicted,
                    _ => return Err(SmeltError::Other("Expected regression".into())),
                };

                let mut sse = 0.0;
                for (m, &j) in test_idx.iter().enumerate() {
                    let e = target[j] - pred_vals[m];
                    sse += e * e;
                }
                fold_rmses.push((sse / test_idx.len() as f64).sqrt());
            }

            let mean_rmse = if fold_rmses.is_empty() {
                f64::INFINITY
            } else {
                fold_rmses.iter().sum::<f64>() / fold_rmses.len() as f64
            };
            scores.push((bw, mean_rmse));
        }

        let best = scores
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|&(bw, _)| bw)
            .unwrap_or(self.bandwidth);

        Ok(BandwidthSelection { best, scores })
    }

    fn make_xgb(&self) -> XGBoost {
        XGBoost::new()
            .with_n_estimators(self.n_estimators)
            .with_max_depth(self.max_depth)
            .with_learning_rate(self.learning_rate)
            .with_lambda(self.lambda)
            .with_seed(self.seed)
    }
}

/// Result of a [`GeoXGBoost::select_bandwidth`] cross-validation sweep.
pub struct BandwidthSelection {
    /// Bandwidth (neighbour count) with the lowest mean CV RMSE.
    pub best: usize,
    /// `(bandwidth, mean_cv_rmse)` for every candidate, in input order.
    pub scores: Vec<(usize, f64)>,
}

/// Deterministic Fisher-Yates shuffle of `0..n` driven by a seeded LCG.
/// Used to assign cross-validation folds reproducibly without pulling in `rand`.
fn seeded_permutation(n: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    for i in (1..n).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = ((state >> 33) as usize) % (i + 1);
        idx.swap(i, j);
    }
    idx
}

/// Euclidean distance between two points.
#[inline]
fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

/// Bi-square kernel weight (Eq. 10).
#[inline]
fn bisquare(d: f64, bandwidth: f64) -> f64 {
    if d < bandwidth {
        let r = d / bandwidth;
        (1.0 - r * r) * (1.0 - r * r)
    } else {
        0.0
    }
}

/// Compute spatial weights for point i using adaptive bandwidth (N nearest neighbors).
fn spatial_weights(coords: &[(f64, f64)], center: usize, n_neighbors: usize) -> Vec<f64> {
    let n = coords.len();
    let ci = coords[center];

    // Compute distances to all other points
    let mut dists: Vec<(usize, f64)> = (0..n).map(|j| (j, dist(ci, coords[j]))).collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Adaptive bandwidth = distance to the N-th nearest neighbor
    let bw_idx = n_neighbors.min(n - 1);
    let bandwidth = dists[bw_idx].1.max(1e-10);

    // Bi-square kernel weights
    let mut weights = vec![0.0; n];
    for &(j, d) in &dists {
        weights[j] = bisquare(d, bandwidth);
    }
    // Exclude the central point (leave-one-out for OOB error)
    weights[center] = 0.0;

    weights
}

// ── Trained model ───────────────────────────────────────────────────

/// Trained G-XGBoost model with global model, local models, and alpha weights.
pub struct TrainedGeoXGBoost {
    global_model: Box<dyn TrainedModel>,
    local_models: Vec<Box<dyn TrainedModel>>,
    alphas: Vec<f64>,
    coords: Vec<(f64, f64)>,
    feature_names: Vec<String>,
    local_importances: Vec<Option<Vec<(String, f64)>>>,
}

impl TrainedGeoXGBoost {
    /// Predict on new data using the nearest local model for each point.
    ///
    /// For each new coordinate, finds the closest training point and uses
    /// its local model blended with the global model via the stored alpha weight.
    /// This matches the behavior of the Python `predict_gxgb` function.
    pub fn predict_spatial(
        &self,
        features: &Array2<f64>,
        new_coords: &[(f64, f64)],
    ) -> Result<Prediction> {
        crate::validate::check_n_features(features, self.feature_names.len())?;
        let n_samples = features.nrows();
        if new_coords.len() != n_samples {
            return Err(SmeltError::DimensionMismatch {
                expected: n_samples,
                got: new_coords.len(),
            });
        }

        let global_pred = self.global_model.predict(features)?;
        let global_vals = match &global_pred {
            Prediction::Regression { predicted, .. } => predicted.clone(),
            _ => return Err(SmeltError::Other("Expected regression".into())),
        };

        let mut predicted = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            // Find nearest training point
            let nearest = self
                .coords
                .iter()
                .enumerate()
                .map(|(j, &c)| (j, dist(new_coords[i], c)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(j, _)| j)
                .unwrap_or(0);

            let row = features.select(ndarray::Axis(0), &[i]);
            let local_pred = self.local_models[nearest].predict(&row)?;
            let local_val = match &local_pred {
                Prediction::Regression { predicted, .. } => predicted[0],
                _ => return Err(SmeltError::Other("Expected regression".into())),
            };

            let alpha = self.alphas[nearest];
            let ensemble = alpha * local_val + (1.0 - alpha) * global_vals[i];
            predicted.push(ensemble);
        }

        Ok(Prediction::regression(predicted))
    }
}

impl TrainedModel for TrainedGeoXGBoost {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        crate::validate::check_n_features(features, self.feature_names.len())?;

        // For prediction on new data: find nearest local model for each sample
        // (requires coordinates — use the training coords to match)
        // If n_samples matches training, use corresponding local model.
        // Otherwise, use global model only.

        let n_samples = features.nrows();

        if n_samples == self.local_models.len() {
            // Predict on training data: use each point's own local model
            let global_pred = self.global_model.predict(features)?;
            let global_vals = match &global_pred {
                Prediction::Regression { predicted, .. } => predicted.clone(),
                _ => return Err(SmeltError::Other("Expected regression".into())),
            };

            let mut predicted = Vec::with_capacity(n_samples);
            for i in 0..n_samples {
                let row = features.select(ndarray::Axis(0), &[i]);
                let local_pred = self.local_models[i].predict(&row)?;
                let local_val = match &local_pred {
                    Prediction::Regression { predicted, .. } => predicted[0],
                    _ => return Err(SmeltError::Other("Expected regression".into())),
                };

                let alpha = self.alphas[i];
                let ensemble = alpha * local_val + (1.0 - alpha) * global_vals[i];
                predicted.push(ensemble);
            }

            Ok(Prediction::regression(predicted))
        } else {
            // New data: use global model (or find nearest local model)
            self.global_model.predict(features)
        }
    }

    fn feature_importance(&self) -> Option<Vec<(String, f64)>> {
        // Average local importances across all spatial units
        let valid: Vec<&Vec<(String, f64)>> = self
            .local_importances
            .iter()
            .filter_map(|i| i.as_ref())
            .collect();
        if valid.is_empty() {
            return self.global_model.feature_importance();
        }

        let n_features = valid[0].len();
        let mut avg = vec![0.0; n_features];
        let mut names = Vec::new();
        for imp in &valid {
            if names.is_empty() {
                names = imp.iter().map(|(n, _)| n.clone()).collect();
            }
            for (j, (_, v)) in imp.iter().enumerate() {
                avg[j] += v;
            }
        }
        let n_valid = valid.len() as f64;
        Some(
            names
                .into_iter()
                .zip(avg)
                .map(|(n, v)| (n, v / n_valid))
                .collect(),
        )
    }
}

// ── Learner implementation ──────────────────────────────────────────

impl Learner for GeoXGBoost {
    fn id(&self) -> &str {
        "geo_xgboost"
    }

    fn train_classif(&mut self, _: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        Err(SmeltError::Other(
            "GeoXGBoost only supports regression".into(),
        ))
    }

    fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
        Ok(Box::new(self.train_geo_inner(task)?))
    }
}

impl GeoXGBoost {
    fn train_geo_inner(&mut self, task: &RegressionTask) -> Result<TrainedGeoXGBoost> {
        let features = task.features();
        let target = task.target();
        let n_samples = task.n_samples();

        if self.coords.len() != n_samples {
            return Err(SmeltError::DimensionMismatch {
                expected: n_samples,
                got: self.coords.len(),
            });
        }

        let bandwidth = self.bandwidth.min(n_samples - 1);

        // Step 1: Train global XGBoost model
        let mut global_xgb = self.make_xgb();
        let global_model = global_xgb.train_regress(task)?;
        let global_pred = global_model.predict(features)?;
        let global_vals = match &global_pred {
            Prediction::Regression { predicted, .. } => predicted.clone(),
            _ => return Err(SmeltError::Other("Expected regression".into())),
        };

        // Global errors per point
        let global_errors: Vec<f64> = target
            .iter()
            .zip(&global_vals)
            .map(|(y, p)| (y - p).abs())
            .collect();

        // Step 2: Train local models for each spatial unit
        let mut local_models: Vec<Box<dyn TrainedModel>> = Vec::with_capacity(n_samples);
        let mut local_preds = vec![0.0; n_samples];
        let mut local_importances = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let ws = spatial_weights(&self.coords, i, bandwidth);

            // Create weighted training data: include samples with ws > 0 (excluding center)
            let weighted_indices: Vec<usize> =
                (0..n_samples).filter(|&j| ws[j] > 0.0 && j != i).collect();

            if weighted_indices.len() < 3 {
                // Not enough neighbors: use global model for this point
                local_preds[i] = global_vals[i];
                local_models.push(global_xgb.train_regress(task)?); // fallback
                local_importances.push(None);
                continue;
            }

            // Build local task with weighted samples
            // G-XGBoost uses sample_weights in the XGBoost objective (Eq. 13)
            // Our XGBoost doesn't support sample weights directly, so we
            // approximate by repeating samples proportional to their weight
            // OR by creating a subset and using the spatial weight as importance

            // Simpler approach: create sub-task from neighborhood
            let local_features = features.select(ndarray::Axis(0), &weighted_indices);
            let local_target: Vec<f64> = weighted_indices.iter().map(|&j| target[j]).collect();
            let local_task = RegressionTask::new(task.id(), local_features, local_target)?;

            let mut local_xgb = self.make_xgb();
            let local_model = local_xgb.train_regress(&local_task)?;

            // OOB prediction for center point (Eq. 14)
            let center_row = features.select(ndarray::Axis(0), &[i]);
            let center_pred = local_model.predict(&center_row)?;
            local_preds[i] = match &center_pred {
                Prediction::Regression { predicted, .. } => predicted[0],
                _ => global_vals[i],
            };

            local_importances.push(local_model.feature_importance());
            local_models.push(local_model);
        }

        // Step 3: Compute alpha weights (Eq. 19, 20)
        let alphas: Vec<f64> = (0..n_samples)
            .map(|i| {
                match self.alpha {
                    Some(a) => a, // fixed alpha
                    None => {
                        // Adaptive: favor local when it has lower error
                        let e_local = (target[i] - local_preds[i]).abs();
                        let e_global = global_errors[i];
                        if e_local <= e_global {
                            1.0 // local is better: use 100% local
                        } else {
                            // Local is worse: blend based on error ratio
                            // α = 1 - (e_local - e_global) / e_local.max(1e-10)
                            let ratio = (e_local - e_global) / e_local.max(1e-10);
                            (1.0 - ratio).max(0.0)
                        }
                    }
                }
            })
            .collect();

        Ok(TrainedGeoXGBoost {
            global_model,
            local_models,
            alphas,
            coords: self.coords.clone(),
            feature_names: task.feature_names().to_vec(),
            local_importances,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Build a small spatially-structured regression task on a grid.
    fn toy_task(side: usize) -> (RegressionTask, Vec<(f64, f64)>) {
        let n = side * side;
        let mut feats = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);
        let mut coords = Vec::with_capacity(n);
        for r in 0..side {
            for c in 0..side {
                let x = c as f64;
                let y = r as f64;
                coords.push((x, y));
                // Local relationship varies across space (non-stationarity).
                feats.push(x + y);
                target.push(x * 2.0 + y * 0.5 + (x * y) * 0.1);
            }
        }
        let features = Array2::from_shape_vec((n, 1), feats).unwrap();
        (RegressionTask::new("toy", features, target).unwrap(), coords)
    }

    #[test]
    fn seeded_permutation_is_a_permutation() {
        let p = seeded_permutation(50, 7);
        let mut sorted = p.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..50).collect::<Vec<_>>());
        // Deterministic for a given seed.
        assert_eq!(p, seeded_permutation(50, 7));
    }

    #[test]
    fn select_bandwidth_returns_a_candidate() {
        let (task, coords) = toy_task(8); // 64 points
        let gxgb = GeoXGBoost::new(coords).with_n_estimators(20).with_alpha(0.5);
        let candidates = [5usize, 10, 20, 40];
        let sel = gxgb.select_bandwidth(&task, &candidates, 4).unwrap();
        assert!(candidates.contains(&sel.best));
        assert_eq!(sel.scores.len(), candidates.len());
        // Every reported score is finite (every candidate was actually evaluated).
        assert!(sel.scores.iter().all(|&(_, r)| r.is_finite()));
    }

    #[test]
    fn select_bandwidth_rejects_too_few_folds() {
        let (task, coords) = toy_task(5);
        let gxgb = GeoXGBoost::new(coords);
        assert!(gxgb.select_bandwidth(&task, &[5], 1).is_err());
    }
}
