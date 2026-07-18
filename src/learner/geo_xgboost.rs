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
//! - Bandwidth selection via the leave-one-out CV criterion (`select_bandwidth`)
//! - Local feature importance

use crate::learner::xgboost::XGBoost;
use crate::learner::{Learner, TrainedModel};
use crate::prediction::Prediction;
use crate::resample::{CrossValidation, Resample};
use crate::task::{RegressionTask, Task};
use crate::{Result, SmeltError};
use ndarray::Array2;
use rayon::prelude::*;

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

/// Minimum recommended adaptive bandwidth (number of nearest neighbours).
///
/// Per Grekousis (author correspondence, 2026-07-11): geographically
/// weighted models are unreliable below ~30 units around each location, so
/// [`GeoXGBoost::select_bandwidth`] rejects smaller candidates outright.
/// `with_bandwidth` still accepts smaller values as an explicit override
/// (toy datasets, quick tests) -- documented there. If a fixed-DISTANCE
/// bandwidth is ever added (only KNN-adaptive exists today), it needs the
/// companion check: count how many locations end up with fewer than this
/// many neighbours at that distance, and advise raising it.
pub const MIN_BANDWIDTH: usize = 30;

impl GeoXGBoost {
    /// Creates a `GeoXGBoost` learner from spatial unit coordinates, with
    /// default hyperparameters and adaptive alpha weighting.
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

    /// Sets the number of nearest neighbors used by the bi-square spatial
    /// kernel; use `select_bandwidth` to pick this via leave-one-out CV.
    ///
    /// Convention (aligned with Grekousis's reference implementation): `bw`
    /// counts neighbours AROUND the central point, which is not included;
    /// the kernel distance `d_h` is the distance to the `bw`-th neighbour,
    /// which therefore itself receives bi-square weight 0. Values below
    /// [`MIN_BANDWIDTH`] are accepted here as an explicit override, but
    /// geographically weighted fits are unreliable that small --
    /// `select_bandwidth` refuses them.
    pub fn with_bandwidth(mut self, bw: usize) -> Self {
        self.bandwidth = bw;
        self
    }
    /// Sets the number of boosting rounds for both the global and local
    /// XGBoost models.
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }
    /// Sets the maximum tree depth for both the global and local XGBoost
    /// models.
    pub fn with_max_depth(mut self, d: usize) -> Self {
        self.max_depth = d;
        self
    }
    /// Sets the shrinkage applied to each tree's contribution in both the
    /// global and local XGBoost models.
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }
    /// Sets the L2 regularization strength for both the global and local
    /// XGBoost models.
    pub fn with_lambda(mut self, l: f64) -> Self {
        self.lambda = l;
        self
    }
    /// Fixes the global/local blending weight (0.0 = only global, 1.0 =
    /// only local); when unset, alpha is computed adaptively per Eq. 20.
    pub fn with_alpha(mut self, a: f64) -> Self {
        self.alpha = Some(a);
        self
    }
    /// Sets the RNG seed used by the underlying XGBoost models.
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

    /// Select the optimal bandwidth (number of nearest neighbours) by minimising
    /// the leave-one-out cross-validation (CV) criterion of Grekousis (2025, Eq. 11).
    ///
    /// For each candidate bandwidth `h` and each location `i`, a *local* model is
    /// calibrated on `i`'s spatially-weighted neighbours **excluding `i` itself**,
    /// and used to predict `y_i`. The CV criterion is the resulting leave-one-out
    /// error, `sqrt( mean_i (y_i - ŷ_{≠i}(h))² )`; the bandwidth minimising it is
    /// returned, together with the per-candidate scores.
    ///
    /// This is deliberately a property of the **local model alone**: the global
    /// model and the ensemble weight `alpha` are *not* involved, because bandwidth
    /// is tuned before the global/local blending step (cf. Grekousis, 2025). A
    /// too-small neighbourhood leaves each excluded location with too few points
    /// to predict well, so the criterion spikes — which is precisely why the LOO
    /// criterion, not the ensemble's hold-out RMSE, is the correct objective.
    ///
    /// `n_estimators`, `max_depth`, `learning_rate`, `lambda` and `seed` are held
    /// at their current values.
    pub fn select_bandwidth(
        &self,
        task: &RegressionTask,
        candidates: &[usize],
    ) -> Result<BandwidthSelection> {
        let n = task.n_samples();
        if self.coords.len() != n {
            return Err(SmeltError::DimensionMismatch {
                expected: n,
                got: self.coords.len(),
            });
        }
        crate::validate::check_coords_finite(&self.coords)?;
        if candidates.is_empty() {
            return Err(SmeltError::InvalidParameter(
                "select_bandwidth requires at least one candidate bandwidth".into(),
            ));
        }
        if let Some(&bad) = candidates.iter().find(|&&c| c < MIN_BANDWIDTH) {
            return Err(SmeltError::InvalidParameter(format!(
                "candidate bandwidth {bad} is below the minimum of {MIN_BANDWIDTH} neighbours: \
                 geographically weighted models are unreliable with fewer than ~30 units per \
                 neighbourhood (Grekousis). Use with_bandwidth() directly if you really need a \
                 smaller value on toy data"
            )));
        }
        // 5th audit, M-1: with n <= MIN_BANDWIDTH every candidate (all >=
        // MIN_BANDWIDTH per the check above) would be clamped to n-1 inside
        // `loo_cv_criterion`, producing a fictitious sweep — identical
        // scores for every candidate and a reported "best" whose effective
        // neighbourhood violates the documented 30-neighbour minimum.
        // Erroring here is the conservative option (a warning-plus-proceed
        // mode would need to be agreed with Grekousis first, since it would
        // report results the method's own guidance calls unreliable).
        if n - 1 < MIN_BANDWIDTH {
            return Err(SmeltError::InvalidParameter(format!(
                "dataset with n={n} samples cannot satisfy the {MIN_BANDWIDTH}-neighbour minimum: \
                 each location has at most n-1={} neighbours, so every candidate bandwidth would \
                 silently clamp to the same value and the selection sweep would be meaningless. \
                 Bandwidth selection needs at least {} samples; use with_bandwidth() directly for \
                 toy data",
                n - 1,
                MIN_BANDWIDTH + 1
            )));
        }

        let mut scores: Vec<(usize, f64)> = Vec::with_capacity(candidates.len());
        for &bw in candidates {
            scores.push((bw, self.loo_cv_criterion(task, bw)?));
        }

        let best = scores
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|&(bw, _)| bw)
            .unwrap_or(self.bandwidth);

        Ok(BandwidthSelection { best, scores })
    }

    /// Leave-one-out CV criterion (Grekousis 2025, Eq. 11) for a single bandwidth.
    ///
    /// Returns `sqrt( mean_i (y_i - ŷ_{≠i})² )`, where `ŷ_{≠i}` is the prediction
    /// for location `i` from a local model fit on its neighbours, excluding `i`.
    /// Uses the local model only (no global model, no `alpha`).
    fn loo_cv_criterion(&self, task: &RegressionTask, bandwidth: usize) -> Result<f64> {
        let features = task.features();
        let target = task.target();
        let n = task.n_samples();
        let bw = bandwidth.min(n - 1);

        // Each location's leave-one-out fit is independent -> compute in parallel.
        // Each returns Some(squared error) or None where the neighbourhood was
        // too small to fit a local model. NOTE: skipping a location makes the
        // candidate look BETTER, not worse (the skipped ones are exactly the
        // hard, sparse-neighbourhood locations) -- the old comment claimed the
        // opposite. With MIN_BANDWIDTH enforced by select_bandwidth this path
        // only triggers on degenerate geometry (e.g. massively duplicated
        // coordinates), where a local fit is meaningless anyway.
        let per_point: Vec<Result<Option<f64>>> = (0..n)
            .into_par_iter()
            .map(|i| -> Result<Option<f64>> {
                // spatial_weights already zeroes out the centre, so i is excluded.
                let ws = spatial_weights(&self.coords, i, bw);
                let idx: Vec<usize> = (0..n).filter(|&j| ws[j] > 0.0 && j != i).collect();
                if idx.len() < 3 {
                    return Ok(None);
                }

                let local_features = features.select(ndarray::Axis(0), &idx);
                let local_target: Vec<f64> = idx.iter().map(|&j| target[j]).collect();
                let local_weights: Vec<f64> = idx.iter().map(|&j| ws[j]).collect();
                let local_task = RegressionTask::new(task.id(), local_features, local_target)?;

                let mut local_xgb = self.make_xgb().with_sample_weights(local_weights);
                let local_model = local_xgb.train_regress(&local_task)?;

                let center_row = features.select(ndarray::Axis(0), &[i]);
                let pred = local_model.predict(&center_row)?;
                let yhat = match &pred {
                    Prediction::Regression { predicted, .. } => predicted[0],
                    _ => return Err(SmeltError::IncompatiblePrediction("Expected regression".into())),
                };
                let e = target[i] - yhat;
                Ok(Some(e * e))
            })
            .collect();

        let mut sse = 0.0;
        let mut count = 0usize;
        for r in per_point {
            if let Some(se) = r? {
                sse += se;
                count += 1;
            }
        }

        if count == 0 {
            return Ok(f64::INFINITY);
        }
        Ok((sse / count as f64).sqrt())
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
///
/// Convention (confirmed against Grekousis's reference implementation,
/// author correspondence 2026-07-11): the center sits at sorted position 0,
/// so `dists[n_neighbors]` is the n_neighbors-th NEIGHBOUR (center
/// excluded) -- his "rank 51 when 1 is the central unit" for h=50. That
/// neighbour defines `d_h` and thus gets bi-square weight 0 itself; the
/// center's weight is also zeroed (leave-one-out).
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
    /// One per training point; `None` where the neighbourhood was too small and
    /// the global model is used as the local prediction instead.
    local_models: Vec<Option<Box<dyn TrainedModel>>>,
    alphas: Vec<f64>,
    coords: Vec<(f64, f64)>,
    feature_names: Vec<String>,
    local_importances: Vec<Option<Vec<(String, f64)>>>,
}

impl TrainedGeoXGBoost {
    /// Training coordinates, one `(x, y)` per local model (same order as
    /// [`local_importances`](Self::local_importances)).
    pub fn coords(&self) -> &[(f64, f64)] {
        &self.coords
    }

    /// Per-location local-model feature importances, one entry per training
    /// point (in `coords` order). `None` where the neighbourhood was too small
    /// and the global model was used as a fallback. Each inner vector is a list
    /// of `(feature_name, gain)`. This is what lets you *map* how the influence
    /// of each predictor varies across space (spatial non-stationarity).
    pub fn local_importances(&self) -> &[Option<Vec<(String, f64)>>] {
        &self.local_importances
    }

    /// Feature names (internal `x0`, `x1`, ... order).
    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    /// Predict on new data using the nearest local model for each point.
    ///
    /// For each new coordinate, finds the closest training point and uses
    /// its local model blended with the global model via the stored alpha weight.
    /// This matches the behavior of the Python `predict_gxgb` function.
    ///
    /// To get **in-sample fitted values** (the training-set predictions that
    /// exercise each point's own local model), call this with the training
    /// features and [`Self::coords`] — each point then matches itself exactly
    /// (distance 0).
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
        crate::validate::check_coords_finite(new_coords)?;

        let global_pred = self.global_model.predict(features)?;
        let global_vals = match &global_pred {
            Prediction::Regression { predicted, .. } => predicted.clone(),
            _ => return Err(SmeltError::IncompatiblePrediction("Expected regression".into())),
        };

        let mut predicted = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            // Find nearest training point
            let nearest = self
                .coords
                .iter()
                .enumerate()
                .map(|(j, &c)| (j, dist(new_coords[i], c)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(j, _)| j)
                .unwrap_or(0);

            // If the nearest training point had no local model (neighbourhood
            // too small), fall back to the global prediction for this point.
            let ensemble = match &self.local_models[nearest] {
                Some(local_model) => {
                    let row = features.select(ndarray::Axis(0), &[i]);
                    let local_pred = local_model.predict(&row)?;
                    let local_val = match &local_pred {
                        Prediction::Regression { predicted, .. } => predicted[0],
                        _ => return Err(SmeltError::IncompatiblePrediction("Expected regression".into())),
                    };
                    let alpha = self.alphas[nearest];
                    alpha * local_val + (1.0 - alpha) * global_vals[i]
                }
                None => global_vals[i],
            };
            predicted.push(ensemble);
        }

        Ok(Prediction::regression(predicted))
    }
}

impl TrainedModel for TrainedGeoXGBoost {
    /// Global-model-only prediction.
    ///
    /// The `TrainedModel` trait has no notion of spatial coordinates, so this
    /// method cannot know whether `features` is the training set, a genuinely
    /// new dataset, or a new dataset that coincidentally has the same number
    /// of rows as the training set — matching on row count would silently
    /// apply the wrong local model to unrelated points. To get spatially-aware
    /// predictions (including "fitted values" on the training set, by passing
    /// the training coordinates back), use [`TrainedGeoXGBoost::predict_spatial`].
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        crate::validate::check_n_features(features, self.feature_names.len())?;
        self.global_model.predict(features)
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

    fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
        crate::validate::check_no_weights(task.weights(), "GeoXGBoost")?;
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
        crate::validate::check_coords_finite(&self.coords)?;

        // 5th audit, M-1 (train-path companion of the select_bandwidth
        // check): a nominal bandwidth that satisfies the documented
        // 30-neighbour minimum must not be silently clamped BELOW it by the
        // n-1 cap — the caller asked for a compliant neighbourhood and
        // would get an unreliable sub-minimum fit without any signal.
        // Nominal bandwidths already below MIN_BANDWIDTH stay accepted:
        // that is `with_bandwidth`'s documented explicit toy-data override.
        // Err (rather than warn-and-proceed) is the conservative choice;
        // a warning mode would need to be discussed with Grekousis first.
        if self.bandwidth >= MIN_BANDWIDTH && n_samples - 1 < MIN_BANDWIDTH {
            return Err(SmeltError::InvalidParameter(format!(
                "bandwidth {} cannot be honoured with n={} samples: it would be clamped to \
                 n-1={} neighbours, below the {MIN_BANDWIDTH}-neighbour minimum for reliable \
                 geographically weighted fits (Grekousis). Provide at least {} samples, or \
                 explicitly request a sub-minimum bandwidth via with_bandwidth() for toy data",
                self.bandwidth,
                n_samples,
                n_samples - 1,
                MIN_BANDWIDTH + 1
            )));
        }

        let bandwidth = self.bandwidth.min(n_samples - 1);

        // Step 1: Train global XGBoost model
        let mut global_xgb = self.make_xgb();
        let global_model = global_xgb.train_regress(task)?;
        let global_pred = global_model.predict(features)?;
        let global_vals = match &global_pred {
            Prediction::Regression { predicted, .. } => predicted.clone(),
            _ => return Err(SmeltError::IncompatiblePrediction("Expected regression".into())),
        };

        // Global errors per point, out-of-fold (CV) — comparable to the local
        // models' leave-one-out errors. Only needed for the adaptive alpha
        // (Eq. 19-20); skip the extra CV passes when alpha is fixed.
        let global_errors: Vec<f64> = if self.alpha.is_none() {
            self.global_oob_errors(task, &global_vals)?
        } else {
            Vec::new()
        };

        // Step 2: Train local models for each spatial unit.
        // The local models are independent, so we train them in parallel.
        // Each entry is `None` where the neighbourhood was too small (the global
        // model is used for that point at prediction time).
        type LocalFit = (Option<Box<dyn TrainedModel>>, f64, Option<Vec<(String, f64)>>);
        let fits: Vec<Result<LocalFit>> = (0..n_samples)
            .into_par_iter()
            .map(|i| -> Result<LocalFit> {
                let ws = spatial_weights(&self.coords, i, bandwidth);

                // Neighbourhood = points with ws > 0, excluding the centre.
                let weighted_indices: Vec<usize> =
                    (0..n_samples).filter(|&j| ws[j] > 0.0 && j != i).collect();

                if weighted_indices.len() < 3 {
                    // Not enough neighbours: fall back to the global model.
                    return Ok((None, global_vals[i], None));
                }

                // Fit a *weighted* local XGBoost: each neighbour's bi-square
                // kernel weight scales its gradient/hessian, so closer points
                // count more — the spatially-weighted objective of G-XGBoost
                // (Eq. 13), rather than a hard 0/1 subset.
                let local_features = features.select(ndarray::Axis(0), &weighted_indices);
                let local_target: Vec<f64> = weighted_indices.iter().map(|&j| target[j]).collect();
                let local_weights: Vec<f64> = weighted_indices.iter().map(|&j| ws[j]).collect();
                let local_task = RegressionTask::new(task.id(), local_features, local_target)?;

                let mut local_xgb = self.make_xgb().with_sample_weights(local_weights);
                let local_model = local_xgb.train_regress(&local_task)?;

                // OOB prediction for the centre point (Eq. 14).
                let center_row = features.select(ndarray::Axis(0), &[i]);
                let center_pred = local_model.predict(&center_row)?;
                let pred = match &center_pred {
                    Prediction::Regression { predicted, .. } => predicted[0],
                    _ => global_vals[i],
                };
                let imp = local_model.feature_importance();
                Ok((Some(local_model), pred, imp))
            })
            .collect();

        let mut local_models: Vec<Option<Box<dyn TrainedModel>>> = Vec::with_capacity(n_samples);
        let mut local_preds = vec![0.0; n_samples];
        let mut local_importances = Vec::with_capacity(n_samples);
        for (i, fit) in fits.into_iter().enumerate() {
            let (model, pred, imp) = fit?;
            local_models.push(model);
            local_preds[i] = pred;
            local_importances.push(imp);
        }

        // Step 3: Compute alpha weights (Eq. 19, 20). Both errors are
        // out-of-sample (e_local: leave-one-out; e_global: out-of-fold CV) so
        // the comparison is apples-to-apples — comparing e_local against an
        // in-sample global residual would systematically favour the global
        // model, since it has already seen every training point.
        let alphas: Vec<f64> = (0..n_samples)
            .map(|i| {
                match self.alpha {
                    Some(a) => a, // fixed alpha
                    None => {
                        // Adaptive: favor local when it has lower OOS error
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

    /// Out-of-fold absolute errors for the global model, via k-fold CV.
    ///
    /// The adaptive alpha (Eq. 19-20) compares the local models' leave-one-out
    /// error against the global model's error; using the global model's
    /// in-sample residual for that comparison is biased low (the global model
    /// has already seen every point) and systematically favours it. This
    /// refits the global model on `k` folds and returns each point's
    /// held-out residual instead, so both sides of the comparison are
    /// out-of-sample.
    fn global_oob_errors(&self, task: &RegressionTask, global_vals: &[f64]) -> Result<Vec<f64>> {
        let n_samples = task.n_samples();
        let target = task.target();
        let folds = 5.min(n_samples);

        if folds < 2 {
            // Too few points to cross-validate: fall back to the in-sample
            // residual (biased, but there is no out-of-sample alternative).
            return Ok(target.iter().zip(global_vals).map(|(y, p)| (y - p).abs()).collect());
        }

        let cv = CrossValidation::new(folds).with_seed(self.seed);
        let features = task.features();

        let fold_errors: Vec<Result<Vec<(usize, f64)>>> = cv
            .splits(n_samples)?
            .into_par_iter()
            .map(|(train_idx, test_idx)| -> Result<Vec<(usize, f64)>> {
                let fold_features = features.select(ndarray::Axis(0), &train_idx);
                let fold_target: Vec<f64> = train_idx.iter().map(|&j| target[j]).collect();
                let fold_task = RegressionTask::new(task.id(), fold_features, fold_target)?;

                let mut fold_xgb = self.make_xgb();
                let fold_model = fold_xgb.train_regress(&fold_task)?;

                let test_features = features.select(ndarray::Axis(0), &test_idx);
                let test_pred = fold_model.predict(&test_features)?;
                let test_vals = match &test_pred {
                    Prediction::Regression { predicted, .. } => predicted.clone(),
                    _ => return Err(SmeltError::IncompatiblePrediction("Expected regression".into())),
                };
                Ok(test_idx
                    .into_iter()
                    .zip(test_vals)
                    .map(|(idx, pred)| (idx, (target[idx] - pred).abs()))
                    .collect())
            })
            .collect();

        let mut errors = vec![0.0; n_samples];
        for fold in fold_errors {
            for (idx, err) in fold? {
                errors[idx] = err;
            }
        }
        Ok(errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use rand::prelude::*;

    /// Regression test (4th audit, HIGH-3): a single NaN coordinate used to
    /// PANIC the process inside rayon — `slice::sort` (Rust ≥ 1.81) detects
    /// the non-total order that `partial_cmp().unwrap_or(Equal)` induces on
    /// NaN distances. A missing georeference must be a clean `Err`, not a
    /// crash.
    #[test]
    fn non_finite_coordinates_are_rejected_at_train_and_predict() {
        let n = 20;
        let features = Array2::from_shape_fn((n, 1), |(i, _)| i as f64);
        let target: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
        let task = RegressionTask::new("nan-coords", features.clone(), target).unwrap();

        let mut coords: Vec<(f64, f64)> = (0..n).map(|i| (i as f64, 0.0)).collect();
        coords[3] = (0.0, f64::NAN);
        let mut gxgb = GeoXGBoost::new(coords).with_n_estimators(5);
        let Err(err) = gxgb.train_geo(&task) else {
            panic!("NaN training coordinate must be rejected")
        };
        assert!(
            err.to_string().contains("index 3"),
            "error should name the offending coordinate: {err}"
        );

        let good_coords: Vec<(f64, f64)> = (0..n).map(|i| (i as f64, 0.0)).collect();
        // Explicit sub-minimum bandwidth: n=20 can't honour the default 30
        // without clamping below the documented minimum (rejected since the
        // 5th audit's M-1 fix); this test is about NaN coords, not that.
        let mut gxgb = GeoXGBoost::new(good_coords.clone())
            .with_n_estimators(5)
            .with_bandwidth(10);
        let trained = gxgb.train_geo(&task).unwrap();
        let Err(err) = trained.predict_spatial(
            &features.slice(ndarray::s![0..1, ..]).to_owned(),
            &[(f64::NAN, 0.0)],
        ) else {
            panic!("NaN query coordinate must be rejected")
        };
        assert!(err.to_string().contains("index 0"), "got: {err}");

        // select_bandwidth shares the same guard.
        let mut bad = good_coords;
        bad[9] = (f64::NEG_INFINITY, 0.0);
        let gxgb = GeoXGBoost::new(bad).with_n_estimators(5);
        let Err(err) = gxgb.select_bandwidth(&task, &[4, 8]) else {
            panic!("non-finite coordinate must be rejected in select_bandwidth")
        };
        assert!(err.to_string().contains("index 9"), "got: {err}");
    }

    /// Regression test: the adaptive alpha (Eq. 19-20) must compare the local
    /// model's leave-one-out error against an out-of-fold error for the
    /// global model, not its in-sample residual. A flexible global model on a
    /// small dataset can nearly memorize the training set (in-sample error
    /// close to 0) while its true generalization error is much higher; using
    /// the in-sample residual would make the global model look artificially
    /// competitive against the (genuinely out-of-sample) local models.
    #[test]
    fn global_oob_errors_are_not_optimistic_like_in_sample_residuals() {
        let n = 40;
        let mut rng = StdRng::seed_from_u64(7);
        let mut feats = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);
        let mut coords = Vec::with_capacity(n);
        for i in 0..n {
            let x = i as f64;
            coords.push((x, 0.0));
            feats.push(x);
            target.push(x * 0.3 + rng.random::<f64>() * 5.0); // noisy signal
        }
        let features = Array2::from_shape_vec((n, 1), feats).unwrap();
        let task = RegressionTask::new("oob", features.clone(), target.clone()).unwrap();

        // Deep, many-estimator, low-regularization XGBoost: with n=40 and a
        // single feature this essentially memorizes the training set.
        let gxgb = GeoXGBoost::new(coords)
            .with_n_estimators(300)
            .with_max_depth(8)
            .with_learning_rate(0.5)
            .with_lambda(0.01);

        let mut global_xgb = gxgb.make_xgb();
        let global_model = global_xgb.train_regress(&task).unwrap();
        let global_pred = global_model.predict(&features).unwrap();
        let Prediction::Regression { predicted: global_vals, .. } = global_pred else {
            panic!("expected regression")
        };

        let in_sample: Vec<f64> = target
            .iter()
            .zip(&global_vals)
            .map(|(y, p)| (y - p).abs())
            .collect();
        let oob = gxgb.global_oob_errors(&task, &global_vals).unwrap();

        let mean_in_sample: f64 = in_sample.iter().sum::<f64>() / n as f64;
        let mean_oob: f64 = oob.iter().sum::<f64>() / n as f64;

        assert!(
            mean_oob > mean_in_sample * 2.0,
            "OOB error ({mean_oob:.4}) should be markedly higher than the \
             optimistic in-sample residual ({mean_in_sample:.4}) once the \
             global model overfits"
        );
    }

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
    fn select_bandwidth_returns_a_candidate() {
        let (task, coords) = toy_task(8); // 64 points
        let gxgb = GeoXGBoost::new(coords).with_n_estimators(20);
        let candidates = [30usize, 40, 50];
        let sel = gxgb.select_bandwidth(&task, &candidates).unwrap();
        assert!(candidates.contains(&sel.best));
        assert_eq!(sel.scores.len(), candidates.len());
        // Every reported LOO criterion is finite (each candidate was evaluated).
        assert!(sel.scores.iter().all(|&(_, r)| r.is_finite()));
    }

    /// Regression test: `predict()` (the `TrainedModel` trait method) must be
    /// global-only, never keyed off row count. Previously, `predict()` checked
    /// `features.nrows() == local_models.len()` and, if so, applied each
    /// training point's local model by row *position* — silently wrong for
    /// any dataset that happens to have the same row count as training but
    /// is not, in fact, the training set (no coordinates to verify against).
    #[test]
    fn predict_is_global_only_never_positional_local_models() {
        let (task, coords) = toy_task(4); // 16 points
        let features = task.features().clone();
        let mut gxgb = GeoXGBoost::new(coords).with_n_estimators(20).with_bandwidth(4);
        let model = gxgb.train_geo(&task).unwrap();

        // Same row count as training (16), predicted through the trait method.
        let via_trait = TrainedModel::predict(&model, &features).unwrap();
        let Prediction::Regression { predicted: via_trait_vals, .. } = via_trait else {
            panic!("expected regression");
        };

        // Must match the global model's own prediction exactly — no local
        // model or alpha blending involved, regardless of row count matching.
        let global_only = model.global_model.predict(&features).unwrap();
        let Prediction::Regression { predicted: global_vals, .. } = global_only else {
            panic!("expected regression");
        };

        assert_eq!(via_trait_vals, global_vals, "predict() must equal the global model alone");
    }

    /// Per Grekousis (2026-07-11): neighbourhoods below ~30 units make
    /// geographically weighted fits unreliable, so bandwidth SELECTION
    /// refuses them outright; `with_bandwidth` stays available as the
    /// explicit toy-data override.
    #[test]
    fn select_bandwidth_rejects_candidates_below_minimum() {
        let (task, coords) = toy_task(8); // 64 points
        let gxgb = GeoXGBoost::new(coords).with_n_estimators(20);
        let Err(err) = gxgb.select_bandwidth(&task, &[10, 40]) else {
            panic!("candidate below MIN_BANDWIDTH must be rejected")
        };
        let msg = format!("{err}");
        assert!(msg.contains("30") && msg.contains("10"), "got: {msg}");
    }

    #[test]
    fn select_bandwidth_rejects_empty_grid() {
        let (task, coords) = toy_task(5);
        let gxgb = GeoXGBoost::new(coords);
        assert!(gxgb.select_bandwidth(&task, &[]).is_err());
    }

    /// Regression test (5th audit, M-1): with n <= MIN_BANDWIDTH the old
    /// code clamped every candidate to n-1 inside `loo_cv_criterion`,
    /// producing a fictitious sweep — identical scores for all candidates
    /// and a "best" whose effective neighbourhood (n-1 < 30) violated the
    /// documented minimum the nominal-value validation claims to enforce.
    /// It must be a clear `Err` naming n and the minimum instead.
    #[test]
    fn select_bandwidth_rejects_datasets_too_small_for_the_minimum() {
        let (task, coords) = toy_task(5); // n = 25 < MIN_BANDWIDTH + 1
        let gxgb = GeoXGBoost::new(coords).with_n_estimators(5);
        let Err(err) = gxgb.select_bandwidth(&task, &[30, 40, 50]) else {
            panic!("n=25 cannot satisfy the 30-neighbour minimum; the sweep must be rejected")
        };
        let msg = format!("{err}");
        assert!(
            msg.contains("n=25") && msg.contains("30"),
            "error must name n and the minimum: {msg}"
        );
    }

    /// M-1 train-path companion: a nominal bandwidth that meets the
    /// 30-neighbour minimum must not be silently clamped below it by the
    /// n-1 cap — that's an `Err` — while an explicitly sub-minimum
    /// bandwidth (the documented `with_bandwidth` toy-data override, no
    /// clamp below the minimum involved) keeps training.
    #[test]
    fn train_rejects_min_compliant_bandwidth_that_would_clamp_below_minimum() {
        let (task, coords) = toy_task(5); // n = 25

        // Default bandwidth (30, exactly the minimum) can't be honoured.
        let mut default_bw = GeoXGBoost::new(coords.clone()).with_n_estimators(5);
        let Err(err) = default_bw.train_geo(&task) else {
            panic!("bandwidth 30 with n=25 clamps to 24 < 30 and must be rejected")
        };
        let msg = format!("{err}");
        assert!(
            msg.contains("30") && msg.contains("24"),
            "error must name the requested bandwidth and the clamped value: {msg}"
        );

        // Explicit sub-minimum override still works (documented behavior).
        let mut small_bw = GeoXGBoost::new(coords)
            .with_n_estimators(5)
            .with_bandwidth(8);
        small_bw
            .train_geo(&task)
            .expect("explicit sub-minimum bandwidth must remain a valid toy-data override");
    }
}
