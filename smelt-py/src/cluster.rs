//! Unsupervised clustering and anomaly detection: KMeans, DBSCAN,
//! IsolationForest. These don't go through the `Learner`/`TrainedModel`
//! pipeline (no `Task`, no `predict` against unseen labels) -- each wrapper
//! just calls the underlying Rust `fit`/`fit_predict` directly and returns
//! plain numpy arrays / Python values, matching `Smote`/`SpatialSmote`'s
//! shape in `preprocess.rs` rather than `define_learner!`'s.

use crate::common::{smelt_err, to_array2};
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

// ── KMeans ───────────────────────────────────────────────────────────

#[pyclass]
pub(crate) struct KMeans {
    k: usize,
    max_iter: usize,
    n_init: usize,
    seed: u64,
}

#[pymethods]
impl KMeans {
    #[new]
    #[pyo3(signature = (k, max_iter=300, n_init=10, seed=42))]
    fn new(k: usize, max_iter: usize, n_init: usize, seed: u64) -> PyResult<Self> {
        // k=0 used to surface as a PanicException from an ndarray index
        // assert deep inside fit (audit M-17); k > n_samples is validated
        // by the core at fit time, but k=0 never reached that check.
        if k == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "k must be at least 1",
            ));
        }
        Ok(Self { k, max_iter, n_init, seed })
    }

    /// Fit and return `(labels, centroids)`. `labels[i]` is the cluster
    /// index for sample `i`; `centroids` is `(n_clusters, n_features)`.
    fn fit_predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<(Vec<i32>, Bound<'py, PyArray2<f64>>)> {
        let features = to_array2(x);
        let result = py
            .allow_threads(|| {
                smelt_ml::prelude::KMeans::new(self.k)
                    .with_max_iter(self.max_iter)
                    .with_n_init(self.n_init)
                    .with_seed(self.seed)
                    .fit(&features)
            })
            .map_err(smelt_err)?;
        let centroids = result
            .centroids
            .expect("KMeans::fit always sets centroids");
        Ok((result.labels, PyArray2::from_owned_array(py, centroids)))
    }

    /// Silhouette score of a clustering (range [-1, 1], higher is better).
    /// `labels` is a previous `fit_predict` result's first element.
    fn silhouette_score(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        labels: Vec<i32>,
    ) -> PyResult<f64> {
        let features = to_array2(x);
        // A labels/rows length mismatch used to panic with a raw
        // index-out-of-bounds inside the O(n²) loop (audit M-17).
        if labels.len() != features.nrows() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "labels has {} entries but x has {} rows -- pass the labels fit_predict \
                 returned for this same x",
                labels.len(),
                features.nrows()
            )));
        }
        let n_clusters = labels.iter().filter(|&&l| l >= 0).map(|&l| l as usize).max().map_or(0, |m| m + 1);
        let result = smelt_ml::prelude::ClusterResult {
            labels,
            n_clusters,
            centroids: None,
        };
        // O(n²) pairwise distances: release the GIL like the other fits here.
        Ok(py.allow_threads(|| result.silhouette_score(&features)))
    }
}

// ── DBSCAN ───────────────────────────────────────────────────────────

#[pyclass]
pub(crate) struct DBSCAN {
    eps: f64,
    min_pts: usize,
}

#[pymethods]
impl DBSCAN {
    #[new]
    #[pyo3(signature = (eps, min_pts=5))]
    fn new(eps: f64, min_pts: usize) -> Self {
        Self { eps, min_pts }
    }

    /// Fit and return cluster labels (`-1` = noise).
    fn fit_predict(&self, py: Python<'_>, x: PyReadonlyArray2<'_, f64>) -> PyResult<Vec<i32>> {
        let features = to_array2(x);
        let result = py
            .allow_threads(|| smelt_ml::prelude::DBSCAN::new(self.eps, self.min_pts).fit(&features))
            .map_err(smelt_err)?;
        Ok(result.labels)
    }
}

// ── IsolationForest ────────────────────────────────────────────────────

#[pyclass]
pub(crate) struct IsolationForest {
    n_estimators: usize,
    max_samples: Option<usize>,
    contamination: f64,
    seed: u64,
}

#[pymethods]
impl IsolationForest {
    #[new]
    #[pyo3(signature = (n_estimators=100, max_samples=None, contamination=0.1, seed=42))]
    fn new(n_estimators: usize, max_samples: Option<usize>, contamination: f64, seed: u64) -> Self {
        Self { n_estimators, max_samples, contamination, seed }
    }

    /// Fit and return `(scores, labels)`. `scores` are anomaly scores in
    /// `[0, 1]` (higher = more anomalous); `labels` are `1` (anomaly) or `0`
    /// (normal), thresholded by `contamination`.
    fn fit_predict(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<(Vec<f64>, Vec<i32>)> {
        let features = to_array2(x);
        let mut iforest = smelt_ml::prelude::IsolationForest::new()
            .with_n_estimators(self.n_estimators)
            .with_contamination(self.contamination)
            .with_seed(self.seed);
        if let Some(m) = self.max_samples {
            iforest = iforest.with_max_samples(m);
        }
        let result = py
            .allow_threads(|| iforest.fit_predict(&features))
            .map_err(smelt_err)?;
        Ok((result.scores, result.labels))
    }
}
