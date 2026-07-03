//! Resampling strategy wrappers: CrossValidation, SpatialBlockCV,
//! SpatialBufferCV, StratifiedCV, GroupCV.

use crate::common::{parse_coords, smelt_err};
use pyo3::prelude::*;

// ── CrossValidation ────────────────────────────────────────────────────

#[pyclass]
pub(crate) struct CrossValidation {
    inner: smelt_ml::prelude::CrossValidation,
}

#[pymethods]
impl CrossValidation {
    #[new]
    #[pyo3(signature = (n_folds=5, seed=42))]
    fn new(n_folds: usize, seed: u64) -> Self {
        Self {
            inner: smelt_ml::prelude::CrossValidation::new(n_folds).with_seed(seed),
        }
    }

    fn splits(&self, n_samples: usize) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
        use smelt_ml::prelude::Resample;
        self.inner.splits(n_samples).map_err(smelt_err)
    }
}

// ── SpatialBlockCV ─────────────────────────────────────────────────────

#[pyclass]
pub(crate) struct SpatialBlockCV {
    inner: smelt_ml::prelude::SpatialBlockCV,
}

#[pymethods]
impl SpatialBlockCV {
    #[new]
    fn new(n_folds: usize, coords: &Bound<'_, PyAny>) -> PyResult<Self> {
        let parsed = parse_coords(coords)?;
        Ok(Self {
            inner: smelt_ml::prelude::SpatialBlockCV::new(n_folds, parsed),
        })
    }

    fn splits(&self, n_samples: usize) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
        use smelt_ml::prelude::Resample;
        self.inner.splits(n_samples).map_err(smelt_err)
    }
}

// ── SpatialBufferCV ────────────────────────────────────────────────────

#[pyclass]
pub(crate) struct SpatialBufferCV {
    inner: smelt_ml::resample::SpatialBufferCV,
}

#[pymethods]
impl SpatialBufferCV {
    /// Buffered k-fold spatial CV. Training samples within `buffer_distance`
    /// of any test sample are excluded, reducing spatial autocorrelation leakage.
    ///
    /// For Spatial Leave-One-Out behaviour, set `n_folds = n_samples`.
    #[new]
    #[pyo3(signature = (n_folds, coords, buffer_distance, seed=42))]
    fn new(
        n_folds: usize,
        coords: &Bound<'_, PyAny>,
        buffer_distance: f64,
        seed: u64,
    ) -> PyResult<Self> {
        let parsed = parse_coords(coords)?;
        Ok(Self {
            inner: smelt_ml::resample::SpatialBufferCV::new(n_folds, parsed, buffer_distance)
                .with_seed(seed),
        })
    }

    fn splits(&self, n_samples: usize) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
        use smelt_ml::prelude::Resample;
        self.inner.splits(n_samples).map_err(smelt_err)
    }
}

// ── StratifiedCV ───────────────────────────────────────────────────────

#[pyclass]
pub(crate) struct StratifiedCV {
    inner: smelt_ml::prelude::StratifiedCV,
}

#[pymethods]
impl StratifiedCV {
    /// Stratified k-fold: each fold preserves the overall class proportions.
    /// `labels` are the classification target values (0-indexed class ids).
    #[new]
    #[pyo3(signature = (n_folds, labels, seed=42))]
    fn new(n_folds: usize, labels: Vec<usize>, seed: u64) -> Self {
        Self {
            inner: smelt_ml::prelude::StratifiedCV::new(n_folds, labels).with_seed(seed),
        }
    }

    fn splits(&self, n_samples: usize) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
        use smelt_ml::prelude::Resample;
        self.inner.splits(n_samples).map_err(smelt_err)
    }
}

// ── GroupCV ────────────────────────────────────────────────────────────

#[pyclass]
pub(crate) struct GroupCV {
    inner: smelt_ml::prelude::GroupCV,
}

#[pymethods]
impl GroupCV {
    /// Group k-fold: every sample sharing a group id stays in the same
    /// fold, so a group never spans both train and test.
    #[new]
    #[pyo3(signature = (n_folds, groups, seed=42))]
    fn new(n_folds: usize, groups: Vec<usize>, seed: u64) -> Self {
        Self {
            inner: smelt_ml::prelude::GroupCV::new(n_folds, groups).with_seed(seed),
        }
    }

    fn splits(&self, n_samples: usize) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
        use smelt_ml::prelude::Resample;
        self.inner.splits(n_samples).map_err(smelt_err)
    }
}

