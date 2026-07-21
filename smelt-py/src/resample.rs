//! Resampling strategy wrappers: CrossValidation, RepeatedCV, LeaveOneOut,
//! Bootstrap, SpatialBlockCV, SpatialBufferCV, StratifiedCV, GroupCV,
//! TimeSeriesCV.

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

// ── RepeatedCV ─────────────────────────────────────────────────────────

#[pyclass]
pub(crate) struct RepeatedCV {
    inner: smelt_ml::prelude::RepeatedCV,
}

#[pymethods]
impl RepeatedCV {
    /// Repeated k-fold CV: runs plain k-fold `n_repeats` times with a
    /// different shuffle each time, yielding `n_folds * n_repeats` splits.
    /// Averaging over them reduces the variance from a single fold assignment.
    #[new]
    #[pyo3(signature = (n_folds=5, n_repeats=10, seed=42))]
    fn new(n_folds: usize, n_repeats: usize, seed: u64) -> Self {
        Self {
            inner: smelt_ml::prelude::RepeatedCV::new(n_folds, n_repeats).with_seed(seed),
        }
    }

    fn splits(&self, n_samples: usize) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
        use smelt_ml::prelude::Resample;
        self.inner.splits(n_samples).map_err(smelt_err)
    }
}

// ── LeaveOneOut ────────────────────────────────────────────────────────

#[pyclass]
pub(crate) struct LeaveOneOut {
    inner: smelt_ml::prelude::LeaveOneOut,
}

#[pymethods]
impl LeaveOneOut {
    /// Leave-one-out CV: one split per sample, each holding out a single
    /// point and training on the rest. Deterministic (no seed).
    #[new]
    fn new() -> Self {
        Self {
            inner: smelt_ml::prelude::LeaveOneOut,
        }
    }

    fn splits(&self, n_samples: usize) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
        use smelt_ml::prelude::Resample;
        self.inner.splits(n_samples).map_err(smelt_err)
    }
}

// ── Bootstrap ──────────────────────────────────────────────────────────

#[pyclass]
pub(crate) struct Bootstrap {
    inner: smelt_ml::prelude::Bootstrap,
}

#[pymethods]
impl Bootstrap {
    /// Bootstrap resampling: each resample draws `n_samples` train indices
    /// with replacement; the never-drawn out-of-bag (~36.8%) become the test
    /// set. Draws with an empty OOB set are skipped, so exactly `n_resamples`
    /// usable splits are returned (requires `n_samples >= 2`).
    #[new]
    #[pyo3(signature = (n_resamples=30, seed=42))]
    fn new(n_resamples: usize, seed: u64) -> Self {
        Self {
            inner: smelt_ml::prelude::Bootstrap::new(n_resamples).with_seed(seed),
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
    /// `block_size`, when given, fixes the grid cell side length (in the
    /// same units as `coords`) independently of `n_folds` — otherwise the
    /// grid resolution is derived from `n_folds` alone
    /// (`ceil(sqrt(n_folds))` per side), which cannot express e.g. "2 km
    /// blocks" over a large extent without inflating `n_folds`.
    #[new]
    #[pyo3(signature = (n_folds, coords, block_size=None))]
    fn new(n_folds: usize, coords: &Bound<'_, PyAny>, block_size: Option<f64>) -> PyResult<Self> {
        let parsed = parse_coords(coords)?;
        let inner = match block_size {
            Some(bs) => smelt_ml::prelude::SpatialBlockCV::with_block_size(n_folds, parsed, bs),
            None => smelt_ml::prelude::SpatialBlockCV::new(n_folds, parsed),
        };
        Ok(Self { inner })
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

    /// O(n^2) in `n_samples` (every fold rechecks every point against the
    /// buffer distance) and typically called with `n_folds = n_samples` for
    /// spatial leave-one-out — release the GIL for the computation.
    fn splits(&self, py: Python<'_>, n_samples: usize) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
        use smelt_ml::prelude::Resample;
        py.allow_threads(|| self.inner.splits(n_samples))
            .map_err(smelt_err)
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

// ── TimeSeriesCV ───────────────────────────────────────────────────────

#[pyclass]
pub(crate) struct TimeSeriesCV {
    inner: smelt_ml::prelude::TimeSeriesCV,
}

#[pymethods]
impl TimeSeriesCV {
    /// Rolling-origin (walk-forward) CV for time-ordered data: every split
    /// trains strictly on the past and tests on the next `horizon` samples.
    /// Rows are assumed sorted by time (index 0 = oldest).
    ///
    /// `min_train_size`/`step` default to `horizon` (contiguous test
    /// windows); `max_window` switches to a sliding training window of that
    /// size; `gap` leaves an embargo between train end and test start.
    #[new]
    #[pyo3(signature = (horizon, min_train_size=None, step=None, max_window=None, gap=0))]
    fn new(
        horizon: usize,
        min_train_size: Option<usize>,
        step: Option<usize>,
        max_window: Option<usize>,
        gap: usize,
    ) -> Self {
        let mut inner = smelt_ml::prelude::TimeSeriesCV::new(horizon).with_gap(gap);
        if let Some(m) = min_train_size {
            inner = inner.with_min_train_size(m);
        }
        if let Some(s) = step {
            inner = inner.with_step(s);
        }
        if let Some(w) = max_window {
            inner = inner.with_sliding_window(w);
        }
        Self { inner }
    }

    fn splits(&self, n_samples: usize) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
        use smelt_ml::prelude::Resample;
        self.inner.splits(n_samples).map_err(smelt_err)
    }
}
