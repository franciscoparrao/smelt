//! Preprocessing wrappers.

use crate::common::{extract_class_labels, parse_coords, smelt_err, to_array2};
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

// ── StandardScaler ─────────────────────────────────────────────────────

#[pyclass]
pub(crate) struct StandardScaler {
    inner: smelt_ml::prelude::StandardScaler,
}

#[pymethods]
impl StandardScaler {
    #[new]
    fn new() -> Self {
        Self {
            inner: smelt_ml::prelude::StandardScaler::new(),
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        use smelt_ml::prelude::Transformer;
        self.inner.fit(&to_array2(x)).map_err(smelt_err)
    }

    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        use smelt_ml::prelude::Transformer;
        let r = self.inner.transform(&to_array2(x)).map_err(smelt_err)?;
        Ok(PyArray2::from_owned_array(py, r))
    }

    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        use smelt_ml::prelude::Transformer;
        let r = self.inner.fit_transform(&to_array2(x)).map_err(smelt_err)?;
        Ok(PyArray2::from_owned_array(py, r))
    }
}

// ── Smote ────────────────────────────────────────────────────────────

#[pyclass]
pub(crate) struct Smote {
    k_neighbors: usize,
    seed: u64,
}

#[pymethods]
impl Smote {
    #[new]
    #[pyo3(signature = (k_neighbors=5, seed=42))]
    fn new(k_neighbors: usize, seed: u64) -> Self {
        Self { k_neighbors, seed }
    }

    /// Oversample minority classes by interpolating between a minority
    /// instance and one of its k-nearest same-class neighbors. `y` must be
    /// non-negative integer class labels. Returns `(x_balanced, y_balanced)`.
    fn balance<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
        y: &Bound<'_, PyAny>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Vec<usize>)> {
        use smelt_ml::task::Task;
        let target = extract_class_labels(y)?;
        let task = smelt_ml::task::ClassificationTask::new("py", to_array2(x), target)
            .map_err(smelt_err)?;
        let smote = smelt_ml::prelude::Smote::new()
            .with_k_neighbors(self.k_neighbors)
            .with_seed(self.seed);
        let balanced = py.allow_threads(|| smote.balance(&task)).map_err(smelt_err)?;
        Ok((
            PyArray2::from_owned_array(py, balanced.features().clone()),
            balanced.target().to_vec(),
        ))
    }
}

// ── SpatialSmote ─────────────────────────────────────────────────────

#[pyclass]
pub(crate) struct SpatialSmote {
    k_neighbors: usize,
    seed: u64,
    max_spatial_distance: Option<f64>,
}

#[pymethods]
impl SpatialSmote {
    /// `max_spatial_distance`: when set, restricts candidate neighbors to
    /// those within this spatial distance -- a minority sample with no
    /// same-class neighbor within the cutoff contributes no synthetic
    /// samples for that draw (graceful degradation, not an error).
    #[new]
    #[pyo3(signature = (k_neighbors=5, seed=42, max_spatial_distance=None))]
    fn new(k_neighbors: usize, seed: u64, max_spatial_distance: Option<f64>) -> Self {
        Self {
            k_neighbors,
            seed,
            max_spatial_distance,
        }
    }

    /// Oversample minority classes, respecting spatial proximity. `coords`
    /// is an (N, 2) array-like of (x, y) per sample. `y` must be
    /// non-negative integer class labels. Returns `(x_balanced, y_balanced,
    /// coords_balanced)` -- `coords_balanced` includes an interpolated
    /// coordinate for each synthetic sample, since raw arrays carry no
    /// notion of spatial location on their own.
    fn balance<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
        y: &Bound<'_, PyAny>,
        coords: &Bound<'_, PyAny>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Vec<usize>, Bound<'py, PyArray2<f64>>)> {
        use smelt_ml::task::Task;
        let target = extract_class_labels(y)?;
        let features = to_array2(x);
        let parsed_coords = parse_coords(coords)?;
        if parsed_coords.len() != features.nrows() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "coords length ({}) must match number of samples ({})",
                parsed_coords.len(),
                features.nrows()
            )));
        }
        let task =
            smelt_ml::task::ClassificationTask::new("py", features, target).map_err(smelt_err)?;

        let mut smote = smelt_ml::prelude::SpatialSmote::new()
            .with_k_neighbors(self.k_neighbors)
            .with_seed(self.seed);
        if let Some(d) = self.max_spatial_distance {
            smote = smote.with_max_spatial_distance(d);
        }
        let (balanced, new_coords) = py
            .allow_threads(|| smote.balance(&task, &parsed_coords))
            .map_err(smelt_err)?;

        let n = new_coords.len();
        let flat: Vec<f64> = new_coords.iter().flat_map(|&(x, y)| [x, y]).collect();
        let coord_arr = ndarray::Array2::from_shape_vec((n, 2), flat)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok((
            PyArray2::from_owned_array(py, balanced.features().clone()),
            balanced.target().to_vec(),
            PyArray2::from_owned_array(py, coord_arr),
        ))
    }
}

