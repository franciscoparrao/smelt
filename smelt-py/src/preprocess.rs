//! Preprocessing wrappers.

use crate::common::{smelt_err, to_array2};
use numpy::{PyArray2, PyReadonlyArray2};
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

