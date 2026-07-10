//! Data loading: CSV and (behind the `parquet` Cargo feature) Parquet files.
//! Returns plain `(x, y, feature_names)` tuples rather than a `Task` object
//! -- there's no Python-facing `Task` type, and every learner wrapper's
//! `fit(x, y)` already expects exactly this shape.

use crate::common::smelt_err;
use numpy::PyArray2;
use pyo3::prelude::*;
use smelt_ml::task::Task;

#[pyclass]
pub(crate) struct CsvLoader {
    path: String,
    target: Option<String>,
    delimiter: u8,
    max_rows: Option<usize>,
    categorical: Vec<String>,
}

#[pymethods]
impl CsvLoader {
    #[new]
    #[pyo3(signature = (path, target, delimiter=",".to_string(), max_rows=None, categorical=None))]
    fn new(
        path: String,
        target: String,
        delimiter: String,
        max_rows: Option<usize>,
        categorical: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let delimiter_byte = delimiter.as_bytes();
        if delimiter_byte.len() != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "delimiter must be a single ASCII character",
            ));
        }
        Ok(Self {
            path,
            target: Some(target),
            delimiter: delimiter_byte[0],
            max_rows,
            categorical: categorical.unwrap_or_default(),
        })
    }

    /// Load as a classification task. Returns `(x, y, feature_names)` where
    /// `y` is integer class labels.
    fn load_classif<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Vec<usize>, Vec<String>)> {
        let task = self.build().load_classif().map_err(smelt_err)?;
        Ok((
            PyArray2::from_owned_array(py, task.features().clone()),
            task.target().to_vec(),
            task.feature_names().to_vec(),
        ))
    }

    /// Load as a regression task. Returns `(x, y, feature_names)` where `y`
    /// is continuous values.
    fn load_regress<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Vec<f64>, Vec<String>)> {
        let task = self.build().load_regress().map_err(smelt_err)?;
        Ok((
            PyArray2::from_owned_array(py, task.features().clone()),
            task.target().to_vec(),
            task.feature_names().to_vec(),
        ))
    }
}

impl CsvLoader {
    fn build(&self) -> smelt_ml::data::CsvLoader {
        let cats: Vec<&str> = self.categorical.iter().map(String::as_str).collect();
        let mut loader = smelt_ml::data::CsvLoader::from_path(&self.path)
            .delimiter(self.delimiter)
            .categorical(&cats);
        if let Some(t) = &self.target {
            loader = loader.target(t);
        }
        if let Some(n) = self.max_rows {
            loader = loader.max_rows(n);
        }
        loader
    }
}

#[cfg(feature = "parquet")]
#[pyclass]
pub(crate) struct ParquetLoader {
    path: String,
    target: Option<String>,
    categorical: Vec<String>,
}

#[cfg(feature = "parquet")]
#[pymethods]
impl ParquetLoader {
    #[new]
    #[pyo3(signature = (path, target, categorical=None))]
    fn new(path: String, target: String, categorical: Option<Vec<String>>) -> Self {
        Self {
            path,
            target: Some(target),
            categorical: categorical.unwrap_or_default(),
        }
    }

    /// Load as a classification task. Returns `(x, y, feature_names)` where
    /// `y` is integer class labels.
    fn load_classif<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Vec<usize>, Vec<String>)> {
        let task = self.build().load_classif().map_err(smelt_err)?;
        Ok((
            PyArray2::from_owned_array(py, task.features().clone()),
            task.target().to_vec(),
            task.feature_names().to_vec(),
        ))
    }

    /// Load as a regression task. Returns `(x, y, feature_names)` where `y`
    /// is continuous values.
    fn load_regress<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Vec<f64>, Vec<String>)> {
        let task = self.build().load_regress().map_err(smelt_err)?;
        Ok((
            PyArray2::from_owned_array(py, task.features().clone()),
            task.target().to_vec(),
            task.feature_names().to_vec(),
        ))
    }
}

#[cfg(feature = "parquet")]
impl ParquetLoader {
    fn build(&self) -> smelt_ml::data::ParquetLoader {
        let cats: Vec<&str> = self.categorical.iter().map(String::as_str).collect();
        let mut loader = smelt_ml::data::ParquetLoader::from_path(&self.path).categorical(&cats);
        if let Some(t) = &self.target {
            loader = loader.target(t);
        }
        loader
    }
}
