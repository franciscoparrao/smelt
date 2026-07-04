//! Parquet file import into Tasks. Requires the `parquet` feature (pulls in
//! `polars` as an optional dependency, per the audit's recommendation to gate
//! Arrow/Parquet support behind a feature rather than force it on every
//! consumer of this crate).

use crate::preprocess::LabelEncoder;
use crate::task::{ClassificationTask, RegressionTask};
use crate::{Result, SmeltError};
use ndarray::Array2;
use polars::prelude::*;
use std::path::Path;

fn polars_err(e: PolarsError) -> SmeltError {
    SmeltError::Parquet(e.to_string())
}

/// Loads a Parquet file into a classification or regression Task.
///
/// Unlike [`crate::data::CsvLoader`], columns are typed by the Parquet
/// schema instead of sniffed from strings: numeric/boolean columns are cast
/// straight to `f64` (nulls become `f64::NAN`), and string columns are
/// label-encoded and marked `FeatureType::Categorical`, same as CSV's
/// auto-detection. Use [`ParquetLoader::categorical`] to force a numeric
/// column to be treated as categorical instead.
///
/// # Examples
///
/// ```no_run
/// use smelt_ml::data::ParquetLoader;
///
/// let task = ParquetLoader::from_path("iris.parquet")
///     .target("species")
///     .load_classif()
///     .unwrap();
/// ```
pub struct ParquetLoader {
    path: String,
    target_col: Option<String>,
    categorical_cols: Vec<String>,
}

impl ParquetLoader {
    pub fn from_path(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_string_lossy().to_string(),
            target_col: None,
            categorical_cols: Vec::new(),
        }
    }

    pub fn target(mut self, col: &str) -> Self {
        self.target_col = Some(col.to_string());
        self
    }

    /// Force these feature columns to be treated as categorical (label-encoded)
    /// even if their values are numeric. String columns are auto-detected as
    /// categorical without needing this.
    pub fn categorical(mut self, cols: &[&str]) -> Self {
        self.categorical_cols = cols.iter().map(|c| c.to_string()).collect();
        self
    }

    fn read_df(&self) -> Result<DataFrame> {
        let file = std::fs::File::open(&self.path)?;
        ParquetReader::new(file).finish().map_err(polars_err)
    }

    fn find_target_col(&self, df: &DataFrame) -> Result<usize> {
        let target_name = self
            .target_col
            .as_ref()
            .ok_or_else(|| SmeltError::Other("target column not specified".into()))?;
        df.get_column_names()
            .iter()
            .position(|h| h.as_str() == target_name)
            .ok_or_else(|| SmeltError::FeatureNotFound(target_name.clone()))
    }

    /// Cast a column to `f64`, mapping nulls to `NaN`. `n` is the row count
    /// (`DataFrame::height`), passed in rather than read off the chunked
    /// array to avoid pulling in polars' internal `Length` trait just for
    /// this.
    fn column_to_f64(column: &Column, n: usize) -> Result<Vec<f64>> {
        let casted = column.cast(&DataType::Float64).map_err(polars_err)?;
        let ca = casted.f64().map_err(polars_err)?;
        Ok((0..n).map(|i| ca.get(i).unwrap_or(f64::NAN)).collect())
    }

    /// Cast a column to string, mapping nulls to `None`. See `column_to_f64`
    /// for why `n` is passed in explicitly.
    fn column_to_strings(column: &Column, n: usize) -> Result<Vec<Option<String>>> {
        let casted = column.cast(&DataType::String).map_err(polars_err)?;
        let ca = casted.str().map_err(polars_err)?;
        Ok((0..n).map(|i| ca.get(i).map(str::to_string)).collect())
    }

    /// Parse feature columns with missing-value and categorical handling.
    /// Returns (feature_names, features, categorical_column_indices). A
    /// column is categorical when it is listed in `self.categorical_cols` or
    /// its Parquet dtype is neither numeric nor boolean; its values are
    /// label-encoded to integer codes. Nulls become NaN either way.
    fn parse_features(
        &self,
        df: &DataFrame,
        target_idx: usize,
    ) -> Result<(Vec<String>, Array2<f64>, Vec<usize>)> {
        let names = df.get_column_names();
        for c in &self.categorical_cols {
            if !names.iter().any(|h| h.as_str() == c) {
                return Err(SmeltError::FeatureNotFound(c.clone()));
            }
        }

        let columns = df.columns();
        let feature_cols: Vec<usize> = (0..columns.len()).filter(|&i| i != target_idx).collect();
        let feature_names: Vec<String> = feature_cols
            .iter()
            .map(|&i| columns[i].name().to_string())
            .collect();

        let n_samples = df.height();
        let mut features = Array2::zeros((n_samples, feature_cols.len()));
        let mut cat_indices = Vec::new();

        for (j, &ci) in feature_cols.iter().enumerate() {
            let column = &columns[ci];
            let forced_cat = self
                .categorical_cols
                .iter()
                .any(|c| c.as_str() == column.name().as_str());
            let is_numeric = column.dtype().is_numeric() || column.dtype().is_bool();

            if is_numeric && !forced_cat {
                let values = Self::column_to_f64(column, n_samples)?;
                for (i, v) in values.into_iter().enumerate() {
                    features[[i, j]] = v;
                }
            } else {
                let strings = Self::column_to_strings(column, n_samples)?;
                let present: Vec<&str> =
                    strings.iter().filter_map(|s| s.as_deref()).collect();
                let encoder = LabelEncoder::fit(&present);
                for (i, cell) in strings.iter().enumerate() {
                    features[[i, j]] = match cell {
                        Some(v) => encoder.encode(&[v.as_str()])?[0] as f64,
                        None => f64::NAN,
                    };
                }
                cat_indices.push(j);
            }
        }

        Ok((feature_names, features, cat_indices))
    }

    /// Load as a classification task. A numeric target is truncated to
    /// `usize` (must be non-negative integers); a string target is
    /// label-encoded.
    pub fn load_classif(&self) -> Result<ClassificationTask> {
        let df = self.read_df()?;
        let target_idx = self.find_target_col(&df)?;
        let target_col = &df.columns()[target_idx];
        let n_samples = df.height();

        let target: Vec<usize> = if target_col.dtype().is_numeric() {
            Self::column_to_f64(target_col, n_samples)?
                .into_iter()
                .map(|v| {
                    if v.is_nan() || v < 0.0 || v.fract() != 0.0 {
                        Err(SmeltError::Parquet(format!(
                            "cannot parse target '{v}' as a non-negative integer"
                        )))
                    } else {
                        Ok(v as usize)
                    }
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            let strings = Self::column_to_strings(target_col, n_samples)?;
            let owned: Vec<&str> = strings
                .iter()
                .map(|s| {
                    s.as_deref().ok_or_else(|| {
                        SmeltError::Parquet(
                            "target column contains a null value".into(),
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let encoder = LabelEncoder::fit(&owned);
            encoder.encode(&owned)?
        };

        let (feature_names, features, cat_indices) = self.parse_features(&df, target_idx)?;

        ClassificationTask::new("parquet", features, target)?
            .with_feature_names(feature_names)?
            .with_categorical_features(&cat_indices)
    }

    /// Load as a regression task. Target values must be numeric and non-null.
    pub fn load_regress(&self) -> Result<RegressionTask> {
        let df = self.read_df()?;
        let target_idx = self.find_target_col(&df)?;
        let target_col = &df.columns()[target_idx];
        let n_samples = df.height();

        let target: Vec<f64> = Self::column_to_f64(target_col, n_samples)?
            .into_iter()
            .map(|v| {
                if v.is_nan() {
                    Err(SmeltError::Parquet("target column contains a null/NaN value".into()))
                } else {
                    Ok(v)
                }
            })
            .collect::<Result<Vec<_>>>()?;

        let (feature_names, features, cat_indices) = self.parse_features(&df, target_idx)?;

        RegressionTask::new("parquet", features, target)?
            .with_feature_names(feature_names)?
            .with_categorical_features(&cat_indices)
    }
}
