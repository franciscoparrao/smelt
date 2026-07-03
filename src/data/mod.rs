//! Data loading: CSV and (optionally) Parquet file import into Tasks.

use crate::preprocess::LabelEncoder;
use crate::task::{ClassificationTask, RegressionTask};
use crate::{Result, SmeltError};
use ndarray::Array2;
use std::path::Path;

#[cfg(feature = "parquet")]
mod parquet;
#[cfg(feature = "parquet")]
pub use parquet::ParquetLoader;

/// Loads a CSV file into a classification or regression Task.
///
/// Missing values (empty cells, `NA`, `NaN`, `null`, `?`, `N/A` — case
/// insensitive) are loaded as `f64::NAN`. Feature columns whose non-missing
/// values are not all numeric are auto-detected as categorical: their values
/// are label-encoded to integer codes and the column is marked
/// `FeatureType::Categorical` on the resulting task. Use [`CsvLoader::categorical`]
/// to force specific columns to be treated as categorical.
///
/// # Examples
///
/// ```no_run
/// use smelt_ml::data::CsvLoader;
///
/// let task = CsvLoader::from_path("iris.csv")
///     .target("species")
///     .load_classif()
///     .unwrap();
/// ```
pub struct CsvLoader {
    path: String,
    target_col: Option<String>,
    delimiter: u8,
    max_rows: Option<usize>,
    categorical_cols: Vec<String>,
}

/// True when a CSV cell should be loaded as a missing value (`f64::NAN`).
fn is_missing(s: &str) -> bool {
    let t = s.trim();
    t.is_empty()
        || t == "?"
        || t.eq_ignore_ascii_case("na")
        || t.eq_ignore_ascii_case("nan")
        || t.eq_ignore_ascii_case("null")
        || t.eq_ignore_ascii_case("n/a")
}

impl CsvLoader {
    pub fn from_path(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_string_lossy().to_string(),
            target_col: None,
            delimiter: b',',
            max_rows: None,
            categorical_cols: Vec::new(),
        }
    }

    pub fn target(mut self, col: &str) -> Self {
        self.target_col = Some(col.to_string());
        self
    }

    /// Force these feature columns to be treated as categorical (label-encoded)
    /// even if their values parse as numbers. String columns are auto-detected
    /// as categorical without needing this.
    pub fn categorical(mut self, cols: &[&str]) -> Self {
        self.categorical_cols = cols.iter().map(|c| c.to_string()).collect();
        self
    }

    pub fn delimiter(mut self, d: u8) -> Self {
        self.delimiter = d;
        self
    }

    /// Limit the number of rows to read (prevents OOM on large files).
    pub fn max_rows(mut self, n: usize) -> Self {
        self.max_rows = Some(n);
        self
    }

    fn read_raw(&self) -> Result<(Vec<String>, Vec<Vec<String>>)> {
        let mut rdr = csv::ReaderBuilder::new()
            .delimiter(self.delimiter)
            .from_path(&self.path)
            .map_err(|e| SmeltError::Csv(e.to_string()))?;

        let headers: Vec<String> = rdr
            .headers()
            .map_err(|e| SmeltError::Csv(e.to_string()))?
            .iter()
            .map(|h| h.to_string())
            .collect();

        let mut rows = Vec::new();
        for result in rdr.records() {
            if let Some(max) = self.max_rows
                && rows.len() >= max
            {
                break;
            }
            let record = result.map_err(|e| SmeltError::Csv(e.to_string()))?;
            let row: Vec<String> = record.iter().map(|f| f.to_string()).collect();
            rows.push(row);
        }

        if rows.is_empty() {
            return Err(SmeltError::EmptyDataset);
        }

        Ok((headers, rows))
    }

    fn find_target_col(&self, headers: &[String]) -> Result<usize> {
        let target_name = self
            .target_col
            .as_ref()
            .ok_or_else(|| SmeltError::Other("target column not specified".into()))?;
        headers
            .iter()
            .position(|h| h == target_name)
            .ok_or_else(|| SmeltError::FeatureNotFound(target_name.clone()))
    }

    /// Parse feature columns with missing-value and categorical handling.
    /// Returns (feature_names, features, categorical_column_indices).
    /// A column is categorical when it is listed in `self.categorical_cols` or
    /// when any non-missing value fails to parse as a number; its values are
    /// label-encoded to integer codes. Missing cells become NaN either way.
    fn parse_features(
        &self,
        headers: &[String],
        rows: &[Vec<String>],
        target_idx: usize,
    ) -> Result<(Vec<String>, Array2<f64>, Vec<usize>)> {
        for c in &self.categorical_cols {
            if !headers.contains(c) {
                return Err(SmeltError::FeatureNotFound(c.clone()));
            }
        }

        let feature_cols: Vec<usize> = (0..headers.len()).filter(|&i| i != target_idx).collect();
        let feature_names: Vec<String> =
            feature_cols.iter().map(|&i| headers[i].clone()).collect();

        let n_samples = rows.len();
        let mut features = Array2::zeros((n_samples, feature_cols.len()));
        let mut cat_indices = Vec::new();

        for (j, &col) in feature_cols.iter().enumerate() {
            let forced_cat = self.categorical_cols.contains(&headers[col]);
            let all_numeric = rows
                .iter()
                .all(|r| is_missing(&r[col]) || r[col].trim().parse::<f64>().is_ok());

            if all_numeric && !forced_cat {
                for (i, row) in rows.iter().enumerate() {
                    features[[i, j]] = if is_missing(&row[col]) {
                        f64::NAN
                    } else {
                        row[col].trim().parse::<f64>().unwrap()
                    };
                }
            } else {
                let present: Vec<&str> = rows
                    .iter()
                    .map(|r| r[col].trim())
                    .filter(|s| !is_missing(s))
                    .collect();
                let encoder = LabelEncoder::fit(&present);
                for (i, row) in rows.iter().enumerate() {
                    let cell = row[col].trim();
                    features[[i, j]] = if is_missing(cell) {
                        f64::NAN
                    } else {
                        encoder.encode(&[cell])?[0] as f64
                    };
                }
                cat_indices.push(j);
            }
        }

        Ok((feature_names, features, cat_indices))
    }

    /// Load as a classification task. Target values are parsed as usize or auto-encoded.
    pub fn load_classif(&self) -> Result<ClassificationTask> {
        let (headers, rows) = self.read_raw()?;
        let target_idx = self.find_target_col(&headers)?;

        // Try parsing target as usize first, fall back to label encoding
        let target_strings: Vec<String> = rows.iter().map(|r| r[target_idx].clone()).collect();

        let target: Vec<usize> = match target_strings[0].parse::<usize>() {
            Ok(_) => target_strings
                .iter()
                .map(|s| {
                    s.parse::<usize>().map_err(|_| {
                        SmeltError::Csv(format!("cannot parse target '{}' as integer", s))
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            Err(_) => {
                let encoder = LabelEncoder::fit(
                    &target_strings
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>(),
                );
                encoder.encode(
                    &target_strings
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>(),
                )?
            }
        };

        let (feature_names, features, cat_indices) =
            self.parse_features(&headers, &rows, target_idx)?;

        ClassificationTask::new("csv", features, target)?
            .with_feature_names(feature_names)?
            .with_categorical_features(&cat_indices)
    }

    /// Load as a regression task. Target values must be numeric.
    pub fn load_regress(&self) -> Result<RegressionTask> {
        let (headers, rows) = self.read_raw()?;
        let target_idx = self.find_target_col(&headers)?;

        let target: Vec<f64> = rows
            .iter()
            .map(|r| {
                r[target_idx].parse::<f64>().map_err(|_| {
                    SmeltError::Csv(format!("cannot parse target '{}' as number", r[target_idx]))
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let (feature_names, features, cat_indices) =
            self.parse_features(&headers, &rows, target_idx)?;

        RegressionTask::new("csv", features, target)?
            .with_feature_names(feature_names)?
            .with_categorical_features(&cat_indices)
    }
}
