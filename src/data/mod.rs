//! Data loading: CSV file import into Tasks.

use std::path::Path;
use ndarray::Array2;
use crate::task::{ClassificationTask, RegressionTask};
use crate::preprocess::LabelEncoder;
use crate::{SmeltError, Result};

/// Loads a CSV file into a classification or regression Task.
///
/// # Examples
///
/// ```no_run
/// use smelt::data::CsvLoader;
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
}

impl CsvLoader {
    pub fn from_path(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_string_lossy().to_string(),
            target_col: None,
            delimiter: b',',
        }
    }

    pub fn target(mut self, col: &str) -> Self {
        self.target_col = Some(col.to_string());
        self
    }

    pub fn delimiter(mut self, d: u8) -> Self {
        self.delimiter = d;
        self
    }

    fn read_raw(&self) -> Result<(Vec<String>, Vec<Vec<String>>)> {
        let mut rdr = csv::ReaderBuilder::new()
            .delimiter(self.delimiter)
            .from_path(&self.path)
            .map_err(|e| SmeltError::Csv(e.to_string()))?;

        let headers: Vec<String> = rdr.headers()
            .map_err(|e| SmeltError::Csv(e.to_string()))?
            .iter()
            .map(|h| h.to_string())
            .collect();

        let mut rows = Vec::new();
        for result in rdr.records() {
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
        let target_name = self.target_col.as_ref()
            .ok_or_else(|| SmeltError::Other("target column not specified".into()))?;
        headers.iter().position(|h| h == target_name)
            .ok_or_else(|| SmeltError::FeatureNotFound(target_name.clone()))
    }

    /// Load as a classification task. Target values are parsed as usize or auto-encoded.
    pub fn load_classif(&self) -> Result<ClassificationTask> {
        let (headers, rows) = self.read_raw()?;
        let target_idx = self.find_target_col(&headers)?;

        // Try parsing target as usize first, fall back to label encoding
        let target_strings: Vec<String> = rows.iter()
            .map(|r| r[target_idx].clone())
            .collect();

        let target: Vec<usize> = match target_strings[0].parse::<usize>() {
            Ok(_) => {
                target_strings.iter()
                    .map(|s| s.parse::<usize>()
                        .map_err(|_| SmeltError::Csv(format!("cannot parse target '{}' as integer", s))))
                    .collect::<Result<Vec<_>>>()?
            }
            Err(_) => {
                let encoder = LabelEncoder::fit(&target_strings.iter().map(|s| s.as_str()).collect::<Vec<_>>());
                encoder.encode(&target_strings.iter().map(|s| s.as_str()).collect::<Vec<_>>())?
            }
        };

        // Parse feature columns
        let feature_names: Vec<String> = headers.iter().enumerate()
            .filter(|&(i, _)| i != target_idx)
            .map(|(_, h)| h.clone())
            .collect();

        let n_features = feature_names.len();
        let n_samples = rows.len();
        let mut features = Array2::zeros((n_samples, n_features));

        for (i, row) in rows.iter().enumerate() {
            let mut j = 0;
            for (col, val) in row.iter().enumerate() {
                if col == target_idx {
                    continue;
                }
                features[[i, j]] = val.parse::<f64>()
                    .map_err(|_| SmeltError::Csv(
                        format!("cannot parse '{}' as number at row {}, col '{}'", val, i + 1, headers[col])
                    ))?;
                j += 1;
            }
        }

        ClassificationTask::new("csv", features, target)?
            .with_feature_names(feature_names)
    }

    /// Load as a regression task. Target values must be numeric.
    pub fn load_regress(&self) -> Result<RegressionTask> {
        let (headers, rows) = self.read_raw()?;
        let target_idx = self.find_target_col(&headers)?;

        let target: Vec<f64> = rows.iter()
            .map(|r| r[target_idx].parse::<f64>()
                .map_err(|_| SmeltError::Csv(
                    format!("cannot parse target '{}' as number", r[target_idx])
                )))
            .collect::<Result<Vec<_>>>()?;

        let feature_names: Vec<String> = headers.iter().enumerate()
            .filter(|&(i, _)| i != target_idx)
            .map(|(_, h)| h.clone())
            .collect();

        let n_features = feature_names.len();
        let n_samples = rows.len();
        let mut features = Array2::zeros((n_samples, n_features));

        for (i, row) in rows.iter().enumerate() {
            let mut j = 0;
            for (col, val) in row.iter().enumerate() {
                if col == target_idx {
                    continue;
                }
                features[[i, j]] = val.parse::<f64>()
                    .map_err(|_| SmeltError::Csv(
                        format!("cannot parse '{}' as number at row {}, col '{}'", val, i + 1, headers[col])
                    ))?;
                j += 1;
            }
        }

        RegressionTask::new("csv", features, target)?
            .with_feature_names(feature_names)
    }
}
