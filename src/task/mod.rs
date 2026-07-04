//! Tasks define the problem: data + target + type (classification/regression).
//!
//! Inspired by mlr3's `Task` system.

use crate::{Result, SmeltError};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
// validate module used by learner predict methods

/// Type of a feature column.
///
/// Learners that understand categorical features (the boosting engines) read
/// this to choose categorical split finding / target encoding; every other
/// learner treats the integer codes as ordinary numeric values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureType {
    /// Continuous numeric feature (the default).
    Numeric,
    /// Categorical feature stored as integer codes `0..n_categories`.
    /// NaN is allowed and means "missing category".
    Categorical {
        /// Number of distinct categories (max code + 1).
        n_categories: usize,
    },
}

/// Validate that `columns` can be marked categorical in `features` and return
/// the per-column feature types: codes must be non-negative integers (NaN is
/// allowed as missing).
fn build_feature_types(features: &Array2<f64>, columns: &[usize]) -> Result<Vec<FeatureType>> {
    let mut types = vec![FeatureType::Numeric; features.ncols()];
    for &col in columns {
        if col >= features.ncols() {
            return Err(SmeltError::InvalidParameter(format!(
                "categorical feature index {col} out of range ({} features)",
                features.ncols()
            )));
        }
        let mut max_code = 0usize;
        for (row, &v) in features.column(col).iter().enumerate() {
            if v.is_nan() {
                continue;
            }
            if v < 0.0 || v.fract() != 0.0 {
                return Err(SmeltError::InvalidParameter(format!(
                    "categorical feature {col} has non-integer or negative code {v} at row {row}"
                )));
            }
            max_code = max_code.max(v as usize);
        }
        types[col] = FeatureType::Categorical {
            n_categories: max_code + 1,
        };
    }
    Ok(types)
}

/// Core trait for all task types.
pub trait Task {
    /// Task identifier.
    fn id(&self) -> &str;
    /// Feature matrix (n_samples × n_features).
    fn features(&self) -> &Array2<f64>;
    /// Number of samples.
    fn n_samples(&self) -> usize {
        self.features().nrows()
    }
    /// Number of features.
    fn n_features(&self) -> usize {
        self.features().ncols()
    }
    /// Feature names.
    fn feature_names(&self) -> &[String];
    /// Per-column feature types (Numeric unless marked categorical).
    fn feature_types(&self) -> &[FeatureType];
    /// Indices of the categorical feature columns.
    fn categorical_features(&self) -> Vec<usize> {
        self.feature_types()
            .iter()
            .enumerate()
            .filter(|(_, t)| matches!(t, FeatureType::Categorical { .. }))
            .map(|(i, _)| i)
            .collect()
    }
}

/// Classification task with discrete target labels.
#[derive(Debug)]
pub struct ClassificationTask {
    id: String,
    features: Array2<f64>,
    target: Vec<usize>,
    feature_names: Vec<String>,
    class_names: Vec<String>,
    feature_types: Vec<FeatureType>,
}

impl ClassificationTask {
    /// Create a classification task from a feature matrix and integer class
    /// labels; class names default to `"class_0"`, `"class_1"`, etc.
    pub fn new(id: impl Into<String>, features: Array2<f64>, target: Vec<usize>) -> Result<Self> {
        if features.nrows() == 0 {
            return Err(SmeltError::EmptyDataset);
        }
        if features.nrows() != target.len() {
            return Err(SmeltError::DimensionMismatch {
                expected: features.nrows(),
                got: target.len(),
            });
        }
        if features.ncols() == 0 {
            return Err(SmeltError::InvalidParameter(
                "features must have at least 1 column".into(),
            ));
        }
        let n_features = features.ncols();
        let n_classes = target.iter().copied().max().unwrap_or(0) + 1;
        Ok(Self {
            id: id.into(),
            feature_names: (0..n_features).map(|i| format!("x{i}")).collect(),
            feature_types: vec![FeatureType::Numeric; n_features],
            features,
            target,
            class_names: (0..n_classes).map(|i| format!("class_{i}")).collect(),
        })
    }

    /// Mark feature columns as categorical. Values in those columns must be
    /// non-negative integer codes (NaN allowed as missing category).
    ///
    /// Recomputes `n_categories` per column from the codes actually present
    /// in `self.features` (the max code + 1). If this task is a subset of a
    /// larger dataset (e.g. a CV fold), this can under-count categories the
    /// subset happens not to contain — use [`Self::with_feature_types`] to
    /// copy the exact types (and `n_categories`) from the parent task instead.
    pub fn with_categorical_features(mut self, columns: &[usize]) -> Result<Self> {
        self.feature_types = build_feature_types(&self.features, columns)?;
        Ok(self)
    }

    /// Set feature types directly (e.g. copied from another task's
    /// [`Task::feature_types`]), bypassing the from-data recomputation
    /// `with_categorical_features` does. Must match the number of feature
    /// columns.
    pub fn with_feature_types(mut self, types: Vec<FeatureType>) -> Result<Self> {
        if types.len() != self.features.ncols() {
            return Err(SmeltError::DimensionMismatch {
                expected: self.features.ncols(),
                got: types.len(),
            });
        }
        self.feature_types = types;
        Ok(self)
    }

    /// Set custom feature names; must match the number of feature columns.
    pub fn with_feature_names(mut self, names: Vec<String>) -> Result<Self> {
        if names.len() != self.features.ncols() {
            return Err(SmeltError::DimensionMismatch {
                expected: self.features.ncols(),
                got: names.len(),
            });
        }
        self.feature_names = names;
        Ok(self)
    }

    /// Set custom class names, indexed by class label.
    pub fn with_class_names(mut self, names: Vec<String>) -> Self {
        self.class_names = names;
        self
    }

    /// Integer target labels, one per sample.
    pub fn target(&self) -> &[usize] {
        &self.target
    }
    /// Number of distinct classes.
    pub fn n_classes(&self) -> usize {
        self.class_names.len()
    }
    /// Class names, indexed by class label.
    pub fn class_names(&self) -> &[String] {
        &self.class_names
    }
}

impl Task for ClassificationTask {
    fn id(&self) -> &str {
        &self.id
    }
    fn features(&self) -> &Array2<f64> {
        &self.features
    }
    fn feature_names(&self) -> &[String] {
        &self.feature_names
    }
    fn feature_types(&self) -> &[FeatureType] {
        &self.feature_types
    }
}

/// Regression task with continuous target values.
#[derive(Debug)]
pub struct RegressionTask {
    id: String,
    features: Array2<f64>,
    target: Vec<f64>,
    feature_names: Vec<String>,
    feature_types: Vec<FeatureType>,
}

impl RegressionTask {
    /// Create a regression task from a feature matrix and continuous target
    /// values.
    pub fn new(id: impl Into<String>, features: Array2<f64>, target: Vec<f64>) -> Result<Self> {
        if features.nrows() == 0 {
            return Err(SmeltError::EmptyDataset);
        }
        if features.nrows() != target.len() {
            return Err(SmeltError::DimensionMismatch {
                expected: features.nrows(),
                got: target.len(),
            });
        }
        if features.ncols() == 0 {
            return Err(SmeltError::InvalidParameter(
                "features must have at least 1 column".into(),
            ));
        }
        let n_features = features.ncols();
        Ok(Self {
            id: id.into(),
            feature_names: (0..n_features).map(|i| format!("x{i}")).collect(),
            feature_types: vec![FeatureType::Numeric; n_features],
            features,
            target,
        })
    }

    /// Mark feature columns as categorical. Values in those columns must be
    /// non-negative integer codes (NaN allowed as missing category).
    ///
    /// Recomputes `n_categories` per column from the codes actually present
    /// in `self.features` (the max code + 1). If this task is a subset of a
    /// larger dataset (e.g. a CV fold), this can under-count categories the
    /// subset happens not to contain — use [`Self::with_feature_types`] to
    /// copy the exact types (and `n_categories`) from the parent task instead.
    pub fn with_categorical_features(mut self, columns: &[usize]) -> Result<Self> {
        self.feature_types = build_feature_types(&self.features, columns)?;
        Ok(self)
    }

    /// Set feature types directly (e.g. copied from another task's
    /// [`Task::feature_types`]), bypassing the from-data recomputation
    /// `with_categorical_features` does. Must match the number of feature
    /// columns.
    pub fn with_feature_types(mut self, types: Vec<FeatureType>) -> Result<Self> {
        if types.len() != self.features.ncols() {
            return Err(SmeltError::DimensionMismatch {
                expected: self.features.ncols(),
                got: types.len(),
            });
        }
        self.feature_types = types;
        Ok(self)
    }

    /// Set custom feature names; must match the number of feature columns.
    pub fn with_feature_names(mut self, names: Vec<String>) -> Result<Self> {
        if names.len() != self.features.ncols() {
            return Err(SmeltError::DimensionMismatch {
                expected: self.features.ncols(),
                got: names.len(),
            });
        }
        self.feature_names = names;
        Ok(self)
    }

    /// Continuous target values, one per sample.
    pub fn target(&self) -> &[f64] {
        &self.target
    }
}

impl Task for RegressionTask {
    fn id(&self) -> &str {
        &self.id
    }
    fn features(&self) -> &Array2<f64> {
        &self.features
    }
    fn feature_names(&self) -> &[String] {
        &self.feature_names
    }
    fn feature_types(&self) -> &[FeatureType] {
        &self.feature_types
    }
}
