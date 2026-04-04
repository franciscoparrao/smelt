//! Tasks define the problem: data + target + type (classification/regression).
//!
//! Inspired by mlr3's `Task` system.

use crate::{Result, SmeltError};
use ndarray::Array2;
// validate module used by learner predict methods

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
}

/// Classification task with discrete target labels.
#[derive(Debug)]
pub struct ClassificationTask {
    id: String,
    features: Array2<f64>,
    target: Vec<usize>,
    feature_names: Vec<String>,
    class_names: Vec<String>,
}

impl ClassificationTask {
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
            features,
            target,
            class_names: (0..n_classes).map(|i| format!("class_{i}")).collect(),
        })
    }

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

    pub fn with_class_names(mut self, names: Vec<String>) -> Self {
        self.class_names = names;
        self
    }

    pub fn target(&self) -> &[usize] {
        &self.target
    }
    pub fn n_classes(&self) -> usize {
        self.class_names.len()
    }
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
}

/// Regression task with continuous target values.
#[derive(Debug)]
pub struct RegressionTask {
    id: String,
    features: Array2<f64>,
    target: Vec<f64>,
    feature_names: Vec<String>,
}

impl RegressionTask {
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
            features,
            target,
        })
    }

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
}
