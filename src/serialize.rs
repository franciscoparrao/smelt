//! Model serialization: save and load trained models as JSON.

use std::path::Path;
use std::fs;
use ndarray::Array2;
use serde::{Serialize, Deserialize};
use crate::learner::TrainedModel;
use crate::learner::tree::decision_tree::TrainedDecisionTree;
use crate::learner::tree::random_forest::TrainedRandomForest;
use crate::learner::tree::gradient_boosting::TrainedGradientBoosting;
use crate::learner::knn::{TrainedKnnClassifier, TrainedKnnRegressor};
use crate::learner::linear_regression::TrainedLinearRegression;
use crate::learner::logistic_regression::TrainedLogisticRegression;
use crate::prediction::Prediction;
use crate::{SmeltError, Result};

/// A serializable wrapper for all built-in trained model types.
///
/// Supports JSON serialization via `save_json` / `load_json`.
/// Note: `Bagging` and `Pipeline` trained models cannot be serialized
/// because they contain trait objects internally.
///
/// # Examples
///
/// ```no_run
/// use smelt::serialize::{SerializableModel, save_json, load_json};
///
/// // save_json(&model, "model.json").unwrap();
/// // let loaded = load_json("model.json").unwrap();
/// // let pred = loaded.predict(&features).unwrap();
/// ```
#[derive(Serialize, Deserialize)]
#[serde(tag = "model_type")]
pub enum SerializableModel {
    DecisionTree(TrainedDecisionTree),
    RandomForest(TrainedRandomForest),
    GradientBoosting(TrainedGradientBoosting),
    KnnClassifier(TrainedKnnClassifier),
    KnnRegressor(TrainedKnnRegressor),
    LinearRegression(TrainedLinearRegression),
    LogisticRegression(TrainedLogisticRegression),
}

impl SerializableModel {
    /// Make predictions using the underlying model.
    pub fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        match self {
            Self::DecisionTree(m) => m.predict(features),
            Self::RandomForest(m) => m.predict(features),
            Self::GradientBoosting(m) => m.predict(features),
            Self::KnnClassifier(m) => m.predict(features),
            Self::KnnRegressor(m) => m.predict(features),
            Self::LinearRegression(m) => m.predict(features),
            Self::LogisticRegression(m) => m.predict(features),
        }
    }

    /// Get feature importance if available.
    pub fn feature_importance(&self) -> Option<Vec<(String, f64)>> {
        match self {
            Self::DecisionTree(m) => m.feature_importance(),
            Self::RandomForest(m) => m.feature_importance(),
            Self::GradientBoosting(m) => m.feature_importance(),
            Self::KnnClassifier(m) => m.feature_importance(),
            Self::KnnRegressor(m) => m.feature_importance(),
            Self::LinearRegression(m) => m.feature_importance(),
            Self::LogisticRegression(m) => m.feature_importance(),
        }
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| SmeltError::Json(e.to_string()))
    }

    /// Deserialize from JSON string.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| SmeltError::Json(e.to_string()))
    }
}

/// Save a model to a JSON file.
pub fn save_json(model: &SerializableModel, path: impl AsRef<Path>) -> Result<()> {
    let json = model.to_json()?;
    fs::write(path, json)?;
    Ok(())
}

/// Load a model from a JSON file.
pub fn load_json(path: impl AsRef<Path>) -> Result<SerializableModel> {
    let json = fs::read_to_string(path)?;
    SerializableModel::from_json(&json)
}
