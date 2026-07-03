//! Model serialization: save and load trained models as JSON.

use crate::learner::TrainedModel;
use crate::learner::adaboost::TrainedAdaBoost;
use crate::learner::catboost::TrainedCatBoost;
use crate::learner::knn::{TrainedKnnClassifier, TrainedKnnRegressor};
use crate::learner::lightgbm::TrainedLightGBM;
use crate::learner::linear_regression::TrainedLinearRegression;
use crate::learner::logistic_regression::TrainedLogisticRegression;
use crate::learner::naive_bayes::TrainedGaussianNB;
use crate::learner::regularized::TrainedRegularizedRegression;
use crate::learner::svm::TrainedLinearSVM;
use crate::learner::tree::decision_tree::TrainedDecisionTree;
use crate::learner::tree::extra_trees::TrainedExtraTrees;
use crate::learner::tree::gradient_boosting::TrainedGradientBoosting;
use crate::learner::tree::random_forest::TrainedRandomForest;
use crate::learner::xgboost::TrainedXGBoost;
use crate::prediction::Prediction;
use crate::{Result, SmeltError};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Serialization format version. Bump when `SerializableModel`'s wire format
/// changes in a way that breaks reading older files, and add an explicit
/// migration/error path for the old version(s) below.
pub const SERIALIZATION_FORMAT_VERSION: u32 = 1;

/// A serializable wrapper for all built-in trained model types.
///
/// Supports JSON serialization via `save_json` / `load_json`.
/// Note: `Bagging`, `Pipeline`, `Stacking`, `GeoXGBoost` and other trained
/// models that hold `Box<dyn TrainedModel>` internally cannot be serialized
/// this way (trait objects aren't representable); `QuantileForest` is
/// likewise excluded pending a dedicated variant.
///
/// # Examples
///
/// ```no_run
/// use smelt_ml::serialize::{SerializableModel, save_json, load_json};
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
    XGBoost(TrainedXGBoost),
    LightGBM(TrainedLightGBM),
    CatBoost(TrainedCatBoost),
    ExtraTrees(TrainedExtraTrees),
    AdaBoost(TrainedAdaBoost),
    LinearSVM(TrainedLinearSVM),
    GaussianNB(TrainedGaussianNB),
    RegularizedRegression(TrainedRegularizedRegression),
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
            Self::XGBoost(m) => m.predict(features),
            Self::LightGBM(m) => m.predict(features),
            Self::CatBoost(m) => m.predict(features),
            Self::ExtraTrees(m) => m.predict(features),
            Self::AdaBoost(m) => m.predict(features),
            Self::LinearSVM(m) => m.predict(features),
            Self::GaussianNB(m) => m.predict(features),
            Self::RegularizedRegression(m) => m.predict(features),
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
            Self::XGBoost(m) => m.feature_importance(),
            Self::LightGBM(m) => m.feature_importance(),
            Self::CatBoost(m) => m.feature_importance(),
            Self::ExtraTrees(m) => m.feature_importance(),
            Self::AdaBoost(m) => m.feature_importance(),
            Self::LinearSVM(m) => m.feature_importance(),
            Self::GaussianNB(m) => m.feature_importance(),
            Self::RegularizedRegression(m) => m.feature_importance(),
        }
    }

    /// Serialize to a raw JSON string (no format/version wrapper). Prefer
    /// [`save_json`] for files, which wraps this with a version header so a
    /// future incompatible format change can be detected and reported
    /// clearly instead of failing deserialization with a confusing error.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| SmeltError::Json(e.to_string()))
    }

    /// Deserialize from a raw JSON string produced by [`Self::to_json`].
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| SmeltError::Json(e.to_string()))
    }
}

/// On-disk envelope: pairs a [`SerializableModel`] with the format version
/// and the smelt-ml version that wrote it, so a future format change can be
/// detected at load time instead of failing with an opaque serde error.
#[derive(Serialize)]
struct ModelFileRef<'a> {
    format_version: u32,
    smelt_version: &'static str,
    model: &'a SerializableModel,
}

#[derive(Deserialize)]
struct ModelFile {
    format_version: u32,
    smelt_version: String,
    model: SerializableModel,
}

/// Save a model to a JSON file, tagged with the current format/crate version.
pub fn save_json(model: &SerializableModel, path: impl AsRef<Path>) -> Result<()> {
    let file = ModelFileRef {
        format_version: SERIALIZATION_FORMAT_VERSION,
        smelt_version: env!("CARGO_PKG_VERSION"),
        model,
    };
    let json = serde_json::to_string_pretty(&file).map_err(|e| SmeltError::Json(e.to_string()))?;
    fs::write(path, json)?;
    Ok(())
}

/// Load a model from a JSON file written by [`save_json`].
/// Rejects files larger than 100MB to prevent OOM.
pub fn load_json(path: impl AsRef<Path>) -> Result<SerializableModel> {
    let metadata = fs::metadata(&path)?;
    if metadata.len() > 100_000_000 {
        return Err(SmeltError::Other("Model file too large (>100MB)".into()));
    }
    let json = fs::read_to_string(path)?;
    let file: ModelFile =
        serde_json::from_str(&json).map_err(|e| SmeltError::Json(e.to_string()))?;
    if file.format_version != SERIALIZATION_FORMAT_VERSION {
        return Err(SmeltError::Other(format!(
            "model file has serialization format version {}, this build of smelt-ml ({}) expects version {} (file was written by smelt-ml {})",
            file.format_version,
            env!("CARGO_PKG_VERSION"),
            SERIALIZATION_FORMAT_VERSION,
            file.smelt_version,
        )));
    }
    Ok(file.model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::xgboost::{TrainedXGBoost, XGBMode};
    use ndarray::array;

    /// A trivial constant-predicting XGBoost model (no trees), built directly
    /// from `TrainedXGBoost`'s `pub(crate)` fields -- avoids depending on the
    /// internals of a real training run just to exercise serialization.
    fn constant_xgb(value: f64) -> TrainedXGBoost {
        TrainedXGBoost {
            trees: vec![],
            initial: vec![value],
            learning_rate: 0.1,
            mode: XGBMode::Regression,
            feature_names: vec!["x0".to_string()],
            feature_importances: vec![0.0],
            transform: Default::default(),
        }
    }

    /// Regression test: XGBoost/LightGBM/CatBoost (and 5 others) derive
    /// Serialize but, until now, had no variant in `SerializableModel` --
    /// save_json/load_json simply couldn't represent them despite the
    /// flagship gradient boosting engines being the whole point of
    /// persisting a trained model.
    #[test]
    fn xgboost_roundtrips_through_save_load() {
        let features = array![[0.0], [1.0], [2.0]];
        let model = SerializableModel::XGBoost(constant_xgb(7.0));

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("xgb.json");
        save_json(&model, &path).unwrap();
        let loaded = load_json(&path).unwrap();

        let pred = loaded.predict(&features).unwrap();
        let Prediction::Regression { predicted, .. } = pred else {
            panic!("expected regression")
        };
        assert_eq!(predicted, vec![7.0, 7.0, 7.0]);
    }

    /// Version-mismatch rejection: a file claiming a different format
    /// version must fail with a clear error naming the mismatch, not an
    /// opaque serde parse failure on the next format-breaking change.
    #[test]
    fn load_json_rejects_mismatched_format_version() {
        let model = SerializableModel::XGBoost(constant_xgb(1.0));

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.json");
        save_json(&model, &path).unwrap();

        // Tamper with the format_version field.
        let json = fs::read_to_string(&path).unwrap();
        let bumped = json.replacen("\"format_version\": 1", "\"format_version\": 999", 1);
        assert_ne!(json, bumped, "the replacement should have matched something");
        fs::write(&path, bumped).unwrap();

        match load_json(&path) {
            Err(err) => assert!(
                format!("{err}").contains("999"),
                "error should mention the mismatched version: {err}"
            ),
            Ok(_) => panic!("expected a version-mismatch error"),
        }
    }
}
