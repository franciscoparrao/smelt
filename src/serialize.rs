//! Model serialization: save and load trained models as JSON.

use crate::learner::TrainedModel;
use crate::learner::adaboost::TrainedAdaBoost;
use crate::learner::adaptive_rf::TrainedAdaptiveRandomForest;
use crate::learner::catboost::TrainedCatBoost;
use crate::learner::ebm::TrainedEBM;
use crate::learner::elm::TrainedELM;
use crate::learner::hoeffding::TrainedHoeffdingTree;
use crate::learner::knn::{TrainedKnnClassifier, TrainedKnnRegressor};
use crate::learner::lightgbm::TrainedLightGBM;
use crate::learner::linear_regression::TrainedLinearRegression;
use crate::learner::logistic_regression::TrainedLogisticRegression;
use crate::learner::mondrian::{TrainedMondrianForest, TrainedMondrianTree};
use crate::learner::naive_bayes::TrainedGaussianNB;
use crate::learner::oblique::{TrainedObliqueForest, TrainedObliqueTree};
use crate::learner::quantile::TrainedQuantileGB;
use crate::learner::quantile_forest::TrainedQuantileForest;
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
/// Note: `Bagging`, `Pipeline`, `Stacking`, `GeoXGBoost`, `DeepForest`,
/// `KrigingHybrid`, `DynamicEnsemble` and `CostSensitiveClassifier` hold
/// `Box<dyn TrainedModel>` internally and cannot be represented this way
/// (trait objects aren't (de)serializable). Every other built-in trained
/// model -- including the self-contained streaming/ensemble learners added
/// in 2026-07 (Mondrian, ELM, AdaptiveRandomForest, HoeffdingTree, the
/// Oblique tree/forest, EBM, QuantileGB, QuantileForest) -- has a variant
/// below.
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
#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "model_type")]
pub enum SerializableModel {
    /// A trained decision tree.
    DecisionTree(TrainedDecisionTree),
    /// A trained random forest.
    RandomForest(TrainedRandomForest),
    /// A trained gradient boosting model.
    GradientBoosting(TrainedGradientBoosting),
    /// A trained k-nearest-neighbors classifier.
    KnnClassifier(TrainedKnnClassifier),
    /// A trained k-nearest-neighbors regressor.
    KnnRegressor(TrainedKnnRegressor),
    /// A trained linear regression model.
    LinearRegression(TrainedLinearRegression),
    /// A trained logistic regression model.
    LogisticRegression(TrainedLogisticRegression),
    /// A trained XGBoost model.
    XGBoost(TrainedXGBoost),
    /// A trained LightGBM model.
    LightGBM(TrainedLightGBM),
    /// A trained CatBoost model.
    CatBoost(TrainedCatBoost),
    /// A trained extra-trees ensemble.
    ExtraTrees(TrainedExtraTrees),
    /// A trained AdaBoost ensemble.
    AdaBoost(TrainedAdaBoost),
    /// A trained linear support vector machine.
    LinearSVM(TrainedLinearSVM),
    /// A trained Gaussian naive Bayes classifier.
    GaussianNB(TrainedGaussianNB),
    /// A trained regularized (ridge/lasso) regression model.
    RegularizedRegression(TrainedRegularizedRegression),
    /// A trained Mondrian tree.
    MondrianTree(TrainedMondrianTree),
    /// A trained Mondrian forest.
    MondrianForest(TrainedMondrianForest),
    /// A trained Extreme Learning Machine.
    ExtremeLearningMachine(TrainedELM),
    /// A trained Adaptive Random Forest.
    AdaptiveRandomForest(TrainedAdaptiveRandomForest),
    /// A trained (batch-constructed) Hoeffding tree.
    HoeffdingTree(TrainedHoeffdingTree),
    /// A trained oblique decision tree.
    ObliqueTree(TrainedObliqueTree),
    /// A trained oblique forest.
    ObliqueForest(TrainedObliqueForest),
    /// A trained Explainable Boosting Machine.
    EBM(TrainedEBM),
    /// A trained Quantile Gradient Boosting regressor.
    QuantileGB(TrainedQuantileGB),
    /// A trained Quantile Regression Forest.
    QuantileForest(TrainedQuantileForest),
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
            Self::MondrianTree(m) => m.predict(features),
            Self::MondrianForest(m) => m.predict(features),
            Self::ExtremeLearningMachine(m) => m.predict(features),
            Self::AdaptiveRandomForest(m) => m.predict(features),
            Self::HoeffdingTree(m) => m.predict(features),
            Self::ObliqueTree(m) => m.predict(features),
            Self::ObliqueForest(m) => m.predict(features),
            Self::EBM(m) => m.predict(features),
            Self::QuantileGB(m) => m.predict(features),
            Self::QuantileForest(m) => m.predict(features),
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
            Self::MondrianTree(m) => m.feature_importance(),
            Self::MondrianForest(m) => m.feature_importance(),
            Self::ExtremeLearningMachine(m) => m.feature_importance(),
            Self::AdaptiveRandomForest(m) => m.feature_importance(),
            Self::HoeffdingTree(m) => m.feature_importance(),
            Self::ObliqueTree(m) => m.feature_importance(),
            Self::ObliqueForest(m) => m.feature_importance(),
            Self::EBM(m) => m.feature_importance(),
            Self::QuantileGB(m) => m.feature_importance(),
            Self::QuantileForest(m) => m.feature_importance(),
        }
    }

    /// Name of this variant (matches its `model_type` tag in JSON), for
    /// callers that need to check a loaded file holds the model type they
    /// expect (e.g. smelt-py's `load()`, which would otherwise silently
    /// accept a file saved from a different learner and forward `predict`
    /// to the wrong underlying model).
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::DecisionTree(_) => "DecisionTree",
            Self::RandomForest(_) => "RandomForest",
            Self::GradientBoosting(_) => "GradientBoosting",
            Self::KnnClassifier(_) => "KnnClassifier",
            Self::KnnRegressor(_) => "KnnRegressor",
            Self::LinearRegression(_) => "LinearRegression",
            Self::LogisticRegression(_) => "LogisticRegression",
            Self::XGBoost(_) => "XGBoost",
            Self::LightGBM(_) => "LightGBM",
            Self::CatBoost(_) => "CatBoost",
            Self::ExtraTrees(_) => "ExtraTrees",
            Self::AdaBoost(_) => "AdaBoost",
            Self::LinearSVM(_) => "LinearSVM",
            Self::GaussianNB(_) => "GaussianNB",
            Self::RegularizedRegression(_) => "RegularizedRegression",
            Self::MondrianTree(_) => "MondrianTree",
            Self::MondrianForest(_) => "MondrianForest",
            Self::ExtremeLearningMachine(_) => "ExtremeLearningMachine",
            Self::AdaptiveRandomForest(_) => "AdaptiveRandomForest",
            Self::HoeffdingTree(_) => "HoeffdingTree",
            Self::ObliqueTree(_) => "ObliqueTree",
            Self::ObliqueForest(_) => "ObliqueForest",
            Self::EBM(_) => "EBM",
            Self::QuantileGB(_) => "QuantileGB",
            Self::QuantileForest(_) => "QuantileForest",
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

impl TrainedModel for SerializableModel {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        SerializableModel::predict(self, features)
    }

    fn feature_importance(&self) -> Option<Vec<(String, f64)>> {
        SerializableModel::feature_importance(self)
    }

    /// Round-trips a loaded model back to its own `SerializableModel` form,
    /// so re-saving after a load (e.g. converting a legacy file to the
    /// current format) works the same as saving a freshly trained model.
    fn to_serializable(&self) -> Option<SerializableModel> {
        Some(self.clone())
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
///
/// Files written before the versioned envelope existed (smelt-ml < 2.0, raw
/// `SerializableModel` JSON with no `format_version`/`smelt_version`/`model`
/// wrapper) are rejected with a message naming that specifically, rather
/// than the opaque `missing field \`format_version\`` serde error the
/// envelope itself was introduced to avoid.
pub fn load_json(path: impl AsRef<Path>) -> Result<SerializableModel> {
    let metadata = fs::metadata(&path)?;
    if metadata.len() > 100_000_000 {
        return Err(SmeltError::Json("Model file too large (>100MB)".into()));
    }
    let json = fs::read_to_string(path)?;
    let raw: serde_json::Value =
        serde_json::from_str(&json).map_err(|e| SmeltError::Json(e.to_string()))?;
    if raw.get("format_version").is_none() {
        return Err(SmeltError::Json(
            "this file has no `format_version` field, so it predates the versioned \
             model-file envelope introduced in smelt-ml 2.0 -- its raw pre-2.0 format \
             is not supported by this version; re-train and re-save the model with \
             the current smelt-ml release"
                .into(),
        ));
    }
    let file: ModelFile =
        serde_json::from_value(raw).map_err(|e| SmeltError::Json(e.to_string()))?;
    if file.format_version != SERIALIZATION_FORMAT_VERSION {
        return Err(SmeltError::Json(format!(
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

    /// A trivial constant-predicting ELM (zero input weights/bias, so the
    /// hidden layer's sigmoid activation is always exactly 0.5 regardless
    /// of input), built directly from `TrainedELM`'s `pub(crate)` fields --
    /// representative of the 2026-07 self-contained streaming/ensemble
    /// learners (Mondrian, ELM, AdaptiveRandomForest, HoeffdingTree, the
    /// Oblique tree/forest, EBM, QuantileGB, QuantileForest) that gained a
    /// `SerializableModel` variant in this same change.
    fn constant_elm(value: f64) -> TrainedELM {
        TrainedELM {
            input_weights: Array2::zeros((1, 1)),
            biases: ndarray::Array1::zeros(1),
            output_weights: ndarray::Array1::from_vec(vec![2.0 * value]).insert_axis(ndarray::Axis(1)),
            activation: crate::learner::elm::Activation::Sigmoid,
            is_classifier: false,
            n_features: 1,
            feature_mean: ndarray::Array1::zeros(1),
            feature_std: ndarray::Array1::from_vec(vec![1.0]),
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

    /// Regression test (M-10, `docs/auditoria_motor_2026-07-05.md`): none of
    /// the self-contained learners added 2026-07-04 (Mondrian, ELM,
    /// AdaptiveRandomForest, HoeffdingTree, Oblique tree/forest, EBM,
    /// QuantileGB, QuantileForest) derived `Serialize`/`Deserialize` or had
    /// a `SerializableModel` variant, despite holding no `Box<dyn
    /// TrainedModel>` internals (the actual reason `Bagging`/`Stacking`/
    /// `DeepForest`/etc. are excluded) -- an oversight, not a deliberate
    /// exclusion. This exercises one representative end-to-end
    /// save/load/predict round trip; the other 7 share the same
    /// `#[derive(Serialize, Deserialize)]` + enum-variant wiring, verified
    /// structurally by this crate compiling at all (the derive macro fails
    /// to expand if any field's type doesn't itself implement the trait).
    #[test]
    fn elm_roundtrips_through_save_load() {
        let features = array![[0.0], [1.0], [2.0]];
        let model = SerializableModel::ExtremeLearningMachine(constant_elm(3.0));

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("elm.json");
        save_json(&model, &path).unwrap();
        let loaded = load_json(&path).unwrap();

        let pred = loaded.predict(&features).unwrap();
        let Prediction::Regression { predicted, .. } = pred else {
            panic!("expected regression")
        };
        for p in predicted {
            assert!((p - 3.0).abs() < 1e-9, "expected constant 3.0, got {p}");
        }
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

    /// Regression test (M-14, `docs/auditoria_motor_2026-07-05.md`): a file
    /// written before the versioned envelope existed (pre-2.0 raw
    /// `SerializableModel::to_json()` output, no `format_version` wrapper)
    /// used to fail `load_json` with an opaque
    /// `Json("missing field \`format_version\`")` -- exactly the confusing
    /// serde error the envelope was introduced to avoid. It should instead
    /// name the actual problem (a pre-2.0 legacy file format).
    #[test]
    fn load_json_names_pre_envelope_legacy_files_explicitly() {
        let model = SerializableModel::XGBoost(constant_xgb(1.0));
        let legacy_json = model.to_json().unwrap(); // no envelope, as pre-2.0 save_json wrote

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("legacy.json");
        fs::write(&path, legacy_json).unwrap();

        match load_json(&path) {
            Err(err) => {
                let msg = format!("{err}");
                assert!(
                    msg.contains("format_version") && msg.contains("pre-2.0"),
                    "error should explicitly name the missing envelope / pre-2.0 format, got: {msg}"
                );
                assert!(
                    !msg.contains("missing field"),
                    "error should not be the raw opaque serde message: {msg}"
                );
            }
            Ok(_) => panic!("expected a legacy-format error"),
        }
    }
}
