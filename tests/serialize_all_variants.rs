//! Roundtrip coverage for EVERY `SerializableModel` variant (4th audit,
//! HIGH-2): `save_json` → `load_json` must reproduce predictions exactly
//! for each variant, trained with each learner's DEFAULT hyperparameters.
//!
//! The defaults matter: the original seven-model spot check missed that
//! MondrianTree/MondrianForest with the default infinite `lifetime`
//! serialized every leaf's `tau` as JSON `null` (unloadable), and that
//! HoeffdingTree/AdaptiveRandomForest's `HashMap<usize, _>` keys only
//! decode on serde_json's text path — `save()` succeeded while `load()`
//! could never read the file back. A structural "it compiles" check cannot
//! catch either; only an executed roundtrip per variant can.

use ndarray::Array2;
use smelt_ml::learner::TrainedModel;
use smelt_ml::learner::quantile::QuantileGB;
use smelt_ml::prelude::*;
use smelt_ml::serialize::{SerializableModel, load_json, save_json};
use std::collections::BTreeSet;

fn classif_task() -> ClassificationTask {
    let n = 40;
    // Column 2 holds integer category codes so the catboost_categorical
    // entry exercises real target-statistic encodings; every other learner
    // just sees it as a numeric feature.
    let features = Array2::from_shape_fn((n, 3), |(i, j)| {
        if j == 2 {
            (i % 3) as f64
        } else {
            (i as f64 * 0.37 + j as f64 * 1.71).sin() + if i % 2 == 0 { 1.5 } else { -1.5 }
        }
    });
    let target: Vec<usize> = (0..n).map(|i| i % 2).collect();
    ClassificationTask::new("roundtrip-classif", features, target).unwrap()
}

fn regress_task() -> RegressionTask {
    let n = 40;
    // Distinct per-column frequencies: equal-frequency phase-shifted
    // cosines are linear combinations of one sin/cos pair, which makes the
    // normal equation singular for LinearRegression.
    let features =
        Array2::from_shape_fn((n, 3), |(i, j)| (i as f64 * (0.31 + 0.17 * j as f64)).cos());
    let target: Vec<f64> = (0..n)
        .map(|i| (i as f64 * 0.17).sin() * 3.0 + 1.0)
        .collect();
    RegressionTask::new("roundtrip-regress", features, target).unwrap()
}

/// Saves, reloads, and compares predictions bit-for-bit (via the
/// `Prediction` serde form, which covers classes, values, AND
/// probabilities). Returns the covered variant's `type_name`.
fn assert_roundtrip(
    name: &str,
    trained: Box<dyn TrainedModel>,
    features: &Array2<f64>,
) -> &'static str {
    let serial = trained
        .to_serializable()
        .unwrap_or_else(|| panic!("{name}: to_serializable() returned None"));
    let type_name = serial.type_name();

    let path = std::env::temp_dir().join(format!(
        "smelt_roundtrip_{}_{}.json",
        name,
        std::process::id()
    ));
    save_json(&serial, &path).unwrap_or_else(|e| panic!("{name}: save_json failed: {e}"));
    let loaded: SerializableModel = load_json(&path)
        .unwrap_or_else(|e| panic!("{name}: load_json failed on its own save_json output: {e}"));
    std::fs::remove_file(&path).ok();

    let before = trained
        .predict(features)
        .unwrap_or_else(|e| panic!("{name}: predict before save failed: {e}"));
    let after = loaded
        .predict(features)
        .unwrap_or_else(|e| panic!("{name}: predict after load failed: {e}"));
    assert_eq!(
        serde_json::to_string(&before).unwrap(),
        serde_json::to_string(&after).unwrap(),
        "{name}: predictions changed across the save/load roundtrip"
    );
    type_name
}

#[test]
fn every_serializable_model_variant_survives_a_save_load_roundtrip() {
    let classif = classif_task();
    let regress = regress_task();

    let classif_learners: Vec<(&str, Box<dyn Learner>)> = vec![
        ("decision_tree", Box::new(DecisionTree::new())),
        ("random_forest", Box::new(RandomForest::new())),
        ("knn_classifier", Box::new(KNearestNeighbors::new(3))),
        ("logistic_regression", Box::new(LogisticRegression::new())),
        ("xgboost", Box::new(XGBoost::new())),
        ("lightgbm", Box::new(LightGBM::new())),
        ("catboost", Box::new(CatBoost::new())),
        // With a categorical feature on purpose: `cat_encodings` (integer-
        // keyed nested maps, only populated when cat_features is non-empty)
        // was unloadable as JSON objects — same bug class as HoeffdingTree.
        (
            "catboost_categorical",
            Box::new(CatBoost::new().with_cat_features(vec![2])),
        ),
        ("extra_trees", Box::new(ExtraTrees::new())),
        ("adaboost", Box::new(AdaBoost::new())),
        ("linear_svm", Box::new(LinearSVM::new())),
        ("gaussian_nb", Box::new(GaussianNB::new())),
        // Default (infinite) lifetime on purpose: the discriminating config
        // for the `tau` = INFINITY → JSON null bug.
        ("mondrian_tree", Box::new(MondrianTree::new())),
        ("mondrian_forest", Box::new(MondrianForest::new())),
        ("elm", Box::new(ExtremeLearningMachine::new())),
        (
            "adaptive_random_forest",
            Box::new(AdaptiveRandomForest::new()),
        ),
        ("hoeffding_tree", Box::new(HoeffdingTree::new())),
        ("oblique_tree", Box::new(ObliqueTree::new())),
        ("oblique_forest", Box::new(ObliqueForest::new())),
        ("ebm", Box::new(EBM::new())),
    ];
    let regress_learners: Vec<(&str, Box<dyn Learner>)> = vec![
        ("gradient_boosting", Box::new(GradientBoosting::new())),
        ("knn_regressor", Box::new(KNearestNeighbors::new(3))),
        ("linear_regression", Box::new(LinearRegression::new())),
        ("ridge", Box::new(Ridge::new(1.0))),
        ("quantile_gb", Box::new(QuantileGB::new(0.5))),
        ("quantile_forest", Box::new(QuantileForest::new())),
    ];

    let mut covered: BTreeSet<&'static str> = BTreeSet::new();
    for (name, mut learner) in classif_learners {
        let trained = learner
            .train_classif(&classif)
            .unwrap_or_else(|e| panic!("{name}: train_classif failed: {e}"));
        covered.insert(assert_roundtrip(name, trained, classif.features()));
    }
    for (name, mut learner) in regress_learners {
        let trained = learner
            .train_regress(&regress)
            .unwrap_or_else(|e| panic!("{name}: train_regress failed: {e}"));
        covered.insert(assert_roundtrip(name, trained, regress.features()));
    }

    // Pin the variant count: adding a SerializableModel variant without
    // adding its roundtrip here must fail this assertion.
    assert_eq!(
        covered.len(),
        25,
        "expected all 25 SerializableModel variants exercised, got {}: {covered:?}",
        covered.len()
    );
}

/// Rewrite a saved CatBoost model file's `cat_encodings` from the current
/// vec-of-pairs wire form into the nested-JSON-object form that smelt-ml
/// 2.0.x–3.0.0 wrote (string keys), byte-for-byte what those releases'
/// default serde derive produced: `[{"col": {"code": enc, ...}, ...}, ...]`
/// (or `[{}]` with no categorical features). Returns the rewritten envelope.
fn to_legacy_cat_encodings_form(envelope_json: &str) -> String {
    let mut raw: serde_json::Value = serde_json::from_str(envelope_json).unwrap();
    let enc = raw["model"]["cat_encodings"].take();
    let outer = enc.as_array().expect("cat_encodings must be an array");
    let legacy: Vec<serde_json::Value> = outer
        .iter()
        .map(|per_output| {
            let mut obj = serde_json::Map::new();
            for pair in per_output.as_array().expect("pairs form: outer vec") {
                let pair = pair.as_array().expect("pairs form: (col, cats) tuple");
                let col = pair[0].as_u64().unwrap().to_string();
                let mut cats = serde_json::Map::new();
                for cat_pair in pair[1].as_array().expect("pairs form: cats vec") {
                    let cat_pair = cat_pair.as_array().unwrap();
                    cats.insert(cat_pair[0].to_string(), cat_pair[1].clone());
                }
                obj.insert(col, serde_json::Value::Object(cats));
            }
            serde_json::Value::Object(obj)
        })
        .collect();
    raw["model"]["cat_encodings"] = serde_json::Value::Array(legacy);
    serde_json::to_string_pretty(&raw).unwrap()
}

/// Regression test (5th audit, M-2): the vec-of-pairs `cat_encodings_serde`
/// introduced for the HIGH-2 fix rejected the legacy JSON-object wire form
/// that smelt-ml 2.0.x–3.0.0 wrote — `"cat_encodings": [{}]` (no
/// categorical features) or string-keyed nested maps (populated) — with an
/// opaque `invalid type: map, expected a sequence` serde error, inside the
/// same `format_version` 1 the envelope check waves through. Both legacy
/// forms must now load and predict identically to the original model, and
/// the current pairs form must keep roundtripping bit-identically.
#[test]
fn catboost_legacy_object_form_cat_encodings_still_loads() {
    let task = classif_task();
    let tmp = std::env::temp_dir();
    let pid = std::process::id();

    // (a) populated cat_features → legacy string-keyed nested maps.
    // (b) no cat_features → legacy `[{}]` / `[{}, {}, ...]` empty objects.
    let configs: Vec<(&str, Box<dyn Learner>)> = vec![
        (
            "legacy_cat",
            Box::new(CatBoost::new().with_cat_features(vec![2])),
        ),
        ("legacy_nocat", Box::new(CatBoost::new())),
    ];

    for (name, mut learner) in configs {
        let trained = learner.train_classif(&task).unwrap();
        let before = serde_json::to_string(&trained.predict(task.features()).unwrap()).unwrap();
        let serial = trained.to_serializable().unwrap();

        let path = tmp.join(format!("smelt_catboost_{name}_{pid}.json"));
        save_json(&serial, &path).unwrap();
        let current_json = std::fs::read_to_string(&path).unwrap();

        // Sanity: the rewrite really produced the legacy object form (every
        // per-output entry is a JSON object, not a vec of pairs).
        let legacy_json = to_legacy_cat_encodings_form(&current_json);
        let reparsed: serde_json::Value = serde_json::from_str(&legacy_json).unwrap();
        let entries = reparsed["model"]["cat_encodings"].as_array().unwrap();
        assert!(
            !entries.is_empty() && entries.iter().all(|e| e.is_object()),
            "{name}: rewrite should have produced JSON objects, got: {entries:?}"
        );
        std::fs::write(&path, &legacy_json).unwrap();

        // Legacy form must load through the full envelope path and predict
        // bit-identically.
        let loaded = load_json(&path)
            .unwrap_or_else(|e| panic!("{name}: legacy 2.0.x–3.0.0 wire form must load: {e}"));
        let after = serde_json::to_string(&loaded.predict(task.features()).unwrap()).unwrap();
        assert_eq!(
            before, after,
            "{name}: legacy-form load changed predictions"
        );

        // The current pairs form still roundtrips bit-identically (the dual
        // deserializer must not have disturbed the new path).
        std::fs::write(&path, &current_json).unwrap();
        let reloaded = load_json(&path).unwrap();
        let after_current =
            serde_json::to_string(&reloaded.predict(task.features()).unwrap()).unwrap();
        assert_eq!(
            before, after_current,
            "{name}: pairs-form load changed predictions"
        );

        std::fs::remove_file(&path).ok();
    }
}
