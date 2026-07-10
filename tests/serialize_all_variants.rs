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
    let target: Vec<f64> = (0..n).map(|i| (i as f64 * 0.17).sin() * 3.0 + 1.0).collect();
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
    let loaded: SerializableModel =
        load_json(&path).unwrap_or_else(|e| panic!("{name}: load_json failed on its own save_json output: {e}"));
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
        ("adaptive_random_forest", Box::new(AdaptiveRandomForest::new())),
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
