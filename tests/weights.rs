//! Fase A of per-sample weights: the crate carries weights honestly without
//! any learner consuming them yet.
//!
//! - every learner rejects a weighted task with a clear `InvalidParameter`
//!   (never silently ignores the weights),
//! - CV fold tasks slice weights with the same indices as features/target
//!   (probe learner, same technique as the M-3 feature_types test),
//! - Pipeline propagates weights through row-preserving transformer stages,
//! - resamplers (Smote/Adasyn/SpatialSmote, and the Pipeline resampler
//!   stage) reject weighted tasks: synthetic samples' weights are undefined.

use ndarray::Array2;
use smelt_ml::learner::TrainedModel;
use smelt_ml::prelude::*;
use std::sync::{Arc, Mutex};

// ── probe learner: captures the weights of the task it is trained on ──

/// Recorded per training call: (weights if any, classif target, regress target).
type Seen = Arc<Mutex<Vec<(Option<Vec<f64>>, Vec<usize>, Vec<f64>)>>>;

struct WeightProbe {
    seen: Seen,
}

struct DummyModel {
    classif: bool,
}

impl TrainedModel for DummyModel {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        let n = features.nrows();
        Ok(if self.classif {
            Prediction::Classification {
                predicted: vec![0; n],
                truth: None,
                probabilities: None,
            }
        } else {
            Prediction::Regression {
                predicted: vec![0.0; n],
                truth: None,
            }
        })
    }
}

impl Learner for WeightProbe {
    fn id(&self) -> &str {
        "weight_probe"
    }
    fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        self.seen.lock().unwrap().push((
            task.weights().map(|w| w.to_vec()),
            task.target().to_vec(),
            Vec::new(),
        ));
        Ok(Box::new(DummyModel { classif: true }))
    }
    fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
        self.seen.lock().unwrap().push((
            task.weights().map(|w| w.to_vec()),
            Vec::new(),
            task.target().to_vec(),
        ));
        Ok(Box::new(DummyModel { classif: false }))
    }
}

fn make_probe() -> (WeightProbe, Seen) {
    let seen: Seen = Arc::new(Mutex::new(Vec::new()));
    (WeightProbe { seen: seen.clone() }, seen)
}

// ── fixtures: weight_i is a deterministic function of target_i, so any
// fold/reorder misalignment between weights and rows is detectable ──

/// 20 samples; target_i = i % 2, weight_i = i + 1 encoded via feature 0 = i.
fn weighted_classif_task(n: usize) -> ClassificationTask {
    let features = Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { i as f64 } else { 1.0 });
    let target: Vec<usize> = (0..n).map(|i| i % 2).collect();
    let weights: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
    ClassificationTask::new("wc", features, target)
        .unwrap()
        .with_weights(weights)
}

/// n samples; target_i = i, weight_i = target_i + 1 — alignment is checkable
/// from the fold task alone.
fn weighted_regress_task(n: usize) -> RegressionTask {
    let features = Array2::from_shape_fn((n, 1), |(i, _)| i as f64);
    let target: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let weights: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
    RegressionTask::new("wr", features, target)
        .unwrap()
        .with_weights(weights)
}

// ── learner guard ──

#[test]
fn learners_reject_weighted_tasks_with_a_clear_error() {
    // DecisionTree became weight-aware in Fase B (its positive tests live
    // in tests/weights_trees.rs), so the classification-side guard is now
    // exercised through KNN, which remains weight-unaware.
    let task = weighted_classif_task(10);
    let mut knn_c = KNearestNeighbors::new(3);
    let err = knn_c.train_classif(&task).map(|_| ()).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("KNearestNeighbors") && msg.contains("does not support sample weights"),
        "guard must name the learner and the problem: {msg}"
    );

    // A composite wrapper guards too (weights are NOT forwarded silently
    // into its internal bootstrap/fold tasks), even when its base learner
    // is itself weight-aware.
    let mut bag = Bagging::new(|| Box::new(DecisionTree::default())).with_n_estimators(3);
    let err = bag.train_classif(&task).map(|_| ()).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("Bagging") && msg.contains("does not support sample weights"),
        "composite guard must name the wrapper: {msg}"
    );

    // Fase B3 made Ridge (and the other linear models) weight-aware, so the
    // regression-side guard is now exercised through KNN, which remains
    // weight-unaware.
    let rtask = weighted_regress_task(10);
    let mut knn = KNearestNeighbors::new(3);
    let err = knn.train_regress(&rtask).map(|_| ()).unwrap_err();
    assert!(
        format!("{err}").contains("does not support sample weights"),
        "regression guard: {err}"
    );
}

#[test]
fn supports_weights_defaults_to_false() {
    // DecisionTree (Fase B, tests/weights_trees.rs), the boosting engines
    // (Fase B2, tests/weights_boosting.rs) and the linear models (Fase B3,
    // tests/weights_linear.rs) now consume weights — their positive
    // assertions live in their own files. LinearSVM remains weight-unaware.
    assert!(!LinearSVM::new().supports_weights());
    // trait-object dispatch works too (the future registry-properties seam)
    let learner: Box<dyn Learner> = Box::new(KNearestNeighbors::new(3));
    assert!(!learner.supports_weights());
}

// ── CV fold slicing ──

#[test]
fn cv_fold_tasks_carry_weights_sliced_with_the_same_indices() {
    let n = 20;
    let task = weighted_regress_task(n);
    let (mut probe, seen) = make_probe();
    let cv = CrossValidation::new(4);
    benchmark::resample_regress(&mut probe, &task, &cv, &[]).unwrap();

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 4, "one training call per fold");
    for (fold, (weights, _, target)) in seen.iter().enumerate() {
        let weights = weights
            .as_ref()
            .unwrap_or_else(|| panic!("fold {fold}: weights were dropped on the fold rebuild"));
        assert_eq!(
            weights.len(),
            target.len(),
            "fold {fold}: one weight per fold row"
        );
        assert_eq!(target.len(), n - n / 4, "fold {fold}: expected train size");
        for (j, (&w, &t)) in weights.iter().zip(target.iter()).enumerate() {
            assert_eq!(
                w,
                t + 1.0,
                "fold {fold}, row {j}: weight misaligned with its sample (weight={w}, target={t})"
            );
        }
    }
}

#[test]
fn cv_fold_tasks_carry_weights_for_classification_too() {
    let n = 20;
    let task = weighted_classif_task(n);
    let (mut probe, seen) = make_probe();
    let cv = CrossValidation::new(4);
    benchmark::resample_classif(&mut probe, &task, &cv, &[]).unwrap();

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 4);
    for (fold, (weights, target, _)) in seen.iter().enumerate() {
        let weights = weights
            .as_ref()
            .unwrap_or_else(|| panic!("fold {fold}: weights were dropped on the fold rebuild"));
        assert_eq!(weights.len(), target.len());
        for (j, (&w, &t)) in weights.iter().zip(target.iter()).enumerate() {
            // weight_i = i + 1 and target_i = i % 2, so weight parity must
            // match: even weight-1 ⇔ label 0, odd ⇔ label 1.
            assert_eq!(
                (w as usize - 1) % 2,
                t,
                "fold {fold}, row {j}: weight {w} misaligned with label {t}"
            );
        }
    }
}

#[test]
fn unweighted_tasks_still_produce_unweighted_fold_tasks() {
    let features = Array2::from_shape_fn((12, 1), |(i, _)| i as f64);
    let target: Vec<f64> = (0..12).map(|i| i as f64).collect();
    let task = RegressionTask::new("plain", features, target).unwrap();
    let (mut probe, seen) = make_probe();
    benchmark::resample_regress(&mut probe, &task, &CrossValidation::new(3), &[]).unwrap();
    for (weights, _, _) in seen.lock().unwrap().iter() {
        assert!(weights.is_none(), "no weights must be fabricated");
    }
}

// ── Pipeline propagation ──

#[test]
fn pipeline_propagates_weights_through_row_preserving_transformers() {
    let task = weighted_classif_task(10);
    let expected = task.weights().unwrap().to_vec();
    let (probe, seen) = make_probe();
    let mut pipe = Pipeline::new(vec![Box::new(StandardScaler::new())], Box::new(probe));
    pipe.train_classif(&task).unwrap();

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 1);
    assert_eq!(
        seen[0].0.as_deref(),
        Some(expected.as_slice()),
        "weights must reach the final learner unchanged through scaler stages"
    );

    // regression side
    let rtask = weighted_regress_task(10);
    let expected = rtask.weights().unwrap().to_vec();
    let (probe, seen) = make_probe();
    let mut pipe = Pipeline::new(vec![Box::new(StandardScaler::new())], Box::new(probe));
    pipe.train_regress(&rtask).unwrap();
    assert_eq!(seen.lock().unwrap()[0].0.as_deref(), Some(expected.as_slice()));
}

#[test]
fn pipeline_with_a_learner_guard_still_rejects_weights_end_to_end() {
    // Pipeline itself propagates; the inner learner's guard must fire, so a
    // weighted task through a pipeline is never silently unweighted.
    let task = weighted_classif_task(10);
    let mut pipe = Pipeline::new(
        vec![Box::new(StandardScaler::new())],
        Box::new(KNearestNeighbors::new(1)),
    );
    let err = pipe.train_classif(&task).map(|_| ()).unwrap_err();
    assert!(
        format!("{err}").contains("does not support sample weights"),
        "inner learner guard must surface through the Pipeline: {err}"
    );
}

// ── resamplers reject weighted tasks ──

fn assert_resampling_rejected(err: SmeltError) {
    let msg = format!("{err}");
    assert!(
        msg.contains("resampling a weighted task is not supported"),
        "expected the weighted-resampling rejection, got: {msg}"
    );
}

#[test]
fn smote_rejects_a_weighted_task() {
    let task = weighted_classif_task(10);
    assert_resampling_rejected(Smote::new().with_k_neighbors(1).balance(&task).map(|_| ()).unwrap_err());
}

#[test]
fn adasyn_rejects_a_weighted_task() {
    let task = weighted_classif_task(10);
    assert_resampling_rejected(
        Adasyn::new().with_k_neighbors(1).balance(&task).map(|_| ()).unwrap_err(),
    );
}

#[test]
fn spatial_smote_rejects_a_weighted_task() {
    let task = weighted_classif_task(10);
    let coords: Vec<(f64, f64)> = (0..10).map(|i| (i as f64, 0.0)).collect();
    assert_resampling_rejected(
        SpatialSmote::new()
            .with_k_neighbors(1)
            .balance(&task, &coords)
            .map(|_| ()).unwrap_err(),
    );
}

#[test]
fn pipeline_resampler_stage_rejects_a_weighted_task() {
    let task = weighted_classif_task(10);
    let mut pipe = Pipeline::new(vec![], Box::new(KNearestNeighbors::new(1)))
        .with_resampler(Box::new(Smote::new().with_k_neighbors(1)));
    assert_resampling_rejected(pipe.train_classif(&task).map(|_| ()).unwrap_err());
}
