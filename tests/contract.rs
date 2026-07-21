//! Contract autotest — the smelt analogue of mlr3's learner `autotest`.
//!
//! For every id in [`registered_learner_ids`], this constructs the learner and
//! checks that each field of its declared [`LearnerProperties`] matches the
//! learner's actually-observable behaviour. A learner whose metadata lies — or
//! whose behaviour drifts away from its metadata — fails here loudly, which is
//! the entire point: the properties table is only worth having if it is true.
//!
//! Each test loops over all learners and accumulates *every* mismatch before
//! failing, so one run surfaces the full list of offenders rather than just the
//! first. Panics inside `predict` are caught and reported as failures too (the
//! crate's zero-panic-on-bad-input standard).

use ndarray::Array2;
use smelt_ml::learner::{TrainedModel, learner_from_id, registered_learner_ids};
use smelt_ml::prelude::*;
use std::panic::{AssertUnwindSafe, catch_unwind};

const N: usize = 40;
const F: usize = 3;

/// Classification data with clear signal in feature 0 (so any tree splits and
/// reports non-zero importance) plus secondary signal in features 1/2.
fn classif_features_target() -> (Array2<f64>, Vec<usize>) {
    let mut x = Array2::zeros((N, F));
    let mut y = vec![0usize; N];
    for i in 0..N {
        let c = i % 2;
        y[i] = c;
        x[[i, 0]] = c as f64 + 0.1 * (i % 5) as f64; // strongly separates the two classes
        x[[i, 1]] = (c as f64) * 2.0 - 0.05 * i as f64;
        x[[i, 2]] = 0.3 * i as f64 - c as f64;
    }
    (x, y)
}

/// Regression data with a genuine linear relationship, so coefficient- and
/// impurity-based importances are non-zero for a normal fit.
fn regress_features_target() -> (Array2<f64>, Vec<f64>) {
    let mut x = Array2::zeros((N, F));
    let mut y = vec![0.0; N];
    for i in 0..N {
        let a = i as f64;
        x[[i, 0]] = a;
        x[[i, 1]] = (a * 0.5).sin();
        x[[i, 2]] = a * a * 0.01;
        y[i] = 2.0 * x[[i, 0]] + 3.0 * x[[i, 1]] - x[[i, 2]] + 1.0;
    }
    (x, y)
}

fn classif_task() -> ClassificationTask {
    let (x, y) = classif_features_target();
    ClassificationTask::new("contract_cls", x, y).unwrap()
}

fn regress_task() -> RegressionTask {
    let (x, y) = regress_features_target();
    RegressionTask::new("contract_reg", x, y).unwrap()
}

/// Train on whichever task type the learner declares support for (classification
/// preferred when both), returning the fresh trained model.
fn train_supported(id: &str, props: LearnerProperties) -> Box<dyn TrainedModel> {
    let mut l = learner_from_id(id).unwrap();
    if props.supports_classification {
        l.train_classif(&classif_task())
            .unwrap_or_else(|e| panic!("{id}: train_classif failed unexpectedly: {e}"))
    } else {
        l.train_regress(&regress_task())
            .unwrap_or_else(|e| panic!("{id}: train_regress failed unexpectedly: {e}"))
    }
}

fn report(name: &str, failures: Vec<String>) {
    assert!(
        failures.is_empty(),
        "\n{name}: {} contract violation(s):\n  - {}\n",
        failures.len(),
        failures.join("\n  - ")
    );
}

/// `supports_classification` / `supports_regression` must match which of
/// `train_classif`/`train_regress` actually succeeds vs. returns the default
/// "does not support X" error. This is the machine version of the manual
/// task-heuristic audits.
#[test]
fn contract_task_support() {
    let mut failures = Vec::new();
    for &id in registered_learner_ids() {
        let props = learner_from_id(id).unwrap().properties();

        let mut lc = learner_from_id(id).unwrap();
        match (
            props.supports_classification,
            lc.train_classif(&classif_task()),
        ) {
            (true, Err(e)) => failures.push(format!(
                "{id}: declares classification but train_classif errored: {e}"
            )),
            (false, Ok(_)) => failures.push(format!(
                "{id}: does NOT declare classification but train_classif succeeded"
            )),
            _ => {}
        }

        let mut lr = learner_from_id(id).unwrap();
        match (props.supports_regression, lr.train_regress(&regress_task())) {
            (true, Err(e)) => failures.push(format!(
                "{id}: declares regression but train_regress errored: {e}"
            )),
            (false, Ok(_)) => failures.push(format!(
                "{id}: does NOT declare regression but train_regress succeeded"
            )),
            _ => {}
        }
    }
    report("contract_task_support", failures);
}

/// `supports_weights` (the property) must equal `Learner::supports_weights()`
/// (non-divergence — they share one source of truth) AND match observable
/// behaviour: a weighted task trains when `true`, and is rejected by
/// `check_no_weights` when `false`.
#[test]
fn contract_weights() {
    let mut failures = Vec::new();
    for &id in registered_learner_ids() {
        let l = learner_from_id(id).unwrap();
        let props = l.properties();

        if l.supports_weights() != props.supports_weights {
            failures.push(format!(
                "{id}: supports_weights() = {} but properties().supports_weights = {} (divergent!)",
                l.supports_weights(),
                props.supports_weights
            ));
        }

        let weights = vec![1.0_f64; N];
        let result = if props.supports_classification {
            let (x, y) = classif_features_target();
            let task = ClassificationTask::new("w", x, y)
                .unwrap()
                .with_weights(weights);
            learner_from_id(id)
                .unwrap()
                .train_classif(&task)
                .map(|_| ())
        } else {
            let (x, y) = regress_features_target();
            let task = RegressionTask::new("w", x, y)
                .unwrap()
                .with_weights(weights);
            learner_from_id(id)
                .unwrap()
                .train_regress(&task)
                .map(|_| ())
        };

        match (props.supports_weights, result) {
            (true, Err(e)) => {
                failures.push(format!("{id}: declares weights but a weighted task errored: {e}"))
            }
            (false, Ok(())) => failures.push(format!(
                "{id}: does NOT declare weights but a weighted task trained (weights silently ignored?)"
            )),
            _ => {}
        }
    }
    report("contract_weights", failures);
}

/// `supports_nan`: a task with a NaN feature trains when `true`, and is
/// rejected by `check_no_nan` when `false`.
#[test]
fn contract_nan() {
    let mut failures = Vec::new();
    for &id in registered_learner_ids() {
        let props = learner_from_id(id).unwrap().properties();

        let result = if props.supports_classification {
            let (mut x, y) = classif_features_target();
            x[[0, 0]] = f64::NAN;
            let task = ClassificationTask::new("nan", x, y).unwrap();
            learner_from_id(id)
                .unwrap()
                .train_classif(&task)
                .map(|_| ())
        } else {
            let (mut x, y) = regress_features_target();
            x[[0, 0]] = f64::NAN;
            let task = RegressionTask::new("nan", x, y).unwrap();
            learner_from_id(id)
                .unwrap()
                .train_regress(&task)
                .map(|_| ())
        };

        match (props.supports_nan, result) {
            (true, Err(e)) => failures.push(format!(
                "{id}: declares NaN support but a NaN feature errored: {e}"
            )),
            (false, Ok(())) => failures.push(format!(
                "{id}: does NOT declare NaN support but trained on a NaN feature"
            )),
            _ => {}
        }
    }
    report("contract_nan", failures);
}

/// `supports_proba` (classification only): the prediction carries a valid
/// per-class distribution (`Some`, width `n_classes`, rows sum to ~1, argmax ==
/// hard label) when `true`, or only hard labels (`None`) when `false`. We do
/// NOT assert probabilities are fractional — a pure leaf legitimately yields a
/// one-hot row on separable data — only that a declared distribution is present
/// and consistent with the label.
#[test]
fn contract_proba() {
    let mut failures = Vec::new();
    for &id in registered_learner_ids() {
        let props = learner_from_id(id).unwrap().properties();
        if !props.supports_classification {
            continue;
        }
        let mut l = learner_from_id(id).unwrap();
        let task = classif_task();
        let model = match l.train_classif(&task) {
            Ok(m) => m,
            Err(e) => {
                failures.push(format!(
                    "{id}: train_classif errored during proba check: {e}"
                ));
                continue;
            }
        };
        let pred = match model.predict(task.features()) {
            Ok(p) => p,
            Err(e) => {
                failures.push(format!("{id}: predict errored during proba check: {e}"));
                continue;
            }
        };
        let Prediction::Classification {
            predicted,
            probabilities,
            ..
        } = pred
        else {
            failures.push(format!(
                "{id}: classification predict did not return a Classification"
            ));
            continue;
        };
        match (props.supports_proba, probabilities) {
            (true, None) => failures.push(format!(
                "{id}: declares proba but predict returned probabilities = None"
            )),
            (true, Some(rows)) => {
                let n_classes = task.n_classes();
                for (i, row) in rows.iter().enumerate() {
                    if row.len() != n_classes {
                        failures.push(format!(
                            "{id}: proba row {i} width {} != n_classes {n_classes}",
                            row.len()
                        ));
                        break;
                    }
                    let sum: f64 = row.iter().sum();
                    if (sum - 1.0).abs() > 1e-3 {
                        failures.push(format!("{id}: proba row {i} sums to {sum}, not ~1"));
                        break;
                    }
                    let argmax = row
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(k, _)| k)
                        .unwrap();
                    if argmax != predicted[i] {
                        failures.push(format!(
                            "{id}: proba argmax {argmax} != hard label {} at row {i}",
                            predicted[i]
                        ));
                        break;
                    }
                }
            }
            (false, Some(_)) => failures.push(format!(
                "{id}: does NOT declare proba but predict returned Some(probabilities)"
            )),
            (false, None) => {}
        }
    }
    report("contract_proba", failures);
}

/// `provides_feature_importance`: after a normal fit, `feature_importance()` is
/// `Some` with exactly `n_features` entries (the positional contract RFE
/// relies on) when `true`, or `None` when `false`.
#[test]
fn contract_feature_importance() {
    let mut failures = Vec::new();
    for &id in registered_learner_ids() {
        let props = learner_from_id(id).unwrap().properties();
        let model = train_supported(id, props);
        match (
            props.provides_feature_importance,
            model.feature_importance(),
        ) {
            (true, None) => failures.push(format!(
                "{id}: declares feature importance but feature_importance() returned None"
            )),
            (true, Some(fi)) => {
                if fi.len() != F {
                    failures.push(format!(
                        "{id}: feature_importance() len {} != n_features {F}",
                        fi.len()
                    ));
                }
            }
            (false, Some(_)) => failures.push(format!(
                "{id}: does NOT declare feature importance but feature_importance() returned Some"
            )),
            (false, None) => {}
        }
    }
    report("contract_feature_importance", failures);
}

/// `serializable`: `to_serializable()` is `Some` after a fit when `true`,
/// `None` when `false`.
#[test]
fn contract_serializable() {
    let mut failures = Vec::new();
    for &id in registered_learner_ids() {
        let props = learner_from_id(id).unwrap().properties();
        let model = train_supported(id, props);
        match (props.serializable, model.to_serializable().is_some()) {
            (true, false) => failures.push(format!(
                "{id}: declares serializable but to_serializable() returned None"
            )),
            (false, true) => failures.push(format!(
                "{id}: does NOT declare serializable but to_serializable() returned Some"
            )),
            _ => {}
        }
    }
    report("contract_serializable", failures);
}

/// Universal zero-panic contract: `predict` on a feature matrix with the wrong
/// number of columns must return `Err`, never panic, for every learner.
#[test]
fn contract_predict_wrong_features_errors_not_panics() {
    let mut failures = Vec::new();
    for &id in registered_learner_ids() {
        let props = learner_from_id(id).unwrap().properties();
        let model = train_supported(id, props);
        let wrong = Array2::<f64>::zeros((2, F + 1));
        match catch_unwind(AssertUnwindSafe(|| model.predict(&wrong))) {
            Err(_) => failures.push(format!(
                "{id}: predict PANICKED on wrong n_features (must return Err instead)"
            )),
            Ok(Ok(_)) => failures.push(format!(
                "{id}: predict accepted a wrong-width feature matrix instead of erroring"
            )),
            Ok(Err(_)) => {}
        }
    }
    report(
        "contract_predict_wrong_features_errors_not_panics",
        failures,
    );
}
