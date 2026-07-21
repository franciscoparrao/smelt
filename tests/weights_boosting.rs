//! Fase B2 of per-sample weights: real consumption in the external boosting
//! engines (XGBoost, LightGBM, CatBoost).
//!
//! Oracles:
//! 1. **Duplication**: integer weight k ≡ the row physically duplicated k
//!    times, in each engine's most deterministic configuration (no
//!    subsample/GOSS/colsample). Compared with relative tolerance 1e-9,
//!    documented below (`assert_close`): the two runs are mathematically
//!    identical but accumulate gradient/hessian sums in different orders
//!    (k separate additions of `g` vs one addition of `k·g`), so
//!    reassociation-level float drift is expected; anything beyond ulp
//!    scale would mean a split flipped, which the decisive step-function
//!    targets used here rule out.
//! 2. **All-ones ≡ no weights**: bit-identical, all 3 engines, same seed.
//! 3. **Weight 0 ≡ row removed**: XGBoost exact mode bit-identical; the
//!    histogram engines with the same construction (the zero-weight row's
//!    features duplicate an existing row, so bin boundaries are unchanged)
//!    under the 1e-9 tolerance — histogram *subtraction* picks the smaller
//!    sibling by row count, which includes the zero-weight phantom row, so
//!    the scan-vs-subtract side can differ between the two runs and drift
//!    by reassociation ulps.
//! 4. **Builder vs task (XGBoost)**: both set → clear `InvalidParameter`;
//!    builder-only unchanged (its own tests in xgboost.rs still pass);
//!    task-only ≡ builder-only bit-identical.
//! 5. **Semantics**: a 100x-weighted sample drags the *local* prediction
//!    (statistical, wide margins).

use ndarray::Array2;
use smelt_ml::learner::TrainedModel;
use smelt_ml::prelude::*;

// ── helpers ─────────────────────────────────────────────────────────

fn regress_preds(model: &dyn TrainedModel, features: &Array2<f64>) -> Vec<f64> {
    match model.predict(features).unwrap() {
        Prediction::Regression { predicted, .. } => predicted,
        _ => panic!("expected regression prediction"),
    }
}

fn classif_probs(model: &dyn TrainedModel, features: &Array2<f64>) -> Vec<Vec<f64>> {
    match model.predict(features).unwrap() {
        Prediction::Classification {
            probabilities: Some(p),
            ..
        } => p,
        _ => panic!("expected classification prediction with probabilities"),
    }
}

/// Relative tolerance 1e-9 (see module doc, oracle 1): the weighted and
/// duplicated runs accumulate the same sums in different orders.
fn assert_close(a: &[f64], b: &[f64], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length mismatch");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let tol = 1e-9 * x.abs().max(y.abs()).max(1.0);
        assert!(
            (x - y).abs() <= tol,
            "{what}: row {i} differs beyond reassociation tolerance: {x} vs {y}"
        );
    }
}

fn assert_probs_close(a: &[Vec<f64>], b: &[Vec<f64>], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length mismatch");
    for (i, (pa, pb)) in a.iter().zip(b).enumerate() {
        assert_close(pa, pb, &format!("{what} (row {i} probs)"));
    }
}

/// Base fixture for the duplication oracle: 50 rows, 2 low-cardinality
/// features (10 and 7 unique integer values — few enough that every engine's
/// binning gives each value its own bin, so weighted and duplicated runs see
/// identical bin boundaries), decisive integer step targets, integer weights
/// in {1, 2, 3}.
fn dup_fixture() -> (Array2<f64>, Vec<f64>, Vec<usize>, Vec<f64>) {
    let n = 50;
    let features = Array2::from_shape_fn((n, 2), |(i, j)| {
        if j == 0 {
            (i % 10) as f64
        } else {
            ((i * 3) % 7) as f64
        }
    });
    let regress_target: Vec<f64> = (0..n)
        .map(|i| {
            let x0 = (i % 10) as f64;
            let x1 = ((i * 3) % 7) as f64;
            (if x0 >= 5.0 { 10.0 } else { 0.0 }) + (if x1 >= 3.0 { 3.0 } else { 0.0 })
        })
        .collect();
    let classif_target: Vec<usize> = (0..n).map(|i| usize::from(i % 10 >= 5)).collect();
    let weights: Vec<f64> = (0..n).map(|i| (1 + i % 3) as f64).collect();
    (features, regress_target, classif_target, weights)
}

/// Physically duplicate row i `weights[i]` times (contiguously).
fn duplicate_rows(features: &Array2<f64>, weights: &[f64]) -> (Array2<f64>, Vec<usize>) {
    let mut rows: Vec<f64> = Vec::new();
    let mut origin: Vec<usize> = Vec::new();
    for (i, &w) in weights.iter().enumerate().take(features.nrows()) {
        for _ in 0..w as usize {
            rows.extend(features.row(i).iter().copied());
            origin.push(i);
        }
    }
    let n = origin.len();
    (
        Array2::from_shape_vec((n, features.ncols()), rows).unwrap(),
        origin,
    )
}

// ── oracle 1: integer weight k ≡ row duplicated k times ─────────────

fn dup_oracle_regress<F>(train: F, name: &str)
where
    F: Fn(&RegressionTask) -> Box<dyn TrainedModel>,
{
    let (features, target, _, weights) = dup_fixture();
    let (dup_features, origin) = duplicate_rows(&features, &weights);
    let dup_target: Vec<f64> = origin.iter().map(|&i| target[i]).collect();

    let weighted_task = RegressionTask::new("w", features.clone(), target.clone())
        .unwrap()
        .with_weights(weights);
    let dup_task = RegressionTask::new("d", dup_features, dup_target).unwrap();

    let p_w = regress_preds(&*train(&weighted_task), &features);
    let p_d = regress_preds(&*train(&dup_task), &features);
    assert_close(&p_w, &p_d, &format!("{name} regress duplication oracle"));
}

fn dup_oracle_classif<F>(train: F, name: &str)
where
    F: Fn(&ClassificationTask) -> Box<dyn TrainedModel>,
{
    let (features, _, target, weights) = dup_fixture();
    let (dup_features, origin) = duplicate_rows(&features, &weights);
    let dup_target: Vec<usize> = origin.iter().map(|&i| target[i]).collect();

    let weighted_task = ClassificationTask::new("w", features.clone(), target.clone())
        .unwrap()
        .with_weights(weights);
    let dup_task = ClassificationTask::new("d", dup_features, dup_target).unwrap();

    let p_w = classif_probs(&*train(&weighted_task), &features);
    let p_d = classif_probs(&*train(&dup_task), &features);
    assert_probs_close(&p_w, &p_d, &format!("{name} classif duplication oracle"));
}

/// XGBoost runs in exact-greedy mode here (n ≤ n_bins on both sides: 50
/// weighted rows, at most 150 duplicated rows, default n_bins 256), so the
/// only weighted-vs-duplicated divergence is summation order.
#[test]
fn xgboost_integer_weight_equals_duplication_regress() {
    dup_oracle_regress(
        |task| {
            XGBoost::new()
                .with_n_estimators(30)
                .with_max_depth(3)
                .with_learning_rate(0.3)
                .train_regress(task)
                .unwrap()
        },
        "xgboost",
    );
}

#[test]
fn xgboost_integer_weight_equals_duplication_classif() {
    dup_oracle_classif(
        |task| {
            XGBoost::new()
                .with_n_estimators(30)
                .with_max_depth(3)
                .with_learning_rate(0.3)
                .train_classif(task)
                .unwrap()
        },
        "xgboost",
    );
}

/// LightGBM: plain-GBDT defaults (top_rate=1.0/other_rate=0.0 — GOSS off,
/// subsample=1.0, colsample=1.0). Default 255 bins over ≤10 unique feature
/// values means one bin per value: no bin collision, identical boundaries
/// for the weighted and duplicated datasets (`HistBins` builds boundaries
/// from *unique* values, which duplication cannot change).
#[test]
fn lightgbm_integer_weight_equals_duplication_regress() {
    dup_oracle_regress(
        |task| {
            LightGBM::new()
                .with_n_estimators(30)
                .with_num_leaves(8)
                .train_regress(task)
                .unwrap()
        },
        "lightgbm",
    );
}

#[test]
fn lightgbm_integer_weight_equals_duplication_classif() {
    dup_oracle_classif(
        |task| {
            LightGBM::new()
                .with_n_estimators(30)
                .with_num_leaves(8)
                .train_classif(task)
                .unwrap()
        },
        "lightgbm",
    );
}

/// CatBoost: default 64 bins over ≤10 unique values — one bin per value, no
/// collision. Histogram accumulation is f32, but split choices on this
/// decisive step target are far apart in gain, and leaf values are computed
/// from the f64 gradient/hessian sums, so the 1e-9 tolerance holds. No
/// categorical features: ordered target statistics depend on a permutation
/// of row indices, which cannot be equivalent between an n-row weighted and
/// an m-row duplicated dataset (documented scope limit of this oracle).
#[test]
fn catboost_integer_weight_equals_duplication_regress() {
    dup_oracle_regress(
        |task| {
            CatBoost::new()
                .with_n_estimators(30)
                .with_depth(3)
                .train_regress(task)
                .unwrap()
        },
        "catboost",
    );
}

#[test]
fn catboost_integer_weight_equals_duplication_classif() {
    dup_oracle_classif(
        |task| {
            CatBoost::new()
                .with_n_estimators(30)
                .with_depth(3)
                .train_classif(task)
                .unwrap()
        },
        "catboost",
    );
}

// ── oracle 2: all-ones weights ≡ no weights (bit-identical) ─────────

fn all_ones_regress<F>(train: F, name: &str)
where
    F: Fn(&RegressionTask) -> Box<dyn TrainedModel>,
{
    let (features, target, _, _) = dup_fixture();
    let plain = RegressionTask::new("p", features.clone(), target.clone()).unwrap();
    let ones = RegressionTask::new("o", features.clone(), target)
        .unwrap()
        .with_weights(vec![1.0; features.nrows()]);
    assert_eq!(
        regress_preds(&*train(&plain), &features),
        regress_preds(&*train(&ones), &features),
        "{name}: all-ones weights must be bit-identical to no weights (regression)"
    );
}

fn all_ones_classif<F>(train: F, name: &str)
where
    F: Fn(&ClassificationTask) -> Box<dyn TrainedModel>,
{
    let (features, _, target, _) = dup_fixture();
    let plain = ClassificationTask::new("p", features.clone(), target.clone()).unwrap();
    let ones = ClassificationTask::new("o", features.clone(), target)
        .unwrap()
        .with_weights(vec![1.0; features.nrows()]);
    assert_eq!(
        classif_probs(&*train(&plain), &features),
        classif_probs(&*train(&ones), &features),
        "{name}: all-ones weights must be bit-identical to no weights (classification)"
    );
}

#[test]
fn xgboost_all_ones_weights_bit_identical_to_unweighted() {
    all_ones_regress(
        |t| {
            XGBoost::new()
                .with_n_estimators(20)
                .train_regress(t)
                .unwrap()
        },
        "xgboost",
    );
    all_ones_classif(
        |t| {
            XGBoost::new()
                .with_n_estimators(20)
                .train_classif(t)
                .unwrap()
        },
        "xgboost",
    );
}

#[test]
fn lightgbm_all_ones_weights_bit_identical_to_unweighted() {
    all_ones_regress(
        |t| {
            LightGBM::new()
                .with_n_estimators(20)
                .train_regress(t)
                .unwrap()
        },
        "lightgbm",
    );
    all_ones_classif(
        |t| {
            LightGBM::new()
                .with_n_estimators(20)
                .train_classif(t)
                .unwrap()
        },
        "lightgbm",
    );
}

#[test]
fn catboost_all_ones_weights_bit_identical_to_unweighted() {
    all_ones_regress(
        |t| {
            CatBoost::new()
                .with_n_estimators(20)
                .train_regress(t)
                .unwrap()
        },
        "catboost",
    );
    all_ones_classif(
        |t| {
            CatBoost::new()
                .with_n_estimators(20)
                .train_classif(t)
                .unwrap()
        },
        "catboost",
    );
}

// ── oracle 3: weight 0 ≡ row removed ────────────────────────────────

/// `(base_features, base_target, poisoned_features, poisoned_target, weights)`
/// for the weight-0 oracle.
type ZeroWeightFixture = (Array2<f64>, Vec<f64>, Array2<f64>, Vec<f64>, Vec<f64>);

/// 30 clean rows plus one poison row whose *features duplicate row 7* (so
/// bin boundaries / exact-mode split candidates are unchanged by its
/// presence) but whose target is wildly wrong. With weight 0 it must be as
/// if the row were never there.
fn zero_weight_fixture() -> ZeroWeightFixture {
    let n = 30;
    let base = Array2::from_shape_fn((n, 1), |(i, _)| (i % 10) as f64);
    let target: Vec<f64> = (0..n)
        .map(|i| if (i % 10) as f64 >= 5.0 { 10.0 } else { 0.0 })
        .collect();

    let mut rows: Vec<f64> = base.iter().copied().collect();
    rows.push(base[[7, 0]]); // phantom row: same features as row 7
    let mut poisoned_target = target.clone();
    poisoned_target.push(999.0); // wildly wrong target, weight 0
    let mut weights = vec![1.0; n];
    weights.push(0.0);
    let poisoned = Array2::from_shape_vec((n + 1, 1), rows).unwrap();
    (base, target, poisoned, poisoned_target, weights)
}

/// XGBoost exact mode (n = 31 ≤ 256): bit-identical.
#[test]
fn xgboost_zero_weight_row_bit_identical_to_removed_row() {
    let (base, target, poisoned, poisoned_target, weights) = zero_weight_fixture();
    let removed = RegressionTask::new("r", base.clone(), target).unwrap();
    let zeroed = RegressionTask::new("z", poisoned, poisoned_target)
        .unwrap()
        .with_weights(weights);

    let cfg = || XGBoost::new().with_n_estimators(25).with_max_depth(3);
    assert_eq!(
        regress_preds(&*cfg().train_regress(&removed).unwrap(), &base),
        regress_preds(&*cfg().train_regress(&zeroed).unwrap(), &base),
        "xgboost exact mode: a weight-0 row must be bit-identical to its absence"
    );
}

#[test]
fn xgboost_zero_weight_row_classif_matches_removed_row() {
    let (base, target, poisoned, _, weights) = zero_weight_fixture();
    let labels: Vec<usize> = target.iter().map(|&y| usize::from(y > 5.0)).collect();
    let mut poisoned_labels = labels.clone();
    poisoned_labels.push(0); // wrong label (row 7's group is label 1), weight 0
    let removed = ClassificationTask::new("r", base.clone(), labels).unwrap();
    let zeroed = ClassificationTask::new("z", poisoned, poisoned_labels)
        .unwrap()
        .with_weights(weights);

    let cfg = || XGBoost::new().with_n_estimators(25).with_max_depth(3);
    assert_eq!(
        classif_probs(&*cfg().train_classif(&removed).unwrap(), &base),
        classif_probs(&*cfg().train_classif(&zeroed).unwrap(), &base),
        "xgboost exact mode: a weight-0 row must be bit-identical to its absence (classif)"
    );
}

/// Histogram engines: same construction (phantom features duplicate an
/// existing row, so `HistBins` boundaries are unchanged). Compared under the
/// 1e-9 tolerance: the phantom row contributes exactly 0.0 to every
/// histogram, but it *does* count toward the row-count that picks which
/// sibling gets scanned vs derived by subtraction, so the two runs may
/// differ by float reassociation.
#[test]
fn lightgbm_zero_weight_row_matches_removed_row() {
    let (base, target, poisoned, poisoned_target, weights) = zero_weight_fixture();
    let removed = RegressionTask::new("r", base.clone(), target).unwrap();
    let zeroed = RegressionTask::new("z", poisoned, poisoned_target)
        .unwrap()
        .with_weights(weights);

    let cfg = || LightGBM::new().with_n_estimators(25).with_num_leaves(8);
    assert_close(
        &regress_preds(&*cfg().train_regress(&removed).unwrap(), &base),
        &regress_preds(&*cfg().train_regress(&zeroed).unwrap(), &base),
        "lightgbm: weight-0 row vs removed row",
    );
}

#[test]
fn catboost_zero_weight_row_matches_removed_row() {
    let (base, target, poisoned, poisoned_target, weights) = zero_weight_fixture();
    let removed = RegressionTask::new("r", base.clone(), target).unwrap();
    let zeroed = RegressionTask::new("z", poisoned, poisoned_target)
        .unwrap()
        .with_weights(weights);

    let cfg = || CatBoost::new().with_n_estimators(25).with_depth(3);
    assert_close(
        &regress_preds(&*cfg().train_regress(&removed).unwrap(), &base),
        &regress_preds(&*cfg().train_regress(&zeroed).unwrap(), &base),
        "catboost: weight-0 row vs removed row",
    );
}

// ── oracle 4: XGBoost builder weights vs task weights ───────────────

#[test]
fn xgboost_rejects_weights_from_both_builder_and_task() {
    let (features, target, labels, weights) = dup_fixture();

    let rtask = RegressionTask::new("r", features.clone(), target)
        .unwrap()
        .with_weights(weights.clone());
    let err = XGBoost::new()
        .with_sample_weights(weights.clone())
        .train_regress(&rtask)
        .map(|_| ())
        .unwrap_err();
    let msg = format!("{err}");
    assert!(
        matches!(err, SmeltError::InvalidParameter(_)) && msg.contains("both"),
        "conflict must be a clear InvalidParameter naming both routes: {msg}"
    );

    let ctask = ClassificationTask::new("c", features, labels)
        .unwrap()
        .with_weights(weights.clone());
    let err = XGBoost::new()
        .with_sample_weights(weights)
        .train_classif(&ctask)
        .map(|_| ())
        .unwrap_err();
    assert!(
        format!("{err}").contains("both"),
        "classif conflict must error too: {err}"
    );
}

#[test]
fn xgboost_task_weights_bit_identical_to_builder_weights() {
    let (features, target, labels, weights) = dup_fixture();

    // regression
    let task_route = RegressionTask::new("t", features.clone(), target.clone())
        .unwrap()
        .with_weights(weights.clone());
    let builder_task = RegressionTask::new("b", features.clone(), target).unwrap();
    let p_task = regress_preds(
        &*XGBoost::new()
            .with_n_estimators(25)
            .train_regress(&task_route)
            .unwrap(),
        &features,
    );
    let p_builder = regress_preds(
        &*XGBoost::new()
            .with_n_estimators(25)
            .with_sample_weights(weights.clone())
            .train_regress(&builder_task)
            .unwrap(),
        &features,
    );
    assert_eq!(
        p_task, p_builder,
        "task-route and builder-route weights must produce the identical model (regression)"
    );

    // classification
    let task_route = ClassificationTask::new("t", features.clone(), labels.clone())
        .unwrap()
        .with_weights(weights.clone());
    let builder_task = ClassificationTask::new("b", features.clone(), labels).unwrap();
    let p_task = classif_probs(
        &*XGBoost::new()
            .with_n_estimators(25)
            .train_classif(&task_route)
            .unwrap(),
        &features,
    );
    let p_builder = classif_probs(
        &*XGBoost::new()
            .with_n_estimators(25)
            .with_sample_weights(weights)
            .train_classif(&builder_task)
            .unwrap(),
        &features,
    );
    assert_eq!(
        p_task, p_builder,
        "task-route and builder-route weights must produce the identical model (classification)"
    );
}

// ── oracle 5: a 100x-weighted sample drags the local prediction ─────

/// 40 clean step-function rows plus one contrarian point at x=5 with target
/// 10 (the local consensus there is 0). Unweighted, the model must mostly
/// ignore it; with weight 100 it must dominate its neighborhood — while the
/// rest of the curve (e.g. x=2) stays put.
fn local_drag_fixture() -> (Array2<f64>, Vec<f64>, Vec<f64>) {
    let n = 41;
    let mut xs: Vec<f64> = (0..40).map(|i| i as f64).collect();
    let mut target: Vec<f64> = xs
        .iter()
        .map(|&x| if x < 20.0 { 0.0 } else { 10.0 })
        .collect();
    xs.push(5.0);
    target.push(10.0); // contrarian point
    let mut weights = vec![1.0; n];
    weights[n - 1] = 100.0;
    (Array2::from_shape_vec((n, 1), xs).unwrap(), target, weights)
}

fn local_drag_check<F>(train: F, name: &str)
where
    F: Fn(&RegressionTask) -> Box<dyn TrainedModel>,
{
    let (features, target, weights) = local_drag_fixture();
    let probe = Array2::from_shape_vec((2, 1), vec![5.0, 12.0]).unwrap();

    let plain = RegressionTask::new("p", features.clone(), target.clone()).unwrap();
    let p0 = regress_preds(&*train(&plain), &probe);
    assert!(
        p0[0] < 6.0,
        "{name}: unweighted, the single contrarian point must not dominate x=5, got {}",
        p0[0]
    );

    let weighted = RegressionTask::new("w", features, target)
        .unwrap()
        .with_weights(weights);
    let p1 = regress_preds(&*train(&weighted), &probe);
    assert!(
        p1[0] > 6.0,
        "{name}: with weight 100 the contrarian point must drag x=5 toward 10, got {}",
        p1[0]
    );
    assert!(
        p1[1] < 3.0,
        "{name}: the drag must stay local — x=12 should remain near 0, got {}",
        p1[1]
    );
}

#[test]
fn xgboost_heavy_weight_drags_local_prediction() {
    local_drag_check(
        |t| {
            XGBoost::new()
                .with_n_estimators(60)
                .with_max_depth(4)
                .with_learning_rate(0.3)
                .train_regress(t)
                .unwrap()
        },
        "xgboost",
    );
}

#[test]
fn lightgbm_heavy_weight_drags_local_prediction() {
    local_drag_check(
        |t| {
            LightGBM::new()
                .with_n_estimators(60)
                .with_num_leaves(16)
                .with_learning_rate(0.3)
                .train_regress(t)
                .unwrap()
        },
        "lightgbm",
    );
}

#[test]
fn catboost_heavy_weight_drags_local_prediction() {
    local_drag_check(
        |t| {
            CatBoost::new()
                .with_n_estimators(60)
                .with_depth(4)
                .with_learning_rate(0.3)
                .train_regress(t)
                .unwrap()
        },
        "catboost",
    );
}

/// GOSS composes with sample weights (weights scale the gradients FIRST,
/// then GOSS's top-|gradient| selection sees the weighted values, as in the
/// official implementation): the heavy contrarian point must still win its
/// neighborhood with GOSS enabled at the paper's rates.
#[test]
fn lightgbm_goss_operates_on_weighted_gradients() {
    local_drag_check(
        |t| {
            LightGBM::new()
                .with_n_estimators(60)
                .with_num_leaves(16)
                .with_learning_rate(0.3)
                .with_top_rate(0.2)
                .with_other_rate(0.1)
                .train_regress(t)
                .unwrap()
        },
        "lightgbm+goss",
    );
}

// ── classification prior shift via task weights ─────────────────────

fn classif_prior_shift<F>(train: F, name: &str)
where
    F: Fn(&ClassificationTask) -> Box<dyn TrainedModel>,
{
    let n = 20;
    let features = Array2::<f64>::zeros((n, 1)); // constant feature: no split possible
    let target: Vec<usize> = (0..n).map(|i| usize::from(i >= n / 2)).collect();

    let plain = ClassificationTask::new("p", features.clone(), target.clone()).unwrap();
    let p0 = classif_probs(&*train(&plain), &features)[0][1];
    assert!(
        (0.3..=0.7).contains(&p0),
        "{name}: unweighted P(class=1) should be ~0.5, got {p0}"
    );

    // Class 1 weighted 9x -> weighted positive fraction 0.9.
    let weights: Vec<f64> = (0..n).map(|i| if i >= n / 2 { 9.0 } else { 1.0 }).collect();
    let weighted = ClassificationTask::new("w", features.clone(), target)
        .unwrap()
        .with_weights(weights);
    let p1 = classif_probs(&*train(&weighted), &features)[0][1];
    assert!(
        p1 > 0.75,
        "{name}: 9x class-1 weights should pull P(class=1) toward 0.9, got {p1}"
    );
}

#[test]
fn xgboost_task_weights_shift_classification() {
    classif_prior_shift(
        |t| {
            XGBoost::new()
                .with_n_estimators(30)
                .train_classif(t)
                .unwrap()
        },
        "xgboost",
    );
}

#[test]
fn lightgbm_task_weights_shift_classification() {
    classif_prior_shift(
        |t| {
            LightGBM::new()
                .with_n_estimators(30)
                .train_classif(t)
                .unwrap()
        },
        "lightgbm",
    );
}

#[test]
fn catboost_task_weights_shift_classification() {
    classif_prior_shift(
        |t| {
            CatBoost::new()
                .with_n_estimators(30)
                .train_classif(t)
                .unwrap()
        },
        "catboost",
    );
}

// ── trait metadata ──────────────────────────────────────────────────

#[test]
fn boosting_engines_declare_weight_support() {
    assert!(XGBoost::new().supports_weights());
    assert!(LightGBM::new().supports_weights());
    assert!(CatBoost::new().supports_weights());
    // trait-object dispatch (the registry-properties seam)
    let l: Box<dyn Learner> = Box::new(XGBoost::new());
    assert!(l.supports_weights());
}
