//! Fase B3 of per-sample weights: the LINEAR models consume weights.
//!
//! Covered learners: LinearRegression (weighted normal equations),
//! Ridge (weighted normal equations, unscaled penalty — sklearn's
//! convention), Lasso / ElasticNet (weighted coordinate descent normalized
//! by total weight Σw, algebraically identical to sklearn's "rescale
//! sample_weight to sum to n_samples" convention), LogisticRegression
//! (weighted gradient + weighted standardization statistics) and
//! ExtremeLearningMachine (weighted ridge output solve + weighted
//! standardization statistics; the seeded random projection is unchanged).
//! LinearSVM is deliberately NOT weight-aware in this phase (per-sample
//! shuffled SGD has no duplication-exact weighted form) and must keep its
//! guard.
//!
//! Oracles, for every covered learner:
//! 1. integer weight k ≡ row duplicated k times (exact for hand-rolled
//!    accumulations, ≤1e-9 for paths through ndarray `dot`/the Gaussian
//!    solver, where summation-order ulps are amplified by the solve);
//! 2. all-ones weights ≡ no weights, bit-identical;
//! 3. weight 0 ≡ row absent (same tolerances as oracle 1, and the zeroed
//!    row's values may be arbitrary finite garbage);
//! 4. golden: weighted OLS/Ridge/Lasso against sklearn 1.8.0
//!    `fit(X, y, sample_weight=w)` predictions.
//!
//! All fixtures use ≥2 feature columns on purpose: the bit-identity of
//! oracle 2 for the standardizing learners relies on ndarray's sequential
//! column folds, which is the code path for non-contiguous columns.

use ndarray::Array2;
use smelt_ml::learner::TrainedModel;
use smelt_ml::prelude::*;

// ── fixtures ─────────────────────────────────────────────────────────────

/// Deterministic, well-conditioned regression data: 3 features, linear
/// signal plus a small deterministic "noise" term.
fn regress_data(n: usize) -> (Array2<f64>, Vec<f64>) {
    let features = Array2::from_shape_fn((n, 3), |(i, j)| {
        let t = i as f64;
        match j {
            0 => (t * 0.37).sin() * 2.0 + t * 0.1,
            1 => (t * 0.71).cos() + 1.5,
            _ => (t * 0.13).sin() * (t * 0.29).cos(),
        }
    });
    let target: Vec<f64> = (0..n)
        .map(|i| {
            let x = features.row(i);
            1.5 * x[0] - 2.0 * x[1] + 0.7 * x[2] + 0.3 * (i as f64 * 0.53).sin() + 2.0
        })
        .collect();
    (features, target)
}

/// Deterministic, NON-separable binary classification data (2 features).
/// Non-separable keeps logistic regression's optimum finite.
fn classif_data(n: usize) -> (Array2<f64>, Vec<usize>) {
    let features = Array2::from_shape_fn((n, 2), |(i, j)| {
        let t = i as f64;
        if j == 0 { (t * 0.41).sin() + t * 0.05 } else { (t * 0.67).cos() * 1.3 }
    });
    let target: Vec<usize> = (0..n)
        .map(|i| {
            let x = features.row(i);
            // linear boundary with deterministic label flips (~each 7th)
            let raw = x[0] + 0.8 * x[1] > 0.6;
            if i % 7 == 3 { usize::from(!raw) } else { usize::from(raw) }
        })
        .collect();
    (features, target)
}

/// Integer duplication counts 1..=3, at least one of each.
fn dup_counts(n: usize) -> Vec<usize> {
    (0..n).map(|i| i % 3 + 1).collect()
}

/// Expands rows in place: row i appears counts[i] consecutive times.
fn expand<T: Clone>(features: &Array2<f64>, target: &[T], counts: &[usize]) -> (Array2<f64>, Vec<T>) {
    let total: usize = counts.iter().sum();
    let p = features.ncols();
    let mut out = Array2::zeros((total, p));
    let mut t_out = Vec::with_capacity(total);
    let mut r = 0;
    for (i, &c) in counts.iter().enumerate() {
        for _ in 0..c {
            for j in 0..p {
                out[[r, j]] = features[[i, j]];
            }
            t_out.push(target[i].clone());
            r += 1;
        }
    }
    (out, t_out)
}

// ── prediction extraction / comparison helpers ───────────────────────────

fn reg_preds(model: &dyn TrainedModel, x: &Array2<f64>) -> Vec<f64> {
    match model.predict(x).unwrap() {
        Prediction::Regression { predicted, .. } => predicted,
        _ => panic!("expected regression prediction"),
    }
}

fn classif_probs(model: &dyn TrainedModel, x: &Array2<f64>) -> Vec<Vec<f64>> {
    match model.predict(x).unwrap() {
        Prediction::Classification { probabilities, .. } => {
            probabilities.expect("probabilities expected")
        }
        _ => panic!("expected classification prediction"),
    }
}

/// |a-b| ≤ tol·max(1, |a|, |b|) elementwise.
fn assert_close(a: &[f64], b: &[f64], tol: f64, ctx: &str) {
    assert_eq!(a.len(), b.len(), "{ctx}: length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        let scale = 1.0_f64.max(x.abs()).max(y.abs());
        assert!(
            (x - y).abs() <= tol * scale,
            "{ctx}: element {i} differs beyond tol {tol}: {x} vs {y} (diff {})",
            (x - y).abs()
        );
    }
}

fn assert_probs_close(a: &[Vec<f64>], b: &[Vec<f64>], tol: f64, ctx: &str) {
    assert_eq!(a.len(), b.len(), "{ctx}: row count mismatch");
    for (i, (ra, rb)) in a.iter().zip(b).enumerate() {
        assert_close(ra, rb, tol, &format!("{ctx} (row {i})"));
    }
}

// ── the regression learner roster ────────────────────────────────────────

/// (name, factory, duplication/zero-weight tolerance) roster entry.
type LearnerEntry = (&'static str, Box<dyn Fn() -> Box<dyn Learner>>, f64);

/// Iterative learners are configured to converge far past the comparison
/// tolerance.
fn regress_learners() -> Vec<LearnerEntry> {
    vec![
        ("linear_regression", Box::new(|| Box::new(LinearRegression::new()) as Box<dyn Learner>), 1e-9),
        ("ridge", Box::new(|| Box::new(Ridge::new(0.5)) as Box<dyn Learner>), 1e-9),
        (
            "lasso",
            Box::new(|| {
                Box::new(Lasso::new(0.05).with_tol(1e-12).with_max_iter(200_000)) as Box<dyn Learner>
            }),
            1e-8,
        ),
        (
            "elastic_net",
            Box::new(|| {
                Box::new(ElasticNet::new(0.05, 0.5).with_tol(1e-12).with_max_iter(200_000))
                    as Box<dyn Learner>
            }),
            1e-8,
        ),
        (
            "elm",
            Box::new(|| {
                Box::new(ExtremeLearningMachine::new().with_n_hidden(16).with_seed(7))
                    as Box<dyn Learner>
            }),
            1e-7,
        ),
    ]
}

// ── oracle 1: integer weight k ≡ row duplicated k times ─────────────────

#[test]
fn regression_weight_k_equals_k_duplicated_rows() {
    let n = 24;
    let (features, target) = regress_data(n);
    let counts = dup_counts(n);
    let weights: Vec<f64> = counts.iter().map(|&c| c as f64).collect();

    let weighted_task = RegressionTask::new("w", features.clone(), target.clone())
        .unwrap()
        .with_weights(weights);
    let (dup_features, dup_target) = expand(&features, &target, &counts);
    let dup_task = RegressionTask::new("d", dup_features, dup_target).unwrap();

    for (name, factory, tol) in regress_learners() {
        let m_w = factory().train_regress(&weighted_task).unwrap();
        let m_d = factory().train_regress(&dup_task).unwrap();
        assert_close(
            &reg_preds(m_w.as_ref(), &features),
            &reg_preds(m_d.as_ref(), &features),
            tol,
            &format!("{name}: weight-k vs k-duplicates"),
        );
    }
}

#[test]
fn classification_weight_k_equals_k_duplicated_rows() {
    let n = 24;
    let (features, target) = classif_data(n);
    let counts = dup_counts(n);
    let weights: Vec<f64> = counts.iter().map(|&c| c as f64).collect();

    let weighted_task = ClassificationTask::new("w", features.clone(), target.clone())
        .unwrap()
        .with_weights(weights);
    let (dup_features, dup_target) = expand(&features, &target, &counts);
    let dup_task = ClassificationTask::new("d", dup_features, dup_target).unwrap();

    // LogisticRegression: tol tiny enough that early stopping never fires,
    // so both runs execute exactly max_iter identical GD steps.
    let mk_logreg = || LogisticRegression::new().with_tol(1e-14).with_max_iter(2000);
    let m_w = mk_logreg().train_classif(&weighted_task).unwrap();
    let m_d = mk_logreg().train_classif(&dup_task).unwrap();
    assert_probs_close(
        &classif_probs(m_w.as_ref(), &features),
        &classif_probs(m_d.as_ref(), &features),
        1e-9,
        "logistic_regression: weight-k vs k-duplicates",
    );

    // ELM classification (same seed ⇒ same random projection).
    let mk_elm = || ExtremeLearningMachine::new().with_n_hidden(16).with_seed(11);
    let m_w = mk_elm().train_classif(&weighted_task).unwrap();
    let m_d = mk_elm().train_classif(&dup_task).unwrap();
    assert_probs_close(
        &classif_probs(m_w.as_ref(), &features),
        &classif_probs(m_d.as_ref(), &features),
        1e-7,
        "elm: weight-k vs k-duplicates",
    );
}

// ── oracle 2: all-ones weights ≡ no weights, bit-identical ──────────────

#[test]
fn regression_all_ones_weights_are_bit_identical_to_unweighted() {
    let n = 24;
    let (features, target) = regress_data(n);
    let plain = RegressionTask::new("p", features.clone(), target.clone()).unwrap();
    let ones = RegressionTask::new("o", features.clone(), target)
        .unwrap()
        .with_weights(vec![1.0; n]);

    for (name, factory, _) in regress_learners() {
        let m_p = factory().train_regress(&plain).unwrap();
        let m_o = factory().train_regress(&ones).unwrap();
        assert_eq!(
            reg_preds(m_p.as_ref(), &features),
            reg_preds(m_o.as_ref(), &features),
            "{name}: all-ones weights must be bit-identical to unweighted"
        );
    }
}

#[test]
fn classification_all_ones_weights_are_bit_identical_to_unweighted() {
    let n = 24;
    let (features, target) = classif_data(n);
    let plain = ClassificationTask::new("p", features.clone(), target.clone()).unwrap();
    let ones = ClassificationTask::new("o", features.clone(), target)
        .unwrap()
        .with_weights(vec![1.0; n]);

    let mk_logreg = || LogisticRegression::new();
    assert_eq!(
        classif_probs(mk_logreg().train_classif(&plain).unwrap().as_ref(), &features),
        classif_probs(mk_logreg().train_classif(&ones).unwrap().as_ref(), &features),
        "logistic_regression: all-ones weights must be bit-identical to unweighted"
    );

    let mk_elm = || ExtremeLearningMachine::new().with_n_hidden(16).with_seed(11);
    assert_eq!(
        classif_probs(mk_elm().train_classif(&plain).unwrap().as_ref(), &features),
        classif_probs(mk_elm().train_classif(&ones).unwrap().as_ref(), &features),
        "elm: all-ones weights must be bit-identical to unweighted"
    );
}

// ── oracle 3: weight 0 ≡ row absent ─────────────────────────────────────

#[test]
fn regression_zero_weight_rows_are_excluded_from_the_fit() {
    let n = 24;
    let (features, target) = regress_data(n);

    // Poison two extra rows with large finite garbage and give them weight 0.
    let mut full = Array2::zeros((n + 2, 3));
    for i in 0..n {
        for j in 0..3 {
            full[[i, j]] = features[[i, j]];
        }
    }
    for j in 0..3 {
        full[[n, j]] = 1.0e6 * (j as f64 + 1.0);
        full[[n + 1, j]] = -7.5e5 * (j as f64 + 2.0);
    }
    let mut full_target = target.clone();
    full_target.push(9.9e5);
    full_target.push(-3.3e5);
    let mut weights = vec![1.0; n];
    weights.extend([0.0, 0.0]);

    let weighted_task = RegressionTask::new("z", full, full_target)
        .unwrap()
        .with_weights(weights);
    let subset_task = RegressionTask::new("s", features.clone(), target).unwrap();

    for (name, factory, tol) in regress_learners() {
        let m_w = factory().train_regress(&weighted_task).unwrap();
        let m_s = factory().train_regress(&subset_task).unwrap();
        assert_close(
            &reg_preds(m_w.as_ref(), &features),
            &reg_preds(m_s.as_ref(), &features),
            tol,
            &format!("{name}: zero-weight rows vs rows removed"),
        );
    }
}

#[test]
fn classification_zero_weight_rows_are_excluded_from_the_fit() {
    let n = 24;
    let (features, target) = classif_data(n);

    let mut full = Array2::zeros((n + 2, 2));
    for i in 0..n {
        for j in 0..2 {
            full[[i, j]] = features[[i, j]];
        }
    }
    // Garbage rows: moderate magnitudes with WRONG labels; weight 0 must
    // erase them from the standardization stats and the gradient alike.
    full[[n, 0]] = 50.0;
    full[[n, 1]] = -40.0;
    full[[n + 1, 0]] = -35.0;
    full[[n + 1, 1]] = 60.0;
    let mut full_target = target.clone();
    full_target.push(0);
    full_target.push(1);
    let mut weights = vec![1.0; n];
    weights.extend([0.0, 0.0]);

    let weighted_task = ClassificationTask::new("z", full, full_target)
        .unwrap()
        .with_weights(weights);
    let subset_task = ClassificationTask::new("s", features.clone(), target).unwrap();

    let mk_logreg = || LogisticRegression::new().with_tol(1e-14).with_max_iter(2000);
    assert_probs_close(
        &classif_probs(mk_logreg().train_classif(&weighted_task).unwrap().as_ref(), &features),
        &classif_probs(mk_logreg().train_classif(&subset_task).unwrap().as_ref(), &features),
        1e-9,
        "logistic_regression: zero-weight rows vs rows removed",
    );

    let mk_elm = || ExtremeLearningMachine::new().with_n_hidden(16).with_seed(11);
    assert_probs_close(
        &classif_probs(mk_elm().train_classif(&weighted_task).unwrap().as_ref(), &features),
        &classif_probs(mk_elm().train_classif(&subset_task).unwrap().as_ref(), &features),
        1e-7,
        "elm: zero-weight rows vs rows removed",
    );
}

// ── oracle 4: golden vs sklearn 1.8.0 ───────────────────────────────────
//
// Generated by scratchpad/golden_weights.py (sklearn 1.8.0, numpy):
//   LinearRegression().fit(X, y, sample_weight=w).predict(Xt)
//   Ridge(alpha=0.7).fit(X, y, sample_weight=w).predict(Xt)
//   Lasso(alpha=0.1, tol=1e-14, max_iter=1_000_000).fit(X, y, sample_weight=w).predict(Xt)
// Smelt's Ridge alpha maps 1:1 to sklearn's (both solve
// min Σ w_i·r_i² + alpha·‖coef‖², intercept unpenalized). Smelt's
// Lasso normalizes the data-fit term by Σw, which equals sklearn's
// documented rescaling of sample_weight to sum to n_samples.

fn golden_data() -> (Array2<f64>, Vec<f64>, Vec<f64>, Array2<f64>) {
    let x = ndarray::array![
        [0.5, 1.2],
        [1.5, 0.3],
        [2.0, 2.5],
        [3.1, 1.1],
        [0.2, 3.3],
        [4.0, 2.2],
        [2.7, 0.8],
        [1.1, 1.9]
    ];
    let y = vec![2.1, 1.3, 5.2, 4.0, 4.9, 7.3, 3.1, 3.9];
    let w = vec![0.5, 2.0, 1.5, 0.7, 3.0, 1.0, 2.5, 0.3];
    let x_test = ndarray::array![[1.0, 1.0], [2.5, 1.5], [0.0, 2.0], [3.5, 0.5]];
    (x, y, w, x_test)
}

#[test]
fn weighted_ols_matches_sklearn_golden() {
    let (x, y, w, x_test) = golden_data();
    let task = RegressionTask::new("g", x, y).unwrap().with_weights(w);
    let model = LinearRegression::new().train_regress(&task).unwrap();
    let expected = [1.88424156, 4.22897327, 2.5475605, 3.56565433];
    assert_close(
        &reg_preds(model.as_ref(), &x_test),
        &expected,
        1e-7,
        "weighted OLS vs sklearn LinearRegression",
    );
}

#[test]
fn weighted_ridge_matches_sklearn_golden() {
    let (x, y, w, x_test) = golden_data();
    let task = RegressionTask::new("g", x, y).unwrap().with_weights(w);
    let model = Ridge::new(0.7).train_regress(&task).unwrap();
    let expected = [2.04482625, 4.1921026, 2.68158694, 3.55534191];
    assert_close(
        &reg_preds(model.as_ref(), &x_test),
        &expected,
        1e-7,
        "weighted Ridge vs sklearn Ridge(alpha=0.7)",
    );
}

#[test]
fn weighted_lasso_matches_sklearn_golden() {
    let (x, y, w, x_test) = golden_data();
    let task = RegressionTask::new("g", x, y).unwrap().with_weights(w);
    let model = Lasso::new(0.1)
        .with_tol(1e-12)
        .with_max_iter(500_000)
        .train_regress(&task)
        .unwrap();
    let expected = [2.09589179, 4.16762905, 2.74888617, 3.51463467];
    assert_close(
        &reg_preds(model.as_ref(), &x_test),
        &expected,
        1e-5,
        "weighted Lasso vs sklearn Lasso(alpha=0.1)",
    );
}

// ── metadata + behavioural sanity ───────────────────────────────────────

#[test]
fn linear_learners_declare_weight_support() {
    assert!(LinearRegression::new().supports_weights());
    assert!(Ridge::new(1.0).supports_weights());
    assert!(Lasso::new(0.1).supports_weights());
    assert!(ElasticNet::new(0.1, 0.5).supports_weights());
    assert!(LogisticRegression::new().supports_weights());
    assert!(ExtremeLearningMachine::new().supports_weights());
    // LinearSVM stays weight-unaware in this phase (shuffled per-sample SGD
    // has no duplication-exact weighted form): guard + flag must agree.
    assert!(!LinearSVM::new().supports_weights());
}

#[test]
fn linear_svm_still_rejects_weighted_tasks() {
    let (features, target) = classif_data(12);
    let task = ClassificationTask::new("svmw", features, target)
        .unwrap()
        .with_weights(vec![1.0; 12]);
    let err = LinearSVM::new().train_classif(&task).map(|_| ()).unwrap_err();
    assert!(
        format!("{err}").contains("LinearSVM") && format!("{err}").contains("does not support sample weights"),
        "guard must name the learner: {err}"
    );
}

/// Skewed weights must actually change the fit — guards against a learner
/// that flips `supports_weights` but silently drops the weights.
#[test]
fn skewed_weights_change_the_fit() {
    let n = 24;
    let (features, target) = regress_data(n);
    let plain = RegressionTask::new("p", features.clone(), target.clone()).unwrap();
    let mut weights = vec![1.0e-3; n];
    for w in weights.iter_mut().take(6) {
        *w = 50.0;
    }
    let skewed = RegressionTask::new("s", features.clone(), target)
        .unwrap()
        .with_weights(weights);

    for (name, factory, _) in regress_learners() {
        let p_plain = reg_preds(factory().train_regress(&plain).unwrap().as_ref(), &features);
        let p_skew = reg_preds(factory().train_regress(&skewed).unwrap().as_ref(), &features);
        let max_diff = p_plain
            .iter()
            .zip(&p_skew)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff > 1e-6,
            "{name}: skewed weights left the fit unchanged (max diff {max_diff}) — weights are being ignored"
        );
    }
}
