//! Fase B1 of per-sample weights: real weight consumption in the tree
//! learners (DecisionTree, RandomForest, ExtraTrees, GradientBoosting).
//!
//! Oracles, in the order of the phase plan:
//! 1. DecisionTree exactness — integer weight k ≡ row duplicated k times
//!    (bit-identical predictions, Gini and MSE), and weight 0 ≡ row removed
//!    (bit-identical).
//! 2. All-ones weights ≡ no weights — bit-identical for DT/RF/ET/GBM,
//!    classification and regression, same seed.
//! 3. RF/ET semantics (statistical, wide margins, fixed seeds) — a
//!    zero-weight outlier has no influence; a 100x-weighted sample
//!    dominates its neighborhood.
//! 4. Numerics — the weighted MSE sweep keeps the HIGH-1 centered
//!    accumulation, so a 1e8 additive target offset does not degrade splits.
//! 5. No-change regression for the unweighted path: the unweighted
//!    `TreeBuilder` code paths were not edited at all (weighted logic lives
//!    in separate `*_weighted` twins dispatched on `task.weights()`), so
//!    the existing suite plus oracle 2 covers it.

use ndarray::Array2;
use smelt_ml::prelude::*;

// ── helpers ──────────────────────────────────────────────────────────────

/// Expand each row `i` of `(features, target)` into `weights[i]` copies
/// (weights must be non-negative integers stored as f64).
fn duplicate_rows<T: Clone>(
    features: &Array2<f64>,
    target: &[T],
    weights: &[f64],
) -> (Array2<f64>, Vec<T>) {
    let mut rows: Vec<f64> = Vec::new();
    let mut tgt = Vec::new();
    for i in 0..features.nrows() {
        let k = weights[i] as usize;
        assert_eq!(k as f64, weights[i], "test weights must be integers");
        for _ in 0..k {
            rows.extend(features.row(i).iter().copied());
            tgt.push(target[i].clone());
        }
    }
    let arr = Array2::from_shape_vec((tgt.len(), features.ncols()), rows).unwrap();
    (arr, tgt)
}

fn assert_classif_bit_identical(a: &Prediction, b: &Prediction, ctx: &str) {
    let (
        Prediction::Classification {
            predicted: pa,
            probabilities: ppa,
            ..
        },
        Prediction::Classification {
            predicted: pb,
            probabilities: ppb,
            ..
        },
    ) = (a, b)
    else {
        panic!("{ctx}: expected classification predictions");
    };
    assert_eq!(pa, pb, "{ctx}: predicted classes differ");
    let (ppa, ppb) = (ppa.as_ref().unwrap(), ppb.as_ref().unwrap());
    for (i, (ra, rb)) in ppa.iter().zip(ppb).enumerate() {
        for (j, (x, y)) in ra.iter().zip(rb).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{ctx}: probability [{i}][{j}] differs: {x} vs {y}"
            );
        }
    }
}

fn assert_regress_bit_identical(a: &Prediction, b: &Prediction, ctx: &str) {
    let (
        Prediction::Regression { predicted: pa, .. },
        Prediction::Regression { predicted: pb, .. },
    ) = (a, b)
    else {
        panic!("{ctx}: expected regression predictions");
    };
    for (i, (x, y)) in pa.iter().zip(pb).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{ctx}: prediction [{i}] differs: {x} vs {y}"
        );
    }
}

fn regress_values(p: &Prediction) -> Vec<f64> {
    let Prediction::Regression { predicted, .. } = p else {
        panic!("expected regression");
    };
    predicted.clone()
}

fn classif_values(p: &Prediction) -> Vec<usize> {
    let Prediction::Classification { predicted, .. } = p else {
        panic!("expected classification");
    };
    predicted.clone()
}

// ── fixtures with all-integer values, so every impurity accumulation is
// exact and the duplication equivalence is exact down to the last bit ──

/// 12 distinct rows, 3 classes as a clean function of the two features.
fn classif_fixture() -> (Array2<f64>, Vec<usize>) {
    let mut rows = Vec::new();
    let mut target = Vec::new();
    for x0 in 0..6i64 {
        for x1 in [0i64, 10] {
            rows.extend([x0 as f64, x1 as f64]);
            target.push(if x1 >= 5 {
                2
            } else if x0 >= 3 {
                1
            } else {
                0
            });
        }
    }
    (Array2::from_shape_vec((12, 2), rows).unwrap(), target)
}

/// 10 rows, integer targets with two clear levels plus within-level steps.
fn regress_fixture() -> (Array2<f64>, Vec<f64>) {
    let features = Array2::from_shape_fn((10, 1), |(i, _)| i as f64);
    let target = vec![0.0, 0.0, 1.0, 1.0, 2.0, 10.0, 10.0, 11.0, 11.0, 12.0];
    (features, target)
}

/// Non-integer, deterministic dataset for the all-ones oracles (exercises
/// full-mantissa arithmetic, not just the exact-integer fast case).
fn noisy_regress_fixture(n: usize) -> (Array2<f64>, Vec<f64>) {
    let features = Array2::from_shape_fn((n, 2), |(i, j)| {
        (i as f64 * 12.9898 + j as f64 * 78.233).sin() * 10.0
    });
    let target: Vec<f64> = (0..n)
        .map(|i| features[[i, 0]] * 2.0 - features[[i, 1]] + (i as f64 * 0.7).cos())
        .collect();
    (features, target)
}

fn noisy_classif_fixture(n: usize) -> (Array2<f64>, Vec<usize>) {
    let (features, _) = noisy_regress_fixture(n);
    let target: Vec<usize> = (0..n)
        .map(|i| {
            let s = features[[i, 0]] + features[[i, 1]];
            if s < -5.0 {
                0
            } else if s < 5.0 {
                1
            } else {
                2
            }
        })
        .collect();
    (features, target)
}

// ── oracle 1: DecisionTree exactness ─────────────────────────────────────

#[test]
fn dt_classif_integer_weights_equal_row_duplication_bit_identical() {
    let (features, target) = classif_fixture();
    let weights: Vec<f64> = (0..target.len()).map(|i| (i % 3 + 1) as f64).collect();

    let weighted_task = ClassificationTask::new("w", features.clone(), target.clone())
        .unwrap()
        .with_weights(weights.clone());
    let (dup_features, dup_target) = duplicate_rows(&features, &target, &weights);
    let dup_task = ClassificationTask::new("dup", dup_features, dup_target).unwrap();

    let ma = DecisionTree::default().train_classif(&weighted_task).unwrap();
    let mb = DecisionTree::default().train_classif(&dup_task).unwrap();

    let pa = ma.predict(&features).unwrap();
    let pb = mb.predict(&features).unwrap();
    assert_classif_bit_identical(&pa, &pb, "weight k vs k duplicates (gini)");
}

#[test]
fn dt_regress_integer_weights_equal_row_duplication_bit_identical() {
    let (features, target) = regress_fixture();
    let weights: Vec<f64> = (0..target.len()).map(|i| (i % 3 + 1) as f64).collect();

    let weighted_task = RegressionTask::new("w", features.clone(), target.clone())
        .unwrap()
        .with_weights(weights.clone());
    let (dup_features, dup_target) = duplicate_rows(&features, &target, &weights);
    let dup_task = RegressionTask::new("dup", dup_features, dup_target).unwrap();

    let ma = DecisionTree::default().train_regress(&weighted_task).unwrap();
    let mb = DecisionTree::default().train_regress(&dup_task).unwrap();

    let pa = ma.predict(&features).unwrap();
    let pb = mb.predict(&features).unwrap();
    assert_regress_bit_identical(&pa, &pb, "weight k vs k duplicates (mse)");
}

#[test]
fn dt_classif_zero_weight_equals_row_removal_bit_identical() {
    let (features, target) = classif_fixture();
    // Zero out rows 1, 6, 10; keep mixed weights on the rest so the removed
    // variant also runs through the weighted path.
    let mut weights: Vec<f64> = (0..target.len()).map(|i| (i % 3 + 1) as f64).collect();
    for &z in &[1usize, 6, 10] {
        weights[z] = 0.0;
    }

    let weighted_task = ClassificationTask::new("w", features.clone(), target.clone())
        .unwrap()
        .with_weights(weights.clone());

    let keep: Vec<usize> = (0..target.len()).filter(|i| weights[*i] > 0.0).collect();
    let kept_features = Array2::from_shape_fn((keep.len(), features.ncols()), |(r, c)| {
        features[[keep[r], c]]
    });
    let kept_target: Vec<usize> = keep.iter().map(|&i| target[i]).collect();
    let kept_weights: Vec<f64> = keep.iter().map(|&i| weights[i]).collect();
    let removed_task = ClassificationTask::new("rm", kept_features, kept_target)
        .unwrap()
        .with_weights(kept_weights);

    let ma = DecisionTree::default().train_classif(&weighted_task).unwrap();
    let mb = DecisionTree::default().train_classif(&removed_task).unwrap();

    let pa = ma.predict(&features).unwrap();
    let pb = mb.predict(&features).unwrap();
    assert_classif_bit_identical(&pa, &pb, "weight 0 vs row removed (gini)");
}

#[test]
fn dt_regress_zero_weight_equals_row_removal_bit_identical() {
    let (features, target) = regress_fixture();
    let mut weights: Vec<f64> = (0..target.len()).map(|i| (i % 3 + 1) as f64).collect();
    for &z in &[0usize, 4, 7] {
        weights[z] = 0.0;
    }

    let weighted_task = RegressionTask::new("w", features.clone(), target.clone())
        .unwrap()
        .with_weights(weights.clone());

    let keep: Vec<usize> = (0..target.len()).filter(|i| weights[*i] > 0.0).collect();
    let kept_features = Array2::from_shape_fn((keep.len(), features.ncols()), |(r, c)| {
        features[[keep[r], c]]
    });
    let kept_target: Vec<f64> = keep.iter().map(|&i| target[i]).collect();
    let kept_weights: Vec<f64> = keep.iter().map(|&i| weights[i]).collect();
    let removed_task = RegressionTask::new("rm", kept_features, kept_target)
        .unwrap()
        .with_weights(kept_weights);

    let ma = DecisionTree::default().train_regress(&weighted_task).unwrap();
    let mb = DecisionTree::default().train_regress(&removed_task).unwrap();

    let pa = ma.predict(&features).unwrap();
    let pb = mb.predict(&features).unwrap();
    assert_regress_bit_identical(&pa, &pb, "weight 0 vs row removed (mse)");
}

// ── oracle 2: all-ones weights ≡ unweighted, bit-identical, all 4 learners ──

#[test]
fn dt_all_ones_weights_equal_unweighted_bit_identical() {
    let (cf, ct) = noisy_classif_fixture(60);
    let plain = ClassificationTask::new("p", cf.clone(), ct.clone()).unwrap();
    let ones = ClassificationTask::new("o", cf.clone(), ct)
        .unwrap()
        .with_weights(vec![1.0; 60]);
    let ma = DecisionTree::default().train_classif(&plain).unwrap();
    let mb = DecisionTree::default().train_classif(&ones).unwrap();
    assert_classif_bit_identical(
        &ma.predict(&cf).unwrap(),
        &mb.predict(&cf).unwrap(),
        "DT classif all-ones",
    );

    let (rf, rt) = noisy_regress_fixture(60);
    let plain = RegressionTask::new("p", rf.clone(), rt.clone()).unwrap();
    let ones = RegressionTask::new("o", rf.clone(), rt)
        .unwrap()
        .with_weights(vec![1.0; 60]);
    let ma = DecisionTree::default().train_regress(&plain).unwrap();
    let mb = DecisionTree::default().train_regress(&ones).unwrap();
    assert_regress_bit_identical(
        &ma.predict(&rf).unwrap(),
        &mb.predict(&rf).unwrap(),
        "DT regress all-ones",
    );
}

#[test]
fn rf_all_ones_weights_equal_unweighted_bit_identical() {
    let (cf, ct) = noisy_classif_fixture(60);
    let plain = ClassificationTask::new("p", cf.clone(), ct.clone()).unwrap();
    let ones = ClassificationTask::new("o", cf.clone(), ct)
        .unwrap()
        .with_weights(vec![1.0; 60]);
    let ma = RandomForest::new()
        .with_n_estimators(15)
        .with_seed(7)
        .train_classif(&plain)
        .unwrap();
    let mb = RandomForest::new()
        .with_n_estimators(15)
        .with_seed(7)
        .train_classif(&ones)
        .unwrap();
    assert_classif_bit_identical(
        &ma.predict(&cf).unwrap(),
        &mb.predict(&cf).unwrap(),
        "RF classif all-ones",
    );

    let (rf_, rt) = noisy_regress_fixture(60);
    let plain = RegressionTask::new("p", rf_.clone(), rt.clone()).unwrap();
    let ones = RegressionTask::new("o", rf_.clone(), rt)
        .unwrap()
        .with_weights(vec![1.0; 60]);
    let ma = RandomForest::new()
        .with_n_estimators(15)
        .with_seed(7)
        .train_regress(&plain)
        .unwrap();
    let mb = RandomForest::new()
        .with_n_estimators(15)
        .with_seed(7)
        .train_regress(&ones)
        .unwrap();
    assert_regress_bit_identical(
        &ma.predict(&rf_).unwrap(),
        &mb.predict(&rf_).unwrap(),
        "RF regress all-ones",
    );
}

#[test]
fn et_all_ones_weights_equal_unweighted_bit_identical() {
    let (cf, ct) = noisy_classif_fixture(60);
    let plain = ClassificationTask::new("p", cf.clone(), ct.clone()).unwrap();
    let ones = ClassificationTask::new("o", cf.clone(), ct)
        .unwrap()
        .with_weights(vec![1.0; 60]);
    let ma = ExtraTrees::new()
        .with_n_estimators(15)
        .with_seed(7)
        .train_classif(&plain)
        .unwrap();
    let mb = ExtraTrees::new()
        .with_n_estimators(15)
        .with_seed(7)
        .train_classif(&ones)
        .unwrap();
    assert_classif_bit_identical(
        &ma.predict(&cf).unwrap(),
        &mb.predict(&cf).unwrap(),
        "ET classif all-ones",
    );

    let (rf_, rt) = noisy_regress_fixture(60);
    let plain = RegressionTask::new("p", rf_.clone(), rt.clone()).unwrap();
    let ones = RegressionTask::new("o", rf_.clone(), rt)
        .unwrap()
        .with_weights(vec![1.0; 60]);
    let ma = ExtraTrees::new()
        .with_n_estimators(15)
        .with_seed(7)
        .train_regress(&plain)
        .unwrap();
    let mb = ExtraTrees::new()
        .with_n_estimators(15)
        .with_seed(7)
        .train_regress(&ones)
        .unwrap();
    assert_regress_bit_identical(
        &ma.predict(&rf_).unwrap(),
        &mb.predict(&rf_).unwrap(),
        "ET regress all-ones",
    );
}

#[test]
fn gbm_all_ones_weights_equal_unweighted_bit_identical() {
    // Binary classification (covers train_binary + weighted Newton refit
    // scaling, which must be a bit-exact no-op at w=1).
    let (cf, ct3) = noisy_classif_fixture(60);
    let ct: Vec<usize> = ct3.iter().map(|&c| usize::from(c >= 1)).collect();
    let plain = ClassificationTask::new("p", cf.clone(), ct.clone()).unwrap();
    let ones = ClassificationTask::new("o", cf.clone(), ct)
        .unwrap()
        .with_weights(vec![1.0; 60]);
    let ma = GradientBoosting::new()
        .with_n_estimators(20)
        .with_seed(7)
        .train_classif(&plain)
        .unwrap();
    let mb = GradientBoosting::new()
        .with_n_estimators(20)
        .with_seed(7)
        .train_classif(&ones)
        .unwrap();
    assert_classif_bit_identical(
        &ma.predict(&cf).unwrap(),
        &mb.predict(&cf).unwrap(),
        "GBM binary all-ones",
    );

    // Multiclass (train_multiclass path)
    let plain3 = ClassificationTask::new("p3", cf.clone(), ct3.clone()).unwrap();
    let ones3 = ClassificationTask::new("o3", cf.clone(), ct3)
        .unwrap()
        .with_weights(vec![1.0; 60]);
    let ma = GradientBoosting::new()
        .with_n_estimators(10)
        .with_seed(7)
        .train_classif(&plain3)
        .unwrap();
    let mb = GradientBoosting::new()
        .with_n_estimators(10)
        .with_seed(7)
        .train_classif(&ones3)
        .unwrap();
    assert_classif_bit_identical(
        &ma.predict(&cf).unwrap(),
        &mb.predict(&cf).unwrap(),
        "GBM multiclass all-ones",
    );

    // Regression (weighted initial mean must equal the plain mean at w=1)
    let (rf_, rt) = noisy_regress_fixture(60);
    let plain = RegressionTask::new("p", rf_.clone(), rt.clone()).unwrap();
    let ones = RegressionTask::new("o", rf_.clone(), rt)
        .unwrap()
        .with_weights(vec![1.0; 60]);
    let ma = GradientBoosting::new()
        .with_n_estimators(20)
        .with_seed(7)
        .train_regress(&plain)
        .unwrap();
    let mb = GradientBoosting::new()
        .with_n_estimators(20)
        .with_seed(7)
        .train_regress(&ones)
        .unwrap();
    assert_regress_bit_identical(
        &ma.predict(&rf_).unwrap(),
        &mb.predict(&rf_).unwrap(),
        "GBM regress all-ones",
    );
}

// ── GBM subsample keeps drawing uniformly, then excludes weight-0 rows:
// with subsample=1.0 (deterministic) weight 0 ≡ row removed, bit-exact ──

#[test]
fn gbm_regress_zero_weight_equals_row_removal_bit_identical() {
    let (features, target) = regress_fixture();
    let mut weights = vec![1.0; target.len()];
    weights[3] = 0.0;
    weights[8] = 0.0;

    let weighted_task = RegressionTask::new("w", features.clone(), target.clone())
        .unwrap()
        .with_weights(weights.clone());

    let keep: Vec<usize> = (0..target.len()).filter(|i| weights[*i] > 0.0).collect();
    let kept_features = Array2::from_shape_fn((keep.len(), features.ncols()), |(r, c)| {
        features[[keep[r], c]]
    });
    let kept_target: Vec<f64> = keep.iter().map(|&i| target[i]).collect();
    let removed_task = RegressionTask::new("rm", kept_features, kept_target)
        .unwrap()
        .with_weights(vec![1.0; keep.len()]);

    let ma = GradientBoosting::new()
        .with_n_estimators(30)
        .train_regress(&weighted_task)
        .unwrap();
    let mb = GradientBoosting::new()
        .with_n_estimators(30)
        .train_regress(&removed_task)
        .unwrap();

    let pa = ma.predict(&features).unwrap();
    let pb = mb.predict(&features).unwrap();
    assert_regress_bit_identical(&pa, &pb, "GBM weight 0 vs row removed");
}

#[test]
fn gbm_binary_zero_weight_equals_row_removal_bit_identical() {
    let (features, target3) = classif_fixture();
    let target: Vec<usize> = target3.iter().map(|&c| usize::from(c >= 1)).collect();
    let mut weights = vec![1.0; target.len()];
    weights[2] = 0.0;
    weights[9] = 0.0;

    let weighted_task = ClassificationTask::new("w", features.clone(), target.clone())
        .unwrap()
        .with_weights(weights.clone());

    let keep: Vec<usize> = (0..target.len()).filter(|i| weights[*i] > 0.0).collect();
    let kept_features = Array2::from_shape_fn((keep.len(), features.ncols()), |(r, c)| {
        features[[keep[r], c]]
    });
    let kept_target: Vec<usize> = keep.iter().map(|&i| target[i]).collect();
    let removed_task = ClassificationTask::new("rm", kept_features, kept_target)
        .unwrap()
        .with_weights(vec![1.0; keep.len()]);

    let ma = GradientBoosting::new()
        .with_n_estimators(20)
        .train_classif(&weighted_task)
        .unwrap();
    let mb = GradientBoosting::new()
        .with_n_estimators(20)
        .train_classif(&removed_task)
        .unwrap();

    let pa = ma.predict(&features).unwrap();
    let pb = mb.predict(&features).unwrap();
    assert_classif_bit_identical(&pa, &pb, "GBM binary weight 0 vs row removed");
}

// ── oracle 3: RF/ET weight semantics (statistical, wide margins) ─────────

#[test]
fn rf_et_zero_weight_outlier_has_no_influence() {
    // 40 clean points y≈1.0 plus one wild outlier (y=1000) at x=0.51.
    let n = 41;
    let mut xs: Vec<f64> = (0..40).map(|i| i as f64 / 39.0).collect();
    xs.push(0.51);
    let features = Array2::from_shape_fn((n, 1), |(i, _)| xs[i]);
    let mut target: Vec<f64> = vec![1.0; 40];
    target.push(1000.0);

    let mut weights = vec![1.0; n];
    weights[n - 1] = 0.0;

    let probe = Array2::from_shape_vec((1, 1), vec![0.51]).unwrap();

    for (name, learner) in [
        (
            "RandomForest",
            Box::new(RandomForest::new().with_n_estimators(50).with_seed(3)) as Box<dyn Learner>,
        ),
        (
            "ExtraTrees",
            Box::new(ExtraTrees::new().with_n_estimators(50).with_seed(3)),
        ),
    ] {
        let mut learner = learner;

        // Outlier active (weight 1): prediction at its location is pulled
        // far up — this is the contrast that proves the margin is real.
        let active = RegressionTask::new("a", features.clone(), target.clone())
            .unwrap()
            .with_weights(vec![1.0; n]);
        let pred_active = regress_values(
            &learner
                .train_regress(&active)
                .unwrap()
                .predict(&probe)
                .unwrap(),
        )[0];
        assert!(
            pred_active > 100.0,
            "{name}: with weight 1 the outlier should pull the prediction at its \
             location way up, got {pred_active}"
        );

        // Outlier excluded (weight 0): prediction stays at the clean level.
        let excluded = RegressionTask::new("e", features.clone(), target.clone())
            .unwrap()
            .with_weights(weights.clone());
        let pred_excluded = regress_values(
            &learner
                .train_regress(&excluded)
                .unwrap()
                .predict(&probe)
                .unwrap(),
        )[0];
        assert!(
            (pred_excluded - 1.0).abs() < 0.5,
            "{name}: a zero-weight outlier must not influence the fit, got {pred_excluded}"
        );
    }
}

#[test]
fn rf_et_heavily_weighted_sample_dominates_its_neighborhood() {
    // One class-1 point (x=0.2) surrounded by class-0 points; shallow trees
    // so single points cannot be isolated — only the weights can flip the
    // leaf's majority. Far away, a clean class-1 cluster at x≈1.0 keeps two
    // classes in the task.
    let xs = [0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30, 0.95, 1.0, 1.05];
    let target = vec![0usize, 0, 0, 1, 0, 0, 0, 1, 1, 1];
    let n = xs.len();
    let features = Array2::from_shape_fn((n, 1), |(i, _)| xs[i]);
    let probe = Array2::from_shape_vec((1, 1), vec![0.2]).unwrap();

    for (name, mk) in [
        (
            "RandomForest",
            (|| {
                Box::new(
                    RandomForest::new()
                        .with_n_estimators(50)
                        .with_max_depth(1)
                        .with_seed(3),
                ) as Box<dyn Learner>
            }) as fn() -> Box<dyn Learner>,
        ),
        ("ExtraTrees", || {
            Box::new(
                ExtraTrees::new()
                    .with_n_estimators(50)
                    .with_max_depth(1)
                    .with_seed(3),
            )
        }),
    ] {
        // Unweighted: the lone class-1 point is outvoted in its leaf.
        let plain = ClassificationTask::new("p", features.clone(), target.clone()).unwrap();
        let pred_plain =
            classif_values(&mk().train_classif(&plain).unwrap().predict(&probe).unwrap())[0];
        assert_eq!(
            pred_plain, 0,
            "{name}: unweighted, the lone minority point must be outvoted"
        );

        // Weight 100: it dominates every leaf that contains it.
        let mut weights = vec![1.0; n];
        weights[3] = 100.0;
        let heavy = ClassificationTask::new("h", features.clone(), target.clone())
            .unwrap()
            .with_weights(weights);
        let pred_heavy =
            classif_values(&mk().train_classif(&heavy).unwrap().predict(&probe).unwrap())[0];
        assert_eq!(
            pred_heavy, 1,
            "{name}: a 100x-weighted sample must dominate its neighborhood"
        );
    }
}

// ── oracle 4: weighted MSE keeps the HIGH-1 offset invariance ────────────

#[test]
fn weighted_mse_split_is_invariant_to_large_target_offsets() {
    // Step signal at x=5 with unit-scale pseudo-noise and non-trivial
    // weights. If the weighted sweep accumulated E[y²]−E[y]² on raw
    // targets, eps·offset² would swamp the true variance at offset 1e8 and
    // the depth-1 split would land anywhere; centered on the weighted node
    // mean, the split must stay at the step for every offset.
    let n = 400;
    let features = Array2::from_shape_fn((n, 1), |(i, _)| i as f64 / n as f64 * 10.0);
    let base: Vec<f64> = (0..n)
        .map(|i| {
            let x = features[[i, 0]];
            let step = if x < 5.0 { 0.0 } else { 4.0 };
            step + 0.3 * ((i as f64 * 12.9898).sin())
        })
        .collect();
    let weights: Vec<f64> = (0..n).map(|i| [0.5, 1.0, 2.0, 3.0][i % 4]).collect();

    let probe = Array2::from_shape_vec((2, 1), vec![2.5, 7.5]).unwrap();

    for offset in [0.0, 1e6, 1e8] {
        let target: Vec<f64> = base.iter().map(|y| y + offset).collect();
        let task = RegressionTask::new("off", features.clone(), target)
            .unwrap()
            .with_weights(weights.clone());
        let model = DecisionTree::new()
            .with_max_depth(1)
            .train_regress(&task)
            .unwrap();
        let preds = regress_values(&model.predict(&probe).unwrap());
        let low = preds[0] - offset;
        let high = preds[1] - offset;
        assert!(
            low.abs() < 0.5,
            "offset {offset:e}: left-of-step prediction should stay ≈0, got {low}"
        );
        assert!(
            (high - 4.0).abs() < 0.5,
            "offset {offset:e}: right-of-step prediction should stay ≈4, got {high}"
        );
    }
}

// ── weighted GBM learns the weighted target, not just the unweighted one ──

#[test]
fn gbm_weighted_initial_and_trees_follow_the_heavy_samples() {
    // Two flat regimes; the right regime's rows get weight 9 vs 1. The
    // weighted initial prediction (weighted mean) must sit far closer to
    // the heavy regime than the plain mean would.
    let n = 20;
    let features = Array2::from_shape_fn((n, 1), |(i, _)| i as f64);
    let target: Vec<f64> = (0..n).map(|i| if i < 10 { 0.0 } else { 10.0 }).collect();
    let weights: Vec<f64> = (0..n).map(|i| if i < 10 { 1.0 } else { 9.0 }).collect();

    let task = RegressionTask::new("wgbm", features.clone(), target)
        .unwrap()
        .with_weights(weights);
    // n_estimators=0 is not allowed to converge the trees away — use 1 tree
    // with depth 1 so the initial value still shows through the shrinkage.
    let model = GradientBoosting::new()
        .with_n_estimators(1)
        .with_max_depth(1)
        .with_learning_rate(0.0)
        .train_regress(&task)
        .unwrap();
    let probe = Array2::from_shape_vec((1, 1), vec![5.0]).unwrap();
    let pred = regress_values(&model.predict(&probe).unwrap())[0];
    // Weighted mean = (10*0*1 + 10*10*9)/(10*1+10*9) = 9.0; plain mean = 5.0.
    assert!(
        (pred - 9.0).abs() < 1e-9,
        "with learning_rate 0 the prediction is exactly the weighted-mean initial, got {pred}"
    );
}

// ── trait metadata ───────────────────────────────────────────────────────

#[test]
fn tree_learners_report_supports_weights() {
    assert!(DecisionTree::default().supports_weights());
    assert!(RandomForest::new().supports_weights());
    assert!(ExtraTrees::new().supports_weights());
    assert!(GradientBoosting::new().supports_weights());
}
