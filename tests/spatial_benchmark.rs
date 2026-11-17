//! Spatial learners and measures composed with the ordinary benchmark /
//! tuning loops through task-carried coordinates
//! (`RegressionTask::with_coords`), `TrainedModel::predict_spatial` and
//! `Measure::score_spatial`.
//!
//! The key property under test is *equivalence*: going through
//! `benchmark::resample_regress` with a `coords_from_task()` learner must
//! give exactly what the hand-written per-fold loop (slice task, build the
//! learner with the fold's coordinates, `train_geo`, `predict_spatial`)
//! gave before this plumbing existed. Nothing about the learners' algorithms
//! changes — only where the coordinates come from.

use ndarray::{Array2, Axis};
use smelt_ml::learner::TrainedModel;
use smelt_ml::prelude::*;

/// 9×10 grid of sites; two aspatial covariates plus a smooth spatial field
/// in the target (so a spatial model has something to pick up). Target is
/// strictly positive so target transforms with a positivity domain apply.
fn spatial_fixture() -> (Array2<f64>, Vec<f64>, Vec<(f64, f64)>) {
    let mut rows = Vec::new();
    let mut target = Vec::new();
    let mut coords = Vec::new();
    let mut state: u64 = 7;
    let mut lcg = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    for r in 0..9 {
        for c in 0..10 {
            let (x, y) = (c as f64 * 1.3 + 0.1 * lcg(), r as f64 * 1.1 + 0.1 * lcg());
            let f1 = lcg() * 4.0 - 2.0;
            let f2 = lcg() * 4.0 - 2.0;
            let field = 2.0 * (x / 4.0).sin() + 1.5 * (y / 3.0).cos();
            rows.push([f1, f2]);
            target.push(10.0 + 1.5 * f1 - 0.8 * f2 + field + 0.2 * (lcg() - 0.5));
            coords.push((x, y));
        }
    }
    let n = rows.len();
    let features = Array2::from_shape_vec((n, 2), rows.concat()).unwrap();
    (features, target, coords)
}

fn geo(bandwidth: usize) -> GeoXGBoost {
    GeoXGBoost::coords_from_task()
        .with_bandwidth(bandwidth)
        .with_n_estimators(20)
        .with_max_depth(2)
        .with_seed(3)
}

fn geo_with(coords: Vec<(f64, f64)>, bandwidth: usize) -> GeoXGBoost {
    GeoXGBoost::new(coords)
        .with_bandwidth(bandwidth)
        .with_n_estimators(20)
        .with_max_depth(2)
        .with_seed(3)
}

/// A fold's task (without coordinates), plus its raw features, target and
/// coordinates for the hand-written spatial loop.
type FoldSlice = (RegressionTask, Array2<f64>, Vec<f64>, Vec<(f64, f64)>);

fn slice_task(task: &RegressionTask, idx: &[usize], coords: &[(f64, f64)]) -> FoldSlice {
    let feats = task.features().select(Axis(0), idx);
    let tgt: Vec<f64> = idx.iter().map(|&i| task.target()[i]).collect();
    let c: Vec<(f64, f64)> = idx.iter().map(|&i| coords[i]).collect();
    (
        RegressionTask::new(task.id(), feats.clone(), tgt.clone()).unwrap(),
        feats,
        tgt,
        c,
    )
}

#[test]
fn geo_xgboost_coords_from_task_through_spatial_block_cv_matches_manual_loop() {
    let (features, target, coords) = spatial_fixture();
    let task = RegressionTask::new("geo", features, target)
        .unwrap()
        .with_coords(coords.clone());
    let cv = SpatialBlockCV::new(3, coords.clone());

    let mut learner = geo(8);
    let bench = benchmark::resample_regress(&mut learner, &task, &cv, &[&Rmse]).unwrap();

    // Hand-written loop, the only way to run GeoXGBoost under spatial CV
    // before task-carried coordinates existed.
    let splits = cv.splits(task.n_samples()).unwrap();
    assert_eq!(bench.scores.len(), splits.len());
    let mut any_differs_from_global = false;
    for (fold, (train_idx, test_idx)) in splits.iter().enumerate() {
        let (train_task, _, _, train_coords) = slice_task(&task, train_idx, &coords);
        let (_, test_feats, test_tgt, test_coords) = slice_task(&task, test_idx, &coords);
        let model = geo_with(train_coords, 8).train_geo(&train_task).unwrap();
        let manual = model
            .predict_spatial(&test_feats, &test_coords)
            .unwrap()
            .with_truth_regress(test_tgt.clone());
        let manual_rmse = Rmse.score(&manual).unwrap();
        assert_eq!(
            bench.scores[fold][0], manual_rmse,
            "fold {fold}: benchmark RMSE must equal the manual predict_spatial loop exactly"
        );
        // And it is the *spatial* prediction, not the global-only `predict`.
        let global = model.predict(&test_feats).unwrap().with_truth_regress(test_tgt);
        if (Rmse.score(&global).unwrap() - manual_rmse).abs() > 1e-9 {
            any_differs_from_global = true;
        }
    }
    assert!(
        any_differs_from_global,
        "spatial and global-only predictions coincided in every fold, so the test \
         could not tell whether predict_spatial was used"
    );
}

#[test]
fn kriging_hybrid_coords_from_task_matches_manual_loop() {
    let (features, target, coords) = spatial_fixture();
    let task = RegressionTask::new("rk", features, target)
        .unwrap()
        .with_coords(coords.clone());
    let cv = SpatialBlockCV::new(3, coords.clone());

    let mut learner =
        KrigingHybrid::coords_from_task(|| Box::new(LinearRegression::new())).with_n_neighbors(8);
    let bench = benchmark::resample_regress(&mut learner, &task, &cv, &[&Rmse]).unwrap();

    let splits = cv.splits(task.n_samples()).unwrap();
    for (fold, (train_idx, test_idx)) in splits.iter().enumerate() {
        let (train_task, _, _, train_coords) = slice_task(&task, train_idx, &coords);
        let (_, test_feats, test_tgt, test_coords) = slice_task(&task, test_idx, &coords);
        let model = KrigingHybrid::new(|| Box::new(LinearRegression::new()), train_coords)
            .with_n_neighbors(8)
            .train_regress_geo(&train_task)
            .unwrap();
        let manual = model
            .predict_spatial(&test_feats, &test_coords)
            .unwrap()
            .with_truth_regress(test_tgt);
        assert_eq!(bench.scores[fold][0], Rmse.score(&manual).unwrap(), "fold {fold}");
    }
}

#[test]
fn task_coords_take_precedence_over_learner_coords_and_legacy_path_is_unchanged() {
    let (features, target, coords) = spatial_fixture();
    let plain = RegressionTask::new("geo", features.clone(), target.clone()).unwrap();
    let with_coords = RegressionTask::new("geo", features, target)
        .unwrap()
        .with_coords(coords.clone());
    let cv = SpatialBlockCV::new(3, coords.clone());

    // Legacy form on a task without coordinates: the learner's full-length
    // coordinates don't match the fold -- same error as before.
    let err = benchmark::resample_regress(&mut geo_with(coords.clone(), 8), &plain, &cv, &[&Rmse])
        .unwrap_err();
    assert!(matches!(err, SmeltError::DimensionMismatch { .. }), "{err}");

    // Legacy form on a task WITH coordinates: the task's (fold-aligned)
    // coordinates win, so it now works and equals coords_from_task().
    let legacy = benchmark::resample_regress(&mut geo_with(coords, 8), &with_coords, &cv, &[&Rmse])
        .unwrap();
    let modern = benchmark::resample_regress(&mut geo(8), &with_coords, &cv, &[&Rmse]).unwrap();
    assert_eq!(legacy.scores, modern.scores);

    // coords_from_task() with no coordinates anywhere: a clear error.
    let err = geo(8).train_regress(&plain).err().expect("must fail");
    assert!(
        matches!(&err, SmeltError::InvalidParameter(m) if m.contains("no coordinates")),
        "{err}"
    );
    let err = KrigingHybrid::coords_from_task(|| Box::new(LinearRegression::new()))
        .train_regress(&plain)
        .err()
        .expect("must fail");
    assert!(
        matches!(&err, SmeltError::InvalidParameter(m) if m.contains("no coordinates")),
        "{err}"
    );
}

#[test]
fn aspatial_learners_and_measures_ignore_task_coords() {
    let (features, target, coords) = spatial_fixture();
    let plain = RegressionTask::new("lin", features.clone(), target.clone()).unwrap();
    let with_coords = RegressionTask::new("lin", features, target)
        .unwrap()
        .with_coords(coords.clone());
    let cv = SpatialBlockCV::new(3, coords);
    let a = benchmark::resample_regress(&mut LinearRegression::new(), &plain, &cv, &[&Rmse, &Mae])
        .unwrap();
    let b = benchmark::resample_regress(
        &mut LinearRegression::new(),
        &with_coords,
        &cv,
        &[&Rmse, &Mae],
    )
    .unwrap();
    assert_eq!(a.scores, b.scores);
}

#[test]
fn moran_residuals_coords_from_task_scores_each_fold_against_its_own_coords() {
    let (features, target, coords) = spatial_fixture();
    let task = RegressionTask::new("moran", features, target)
        .unwrap()
        .with_coords(coords.clone());
    let cv = SpatialBlockCV::new(3, coords.clone());
    let moran = MoranResiduals::coords_from_task().with_weights(SpatialWeights::KNearest { k: 4 });

    // An aspatial model leaves the spatial field in the residuals: Moran's I
    // of the residuals is what tells you a spatial model is warranted.
    let bench =
        benchmark::resample_regress(&mut LinearRegression::new(), &task, &cv, &[&Rmse, &moran])
            .unwrap();
    assert_eq!(bench.measure_ids[1], "regr.moran_residuals");

    let splits = cv.splits(task.n_samples()).unwrap();
    for (fold, (train_idx, test_idx)) in splits.iter().enumerate() {
        let (train_task, _, _, _) = slice_task(&task, train_idx, &coords);
        let (_, test_feats, test_tgt, test_coords) = slice_task(&task, test_idx, &coords);
        let model = LinearRegression::new().train_regress(&train_task).unwrap();
        let pred = model.predict(&test_feats).unwrap();
        let residuals: Vec<f64> = match &pred {
            Prediction::Regression { predicted, .. } => {
                test_tgt.iter().zip(predicted).map(|(t, p)| t - p).collect()
            }
            _ => unreachable!(),
        };
        let manual = morans_i(&residuals, &test_coords, &SpatialWeights::KNearest { k: 4 }).unwrap();
        assert_eq!(bench.scores[fold][1], manual.i, "fold {fold}");
    }
    // ...and the spatial model removes (most of) it.
    let geo_bench = benchmark::resample_regress(&mut geo(8), &task, &cv, &[&Rmse, &moran]).unwrap();
    assert!(
        geo_bench.mean_scores()[1] < bench.mean_scores()[1],
        "GeoXGBoost residual Moran's I {} should be below LinearRegression's {}",
        geo_bench.mean_scores()[1],
        bench.mean_scores()[1]
    );

    // Without task coordinates the coords_from_task() measure cannot score.
    let plain = RegressionTask::new("moran", task.features().clone(), task.target().to_vec()).unwrap();
    let err = benchmark::resample_regress(&mut LinearRegression::new(), &plain, &cv, &[&moran])
        .unwrap_err();
    assert!(matches!(err, SmeltError::InvalidParameter(_)), "{err}");
}

#[test]
fn pipeline_propagates_coords_and_forwards_predict_spatial() {
    let (features, target, coords) = spatial_fixture();
    let task = RegressionTask::new("pipe", features.clone(), target.clone())
        .unwrap()
        .with_coords(coords.clone());

    let mut pipeline = Pipeline::new(vec![Box::new(StandardScaler::new())], Box::new(geo(8)));
    let trained = pipeline.train_regress(&task).unwrap();
    let spatial = trained.predict_spatial(&features, &coords).unwrap();
    let global = trained.predict(&features).unwrap();

    // Manual equivalent: scale, then GeoXGBoost with explicit coordinates.
    let mut scaler = StandardScaler::new();
    let scaled = scaler.fit_transform(&features).unwrap();
    let scaled_task = RegressionTask::new("pipe", scaled.clone(), target).unwrap();
    let manual = geo_with(coords.clone(), 8)
        .train_geo(&scaled_task)
        .unwrap()
        .predict_spatial(&scaled, &coords)
        .unwrap();
    match (&spatial, &manual, &global) {
        (
            Prediction::Regression { predicted: s, .. },
            Prediction::Regression { predicted: m, .. },
            Prediction::Regression { predicted: g, .. },
        ) => {
            assert_eq!(s, m, "pipeline predict_spatial must equal scale-then-GeoXGBoost");
            assert!(
                s.iter().zip(g).any(|(a, b)| (a - b).abs() > 1e-9),
                "predict_spatial fell back to the global-only prediction"
            );
        }
        _ => panic!("expected regression predictions"),
    }

    // And the whole pipeline runs under spatial CV in the benchmark.
    let cv = SpatialBlockCV::new(3, coords);
    let bench = benchmark::resample_regress(&mut pipeline, &task, &cv, &[&Rmse]).unwrap();
    assert!(bench.mean_scores()[0].is_finite());
}

#[test]
fn target_transform_forwards_predict_spatial() {
    let (features, target, coords) = spatial_fixture();
    let task = RegressionTask::new("tt", features.clone(), target.clone())
        .unwrap()
        .with_coords(coords.clone());
    let mut tt = TargetTransformRegressor::new(|| Box::new(geo(8)), TargetTransform::Sqrt);
    let trained = tt.train_regress(&task).unwrap();
    let spatial = trained.predict_spatial(&features, &coords).unwrap();

    let sqrt_task =
        RegressionTask::new("tt", features.clone(), target.iter().map(|y| y.sqrt()).collect())
            .unwrap();
    let inner = geo_with(coords.clone(), 8)
        .train_geo(&sqrt_task)
        .unwrap()
        .predict_spatial(&features, &coords)
        .unwrap();
    match (&spatial, &inner) {
        (Prediction::Regression { predicted: s, .. }, Prediction::Regression { predicted: i, .. }) => {
            for (a, b) in s.iter().zip(i) {
                assert!((a - b * b).abs() < 1e-9);
            }
        }
        _ => panic!("expected regression predictions"),
    }
}

#[test]
fn grid_search_tunes_geo_xgboost_bandwidth_under_spatial_block_cv() {
    let (features, target, coords) = spatial_fixture();
    let task = RegressionTask::new("tune", features, target)
        .unwrap()
        .with_coords(coords.clone());
    let cv = SpatialBlockCV::new(3, coords);

    let mut grid = ParamGrid::new();
    grid.insert("bandwidth".into(), vec![6.0.into(), 12.0.into()]);
    let gs = GridSearch::new(
        |params| Box::new(geo(params["bandwidth"].as_usize().unwrap())),
        grid,
    );
    let result = gs.tune_regress(&task, &cv, &Rmse).unwrap();
    assert_eq!(result.all_results.len(), 2);
    assert!(result.best_score.is_finite());
    // Each candidate's score is exactly what resample_regress gives it.
    for (params, score) in &result.all_results {
        let bw = params["bandwidth"].as_usize().unwrap();
        let bench = benchmark::resample_regress(&mut geo(bw), &task, &cv, &[&Rmse]).unwrap();
        assert_eq!(*score, bench.mean_scores()[0]);
    }
}

#[test]
fn registry_geo_xgboost_runs_in_benchmark_design() {
    let (features, target, coords) = spatial_fixture();
    let task = RegressionTask::new("design", features, target)
        .unwrap()
        .with_coords(coords.clone());
    let cv = SpatialBlockCV::new(3, coords);
    let mut learners: Vec<Box<dyn Learner>> = vec![
        learner_from_id("geo_xgboost").unwrap(),
        learner_from_id("linear_regression").unwrap(),
    ];
    let design = benchmark_regress(&mut learners, &[&task], &cv, &[&Rmse]).unwrap();
    assert_eq!(design.entries.len(), 2);
    assert_eq!(design.entries[0].learner_id, "geo_xgboost");
    assert!(design.entries.iter().all(|e| e.mean_scores[0].is_finite()));
}
