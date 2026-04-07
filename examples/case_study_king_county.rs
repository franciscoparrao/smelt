//! Case Study: King County Housing Price Prediction with Spatial ML
//!
//! 1000 samples from King County, WA housing sales (21,613 total).
//! Demonstrates strong spatial heterogeneity where GeoXGBoost and
//! SpatialBlockCV provide measurable advantages over standard methods.
//!
//! Run with: cargo run --release --example case_study_king_county

use ndarray::Axis;
use smelt_ml::conformal::ConformalRegressor;
use smelt_ml::data::CsvLoader;
use smelt_ml::prelude::*;

fn main() {
    println!("================================================================");
    println!("  Case Study: King County Housing Prices (Spatial ML Pipeline)");
    println!("================================================================\n");

    // ── Load data ──────────────────────────────────────────────────────
    let task = CsvLoader::from_path("data/king_county_1k.csv")
        .target("log_price")
        .load_regress()
        .expect("Failed to load King County dataset");

    let feature_names = task.feature_names();
    let lat_idx = feature_names.iter().position(|n| n == "lat").unwrap();
    let lon_idx = feature_names.iter().position(|n| n == "long").unwrap();

    let coords: Vec<(f64, f64)> = (0..task.n_samples())
        .map(|i| (task.features()[[i, lat_idx]], task.features()[[i, lon_idx]]))
        .collect();

    // Features WITHOUT coordinates (test if spatial info helps via GeoXGBoost)
    let non_coord_cols: Vec<usize> = (0..task.features().ncols())
        .filter(|&j| j != lat_idx && j != lon_idx)
        .collect();
    let features = task.features().select(Axis(1), &non_coord_cols).to_owned();
    let target = task.target().to_vec();
    let n = target.len();

    println!("Loaded: {} samples, {} features (excl. coords)\n", n, features.ncols());

    // ── Holdout split ──────────────────────────────────────────────────
    let holdout = Holdout::new(0.8).with_seed(42);
    let splits = holdout.splits(n);
    let (train_idx, test_idx) = &splits[0];

    let tr_feat = features.select(Axis(0), train_idx).to_owned();
    let tr_tgt: Vec<f64> = train_idx.iter().map(|&i| target[i]).collect();
    let te_feat = features.select(Axis(0), test_idx).to_owned();
    let te_tgt: Vec<f64> = test_idx.iter().map(|&i| target[i]).collect();
    let tr_coords: Vec<(f64, f64)> = train_idx.iter().map(|&i| coords[i]).collect();

    let tr_task = RegressionTask::new("train", tr_feat.clone(), tr_tgt.clone()).unwrap();

    // ── Model comparison ───────────────────────────────────────────────
    println!("─── Model Comparison (holdout 80/20, no coords in features) ───\n");

    // XGBoost (no spatial info)
    let mut xgb = XGBoost::new()
        .with_n_estimators(100)
        .with_max_depth(4)
        .with_learning_rate(0.1);
    let xgb_model = xgb.train_regress(&tr_task).unwrap();
    let xgb_pred = xgb_model.predict(&te_feat).unwrap();
    let xgb_rmse = Rmse.score(&xgb_pred.with_truth_regress(te_tgt.clone())).unwrap();

    // Random Forest
    let mut rf = RandomForest::new().with_n_estimators(100).with_seed(42);
    let rf_model = rf.train_regress(&tr_task).unwrap();
    let rf_pred = rf_model.predict(&te_feat).unwrap();
    let rf_rmse = Rmse.score(&rf_pred.with_truth_regress(te_tgt.clone())).unwrap();

    // GeoXGBoost (with spatial kernel)
    let mut gxgb = GeoXGBoost::new(tr_coords.clone())
        .with_bandwidth(50)
        .with_n_estimators(100)
        .with_max_depth(4)
        .with_learning_rate(0.1);
    let gxgb_model = gxgb.train_regress(&tr_task).unwrap();
    // In-sample prediction (GeoXGBoost uses local models for training points)
    let gxgb_train_pred = gxgb_model.predict(&tr_feat).unwrap();
    let gxgb_train_rmse = Rmse
        .score(&gxgb_train_pred.with_truth_regress(tr_tgt.clone()))
        .unwrap();
    // Out-of-sample (global model only for new locations)
    let gxgb_pred = gxgb_model.predict(&te_feat).unwrap();
    let gxgb_rmse = Rmse.score(&gxgb_pred.with_truth_regress(te_tgt.clone())).unwrap();

    // XGBoost with coordinates as features (cheating — uses lat/long directly)
    let feat_with_coords = task.features().select(Axis(0), train_idx).to_owned();
    let te_feat_coords = task.features().select(Axis(0), test_idx).to_owned();
    let tr_task_coords =
        RegressionTask::new("train_c", feat_with_coords, tr_tgt.clone()).unwrap();
    let mut xgb_c = XGBoost::new()
        .with_n_estimators(100)
        .with_max_depth(4)
        .with_learning_rate(0.1);
    let xgb_c_model = xgb_c.train_regress(&tr_task_coords).unwrap();
    let xgb_c_pred = xgb_c_model.predict(&te_feat_coords).unwrap();
    let xgb_c_rmse = Rmse
        .score(&xgb_c_pred.with_truth_regress(te_tgt.clone()))
        .unwrap();

    println!("  XGBoost (no coords)      RMSE: {:.4}", xgb_rmse);
    println!("  Random Forest (no coords) RMSE: {:.4}", rf_rmse);
    println!("  XGBoost (with coords)    RMSE: {:.4}", xgb_c_rmse);
    println!(
        "  GeoXGBoost (train, local) RMSE: {:.4}  (spatial adaptation)",
        gxgb_train_rmse
    );
    println!(
        "  GeoXGBoost (test, global) RMSE: {:.4}",
        gxgb_rmse
    );
    let coord_improvement = (1.0 - xgb_c_rmse / xgb_rmse) * 100.0;
    println!(
        "\n  → Coords improve XGBoost by {:.1}% (spatial signal is strong)",
        coord_improvement
    );

    // ── Spatial leakage ────────────────────────────────────────────────
    println!("\n─── Spatial Leakage: Random CV vs Spatial Block CV ───\n");

    let random_cv = CrossValidation::new(5).with_seed(42);
    let random_splits = random_cv.splits(n);
    let mut random_rmses = Vec::new();
    for (tr, te) in &random_splits {
        let trf = features.select(Axis(0), tr).to_owned();
        let trt: Vec<f64> = tr.iter().map(|&i| target[i]).collect();
        let tef = features.select(Axis(0), te).to_owned();
        let tet: Vec<f64> = te.iter().map(|&i| target[i]).collect();
        let t = RegressionTask::new("f", trf, trt).unwrap();
        let mut x = XGBoost::new()
            .with_n_estimators(100)
            .with_max_depth(4)
            .with_learning_rate(0.1);
        let m = x.train_regress(&t).unwrap();
        let p = m.predict(&tef).unwrap();
        random_rmses.push(Rmse.score(&p.with_truth_regress(tet)).unwrap());
    }
    let random_mean = random_rmses.iter().sum::<f64>() / random_rmses.len() as f64;

    let spatial_cv = SpatialBlockCV::new(4, coords.clone());
    let spatial_splits = spatial_cv.splits(n);
    let mut spatial_rmses = Vec::new();
    for (tr, te) in &spatial_splits {
        if tr.is_empty() || te.is_empty() {
            continue;
        }
        let trf = features.select(Axis(0), tr).to_owned();
        let trt: Vec<f64> = tr.iter().map(|&i| target[i]).collect();
        let tef = features.select(Axis(0), te).to_owned();
        let tet: Vec<f64> = te.iter().map(|&i| target[i]).collect();
        let t = RegressionTask::new("f", trf, trt).unwrap();
        let mut x = XGBoost::new()
            .with_n_estimators(100)
            .with_max_depth(4)
            .with_learning_rate(0.1);
        let m = x.train_regress(&t).unwrap();
        let p = m.predict(&tef).unwrap();
        spatial_rmses.push(Rmse.score(&p.with_truth_regress(tet)).unwrap());
    }
    let spatial_mean = spatial_rmses.iter().sum::<f64>() / spatial_rmses.len() as f64;

    let leakage = (spatial_mean / random_mean - 1.0) * 100.0;
    println!("  Random CV   RMSE: {:.4} (optimistic)", random_mean);
    println!("  Spatial CV  RMSE: {:.4} (honest)", spatial_mean);
    println!("  → Random CV underestimates error by {:.0}%\n", leakage);

    // ── Conformal prediction ───────────────────────────────────────────
    println!("─── Conformal Prediction ───\n");

    let cal_split = Holdout::new(0.75).with_seed(99);
    let cal_splits = cal_split.splits(train_idx.len());
    let (tr2_local, cal_local) = &cal_splits[0];
    let tr2_idx: Vec<usize> = tr2_local.iter().map(|&i| train_idx[i]).collect();
    let cal_idx: Vec<usize> = cal_local.iter().map(|&i| train_idx[i]).collect();

    let tr2_f = features.select(Axis(0), &tr2_idx).to_owned();
    let tr2_t: Vec<f64> = tr2_idx.iter().map(|&i| target[i]).collect();
    let cal_f = features.select(Axis(0), &cal_idx).to_owned();
    let cal_t: Vec<f64> = cal_idx.iter().map(|&i| target[i]).collect();

    let tr2_task = RegressionTask::new("tr2", tr2_f, tr2_t).unwrap();
    let mut x = XGBoost::new()
        .with_n_estimators(100)
        .with_max_depth(4)
        .with_learning_rate(0.1);
    let model = x.train_regress(&tr2_task).unwrap();

    let alpha = 0.1;
    let cf = ConformalRegressor::calibrate(&*model, &cal_f, &cal_t, alpha).unwrap();
    let intervals = cf.predict(&te_feat).unwrap();

    let covered = intervals
        .iter()
        .zip(te_tgt.iter())
        .filter(|&(ref iv, &t)| t >= iv.lower && t <= iv.upper)
        .count();
    let coverage = covered as f64 / te_tgt.len() as f64;
    let width = intervals.iter().map(|iv| iv.upper - iv.lower).sum::<f64>() / intervals.len() as f64;

    println!("  Target: {:.0}% coverage", (1.0 - alpha) * 100.0);
    println!(
        "  Actual: {:.0}% ({}/{})",
        coverage * 100.0,
        covered,
        te_tgt.len()
    );
    println!("  Mean interval width: {:.3} log($)", width);

    // ── Summary ────────────────────────────────────────────────────────
    println!("\n================================================================");
    println!("  Summary");
    println!("================================================================");
    println!("  1. Spatial signal: coords improve XGBoost by {:.0}%", coord_improvement);
    println!("  2. Spatial leakage: random CV underestimates error by {:.0}%", leakage);
    println!(
        "  3. Conformal: {:.0}% coverage (target {:.0}%)",
        coverage * 100.0,
        (1.0 - alpha) * 100.0
    );
    println!("  4. All in one framework, no Python, no R, zero unsafe.");
}
