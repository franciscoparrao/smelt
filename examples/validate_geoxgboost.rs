//! Validate smelt-ml GeoXGBoost against the Python geoxgboost package.
//!
//! Reads the same data and train/test indices produced by the Python script,
//! runs GeoXGBoost with matched hyperparameters, and outputs RMSE for comparison.
//!
//! Run with: cargo run --release --example validate_geoxgboost

use ndarray::Axis;
use smelt_ml::data::CsvLoader;
use smelt_ml::learner::TrainedModel;
use smelt_ml::prelude::*;
use std::fs;
use std::time::Instant;

fn read_indices(path: &str) -> Vec<usize> {
    fs::read_to_string(path)
        .unwrap_or_else(|_| panic!("Cannot read {path}. Run validate_geoxgboost.py first."))
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.trim().parse::<usize>().unwrap())
        .collect()
}

fn rmse(predicted: &[f64], actual: &[f64]) -> f64 {
    let n = predicted.len() as f64;
    let ss: f64 = predicted
        .iter()
        .zip(actual)
        .map(|(p, a)| (p - a).powi(2))
        .sum();
    (ss / n).sqrt()
}

fn r2(predicted: &[f64], actual: &[f64]) -> f64 {
    let mean = actual.iter().sum::<f64>() / actual.len() as f64;
    let ss_res: f64 = predicted
        .iter()
        .zip(actual)
        .map(|(p, a)| (p - a).powi(2))
        .sum();
    let ss_tot: f64 = actual.iter().map(|a| (a - mean).powi(2)).sum();
    1.0 - ss_res / ss_tot
}

fn main() {
    println!("================================================================");
    println!("  Validate smelt-ml GeoXGBoost vs geoxgboost (Python)");
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

    // Features WITHOUT coordinates (matches Python script)
    let non_coord_cols: Vec<usize> = (0..task.features().ncols())
        .filter(|&j| j != lat_idx && j != lon_idx)
        .collect();
    let features = task.features().select(Axis(1), &non_coord_cols).to_owned();
    let target = task.target().to_vec();
    let n_features = features.ncols();

    println!("Loaded: {} samples, {} features (excl. coords)", target.len(), n_features);

    // ── Load train/test indices from Python ────────────────────────────
    let train_idx = read_indices("paper/replication/geoxgb_train_idx.csv");
    let test_idx = read_indices("paper/replication/geoxgb_test_idx.csv");

    println!("Train: {}, Test: {} (indices from Python)\n", train_idx.len(), test_idx.len());

    let tr_feat = features.select(Axis(0), &train_idx).to_owned();
    let tr_tgt: Vec<f64> = train_idx.iter().map(|&i| target[i]).collect();
    let te_feat = features.select(Axis(0), &test_idx).to_owned();
    let te_tgt: Vec<f64> = test_idx.iter().map(|&i| target[i]).collect();
    let tr_coords: Vec<(f64, f64)> = train_idx.iter().map(|&i| coords[i]).collect();

    let tr_task = RegressionTask::new("train", tr_feat.clone(), tr_tgt.clone()).unwrap();

    // ── Matched hyperparameters ────────────────────────────────────────
    // Must match validate_geoxgboost.py exactly
    let bandwidth = 30;
    let n_estimators = 100;
    let max_depth = 6;
    let learning_rate = 0.3;
    let lambda = 1.0;
    let seed = 42;

    // ── 1. Standard XGBoost (baseline) ─────────────────────────────────
    println!("─── Standard XGBoost (baseline) ───");
    let mut xgb = XGBoost::new()
        .with_n_estimators(n_estimators)
        .with_max_depth(max_depth)
        .with_learning_rate(learning_rate)
        .with_lambda(lambda)
        .with_seed(seed);

    let t0 = Instant::now();
    let xgb_model = xgb.train_regress(&tr_task).unwrap();
    let xgb_train_time = t0.elapsed();

    let xgb_pred = xgb_model.predict(&te_feat).unwrap();
    let xgb_vals = match &xgb_pred {
        Prediction::Regression { predicted, .. } => predicted.clone(),
        _ => panic!("Expected regression"),
    };

    let xgb_rmse = rmse(&xgb_vals, &te_tgt);
    let xgb_r2 = r2(&xgb_vals, &te_tgt);
    println!("RMSE: {:.4}", xgb_rmse);
    println!("R²:   {:.4}", xgb_r2);
    println!("Time: {:.2}s\n", xgb_train_time.as_secs_f64());

    // ── 2. GeoXGBoost ──────────────────────────────────────────────────
    println!("─── GeoXGBoost (smelt-ml) ───");
    // alpha=1.0 → pure local prediction (matches geoxgboost Python defaults:
    // alpha_wt=1, alpha_wt_type='varying' → 100% local, no blending)
    let mut gxgb = GeoXGBoost::new(tr_coords.clone())
        .with_bandwidth(bandwidth)
        .with_n_estimators(n_estimators)
        .with_max_depth(max_depth)
        .with_learning_rate(learning_rate)
        .with_lambda(lambda)
        .with_alpha(1.0)
        .with_seed(seed);

    let t0 = Instant::now();
    // Use train_geo() for concrete type with predict_spatial()
    let gxgb_model = gxgb.train_geo(&tr_task).unwrap();
    let gxgb_train_time = t0.elapsed();

    // In-sample predictions (local models, via predict_spatial with the
    // training coords — predict() alone is global-only by design)
    let gxgb_train_pred = gxgb_model.predict_spatial(&tr_feat, &tr_coords).unwrap();
    let gxgb_train_vals = match &gxgb_train_pred {
        Prediction::Regression { predicted, .. } => predicted.clone(),
        _ => panic!("Expected regression"),
    };
    let gxgb_train_rmse = rmse(&gxgb_train_vals, &tr_tgt);
    let gxgb_train_r2 = r2(&gxgb_train_vals, &tr_tgt);

    // Out-of-sample predictions with spatial nearest-neighbor lookup
    let te_coords: Vec<(f64, f64)> = test_idx.iter().map(|&i| coords[i]).collect();
    let gxgb_pred = gxgb_model.predict_spatial(&te_feat, &te_coords).unwrap();
    let gxgb_vals = match &gxgb_pred {
        Prediction::Regression { predicted, .. } => predicted.clone(),
        _ => panic!("Expected regression"),
    };

    let gxgb_rmse = rmse(&gxgb_vals, &te_tgt);
    let gxgb_r2 = r2(&gxgb_vals, &te_tgt);

    println!("In-sample  RMSE: {:.4}, R²: {:.4}", gxgb_train_rmse, gxgb_train_r2);
    println!("Out-sample RMSE: {:.4}, R²: {:.4}", gxgb_rmse, gxgb_r2);
    println!("Time: {:.2}s\n", gxgb_train_time.as_secs_f64());

    // ── Save predictions for comparison ────────────────────────────────
    let pred_str: String = gxgb_vals
        .iter()
        .map(|v| format!("{:.6}", v))
        .collect::<Vec<_>>()
        .join("\n");
    fs::write("paper/replication/smelt_geoxgb_test_preds.csv", &pred_str).unwrap();

    let xgb_pred_str: String = xgb_vals
        .iter()
        .map(|v| format!("{:.6}", v))
        .collect::<Vec<_>>()
        .join("\n");
    fs::write("paper/replication/smelt_xgb_test_preds.csv", &xgb_pred_str).unwrap();

    // ── Load Python predictions for direct comparison ──────────────────
    let py_geoxgb: Vec<f64> = fs::read_to_string("paper/replication/geoxgb_test_preds.csv")
        .unwrap_or_default()
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.trim().parse::<f64>().unwrap_or(0.0))
        .collect();

    let py_xgb: Vec<f64> = fs::read_to_string("paper/replication/xgb_test_preds.csv")
        .unwrap_or_default()
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.trim().parse::<f64>().unwrap_or(0.0))
        .collect();

    println!("═══════════════════════════════════════════════════════════");
    println!("  VALIDATION SUMMARY");
    println!("═══════════════════════════════════════════════════════════");
    println!("{:<30} {:>8} {:>8}", "Method", "RMSE", "R²");
    println!("{}", "-".repeat(50));
    println!("{:<30} {:>8.4} {:>8.4}", "XGBoost (smelt-ml)", xgb_rmse, xgb_r2);
    println!("{:<30} {:>8.4} {:>8.4}", "GeoXGBoost (smelt-ml)", gxgb_rmse, gxgb_r2);

    if !py_xgb.is_empty() {
        let py_xgb_rmse = rmse(&py_xgb, &te_tgt);
        let py_xgb_r2 = r2(&py_xgb, &te_tgt);
        println!("{:<30} {:>8.4} {:>8.4}", "XGBoost (official C++)", py_xgb_rmse, py_xgb_r2);
    }
    if !py_geoxgb.is_empty() {
        let py_geoxgb_rmse = rmse(&py_geoxgb, &te_tgt);
        let py_geoxgb_r2 = r2(&py_geoxgb, &te_tgt);
        println!("{:<30} {:>8.4} {:>8.4}", "geoxgboost (Grekousis)", py_geoxgb_rmse, py_geoxgb_r2);
    }

    // Per-point correlation between implementations
    if !py_geoxgb.is_empty() && py_geoxgb.len() == gxgb_vals.len() {
        let pred_diff: Vec<f64> = gxgb_vals
            .iter()
            .zip(&py_geoxgb)
            .map(|(a, b)| (a - b).abs())
            .collect();
        let mean_diff: f64 = pred_diff.iter().sum::<f64>() / pred_diff.len() as f64;
        let max_diff: f64 = pred_diff.iter().cloned().fold(0.0_f64, f64::max);
        println!("\n─── Per-point prediction difference (GeoXGBoost) ───");
        println!("Mean |smelt - geoxgboost|: {:.4}", mean_diff);
        println!("Max  |smelt - geoxgboost|: {:.4}", max_diff);
    }

    println!("\nDone. Predictions saved to paper/replication/smelt_*.csv");
}
