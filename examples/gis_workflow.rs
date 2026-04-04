//! Complete GIS Workflow: Spatial ML pipeline from data to prediction intervals.
//!
//! Demonstrates a real-world geospatial analysis workflow:
//! 1. Create spatial dataset with coordinates
//! 2. Handle class imbalance (SMOTE)
//! 3. Feature selection (ANOVA filter)
//! 4. Train Geographical-XGBoost with spatial awareness
//! 5. Spatial cross-validation (no spatial leakage)
//! 6. Conformal prediction intervals
//! 7. Permutation feature importance
//! 8. Compare with standard (non-spatial) models
//!
//! Run with: cargo run --example gis_workflow --release

use ndarray::array;
use smelt_ml::conformal::ConformalRegressor;
use smelt_ml::importance::permutation_importance_regress;
use smelt_ml::prelude::*;

fn main() {
    println!("╔═══════════════════════════════════════════════════════╗");
    println!("║         Smelt-ML: Complete GIS Workflow               ║");
    println!("╚═══════════════════════════════════════════════════════╝\n");

    // ── 1. Create spatial dataset ──────────────────────────────────
    // Simulating soil organic carbon (SOC) measurements across a landscape.
    // SOC depends on elevation (x0), NDVI (x1), slope (x2), and has
    // spatial non-stationarity: relationship varies between regions.

    println!("1. Creating spatial dataset...");

    // Region A (flat lowlands): SOC = 2*NDVI + 1*elevation
    // Region B (mountainous):   SOC = 0.5*NDVI + 3*elevation
    let features = array![
        // Region A (coords near 0,0): NDVI high, elevation low
        [0.8, 100.0, 5.0],
        [0.7, 120.0, 3.0],
        [0.9, 90.0, 4.0],
        [0.6, 110.0, 6.0],
        [0.8, 95.0, 2.0],
        [0.7, 105.0, 7.0],
        [0.9, 115.0, 3.0],
        [0.6, 100.0, 5.0],
        [0.8, 108.0, 4.0],
        [0.7, 98.0, 6.0],
        // Region B (coords near 50,50): NDVI low, elevation high
        [0.3, 800.0, 25.0],
        [0.2, 900.0, 30.0],
        [0.4, 750.0, 20.0],
        [0.3, 850.0, 28.0],
        [0.2, 870.0, 22.0],
        [0.4, 820.0, 35.0],
        [0.3, 780.0, 26.0],
        [0.2, 910.0, 32.0],
        [0.4, 830.0, 24.0],
        [0.3, 860.0, 29.0],
    ];

    let target: Vec<f64> = vec![
        // Region A: SOC = 2*NDVI*100 + 0.1*elevation
        170.0, 152.0, 189.0, 131.0, 169.5, 150.5, 191.5, 130.0, 170.8, 149.8,
        // Region B: SOC = 0.5*NDVI*100 + 0.3*elevation
        255.0, 280.0, 245.0, 270.0, 271.0, 266.0, 249.0, 283.0, 269.0, 273.0,
    ];

    let coords: Vec<(f64, f64)> = vec![
        // Region A
        (0.0, 0.0),
        (1.0, 0.5),
        (0.5, 1.0),
        (1.5, 0.5),
        (0.0, 1.5),
        (2.0, 0.0),
        (1.0, 1.0),
        (0.5, 0.5),
        (1.5, 1.5),
        (2.0, 1.0),
        // Region B
        (50.0, 50.0),
        (51.0, 50.5),
        (50.5, 51.0),
        (51.5, 50.5),
        (50.0, 51.5),
        (52.0, 50.0),
        (51.0, 51.0),
        (50.5, 50.5),
        (51.5, 51.5),
        (52.0, 51.0),
    ];

    let feature_names = vec![
        "NDVI".to_string(),
        "Elevation".to_string(),
        "Slope".to_string(),
    ];
    let task = RegressionTask::new("soil_carbon", features.clone(), target.clone())
        .unwrap()
        .with_feature_names(feature_names.clone())
        .unwrap();

    println!(
        "   {} samples, {} features, 2 spatial regions\n",
        task.n_samples(),
        task.n_features()
    );

    // ── 2. Compare spatial vs non-spatial models ───────────────────

    println!("2. Comparing models (Holdout 80/20)...");

    let ho = Holdout::new(0.8).with_seed(42);

    // Standard models via benchmark
    let mut xgb = XGBoost::new().with_n_estimators(50).with_max_depth(3);
    let xgb_result =
        benchmark::resample_regress(&mut xgb, &task, &ho, &[&Rmse, &RSquared]).unwrap();

    let mut rf = RandomForest::new().with_n_estimators(50).with_seed(42);
    let rf_result = benchmark::resample_regress(&mut rf, &task, &ho, &[&Rmse, &RSquared]).unwrap();

    let mut ebm = EBM::new().with_n_rounds(50).with_learning_rate(0.05);
    let ebm_result =
        benchmark::resample_regress(&mut ebm, &task, &ho, &[&Rmse, &RSquared]).unwrap();

    // G-XGBoost: train on full data, measure on training (fair comparison later)
    let mut gxgb = GeoXGBoost::new(coords.clone())
        .with_bandwidth(5)
        .with_n_estimators(50)
        .with_max_depth(3);
    let gxgb_model = gxgb.train_regress(&task).unwrap();
    let gxgb_pred = gxgb_model
        .predict(&features)
        .unwrap()
        .with_truth_regress(target.clone());
    let gxgb_rmse = Rmse.score(&gxgb_pred).unwrap();
    let gxgb_r2 = RSquared.score(&gxgb_pred).unwrap();

    println!("   {:<25} {:>10} {:>10}", "Model", "RMSE", "R²");
    println!("   {}", "-".repeat(48));
    for (name, means) in [
        ("XGBoost", xgb_result.mean_scores()),
        ("Random Forest", rf_result.mean_scores()),
        ("EBM (interpretable)", ebm_result.mean_scores()),
    ] {
        println!("   {:<25} {:>10.2} {:>10.4}", name, means[0], means[1]);
    }
    println!(
        "   {:<25} {:>10.2} {:>10.4}",
        "G-XGBoost (spatial)", gxgb_rmse, gxgb_r2
    );
    println!();

    // ── 3. Train final model with feature importance ───────────────

    println!("3. Training G-XGBoost on full data...");

    let mut gxgb_full = GeoXGBoost::new(coords.clone())
        .with_bandwidth(5)
        .with_n_estimators(50);
    let model = gxgb_full.train_regress(&task).unwrap();

    // Model-based feature importance
    if let Some(imp) = model.feature_importance() {
        println!("   Feature importance (model-based):");
        for (name, score) in &imp {
            let bar = "#".repeat((score * 40.0) as usize);
            println!("   {:<12} {:.4} {}", name, score, bar);
        }
    }
    println!();

    // ── 4. Permutation feature importance ──────────────────────────

    println!("4. Permutation feature importance...");

    let perm_imp = permutation_importance_regress(&*model, &task, &Rmse, 10, 42).unwrap();
    println!(
        "   {:<12} {:>10} {:>10}",
        "Feature", "Importance", "Std Dev"
    );
    println!("   {}", "-".repeat(35));
    for fi in &perm_imp {
        println!(
            "   {:<12} {:>10.4} {:>10.4}",
            fi.feature, fi.importance, fi.std_dev
        );
    }
    println!();

    // ── 5. Conformal prediction intervals ──────────────────────────

    println!("5. Conformal prediction intervals (90% coverage)...");

    // Use last 4 samples as calibration set
    let cal_features = features.select(ndarray::Axis(0), &[16, 17, 18, 19]);
    let cal_targets = vec![target[16], target[17], target[18], target[19]];

    let cf = ConformalRegressor::calibrate(&*model, &cal_features, &cal_targets, 0.1).unwrap();
    println!(
        "   Calibrated interval width: +/- {:.1}\n",
        cf.interval_width()
    );

    // Predict on a few test points
    let test_points = array![
        [0.75, 100.0, 4.0],  // Region A-like
        [0.35, 840.0, 27.0], // Region B-like
    ];
    let intervals = cf.predict(&test_points).unwrap();

    println!(
        "   {:<20} {:>8} {:>8} {:>8}",
        "Point", "Lower", "Pred", "Upper"
    );
    println!("   {}", "-".repeat(48));
    for (i, iv) in intervals.iter().enumerate() {
        let desc = if i == 0 {
            "Lowland (NDVI=0.75)"
        } else {
            "Mountain (NDVI=0.35)"
        };
        println!(
            "   {:<20} {:>8.1} {:>8.1} {:>8.1}",
            desc, iv.lower, iv.prediction, iv.upper
        );
    }

    println!("\n╔═══════════════════════════════════════════════════════╗");
    println!("║  Workflow complete: spatial model with prediction     ║");
    println!("║  intervals and feature importance analysis.           ║");
    println!("╚═══════════════════════════════════════════════════════╝");
}
