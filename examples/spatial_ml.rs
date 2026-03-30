//! Spatial ML: Geographical-XGBoost and spatial cross-validation.
//!
//! Run with: cargo run --example spatial_ml

use ndarray::array;
use smelt_ml::prelude::*;

fn main() {
    // Simulated spatial dataset: temperature varies with location
    // Left region: y ~ x, Right region: y ~ -x + 20
    let features = array![
        [1.0], [2.0], [3.0], [4.0], [5.0],       // left region
        [1.0], [2.0], [3.0], [4.0], [5.0],        // right region (same features, different target)
    ];
    let target = vec![
        1.0, 2.0, 3.0, 4.0, 5.0,       // y = x
        19.0, 18.0, 17.0, 16.0, 15.0,   // y = -x + 20
    ];
    let coords: Vec<(f64, f64)> = vec![
        (0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0),
        (100.0, 0.0), (101.0, 0.0), (102.0, 0.0), (103.0, 0.0), (104.0, 0.0),
    ];

    let task = RegressionTask::new("spatial", features.clone(), target.clone()).unwrap();

    // 1. Standard XGBoost (ignores spatial structure)
    let mut xgb = XGBoost::new().with_n_estimators(50);
    let xgb_model = xgb.train_regress(&task).unwrap();
    let xgb_pred = xgb_model.predict(&features).unwrap()
        .with_truth_regress(target.clone());
    let xgb_rmse = Rmse.score(&xgb_pred).unwrap();

    // 2. Geographical-XGBoost (spatial awareness)
    let mut gxgb = GeoXGBoost::new(coords.clone())
        .with_bandwidth(4)
        .with_n_estimators(50);
    let gxgb_model = gxgb.train_regress(&task).unwrap();
    let gxgb_pred = gxgb_model.predict(&features).unwrap()
        .with_truth_regress(target.clone());
    let gxgb_rmse = Rmse.score(&gxgb_pred).unwrap();

    println!("=== Spatial Heterogeneity Demo ===");
    println!("Two regions with opposite relationships (y=x vs y=-x+20)\n");
    println!("Standard XGBoost RMSE:      {:.4}", xgb_rmse);
    println!("Geographical-XGBoost RMSE:  {:.4}", gxgb_rmse);
    println!();

    // 3. Spatial Cross-Validation
    println!("=== Spatial Cross-Validation ===");
    let spatial_cv = SpatialBlockCV::new(2, coords);
    let splits = spatial_cv.splits(task.n_samples());

    for (i, (train, test)) in splits.iter().enumerate() {
        println!("Fold {}: train={} samples, test={} samples", i + 1, train.len(), test.len());
    }

    // Evaluate with spatial CV
    let mut xgb2 = XGBoost::new().with_n_estimators(50);
    let result = benchmark::resample_regress(&mut xgb2, &task, &spatial_cv, &[&Rmse]).unwrap();
    println!("Spatial CV RMSE: {:.4}", result.mean_scores()[0]);
}
