//! Conformal Prediction: prediction intervals with coverage guarantees.
//!
//! Run with: cargo run --example conformal_prediction

use ndarray::array;
use smelt_ml::conformal::{ConformalClassifier, ConformalRegressor};
use smelt_ml::prelude::*;

fn main() {
    // === Regression: prediction intervals ===
    println!("=== Conformal Regression (90% coverage) ===\n");

    let features = array![
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
        [6.0],
        [7.0],
        [8.0],
        [9.0],
        [10.0],
    ];
    let target = vec![2.1, 3.9, 6.2, 7.8, 10.1, 11.9, 14.2, 15.8, 18.1, 19.9];

    // Train on first 7, calibrate on last 3
    let train_features = features.slice(ndarray::s![..7, ..]).to_owned();
    let train_target = target[..7].to_vec();
    let cal_features = features.slice(ndarray::s![7.., ..]).to_owned();
    let cal_target = &target[7..];

    let train_task = RegressionTask::new("train", train_features, train_target).unwrap();
    let mut xgb = XGBoost::new().with_n_estimators(50);
    let model = xgb.train_regress(&train_task).unwrap();

    // Calibrate: alpha=0.1 for 90% coverage
    let cf = ConformalRegressor::calibrate(&*model, &cal_features, cal_target, 0.1).unwrap();
    println!(
        "Calibrated interval width: +/- {:.2}\n",
        cf.interval_width()
    );

    // Predict with intervals
    let test = array![[3.5], [5.5], [8.5]];
    let intervals = cf.predict(&test).unwrap();

    println!("{:<10} {:>10} {:>10} {:>10}", "x", "Lower", "Pred", "Upper");
    println!("{}", "-".repeat(42));
    for (i, iv) in intervals.iter().enumerate() {
        println!(
            "{:<10.1} {:>10.2} {:>10.2} {:>10.2}",
            test[[i, 0]],
            iv.lower,
            iv.prediction,
            iv.upper
        );
    }

    // === Classification: prediction sets ===
    println!("\n=== Conformal Classification (90% coverage) ===\n");

    let features_c = array![[0.0], [0.5], [1.0], [1.5], [2.0], [2.5], [3.0], [3.5],];
    let target_c = vec![0, 0, 0, 0, 1, 1, 1, 1];
    let task_c = ClassificationTask::new("classif", features_c.clone(), target_c.clone()).unwrap();

    let mut dt = DecisionTree::default();
    let model_c = dt.train_classif(&task_c).unwrap();

    let cal_feat_c = array![[1.0], [2.0], [0.5], [2.5]];
    let cal_target_c = vec![0, 1, 0, 1];

    let cf_c = ConformalClassifier::calibrate(&*model_c, &cal_feat_c, &cal_target_c, 0.1).unwrap();
    let sets = cf_c.predict(&array![[0.3], [1.5], [2.8]]).unwrap();

    for (i, s) in sets.iter().enumerate() {
        println!(
            "x={:.1}: predicted={}, prediction_set={:?}",
            [0.3, 1.5, 2.8][i],
            s.prediction,
            s.prediction_set
        );
    }
    println!("\nPrediction sets contain the true class with >= 90% probability.");
}
