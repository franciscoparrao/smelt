//! Performance benchmark: smelt-ml XGBoost vs official XGBoost timing.
//!
//! Run with: cargo test --test xgboost_perf --release -- --nocapture

use ndarray::Array2;
use smelt_ml::prelude::*;
use std::time::Instant;

fn load_dataset(prefix: &str, n: usize) -> (Array2<f64>, Vec<f64>) {
    let x_str = std::fs::read_to_string(format!("/tmp/bench_{prefix}_{n}_X.csv"))
        .unwrap_or_else(|_| panic!("Run tests/xgboost_perf.py first"));
    let y_str = std::fs::read_to_string(format!("/tmp/bench_{prefix}_{n}_y.csv")).unwrap();

    let rows: Vec<Vec<f64>> = x_str
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.split(',').map(|v| v.trim().parse().unwrap()).collect())
        .collect();
    let n_samples = rows.len();
    let n_features = rows[0].len();
    let mut features = Array2::zeros((n_samples, n_features));
    for (i, row) in rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            features[[i, j]] = v;
        }
    }
    let target: Vec<f64> = y_str
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.trim().parse().unwrap())
        .collect();
    (features, target)
}

#[test]
fn xgboost_performance_classif() {
    let sizes = [100, 500, 1000, 5000, 10000];

    println!("\n=== smelt-ml XGBoost — Classification Benchmark ===");
    println!(
        "{:>7} {:>8} {:>5} {:>10} {:>6}",
        "N", "Features", "Trees", "Time (ms)", "Acc"
    );
    println!("{}", "-".repeat(45));

    for &n in &sizes {
        let (features, target_f64) = load_dataset("classif", n);
        let target: Vec<usize> = target_f64.iter().map(|&v| v as usize).collect();
        let task = ClassificationTask::new("bench", features.clone(), target).unwrap();

        let mut xgb = XGBoost::new()
            .with_n_estimators(100)
            .with_max_depth(6)
            .with_learning_rate(0.3)
            .with_lambda(1.0)
            .with_seed(42);

        let t0 = Instant::now();
        let model = xgb.train_classif(&task).unwrap();
        let elapsed = t0.elapsed().as_secs_f64() * 1000.0;

        let pred = model
            .predict(&features)
            .unwrap()
            .with_truth_classif(task.target().to_vec());
        let acc = Accuracy.score(&pred).unwrap();

        println!("{n:>7} {:>8} {:>5} {elapsed:>10.1} {acc:>6.4}", 20, 100);
    }
}

#[test]
fn xgboost_performance_regress() {
    let sizes = [100, 500, 1000, 5000, 10000];

    println!("\n=== smelt-ml XGBoost — Regression Benchmark ===");
    println!(
        "{:>7} {:>8} {:>5} {:>10}",
        "N", "Features", "Trees", "Time (ms)"
    );
    println!("{}", "-".repeat(38));

    for &n in &sizes {
        let (features, target) = load_dataset("regress", n);
        let task = RegressionTask::new("bench", features, target).unwrap();

        let mut xgb = XGBoost::new()
            .with_n_estimators(100)
            .with_max_depth(6)
            .with_learning_rate(0.3)
            .with_lambda(1.0)
            .with_seed(42);

        let t0 = Instant::now();
        let _model = xgb.train_regress(&task).unwrap();
        let elapsed = t0.elapsed().as_secs_f64() * 1000.0;

        println!("{n:>7} {:>8} {:>5} {elapsed:>10.1}", 20, 100);
    }
}
