//! Prediction (inference) time benchmark.
//!
//! Measures time to predict on 1000 new samples after training on N=10K.
//! Run with: cargo run --release --example benchmark_prediction

use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use smelt_ml::prelude::*;
use std::time::Instant;

fn sample_normal(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.random::<f64>().max(1e-15);
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn main() {
    let n_train = 10_000;
    let n_test = 1_000;
    let n_runs = 100;
    let nf = 20;
    let mut rng = StdRng::seed_from_u64(42);

    // Generate data
    let mut train_feat = Array2::zeros((n_train, nf));
    for i in 0..n_train {
        for j in 0..nf {
            train_feat[[i, j]] = sample_normal(&mut rng);
        }
    }
    let weights: Vec<f64> = (0..10).map(|_| sample_normal(&mut rng)).collect();
    let train_target: Vec<usize> = (0..n_train)
        .map(|i| {
            let s: f64 = (0..10).map(|j| train_feat[[i, j]] * weights[j]).sum();
            if s > 0.0 { 1 } else { 0 }
        })
        .collect();

    let mut test_feat = Array2::zeros((n_test, nf));
    for i in 0..n_test {
        for j in 0..nf {
            test_feat[[i, j]] = sample_normal(&mut rng);
        }
    }

    let task = ClassificationTask::new("bench", train_feat, train_target).unwrap();

    println!("============================================================");
    println!("  Prediction Time Benchmark (train N={}, predict N={})", n_train, n_test);
    println!("  {} runs each, classification", n_runs);
    println!("============================================================\n");

    // XGBoost
    let mut xgb = XGBoost::new()
        .with_n_estimators(100)
        .with_max_depth(6)
        .with_learning_rate(0.3);
    let xgb_model = xgb.train_classif(&task).unwrap();
    let mut times = Vec::with_capacity(n_runs);
    for _ in 0..n_runs {
        let t0 = Instant::now();
        let _ = xgb_model.predict(&test_feat).unwrap();
        times.push(t0.elapsed().as_secs_f64() * 1_000_000.0); // microseconds
    }
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let std = (times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64).sqrt();
    println!("  XGBoost      {:>8.0} +/- {:>5.0} us", mean, std);

    // CatBoost
    let mut cb = CatBoost::new()
        .with_n_estimators(100)
        .with_depth(6)
        .with_learning_rate(0.3);
    let cb_model = cb.train_classif(&task).unwrap();
    let mut times = Vec::with_capacity(n_runs);
    for _ in 0..n_runs {
        let t0 = Instant::now();
        let _ = cb_model.predict(&test_feat).unwrap();
        times.push(t0.elapsed().as_secs_f64() * 1_000_000.0);
    }
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let std = (times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64).sqrt();
    println!("  CatBoost     {:>8.0} +/- {:>5.0} us", mean, std);

    // Random Forest
    let mut rf = RandomForest::new().with_n_estimators(100).with_seed(42);
    let rf_model = rf.train_classif(&task).unwrap();
    let mut times = Vec::with_capacity(n_runs);
    for _ in 0..n_runs {
        let t0 = Instant::now();
        let _ = rf_model.predict(&test_feat).unwrap();
        times.push(t0.elapsed().as_secs_f64() * 1_000_000.0);
    }
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let std = (times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64).sqrt();
    println!("  Random Forest {:>7.0} +/- {:>5.0} us", mean, std);

    // Decision Tree
    let mut dt = DecisionTree::default();
    let dt_model = dt.train_classif(&task).unwrap();
    let mut times = Vec::with_capacity(n_runs);
    for _ in 0..n_runs {
        let t0 = Instant::now();
        let _ = dt_model.predict(&test_feat).unwrap();
        times.push(t0.elapsed().as_secs_f64() * 1_000_000.0);
    }
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let std = (times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64).sqrt();
    println!("  Decision Tree {:>7.0} +/- {:>5.0} us", mean, std);

    println!("\n  (times in microseconds for {} samples)", n_test);
}
