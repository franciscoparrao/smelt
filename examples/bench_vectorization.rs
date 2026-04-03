//! Quick benchmark to measure target-cpu=native effect.
//! Run twice:
//!   RUSTFLAGS="" cargo run --release --example bench_vectorization
//!   RUSTFLAGS="-C target-cpu=native" cargo run --release --example bench_vectorization

use ndarray::Array2;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::Rng;
use smelt_ml::prelude::*;
use std::time::Instant;

fn sample_normal(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.random::<f64>().max(1e-15);
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn main() {
    let n = 10_000;
    let n_features = 20;
    let n_runs = 5;
    let mut rng = StdRng::seed_from_u64(42);

    let mut features = Array2::zeros((n, n_features));
    for i in 0..n {
        for j in 0..n_features {
            features[[i, j]] = sample_normal(&mut rng);
        }
    }
    let weights: Vec<f64> = (0..10).map(|_| sample_normal(&mut rng)).collect();
    let target: Vec<usize> = (0..n)
        .map(|i| {
            let s: f64 = (0..10).map(|j| features[[i, j]] * weights[j]).sum();
            if s > 0.0 { 1 } else { 0 }
        })
        .collect();

    let task = ClassificationTask::new("bench", features, target).unwrap();

    // XGBoost
    let mut times = Vec::new();
    for _ in 0..n_runs {
        let mut xgb = XGBoost::new().with_n_estimators(100).with_max_depth(6).with_learning_rate(0.3);
        let t0 = Instant::now();
        let _ = xgb.train_classif(&task).unwrap();
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let mean: f64 = times.iter().sum::<f64>() / times.len() as f64;
    println!("XGBoost  N=10000: {:.1} ms (5-run mean)", mean);

    // CatBoost
    let mut times = Vec::new();
    for _ in 0..n_runs {
        let mut cb = CatBoost::new().with_n_estimators(100).with_depth(6).with_learning_rate(0.3);
        let t0 = Instant::now();
        let _ = cb.train_classif(&task).unwrap();
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let mean: f64 = times.iter().sum::<f64>() / times.len() as f64;
    println!("CatBoost N=10000: {:.1} ms (5-run mean)", mean);
}
