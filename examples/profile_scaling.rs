//! Profiling target for understanding scaling behavior.
//! Run with: cargo flamegraph --example profile_scaling -- 100000
//! Or for perf stat: perf stat cargo run --release --example profile_scaling -- 10000

use ndarray::Array2;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use smelt_ml::prelude::*;

fn sample_normal(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.random::<f64>().max(1e-15);
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let mut rng = StdRng::seed_from_u64(42);
    let mut features = Array2::zeros((n, 20));
    for i in 0..n {
        for j in 0..20 {
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

    let task = ClassificationTask::new("prof", features, target).unwrap();
    let mut xgb = XGBoost::new()
        .with_n_estimators(100)
        .with_max_depth(6)
        .with_learning_rate(0.3);

    eprintln!("Training XGBoost with N={}", n);
    let t0 = std::time::Instant::now();
    let _ = xgb.train_classif(&task).unwrap();
    eprintln!("Done in {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
}
