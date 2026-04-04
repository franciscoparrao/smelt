use std::fs;

fn get_peak_rss_kb() -> usize {
    let status = fs::read_to_string("/proc/self/status").unwrap();
    for line in status.lines() {
        if line.starts_with("VmHWM:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            return parts[1].parse().unwrap_or(0);
        }
    }
    0
}

fn main() {
    use ndarray::Array2;
    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use smelt_ml::prelude::*;

    let n = 10_000;
    let mut rng = StdRng::seed_from_u64(42);
    let mut features = Array2::zeros((n, 20));
    for i in 0..n {
        for j in 0..20 {
            let u1: f64 = rng.random::<f64>().max(1e-15);
            let u2: f64 = rng.random::<f64>();
            features[[i, j]] = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        }
    }
    let weights: Vec<f64> = (0..10)
        .map(|_| {
            let u1: f64 = rng.random::<f64>().max(1e-15);
            let u2: f64 = rng.random::<f64>();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        })
        .collect();
    let target: Vec<usize> = (0..n)
        .map(|i| {
            let s: f64 = (0..10).map(|j| features[[i, j]] * weights[j]).sum();
            if s > 0.0 { 1 } else { 0 }
        })
        .collect();

    let task = ClassificationTask::new("bench", features, target).unwrap();

    let mut xgb = XGBoost::new().with_n_estimators(100).with_max_depth(6);
    let _ = xgb.train_classif(&task).unwrap();
    println!(
        "Peak RSS after XGBoost Rust (N=10K): {} MB",
        get_peak_rss_kb() / 1024
    );

    let mut cb = CatBoost::new().with_n_estimators(100).with_depth(6);
    let _ = cb.train_classif(&task).unwrap();
    println!(
        "Peak RSS after CatBoost Rust (N=10K): {} MB",
        get_peak_rss_kb() / 1024
    );
}
