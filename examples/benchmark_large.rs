//! Large-scale benchmark for gradient boosting implementations.
//!
//! Generates synthetic datasets at various sizes (500 to 100K samples),
//! trains XGBoost, LightGBM, and CatBoost 10 times each, and reports
//! mean +/- std training time in milliseconds.
//!
//! Run with: RUSTFLAGS="-C target-cpu=native" cargo run --release --example benchmark_large
//!
//! Output: paper/replication/benchmark_rust_results.json

use ndarray::Array2;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use smelt_ml::prelude::*;
use std::time::Instant;

const N_RUNS: usize = 10;
const SIZES: &[usize] = &[500, 1_000, 5_000, 10_000, 50_000, 100_000];
const N_FEATURES: usize = 20;
const N_INFORMATIVE: usize = 10;
const N_TREES: usize = 100;
const MAX_DEPTH: usize = 6;
const SEED: u64 = 42;

/// Generate a synthetic classification dataset similar to sklearn's make_classification.
fn make_classification_data(n_samples: usize, seed: u64) -> (Array2<f64>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(seed);

    // Generate informative features from normal distribution
    let mut features = Array2::zeros((n_samples, N_FEATURES));
    for i in 0..n_samples {
        for j in 0..N_FEATURES {
            features[[i, j]] = sample_normal(&mut rng);
        }
    }

    // Target based on linear combination of informative features
    let mut weights = vec![0.0f64; N_INFORMATIVE];
    for w in weights.iter_mut() {
        *w = sample_normal(&mut rng);
    }

    let target: Vec<usize> = (0..n_samples)
        .map(|i| {
            let score: f64 = (0..N_INFORMATIVE)
                .map(|j| features[[i, j]] * weights[j])
                .sum();
            if score > 0.0 { 1 } else { 0 }
        })
        .collect();

    (features, target)
}

/// Generate a synthetic regression dataset similar to sklearn's make_regression.
fn make_regression_data(n_samples: usize, seed: u64) -> (Array2<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);

    let mut features = Array2::zeros((n_samples, N_FEATURES));
    for i in 0..n_samples {
        for j in 0..N_FEATURES {
            features[[i, j]] = sample_normal(&mut rng);
        }
    }

    let mut weights = vec![0.0f64; N_INFORMATIVE];
    for w in weights.iter_mut() {
        *w = sample_normal(&mut rng) * 10.0;
    }

    let target: Vec<f64> = (0..n_samples)
        .map(|i| {
            let score: f64 = (0..N_INFORMATIVE)
                .map(|j| features[[i, j]] * weights[j])
                .sum();
            score + sample_normal(&mut rng) * 0.1 // small noise
        })
        .collect();

    (features, target)
}

/// Box-Muller transform for normal distribution sampling.
fn sample_normal(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.random::<f64>().max(1e-15);
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn std_dev(v: &[f64]) -> f64 {
    let m = mean(v);
    let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64;
    var.sqrt()
}

#[derive(serde::Serialize)]
struct BenchResult {
    times_ms: Vec<f64>,
    mean_ms: f64,
    std_ms: f64,
}

#[derive(serde::Serialize)]
struct SizeResults {
    xgboost: BenchResult,
    lightgbm: BenchResult,
    catboost: BenchResult,
}

#[derive(serde::Serialize)]
struct AllResults {
    config: serde_json::Value,
    classification: std::collections::BTreeMap<String, SizeResults>,
    regression: std::collections::BTreeMap<String, SizeResults>,
}

fn bench_xgboost_classif(task: &ClassificationTask, n_runs: usize) -> BenchResult {
    let mut times = Vec::with_capacity(n_runs);
    for _ in 0..n_runs {
        let mut xgb = XGBoost::new()
            .with_n_estimators(N_TREES)
            .with_max_depth(MAX_DEPTH)
            .with_learning_rate(0.3);
        let t0 = Instant::now();
        let _ = xgb.train_classif(task).unwrap();
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let m = mean(&times);
    let s = std_dev(&times);
    BenchResult {
        times_ms: times.iter().map(|t| (*t * 100.0).round() / 100.0).collect(),
        mean_ms: (m * 100.0).round() / 100.0,
        std_ms: (s * 100.0).round() / 100.0,
    }
}

fn bench_xgboost_regress(task: &RegressionTask, n_runs: usize) -> BenchResult {
    let mut times = Vec::with_capacity(n_runs);
    for _ in 0..n_runs {
        let mut xgb = XGBoost::new()
            .with_n_estimators(N_TREES)
            .with_max_depth(MAX_DEPTH)
            .with_learning_rate(0.3);
        let t0 = Instant::now();
        let _ = xgb.train_regress(task).unwrap();
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let m = mean(&times);
    let s = std_dev(&times);
    BenchResult {
        times_ms: times.iter().map(|t| (*t * 100.0).round() / 100.0).collect(),
        mean_ms: (m * 100.0).round() / 100.0,
        std_ms: (s * 100.0).round() / 100.0,
    }
}

fn bench_lightgbm_classif(task: &ClassificationTask, n_runs: usize) -> BenchResult {
    let mut times = Vec::with_capacity(n_runs);
    for _ in 0..n_runs {
        let mut lgb = LightGBM::new()
            .with_n_estimators(N_TREES)
            .with_max_depth(MAX_DEPTH)
            .with_learning_rate(0.1);
        let t0 = Instant::now();
        let _ = lgb.train_classif(task).unwrap();
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let m = mean(&times);
    let s = std_dev(&times);
    BenchResult {
        times_ms: times.iter().map(|t| (*t * 100.0).round() / 100.0).collect(),
        mean_ms: (m * 100.0).round() / 100.0,
        std_ms: (s * 100.0).round() / 100.0,
    }
}

fn bench_lightgbm_regress(task: &RegressionTask, n_runs: usize) -> BenchResult {
    let mut times = Vec::with_capacity(n_runs);
    for _ in 0..n_runs {
        let mut lgb = LightGBM::new()
            .with_n_estimators(N_TREES)
            .with_max_depth(MAX_DEPTH)
            .with_learning_rate(0.1);
        let t0 = Instant::now();
        let _ = lgb.train_regress(task).unwrap();
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let m = mean(&times);
    let s = std_dev(&times);
    BenchResult {
        times_ms: times.iter().map(|t| (*t * 100.0).round() / 100.0).collect(),
        mean_ms: (m * 100.0).round() / 100.0,
        std_ms: (s * 100.0).round() / 100.0,
    }
}

fn bench_catboost_classif(task: &ClassificationTask, n_runs: usize) -> BenchResult {
    let mut times = Vec::with_capacity(n_runs);
    for _ in 0..n_runs {
        let mut cb = CatBoost::new()
            .with_n_estimators(N_TREES)
            .with_depth(MAX_DEPTH)
            .with_learning_rate(0.3);
        let t0 = Instant::now();
        let _ = cb.train_classif(task).unwrap();
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let m = mean(&times);
    let s = std_dev(&times);
    BenchResult {
        times_ms: times.iter().map(|t| (*t * 100.0).round() / 100.0).collect(),
        mean_ms: (m * 100.0).round() / 100.0,
        std_ms: (s * 100.0).round() / 100.0,
    }
}

fn bench_catboost_regress(task: &RegressionTask, n_runs: usize) -> BenchResult {
    let mut times = Vec::with_capacity(n_runs);
    for _ in 0..n_runs {
        let mut cb = CatBoost::new()
            .with_n_estimators(N_TREES)
            .with_depth(MAX_DEPTH)
            .with_learning_rate(0.3);
        let t0 = Instant::now();
        let _ = cb.train_regress(task).unwrap();
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let m = mean(&times);
    let s = std_dev(&times);
    BenchResult {
        times_ms: times.iter().map(|t| (*t * 100.0).round() / 100.0).collect(),
        mean_ms: (m * 100.0).round() / 100.0,
        std_ms: (s * 100.0).round() / 100.0,
    }
}

fn main() {
    println!("{}", "=".repeat(70));
    println!("Benchmark: smelt-ml Gradient Boosting");
    println!(
        "Configuration: {} trees, max_depth={}, {} runs each",
        N_TREES, MAX_DEPTH, N_RUNS
    );
    println!("Features: {} ({} informative)", N_FEATURES, N_INFORMATIVE);
    println!("{}", "=".repeat(70));

    let mut classification = std::collections::BTreeMap::new();
    let mut regression = std::collections::BTreeMap::new();

    for &n in SIZES {
        println!("\n{}", "-".repeat(50));
        println!("N = {}", n);
        println!("{}", "-".repeat(50));

        // Generate data
        let (x_c, y_c) = make_classification_data(n, SEED);
        let (x_r, y_r) = make_regression_data(n, SEED);

        let task_c = ClassificationTask::new("bench", x_c, y_c).unwrap();
        let task_r = RegressionTask::new("bench", x_r, y_r).unwrap();

        // Classification
        print!("  xgboost classif... ");
        let xgb_c = bench_xgboost_classif(&task_c, N_RUNS);
        println!("{:.1} +/- {:.1} ms", xgb_c.mean_ms, xgb_c.std_ms);

        print!("  lightgbm classif... ");
        let lgb_c = bench_lightgbm_classif(&task_c, N_RUNS);
        println!("{:.1} +/- {:.1} ms", lgb_c.mean_ms, lgb_c.std_ms);

        print!("  catboost classif... ");
        let cb_c = bench_catboost_classif(&task_c, N_RUNS);
        println!("{:.1} +/- {:.1} ms", cb_c.mean_ms, cb_c.std_ms);

        classification.insert(
            n.to_string(),
            SizeResults {
                xgboost: xgb_c,
                lightgbm: lgb_c,
                catboost: cb_c,
            },
        );

        // Regression
        print!("  xgboost regress... ");
        let xgb_r = bench_xgboost_regress(&task_r, N_RUNS);
        println!("{:.1} +/- {:.1} ms", xgb_r.mean_ms, xgb_r.std_ms);

        print!("  lightgbm regress... ");
        let lgb_r = bench_lightgbm_regress(&task_r, N_RUNS);
        println!("{:.1} +/- {:.1} ms", lgb_r.mean_ms, lgb_r.std_ms);

        print!("  catboost regress... ");
        let cb_r = bench_catboost_regress(&task_r, N_RUNS);
        println!("{:.1} +/- {:.1} ms", cb_r.mean_ms, cb_r.std_ms);

        regression.insert(
            n.to_string(),
            SizeResults {
                xgboost: xgb_r,
                lightgbm: lgb_r,
                catboost: cb_r,
            },
        );
    }

    let results = AllResults {
        config: serde_json::json!({
            "n_runs": N_RUNS,
            "n_trees": N_TREES,
            "max_depth": MAX_DEPTH,
            "n_features": N_FEATURES,
            "n_informative": N_INFORMATIVE,
            "seed": SEED,
        }),
        classification,
        regression,
    };

    let out_path = "paper/replication/benchmark_rust_results.json";
    let json = serde_json::to_string_pretty(&results).unwrap();
    std::fs::write(out_path, &json).unwrap();
    println!("\nResults saved to {}", out_path);
}
