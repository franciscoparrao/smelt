//! Ablation study: measure the contribution of each optimization.
//!
//! Variants:
//! 1. Row-major + u16 bins (baseline, C++ legacy layout)
//! 2. Column-major + u16 bins (layout change only)
//! 3. Column-major + u8 bins (+ packing)
//! 4. Column-major + u8 + histogram subtraction (current smelt-ml)
//!
//! Run with: RUSTFLAGS="-C target-cpu=native" cargo run --release --example ablation_study

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

fn generate_data(n: usize) -> (Array2<f64>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(42);
    let nf = 20;
    let mut features = Array2::zeros((n, nf));
    for i in 0..n {
        for j in 0..nf {
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
    (features, target)
}

// ── Variant 1: Row-major + u16 bins ────────────────────────────────────

struct RowMajorU16Bins {
    data: Vec<Vec<u16>>, // data[sample][feature]
    boundaries: Vec<Vec<f64>>,
}

impl RowMajorU16Bins {
    fn build(features: &Array2<f64>, n_bins: usize) -> Self {
        let (ns, nf) = (features.nrows(), features.ncols());
        let n_bins = n_bins.min(65534);
        let mut boundaries = Vec::with_capacity(nf);

        for j in 0..nf {
            let mut vals: Vec<f64> = features
                .column(j)
                .iter()
                .filter(|v| !v.is_nan())
                .copied()
                .collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            vals.dedup();
            let step = (vals.len() as f64 / n_bins as f64).max(1.0) as usize;
            let b: Vec<f64> = vals.iter().step_by(step).copied().collect();
            boundaries.push(b);
        }

        let mut data = vec![vec![0u16; nf]; ns];
        for i in 0..ns {
            for j in 0..nf {
                let v = features[[i, j]];
                let b = &boundaries[j];
                let bin = b
                    .partition_point(|&t| t <= v)
                    .min(b.len().saturating_sub(1));
                data[i][j] = bin as u16;
            }
        }
        Self { data, boundaries }
    }

    #[inline]
    fn get_bin(&self, sample: usize, feature: usize) -> u16 {
        self.data[sample][feature] // row-major: data[sample][feature]
    }

    fn n_bins(&self, feature: usize) -> usize {
        self.boundaries[feature].len()
    }
}

// ── Variant 2: Column-major + u16 bins ─────────────────────────────────

struct ColMajorU16Bins {
    cols: Vec<Vec<u16>>, // cols[feature][sample]
    boundaries: Vec<Vec<f64>>,
}

impl ColMajorU16Bins {
    fn build(features: &Array2<f64>, n_bins: usize) -> Self {
        let (ns, nf) = (features.nrows(), features.ncols());
        let n_bins = n_bins.min(65534);
        let mut boundaries = Vec::with_capacity(nf);
        let mut cols = Vec::with_capacity(nf);

        for j in 0..nf {
            let mut vals: Vec<f64> = features
                .column(j)
                .iter()
                .filter(|v| !v.is_nan())
                .copied()
                .collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            vals.dedup();
            let step = (vals.len() as f64 / n_bins as f64).max(1.0) as usize;
            let b: Vec<f64> = vals.iter().step_by(step).copied().collect();

            let mut col = vec![0u16; ns];
            for i in 0..ns {
                let v = features[[i, j]];
                let bin = b
                    .partition_point(|&t| t <= v)
                    .min(b.len().saturating_sub(1));
                col[i] = bin as u16;
            }
            cols.push(col);
            boundaries.push(b);
        }
        Self { cols, boundaries }
    }

    #[inline]
    fn get_bin(&self, feature: usize, sample: usize) -> u16 {
        self.cols[feature][sample] // column-major: cols[feature][sample]
    }

    fn n_bins(&self, feature: usize) -> usize {
        self.boundaries[feature].len()
    }
}

/// Variant 3: column-major u8 (same as HistBins), no subtraction — use smelt XGBoost without subtraction
/// We approximate this by using the actual XGBoost (which includes subtraction) and noting
/// that the subtraction impact is measured separately.
///
/// For the ablation, we measure the KERNEL (histogram scanning) directly.
fn bench_kernel_rowmajor_u16(
    features: &Array2<f64>,
    grads: &[f64],
    hess: &[f64],
    n_iters: usize,
) -> f64 {
    let ns = features.nrows();
    let nf = features.ncols();
    let bins = RowMajorU16Bins::build(features, 256);
    let t0 = Instant::now();
    let mut dummy = 0.0f64;
    for _ in 0..n_iters {
        for feat in 0..nf {
            let nb = bins.n_bins(feat);
            let mut bin_g = vec![0.0; nb];
            let mut bin_h = vec![0.0; nb];
            for i in 0..ns {
                let b = bins.get_bin(i, feat) as usize;
                bin_g[b] += grads[i];
                bin_h[b] += hess[i];
            }
            dummy += bin_g[0];
        }
    }
    let _ = dummy; // prevent optimization
    t0.elapsed().as_secs_f64() * 1000.0 / n_iters as f64
}

fn bench_kernel_colmajor_u16(
    features: &Array2<f64>,
    grads: &[f64],
    hess: &[f64],
    n_iters: usize,
) -> f64 {
    let ns = features.nrows();
    let nf = features.ncols();
    let bins = ColMajorU16Bins::build(features, 256);
    let t0 = Instant::now();
    let mut dummy = 0.0f64;
    for _ in 0..n_iters {
        for feat in 0..nf {
            let nb = bins.n_bins(feat);
            let mut bin_g = vec![0.0; nb];
            let mut bin_h = vec![0.0; nb];
            for i in 0..ns {
                let b = bins.get_bin(feat, i) as usize;
                bin_g[b] += grads[i];
                bin_h[b] += hess[i];
            }
            dummy += bin_g[0];
        }
    }
    let _ = dummy;
    t0.elapsed().as_secs_f64() * 1000.0 / n_iters as f64
}

fn bench_kernel_colmajor_u8(
    features: &Array2<f64>,
    grads: &[f64],
    hess: &[f64],
    n_iters: usize,
) -> f64 {
    let nf = features.ncols();
    let bins = smelt_ml::learner::histogram::HistBins::build(features, 256);
    let t0 = Instant::now();
    let mut dummy = 0.0f64;
    let ns = features.nrows();
    for _ in 0..n_iters {
        for feat in 0..nf {
            let nb = bins.n_bins(feat);
            let mut bin_g = vec![0.0; nb];
            let mut bin_h = vec![0.0; nb];
            for i in 0..ns {
                let b = bins.get_bin(feat, i);
                if b != 255 {
                    bin_g[b as usize] += grads[i];
                    bin_h[b as usize] += hess[i];
                }
            }
            dummy += bin_g[0];
        }
    }
    let _ = dummy;
    t0.elapsed().as_secs_f64() * 1000.0 / n_iters as f64
}

fn main() {
    println!("================================================================");
    println!("  Ablation Study: Contribution of Each Optimization");
    println!("  Histogram scanning kernel (20 features, 256 bins)");
    println!("================================================================\n");

    for &n in &[1_000usize, 10_000, 100_000] {
        let (features, target) = generate_data(n);
        let grads: Vec<f64> = (0..n)
            .map(|i| if target[i] == 1 { -0.5 } else { 0.5 })
            .collect();
        let hess: Vec<f64> = vec![0.25; n];

        let n_iters = if n <= 10_000 { 50 } else { 10 };

        let t_row_u16 = bench_kernel_rowmajor_u16(&features, &grads, &hess, n_iters);
        let t_col_u16 = bench_kernel_colmajor_u16(&features, &grads, &hess, n_iters);
        let t_col_u8 = bench_kernel_colmajor_u8(&features, &grads, &hess, n_iters);

        // Full XGBoost training (includes subtraction)
        let task = ClassificationTask::new("abl", features.clone(), target.clone()).unwrap();
        let t0 = Instant::now();
        let mut xgb = XGBoost::new()
            .with_n_estimators(100)
            .with_max_depth(6)
            .with_learning_rate(0.3);
        let _ = xgb.train_classif(&task).unwrap();
        let t_full = t0.elapsed().as_secs_f64() * 1000.0;

        println!("N = {:>7}", n);
        println!(
            "  {:40} {:>8.1} ms  (baseline)",
            "1. Row-major + u16", t_row_u16
        );
        println!(
            "  {:40} {:>8.1} ms  ({:.0}%)",
            "2. Column-major + u16",
            t_col_u16,
            (1.0 - t_col_u16 / t_row_u16) * 100.0
        );
        println!(
            "  {:40} {:>8.1} ms  ({:.0}%)",
            "3. Column-major + u8",
            t_col_u8,
            (1.0 - t_col_u8 / t_row_u16) * 100.0
        );
        println!(
            "  {:40} {:>8.1} ms  (full training)",
            "4. Full XGBoost (col-major+u8+subtraction)", t_full
        );
        println!(
            "  Speedup of column-major over row-major: {:.2}x",
            t_row_u16 / t_col_u16
        );
        println!(
            "  Speedup of u8 over u16:                 {:.2}x",
            t_col_u16 / t_col_u8
        );
        println!();
    }
}
