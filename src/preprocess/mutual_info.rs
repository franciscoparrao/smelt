//! Shared mutual information utilities for info-theoretic feature selection.
//!
//! Provides conditional MI, joint entropy, and helper functions used by
//! MRMR, JMI, JMIM, and CMIM filters.

const N_BINS: usize = 10;

/// Discretize a continuous slice into `N_BINS` uniform-width bins.
pub fn discretize(values: &[f64]) -> Vec<usize> {
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = (max - min).max(1e-10);
    values
        .iter()
        .map(|&v| (((v - min) / range * N_BINS as f64) as usize).min(N_BINS - 1))
        .collect()
}

/// Standard mutual information I(X; Y) from pre-discretized bins.
pub fn mi_from_bins(x_bins: &[usize], y_bins: &[usize]) -> f64 {
    let n = x_bins.len() as f64;
    let mut joint = [[0u32; N_BINS]; N_BINS];
    let mut x_counts = [0u32; N_BINS];
    let mut y_counts = [0u32; N_BINS];

    for (&xb, &yb) in x_bins.iter().zip(y_bins) {
        joint[xb][yb] += 1;
        x_counts[xb] += 1;
        y_counts[yb] += 1;
    }

    let mut mi = 0.0;
    for xb in 0..N_BINS {
        if x_counts[xb] == 0 {
            continue;
        }
        for yb in 0..N_BINS {
            if joint[xb][yb] == 0 {
                continue;
            }
            let p_xy = joint[xb][yb] as f64 / n;
            let p_x = x_counts[xb] as f64 / n;
            let p_y = y_counts[yb] as f64 / n;
            mi += p_xy * (p_xy / (p_x * p_y)).ln();
        }
    }
    mi.max(0.0)
}

/// Conditional mutual information I(X; Y | Z) from pre-discretized bins.
///
/// Uses the identity: I(X; Y | Z) = H(X, Z) + H(Y, Z) - H(X, Y, Z) - H(Z)
pub fn conditional_mi(x_bins: &[usize], y_bins: &[usize], z_bins: &[usize]) -> f64 {
    let h_xz = joint_entropy(x_bins, z_bins);
    let h_yz = joint_entropy(y_bins, z_bins);
    let h_xyz = triple_entropy(x_bins, y_bins, z_bins);
    let h_z = marginal_entropy(z_bins);
    (h_xz + h_yz - h_xyz - h_z).max(0.0)
}

/// Joint mutual information I(X, Z; Y) = I(Z; Y) + I(X; Y | Z).
pub fn joint_mi(x_bins: &[usize], z_bins: &[usize], y_bins: &[usize]) -> f64 {
    mi_from_bins(z_bins, y_bins) + conditional_mi(x_bins, y_bins, z_bins)
}

/// Shannon entropy H(X) of a single discretized variable.
fn marginal_entropy(bins: &[usize]) -> f64 {
    let n = bins.len() as f64;
    let mut counts = [0u32; N_BINS];
    for &b in bins {
        counts[b] += 1;
    }
    counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / n;
            -p * p.ln()
        })
        .sum()
}

/// Joint entropy H(X, Y) of two discretized variables.
fn joint_entropy(x_bins: &[usize], y_bins: &[usize]) -> f64 {
    let n = x_bins.len() as f64;
    let mut joint = [[0u32; N_BINS]; N_BINS];
    for (&xb, &yb) in x_bins.iter().zip(y_bins) {
        joint[xb][yb] += 1;
    }
    let mut h = 0.0;
    for row in &joint {
        for &c in row {
            if c > 0 {
                let p = c as f64 / n;
                h -= p * p.ln();
            }
        }
    }
    h
}

/// Triple joint entropy H(X, Y, Z) of three discretized variables.
fn triple_entropy(x_bins: &[usize], y_bins: &[usize], z_bins: &[usize]) -> f64 {
    let n = x_bins.len() as f64;
    let mut counts = vec![0u32; N_BINS * N_BINS * N_BINS];
    for i in 0..x_bins.len() {
        let idx = x_bins[i] * N_BINS * N_BINS + y_bins[i] * N_BINS + z_bins[i];
        counts[idx] += 1;
    }
    let mut h = 0.0;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / n;
            h -= p * p.ln();
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mi_identical_vars() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let bins = discretize(&x);
        let mi = mi_from_bins(&bins, &bins);
        // MI of a variable with itself = entropy
        assert!(mi > 0.0, "MI with itself should be positive");
    }

    #[test]
    fn mi_independent_vars() {
        // Two unrelated patterns
        let x: Vec<f64> = (0..100).map(|i| (i % 10) as f64).collect();
        let y: Vec<f64> = (0..100).map(|i| (i / 10) as f64).collect();
        let xb = discretize(&x);
        let yb = discretize(&y);
        let mi = mi_from_bins(&xb, &yb);
        assert!(mi < 0.1, "MI of independent vars should be near 0, got {mi}");
    }

    #[test]
    fn conditional_mi_non_negative() {
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..50).map(|i| (i * 2) as f64).collect();
        let z: Vec<f64> = (0..50).map(|i| (i * 3) as f64).collect();
        let xb = discretize(&x);
        let yb = discretize(&y);
        let zb = discretize(&z);
        let cmi = conditional_mi(&xb, &yb, &zb);
        assert!(cmi >= 0.0, "Conditional MI should be non-negative");
    }
}
