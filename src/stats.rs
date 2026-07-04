//! Statistical tests for model comparison.
//!
//! Non-parametric tests commonly used to establish statistically significant
//! differences between ML models, integrated directly into the framework.
//!
//! # Tests available
//!
//! - [`wilcoxon_signed_rank`]: Compare two models across paired samples (CV folds)
//! - [`friedman_test`]: Compare 3+ models across paired samples
//! - [`nemenyi_posthoc`]: Pairwise comparisons after a significant Friedman test
//! - [`mcnemar_test`]: Compare two classifiers on the same test set
//! - [`sign_test`]: Simplest paired comparison (direction only, ignores magnitude)
//! - [`bootstrap_ci`]: Bootstrap confidence interval for any metric
//!
//! # Example
//!
//! ```
//! use smelt_ml::stats::{wilcoxon_signed_rank, friedman_test, bootstrap_ci};
//!
//! // Compare XGBoost vs Random Forest across 6-fold CV
//! let xgb_scores  = vec![0.92, 0.89, 0.91, 0.90, 0.93, 0.91];
//! let rf_scores   = vec![0.88, 0.87, 0.89, 0.86, 0.90, 0.88];
//!
//! let w = wilcoxon_signed_rank(&xgb_scores, &rf_scores).unwrap();
//! assert!(w.p_value < 0.05); // XGBoost significantly better
//!
//! // Bootstrap 95% CI for XGBoost accuracy
//! let ci = bootstrap_ci(&xgb_scores, 0.95, 10000, 42).unwrap();
//! assert!(ci.lower > 0.85);
//! ```
//!
//! Note on `wilcoxon_signed_rank`'s minimum reachable p-value: with `n` paired
//! samples and no ties in sign, the smallest possible two-sided p-value is
//! `2 / 2^n` (achieved only when every pair favors the same model). With 5
//! folds that floor is `2/32 = 0.0625` -- **already above the conventional
//! 0.05 threshold**, so 5-fold CV can never show "significance" by this test
//! no matter how consistent the folds are. This crate computes the exact
//! p-value (not an approximation) for realistic fold/model counts, so this
//! floor is enforced automatically rather than silently understated.

use crate::{Result, SmeltError};

// ── Result types ───────────────────────────────────────────────────

/// Result of a two-sample statistical test.
#[derive(Debug, Clone)]
pub struct TestResult {
    /// Name of the test.
    pub test: &'static str,
    /// Test statistic value.
    pub statistic: f64,
    /// p-value (probability of observing this result under H0).
    pub p_value: f64,
    /// Whether the null hypothesis is rejected at alpha = 0.05.
    pub significant: bool,
}

/// Result of a Friedman test (k models × n datasets/folds).
#[derive(Debug, Clone)]
pub struct FriedmanResult {
    /// Chi-squared statistic.
    pub statistic: f64,
    /// p-value.
    pub p_value: f64,
    /// Whether the null hypothesis is rejected at alpha = 0.05.
    pub significant: bool,
    /// Average ranks per model (lower = better).
    pub avg_ranks: Vec<f64>,
}

/// Bootstrap confidence interval.
#[derive(Debug, Clone)]
pub struct BootstrapCI {
    /// Point estimate (sample mean).
    pub estimate: f64,
    /// Lower bound of the CI.
    pub lower: f64,
    /// Upper bound of the CI.
    pub upper: f64,
    /// Confidence level (e.g., 0.95).
    pub confidence: f64,
}

/// Result of Nemenyi post-hoc pairwise comparisons.
#[derive(Debug, Clone)]
pub struct NemenyiResult {
    /// Critical difference at alpha = 0.05.
    pub critical_difference: f64,
    /// Pairwise comparisons: (model_i, model_j, rank_diff, significant).
    pub comparisons: Vec<(usize, usize, f64, bool)>,
}

// ── Wilcoxon signed-rank test ──────────────────────────────────────

/// Above this effective sample size, `wilcoxon_signed_rank` falls back from
/// the exact permutation distribution to a continuity- and tie-corrected
/// normal approximation. The exact DP is `O(n * n^2)`; 100 keeps it well
/// under a millisecond while covering every realistic CV/benchmark use case
/// (fold counts and model-comparison sample sizes rarely exceed a few dozen).
const WILCOXON_EXACT_MAX_N: usize = 100;

/// Wilcoxon signed-rank test for paired samples.
///
/// Tests H0: the median difference between pairs is zero.
/// This is the standard test for comparing two ML models across CV folds.
///
/// Computes the **exact** two-sided p-value (via dynamic programming over
/// the `2^n` sign assignments of the ranks, not `2^n` brute enumeration) for
/// `n <= `[`WILCOXON_EXACT_MAX_N`]; beyond that, falls back to a normal
/// approximation with a continuity correction and a tie-variance correction
/// (`Σ(t³-t)/48` over tied-rank groups).
///
/// # Arguments
/// * `a` - Scores from model A (e.g., accuracy per fold)
/// * `b` - Scores from model B
///
/// # Returns
/// `TestResult` with the W statistic and two-sided p-value.
pub fn wilcoxon_signed_rank(a: &[f64], b: &[f64]) -> Result<TestResult> {
    if a.len() != b.len() {
        return Err(SmeltError::DimensionMismatch { expected: a.len(), got: b.len() });
    }

    // Compute differences, exclude zeros
    let mut diffs: Vec<(f64, f64)> = a
        .iter()
        .zip(b)
        .map(|(&ai, &bi)| {
            let d = ai - bi;
            (d.abs(), d)
        })
        .filter(|(abs_d, _)| *abs_d > 1e-15)
        .collect();

    let nr = diffs.len(); // effective n (excluding ties at zero)

    if nr == 0 {
        return Ok(TestResult {
            test: "Wilcoxon signed-rank",
            statistic: 0.0,
            p_value: 1.0,
            significant: false,
        });
    }

    // Rank by absolute difference
    diffs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Assign ranks with tie handling (average ranks), tracking each tie
    // group's size for the normal-approximation tie-variance correction.
    let mut ranks = vec![0.0; nr];
    let mut tie_group_sizes: Vec<usize> = Vec::new();
    let mut i = 0;
    while i < nr {
        let mut j = i;
        while j < nr && (diffs[j].0 - diffs[i].0).abs() < 1e-15 {
            j += 1;
        }
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        for k in i..j {
            ranks[k] = avg_rank;
        }
        tie_group_sizes.push(j - i);
        i = j;
    }

    // W+ = sum of ranks where difference is positive
    let w_plus: f64 = ranks
        .iter()
        .zip(&diffs)
        .filter(|(_, (_, d))| *d > 0.0)
        .map(|(r, _)| r)
        .sum();

    let w_minus: f64 = ranks
        .iter()
        .zip(&diffs)
        .filter(|(_, (_, d))| *d < 0.0)
        .map(|(r, _)| r)
        .sum();

    let w = w_plus.min(w_minus);

    let p_value = if nr <= WILCOXON_EXACT_MAX_N {
        wilcoxon_exact_two_sided_p(&ranks, w_plus)
    } else {
        let n_f = nr as f64;
        let mean_w = n_f * (n_f + 1.0) / 4.0;
        let tie_correction: f64 = tie_group_sizes
            .iter()
            .map(|&t| {
                let t = t as f64;
                t * t * t - t
            })
            .sum::<f64>()
            / 48.0;
        let var_w = (n_f * (n_f + 1.0) * (2.0 * n_f + 1.0) / 24.0 - tie_correction).max(0.0);
        let z = if var_w > 0.0 {
            (((w_plus - mean_w).abs() - 0.5).max(0.0)) / var_w.sqrt()
        } else {
            0.0
        };
        2.0 * standard_normal_cdf(-z)
    };

    Ok(TestResult {
        test: "Wilcoxon signed-rank",
        statistic: w,
        p_value,
        significant: p_value < 0.05,
    })
}

/// Exact two-sided p-value for the Wilcoxon signed-rank statistic: under H0,
/// each of the `nr` ranks independently gets a `+` or `-` sign with
/// probability 0.5, giving `2^nr` equally likely sign assignments. Rather
/// than enumerating all `2^nr` assignments directly, counts how many
/// assignments produce each achievable positive-rank sum via a subset-sum
/// dynamic program (`O(nr * max_sum)`): ranks are doubled first so
/// tie-averaged half-integer ranks (e.g. `1.5`) become integers, then
/// `dp[s]` accumulates the number of subsets of the doubled ranks summing to
/// `s`. Counts are kept as `f64` (not integer) so `nr` well beyond 64 doesn't
/// overflow -- `2^nr` loses exact integer precision for large `nr`, but the
/// resulting *ratio* `n_le / total` stays accurate to double-precision
/// relative error, which is all a p-value needs.
fn wilcoxon_exact_two_sided_p(ranks: &[f64], w_plus: f64) -> f64 {
    let doubled: Vec<i64> = ranks.iter().map(|&r| (r * 2.0).round() as i64).collect();
    let max_sum: i64 = doubled.iter().sum();
    let mut dp = vec![0.0f64; (max_sum + 1) as usize];
    dp[0] = 1.0;
    for &r in &doubled {
        for s in (r..=max_sum).rev() {
            dp[s as usize] += dp[(s - r) as usize];
        }
    }

    let total: f64 = 2f64.powi(doubled.len() as i32);
    let target = (w_plus * 2.0).round() as i64;
    let mut n_le = 0.0f64;
    let mut n_ge = 0.0f64;
    for (s, &cnt) in dp.iter().enumerate() {
        if cnt == 0.0 {
            continue;
        }
        let s = s as i64;
        if s <= target {
            n_le += cnt;
        }
        if s >= target {
            n_ge += cnt;
        }
    }

    let p_one_tail = n_le.min(n_ge) / total;
    (2.0 * p_one_tail).min(1.0)
}

// ── Sign test ──────────────────────────────────────────────────────

/// Sign test for paired samples.
///
/// Tests H0: P(A > B) = P(B > A) = 0.5.
/// Simpler than Wilcoxon (ignores magnitude), but valid for very small n.
pub fn sign_test(a: &[f64], b: &[f64]) -> Result<TestResult> {
    if a.len() != b.len() {
        return Err(SmeltError::DimensionMismatch { expected: a.len(), got: b.len() });
    }

    let mut n_plus = 0usize;
    let mut n_minus = 0usize;
    for (&ai, &bi) in a.iter().zip(b) {
        if ai > bi + 1e-15 {
            n_plus += 1;
        } else if bi > ai + 1e-15 {
            n_minus += 1;
        }
    }

    let n = n_plus + n_minus;
    if n == 0 {
        return Ok(TestResult {
            test: "Sign test",
            statistic: 0.0,
            p_value: 1.0,
            significant: false,
        });
    }

    let k = n_plus.min(n_minus);
    // Two-sided p-value from binomial(n, 0.5)
    let p_value = 2.0 * binomial_cdf(k, n, 0.5);
    let p_value = p_value.min(1.0);

    Ok(TestResult {
        test: "Sign test",
        statistic: k as f64,
        p_value,
        significant: p_value < 0.05,
    })
}

// ── Friedman test ──────────────────────────────────────────────────

/// Friedman test for comparing k ≥ 3 models across n paired samples.
///
/// Tests H0: all models perform equally. If rejected, use [`nemenyi_posthoc`]
/// for pairwise comparisons.
///
/// # Arguments
/// * `scores` - Matrix of scores: `scores[model][fold]`, all same length.
///
/// # Example
/// ```
/// use smelt_ml::stats::friedman_test;
///
/// let dt  = vec![0.80, 0.82, 0.79, 0.81, 0.83];
/// let rf  = vec![0.90, 0.88, 0.91, 0.89, 0.92];
/// let xgb = vec![0.92, 0.89, 0.93, 0.91, 0.94];
/// let result = friedman_test(&[&dt, &rf, &xgb]).unwrap();
/// assert!(result.significant);
/// ```
pub fn friedman_test(scores: &[&[f64]]) -> Result<FriedmanResult> {
    let k = scores.len(); // number of models
    if k < 3 {
        return Err(SmeltError::InvalidParameter(format!(
            "friedman_test requires at least 3 models, got {k}"
        )));
    }
    let n = scores[0].len(); // number of folds/datasets
    if !scores.iter().all(|s| s.len() == n) {
        return Err(SmeltError::InvalidParameter(
            "friedman_test: all models must have the same number of scores".into(),
        ));
    }

    // Rank within each fold (1 = best, k = worst)
    // For "higher is better" metrics, invert: rank 1 = highest score
    let mut rank_sums = vec![0.0; k];

    for j in 0..n {
        let mut indexed: Vec<(usize, f64)> = (0..k).map(|i| (i, scores[i][j])).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Assign ranks with tie handling
        let mut i = 0;
        while i < k {
            let mut end = i;
            while end < k && (indexed[end].1 - indexed[i].1).abs() < 1e-15 {
                end += 1;
            }
            let avg_rank = (i + 1 + end) as f64 / 2.0;
            for idx in i..end {
                rank_sums[indexed[idx].0] += avg_rank;
            }
            i = end;
        }
    }

    let avg_ranks: Vec<f64> = rank_sums.iter().map(|&r| r / n as f64).collect();

    // Friedman chi-squared statistic
    let k_f = k as f64;
    let n_f = n as f64;
    let sum_r2: f64 = rank_sums.iter().map(|r| r * r).sum();

    let chi2 = 12.0 / (n_f * k_f * (k_f + 1.0)) * sum_r2 - 3.0 * n_f * (k_f + 1.0);

    // p-value from chi-squared distribution with k-1 degrees of freedom
    let df = k_f - 1.0;
    let p_value = 1.0 - chi_squared_cdf(chi2, df);

    Ok(FriedmanResult {
        statistic: chi2,
        p_value,
        significant: p_value < 0.05,
        avg_ranks,
    })
}

// ── Nemenyi post-hoc test ──────────────────────────────────────────

/// Nemenyi post-hoc test after a significant Friedman test.
///
/// Computes the critical difference (CD) and identifies which pairs of
/// models differ significantly.
///
/// # Arguments
/// * `friedman` - Result from [`friedman_test`]
/// * `n` - Number of folds/datasets
/// * `k` - Number of models
pub fn nemenyi_posthoc(friedman: &FriedmanResult, n: usize, k: usize) -> NemenyiResult {
    // Critical value q_alpha for Nemenyi at alpha=0.05
    // Approximation: q_alpha ≈ z_alpha / sqrt(2) * sqrt(k(k+1)/(6n))
    // where z_alpha for pairwise at alpha=0.05 with Bonferroni: z = 2.576 (approx)
    // More standard: use studentized range distribution
    // For simplicity, use tabulated critical values for common k
    let q_alpha = match k {
        3 => 2.343,
        4 => 2.569,
        5 => 2.728,
        6 => 2.850,
        7 => 2.949,
        8 => 3.031,
        9 => 3.102,
        10 => 3.164,
        _ => 2.576, // Bonferroni approximation
    };

    let cd = q_alpha * (k as f64 * (k as f64 + 1.0) / (6.0 * n as f64)).sqrt();

    let mut comparisons = Vec::new();
    for i in 0..k {
        for j in (i + 1)..k {
            let diff = (friedman.avg_ranks[i] - friedman.avg_ranks[j]).abs();
            comparisons.push((i, j, diff, diff > cd));
        }
    }

    NemenyiResult {
        critical_difference: cd,
        comparisons,
    }
}

// ── McNemar's test ─────────────────────────────────────────────────

/// McNemar's test for comparing two classifiers on the same test set.
///
/// Tests H0: both classifiers have the same error rate.
///
/// # Arguments
/// * `pred_a` - Predictions from model A
/// * `pred_b` - Predictions from model B
/// * `truth` - True labels
pub fn mcnemar_test(pred_a: &[usize], pred_b: &[usize], truth: &[usize]) -> Result<TestResult> {
    if pred_a.len() != pred_b.len() {
        return Err(SmeltError::DimensionMismatch { expected: pred_a.len(), got: pred_b.len() });
    }
    if pred_a.len() != truth.len() {
        return Err(SmeltError::DimensionMismatch { expected: pred_a.len(), got: truth.len() });
    }

    // Count discordant pairs
    let mut b_count = 0usize; // A correct, B wrong
    let mut c_count = 0usize; // A wrong, B correct

    for i in 0..truth.len() {
        let a_correct = pred_a[i] == truth[i];
        let b_correct = pred_b[i] == truth[i];
        match (a_correct, b_correct) {
            (true, false) => b_count += 1,
            (false, true) => c_count += 1,
            _ => {}
        }
    }

    let b = b_count as f64;
    let c = c_count as f64;

    if b + c < 1.0 {
        return Ok(TestResult {
            test: "McNemar",
            statistic: 0.0,
            p_value: 1.0,
            significant: false,
        });
    }

    // McNemar's chi-squared with continuity correction
    let chi2 = ((b - c).abs() - 1.0).max(0.0).powi(2) / (b + c);
    let p_value = 1.0 - chi_squared_cdf(chi2, 1.0);

    Ok(TestResult {
        test: "McNemar",
        statistic: chi2,
        p_value,
        significant: p_value < 0.05,
    })
}

// ── Bootstrap confidence interval ──────────────────────────────────

/// Bootstrap confidence interval for any metric.
///
/// # Arguments
/// * `scores` - Sample of metric values (e.g., accuracy per fold)
/// * `confidence` - Confidence level (e.g., 0.95 for 95% CI)
/// * `n_bootstrap` - Number of bootstrap resamples (recommended: 10000)
/// * `seed` - Random seed for reproducibility
///
/// # Errors
/// Returns [`SmeltError::EmptyDataset`] if `scores` is empty, and
/// [`SmeltError::InvalidParameter`] if `n_bootstrap == 0` — both used to
/// panic (an empty-range `random_range` and a `usize` underflow,
/// respectively) instead of returning a clean error.
pub fn bootstrap_ci(
    scores: &[f64],
    confidence: f64,
    n_bootstrap: usize,
    seed: u64,
) -> Result<BootstrapCI> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    if scores.is_empty() {
        return Err(SmeltError::EmptyDataset);
    }
    if n_bootstrap == 0 {
        return Err(SmeltError::InvalidParameter(
            "n_bootstrap must be > 0".into(),
        ));
    }

    let n = scores.len();
    let estimate = scores.iter().sum::<f64>() / n as f64;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut boot_means = Vec::with_capacity(n_bootstrap);

    for _ in 0..n_bootstrap {
        let mut sum = 0.0;
        for _ in 0..n {
            let idx = rng.random_range(0..n);
            sum += scores[idx];
        }
        boot_means.push(sum / n as f64);
    }

    boot_means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let alpha = 1.0 - confidence;
    let lo_idx = ((alpha / 2.0) * n_bootstrap as f64) as usize;
    let hi_idx = ((1.0 - alpha / 2.0) * n_bootstrap as f64) as usize;
    let lo_idx = lo_idx.min(n_bootstrap - 1);
    let hi_idx = hi_idx.min(n_bootstrap - 1);

    Ok(BootstrapCI {
        estimate,
        lower: boot_means[lo_idx],
        upper: boot_means[hi_idx],
        confidence,
    })
}

// ── Helper: standard normal CDF ────────────────────────────────────

/// Approximate standard normal CDF using Abramowitz & Stegun formula 7.1.26.
#[allow(clippy::excessive_precision)]
fn standard_normal_cdf(x: f64) -> f64 {
    if x < -8.0 {
        return 0.0;
    }
    if x > 8.0 {
        return 1.0;
    }

    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d = 0.3989422804014327; // 1/sqrt(2*pi)
    let p = d * (-x * x / 2.0).exp();
    let c = t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));

    if x >= 0.0 { 1.0 - p * c } else { p * c }
}

// ── Helper: chi-squared CDF ────────────────────────────────────────

/// Approximate chi-squared CDF using the incomplete gamma function.
fn chi_squared_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    lower_incomplete_gamma(df / 2.0, x / 2.0) / gamma(df / 2.0)
}

/// Lower incomplete gamma function via series expansion.
fn lower_incomplete_gamma(a: f64, x: f64) -> f64 {
    if x < 0.0 {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut term = 1.0 / a;
    sum += term;
    for n in 1..200 {
        term *= x / (a + n as f64);
        sum += term;
        if term.abs() < 1e-15 * sum.abs() {
            break;
        }
    }
    sum * (-x).exp() * x.powf(a)
}

/// Gamma function via Lanczos approximation.
#[allow(clippy::excessive_precision)]
fn gamma(x: f64) -> f64 {
    if x < 0.5 {
        return std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma(1.0 - x));
    }
    let x = x - 1.0;
    let g = 7.0;
    let c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    let mut sum = c[0];
    for (i, &ci) in c.iter().enumerate().skip(1) {
        sum += ci / (x + i as f64);
    }

    let t = x + g + 0.5;
    (2.0 * std::f64::consts::PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * sum
}

// ── Helper: binomial CDF ───────────────────────────────────────────

/// Binomial CDF: P(X ≤ k) where X ~ Binomial(n, p).
fn binomial_cdf(k: usize, n: usize, p: f64) -> f64 {
    let mut sum = 0.0;
    let mut binom_coeff = 1.0;
    for i in 0..=k {
        if i > 0 {
            binom_coeff *= (n - i + 1) as f64 / i as f64;
        }
        sum += binom_coeff * p.powi(i as i32) * (1.0 - p).powi((n - i) as i32);
    }
    sum
}

// ── Convenience: compare_models ────────────────────────────────────

/// Compare two models' CV scores with Wilcoxon test + bootstrap CI.
///
/// Returns a human-readable summary string.
pub fn compare_models(
    name_a: &str,
    scores_a: &[f64],
    name_b: &str,
    scores_b: &[f64],
) -> Result<String> {
    if scores_a.len() != scores_b.len() {
        return Err(SmeltError::DimensionMismatch {
            expected: scores_a.len(),
            got: scores_b.len(),
        });
    }
    if scores_a.is_empty() {
        return Err(SmeltError::EmptyDataset);
    }

    let mean_a = scores_a.iter().sum::<f64>() / scores_a.len() as f64;
    let mean_b = scores_b.iter().sum::<f64>() / scores_b.len() as f64;

    let w = wilcoxon_signed_rank(scores_a, scores_b)?;
    let ci_a = bootstrap_ci(scores_a, 0.95, 10000, 42)?;
    let ci_b = bootstrap_ci(scores_b, 0.95, 10000, 43)?;

    let winner = if mean_a > mean_b { name_a } else { name_b };
    let sig = if w.significant {
        "statistically significant"
    } else {
        "NOT statistically significant"
    };

    Ok(format!(
        "{} vs {}:\n  {} mean={:.4} 95%CI=[{:.4}, {:.4}]\n  {} mean={:.4} 95%CI=[{:.4}, {:.4}]\n  Wilcoxon p={:.4} → {} wins ({})",
        name_a,
        name_b,
        name_a,
        mean_a,
        ci_a.lower,
        ci_a.upper,
        name_b,
        mean_b,
        ci_b.lower,
        ci_b.upper,
        w.p_value,
        winner,
        sig,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wilcoxon_significant() {
        let a = vec![0.92, 0.89, 0.91, 0.90, 0.93, 0.91, 0.90, 0.92, 0.89, 0.91];
        let b = vec![0.82, 0.84, 0.81, 0.83, 0.80, 0.82, 0.81, 0.83, 0.79, 0.82];
        let result = wilcoxon_signed_rank(&a, &b).unwrap();
        assert!(result.p_value < 0.01, "Should be highly significant");
        assert!(result.significant);
    }

    #[test]
    fn test_wilcoxon_not_significant() {
        let a = vec![0.90, 0.89, 0.91, 0.90, 0.89];
        let b = vec![0.89, 0.90, 0.90, 0.91, 0.90];
        let result = wilcoxon_signed_rank(&a, &b).unwrap();
        assert!(!result.significant);
    }

    /// Regression test: the old code used a normal approximation for every
    /// `n`, which could (incorrectly) report `p < 0.05` for as few as 5
    /// folds even when every fold favors the same model -- the true exact
    /// minimum p-value with n=5 and no ties is `2/2^5 = 0.0625`, which can
    /// never clear the 5% threshold. The module's own doctest asserted this
    /// invalid inference before this fix.
    #[test]
    fn five_folds_all_same_sign_cannot_reach_significance() {
        let a = vec![0.92, 0.89, 0.91, 0.90, 0.93];
        let b = vec![0.88, 0.87, 0.89, 0.86, 0.90];
        let result = wilcoxon_signed_rank(&a, &b).unwrap();
        assert!(
            (result.p_value - 0.0625).abs() < 1e-9,
            "exact minimum p-value for n=5, all same sign should be 2/32=0.0625, got {}",
            result.p_value
        );
        assert!(!result.significant, "5 folds can never reach p<0.05 when every pair has the same sign");
    }

    /// Cross-check the DP-based exact p-value against direct 2^n brute-force
    /// enumeration of sign assignments (an independent reimplementation) for
    /// a handful of small, tie-containing rank vectors.
    #[test]
    fn exact_dp_matches_brute_force_enumeration() {
        fn brute_force_two_sided_p(ranks: &[f64], w_plus: f64) -> f64 {
            let n = ranks.len();
            let total = 1u32 << n;
            let mut n_le = 0u32;
            let mut n_ge = 0u32;
            for mask in 0..total {
                let sum: f64 = (0..n)
                    .filter(|&i| mask & (1 << i) != 0)
                    .map(|i| ranks[i])
                    .sum();
                if sum <= w_plus + 1e-9 {
                    n_le += 1;
                }
                if sum >= w_plus - 1e-9 {
                    n_ge += 1;
                }
            }
            (2.0 * (n_le.min(n_ge) as f64) / total as f64).min(1.0)
        }

        let cases: Vec<(Vec<f64>, f64)> = vec![
            (vec![1.0, 2.0, 3.0, 4.0, 5.0], 15.0), // no ties, max sum
            (vec![1.0, 2.0, 3.0, 4.0, 5.0], 9.0),  // no ties, mid sum
            (vec![1.5, 1.5, 3.0, 4.5, 4.5], 10.5), // tie-averaged (half-integer) ranks
            (vec![2.0, 2.0, 2.0, 2.0], 4.0),       // all tied
        ];
        for (ranks, w_plus) in cases {
            let dp_p = wilcoxon_exact_two_sided_p(&ranks, w_plus);
            let bf_p = brute_force_two_sided_p(&ranks, w_plus);
            assert!(
                (dp_p - bf_p).abs() < 1e-9,
                "DP ({dp_p}) and brute-force ({bf_p}) exact p-values disagree for ranks={ranks:?}, w_plus={w_plus}"
            );
        }
    }

    #[test]
    fn test_sign_test() {
        let a = vec![0.9, 0.8, 0.7, 0.85, 0.95];
        let b = vec![0.85, 0.75, 0.65, 0.80, 0.90];
        let result = sign_test(&a, &b).unwrap();
        // All 5 pairs favor a → p = 2 * binom(0, 5, 0.5) = 0.0625
        assert!(result.p_value < 0.1);
    }

    #[test]
    fn test_friedman() {
        let dt = vec![0.80, 0.82, 0.79, 0.81, 0.83];
        let rf = vec![0.90, 0.88, 0.91, 0.89, 0.92];
        let xgb = vec![0.92, 0.89, 0.93, 0.91, 0.94];
        let result = friedman_test(&[&dt, &rf, &xgb]).unwrap();
        assert!(
            result.significant,
            "3 models with clear ordering should be significant"
        );
        // XGBoost should have the best (lowest) average rank
        assert!(result.avg_ranks[2] < result.avg_ranks[0]);
    }

    #[test]
    fn test_bootstrap_ci() {
        let scores = vec![0.90, 0.89, 0.91, 0.90, 0.92, 0.88, 0.91, 0.90, 0.89, 0.91];
        let ci = bootstrap_ci(&scores, 0.95, 10000, 42).unwrap();
        assert!(ci.lower > 0.88);
        assert!(ci.upper < 0.93);
        assert!((ci.estimate - 0.901).abs() < 0.01);
    }

    /// Regression test: `bootstrap_ci` used to panic on an empty `scores`
    /// slice (`rng.random_range(0..0)`) or `n_bootstrap == 0` (a `usize`
    /// underflow computing `n_bootstrap - 1`) instead of returning `Err`.
    #[test]
    fn bootstrap_ci_rejects_empty_scores_and_zero_resamples() {
        assert!(bootstrap_ci(&[], 0.95, 100, 42).is_err());
        assert!(bootstrap_ci(&[0.9, 0.8], 0.95, 0, 42).is_err());
    }

    #[test]
    fn test_mcnemar() {
        // Model A gets 90 right, B gets 80 right, 5 discordant pairs each way + extras
        let truth = vec![0; 100];
        let mut pred_a = vec![0; 100];
        let mut pred_b = vec![0; 100];
        // A wrong, B right: 5 cases
        for i in 0..5 {
            pred_a[i] = 1;
        }
        // A right, B wrong: 20 cases
        for i in 5..25 {
            pred_b[i] = 1;
        }
        let result = mcnemar_test(&pred_a, &pred_b, &truth).unwrap();
        assert!(
            result.significant,
            "20 vs 5 discordant should be significant"
        );
    }

    #[test]
    fn test_nemenyi() {
        let dt = vec![0.80, 0.82, 0.79, 0.81, 0.83];
        let rf = vec![0.90, 0.88, 0.91, 0.89, 0.92];
        let xgb = vec![0.92, 0.89, 0.93, 0.91, 0.94];
        let friedman = friedman_test(&[&dt, &rf, &xgb]).unwrap();
        let nemenyi = nemenyi_posthoc(&friedman, 5, 3);
        assert!(nemenyi.critical_difference > 0.0);
        // DT vs XGBoost should be significant
        let dt_xgb = nemenyi.comparisons.iter().find(|c| c.0 == 0 && c.1 == 2);
        assert!(dt_xgb.is_some());
    }

    #[test]
    fn test_compare_models() {
        let a = vec![0.92, 0.89, 0.91, 0.90, 0.93];
        let b = vec![0.82, 0.84, 0.81, 0.83, 0.80];
        let summary = compare_models("XGBoost", &a, "RF", &b).unwrap();
        assert!(summary.contains("XGBoost wins"));
    }

    #[test]
    fn test_normal_cdf() {
        assert!((standard_normal_cdf(0.0) - 0.5).abs() < 0.001);
        assert!((standard_normal_cdf(1.96) - 0.975).abs() < 0.01);
        assert!((standard_normal_cdf(-1.96) - 0.025).abs() < 0.01);
    }
}
