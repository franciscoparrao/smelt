//! Feature selection filters (mlr3-style).
//!
//! Filters compute a score per feature and `FilterSelector` selects the top-k.
//! Integrates with Pipeline as a Transformer — fit on training data only,
//! preventing data leakage in cross-validation.

use super::mutual_info;
use super::Transformer;
use crate::{Result, SmeltError};
use ndarray::Array2;

/// Trait for feature scoring methods.
pub trait Filter: Send + Sync {
    /// Compute a score for each feature. Higher = more important.
    fn score(&self, features: &Array2<f64>, target: &[f64]) -> Vec<f64>;

    /// Filter identifier.
    fn id(&self) -> &str;
}

/// Selects top-k features based on a Filter's scores.
///
/// Implements `Transformer` so it can be used in Pipeline with
/// automatic fit on training data only (no data leakage in CV).
///
/// # Examples
///
/// ```
/// use smelt_ml::preprocess::filter::FilterSelector;
///
/// // Select top 2 features by variance
/// let selector = FilterSelector::variance(2);
/// ```
#[derive(Clone)]
pub struct FilterSelector {
    filter: FilterBox,
    n_features: usize,
    selected_indices: Option<Vec<usize>>,
}

// Wrapper to make Filter cloneable via trait object
#[derive(Clone)]
struct FilterBox {
    inner: FilterType,
}

#[derive(Clone)]
enum FilterType {
    Variance,
    Correlation,
    AnovaF,
    InformationGain,
    MutualInfo,
    Mrmr,
    Jmi,
    Jmim,
    Cmim,
    Relief,
}

impl FilterBox {
    fn score(&self, features: &Array2<f64>, target: &[f64]) -> Vec<f64> {
        match &self.inner {
            FilterType::Variance => VarianceFilter.score(features, target),
            FilterType::Correlation => CorrelationFilter.score(features, target),
            FilterType::AnovaF => AnovaFFilter.score(features, target),
            FilterType::InformationGain => InformationGainFilter.score(features, target),
            FilterType::MutualInfo => MutualInfoFilter.score(features, target),
            FilterType::Mrmr => MrmrFilter.score(features, target),
            FilterType::Jmi => JmiFilter.score(features, target),
            FilterType::Jmim => JmimFilter.score(features, target),
            FilterType::Cmim => CmimFilter.score(features, target),
            FilterType::Relief => ReliefFilter.score(features, target),
        }
    }

    #[allow(dead_code)]
    fn id(&self) -> &str {
        match &self.inner {
            FilterType::Variance => "variance",
            FilterType::Correlation => "correlation",
            FilterType::AnovaF => "anova_f",
            FilterType::InformationGain => "information_gain",
            FilterType::MutualInfo => "mutual_info",
            FilterType::Mrmr => "mrmr",
            FilterType::Jmi => "jmi",
            FilterType::Jmim => "jmim",
            FilterType::Cmim => "cmim",
            FilterType::Relief => "relief",
        }
    }
}

impl FilterSelector {
    /// Create a selector that keeps the top `n_features` based on filter scores.
    pub fn variance(n_features: usize) -> Self {
        Self {
            filter: FilterBox {
                inner: FilterType::Variance,
            },
            n_features,
            selected_indices: None,
        }
    }

    /// Create a selector that keeps the top `n_features` by absolute
    /// correlation with the target.
    pub fn correlation(n_features: usize) -> Self {
        Self {
            filter: FilterBox {
                inner: FilterType::Correlation,
            },
            n_features,
            selected_indices: None,
        }
    }

    /// Create a selector that keeps the top `n_features` by ANOVA F-statistic
    /// against the target.
    pub fn anova_f(n_features: usize) -> Self {
        Self {
            filter: FilterBox {
                inner: FilterType::AnovaF,
            },
            n_features,
            selected_indices: None,
        }
    }

    /// Create a selector that keeps the top `n_features` by information gain
    /// (reduction in target entropy) with respect to the target.
    pub fn information_gain(n_features: usize) -> Self {
        Self {
            filter: FilterBox {
                inner: FilterType::InformationGain,
            },
            n_features,
            selected_indices: None,
        }
    }

    /// Create a selector that keeps the top `n_features` by mutual
    /// information with the target.
    pub fn mutual_info(n_features: usize) -> Self {
        Self {
            filter: FilterBox {
                inner: FilterType::MutualInfo,
            },
            n_features,
            selected_indices: None,
        }
    }

    /// Minimum Redundancy Maximum Relevance (Peng et al., 2005).
    pub fn mrmr(n_features: usize) -> Self {
        Self {
            filter: FilterBox {
                inner: FilterType::Mrmr,
            },
            n_features,
            selected_indices: None,
        }
    }

    /// Joint Mutual Information (Brown et al., 2012).
    pub fn jmi(n_features: usize) -> Self {
        Self {
            filter: FilterBox {
                inner: FilterType::Jmi,
            },
            n_features,
            selected_indices: None,
        }
    }

    /// Joint Mutual Information Maximization (Brown et al., 2012).
    pub fn jmim(n_features: usize) -> Self {
        Self {
            filter: FilterBox {
                inner: FilterType::Jmim,
            },
            n_features,
            selected_indices: None,
        }
    }

    /// Conditional Mutual Information Maximization (Fleuret, 2004).
    pub fn cmim(n_features: usize) -> Self {
        Self {
            filter: FilterBox {
                inner: FilterType::Cmim,
            },
            n_features,
            selected_indices: None,
        }
    }

    /// RReliefF distance-based filter (Kononenko, 1994).
    pub fn relief(n_features: usize) -> Self {
        Self {
            filter: FilterBox {
                inner: FilterType::Relief,
            },
            n_features,
            selected_indices: None,
        }
    }

    /// Get the selected feature indices (after fitting).
    pub fn selected_indices(&self) -> Option<&[usize]> {
        self.selected_indices.as_deref()
    }
}

impl Transformer for FilterSelector {
    fn id(&self) -> &str {
        "filter_selector"
    }

    fn fit(&mut self, features: &Array2<f64>) -> Result<()> {
        // Unsupervised fallback: use variance filter
        let scores: Vec<f64> = (0..features.ncols())
            .map(|j| {
                let col = features.column(j);
                let mean = col.sum() / col.len() as f64;
                col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / col.len() as f64
            })
            .collect();

        let mut ranked: Vec<(usize, f64)> = scores.into_iter().enumerate().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        self.selected_indices = Some(
            ranked
                .iter()
                .take(self.n_features.min(features.ncols()))
                .map(|(i, _)| *i)
                .collect(),
        );
        // Sort indices to preserve column order
        if let Some(ref mut idx) = self.selected_indices {
            idx.sort();
        }
        Ok(())
    }

    fn fit_supervised(&mut self, features: &Array2<f64>, target: &[f64]) -> Result<()> {
        let scores = self.filter.score(features, target);

        let mut ranked: Vec<(usize, f64)> = scores.into_iter().enumerate().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        self.selected_indices = Some(
            ranked
                .iter()
                .take(self.n_features.min(features.ncols()))
                .map(|(i, _)| *i)
                .collect(),
        );
        if let Some(ref mut idx) = self.selected_indices {
            idx.sort();
        }
        Ok(())
    }

    fn transform(&self, features: &Array2<f64>) -> Result<Array2<f64>> {
        let indices = self
            .selected_indices
            .as_ref()
            .ok_or(SmeltError::NotTrained)?;
        Ok(features.select(ndarray::Axis(1), indices))
    }

    fn transform_names(&self, names: &[String]) -> Result<Vec<String>> {
        let indices = self
            .selected_indices
            .as_ref()
            .ok_or(SmeltError::NotTrained)?;
        Ok(indices.iter().map(|&i| names[i].clone()).collect())
    }

    fn clone_box(&self) -> Box<dyn Transformer> {
        Box::new(self.clone())
    }
}

// ── Filter implementations ─────────────────────────────────────────

/// Variance filter: score = variance of each feature. Removes constant features.
pub struct VarianceFilter;

impl Filter for VarianceFilter {
    fn id(&self) -> &str {
        "variance"
    }

    fn score(&self, features: &Array2<f64>, _target: &[f64]) -> Vec<f64> {
        (0..features.ncols())
            .map(|j| {
                let col = features.column(j);
                let n = col.len() as f64;
                let mean = col.sum() / n;
                col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n
            })
            .collect()
    }
}

/// Correlation filter: score = |correlation(feature, target)|.
pub struct CorrelationFilter;

impl Filter for CorrelationFilter {
    fn id(&self) -> &str {
        "correlation"
    }

    fn score(&self, features: &Array2<f64>, target: &[f64]) -> Vec<f64> {
        let n = features.nrows() as f64;
        let t_mean = target.iter().sum::<f64>() / n;
        let t_std = (target.iter().map(|&t| (t - t_mean).powi(2)).sum::<f64>() / n).sqrt();

        (0..features.ncols())
            .map(|j| {
                let col = features.column(j);
                let f_mean = col.sum() / n;
                let f_std = (col.iter().map(|&v| (v - f_mean).powi(2)).sum::<f64>() / n).sqrt();

                if f_std < 1e-10 || t_std < 1e-10 {
                    return 0.0;
                }

                let cov: f64 = col
                    .iter()
                    .zip(target)
                    .map(|(&f, &t)| (f - f_mean) * (t - t_mean))
                    .sum::<f64>()
                    / n;

                (cov / (f_std * t_std)).abs()
            })
            .collect()
    }
}

/// ANOVA F-test filter: score = between-group variance / within-group variance.
/// Works for classification targets (target as f64 class labels).
pub struct AnovaFFilter;

impl Filter for AnovaFFilter {
    fn id(&self) -> &str {
        "anova_f"
    }

    fn score(&self, features: &Array2<f64>, target: &[f64]) -> Vec<f64> {
        let n = features.nrows();
        let classes: Vec<usize> = target.iter().map(|&t| t as usize).collect();
        let n_classes = classes.iter().max().map_or(0, |&m| m + 1);
        if n_classes < 2 {
            return vec![0.0; features.ncols()];
        }

        (0..features.ncols())
            .map(|j| {
                let col = features.column(j);
                let global_mean = col.sum() / n as f64;

                let mut class_sums = vec![0.0; n_classes];
                let mut class_counts = vec![0usize; n_classes];
                for i in 0..n {
                    class_sums[classes[i]] += col[i];
                    class_counts[classes[i]] += 1;
                }

                // Between-group variance
                let mut ss_between = 0.0;
                for c in 0..n_classes {
                    if class_counts[c] == 0 {
                        continue;
                    }
                    let class_mean = class_sums[c] / class_counts[c] as f64;
                    ss_between += class_counts[c] as f64 * (class_mean - global_mean).powi(2);
                }

                // Within-group variance
                let mut ss_within = 0.0;
                for i in 0..n {
                    let class_mean = class_sums[classes[i]] / class_counts[classes[i]] as f64;
                    ss_within += (col[i] - class_mean).powi(2);
                }

                if ss_within < 1e-10 {
                    return f64::INFINITY;
                }

                let df_between = (n_classes - 1) as f64;
                let df_within = (n - n_classes) as f64;
                if df_within <= 0.0 {
                    return 0.0;
                }

                (ss_between / df_between) / (ss_within / df_within)
            })
            .collect()
    }
}

/// Information Gain filter: score = entropy(target) - conditional_entropy(target | feature).
/// Discretizes continuous features into bins.
pub struct InformationGainFilter;

impl Filter for InformationGainFilter {
    fn id(&self) -> &str {
        "information_gain"
    }

    fn score(&self, features: &Array2<f64>, target: &[f64]) -> Vec<f64> {
        let n = features.nrows() as f64;
        let classes: Vec<usize> = target.iter().map(|&t| t as usize).collect();
        let n_classes = classes.iter().max().map_or(0, |&m| m + 1);

        // Target entropy H(Y)
        let mut class_counts = vec![0usize; n_classes];
        for &c in &classes {
            class_counts[c] += 1;
        }
        let h_target = entropy(&class_counts, n);

        (0..features.ncols())
            .map(|j| {
                let col = features.column(j);
                // Discretize into 10 bins
                let min = col.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let range = (max - min).max(1e-10);
                let n_bins = 10usize;

                let bins: Vec<usize> = col
                    .iter()
                    .map(|&v| (((v - min) / range * n_bins as f64) as usize).min(n_bins - 1))
                    .collect();

                // Conditional entropy H(Y | X_binned)
                let mut bin_class_counts = vec![vec![0usize; n_classes]; n_bins];
                let mut bin_counts = vec![0usize; n_bins];
                for i in 0..features.nrows() {
                    bin_class_counts[bins[i]][classes[i]] += 1;
                    bin_counts[bins[i]] += 1;
                }

                let h_conditional: f64 = (0..n_bins)
                    .map(|b| {
                        if bin_counts[b] == 0 {
                            return 0.0;
                        }
                        let weight = bin_counts[b] as f64 / n;
                        weight * entropy(&bin_class_counts[b], bin_counts[b] as f64)
                    })
                    .sum();

                (h_target - h_conditional).max(0.0)
            })
            .collect()
    }
}

/// Mutual Information filter (continuous approximation via binning).
pub struct MutualInfoFilter;

impl Filter for MutualInfoFilter {
    fn id(&self) -> &str {
        "mutual_info"
    }

    fn score(&self, features: &Array2<f64>, target: &[f64]) -> Vec<f64> {
        // For regression targets, use binned mutual information
        let n = features.nrows();
        let n_bins = 10usize;

        // Bin the target
        let t_min = target.iter().cloned().fold(f64::INFINITY, f64::min);
        let t_max = target.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let t_range = (t_max - t_min).max(1e-10);
        let t_bins: Vec<usize> = target
            .iter()
            .map(|&v| (((v - t_min) / t_range * n_bins as f64) as usize).min(n_bins - 1))
            .collect();

        (0..features.ncols())
            .map(|j| {
                let col = features.column(j);
                let f_min = col.iter().cloned().fold(f64::INFINITY, f64::min);
                let f_max = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let f_range = (f_max - f_min).max(1e-10);
                let f_bins: Vec<usize> = col
                    .iter()
                    .map(|&v| (((v - f_min) / f_range * n_bins as f64) as usize).min(n_bins - 1))
                    .collect();

                // Joint and marginal counts
                let mut joint = vec![vec![0usize; n_bins]; n_bins]; // [f_bin][t_bin]
                let mut f_counts = vec![0usize; n_bins];
                let mut t_counts = vec![0usize; n_bins];

                for i in 0..n {
                    joint[f_bins[i]][t_bins[i]] += 1;
                    f_counts[f_bins[i]] += 1;
                    t_counts[t_bins[i]] += 1;
                }

                // MI = Σ p(x,y) log(p(x,y) / (p(x)p(y)))
                let n_f = n as f64;
                let mut mi = 0.0;
                for fb in 0..n_bins {
                    for tb in 0..n_bins {
                        if joint[fb][tb] == 0 {
                            continue;
                        }
                        let p_joint = joint[fb][tb] as f64 / n_f;
                        let p_f = f_counts[fb] as f64 / n_f;
                        let p_t = t_counts[tb] as f64 / n_f;
                        mi += p_joint * (p_joint / (p_f * p_t)).ln();
                    }
                }
                mi.max(0.0)
            })
            .collect()
    }
}

fn entropy(counts: &[usize], total: f64) -> f64 {
    if total <= 0.0 {
        return 0.0;
    }
    counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / total;
            -p * p.ln()
        })
        .sum()
}

// ── Greedy info-theoretic filters ──────────────────────────────────

/// Helper: greedy sequential selection that assigns descending scores.
/// `criterion` receives (candidate_bins, target_bins, &[selected_bins]) → f64.
fn greedy_select(
    features: &Array2<f64>,
    target: &[f64],
    criterion: fn(&[usize], &[usize], &[&[usize]]) -> f64,
) -> Vec<f64> {
    let p = features.ncols();
    let t_bins = mutual_info::discretize(target);

    // Pre-discretize all features
    let all_bins: Vec<Vec<usize>> = (0..p)
        .map(|j| {
            let col: Vec<f64> = features.column(j).to_vec();
            mutual_info::discretize(&col)
        })
        .collect();

    let mut selected: Vec<usize> = Vec::new();
    let mut available: Vec<bool> = vec![true; p];
    let mut scores = vec![0.0_f64; p];

    for rank in 0..p {
        let sel_bins: Vec<&[usize]> = selected.iter().map(|&i| all_bins[i].as_slice()).collect();

        let mut best_j = 0;
        let mut best_val = f64::NEG_INFINITY;

        for j in 0..p {
            if !available[j] {
                continue;
            }
            let val = criterion(&all_bins[j], &t_bins, &sel_bins);
            if val > best_val {
                best_val = val;
                best_j = j;
            }
        }

        available[best_j] = false;
        selected.push(best_j);
        // Higher score = selected earlier
        scores[best_j] = (p - rank) as f64;
    }

    scores
}

/// MRMR: Minimum Redundancy Maximum Relevance (Peng et al., 2005).
pub struct MrmrFilter;

impl Filter for MrmrFilter {
    fn id(&self) -> &str {
        "mrmr"
    }

    fn score(&self, features: &Array2<f64>, target: &[f64]) -> Vec<f64> {
        greedy_select(features, target, |cand, tgt, selected| {
            let relevance = mutual_info::mi_from_bins(cand, tgt);
            if selected.is_empty() {
                return relevance;
            }
            let redundancy: f64 = selected
                .iter()
                .map(|s| mutual_info::mi_from_bins(cand, s))
                .sum::<f64>()
                / selected.len() as f64;
            relevance - redundancy
        })
    }
}

/// JMI: Joint Mutual Information (Brown et al., 2012).
pub struct JmiFilter;

impl Filter for JmiFilter {
    fn id(&self) -> &str {
        "jmi"
    }

    fn score(&self, features: &Array2<f64>, target: &[f64]) -> Vec<f64> {
        greedy_select(features, target, |cand, tgt, selected| {
            if selected.is_empty() {
                return mutual_info::mi_from_bins(cand, tgt);
            }
            // JMI = Σ_{s in S} I(cand, s; y)
            selected
                .iter()
                .map(|s| mutual_info::joint_mi(cand, s, tgt))
                .sum()
        })
    }
}

/// JMIM: Joint Mutual Information Maximization (Brown et al., 2012).
pub struct JmimFilter;

impl Filter for JmimFilter {
    fn id(&self) -> &str {
        "jmim"
    }

    fn score(&self, features: &Array2<f64>, target: &[f64]) -> Vec<f64> {
        greedy_select(features, target, |cand, tgt, selected| {
            if selected.is_empty() {
                return mutual_info::mi_from_bins(cand, tgt);
            }
            // JMIM = min_{s in S} I(cand, s; y)
            selected
                .iter()
                .map(|s| mutual_info::joint_mi(cand, s, tgt))
                .fold(f64::INFINITY, f64::min)
        })
    }
}

/// CMIM: Conditional Mutual Information Maximization (Fleuret, 2004).
pub struct CmimFilter;

impl Filter for CmimFilter {
    fn id(&self) -> &str {
        "cmim"
    }

    fn score(&self, features: &Array2<f64>, target: &[f64]) -> Vec<f64> {
        greedy_select(features, target, |cand, tgt, selected| {
            if selected.is_empty() {
                return mutual_info::mi_from_bins(cand, tgt);
            }
            // CMIM = min_{s in S} I(cand; y | s)
            selected
                .iter()
                .map(|s| mutual_info::conditional_mi(cand, tgt, s))
                .fold(f64::INFINITY, f64::min)
        })
    }
}

// ── Relief ─────────────────────────────────────────────────────────

/// RReliefF filter for regression (Kononenko, 1994).
pub struct ReliefFilter;

impl Filter for ReliefFilter {
    fn id(&self) -> &str {
        "relief"
    }

    fn score(&self, features: &Array2<f64>, target: &[f64]) -> Vec<f64> {
        let n = features.nrows();
        let p = features.ncols();
        let k = 10.min(n - 1); // number of neighbours

        // Normalize features and target to [0, 1]
        let mut col_min = vec![f64::INFINITY; p];
        let mut col_max = vec![f64::NEG_INFINITY; p];
        for j in 0..p {
            for i in 0..n {
                let v = features[[i, j]];
                if v < col_min[j] {
                    col_min[j] = v;
                }
                if v > col_max[j] {
                    col_max[j] = v;
                }
            }
        }
        let col_range: Vec<f64> = (0..p).map(|j| (col_max[j] - col_min[j]).max(1e-10)).collect();

        let t_min = target.iter().cloned().fold(f64::INFINITY, f64::min);
        let t_max = target.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let t_range = (t_max - t_min).max(1e-10);

        // Precompute pairwise distances (Euclidean on normalized features)
        // For each instance, find k nearest neighbours.
        //
        // RReliefF (Robnik-Šikonja & Kononenko) estimates
        // W[A] = P(diffA | diffC) - P(diffA | equalC), where:
        //   P(diffA|diffC)  ≈ (Σ diffA·diffC·w) / (Σ diffC·w)         =: pos/n_dc
        //   P(diffA|equalC) ≈ (Σ diffA·(1-diffC)·w) / (Σ (1-diffC)·w) =: neg/n_equal
        // The two terms are normalized by DIFFERENT denominators (n_dc vs.
        // n_equal = total_w - n_dc) since they estimate different
        // conditional probabilities; dividing both by n_dc collapses them
        // onto the same scale and understates the "same-target" (noise)
        // term whenever n_equal != n_dc (the common case).
        let mut pos = vec![0.0_f64; p]; // Σ diffA · diffC · w
        let mut neg = vec![0.0_f64; p]; // Σ diffA · (1 - diffC) · w
        let mut n_dc = 0.0_f64; // Σ diffC · w
        let mut total_w = 0.0_f64; // Σ w

        for i in 0..n {
            // Compute distances to all other instances
            let mut dists: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let d: f64 = (0..p)
                        .map(|f| {
                            let diff =
                                (features[[i, f]] - features[[j, f]]) / col_range[f];
                            diff * diff
                        })
                        .sum::<f64>()
                        .sqrt();
                    (j, d)
                })
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Adaptive sigma = distance to k-th neighbour
            let sigma = dists[k.min(dists.len()) - 1].1.max(1e-10);

            for &(j, d) in dists.iter().take(k) {
                let w = (-d * d / (sigma * sigma)).exp();
                let diff_target = ((target[i] - target[j]) / t_range).abs();
                n_dc += diff_target * w;
                total_w += w;

                for f in 0..p {
                    let diff_f = ((features[[i, f]] - features[[j, f]]) / col_range[f]).abs();
                    pos[f] += diff_target * diff_f * w;
                    neg[f] += (1.0 - diff_target) * diff_f * w;
                }
            }
        }

        let n_equal = (total_w - n_dc).max(0.0);
        (0..p)
            .map(|f| {
                let pos_term = if n_dc > 1e-10 { pos[f] / n_dc } else { 0.0 };
                let neg_term = if n_equal > 1e-10 { neg[f] / n_equal } else { 0.0 };
                pos_term - neg_term
            })
            .collect()
    }
}

#[cfg(test)]
mod relief_tests {
    use super::*;
    use ndarray::array;

    /// Golden test against an independent re-implementation of the same
    /// weighted-RReliefF formula, computed in Python/numpy. On this fixture
    /// (7 samples share one target value, 1 outlier differs -- so n_dc is
    /// small relative to n_equal), the M11 normalization bug (dividing BOTH
    /// the positive and negative terms by n_dc, instead of the negative
    /// term by n_equal = total_weight - n_dc) produces wildly different,
    /// mostly-negative scores; the corrected normalization gives small
    /// positive scores.
    #[test]
    fn golden_normalization_matches_independent_reference() {
        let features = array![
            [0.0, 5.0],
            [0.1, 1.0],
            [0.2, 8.0],
            [0.3, 2.0],
            [5.0, 9.0],
            [5.1, 0.0],
            [5.2, 6.0],
            [5.3, 3.0],
        ];
        let target = vec![0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0];

        let scores = ReliefFilter.score(&features, &target);
        let expected = [0.01938867, 0.17590801];
        for (got, exp) in scores.iter().zip(expected) {
            assert!(
                (got - exp).abs() < 1e-6,
                "Relief score {got} should match the independently-computed reference {exp} \
                 (the bug's normalization would give a large negative value like -0.94 or \
                 -0.70 instead)"
            );
        }
    }
}
