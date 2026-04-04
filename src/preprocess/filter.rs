//! Feature selection filters (mlr3-style).
//!
//! Filters compute a score per feature and `FilterSelector` selects the top-k.
//! Integrates with Pipeline as a Transformer — fit on training data only,
//! preventing data leakage in cross-validation.

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
}

impl FilterBox {
    fn score(&self, features: &Array2<f64>, target: &[f64]) -> Vec<f64> {
        match &self.inner {
            FilterType::Variance => VarianceFilter.score(features, target),
            FilterType::Correlation => CorrelationFilter.score(features, target),
            FilterType::AnovaF => AnovaFFilter.score(features, target),
            FilterType::InformationGain => InformationGainFilter.score(features, target),
            FilterType::MutualInfo => MutualInfoFilter.score(features, target),
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

    pub fn correlation(n_features: usize) -> Self {
        Self {
            filter: FilterBox {
                inner: FilterType::Correlation,
            },
            n_features,
            selected_indices: None,
        }
    }

    pub fn anova_f(n_features: usize) -> Self {
        Self {
            filter: FilterBox {
                inner: FilterType::AnovaF,
            },
            n_features,
            selected_indices: None,
        }
    }

    pub fn information_gain(n_features: usize) -> Self {
        Self {
            filter: FilterBox {
                inner: FilterType::InformationGain,
            },
            n_features,
            selected_indices: None,
        }
    }

    pub fn mutual_info(n_features: usize) -> Self {
        Self {
            filter: FilterBox {
                inner: FilterType::MutualInfo,
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
