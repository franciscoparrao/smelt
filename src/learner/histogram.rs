//! Shared histogram binning for gradient boosting engines.
//!
//! Column-major storage with u8 packing for cache-optimal histogram accumulation.
//! Used by XGBoost, LightGBM, and CatBoost.

use crate::task::FeatureType;
use ndarray::Array2;

/// NaN sentinel value (bin index 255).
pub const NAN_BIN: u8 = u8::MAX;

/// Maximum number of real bins (255 is the NaN sentinel). Categorical codes
/// at or above this are clamped into the last bin (overflow bucket).
pub const MAX_BINS: usize = 254;

/// Column-major histogram bins with u8 packing.
///
/// Storage: `cols[feature][sample]` — sequential access per feature
/// for cache-optimal histogram accumulation.
/// Max 254 real bins (255 = NaN sentinel).
///
/// Categorical features (declared via `build_typed`) are binned by category
/// code directly: bin b == category b, and `cat[feature] = Some(n_bins)`.
pub struct HistBins {
    /// Per-feature bin boundary values (upper edge of each bin, in ascending
    /// order) used to map a raw feature value to its bin index.
    pub boundaries: Vec<Vec<f64>>,
    cols: Vec<Vec<u8>>,
    /// `Some(n_categories)` for categorical features, `None` for numeric.
    pub cat: Vec<Option<usize>>,
}

impl HistBins {
    /// Build histogram bins from a feature matrix, all columns numeric.
    /// `n_bins` is capped at 254 (255 reserved for NaN).
    pub fn build(features: &Array2<f64>, n_bins: usize) -> Self {
        Self::build_typed(features, n_bins, &[])
    }

    /// Build histogram bins with per-column feature types. Columns marked
    /// `FeatureType::Categorical` are binned by category code (bin == code,
    /// codes ≥ 254 clamped into bin 253); an empty `types` slice means all
    /// numeric.
    pub fn build_typed(features: &Array2<f64>, n_bins: usize, types: &[FeatureType]) -> Self {
        let n_bins = n_bins.min(MAX_BINS);
        let n_samples = features.nrows();
        let n_features = features.ncols();
        let mut boundaries = Vec::with_capacity(n_features);
        let mut cols = Vec::with_capacity(n_features);
        let mut cat = Vec::with_capacity(n_features);

        for j in 0..n_features {
            if let Some(FeatureType::Categorical { n_categories }) = types.get(j) {
                let nc = (*n_categories).clamp(1, MAX_BINS);
                let mut col = Vec::with_capacity(n_samples);
                for i in 0..n_samples {
                    let val = features[[i, j]];
                    if val.is_nan() {
                        col.push(NAN_BIN);
                    } else {
                        col.push((val as usize).min(nc - 1) as u8);
                    }
                }
                // Dummy boundaries so n_bins()/bin_threshold() stay coherent:
                // "threshold" of a categorical bin is its own code.
                boundaries.push((0..nc).map(|c| c as f64).collect());
                cols.push(col);
                cat.push(Some(nc));
                continue;
            }

            let mut vals: Vec<f64> = features
                .column(j)
                .iter()
                .copied()
                .filter(|v| !v.is_nan())
                .collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            vals.dedup();

            let n_unique = vals.len();
            if n_unique == 0 {
                boundaries.push(vec![f64::INFINITY]);
                cols.push(vec![NAN_BIN; n_samples]);
                cat.push(None);
                continue;
            }

            let step = (n_unique as f64 / n_bins as f64).max(1.0);
            let mut bounds = Vec::new();
            let mut idx = step;
            while (idx as usize) < n_unique {
                bounds.push(vals[idx as usize]);
                idx += step;
            }
            if bounds.is_empty() || *bounds.last().unwrap() <= vals[n_unique - 1] {
                bounds.push(f64::INFINITY);
            }

            let mut col = Vec::with_capacity(n_samples);
            for i in 0..n_samples {
                let val = features[[i, j]];
                if val.is_nan() {
                    col.push(NAN_BIN);
                } else {
                    col.push(
                        bounds
                            .iter()
                            .position(|&b| val < b)
                            .unwrap_or(bounds.len() - 1) as u8,
                    );
                }
            }
            cols.push(col);
            boundaries.push(bounds);
            cat.push(None);
        }

        Self {
            boundaries,
            cols,
            cat,
        }
    }

    /// Number of bins for a feature.
    #[inline]
    pub fn n_bins(&self, feature: usize) -> usize {
        self.boundaries[feature].len()
    }

    /// Bin threshold value.
    #[inline]
    pub fn bin_threshold(&self, feature: usize, bin: usize) -> f64 {
        self.boundaries[feature][bin]
    }

    /// Get bin index for a (feature, sample) pair. 255 = NaN.
    #[inline]
    pub fn get_bin(&self, feature: usize, sample: usize) -> u8 {
        self.cols[feature][sample]
    }
}

/// Optimal categorical partition via the Fisher ordering (as in LightGBM):
/// sort the categories present in the node by gradient/hessian ratio, then
/// scan prefixes as if the feature were ordinal — for convex losses the best
/// left/right partition is always a prefix of that ordering. NaN mass is
/// tried on both sides, like the numeric default-direction scan.
///
/// `bin_g`/`bin_h` are indexed by category code. Categories with no hessian
/// mass in the node are excluded from the ordering (they route right at
/// prediction time, together with unseen categories). Returns
/// `(left_categories_sorted_by_code, gain, nan_goes_left)` for the best
/// positive-gain partition, or `None`.
pub(crate) fn best_categorical_split(
    bin_g: &[f64],
    bin_h: &[f64],
    nan_g: f64,
    nan_h: f64,
    min_child_weight: f64,
    gain_fn: impl Fn(f64, f64, f64, f64) -> f64,
) -> Option<(Vec<u16>, f64, bool)> {
    let mut cats: Vec<usize> = (0..bin_g.len()).filter(|&c| bin_h[c] > 0.0).collect();
    if cats.len() < 2 {
        return None;
    }
    cats.sort_by(|&a, &b| {
        let ra = bin_g[a] / bin_h[a];
        let rb = bin_g[b] / bin_h[b];
        ra.partial_cmp(&rb).unwrap_or(std::cmp::Ordering::Equal)
    });

    let total_g: f64 = cats.iter().map(|&c| bin_g[c]).sum::<f64>() + nan_g;
    let total_h: f64 = cats.iter().map(|&c| bin_h[c]).sum::<f64>() + nan_h;

    let mut best: Option<(usize, f64, bool)> = None; // (prefix_len, gain, nan_left)
    for nan_left in [false, true] {
        if nan_left && nan_h <= 0.0 {
            continue;
        }
        let (mut gl, mut hl) = if nan_left { (nan_g, nan_h) } else { (0.0, 0.0) };
        for k in 0..cats.len() - 1 {
            gl += bin_g[cats[k]];
            hl += bin_h[cats[k]];
            let (gr, hr) = (total_g - gl, total_h - hl);
            if hl < min_child_weight || hr < min_child_weight {
                continue;
            }
            let gain = gain_fn(gl, hl, gr, hr);
            if gain > 0.0 && best.as_ref().is_none_or(|&(_, bg, _)| gain > bg) {
                best = Some((k + 1, gain, nan_left));
            }
        }
    }

    best.map(|(prefix_len, gain, nan_left)| {
        let mut left: Vec<u16> = cats[..prefix_len].iter().map(|&c| c as u16).collect();
        left.sort_unstable();
        (left, gain, nan_left)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test: a binary {0,1} feature must land in two distinct bins.
    /// Previously the boundary-generation loop only pushed the INFINITY sentinel
    /// when the last generated boundary was strictly less than the max value;
    /// with n_unique <= n_bins the last boundary equals the max exactly, so the
    /// sentinel was skipped and both values fell into bin 0 — making binary
    /// (and other low-cardinality, e.g. one-hot) features unsplittable.
    #[test]
    fn binary_feature_gets_two_bins() {
        let features = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 0.0, 1.0]).unwrap();
        let bins = HistBins::build(&features, 256);
        assert_eq!(bins.n_bins(0), 2, "binary feature must produce 2 bins");
        assert_ne!(
            bins.get_bin(0, 0),
            bins.get_bin(0, 1),
            "0.0 and 1.0 must land in different bins"
        );
    }

    /// Same failure mode for any low-cardinality feature (n_unique <= n_bins):
    /// every distinct value must be separable from its neighbor.
    #[test]
    fn low_cardinality_feature_separates_all_values() {
        let features = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let bins = HistBins::build(&features, 256);
        let assigned: Vec<u8> = (0..5).map(|i| bins.get_bin(0, i)).collect();
        for i in 1..5 {
            assert_ne!(
                assigned[i - 1],
                assigned[i],
                "distinct values {} and {} collapsed into the same bin",
                i,
                i + 1
            );
        }
    }
}
