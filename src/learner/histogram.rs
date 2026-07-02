//! Shared histogram binning for gradient boosting engines.
//!
//! Column-major storage with u8 packing for cache-optimal histogram accumulation.
//! Used by XGBoost, LightGBM, and CatBoost.

use ndarray::Array2;

/// NaN sentinel value (bin index 255).
pub const NAN_BIN: u8 = u8::MAX;

/// Column-major histogram bins with u8 packing.
///
/// Storage: `cols[feature][sample]` — sequential access per feature
/// for cache-optimal histogram accumulation.
/// Max 254 real bins (255 = NaN sentinel).
pub struct HistBins {
    pub boundaries: Vec<Vec<f64>>,
    cols: Vec<Vec<u8>>,
}

impl HistBins {
    /// Build histogram bins from a feature matrix.
    /// `n_bins` is capped at 254 (255 reserved for NaN).
    pub fn build(features: &Array2<f64>, n_bins: usize) -> Self {
        let n_bins = n_bins.min(254);
        let n_samples = features.nrows();
        let n_features = features.ncols();
        let mut boundaries = Vec::with_capacity(n_features);
        let mut cols = Vec::with_capacity(n_features);

        for j in 0..n_features {
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
        }

        Self { boundaries, cols }
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
