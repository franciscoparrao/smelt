//! Pre-allocated histogram pool for the histogram subtraction trick.
//!
//! One flat buffer per tree depth level, allocated once per tree.
//! Avoids per-node allocation that caused 20x regression.

use super::histogram::{HistBins, best_categorical_split};
use rayon::prelude::*;

/// Best split found in a pooled histogram:
/// (feature, threshold, gain, nan_goes_left, split_bin, left_cats).
/// `left_cats` is `Some(sorted codes going left)` for categorical features.
pub(crate) type PoolSplit = (usize, f64, f64, bool, usize, Option<Vec<u16>>);

pub(crate) struct HistPool {
    levels: Vec<Vec<f64>>,
    pub max_bins: usize,
    feat_stride: usize,
}

impl HistPool {
    pub fn new(max_depth: usize, n_col: usize, max_bins: usize) -> Self {
        let feat_stride = 2 * max_bins + 2;
        let level_size = n_col * feat_stride;
        let levels = (0..=max_depth).map(|_| vec![0.0; level_size]).collect();
        Self {
            levels,
            max_bins,
            feat_stride,
        }
    }

    /// Populate pool[depth] from per-feature histogram data (sequential copy, O(n_features × n_bins)).
    pub fn store_hists(&mut self, depth: usize, hists: &[(Vec<f64>, Vec<f64>, f64, f64)]) {
        let buf = &mut self.levels[depth];
        buf.fill(0.0);
        let mb = self.max_bins;
        let fs = self.feat_stride;
        for (fi, (bg, bh, ng, nh)) in hists.iter().enumerate() {
            let off = fi * fs;
            let nb = bg.len();
            buf[off..off + nb].copy_from_slice(bg);
            buf[off + mb..off + mb + nb].copy_from_slice(bh);
            buf[off + 2 * mb] = *ng;
            buf[off + 2 * mb + 1] = *nh;
        }
    }

    /// In-place: levels[child] = levels[parent] - levels[child]
    pub fn subtract_in_place(&mut self, parent: usize, child: usize) {
        assert_ne!(parent, child);
        let (p, c) = if parent < child {
            let (left, right) = self.levels.split_at_mut(child);
            (&left[parent], &mut right[0])
        } else {
            let (left, right) = self.levels.split_at_mut(parent);
            (&right[0], &mut left[child])
        };
        for i in 0..c.len() {
            c[i] = p[i] - c[i];
        }
    }

    /// Find best split from histogram at depth.
    pub fn find_best(
        &self,
        depth: usize,
        col_indices: &[usize],
        bins: &HistBins,
        min_cw: f64,
        lambda: f64,
        _alpha: f64,
        gamma: f64,
    ) -> Option<PoolSplit> {
        let buf = &self.levels[depth];
        let mb = self.max_bins;
        let fs = self.feat_stride;

        let results: Vec<Option<PoolSplit>> = col_indices
            .par_iter()
            .enumerate()
            .map(|(fi, &feat)| {
                let off = fi * fs;
                let bin_g = &buf[off..off + mb];
                let bin_h = &buf[off + mb..off + 2 * mb];
                let nan_g = buf[off + 2 * mb];
                let nan_h = buf[off + 2 * mb + 1];
                let nb = bins.n_bins(feat);

                let sg = |gl: f64, hl: f64, gr: f64, hr: f64| -> f64 {
                    0.5 * (gl * gl / (hl + lambda) + gr * gr / (hr + lambda)
                        - (gl + gr) * (gl + gr) / (hl + hr + lambda))
                        - gamma
                };

                if bins.cat[feat].is_some() {
                    return best_categorical_split(
                        &bin_g[..nb],
                        &bin_h[..nb],
                        nan_g,
                        nan_h,
                        min_cw,
                        sg,
                    )
                    .map(|(left_cats, gain, nan_left)| {
                        (feat, f64::NAN, gain, nan_left, 0, Some(left_cats))
                    });
                }

                let total_g: f64 = bin_g[..nb].iter().sum::<f64>() + nan_g;
                let total_h: f64 = bin_h[..nb].iter().sum::<f64>() + nan_h;
                let mut best_gain = 0.0f64;
                let mut best_bin = 0usize;
                let mut best_nan_left = false;

                let (mut gl, mut hl) = (0.0, 0.0);
                for b in 0..nb.saturating_sub(1) {
                    gl += bin_g[b];
                    hl += bin_h[b];
                    let (gr, hr) = (total_g - gl, total_h - hl);
                    if hl < min_cw || hr < min_cw {
                        continue;
                    }
                    let g = sg(gl, hl, gr, hr);
                    if g > best_gain {
                        best_gain = g;
                        best_bin = b;
                        best_nan_left = false;
                    }
                }
                if nan_h > 0.0 {
                    let (mut gl, mut hl) = (nan_g, nan_h);
                    for b in 0..nb.saturating_sub(1) {
                        gl += bin_g[b];
                        hl += bin_h[b];
                        let (gr, hr) = (total_g - gl, total_h - hl);
                        if hl < min_cw || hr < min_cw {
                            continue;
                        }
                        let g = sg(gl, hl, gr, hr);
                        if g > best_gain {
                            best_gain = g;
                            best_bin = b;
                            best_nan_left = true;
                        }
                    }
                }

                if best_gain > 0.0 {
                    Some((
                        feat,
                        bins.bin_threshold(feat, best_bin),
                        best_gain,
                        best_nan_left,
                        best_bin,
                        None,
                    ))
                } else {
                    None
                }
            })
            .collect();

        results
            .into_iter()
            .flatten()
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
    }
}
