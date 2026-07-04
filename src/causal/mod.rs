//! Causal Forest: estimate heterogeneous treatment effects (CATE).
//!
//! Implements the honest causal forest algorithm from Athey & Imbens (2016)
//! and Wager & Athey (2018). Estimates τ(x) = E[Y(1) - Y(0) | X = x]
//! for each observation.
//!
//! Key features:
//! - Honest splitting: separate samples for tree structure and effect estimation
//! - Splitting criterion maximizes heterogeneity in treatment effects
//! - Forest averaging with variance estimates
//! - No assumption of treatment assignment mechanism (works with observational data)
//!
//! References:
//! - Athey, S., & Imbens, G. (2016). Recursive partitioning for heterogeneous causal effects. PNAS.
//! - Wager, S., & Athey, S. (2018). Estimation and inference of heterogeneous treatment effects using random forests. JASA.

use crate::Result;
use crate::SmeltError;
use ndarray::{Array2, ArrayView1};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;

pub mod meta_learners;

/// Estimated causal effect for a single observation.
#[derive(Debug, Clone)]
pub struct CausalEffect {
    /// Estimated conditional average treatment effect τ̂(x).
    pub estimate: f64,
    /// Standard error of the estimate.
    pub std_error: f64,
    /// Lower bound of 95% confidence interval.
    pub ci_lower: f64,
    /// Upper bound of 95% confidence interval.
    pub ci_upper: f64,
}

/// Result of causal forest estimation.
#[derive(Debug)]
pub struct CausalForestResult {
    /// Individual treatment effects for each observation.
    pub effects: Vec<CausalEffect>,
    /// Average treatment effect (ATE) across all observations.
    pub ate: f64,
    /// Standard error of the ATE.
    pub ate_std_error: f64,
    /// Feature importance for treatment effect heterogeneity.
    pub feature_importance: Vec<(String, f64)>,
}

/// Causal Forest for heterogeneous treatment effect estimation.
///
/// # Examples
///
/// ```
/// use smelt_ml::causal::CausalForest;
/// use ndarray::array;
///
/// // Features, binary treatment (0/1), and continuous outcome
/// let features = array![
///     [25.0], [30.0], [35.0], [40.0], [45.0],
///     [25.0], [30.0], [35.0], [40.0], [45.0],
/// ];
/// let treatment = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
/// let outcome = vec![5.0, 6.0, 7.0, 8.0, 9.0, 8.0, 10.0, 12.0, 14.0, 16.0];
///
/// let cf = CausalForest::new()
///     .with_n_estimators(50)
///     .with_seed(42);
/// let result = cf.estimate(&features, &treatment, &outcome,
///     &["age".to_string()]).unwrap();
///
/// println!("ATE: {:.2} +/- {:.2}", result.ate, result.ate_std_error);
/// for (i, effect) in result.effects.iter().enumerate() {
///     println!("Unit {}: tau={:.2}, CI=[{:.2}, {:.2}]",
///         i, effect.estimate, effect.ci_lower, effect.ci_upper);
/// }
/// ```
pub struct CausalForest {
    n_estimators: usize,
    max_depth: Option<usize>,
    min_samples_leaf: usize,
    honesty_fraction: f64,
    subsample_fraction: f64,
    seed: u64,
}

impl Default for CausalForest {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            max_depth: None,
            min_samples_leaf: 5,
            honesty_fraction: 0.5,
            subsample_fraction: 0.5,
            seed: 42,
        }
    }
}

impl CausalForest {
    /// Creates a causal forest with default hyperparameters (100 trees, no
    /// max depth, min leaf size 5, honesty fraction 0.5, seed 42).
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets the number of trees in the forest.
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }
    /// Sets the maximum tree depth (unlimited by default).
    pub fn with_max_depth(mut self, d: usize) -> Self {
        self.max_depth = Some(d);
        self
    }
    /// Sets the minimum number of samples required in each leaf.
    pub fn with_min_samples_leaf(mut self, n: usize) -> Self {
        self.min_samples_leaf = n;
        self
    }
    /// Sets the fraction of each tree's subsample held out for honest
    /// leaf-effect re-estimation, rather than used to choose the tree's
    /// splits.
    pub fn with_honesty_fraction(mut self, f: f64) -> Self {
        self.honesty_fraction = f;
        self
    }
    /// Sets the RNG seed controlling subsampling and the honest train/estimation split.
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Estimate heterogeneous treatment effects.
    ///
    /// - `features`: covariate matrix (n_samples x n_features)
    /// - `treatment`: binary treatment assignment (0 = control, 1 = treated)
    /// - `outcome`: continuous outcome variable
    /// - `feature_names`: names for feature importance output
    pub fn estimate(
        &self,
        features: &Array2<f64>,
        treatment: &[usize],
        outcome: &[f64],
        feature_names: &[String],
    ) -> Result<CausalForestResult> {
        let n_samples = features.nrows();
        let n_features = features.ncols();

        if treatment.len() != n_samples || outcome.len() != n_samples {
            return Err(SmeltError::DimensionMismatch {
                expected: n_samples,
                got: treatment.len().min(outcome.len()),
            });
        }

        // Train forest: each tree produces a tau estimate per observation,
        // plus an in-bag indicator per observation (was it in this tree's
        // subsample at all, train or estimation half) needed for the
        // infinitesimal jackknife variance below.
        let tree_results: Vec<(Vec<Option<f64>>, Vec<f64>, Vec<bool>)> = (0..self.n_estimators)
            .into_par_iter()
            .map(|i| {
                let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(i as u64));

                // Subsample
                let sub_size = (n_samples as f64 * self.subsample_fraction).ceil() as usize;
                let mut sub_indices: Vec<usize> = (0..n_samples).collect();
                sub_indices.shuffle(&mut rng);
                sub_indices.truncate(sub_size);
                let mut in_bag = vec![false; n_samples];
                for &idx in &sub_indices {
                    in_bag[idx] = true;
                }

                // Honest split: training vs estimation
                let split_point =
                    (sub_indices.len() as f64 * (1.0 - self.honesty_fraction)) as usize;
                let train_idx = &sub_indices[..split_point];
                let est_idx = &sub_indices[split_point..];

                // Build the tree STRUCTURE from train_idx only (splits chosen
                // to maximise treatment-effect heterogeneity on the training
                // half). Its leaf tau values are provisional -- computed from
                // train_idx during construction purely so build_node has a
                // criterion to split on -- and are overwritten below.
                let mut importances = vec![0.0; n_features];
                let mut root = build_causal_tree(
                    features,
                    treatment,
                    outcome,
                    train_idx,
                    n_features,
                    self.max_depth,
                    self.min_samples_leaf,
                    &mut importances,
                    &mut rng,
                );

                // Honest step: route the held-out estimation indices down the
                // (already-fixed) tree structure and recompute each leaf's
                // tau using ONLY those points -- the estimation sample never
                // influenced where the splits are, and the training sample
                // never influences the final tau (Athey & Imbens 2016;
                // Wager & Athey 2018). Leaves with no honest treated/control
                // pair are marked invalid rather than falling back to the
                // biased training-sample tau.
                populate_leaf_tau(&mut root, features, est_idx, treatment, outcome);

                // Out-of-bag aggregation: a point that was in this tree's
                // subsample (train_idx OR est_idx -- in_bag) must not use
                // this tree's tau for its own query. If it's in est_idx, its
                // own outcome directly fed the leaf tau that would be
                // returned for it (auto-influence / in-sample optimism); if
                // it's in train_idx, it influenced where the splits are.
                // Only trees where the point never appeared in the subsample
                // contribute to its aggregated estimate, matching how OOB
                // predictions are formed for ordinary random forests.
                let mut all_effects = vec![None; n_samples];
                for idx in 0..n_samples {
                    if in_bag[idx] {
                        continue;
                    }
                    let leaf = find_leaf(&root, features.row(idx));
                    if leaf.honest_valid {
                        all_effects[idx] = Some(leaf.tau);
                    }
                }

                (all_effects, importances, in_bag)
            })
            .collect();

        // Aggregate across trees
        let mut total_importances = vec![0.0; n_features];
        for (_, imp, _) in &tree_results {
            for (j, v) in imp.iter().enumerate() {
                total_importances[j] += v;
            }
        }

        let effects: Vec<CausalEffect> = (0..n_samples)
            .into_par_iter()
            .map(|query| {
                let values: Vec<(f64, &[bool])> = tree_results
                    .iter()
                    .filter_map(|(tree_effects, _, in_bag)| {
                        tree_effects[query].map(|tau| (tau, in_bag.as_slice()))
                    })
                    .collect();
                if values.is_empty() {
                    return CausalEffect {
                        estimate: 0.0,
                        std_error: f64::INFINITY,
                        ci_lower: f64::NEG_INFINITY,
                        ci_upper: f64::INFINITY,
                    };
                }
                let mean_tau =
                    values.iter().map(|(v, _)| v).sum::<f64>() / values.len() as f64;
                let se = infinitesimal_jackknife_se(&values, n_samples);
                CausalEffect {
                    estimate: mean_tau,
                    std_error: se,
                    ci_lower: mean_tau - 1.96 * se,
                    ci_upper: mean_tau + 1.96 * se,
                }
            })
            .collect();

        // ATE: each tree's own contribution is its mean tau over whichever
        // samples it could honestly estimate; the same IJ estimator applies
        // with that per-tree scalar taking the place of "tau_b(x_q)".
        let ate = effects.iter().map(|e| e.estimate).sum::<f64>() / n_samples as f64;
        let tree_ates: Vec<(f64, &[bool])> = tree_results
            .iter()
            .filter_map(|(tree_effects, _, in_bag)| {
                let vals: Vec<f64> = tree_effects.iter().filter_map(|&e| e).collect();
                if vals.is_empty() {
                    None
                } else {
                    Some((vals.iter().sum::<f64>() / vals.len() as f64, in_bag.as_slice()))
                }
            })
            .collect();
        let ate_se = infinitesimal_jackknife_se(&tree_ates, n_samples);

        // Feature importance
        let total_imp: f64 = total_importances.iter().sum();
        let feature_importance = if total_imp > 0.0 {
            feature_names
                .iter()
                .zip(&total_importances)
                .map(|(n, &v)| (n.clone(), v / total_imp))
                .collect()
        } else {
            feature_names.iter().map(|n| (n.clone(), 0.0)).collect()
        };

        Ok(CausalForestResult {
            effects,
            ate,
            ate_std_error: ate_se,
            feature_importance,
        })
    }
}

/// Infinitesimal jackknife variance for a forest-averaged statistic (Wager,
/// Hastie & Efron 2014; used for random forests by Wager & Athey 2018,
/// Sec. 4). `values` holds, for each tree that produced an estimate, its
/// value of the target quantity (a per-point tau or a per-tree mean tau for
/// the ATE) paired with that tree's in-bag indicator over all training
/// points.
///
/// This replaces `sqrt(between-tree variance / n_trees)`, which is NOT a
/// valid standard error for the forest average: it treats tree estimates as
/// independent draws of a fixed quantity, so it shrinks toward zero as more
/// trees are added (correctly reducing Monte Carlo noise in the ensemble
/// average) while saying nothing about the forest's actual sampling
/// uncertainty from having a finite training set — that uncertainty does
/// NOT vanish as n_trees grows. The IJ estimator instead uses the
/// covariance, across trees, between each training point's in-bag status
/// and the tree's prediction: points whose presence/absence in the
/// subsample systematically shifts the prediction contribute to genuine
/// sampling variance, and (unlike the naive formula) this does not collapse
/// to zero as n_trees grows.
///
/// Deliberately uses the *uncorrected* IJ estimator (Σ Cov_i², no
/// finite-B bias subtraction). Wager, Hastie & Efron's bootstrap
/// bias-correction term (n/B · between-tree variance) assumes bootstrap
/// (with-replacement) resampling; `CausalForest` subsamples without
/// replacement, for which that correction constant doesn't directly apply,
/// and empirically it dominates Σ Cov_i² for realistic forest sizes here --
/// clipping the corrected estimate to zero far more often than genuine
/// near-zero variance would warrant. The uncorrected estimator is always
/// non-negative by construction and has a well-known slight upward
/// (conservative) bias at finite B that shrinks as more trees are added; it
/// is the safer choice over silently zeroing out the uncertainty estimate.
fn infinitesimal_jackknife_se(values: &[(f64, &[bool])], n_samples: usize) -> f64 {
    if values.is_empty() {
        return f64::INFINITY;
    }
    let b = values.len() as f64;
    let mean: f64 = values.iter().map(|(v, _)| v).sum::<f64>() / b;

    let v_ij: f64 = (0..n_samples)
        .map(|train_point| {
            let n_bar: f64 = values
                .iter()
                .map(|(_, ib)| if ib[train_point] { 1.0 } else { 0.0 })
                .sum::<f64>()
                / b;
            let cov: f64 = values
                .iter()
                .map(|(v, ib)| {
                    let n_i = if ib[train_point] { 1.0 } else { 0.0 };
                    (n_i - n_bar) * (v - mean)
                })
                .sum::<f64>()
                / b;
            cov * cov
        })
        .sum();

    v_ij.sqrt()
}

// ── Causal tree internals ───────────────────────────────────────────

struct CausalNode {
    /// Leaf tau. Set from train_idx during construction (needed to choose
    /// splits), then overwritten with the honest estimate by
    /// `populate_leaf_tau`. Meaningless for non-leaf (Split) nodes -- only
    /// leaf values are ever read, via `find_leaf`.
    tau: f64,
    /// True once this leaf has been honestly re-estimated from `est_idx` AND
    /// that estimation sample contained at least one treated and one control
    /// unit. `find_leaf` callers must check this before trusting `tau`: a
    /// leaf that reaches no honest data is not a valid estimate, not zero.
    honest_valid: bool,
    /// Split info (None for leaf).
    split: Option<CausalSplit>,
}

struct CausalSplit {
    feature: usize,
    threshold: f64,
    left: Box<CausalNode>,
    right: Box<CausalNode>,
}

fn find_leaf<'a>(node: &'a CausalNode, sample: ArrayView1<f64>) -> &'a CausalNode {
    match &node.split {
        None => node,
        Some(split) => {
            if sample[split.feature] <= split.threshold {
                find_leaf(&split.left, sample)
            } else {
                find_leaf(&split.right, sample)
            }
        }
    }
}

/// Honest step: route `est_idx` down the fixed tree structure and recompute
/// each leaf's tau using only the estimation points that land there. The
/// tree's splits (built from `train_idx`) are never touched, so structure
/// and effect estimates come from disjoint samples throughout.
fn populate_leaf_tau(
    node: &mut CausalNode,
    features: &Array2<f64>,
    est_idx: &[usize],
    treatment: &[usize],
    outcome: &[f64],
) {
    match &mut node.split {
        None => {
            let (tau, n_treated, n_control) = estimate_tau(treatment, outcome, est_idx);
            node.honest_valid = n_treated >= 1 && n_control >= 1;
            if node.honest_valid {
                node.tau = tau;
            }
        }
        Some(split) => {
            let (feature, threshold) = (split.feature, split.threshold);
            let mut left_idx = Vec::new();
            let mut right_idx = Vec::new();
            for &i in est_idx {
                if features[[i, feature]] <= threshold {
                    left_idx.push(i);
                } else {
                    right_idx.push(i);
                }
            }
            populate_leaf_tau(&mut split.left, features, &left_idx, treatment, outcome);
            populate_leaf_tau(&mut split.right, features, &right_idx, treatment, outcome);
        }
    }
}

/// Estimate tau within a set of indices.
fn estimate_tau(treatment: &[usize], outcome: &[f64], indices: &[usize]) -> (f64, usize, usize) {
    let mut sum_treated = 0.0;
    let mut sum_control = 0.0;
    let mut n_treated = 0usize;
    let mut n_control = 0usize;

    for &i in indices {
        if treatment[i] == 1 {
            sum_treated += outcome[i];
            n_treated += 1;
        } else {
            sum_control += outcome[i];
            n_control += 1;
        }
    }

    let tau = if n_treated > 0 && n_control > 0 {
        sum_treated / n_treated as f64 - sum_control / n_control as f64
    } else {
        0.0
    };

    (tau, n_treated, n_control)
}

/// Build a single causal tree.
fn build_causal_tree(
    features: &Array2<f64>,
    treatment: &[usize],
    outcome: &[f64],
    indices: &[usize],
    n_features: usize,
    max_depth: Option<usize>,
    min_samples_leaf: usize,
    importances: &mut Vec<f64>,
    rng: &mut impl Rng,
) -> CausalNode {
    build_node(
        features,
        treatment,
        outcome,
        indices,
        n_features,
        max_depth,
        min_samples_leaf,
        0,
        importances,
        rng,
    )
}

fn build_node(
    features: &Array2<f64>,
    treatment: &[usize],
    outcome: &[f64],
    indices: &[usize],
    n_features: usize,
    max_depth: Option<usize>,
    min_samples_leaf: usize,
    depth: usize,
    importances: &mut Vec<f64>,
    rng: &mut impl Rng,
) -> CausalNode {
    let (tau, n_treated, n_control) = estimate_tau(treatment, outcome, indices);

    // Stopping conditions
    if indices.len() < 2 * min_samples_leaf
        || max_depth.is_some_and(|d| depth >= d)
        || n_treated < 2
        || n_control < 2
    {
        return CausalNode {
            tau,
            honest_valid: false,
            split: None,
        };
    }

    // Find best split: maximize heterogeneity in tau
    // Criterion: n_L * n_R / n² * (tau_L - tau_R)²
    let n = indices.len() as f64;
    let mut best_criterion = 0.0;
    let mut best_split: Option<(usize, f64, Vec<usize>, Vec<usize>)> = None;

    // Random feature subset (like RF)
    let n_try = ((n_features as f64).sqrt().ceil() as usize).max(1);
    let mut feat_indices: Vec<usize> = (0..n_features).collect();
    feat_indices.shuffle(rng);

    for &feat in &feat_indices[..n_try.min(n_features)] {
        let mut sorted: Vec<usize> = indices.to_vec();
        sorted.sort_by(|&a, &b| {
            features[[a, feat]]
                .partial_cmp(&features[[b, feat]])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for s in min_samples_leaf..(sorted.len().saturating_sub(min_samples_leaf)) {
            if (features[[sorted[s], feat]] - features[[sorted[s - 1], feat]]).abs() < f64::EPSILON
            {
                continue;
            }

            let left_idx = &sorted[..s];
            let right_idx = &sorted[s..];

            // Need both treated and control in each child
            let left_treated = left_idx.iter().filter(|&&i| treatment[i] == 1).count();
            let left_control = left_idx.len() - left_treated;
            let right_treated = right_idx.iter().filter(|&&i| treatment[i] == 1).count();
            let right_control = right_idx.len() - right_treated;

            if left_treated < 1 || left_control < 1 || right_treated < 1 || right_control < 1 {
                continue;
            }

            let (tau_l, _, _) = estimate_tau(treatment, outcome, left_idx);
            let (tau_r, _, _) = estimate_tau(treatment, outcome, right_idx);

            let n_l = left_idx.len() as f64;
            let n_r = right_idx.len() as f64;
            let criterion = (n_l * n_r / (n * n)) * (tau_l - tau_r).powi(2);

            if criterion > best_criterion {
                best_criterion = criterion;
                let threshold =
                    (features[[sorted[s - 1], feat]] + features[[sorted[s], feat]]) / 2.0;
                best_split = Some((feat, threshold, left_idx.to_vec(), right_idx.to_vec()));
            }
        }
    }

    match best_split {
        Some((feat, threshold, left_idx, right_idx)) => {
            importances[feat] += best_criterion * indices.len() as f64;

            let left = build_node(
                features,
                treatment,
                outcome,
                &left_idx,
                n_features,
                max_depth,
                min_samples_leaf,
                depth + 1,
                importances,
                rng,
            );
            let right = build_node(
                features,
                treatment,
                outcome,
                &right_idx,
                n_features,
                max_depth,
                min_samples_leaf,
                depth + 1,
                importances,
                rng,
            );

            CausalNode {
                tau,
                honest_valid: false,
                split: Some(CausalSplit {
                    feature: feat,
                    threshold,
                    left: Box::new(left),
                    right: Box::new(right),
                }),
            }
        }
        None => CausalNode {
            tau,
            honest_valid: false,
            split: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Regression test for the honest-splitting bug: `populate_leaf_tau` was
    /// a no-op, so leaf tau came entirely from `train_idx` (the same sample
    /// that chose where the splits are) -- the classic adaptive-estimation
    /// bias honest forests exist to avoid. This builds a two-leaf tree by
    /// hand with an obviously-wrong "training" tau in each leaf, then checks
    /// that `populate_leaf_tau` overwrites it with the value implied by the
    /// estimation sample alone.
    #[test]
    fn populate_leaf_tau_uses_only_estimation_indices() {
        let mut tree = CausalNode {
            tau: 999.0, // deliberately wrong "training" value
            honest_valid: false,
            split: Some(CausalSplit {
                feature: 0,
                threshold: 0.5,
                left: Box::new(CausalNode { tau: 999.0, honest_valid: false, split: None }),
                right: Box::new(CausalNode { tau: 999.0, honest_valid: false, split: None }),
            }),
        };

        // 4 estimation points: 2 route left (x<0.5), 2 route right (x>=0.5).
        let features = array![[0.0], [0.1], [0.9], [1.0]];
        // Left leaf: treated outcome 10, control outcome 0 -> honest tau = 10.
        // Right leaf: treated outcome 3, control outcome 1 -> honest tau = 2.
        let treatment = vec![1, 0, 1, 0];
        let outcome = vec![10.0, 0.0, 3.0, 1.0];
        let est_idx = vec![0, 1, 2, 3];

        populate_leaf_tau(&mut tree, &features, &est_idx, &treatment, &outcome);

        let split = tree.split.as_ref().unwrap();
        assert!(split.left.honest_valid);
        assert!(
            (split.left.tau - 10.0).abs() < 1e-9,
            "left leaf tau should be the honest estimation-sample value (10.0), got {} \
             (999.0 would mean the training-sample value leaked through)",
            split.left.tau
        );
        assert!(split.right.honest_valid);
        assert!(
            (split.right.tau - 2.0).abs() < 1e-9,
            "right leaf tau should be the honest estimation-sample value (2.0), got {}",
            split.right.tau
        );
    }

    /// A leaf with no treated (or no control) unit among the routed
    /// estimation points has no honest estimate and must be marked invalid,
    /// not silently default to 0.0 (a fabricated "no effect" claim).
    #[test]
    fn populate_leaf_tau_marks_leaf_invalid_without_both_groups() {
        let mut leaf = CausalNode { tau: 0.0, honest_valid: false, split: None };
        let features = array![[0.0], [0.1]];
        let treatment = vec![1, 1]; // both treated: no control reaches this leaf
        let outcome = vec![5.0, 6.0];
        let est_idx = vec![0, 1];

        populate_leaf_tau(&mut leaf, &features, &est_idx, &treatment, &outcome);

        assert!(!leaf.honest_valid, "a leaf with no honest control unit must not be marked valid");
    }

    /// End-to-end: samples whose leaf receives no honest estimation data are
    /// excluded from that tree's vote (not given a fabricated tau of 0), and
    /// a homogeneous true effect is still recovered reasonably by the ATE.
    #[test]
    fn honest_forest_recovers_constant_treatment_effect() {
        let n = 200;
        let mut rng_features = StdRng::seed_from_u64(11);
        let mut features_flat = Vec::with_capacity(n);
        let mut treatment = Vec::with_capacity(n);
        let mut outcome = Vec::with_capacity(n);
        for i in 0..n {
            let x: f64 = rng_features.random::<f64>() * 10.0;
            let t = i % 2; // balanced treatment assignment
            // Constant true effect of 5.0, no heterogeneity in x.
            let noise: f64 = (rng_features.random::<f64>() - 0.5) * 0.5;
            let y = 5.0 * t as f64 + x * 0.1 + noise;
            features_flat.push(x);
            treatment.push(t);
            outcome.push(y);
        }
        let features = Array2::from_shape_vec((n, 1), features_flat).unwrap();

        let cf = CausalForest::new().with_n_estimators(50).with_seed(7);
        let result = cf.estimate(&features, &treatment, &outcome, &["x".into()]).unwrap();

        assert!(
            (result.ate - 5.0).abs() < 1.0,
            "ATE should recover the constant true effect (5.0), got {}",
            result.ate
        );
    }

    /// Regression test for the in-sample auto-influence bug: `estimate()`
    /// used to aggregate a tree's tau for point `i` even when tree included
    /// `i` in its own subsample (train_idx or est_idx). If `i` was in
    /// est_idx, `i`'s own outcome directly fed the leaf tau then reported
    /// back to `i` itself. With single-leaf trees (`max_depth(0)`), an
    /// extreme-outlier treated unit's own outcome would dominate its own
    /// leaf's tau in every tree where it landed in est_idx, pulling its
    /// reported CATE far from the common effect all other units share. The
    /// fix restricts aggregation to out-of-bag trees only (trees where the
    /// point never appeared in the subsample at all), so a point's own
    /// outcome can never feed its own reported estimate.
    #[test]
    fn oob_aggregation_excludes_own_outcome_from_own_estimate() {
        let n = 40;
        let mut features = Array2::<f64>::zeros((n, 1));
        let mut treatment = vec![0usize; n];
        let mut outcome = vec![0.0f64; n];
        for i in 0..n {
            features[[i, 0]] = i as f64; // irrelevant covariate: single-leaf trees anyway
            if i % 2 == 1 {
                treatment[i] = 1;
                outcome[i] = 1.0; // common effect ~= 1.0
            }
        }
        let outlier_idx = 1; // a treated unit
        outcome[outlier_idx] = 1_000_000.0; // extreme outlier, own outcome only

        let cf = CausalForest::new()
            .with_n_estimators(300)
            .with_max_depth(0)
            .with_seed(7);
        let result = cf
            .estimate(&features, &treatment, &outcome, &["x".to_string()])
            .unwrap();

        let outlier_effect = result.effects[outlier_idx].estimate;
        assert!(
            (outlier_effect - 1.0).abs() < 5.0,
            "outlier's own reported CATE should reflect the common effect (~1.0) from \
             out-of-bag trees, not be dragged toward its own 1e6 outcome via in-bag \
             self-influence: got {outlier_effect}"
        );
    }

    /// Regression test for the invalid SE: `sqrt(between-tree variance /
    /// n_trees)` treats tree estimates as independent draws of a fixed
    /// quantity, so it shrinks toward zero as n_estimators grows -- exactly
    /// like `sqrt(B_small/B_large)` -- which says the forest's sampling
    /// uncertainty vanishes simply by adding more trees. It doesn't. The IJ
    /// estimator used here still carries a well-known finite-B upward bias
    /// (each per-point covariance is itself a noisy sample statistic across
    /// B trees, and squaring+summing noisy estimates inflates the result at
    /// small B), so it isn't exactly B-invariant either -- but it should
    /// measurably beat the old formula's exact 1/sqrt(B) decay, not merely
    /// match it.
    #[test]
    fn ate_std_error_does_not_vanish_as_n_estimators_grows() {
        let n = 150;
        let mut rng = StdRng::seed_from_u64(21);
        let mut features_flat = Vec::with_capacity(n);
        let mut treatment = Vec::with_capacity(n);
        let mut outcome = Vec::with_capacity(n);
        for i in 0..n {
            let x: f64 = rng.random::<f64>() * 10.0;
            let t = i % 2;
            let noise: f64 = (rng.random::<f64>() - 0.5) * 2.0;
            let y = 3.0 * t as f64 + x * 0.2 + noise;
            features_flat.push(x);
            treatment.push(t);
            outcome.push(y);
        }
        let features = Array2::from_shape_vec((n, 1), features_flat).unwrap();

        let small = CausalForest::new()
            .with_n_estimators(20)
            .with_seed(3)
            .estimate(&features, &treatment, &outcome, &["x".into()])
            .unwrap();
        let large = CausalForest::new()
            .with_n_estimators(300)
            .with_seed(3)
            .estimate(&features, &treatment, &outcome, &["x".into()])
            .unwrap();

        assert!(small.ate_std_error.is_finite() && small.ate_std_error > 0.0);
        assert!(large.ate_std_error.is_finite() && large.ate_std_error > 0.0);

        let ratio = large.ate_std_error / small.ate_std_error;
        let old_buggy_ratio = (20.0_f64 / 300.0).sqrt();
        assert!(
            ratio > old_buggy_ratio * 1.15,
            "IJ SE should measurably beat the old formula's exact 1/sqrt(B) decay: \
             small-B SE={:.4}, large-B SE={:.4}, ratio={:.3} \
             (old sqrt(var/B) bug predicts ratio~{:.3}; got only {:.3} -- too close to the bug)",
            small.ate_std_error,
            large.ate_std_error,
            ratio,
            old_buggy_ratio,
            ratio
        );
    }

    /// Same non-vanishing property at the per-unit (CATE) level, not just
    /// the ATE.
    #[test]
    fn per_unit_std_error_does_not_vanish_as_n_estimators_grows() {
        let n = 150;
        let mut rng = StdRng::seed_from_u64(23);
        let mut features_flat = Vec::with_capacity(n);
        let mut treatment = Vec::with_capacity(n);
        let mut outcome = Vec::with_capacity(n);
        for i in 0..n {
            let x: f64 = rng.random::<f64>() * 10.0;
            let t = i % 2;
            let noise: f64 = (rng.random::<f64>() - 0.5) * 2.0;
            let y = 3.0 * t as f64 + x * 0.2 + noise;
            features_flat.push(x);
            treatment.push(t);
            outcome.push(y);
        }
        let features = Array2::from_shape_vec((n, 1), features_flat).unwrap();

        let small = CausalForest::new()
            .with_n_estimators(20)
            .with_seed(4)
            .estimate(&features, &treatment, &outcome, &["x".into()])
            .unwrap();
        let large = CausalForest::new()
            .with_n_estimators(300)
            .with_seed(4)
            .estimate(&features, &treatment, &outcome, &["x".into()])
            .unwrap();

        let mean_se_small: f64 =
            small.effects.iter().map(|e| e.std_error).sum::<f64>() / n as f64;
        let mean_se_large: f64 =
            large.effects.iter().map(|e| e.std_error).sum::<f64>() / n as f64;

        let ratio = mean_se_large / mean_se_small;
        let old_buggy_ratio = (20.0_f64 / 300.0).sqrt();
        assert!(
            ratio > old_buggy_ratio * 1.15,
            "mean per-unit SE should measurably beat the old formula's exact 1/sqrt(B) decay: \
             small-B mean SE={mean_se_small:.4}, large-B mean SE={mean_se_large:.4}, ratio={ratio:.3} \
             (old bug predicts ratio~{old_buggy_ratio:.3})"
        );
    }
}
