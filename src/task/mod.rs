//! Tasks define the problem: data + target + type (classification/regression).
//!
//! Inspired by mlr3's `Task` system.

use crate::{Result, SmeltError};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
// validate module used by learner predict methods

/// Type of a feature column.
///
/// Learners that understand categorical features (the boosting engines) read
/// this to choose categorical split finding / target encoding; every other
/// learner treats the integer codes as ordinary numeric values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureType {
    /// Continuous numeric feature (the default).
    Numeric,
    /// Categorical feature stored as integer codes `0..n_categories`.
    /// NaN is allowed and means "missing category".
    Categorical {
        /// Number of distinct categories (max code + 1).
        n_categories: usize,
    },
}

/// Validate that `columns` can be marked categorical in `features` and return
/// the per-column feature types: codes must be non-negative integers (NaN is
/// allowed as missing).
fn build_feature_types(features: &Array2<f64>, columns: &[usize]) -> Result<Vec<FeatureType>> {
    let mut types = vec![FeatureType::Numeric; features.ncols()];
    for &col in columns {
        if col >= features.ncols() {
            return Err(SmeltError::InvalidParameter(format!(
                "categorical feature index {col} out of range ({} features)",
                features.ncols()
            )));
        }
        let mut max_code = 0usize;
        for (row, &v) in features.column(col).iter().enumerate() {
            if v.is_nan() {
                continue;
            }
            if v < 0.0 || v.fract() != 0.0 {
                return Err(SmeltError::InvalidParameter(format!(
                    "categorical feature {col} has non-integer or negative code {v} at row {row}"
                )));
            }
            max_code = max_code.max(v as usize);
        }
        types[col] = FeatureType::Categorical {
            n_categories: max_code + 1,
        };
    }
    Ok(types)
}

/// Validate sample weights for `with_weights`: the length must match the
/// number of samples, every weight must be finite and `>= 0`, and at least
/// one must be `> 0` (an individual weight of exactly `0.0` is valid and
/// means "exclude this sample from training").
///
/// Panics with a clear message on the first violation — see
/// `with_weights`'s `# Panics` section for why this is a panic rather than
/// a `Result`.
fn validate_weights(weights: &[f64], n_samples: usize) {
    assert!(
        weights.len() == n_samples,
        "with_weights: {} weight(s) provided for {} sample(s) — one weight per sample is required",
        weights.len(),
        n_samples
    );
    for (i, &w) in weights.iter().enumerate() {
        assert!(
            w.is_finite(),
            "with_weights: weight at index {i} is {w} — all sample weights must be finite"
        );
        assert!(
            w >= 0.0,
            "with_weights: weight at index {i} is {w} — sample weights must be >= 0"
        );
    }
    assert!(
        weights.iter().any(|&w| w > 0.0),
        "with_weights: all {} weights are zero — at least one sample must have positive weight",
        weights.len()
    );
}

/// Core trait for all task types.
pub trait Task {
    /// Task identifier.
    fn id(&self) -> &str;
    /// Feature matrix (n_samples × n_features).
    fn features(&self) -> &Array2<f64>;
    /// Number of samples.
    fn n_samples(&self) -> usize {
        self.features().nrows()
    }
    /// Number of features.
    fn n_features(&self) -> usize {
        self.features().ncols()
    }
    /// Feature names.
    fn feature_names(&self) -> &[String];
    /// Per-column feature types (Numeric unless marked categorical).
    fn feature_types(&self) -> &[FeatureType];
    /// Indices of the categorical feature columns.
    fn categorical_features(&self) -> Vec<usize> {
        self.feature_types()
            .iter()
            .enumerate()
            .filter(|(_, t)| matches!(t, FeatureType::Categorical { .. }))
            .map(|(i, _)| i)
            .collect()
    }
}

/// Classification task with discrete target labels.
#[derive(Debug)]
pub struct ClassificationTask {
    id: String,
    features: Array2<f64>,
    target: Vec<usize>,
    feature_names: Vec<String>,
    class_names: Vec<String>,
    feature_types: Vec<FeatureType>,
    weights: Option<Vec<f64>>,
}

impl ClassificationTask {
    /// Create a classification task from a feature matrix and integer class
    /// labels; class names default to `"class_0"`, `"class_1"`, etc.
    pub fn new(id: impl Into<String>, features: Array2<f64>, target: Vec<usize>) -> Result<Self> {
        if features.nrows() == 0 {
            return Err(SmeltError::EmptyDataset);
        }
        if features.nrows() != target.len() {
            return Err(SmeltError::DimensionMismatch {
                expected: features.nrows(),
                got: target.len(),
            });
        }
        if features.ncols() == 0 {
            return Err(SmeltError::InvalidParameter(
                "features must have at least 1 column".into(),
            ));
        }
        let n_features = features.ncols();
        let n_classes = target.iter().copied().max().unwrap_or(0) + 1;
        Ok(Self {
            id: id.into(),
            feature_names: (0..n_features).map(|i| format!("x{i}")).collect(),
            feature_types: vec![FeatureType::Numeric; n_features],
            features,
            target,
            class_names: (0..n_classes).map(|i| format!("class_{i}")).collect(),
            weights: None,
        })
    }

    /// Mark feature columns as categorical. Values in those columns must be
    /// non-negative integer codes (NaN allowed as missing category).
    ///
    /// Recomputes `n_categories` per column from the codes actually present
    /// in `self.features` (the max code + 1). If this task is a subset of a
    /// larger dataset (e.g. a CV fold), this can under-count categories the
    /// subset happens not to contain — use [`Self::with_feature_types`] to
    /// copy the exact types (and `n_categories`) from the parent task instead.
    pub fn with_categorical_features(mut self, columns: &[usize]) -> Result<Self> {
        self.feature_types = build_feature_types(&self.features, columns)?;
        Ok(self)
    }

    /// Set feature types directly (e.g. copied from another task's
    /// [`Task::feature_types`]), bypassing the from-data recomputation
    /// `with_categorical_features` does. Must match the number of feature
    /// columns.
    pub fn with_feature_types(mut self, types: Vec<FeatureType>) -> Result<Self> {
        if types.len() != self.features.ncols() {
            return Err(SmeltError::DimensionMismatch {
                expected: self.features.ncols(),
                got: types.len(),
            });
        }
        self.feature_types = types;
        Ok(self)
    }

    /// Set custom feature names; must match the number of feature columns.
    pub fn with_feature_names(mut self, names: Vec<String>) -> Result<Self> {
        if names.len() != self.features.ncols() {
            return Err(SmeltError::DimensionMismatch {
                expected: self.features.ncols(),
                got: names.len(),
            });
        }
        self.feature_names = names;
        Ok(self)
    }

    /// Set custom class names, indexed by class label. Providing MORE names
    /// than observed labels is valid and useful (it widens `n_classes` so a
    /// subset/fold task keeps the full class set of its parent — how
    /// Pipeline/Stacking/resamplers propagate names); providing FEWER names
    /// than the highest label + 1 is never valid, since `n_classes()` is
    /// `class_names.len()` and every downstream probability row / class
    /// count would be too narrow.
    ///
    /// # Panics
    ///
    /// Panics immediately if `names` has fewer entries than the highest
    /// target label + 1 — before this check, the mismatch surfaced later as
    /// an opaque index-out-of-bounds panic deep inside whatever consumed
    /// `n_classes()` first (e.g. SMOTE's per-class grouping). Kept as a
    /// panic rather than `Result` because changing the signature would be a
    /// breaking API change (5th audit, LOW-C).
    pub fn with_class_names(mut self, names: Vec<String>) -> Self {
        let required = self.target.iter().copied().max().map_or(0, |m| m + 1);
        assert!(
            names.len() >= required,
            "with_class_names: {} class name(s) provided, but the target's highest label is {} \
             so at least {} name(s) are required",
            names.len(),
            required.saturating_sub(1),
            required
        );
        self.class_names = names;
        self
    }

    /// Attach per-sample weights, one per sample.
    ///
    /// Weights are FREQUENCY / relative-importance weights for **training**:
    /// a sample with weight `k` should influence a weight-aware learner's
    /// fit like `k` copies of that row, and a weight of `0.0` excludes the
    /// sample. They do not affect prediction or measures (for now). No
    /// learner consumes them yet — every learner currently rejects a
    /// weighted task with a clear error via
    /// [`crate::validate::check_no_weights`] rather than silently ignoring
    /// the weights; weight-aware learners land in a later phase.
    ///
    /// # Panics
    ///
    /// Panics immediately (same precedent as [`Self::with_class_names`]:
    /// changing an existing builder chain to `Result` would be a breaking
    /// API change, and an invalid weight vector is a programming error, not
    /// a data condition) if:
    /// - `weights.len() != n_samples`
    /// - any weight is NaN or ±infinity
    /// - any weight is negative
    /// - **all** weights are zero (an individual `0.0` is valid = sample
    ///   excluded; a task where every sample is excluded is not).
    ///
    /// Note for fold slicing: a subset of a validly-weighted task can be
    /// all-zero (every positively-weighted row landed in the other folds);
    /// re-attaching such a slice panics with the same message, which is the
    /// honest outcome — training on an all-zero-weight fold is undefined.
    pub fn with_weights(mut self, weights: Vec<f64>) -> Self {
        validate_weights(&weights, self.features.nrows());
        self.weights = Some(weights);
        self
    }

    /// Per-sample training weights, if any were attached via
    /// [`Self::with_weights`]. `None` means the unweighted default (every
    /// sample counts once).
    pub fn weights(&self) -> Option<&[f64]> {
        self.weights.as_deref()
    }

    /// Integer target labels, one per sample.
    pub fn target(&self) -> &[usize] {
        &self.target
    }
    /// Number of distinct classes.
    pub fn n_classes(&self) -> usize {
        self.class_names.len()
    }
    /// Class names, indexed by class label.
    pub fn class_names(&self) -> &[String] {
        &self.class_names
    }
}

impl Task for ClassificationTask {
    fn id(&self) -> &str {
        &self.id
    }
    fn features(&self) -> &Array2<f64> {
        &self.features
    }
    fn feature_names(&self) -> &[String] {
        &self.feature_names
    }
    fn feature_types(&self) -> &[FeatureType] {
        &self.feature_types
    }
}

/// Regression task with continuous target values.
#[derive(Debug)]
pub struct RegressionTask {
    id: String,
    features: Array2<f64>,
    target: Vec<f64>,
    feature_names: Vec<String>,
    feature_types: Vec<FeatureType>,
    weights: Option<Vec<f64>>,
}

impl RegressionTask {
    /// Create a regression task from a feature matrix and continuous target
    /// values.
    pub fn new(id: impl Into<String>, features: Array2<f64>, target: Vec<f64>) -> Result<Self> {
        if features.nrows() == 0 {
            return Err(SmeltError::EmptyDataset);
        }
        if features.nrows() != target.len() {
            return Err(SmeltError::DimensionMismatch {
                expected: features.nrows(),
                got: target.len(),
            });
        }
        if features.ncols() == 0 {
            return Err(SmeltError::InvalidParameter(
                "features must have at least 1 column".into(),
            ));
        }
        let n_features = features.ncols();
        Ok(Self {
            id: id.into(),
            feature_names: (0..n_features).map(|i| format!("x{i}")).collect(),
            feature_types: vec![FeatureType::Numeric; n_features],
            features,
            target,
            weights: None,
        })
    }

    /// Mark feature columns as categorical. Values in those columns must be
    /// non-negative integer codes (NaN allowed as missing category).
    ///
    /// Recomputes `n_categories` per column from the codes actually present
    /// in `self.features` (the max code + 1). If this task is a subset of a
    /// larger dataset (e.g. a CV fold), this can under-count categories the
    /// subset happens not to contain — use [`Self::with_feature_types`] to
    /// copy the exact types (and `n_categories`) from the parent task instead.
    pub fn with_categorical_features(mut self, columns: &[usize]) -> Result<Self> {
        self.feature_types = build_feature_types(&self.features, columns)?;
        Ok(self)
    }

    /// Set feature types directly (e.g. copied from another task's
    /// [`Task::feature_types`]), bypassing the from-data recomputation
    /// `with_categorical_features` does. Must match the number of feature
    /// columns.
    pub fn with_feature_types(mut self, types: Vec<FeatureType>) -> Result<Self> {
        if types.len() != self.features.ncols() {
            return Err(SmeltError::DimensionMismatch {
                expected: self.features.ncols(),
                got: types.len(),
            });
        }
        self.feature_types = types;
        Ok(self)
    }

    /// Set custom feature names; must match the number of feature columns.
    pub fn with_feature_names(mut self, names: Vec<String>) -> Result<Self> {
        if names.len() != self.features.ncols() {
            return Err(SmeltError::DimensionMismatch {
                expected: self.features.ncols(),
                got: names.len(),
            });
        }
        self.feature_names = names;
        Ok(self)
    }

    /// Attach per-sample weights, one per sample.
    ///
    /// Weights are FREQUENCY / relative-importance weights for **training**:
    /// a sample with weight `k` should influence a weight-aware learner's
    /// fit like `k` copies of that row, and a weight of `0.0` excludes the
    /// sample. They do not affect prediction or measures (for now). No
    /// learner consumes them yet — every learner currently rejects a
    /// weighted task with a clear error via
    /// [`crate::validate::check_no_weights`] rather than silently ignoring
    /// the weights; weight-aware learners land in a later phase.
    ///
    /// # Panics
    ///
    /// Panics immediately (same precedent as
    /// [`ClassificationTask::with_class_names`]: changing an existing
    /// builder chain to `Result` would be a breaking API change, and an
    /// invalid weight vector is a programming error, not a data condition)
    /// if:
    /// - `weights.len() != n_samples`
    /// - any weight is NaN or ±infinity
    /// - any weight is negative
    /// - **all** weights are zero (an individual `0.0` is valid = sample
    ///   excluded; a task where every sample is excluded is not).
    ///
    /// Note for fold slicing: a subset of a validly-weighted task can be
    /// all-zero (every positively-weighted row landed in the other folds);
    /// re-attaching such a slice panics with the same message, which is the
    /// honest outcome — training on an all-zero-weight fold is undefined.
    pub fn with_weights(mut self, weights: Vec<f64>) -> Self {
        validate_weights(&weights, self.features.nrows());
        self.weights = Some(weights);
        self
    }

    /// Per-sample training weights, if any were attached via
    /// [`Self::with_weights`]. `None` means the unweighted default (every
    /// sample counts once).
    pub fn weights(&self) -> Option<&[f64]> {
        self.weights.as_deref()
    }

    /// Continuous target values, one per sample.
    pub fn target(&self) -> &[f64] {
        &self.target
    }
}

impl Task for RegressionTask {
    fn id(&self) -> &str {
        &self.id
    }
    fn features(&self) -> &Array2<f64> {
        &self.features
    }
    fn feature_names(&self) -> &[String] {
        &self.feature_names
    }
    fn feature_types(&self) -> &[FeatureType] {
        &self.feature_types
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn classif_task() -> ClassificationTask {
        let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        ClassificationTask::new("t", features, vec![0, 1, 0, 1]).unwrap()
    }

    fn regress_task() -> RegressionTask {
        let features = array![[1.0], [2.0], [3.0], [4.0]];
        RegressionTask::new("t", features, vec![1.0, 2.0, 3.0, 4.0]).unwrap()
    }

    #[test]
    fn weights_default_to_none() {
        assert_eq!(classif_task().weights(), None);
        assert_eq!(regress_task().weights(), None);
    }

    #[test]
    fn with_weights_attaches_and_accessor_returns_them() {
        let w = vec![1.0, 0.5, 2.0, 0.25];
        let ct = classif_task().with_weights(w.clone());
        assert_eq!(ct.weights(), Some(w.as_slice()));
        let rt = regress_task().with_weights(w.clone());
        assert_eq!(rt.weights(), Some(w.as_slice()));
    }

    #[test]
    fn an_individual_zero_weight_is_valid() {
        // weight 0 = "sample excluded" — only ALL-zero is invalid
        let ct = classif_task().with_weights(vec![0.0, 1.0, 0.0, 1.0]);
        assert_eq!(ct.weights(), Some([0.0, 1.0, 0.0, 1.0].as_slice()));
    }

    #[test]
    #[should_panic(expected = "one weight per sample is required")]
    fn with_weights_panics_on_length_mismatch() {
        classif_task().with_weights(vec![1.0, 1.0, 1.0]);
    }

    #[test]
    #[should_panic(expected = "must be finite")]
    fn with_weights_panics_on_nan() {
        classif_task().with_weights(vec![1.0, f64::NAN, 1.0, 1.0]);
    }

    #[test]
    #[should_panic(expected = "must be finite")]
    fn with_weights_panics_on_infinity() {
        regress_task().with_weights(vec![1.0, f64::INFINITY, 1.0, 1.0]);
    }

    #[test]
    #[should_panic(expected = "must be >= 0")]
    fn with_weights_panics_on_negative_weight() {
        regress_task().with_weights(vec![1.0, -0.5, 1.0, 1.0]);
    }

    #[test]
    #[should_panic(expected = "at least one sample must have positive weight")]
    fn with_weights_panics_when_all_weights_are_zero() {
        classif_task().with_weights(vec![0.0, 0.0, 0.0, 0.0]);
    }
}
