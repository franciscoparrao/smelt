//! Queryable, declarative capability metadata for [`Learner`]s
//! (`Learner::properties`), the smelt analogue of mlr3's `Learner$properties`.
//!
//! Every field describes a **behaviour observable from outside the learner**,
//! not an internal implementation note. The contract autotest
//! (`tests/contract.rs`) constructs each registered learner and checks that
//! the declared value matches what the learner actually does, so a field that
//! lies is a test failure, not silent misinformation. When you add a learner
//! (or change one's behaviour), update its [`Learner::properties`] override and
//! the autotest will hold you to it.
//!
//! [`Learner`]: crate::learner::Learner
//! [`Learner::properties`]: crate::learner::Learner::properties

/// Declarative capability flags for a [`Learner`](crate::learner::Learner).
///
/// `Copy`/`Clone`/`Eq` so it can be compared, stored in tables, and returned
/// by value cheaply. Build one with the fluent constructors
/// ([`classifier`](LearnerProperties::classifier),
/// [`regressor`](LearnerProperties::regressor),
/// [`classifier_regressor`](LearnerProperties::classifier_regressor),
/// [`none`](LearnerProperties::none)) and the `with_*` setters, e.g.
/// `LearnerProperties::classifier_regressor().with_weights().with_proba()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LearnerProperties {
    /// `true` iff [`Learner::train_classif`] trains successfully (rather than
    /// returning the default "does not support classification" error).
    ///
    /// Observable: a trivial [`ClassificationTask`] trains without error.
    ///
    /// [`Learner::train_classif`]: crate::learner::Learner::train_classif
    /// [`ClassificationTask`]: crate::task::ClassificationTask
    pub supports_classification: bool,

    /// `true` iff [`Learner::train_regress`] trains successfully (rather than
    /// returning the default "does not support regression" error).
    ///
    /// Observable: a trivial [`RegressionTask`] trains without error.
    ///
    /// [`Learner::train_regress`]: crate::learner::Learner::train_regress
    /// [`RegressionTask`]: crate::task::RegressionTask
    pub supports_regression: bool,

    /// `true` iff the learner consumes per-sample weights during training.
    ///
    /// Single source of truth for weight support: the default
    /// [`Learner::supports_weights`] method returns *this* field, so the two
    /// can never disagree (see the module docs on `supports_weights`).
    ///
    /// Observable: a weighted task trains without error when `true`; when
    /// `false`, the learner's [`check_no_weights`] guard rejects a weighted
    /// task with [`SmeltError::InvalidParameter`].
    ///
    /// [`Learner::supports_weights`]: crate::learner::Learner::supports_weights
    /// [`check_no_weights`]: crate::validate::check_no_weights
    /// [`SmeltError::InvalidParameter`]: crate::SmeltError::InvalidParameter
    pub supports_weights: bool,

    /// `true` iff a classification prediction carries a real per-class
    /// probability vector (`Prediction::Classification { probabilities:
    /// Some(..), .. }`) rather than only hard labels (`probabilities: None`).
    ///
    /// Observable: after `train_classif`, `predict` returns
    /// `probabilities: Some(rows)` with each row of width `n_classes`, summing
    /// to ~1, and whose argmax equals the hard label. Note that *fractional*
    /// probabilities are data-dependent — a pure decision-tree leaf legitimately
    /// yields a one-hot row on separable data — so the contract checks "emits a
    /// valid distribution consistent with the label", not "always fractional".
    /// Meaningless (and set `false`) for regression-only learners.
    pub supports_proba: bool,

    /// `true` iff the learner accepts `NaN` in the feature matrix (the
    /// gradient-boosting engines, which route missing values through a learned
    /// default direction) rather than rejecting it up front.
    ///
    /// Observable: `false` learners call [`check_no_nan`] and reject a task
    /// with a `NaN` feature via [`SmeltError::InvalidParameter`]; `true`
    /// learners train on it without error.
    ///
    /// [`check_no_nan`]: crate::validate::check_no_nan
    /// [`SmeltError::InvalidParameter`]: crate::SmeltError::InvalidParameter
    pub supports_nan: bool,

    /// `true` iff the learner reads [`Task`] feature-type metadata to perform
    /// native categorical splits (XGBoost/LightGBM one-hot/Fisher grouping,
    /// CatBoost ordered target statistics) rather than treating every column as
    /// numeric.
    ///
    /// Verified by code inspection (the autotest does not construct a
    /// categorical task — that surface is exercised by each engine's own
    /// integration tests); declared here so callers can query it.
    ///
    /// [`Task`]: crate::task::Task
    pub supports_categorical: bool,

    /// `true` iff the trained model's [`TrainedModel::feature_importance`]
    /// returns `Some` after a normal fit.
    ///
    /// Observable: after training on data that induces at least one split /
    /// non-degenerate fit, `feature_importance()` is `Some` with exactly
    /// `n_features` entries **in training-column order** (the positional
    /// contract [`Rfe`](crate::preprocess::Rfe) relies on).
    ///
    /// [`TrainedModel::feature_importance`]: crate::learner::TrainedModel::feature_importance
    pub provides_feature_importance: bool,

    /// `true` iff the trained model's [`TrainedModel::to_serializable`] returns
    /// `Some` (i.e. it has a [`SerializableModel`] variant), so the model can be
    /// round-tripped through `serialize.rs`. The `Box<dyn TrainedModel>`-holding
    /// composites return `None`.
    ///
    /// Observable: after training, `to_serializable()` is `Some`.
    ///
    /// [`TrainedModel::to_serializable`]: crate::learner::TrainedModel::to_serializable
    /// [`SerializableModel`]: crate::serialize::SerializableModel
    pub serializable: bool,
}

impl LearnerProperties {
    /// All-`false` properties — the most conservative baseline and the trait
    /// default. A learner that forgets to override [`Learner::properties`]
    /// therefore *under*-claims (fails the contract autotest loudly) rather
    /// than silently over-claiming a capability it lacks.
    ///
    /// [`Learner::properties`]: crate::learner::Learner::properties
    pub const fn none() -> Self {
        Self {
            supports_classification: false,
            supports_regression: false,
            supports_weights: false,
            supports_proba: false,
            supports_nan: false,
            supports_categorical: false,
            provides_feature_importance: false,
            serializable: false,
        }
    }

    /// Classification-only baseline (`supports_classification = true`, rest
    /// `false`).
    pub const fn classifier() -> Self {
        Self {
            supports_classification: true,
            ..Self::none()
        }
    }

    /// Regression-only baseline (`supports_regression = true`, rest `false`).
    pub const fn regressor() -> Self {
        Self {
            supports_regression: true,
            ..Self::none()
        }
    }

    /// Dual-task baseline (both `supports_classification` and
    /// `supports_regression` `true`, rest `false`).
    pub const fn classifier_regressor() -> Self {
        Self {
            supports_classification: true,
            supports_regression: true,
            ..Self::none()
        }
    }

    /// Marks weight support (see [`supports_weights`](Self::supports_weights)).
    pub const fn with_weights(mut self) -> Self {
        self.supports_weights = true;
        self
    }

    /// Marks probability output (see [`supports_proba`](Self::supports_proba)).
    pub const fn with_proba(mut self) -> Self {
        self.supports_proba = true;
        self
    }

    /// Marks `NaN` tolerance (see [`supports_nan`](Self::supports_nan)).
    pub const fn with_nan(mut self) -> Self {
        self.supports_nan = true;
        self
    }

    /// Marks native categorical-split support (see
    /// [`supports_categorical`](Self::supports_categorical)).
    pub const fn with_categorical(mut self) -> Self {
        self.supports_categorical = true;
        self
    }

    /// Marks feature-importance availability (see
    /// [`provides_feature_importance`](Self::provides_feature_importance)).
    pub const fn with_feature_importance(mut self) -> Self {
        self.provides_feature_importance = true;
        self
    }

    /// Marks serializability (see [`serializable`](Self::serializable)).
    pub const fn with_serializable(mut self) -> Self {
        self.serializable = true;
        self
    }
}
