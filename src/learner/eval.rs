//! Shared held-out eval-set support for early stopping in the boosting engines.
//!
//! Training loss under boosting is (near-)monotonically decreasing, so early
//! stopping that monitors it rarely plateaus and almost never actually fires.
//! Each engine exposes `with_eval_set_regress/_classif` builders storing an
//! `EvalSet`; the validators below check dimensions and task-type coherence at
//! train time.

use crate::{Result, SmeltError};
use ndarray::Array2;

/// Held-out target for early-stopping evaluation, paired with the eval set's
/// features. Must match the task type the model is trained on.
pub(crate) enum EvalTarget {
    Regression(Vec<f64>),
    Classification(Vec<usize>),
}

pub(crate) type EvalSet = Option<(Array2<f64>, EvalTarget)>;

/// Validate an eval set for a regression task and return the
/// (features, target) pair to evaluate on for early stopping, if set.
pub(crate) fn validate_eval_regress(
    eval_set: &EvalSet,
    n_features: usize,
) -> Result<Option<(&Array2<f64>, &[f64])>> {
    match eval_set {
        None => Ok(None),
        Some((ef, EvalTarget::Regression(et))) => {
            if ef.ncols() != n_features {
                return Err(SmeltError::DimensionMismatch {
                    expected: n_features,
                    got: ef.ncols(),
                });
            }
            if et.len() != ef.nrows() {
                return Err(SmeltError::DimensionMismatch {
                    expected: ef.nrows(),
                    got: et.len(),
                });
            }
            Ok(Some((ef, et.as_slice())))
        }
        Some((_, EvalTarget::Classification(_))) => Err(SmeltError::InvalidParameter(
            "eval_set was set via with_eval_set_classif but the model is training a regression task".into(),
        )),
    }
}

/// Validate an eval set for a classification task and return the
/// (features, target) pair to evaluate on for early stopping, if set.
pub(crate) fn validate_eval_classif(
    eval_set: &EvalSet,
    n_features: usize,
) -> Result<Option<(&Array2<f64>, &[usize])>> {
    match eval_set {
        None => Ok(None),
        Some((ef, EvalTarget::Classification(et))) => {
            if ef.ncols() != n_features {
                return Err(SmeltError::DimensionMismatch {
                    expected: n_features,
                    got: ef.ncols(),
                });
            }
            if et.len() != ef.nrows() {
                return Err(SmeltError::DimensionMismatch {
                    expected: ef.nrows(),
                    got: et.len(),
                });
            }
            Ok(Some((ef, et.as_slice())))
        }
        Some((_, EvalTarget::Regression(_))) => Err(SmeltError::InvalidParameter(
            "eval_set was set via with_eval_set_regress but the model is training a classification task".into(),
        )),
    }
}

/// Early-stopping state: tracks the best loss seen and how many rounds have
/// passed without improvement. `update` returns `Some(best_n_trees)` when
/// training should stop (truncate to that many trees and break).
pub(crate) struct EarlyStopper {
    rounds: usize,
    best_loss: f64,
    no_improve: usize,
    best_n: usize,
}

impl EarlyStopper {
    /// `rounds == 0` disables early stopping (update always returns None
    /// and `is_active()` is false).
    pub(crate) fn new(rounds: usize) -> Self {
        Self {
            rounds,
            best_loss: f64::INFINITY,
            no_improve: 0,
            best_n: 0,
        }
    }

    pub(crate) fn is_active(&self) -> bool {
        self.rounds > 0
    }

    /// Record this round's loss; `n_trees` is the tree count that a later
    /// truncate should keep if this round turns out to be the best.
    pub(crate) fn update(&mut self, loss: f64, n_trees: usize) -> Option<usize> {
        if loss < self.best_loss - 1e-10 {
            self.best_loss = loss;
            self.best_n = n_trees;
            self.no_improve = 0;
            None
        } else {
            self.no_improve += 1;
            if self.no_improve >= self.rounds {
                Some(self.best_n)
            } else {
                None
            }
        }
    }
}
