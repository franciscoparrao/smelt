//! Shared out-of-fold (OOF) cross-fitting helpers for R-learner/DR-learner.
//!
//! Generalizes the OOF loop `Stacking::train_regress` already uses for its
//! meta-features: for every `(train_idx, test_idx)` fold from
//! `CrossValidation::splits`, fit on the training rows and predict on the
//! held-out rows, so no unit is ever predicted by a model that saw it
//! during training -- the same discipline that gives R-learner/DR-learner
//! their "quasi-oracle"/doubly-robust bias-removal property.

use super::LearnerFactory;
use crate::prediction::Prediction;
use crate::resample::{CrossValidation, Resample};
use crate::task::{ClassificationTask, RegressionTask};
use crate::{Result, SmeltError};
use ndarray::{Array2, Axis};

/// Out-of-fold regression predictions for every row.
pub(crate) fn oof_regression(
    features: &Array2<f64>,
    target: &[f64],
    factory: &LearnerFactory,
    folds: usize,
    seed: u64,
) -> Result<Vec<f64>> {
    let n = features.nrows();
    let splits = CrossValidation::new(folds).with_seed(seed).splits(n)?;
    let mut out = vec![0.0; n];
    for (train_idx, test_idx) in &splits {
        let train_features = features.select(Axis(0), train_idx);
        let train_target: Vec<f64> = train_idx.iter().map(|&i| target[i]).collect();
        let train_task = RegressionTask::new("oof_regression", train_features, train_target)?;
        let model = factory().train_regress(&train_task)?;

        let test_features = features.select(Axis(0), test_idx);
        let pred = model.predict(&test_features)?;
        let Prediction::Regression { predicted, .. } = pred else {
            return Err(SmeltError::InvalidParameter(
                "oof_regression requires a base learner that produces regression predictions"
                    .into(),
            ));
        };
        for (j, &idx) in test_idx.iter().enumerate() {
            out[idx] = predicted[j];
        }
    }
    Ok(out)
}

/// Out-of-fold P(treatment=1|X) for every row, from a binary classifier.
pub(crate) fn oof_propensity(
    features: &Array2<f64>,
    treatment: &[usize],
    factory: &LearnerFactory,
    folds: usize,
    seed: u64,
) -> Result<Vec<f64>> {
    let n = features.nrows();
    let splits = CrossValidation::new(folds).with_seed(seed).splits(n)?;
    let mut out = vec![0.0; n];
    for (train_idx, test_idx) in &splits {
        let train_target: Vec<usize> = train_idx.iter().map(|&i| treatment[i]).collect();
        if !train_target.contains(&0) || !train_target.contains(&1) {
            return Err(SmeltError::InvalidParameter(
                "a cross-fitting fold has no units in one treatment arm; use fewer folds or more data".into(),
            ));
        }
        let train_features = features.select(Axis(0), train_idx);
        let train_task = ClassificationTask::new("oof_propensity", train_features, train_target)?;
        let model = factory().train_classif(&train_task)?;

        let test_features = features.select(Axis(0), test_idx);
        let pred = model.predict(&test_features)?;
        let Prediction::Classification {
            probabilities: Some(probs),
            ..
        } = pred
        else {
            return Err(SmeltError::InvalidParameter(
                "oof_propensity requires a classifier that produces class probabilities \
                 (e.g. logistic_regression, gaussian_nb, random_forest)"
                    .into(),
            ));
        };
        for (j, &idx) in test_idx.iter().enumerate() {
            // Both arms are guaranteed present in train_target above, so a
            // well-behaved classifier's probability rows are 2 wide; error
            // (not silently default to 0.5) if one somehow isn't.
            out[idx] = *probs[j].get(1).ok_or_else(|| {
                SmeltError::InvalidParameter(
                    "propensity classifier returned fewer than 2 probability columns \
                     despite both treatment arms being present in training data"
                        .into(),
                )
            })?;
        }
    }
    Ok(out)
}

/// Out-of-fold per-arm regression predictions `(mu0_hat, mu1_hat)` for every
/// row: within each fold, fit separately on the fold's control/treated
/// training subset, predict on the held-out rows for both arms.
pub(crate) fn oof_regression_by_arm(
    features: &Array2<f64>,
    treatment: &[usize],
    outcome: &[f64],
    control_factory: &LearnerFactory,
    treated_factory: &LearnerFactory,
    folds: usize,
    seed: u64,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let n = features.nrows();
    let splits = CrossValidation::new(folds).with_seed(seed).splits(n)?;
    let mut mu0 = vec![0.0; n];
    let mut mu1 = vec![0.0; n];
    for (train_idx, test_idx) in &splits {
        let control_train: Vec<usize> = train_idx
            .iter()
            .copied()
            .filter(|&i| treatment[i] == 0)
            .collect();
        let treated_train: Vec<usize> = train_idx
            .iter()
            .copied()
            .filter(|&i| treatment[i] == 1)
            .collect();
        if control_train.is_empty() || treated_train.is_empty() {
            return Err(SmeltError::InvalidParameter(
                "a cross-fitting fold has no units in one treatment arm; use fewer folds or more data".into(),
            ));
        }

        let control_features = features.select(Axis(0), &control_train);
        let control_target: Vec<f64> = control_train.iter().map(|&i| outcome[i]).collect();
        let control_task = RegressionTask::new("oof_control", control_features, control_target)?;
        let control_model = control_factory().train_regress(&control_task)?;

        let treated_features = features.select(Axis(0), &treated_train);
        let treated_target: Vec<f64> = treated_train.iter().map(|&i| outcome[i]).collect();
        let treated_task = RegressionTask::new("oof_treated", treated_features, treated_target)?;
        let treated_model = treated_factory().train_regress(&treated_task)?;

        let test_features = features.select(Axis(0), test_idx);
        let pred0 = control_model.predict(&test_features)?;
        let pred1 = treated_model.predict(&test_features)?;
        let (
            Prediction::Regression { predicted: p0, .. },
            Prediction::Regression { predicted: p1, .. },
        ) = (&pred0, &pred1)
        else {
            return Err(SmeltError::InvalidParameter(
                "oof_regression_by_arm requires base learners that produce regression predictions"
                    .into(),
            ));
        };
        for (j, &idx) in test_idx.iter().enumerate() {
            mu0[idx] = p0[j];
            mu1[idx] = p1[j];
        }
    }
    Ok((mu0, mu1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::LogisticRegression;

    /// Regression test: `oof_propensity` used to silently default a fold's
    /// out-of-fold propensity to 0.5 (`probs[j].get(1).copied().unwrap_or(0.5)`)
    /// whenever that fold's training data had only one treatment arm (the
    /// fitted classifier then has `n_classes == 1`, so index 1 doesn't
    /// exist) -- `oof_regression_by_arm` already errored on the same
    /// condition; this brings `oof_propensity` in line instead of fabricating
    /// a "coin flip" propensity with no error or warning.
    #[test]
    fn errors_when_a_fold_has_only_one_treatment_arm() {
        // 6 samples, only the last is treated; with 6 folds (leave-one-out),
        // the fold that holds out that single treated sample has an
        // all-control training set.
        let features = Array2::from_shape_vec((6, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let treatment = vec![0, 0, 0, 0, 0, 1];
        let factory: LearnerFactory = Box::new(|| Box::new(LogisticRegression::new()));

        let result = oof_propensity(&features, &treatment, &factory, 6, 42);
        assert!(
            result.is_err(),
            "a fold with only one treatment arm must error, not silently return 0.5"
        );
    }
}
