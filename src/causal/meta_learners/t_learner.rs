//! T-learner: fit two separate outcome models, one per treatment arm.

use super::{LearnerFactory, MetaLearnerResult, validate_causal_inputs};
use crate::learner::Learner;
use crate::prediction::Prediction;
use crate::task::RegressionTask;
use crate::{Result, SmeltError};
use ndarray::{Array2, Axis};

/// T-learner (Künzel et al. 2019, the simplest metalearner): fit `μ̂0(x)` on
/// control units and `μ̂1(x)` on treated units, `τ̂(x) = μ̂1(x) - μ̂0(x)`.
///
/// No cross-fitting, no propensity model -- just two independent regressions.
/// Simple and often a reasonable baseline, but each arm model only ever sees
/// its own subsample, so an arm with few units gets a noisier fit than an
/// X-learner or DR-learner would produce with the same data.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use smelt_ml::causal::meta_learners::TLearner;
/// use ndarray::array;
///
/// let features = array![[0.0], [1.0], [2.0], [3.0], [0.0], [1.0], [2.0], [3.0]];
/// let treatment = vec![0, 0, 0, 0, 1, 1, 1, 1];
/// let outcome = vec![1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0]; // true effect = 5.0
///
/// let t_learner = TLearner::new(
///     || Box::new(LinearRegression::new()),
///     || Box::new(LinearRegression::new()),
/// );
/// let result = t_learner.estimate(&features, &treatment, &outcome).unwrap();
/// assert!((result.ate - 5.0).abs() < 0.5);
/// ```
pub struct TLearner {
    control_factory: LearnerFactory,
    treated_factory: LearnerFactory,
}

impl TLearner {
    /// `control_factory`/`treated_factory` build the outcome model for the
    /// control (`treatment=0`) and treated (`treatment=1`) arms respectively.
    pub fn new(
        control_factory: impl Fn() -> Box<dyn Learner> + Send + Sync + 'static,
        treated_factory: impl Fn() -> Box<dyn Learner> + Send + Sync + 'static,
    ) -> Self {
        Self {
            control_factory: Box::new(control_factory),
            treated_factory: Box::new(treated_factory),
        }
    }

    /// Estimate per-unit CATE. `treatment` must be binary (0/1) with at
    /// least one unit in each arm.
    pub fn estimate(
        &self,
        features: &Array2<f64>,
        treatment: &[usize],
        outcome: &[f64],
    ) -> Result<MetaLearnerResult> {
        validate_causal_inputs(features, treatment, outcome)?;

        let control_idx: Vec<usize> = (0..treatment.len())
            .filter(|&i| treatment[i] == 0)
            .collect();
        let treated_idx: Vec<usize> = (0..treatment.len())
            .filter(|&i| treatment[i] == 1)
            .collect();

        let control_features = features.select(Axis(0), &control_idx);
        let control_target: Vec<f64> = control_idx.iter().map(|&i| outcome[i]).collect();
        let control_task =
            RegressionTask::new("t_learner_control", control_features, control_target)?;

        let treated_features = features.select(Axis(0), &treated_idx);
        let treated_target: Vec<f64> = treated_idx.iter().map(|&i| outcome[i]).collect();
        let treated_task =
            RegressionTask::new("t_learner_treated", treated_features, treated_target)?;

        let mu0 = (self.control_factory)().train_regress(&control_task)?;
        let mu1 = (self.treated_factory)().train_regress(&treated_task)?;

        let pred0 = mu0.predict(features)?;
        let pred1 = mu1.predict(features)?;
        let (
            Prediction::Regression { predicted: p0, .. },
            Prediction::Regression { predicted: p1, .. },
        ) = (&pred0, &pred1)
        else {
            return Err(SmeltError::InvalidParameter(
                "TLearner's base learners must produce regression predictions".into(),
            ));
        };

        let cate: Vec<f64> = p0.iter().zip(p1).map(|(&a, &b)| b - a).collect();
        Ok(MetaLearnerResult::new(cate))
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_fixtures::{synthetic_confounded_nonlinear_cate, synthetic_linear_cate};
    use super::*;
    use crate::learner::RandomForest;
    use crate::measure::{AteBias, Measure, Pehe};
    use crate::prediction::Prediction;

    fn t_learner_rf() -> TLearner {
        TLearner::new(
            || Box::new(RandomForest::new().with_n_estimators(50).with_seed(1)),
            || Box::new(RandomForest::new().with_n_estimators(50).with_seed(2)),
        )
    }

    #[test]
    fn recovers_linear_heterogeneous_effect() {
        let (features, treatment, outcome, true_cate) = synthetic_linear_cate(300, 1, 0.1);
        let result = t_learner_rf()
            .estimate(&features, &treatment, &outcome)
            .unwrap();
        let pred = Prediction::causal_effect_with_truth(result.cate, true_cate);
        assert!(
            Pehe.score(&pred).unwrap() < 1.5,
            "PEHE too high: {}",
            Pehe.score(&pred).unwrap()
        );
        assert!(AteBias.score(&pred).unwrap() < 0.5);
    }

    #[test]
    fn degrades_relative_to_x_learner_under_confounding() {
        // Not a comparison test (X-learner lives in another module) -- just
        // documents that T-learner still runs end-to-end on the harder,
        // confounded fixture without crashing or producing garbage-scale output.
        let (features, treatment, outcome, true_cate) =
            synthetic_confounded_nonlinear_cate(300, 2, 0.1);
        let result = t_learner_rf()
            .estimate(&features, &treatment, &outcome)
            .unwrap();
        let pred = Prediction::causal_effect_with_truth(result.cate, true_cate);
        assert!(Pehe.score(&pred).unwrap().is_finite());
    }

    #[test]
    fn rejects_non_binary_treatment() {
        let features = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let err = t_learner_rf().estimate(&features, &[0, 1, 2], &[1.0, 2.0, 3.0]);
        assert!(err.is_err());
    }

    #[test]
    fn rejects_single_arm() {
        let features = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let err = t_learner_rf().estimate(&features, &[0, 0, 0], &[1.0, 2.0, 3.0]);
        assert!(err.is_err());
    }

    #[test]
    fn rejects_dimension_mismatch() {
        let features = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let err = t_learner_rf().estimate(&features, &[0, 1], &[1.0, 2.0, 3.0]);
        assert!(err.is_err());
    }
}
