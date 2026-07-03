//! S-learner: a single outcome model with treatment as an extra feature.

use super::{LearnerFactory, MetaLearnerResult, validate_causal_inputs};
use crate::learner::Learner;
use crate::prediction::Prediction;
use crate::task::RegressionTask;
use crate::{Result, SmeltError};
use ndarray::{Array2, Axis, concatenate};

/// S-learner (Künzel et al. 2019): fit a single outcome model on `(X, T)`
/// with treatment as just another feature, `τ̂(x) = μ̂(x, 1) - μ̂(x, 0)`.
///
/// **Known weaknesses** (Künzel et al. 2019): a *linear* base learner
/// cannot represent any `X*T` interaction at all (there's no interaction
/// term in an additive `[X, T]` linear model), so it can only recover a
/// *constant*, non-heterogeneous effect. A tree/regularized base can
/// represent interactions via splits, but often under-weights or never
/// splits on the single appended treatment column when other features
/// carry more variance, which biases heterogeneous CATE toward zero -- the
/// treatment effect can get "regularized away." Kept here mainly for
/// completeness/comparison; prefer [`super::TLearner`] or [`super::XLearner`]
/// as defaults for heterogeneous effects.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use smelt_ml::causal::meta_learners::SLearner;
/// use ndarray::array;
///
/// let features = array![[0.0], [1.0], [2.0], [3.0], [0.0], [1.0], [2.0], [3.0]];
/// let treatment = vec![0, 0, 0, 0, 1, 1, 1, 1];
/// let outcome = vec![1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0]; // true effect = 5.0
///
/// let s_learner = SLearner::new(|| Box::new(LinearRegression::new()));
/// let result = s_learner.estimate(&features, &treatment, &outcome).unwrap();
/// assert!((result.ate - 5.0).abs() < 0.5);
/// ```
pub struct SLearner {
    factory: LearnerFactory,
}

impl SLearner {
    pub fn new(factory: impl Fn() -> Box<dyn Learner> + Send + Sync + 'static) -> Self {
        Self {
            factory: Box::new(factory),
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

        let n = features.nrows();
        let treatment_col = Array2::from_shape_vec(
            (n, 1),
            treatment.iter().map(|&t| t as f64).collect(),
        )
        .expect("treatment column shape matches n_samples");
        let augmented = concatenate(Axis(1), &[features.view(), treatment_col.view()])
            .map_err(|e| SmeltError::Other(format!("failed to augment features with treatment column: {e}")))?;

        let task = RegressionTask::new("s_learner", augmented, outcome.to_vec())?;
        let model = (self.factory)().train_regress(&task)?;

        let ones = Array2::from_elem((n, 1), 1.0);
        let zeros = Array2::from_elem((n, 1), 0.0);
        let treated_input = concatenate(Axis(1), &[features.view(), ones.view()])
            .map_err(|e| SmeltError::Other(format!("failed to build treated-arm input: {e}")))?;
        let control_input = concatenate(Axis(1), &[features.view(), zeros.view()])
            .map_err(|e| SmeltError::Other(format!("failed to build control-arm input: {e}")))?;

        let pred1 = model.predict(&treated_input)?;
        let pred0 = model.predict(&control_input)?;
        let (Prediction::Regression { predicted: p1, .. }, Prediction::Regression { predicted: p0, .. }) =
            (&pred1, &pred0)
        else {
            return Err(SmeltError::InvalidParameter(
                "SLearner's base learner must produce regression predictions".into(),
            ));
        };

        let cate: Vec<f64> = p1.iter().zip(p0).map(|(&a, &b)| a - b).collect();
        Ok(MetaLearnerResult::new(cate))
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_fixtures::synthetic_linear_cate;
    use super::*;
    use crate::learner::{LinearRegression, RandomForest};
    use crate::measure::{AteBias, Measure, Pehe};
    use crate::prediction::Prediction;

    #[test]
    fn recovers_linear_heterogeneous_effect_with_tree_base() {
        // A plain additive linear base (columns [x0, x1, T], no interaction
        // term) *cannot* represent this fixture's tau(x) = 2*x0 effect at
        // all -- that's a model-misspecification ceiling, not just the
        // "regularized away" bias Kunzel et al. describe, so linear
        // regression isn't a fair test of the S-learner's data flow here.
        // A tree-based learner can split on T combined with x0 thresholds,
        // approximating the interaction the way T/X/R/DR-learner's tests
        // all rely on RandomForest to do.
        let (features, treatment, outcome, true_cate) = synthetic_linear_cate(300, 3, 0.1);
        let s_learner = SLearner::new(|| Box::new(RandomForest::new().with_n_estimators(50).with_seed(1)));
        let result = s_learner.estimate(&features, &treatment, &outcome).unwrap();
        let pred = Prediction::causal_effect_with_truth(result.cate, true_cate);
        assert!(Pehe.score(&pred).unwrap() < 2.5, "PEHE too high: {}", Pehe.score(&pred).unwrap());
        assert!(AteBias.score(&pred).unwrap() < 0.7);
    }

    #[test]
    fn recovers_constant_effect_with_linear_base() {
        // A *constant* (non-heterogeneous) effect has no X*T interaction to
        // miss, so a plain linear base recovers it fine -- this is the case
        // the module doc's example demonstrates too.
        let features = Array2::from_shape_vec((8, 1), vec![0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]).unwrap();
        let treatment = vec![0, 0, 0, 0, 1, 1, 1, 1];
        let outcome = vec![1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0]; // true effect = 5.0
        let s_learner = SLearner::new(|| Box::new(LinearRegression::new()));
        let result = s_learner.estimate(&features, &treatment, &outcome).unwrap();
        assert!((result.ate - 5.0).abs() < 0.5, "ATE = {}", result.ate);
    }

    #[test]
    fn rejects_non_binary_treatment() {
        let features = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let s_learner = SLearner::new(|| Box::new(LinearRegression::new()));
        let err = s_learner.estimate(&features, &[0, 1, 2], &[1.0, 2.0, 3.0]);
        assert!(err.is_err());
    }

    #[test]
    fn rejects_single_arm() {
        let features = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let s_learner = SLearner::new(|| Box::new(LinearRegression::new()));
        let err = s_learner.estimate(&features, &[1, 1, 1], &[1.0, 2.0, 3.0]);
        assert!(err.is_err());
    }
}
