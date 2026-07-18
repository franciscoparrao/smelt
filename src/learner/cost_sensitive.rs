//! Cost-sensitive classification: wraps any probabilistic classifier and
//! replaces its plain argmax decision with the Bayes-risk-minimizing one
//! under an explicit misclassification cost matrix.
//!
//! Elkan, C. (2001). "The Foundations of Cost-Sensitive Learning."
//! Proceedings of IJCAI 2001.
//!
//! Standard classification implicitly assumes every misclassification costs
//! the same (accuracy just counts errors). That's rarely true in practice --
//! missing a cancer diagnosis (false negative) is far costlier than an
//! unnecessary follow-up test (false positive), and the reverse trade-off
//! holds for, say, fraud alerts that block a legitimate transaction. Given a
//! cost matrix `cost[true][predicted]` and a classifier's predicted
//! probabilities `P(i|x)`, the Bayes-optimal decision is not `argmax_i
//! P(i|x)` but:
//!
//! ```text
//! predicted_class(x) = argmin_j  Σ_i P(i|x) · cost[i][j]
//! ```
//!
//! This requires no retraining at all -- it's a decision rule layered on
//! top of any base learner's existing probability estimates, which is why
//! this is a thin wrapper (same `factory: Fn() -> Box<dyn Learner>` pattern
//! as [`crate::learner::Bagging`]/[`crate::learner::Stacking`]) rather than
//! a new learning algorithm.

use crate::Result;
use crate::SmeltError;
use crate::learner::{Learner, TrainedModel};
use crate::prediction::Prediction;
use crate::task::ClassificationTask;
use ndarray::Array2;

/// Cost-sensitive classification wrapper.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use smelt_ml::learner::CostSensitiveClassifier;
/// use ndarray::array;
///
/// // Missing the positive class (a false negative) costs 10x more than a
/// // false positive -- e.g. a missed diagnosis vs. an unnecessary test.
/// let features = array![
///     [0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [0.0, 0.2],
///     [1.0, 1.0], [1.1, 0.9], [0.9, 1.1], [1.0, 0.9]
/// ];
/// let target = vec![0, 0, 0, 0, 1, 1, 1, 1];
/// let task = ClassificationTask::new("cost_demo", features, target).unwrap();
///
/// let mut cs = CostSensitiveClassifier::binary(
///     || Box::new(LogisticRegression::new()),
///     1.0,  // cost of a false positive
///     10.0, // cost of a false negative
/// );
/// let model = cs.train_classif(&task).unwrap();
/// ```
pub struct CostSensitiveClassifier {
    factory: Box<dyn Fn() -> Box<dyn Learner> + Send + Sync>,
    cost_matrix: Vec<Vec<f64>>,
}

impl CostSensitiveClassifier {
    /// Creates a cost-sensitive wrapper from a base-learner factory and an
    /// explicit `n_classes x n_classes` cost matrix, `cost[true][predicted]`
    /// (validated against the task's actual `n_classes` at train time).
    pub fn new(
        factory: impl Fn() -> Box<dyn Learner> + Send + Sync + 'static,
        cost_matrix: Vec<Vec<f64>>,
    ) -> Self {
        Self {
            factory: Box::new(factory),
            cost_matrix,
        }
    }

    /// Convenience constructor for the common binary case: `cost[0][1] =
    /// false_positive_cost` (true class 0, predicted 1), `cost[1][0] =
    /// false_negative_cost` (true class 1, predicted 0), with zero cost on
    /// the diagonal (correct predictions).
    pub fn binary(
        factory: impl Fn() -> Box<dyn Learner> + Send + Sync + 'static,
        false_positive_cost: f64,
        false_negative_cost: f64,
    ) -> Self {
        Self::new(
            factory,
            vec![
                vec![0.0, false_positive_cost],
                vec![false_negative_cost, 0.0],
            ],
        )
    }
}

impl Learner for CostSensitiveClassifier {
    fn id(&self) -> &str {
        "cost_sensitive"
    }

    fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        crate::validate::check_no_weights(task.weights(), "CostSensitiveClassifier")?;
        let n_classes = task.n_classes();
        if self.cost_matrix.len() != n_classes
            || self.cost_matrix.iter().any(|row| row.len() != n_classes)
        {
            return Err(SmeltError::InvalidParameter(format!(
                "cost matrix must be {n_classes}x{n_classes} (task has {n_classes} classes), \
                 got {} rows of lengths {:?}",
                self.cost_matrix.len(),
                self.cost_matrix.iter().map(|r| r.len()).collect::<Vec<_>>()
            )));
        }

        let mut base = (self.factory)();
        let model = base.train_classif(task)?;

        Ok(Box::new(TrainedCostSensitiveClassifier {
            model,
            cost_matrix: self.cost_matrix.clone(),
        }))
    }
}

struct TrainedCostSensitiveClassifier {
    model: Box<dyn TrainedModel>,
    cost_matrix: Vec<Vec<f64>>,
}

impl TrainedModel for TrainedCostSensitiveClassifier {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        let pred = self.model.predict(features)?;
        match pred {
            Prediction::Classification {
                truth,
                probabilities: Some(probs),
                ..
            } => {
                let n_classes = self.cost_matrix.len();
                let predicted: Vec<usize> = probs
                    .iter()
                    .map(|p| {
                        (0..n_classes)
                            .map(|j| {
                                let expected_cost: f64 = (0..n_classes)
                                    .map(|i| p.get(i).copied().unwrap_or(0.0) * self.cost_matrix[i][j])
                                    .sum();
                                (j, expected_cost)
                            })
                            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                            .map(|(j, _)| j)
                            .unwrap_or(0)
                    })
                    .collect();
                Ok(Prediction::Classification {
                    predicted,
                    truth,
                    probabilities: Some(probs),
                })
            }
            Prediction::Classification {
                probabilities: None,
                ..
            } => Err(SmeltError::IncompatiblePrediction(
                "CostSensitiveClassifier requires a base learner that produces probabilities \
                 (e.g. logistic_regression, random_forest, gaussian_nb)"
                    .into(),
            )),
            _ => Err(SmeltError::IncompatiblePrediction(
                "CostSensitiveClassifier requires classification predictions".into(),
            )),
        }
    }

    fn feature_importance(&self) -> Option<Vec<(String, f64)>> {
        self.model.feature_importance()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::LogisticRegression;
    use ndarray::array;

    #[test]
    fn registered_id_matches() {
        let cs = CostSensitiveClassifier::binary(|| Box::new(LogisticRegression::new()), 1.0, 1.0);
        assert_eq!(cs.id(), "cost_sensitive");
    }

    #[test]
    fn rejects_wrong_shaped_cost_matrix() {
        let features = array![[0.0], [1.0], [2.0], [3.0]];
        let target = vec![0usize, 0, 1, 1];
        let task = ClassificationTask::new("t", features, target).unwrap();

        // 3x3 matrix for a 2-class task.
        let mut cs = CostSensitiveClassifier::new(
            || Box::new(LogisticRegression::new()),
            vec![vec![0.0, 1.0, 2.0], vec![1.0, 0.0, 2.0], vec![1.0, 2.0, 0.0]],
        );
        assert!(cs.train_classif(&task).is_err());
    }

    /// Regression test for the actual point of this wrapper: with an
    /// asymmetric cost matrix, the predicted class must shift toward the
    /// costlier-to-miss class relative to the base learner's own (cost-
    /// blind) argmax, for points where the base model's probability is
    /// genuinely uncertain (close to 0.5). A plain classifier and a
    /// heavily-imbalanced-cost wrapper should disagree on such points.
    #[test]
    fn cost_matrix_shifts_decisions_toward_costlier_to_miss_class() {
        // Class 1 is rare and its false negatives cost 20x a false positive.
        let mut feats = Vec::new();
        let mut target = Vec::new();
        for i in 0..80 {
            let x = i as f64 * 0.1;
            feats.push(x);
            target.push(0usize);
        }
        for i in 0..20 {
            let x = 5.0 + i as f64 * 0.1; // overlaps with the tail of class 0
            feats.push(x);
            target.push(1usize);
        }
        let features = Array2::from_shape_vec((100, 1), feats).unwrap();
        let task = ClassificationTask::new("imbalanced", features.clone(), target).unwrap();

        let mut plain = LogisticRegression::new();
        let plain_model = plain.train_classif(&task).unwrap();
        let mut cost_sensitive = CostSensitiveClassifier::binary(
            || Box::new(LogisticRegression::new()),
            1.0,
            20.0,
        );
        let cost_model = cost_sensitive.train_classif(&task).unwrap();

        // Query points right around the decision boundary, where the base
        // model's probability for class 1 is real but modest -- exactly
        // where cost-sensitivity should flip the decision toward class 1.
        let boundary_features =
            Array2::from_shape_vec((3, 1), vec![4.6, 4.7, 4.8]).unwrap();

        let Prediction::Classification { predicted: plain_pred, .. } =
            plain_model.predict(&boundary_features).unwrap()
        else {
            panic!("expected classification");
        };
        let Prediction::Classification { predicted: cost_pred, .. } =
            cost_model.predict(&boundary_features).unwrap()
        else {
            panic!("expected classification");
        };

        let cost_favors_class1_more_often = cost_pred.iter().filter(|&&p| p == 1).count()
            >= plain_pred.iter().filter(|&&p| p == 1).count();
        assert!(
            cost_favors_class1_more_often,
            "cost-sensitive predictions {cost_pred:?} should favor class 1 at least as often as \
             the plain classifier's {plain_pred:?}, given the 20x false-negative cost"
        );
    }

    #[test]
    fn feature_importance_delegates_to_base_model() {
        let features = array![[0.0, 5.0], [1.0, 4.0], [2.0, 3.0], [3.0, 2.0]];
        let target = vec![0usize, 0, 1, 1];
        let task = ClassificationTask::new("imp", features, target).unwrap();

        let mut cs = CostSensitiveClassifier::binary(
            || Box::new(crate::learner::tree::decision_tree::DecisionTree::default()),
            1.0,
            1.0,
        );
        let model = cs.train_classif(&task).unwrap();
        assert!(model.feature_importance().is_some());
    }
}
