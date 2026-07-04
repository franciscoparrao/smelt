//! Recursive Feature Elimination (RFE) — wrapper feature selection.
//!
//! Iteratively removes the least important feature based on a learner's
//! feature importance until the desired number of features is reached.

use super::Transformer;
use crate::learner::Learner;
use crate::task::{ClassificationTask, RegressionTask};
use crate::{Result, SmeltError};
use ndarray::Array2;

/// Recursive Feature Elimination.
///
/// Uses a model's feature importance to iteratively remove the weakest
/// feature until `n_features_to_select` remain.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use smelt_ml::preprocess::RFE;
/// use ndarray::array;
///
/// let features = array![
///     [0.0, 42.0, 99.0], [0.1, 13.0, 55.0],
///     [1.0, 42.0, 99.0], [1.1, 13.0, 55.0],
/// ];
/// let target = vec![0, 0, 1, 1];
/// let task = ClassificationTask::new("rfe", features, target).unwrap();
///
/// let rfe = RFE::classif(
///     || Box::new(DecisionTree::default()),
///     2,  // keep 2 features
/// );
/// ```
pub struct RFE {
    learner_factory: Box<dyn Fn() -> Box<dyn Learner> + Send + Sync>,
    n_features_to_select: usize,
    is_classifier: bool,
    selected_indices: Option<Vec<usize>>,
    n_features_in: Option<usize>,
}

impl RFE {
    /// Create RFE for classification.
    pub fn classif(
        factory: impl Fn() -> Box<dyn Learner> + Send + Sync + 'static,
        n_features: usize,
    ) -> Self {
        Self {
            learner_factory: Box::new(factory),
            n_features_to_select: n_features,
            is_classifier: true,
            selected_indices: None,
            n_features_in: None,
        }
    }

    /// Create RFE for regression.
    pub fn regress(
        factory: impl Fn() -> Box<dyn Learner> + Send + Sync + 'static,
        n_features: usize,
    ) -> Self {
        Self {
            learner_factory: Box::new(factory),
            n_features_to_select: n_features,
            is_classifier: false,
            selected_indices: None,
            n_features_in: None,
        }
    }

    /// Indices of the features RFE selected as important, or `None` if not
    /// yet fitted.
    pub fn selected_indices(&self) -> Option<&[usize]> {
        self.selected_indices.as_deref()
    }
}

impl Clone for RFE {
    fn clone(&self) -> Self {
        // Can't clone the factory closure, but we can share the fitted state
        Self {
            learner_factory: Box::new(|| panic!("cloned RFE cannot create new learners")),
            n_features_to_select: self.n_features_to_select,
            is_classifier: self.is_classifier,
            selected_indices: self.selected_indices.clone(),
            n_features_in: self.n_features_in,
        }
    }
}

impl Transformer for RFE {
    fn id(&self) -> &str {
        "rfe"
    }

    fn fit(&mut self, features: &Array2<f64>) -> Result<()> {
        // Without target, just select first n features
        self.n_features_in = Some(features.ncols());
        self.selected_indices =
            Some((0..self.n_features_to_select.min(features.ncols())).collect());
        Ok(())
    }

    fn fit_supervised(&mut self, features: &Array2<f64>, target: &[f64]) -> Result<()> {
        let n_total = features.ncols();
        self.n_features_in = Some(n_total);
        let mut remaining: Vec<usize> = (0..n_total).collect();

        while remaining.len() > self.n_features_to_select {
            // Select current feature subset
            let sub_features = features.select(ndarray::Axis(1), &remaining);
            let names: Vec<String> = remaining.iter().map(|i| format!("f{i}")).collect();

            // Train model and get importance
            let mut learner = (self.learner_factory)();
            let importance = if self.is_classifier {
                let int_target: Vec<usize> = target.iter().map(|&t| t as usize).collect();
                let task = ClassificationTask::new("rfe", sub_features, int_target)?
                    .with_feature_names(names)?;
                let model = learner.train_classif(&task)?;
                model.feature_importance()
            } else {
                let task = RegressionTask::new("rfe", sub_features, target.to_vec())?
                    .with_feature_names(names)?;
                let model = learner.train_regress(&task)?;
                model.feature_importance()
            };

            // Remove least important feature
            match importance {
                Some(imp) => {
                    let least_idx = imp
                        .iter()
                        .enumerate()
                        .min_by(|a, b| {
                            a.1.1
                                .partial_cmp(&b.1.1)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .unwrap()
                        .0;
                    remaining.remove(least_idx);
                }
                None => {
                    // No importance available: remove last
                    remaining.pop();
                }
            }
        }

        remaining.sort();
        self.selected_indices = Some(remaining);
        Ok(())
    }

    fn transform(&self, features: &Array2<f64>) -> Result<Array2<f64>> {
        let indices = self
            .selected_indices
            .as_ref()
            .ok_or(SmeltError::NotTrained)?;
        Ok(features.select(ndarray::Axis(1), indices))
    }

    fn transform_names(&self, names: &[String]) -> Result<Vec<String>> {
        let indices = self
            .selected_indices
            .as_ref()
            .ok_or(SmeltError::NotTrained)?;
        Ok(indices.iter().map(|&i| names[i].clone()).collect())
    }

    fn clone_box(&self) -> Box<dyn Transformer> {
        Box::new(self.clone())
    }
}
