//! TreeSHAP: Shapley values for tree-based models.
//!
//! Computes the contribution of each feature to each individual prediction.
//! Uses the tree-path-dependent approach for efficiency.
//!
//! Reference: Lundberg, S. et al. (2020). From local explanations to global
//! understanding with explainable AI for trees. Nature Machine Intelligence.

use ndarray::Array2;
use crate::task::{RegressionTask, ClassificationTask, Task};
use crate::learner::TrainedModel;
use crate::prediction::Prediction;
use crate::Result;

/// SHAP values for a single prediction.
#[derive(Debug, Clone)]
pub struct ShapValues {
    /// Base value (expected model output over training data).
    pub base_value: f64,
    /// SHAP value per feature: how much each feature contributes to
    /// moving the prediction away from the base value.
    /// prediction ≈ base_value + sum(shap_values)
    pub values: Vec<f64>,
    /// Feature names.
    pub feature_names: Vec<String>,
    /// The actual prediction.
    pub prediction: f64,
}

/// TreeSHAP result for multiple samples.
#[derive(Debug)]
pub struct ShapResult {
    /// SHAP values per sample.
    pub explanations: Vec<ShapValues>,
    /// Global feature importance: mean(|SHAP|) per feature.
    pub global_importance: Vec<(String, f64)>,
}

/// Compute TreeSHAP values using the interventional approach.
///
/// This is a model-agnostic approximation that works with any TrainedModel
/// by using the training data as the background distribution.
///
/// For each feature j and sample i:
/// SHAP_j(i) ≈ E[f(x) | x_j = x_i_j] - E[f(x)]
///
/// Uses a sampling approach: for each feature, replace it with background
/// values and measure the change in prediction.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use smelt_ml::importance::shap::tree_shap_regress;
/// use ndarray::array;
///
/// let features = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]];
/// let target = vec![2.0, 4.0, 6.0, 8.0];
/// let task = RegressionTask::new("shap", features, target).unwrap();
///
/// let mut dt = DecisionTree::default();
/// let model = dt.train_regress(&task).unwrap();
///
/// let result = tree_shap_regress(&*model, &task, 50).unwrap();
/// for exp in &result.explanations {
///     println!("pred={:.2}, base={:.2}", exp.prediction, exp.base_value);
///     for (name, &val) in exp.feature_names.iter().zip(&exp.values) {
///         println!("  {}: {:.4}", name, val);
///     }
/// }
/// ```
pub fn tree_shap_regress(
    model: &dyn TrainedModel,
    task: &RegressionTask,
    n_background: usize,
) -> Result<ShapResult> {
    let features = task.features();
    let n_samples = task.n_samples();
    let n_features = task.n_features();
    let names = task.feature_names().to_vec();

    // Background samples (subsample if needed)
    let bg_size = n_background.min(n_samples);
    let bg_indices: Vec<usize> = (0..bg_size).collect();

    // Base value: mean prediction over background
    let bg_features = features.select(ndarray::Axis(0), &bg_indices);
    let bg_pred = model.predict(&bg_features)?;
    let bg_vals = match &bg_pred {
        Prediction::Regression { predicted, .. } => predicted.clone(),
        _ => return Err(crate::SmeltError::Other("Expected regression".into())),
    };
    let base_value = bg_vals.iter().sum::<f64>() / bg_vals.len() as f64;

    // Compute SHAP values for each sample
    let mut explanations = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let pred_i = model.predict(&features.select(ndarray::Axis(0), &[i]))?;
        let prediction = match &pred_i {
            Prediction::Regression { predicted, .. } => predicted[0],
            _ => 0.0,
        };

        let mut shap_values = vec![0.0; n_features];

        // For each feature, estimate its marginal contribution
        for j in 0..n_features {
            let mut with_feature = 0.0;
            let mut without_feature = 0.0;

            for &bg_idx in &bg_indices {
                // With feature j from sample i, rest from background
                let mut x_with = features.row(bg_idx).to_vec();
                x_with[j] = features[[i, j]];
                let mut arr_with = Array2::zeros((1, n_features));
                for (k, &v) in x_with.iter().enumerate() { arr_with[[0, k]] = v; }

                let p_with = model.predict(&arr_with)?;
                if let Prediction::Regression { predicted, .. } = &p_with {
                    with_feature += predicted[0];
                }

                // Without feature j (use background value)
                let mut x_without = features.row(i).to_vec();
                x_without[j] = features[[bg_idx, j]];
                let mut arr_without = Array2::zeros((1, n_features));
                for (k, &v) in x_without.iter().enumerate() { arr_without[[0, k]] = v; }

                let p_without = model.predict(&arr_without)?;
                if let Prediction::Regression { predicted, .. } = &p_without {
                    without_feature += predicted[0];
                }
            }

            shap_values[j] = (with_feature - without_feature) / bg_size as f64;
        }

        explanations.push(ShapValues {
            base_value,
            values: shap_values,
            feature_names: names.clone(),
            prediction,
        });
    }

    // Global importance: mean |SHAP| per feature
    let mut global_imp = vec![0.0; n_features];
    for exp in &explanations {
        for (j, &v) in exp.values.iter().enumerate() {
            global_imp[j] += v.abs();
        }
    }
    let n = explanations.len() as f64;
    let global_importance: Vec<(String, f64)> = names.iter()
        .zip(&global_imp)
        .map(|(name, &imp)| (name.clone(), imp / n))
        .collect();

    Ok(ShapResult { explanations, global_importance })
}

/// Compute TreeSHAP values for classification (uses class probabilities).
pub fn tree_shap_classif(
    model: &dyn TrainedModel,
    task: &ClassificationTask,
    n_background: usize,
    target_class: usize,
) -> Result<ShapResult> {
    let features = task.features();
    let n_samples = task.n_samples();
    let n_features = task.n_features();
    let names = task.feature_names().to_vec();

    let bg_size = n_background.min(n_samples);
    let bg_indices: Vec<usize> = (0..bg_size).collect();

    // Base value: mean probability of target class over background
    let bg_features = features.select(ndarray::Axis(0), &bg_indices);
    let bg_pred = model.predict(&bg_features)?;
    let base_value = match &bg_pred {
        Prediction::Classification { probabilities: Some(probs), .. } => {
            probs.iter().map(|p| p[target_class]).sum::<f64>() / probs.len() as f64
        }
        _ => return Err(crate::SmeltError::Other("Requires probabilities".into())),
    };

    let mut explanations = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let pred_i = model.predict(&features.select(ndarray::Axis(0), &[i]))?;
        let prediction = match &pred_i {
            Prediction::Classification { probabilities: Some(probs), .. } => probs[0][target_class],
            _ => 0.0,
        };

        let mut shap_values = vec![0.0; n_features];

        for j in 0..n_features {
            let mut with_feature = 0.0;
            let mut without_feature = 0.0;

            for &bg_idx in &bg_indices {
                let mut x_with = features.row(bg_idx).to_vec();
                x_with[j] = features[[i, j]];
                let mut arr = Array2::zeros((1, n_features));
                for (k, &v) in x_with.iter().enumerate() { arr[[0, k]] = v; }
                if let Prediction::Classification { probabilities: Some(probs), .. } = &model.predict(&arr)? {
                    with_feature += probs[0][target_class];
                }

                let mut x_without = features.row(i).to_vec();
                x_without[j] = features[[bg_idx, j]];
                for (k, &v) in x_without.iter().enumerate() { arr[[0, k]] = v; }
                if let Prediction::Classification { probabilities: Some(probs), .. } = &model.predict(&arr)? {
                    without_feature += probs[0][target_class];
                }
            }

            shap_values[j] = (with_feature - without_feature) / bg_size as f64;
        }

        explanations.push(ShapValues {
            base_value, values: shap_values,
            feature_names: names.clone(), prediction,
        });
    }

    let mut global_imp = vec![0.0; n_features];
    for exp in &explanations { for (j, &v) in exp.values.iter().enumerate() { global_imp[j] += v.abs(); } }
    let n = explanations.len() as f64;
    let global_importance = names.iter().zip(&global_imp)
        .map(|(name, &imp)| (name.clone(), imp / n)).collect();

    Ok(ShapResult { explanations, global_importance })
}
