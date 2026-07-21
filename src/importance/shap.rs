//! Permutation SHAP: Shapley-value feature attributions for any model.
//!
//! Computes the contribution of each feature to each individual prediction
//! via sampling-based Shapley value estimation (Štrumbelj & Kononenko 2010;
//! the same interventional/permutation approach as `shap.PermutationExplainer`).
//! Model-agnostic: works with any `TrainedModel` through `predict()` alone,
//! not a tree-structure-specific algorithm (despite the historical module
//! name "TreeSHAP" -- kept for API stability, see below).
//!
//! For a target point `x`, a background point `b`, and a random permutation
//! `π` of the features, walk from `b` to `x` one feature at a time in the
//! order given by `π`, crediting each feature with the change in prediction
//! caused by revealing it. Averaged over many permutations and background
//! draws, this converges to the exact Shapley value and satisfies the
//! efficiency property by construction: for every single (permutation,
//! background) draw, the per-feature contributions telescope to exactly
//! `f(x) - f(b)`, so `prediction ≈ base_value + sum(shap_values)` holds (up
//! to Monte Carlo noise that shrinks with more permutations).
//!
//! Reference: Štrumbelj, E., & Kononenko, I. (2010). An efficient
//! explanation of individual classifications using game theory. JMLR.

use crate::Result;
use crate::learner::TrainedModel;
use crate::prediction::Prediction;
use crate::task::{ClassificationTask, RegressionTask, Task};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

/// SHAP values for a single prediction.
#[derive(Debug, Clone)]
pub struct ShapValues {
    /// Base value (expected model output over the background sample).
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

/// SHAP result for multiple samples.
#[derive(Debug)]
pub struct ShapResult {
    /// SHAP values per sample.
    pub explanations: Vec<ShapValues>,
    /// Global feature importance: mean(|SHAP|) per feature.
    pub global_importance: Vec<(String, f64)>,
}

/// Random permutations averaged per background point, per explained row.
/// Reduces variance in the per-feature split without affecting the
/// efficiency property (each permutation, for a fixed background point,
/// telescopes to the same total `f(x) - f(bg)` regardless of order). Total
/// cost per explained row is `bg_size * PERMUTATIONS_PER_BG * n_features`
/// model predictions.
const PERMUTATIONS_PER_BG: usize = 3;

fn row_as_array(row: &[f64]) -> Array2<f64> {
    Array2::from_shape_vec((1, row.len()), row.to_vec()).unwrap()
}

/// Permutation-SHAP attributions for one row, averaged over `PERMUTATIONS_PER_BG`
/// random feature orderings against *every* background point in `bg_indices`
/// (not a resampled subset of it). `predict_scalar` extracts the target
/// scalar (a regression value, or a class probability) from a `Prediction`.
///
/// Using the exact same background set that `base_value` was averaged over
/// -- each point contributing equally, `PERMUTATIONS_PER_BG` times -- is
/// what makes the efficiency property exact rather than approximate: for a
/// single (permutation, background point) draw, the per-feature increments
/// telescope to exactly `f(x) - f(bg)` regardless of the permutation order,
/// so averaging over the same background set that produced `base_value`
/// gives `sum(shap) = f(x) - base_value` exactly (mod floating point),
/// with no Monte Carlo residual left over from an independent resampling.
fn permutation_shap_row(
    model: &dyn TrainedModel,
    x: &[f64],
    bg_indices: &[usize],
    features: &Array2<f64>,
    n_features: usize,
    rng: &mut StdRng,
    predict_scalar: &dyn Fn(&Prediction) -> Result<f64>,
) -> Result<Vec<f64>> {
    let mut shap = vec![0.0; n_features];
    let mut perm: Vec<usize> = (0..n_features).collect();

    for &bg_idx in bg_indices {
        for _ in 0..PERMUTATIONS_PER_BG {
            perm.shuffle(rng);
            let mut hybrid = features.row(bg_idx).to_vec();

            let mut prev = predict_scalar(&model.predict(&row_as_array(&hybrid))?)?;
            for &j in &perm {
                hybrid[j] = x[j];
                let next = predict_scalar(&model.predict(&row_as_array(&hybrid))?)?;
                shap[j] += next - prev;
                prev = next;
            }
        }
    }

    let total_draws = (bg_indices.len() * PERMUTATIONS_PER_BG) as f64;
    for v in shap.iter_mut() {
        *v /= total_draws;
    }
    Ok(shap)
}

/// Randomly sample (without replacement) up to `n_background` row indices
/// out of `n_samples`, seeded for reproducibility. A fixed prefix (the
/// first `n_background` rows) would silently bias the base value and every
/// permutation draw whenever the data is sorted by target, by location, or
/// by any other structure -- common in the geospatial datasets this crate
/// targets.
fn sample_background_indices(n_samples: usize, n_background: usize, seed: u64) -> Vec<usize> {
    let bg_size = n_background.min(n_samples);
    let mut indices: Vec<usize> = (0..n_samples).collect();
    let mut rng = StdRng::seed_from_u64(seed);
    indices.shuffle(&mut rng);
    indices.truncate(bg_size);
    indices
}

fn global_importance_from(
    explanations: &[ShapValues],
    names: &[String],
    n_features: usize,
) -> Vec<(String, f64)> {
    let mut global_imp = vec![0.0; n_features];
    for exp in explanations {
        for (j, &v) in exp.values.iter().enumerate() {
            global_imp[j] += v.abs();
        }
    }
    let n = explanations.len().max(1) as f64;
    names
        .iter()
        .zip(&global_imp)
        .map(|(name, &imp)| (name.clone(), imp / n))
        .collect()
}

/// Compute permutation-SHAP values for a regression model.
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

    let bg_indices = sample_background_indices(n_samples, n_background, 42);
    let predict_scalar = |p: &Prediction| -> Result<f64> {
        match p {
            Prediction::Regression { predicted, .. } => Ok(predicted[0]),
            _ => Err(crate::SmeltError::IncompatiblePrediction(
                "Expected regression".into(),
            )),
        }
    };

    let bg_features = features.select(ndarray::Axis(0), &bg_indices);
    let bg_pred = model.predict(&bg_features)?;
    let bg_vals = match &bg_pred {
        Prediction::Regression { predicted, .. } => predicted.clone(),
        _ => {
            return Err(crate::SmeltError::IncompatiblePrediction(
                "Expected regression".into(),
            ));
        }
    };
    let base_value = bg_vals.iter().sum::<f64>() / bg_vals.len() as f64;

    let mut rng = StdRng::seed_from_u64(7);
    let mut explanations = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let prediction = predict_scalar(&model.predict(&features.select(ndarray::Axis(0), &[i]))?)?;
        let x = features.row(i).to_vec();
        let values = permutation_shap_row(
            model,
            &x,
            &bg_indices,
            features,
            n_features,
            &mut rng,
            &predict_scalar,
        )?;
        explanations.push(ShapValues {
            base_value,
            values,
            feature_names: names.clone(),
            prediction,
        });
    }

    let global_importance = global_importance_from(&explanations, &names, n_features);
    Ok(ShapResult {
        explanations,
        global_importance,
    })
}

/// Compute permutation-SHAP values for classification (targets one class's
/// probability).
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

    let bg_indices = sample_background_indices(n_samples, n_background, 42);
    let predict_scalar = move |p: &Prediction| -> Result<f64> {
        match p {
            Prediction::Classification {
                probabilities: Some(probs),
                ..
            } => probs
                .first()
                .and_then(|row| row.get(target_class))
                .copied()
                .ok_or_else(|| {
                    crate::SmeltError::InvalidParameter(format!(
                        "target_class {target_class} out of range"
                    ))
                }),
            _ => Err(crate::SmeltError::IncompatiblePrediction(
                "Requires probabilities".into(),
            )),
        }
    };

    let bg_features = features.select(ndarray::Axis(0), &bg_indices);
    let bg_pred = model.predict(&bg_features)?;
    let base_value = match &bg_pred {
        Prediction::Classification {
            probabilities: Some(probs),
            ..
        } => probs.iter().map(|p| p[target_class]).sum::<f64>() / probs.len() as f64,
        _ => {
            return Err(crate::SmeltError::IncompatiblePrediction(
                "Requires probabilities".into(),
            ));
        }
    };

    let mut rng = StdRng::seed_from_u64(7);
    let mut explanations = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let prediction = predict_scalar(&model.predict(&features.select(ndarray::Axis(0), &[i]))?)?;
        let x = features.row(i).to_vec();
        let values = permutation_shap_row(
            model,
            &x,
            &bg_indices,
            features,
            n_features,
            &mut rng,
            &predict_scalar,
        )?;
        explanations.push(ShapValues {
            base_value,
            values,
            feature_names: names.clone(),
            prediction,
        });
    }

    let global_importance = global_importance_from(&explanations, &names, n_features);
    Ok(ShapResult {
        explanations,
        global_importance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::Learner;
    use crate::learner::tree::decision_tree::DecisionTree;
    use ndarray::array;

    /// Efficiency property (the defining property of Shapley values): the
    /// prediction must equal base_value + sum(shap_values), up to Monte
    /// Carlo noise from a finite number of permutations. The previous
    /// implementation (a simple with/without marginal-contribution
    /// difference) had no reason to satisfy this exactly and didn't.
    #[test]
    fn shap_values_satisfy_the_efficiency_property() {
        let features = array![
            [1.0, 10.0, -3.0],
            [2.0, 8.0, 1.0],
            [3.0, 6.0, 4.0],
            [4.0, 4.0, -1.0],
            [5.0, 2.0, 2.0],
            [6.0, 0.0, -2.0],
        ];
        let target = vec![2.0, 5.0, 9.0, 7.0, 12.0, 6.0];
        let task = RegressionTask::new("shap_eff", features, target).unwrap();

        let mut dt = DecisionTree::default();
        let model = dt.train_regress(&task).unwrap();
        let result = tree_shap_regress(&*model, &task, 6).unwrap();

        for exp in &result.explanations {
            let reconstructed: f64 = exp.base_value + exp.values.iter().sum::<f64>();
            assert!(
                (reconstructed - exp.prediction).abs() < 1e-6,
                "efficiency violated: base_value ({:.4}) + sum(shap) ({:.4}) = {:.4}, \
                 expected prediction {:.4}",
                exp.base_value,
                exp.values.iter().sum::<f64>(),
                reconstructed,
                exp.prediction
            );
        }
    }

    #[test]
    fn shap_values_satisfy_efficiency_for_classification() {
        let features = array![[0.0], [0.5], [1.0], [1.5], [2.0], [2.5], [3.0], [3.5]];
        let target = vec![0, 0, 0, 0, 1, 1, 1, 1];
        let task = ClassificationTask::new("shap_eff_c", features, target).unwrap();

        let mut dt = DecisionTree::default();
        let model = dt.train_classif(&task).unwrap();
        let result = tree_shap_classif(&*model, &task, 8, 1).unwrap();

        for exp in &result.explanations {
            let reconstructed: f64 = exp.base_value + exp.values.iter().sum::<f64>();
            assert!(
                (reconstructed - exp.prediction).abs() < 1e-6,
                "efficiency violated: {:.4} + {:.4} != {:.4}",
                exp.base_value,
                exp.values.iter().sum::<f64>(),
                exp.prediction
            );
        }
    }

    /// Background sampling must not just take the first N rows -- data
    /// sorted by target (common after a groupby/sort in a pipeline) would
    /// silently bias the base value.
    #[test]
    fn background_sample_is_not_just_the_first_rows() {
        let n = 50;
        let indices = sample_background_indices(n, 10, 42);
        assert_eq!(indices.len(), 10);
        assert!(
            indices.iter().any(|&i| i >= 10),
            "background sample should not be limited to the first 10 rows: {indices:?}"
        );
    }
}
