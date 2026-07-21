//! Extreme Learning Machine (ELM): a single-hidden-layer feedforward
//! network whose input-to-hidden weights and biases are fixed random
//! values (never trained) -- only the hidden-to-output weights are learned,
//! via a closed-form ridge-regularized least-squares solve. No iterative
//! backpropagation at all, which is what makes it "extreme"-ly fast to fit.
//!
//! Huang, G.-B., Zhu, Q.-Y., & Siew, C.-K. (2006). "Extreme learning
//! machine: Theory and applications." Neurocomputing, 70(1-3), 489-501.

use crate::learner::{Learner, LearnerProperties, TrainedModel};
use crate::prediction::Prediction;
use crate::task::{ClassificationTask, RegressionTask, Task};
use crate::{Result, SmeltError};
use ndarray::{Array1, Array2};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

/// Hidden-layer activation function.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Activation {
    /// `1 / (1 + e^-x)`. Default -- the original paper's activation.
    Sigmoid,
    /// `tanh(x)`.
    Tanh,
    /// `max(0, x)`.
    Relu,
}

impl Activation {
    fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::Relu => x.max(0.0),
        }
    }
}

/// Computes per-feature mean/std over `features` and returns
/// `(standardized, mean, std)`, where `standardized` has mean 0 and unit
/// variance per column (a constant column gets `std` floored to `1.0`,
/// leaving it as literal zeros rather than dividing by zero -- a constant
/// column carries no information regardless of scale). ELM's fixed random
/// input-to-hidden weights are drawn `Uniform(-1,1)`, which implicitly
/// assumes inputs already sit on an O(1) scale (Huang et al.'s original
/// benchmarks are normalized). With features on the realistic real-world
/// scales common in this crate's GIS-adjacent niche (UTM coordinates,
/// elevations in meters), `w . x` saturates the sigmoid/tanh activation and
/// the hidden layer degenerates -- the model still "trains" without error
/// but predicts near-useless output. Standardizing internally (and storing
/// `mean`/`std` on the trained model to apply identically at predict time,
/// see [`standardize_apply`]) makes ELM scale-invariant like the rest of
/// the catalog, at zero asymptotic cost.
fn standardize_fit(features: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let n = features.nrows() as f64;
    let mean = features
        .mean_axis(ndarray::Axis(0))
        .expect("features has at least one row");
    let n_features = mean.len();
    let mut std = Array1::zeros(n_features);
    for j in 0..n_features {
        let var = features
            .column(j)
            .iter()
            .map(|&v| (v - mean[j]).powi(2))
            .sum::<f64>()
            / n;
        std[j] = if var.sqrt() < 1e-12 { 1.0 } else { var.sqrt() };
    }
    let standardized = standardize_apply(features, &mean, &std);
    (standardized, mean, std)
}

/// Sample-weighted variant of [`standardize_fit`]: mean `Σ w·v / Σw` and
/// variance `Σ w·(v-mean)² / Σw` per column. With per-sample weights the
/// standardization statistics must be weighted, or a weight of `k` and `k`
/// duplicated rows would standardize the same data differently — breaking
/// the weight-k ≡ k-duplicates equivalence before the output solve runs.
/// The unweighted path keeps calling [`standardize_fit`] untouched.
fn standardize_fit_weighted(
    features: &Array2<f64>,
    weights: &[f64],
) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let total: f64 = weights.iter().sum();
    let n_features = features.ncols();
    let mut mean = Array1::zeros(n_features);
    let mut std = Array1::zeros(n_features);
    for j in 0..n_features {
        let col = features.column(j);
        let m = col.iter().zip(weights).map(|(&v, &w)| w * v).sum::<f64>() / total;
        let var = col
            .iter()
            .zip(weights)
            .map(|(&v, &w)| w * (v - m).powi(2))
            .sum::<f64>()
            / total;
        mean[j] = m;
        std[j] = if var.sqrt() < 1e-12 { 1.0 } else { var.sqrt() };
    }
    let standardized = standardize_apply(features, &mean, &std);
    (standardized, mean, std)
}

/// Applies a previously-fit `(mean, std)` standardization to `features`.
/// Used both to finish [`standardize_fit`] and, at predict time, to apply
/// the exact same transform the model was trained on.
fn standardize_apply(features: &Array2<f64>, mean: &Array1<f64>, std: &Array1<f64>) -> Array2<f64> {
    let mut out = features.clone();
    for j in 0..mean.len() {
        for i in 0..features.nrows() {
            out[[i, j]] = (features[[i, j]] - mean[j]) / std[j];
        }
    }
    out
}

/// Solves the symmetric positive-definite system `Ax = b` via Gaussian
/// elimination with partial pivoting -- `A = HᵀH + λI` is SPD for `λ > 0`,
/// same solver shape as `regularized.rs::solve` for Ridge, hand-rolled
/// separately per this crate's per-module numeric-routine convention (see
/// `kriging_hybrid.rs`'s docs) rather than factored into a shared helper.
fn solve_spd(a: &Array2<f64>, b: &Array1<f64>) -> Option<Array1<f64>> {
    let n = a.nrows();
    let mut aug = Array2::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            let val = aug[[row, col]].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            return None;
        }
        if max_row != col {
            for j in 0..=n {
                aug.swap((col, j), (max_row, j));
            }
        }
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / aug[[col, col]];
            for j in col..=n {
                aug[[row, j]] -= factor * aug[[col, j]];
            }
        }
    }
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        x[i] = aug[[i, n]];
        for j in (i + 1)..n {
            x[i] -= aug[[i, j]] * x[j];
        }
        x[i] /= aug[[i, i]];
    }
    Some(x)
}

/// Extreme Learning Machine.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use ndarray::array;
///
/// let features = array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]];
/// let target = vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0];
/// let task = RegressionTask::new("elm", features, target).unwrap();
///
/// let mut elm = ExtremeLearningMachine::new().with_n_hidden(20).with_seed(1);
/// let model = elm.train_regress(&task).unwrap();
/// ```
pub struct ExtremeLearningMachine {
    n_hidden: usize,
    activation: Activation,
    regularization: f64,
    seed: u64,
}

impl Default for ExtremeLearningMachine {
    fn default() -> Self {
        Self::new()
    }
}

impl ExtremeLearningMachine {
    /// Creates an ELM with 100 hidden units, sigmoid activation, ridge
    /// regularization `1e-3` (stabilizes the output-weight solve and
    /// generalizes better than the pure Moore-Penrose pseudoinverse the
    /// original paper describes for the un-regularized case), and seed 42.
    pub fn new() -> Self {
        Self {
            n_hidden: 100,
            activation: Activation::Sigmoid,
            regularization: 1e-3,
            seed: 42,
        }
    }
    /// Sets the number of random hidden units. `0` is stored as-is and
    /// rejected with a clear error at train time (5th audit, LOW-D) — the
    /// old silent `max(1)` clamp trained a different model than requested
    /// without any signal.
    pub fn with_n_hidden(mut self, n: usize) -> Self {
        self.n_hidden = n;
        self
    }

    /// A zero-width hidden layer has no basis functions to regress on: the
    /// model would degenerate to a constant. The builder doesn't return
    /// `Result`, so the check runs at the start of training.
    fn check_n_hidden(&self) -> Result<()> {
        if self.n_hidden == 0 {
            return Err(SmeltError::InvalidParameter(
                "elm n_hidden must be at least 1 (got 0): with no hidden units there is \
                 nothing to regress the outputs on"
                    .into(),
            ));
        }
        Ok(())
    }
    /// Sets the hidden-layer activation function.
    pub fn with_activation(mut self, a: Activation) -> Self {
        self.activation = a;
        self
    }
    /// Sets the ridge penalty applied to the output-weight solve.
    pub fn with_regularization(mut self, l: f64) -> Self {
        self.regularization = l;
        self
    }
    /// Sets the RNG seed for the random input weights and biases.
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Random input-to-hidden weights (`n_features x n_hidden`) and biases
    /// (`n_hidden`), drawn `Uniform(-1, 1)` and then fixed for the model's
    /// lifetime -- these are the "extreme" (never trained) part.
    fn random_hidden_layer(&self, n_features: usize) -> (Array2<f64>, Array1<f64>) {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let w = Array2::from_shape_fn((n_features, self.n_hidden), |_| rng.random_range(-1.0..1.0));
        let b = Array1::from_shape_fn(self.n_hidden, |_| rng.random_range(-1.0..1.0));
        (w, b)
    }

    fn hidden_output(
        &self,
        features: &Array2<f64>,
        w: &Array2<f64>,
        b: &Array1<f64>,
    ) -> Array2<f64> {
        let mut h = features.dot(w);
        for mut row in h.rows_mut() {
            for (v, &bias) in row.iter_mut().zip(b.iter()) {
                *v = self.activation.apply(*v + bias);
            }
        }
        h
    }

    /// Solves `(HᵀH + λI) β = Hᵀ T` one output column at a time — or, with
    /// per-sample weights, the weighted ridge system `(HᵀWH + λI) β = HᵀWT`
    /// with `W = diag(weights)`. The ridge penalty `λ` is NOT scaled by the
    /// total weight, so an integer weight `k` is exactly equivalent to
    /// duplicating that row `k` times (same convention as [`super::Ridge`]).
    /// `W` is applied by scaling the rows of one copy of `H` ((WH)ᵀH =
    /// HᵀWH), so the unweighted path is bit-identical to the historical
    /// code and all-ones weights reproduce it exactly.
    fn solve_output_weights(
        &self,
        h: &Array2<f64>,
        targets: &Array2<f64>,
        weights: Option<&[f64]>,
    ) -> Result<Array2<f64>> {
        let n_hidden = h.ncols();
        let n_outputs = targets.ncols();
        let (mut hth, hty) = match weights {
            None => (h.t().dot(h), h.t().dot(targets)),
            Some(w) => {
                let mut hw = h.clone();
                for (i, &wi) in w.iter().enumerate() {
                    hw.row_mut(i).mapv_inplace(|v| wi * v);
                }
                (hw.t().dot(h), hw.t().dot(targets))
            }
        };
        for j in 0..n_hidden {
            hth[[j, j]] += self.regularization;
        }

        let mut beta = Array2::zeros((n_hidden, n_outputs));
        for k in 0..n_outputs {
            let col = hty.column(k).to_owned();
            let solved = solve_spd(&hth, &col).ok_or_else(|| {
                SmeltError::NumericalError(
                    "singular system solving ELM output weights -- try increasing regularization"
                        .into(),
                )
            })?;
            beta.column_mut(k).assign(&solved);
        }
        Ok(beta)
    }
}

/// A trained Extreme Learning Machine.
#[derive(Clone, Serialize, Deserialize)]
pub struct TrainedELM {
    pub(crate) input_weights: Array2<f64>,
    pub(crate) biases: Array1<f64>,
    pub(crate) output_weights: Array2<f64>,
    pub(crate) activation: Activation,
    pub(crate) is_classifier: bool,
    pub(crate) n_features: usize,
    pub(crate) feature_mean: Array1<f64>,
    pub(crate) feature_std: Array1<f64>,
}

impl TrainedModel for TrainedELM {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        crate::validate::check_n_features(features, self.n_features)?;
        let standardized = standardize_apply(features, &self.feature_mean, &self.feature_std);
        let mut h = standardized.dot(&self.input_weights);
        for mut row in h.rows_mut() {
            for (v, &bias) in row.iter_mut().zip(self.biases.iter()) {
                *v = self.activation.apply(*v + bias);
            }
        }
        let raw = h.dot(&self.output_weights);

        if self.is_classifier {
            let mut predicted = Vec::with_capacity(raw.nrows());
            let mut probabilities = Vec::with_capacity(raw.nrows());
            for row in raw.rows() {
                // Softmax over the raw (regressed one-hot) scores turns
                // them into a valid probability distribution for output.
                let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp: Vec<f64> = row.iter().map(|&v| (v - max).exp()).collect();
                let sum: f64 = exp.iter().sum::<f64>().max(1e-300);
                let probs: Vec<f64> = exp.iter().map(|&v| v / sum).collect();
                let pred = probs
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                predicted.push(pred);
                probabilities.push(probs);
            }
            Ok(Prediction::Classification {
                predicted,
                truth: None,
                probabilities: Some(probabilities),
            })
        } else {
            let predicted: Vec<f64> = raw.column(0).to_vec();
            Ok(Prediction::regression(predicted))
        }
    }

    fn to_serializable(&self) -> Option<crate::serialize::SerializableModel> {
        Some(crate::serialize::SerializableModel::ExtremeLearningMachine(
            self.clone(),
        ))
    }
}

impl Learner for ExtremeLearningMachine {
    fn id(&self) -> &str {
        "elm"
    }

    fn properties(&self) -> LearnerProperties {
        LearnerProperties::classifier_regressor()
            .with_weights()
            .with_proba()
            .with_serializable()
    }

    fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        self.check_n_hidden()?;
        crate::validate::check_no_nan(task.features())?;
        let features = task.features();
        let target = task.target();
        let n_classes = task.n_classes();
        let n_features = task.n_features();
        let sample_weights = task.weights();

        let (standardized, feature_mean, feature_std) = match sample_weights {
            None => standardize_fit(features),
            Some(sw) => standardize_fit_weighted(features, sw),
        };
        // The random projection depends only on the seed and n_features,
        // never on the weights: same seed ⇒ same hidden layer.
        let (w, b) = self.random_hidden_layer(n_features);
        let h = self.hidden_output(&standardized, &w, &b);

        let mut one_hot = Array2::zeros((task.n_samples(), n_classes));
        for (i, &label) in target.iter().enumerate() {
            one_hot[[i, label]] = 1.0;
        }

        let output_weights = self.solve_output_weights(&h, &one_hot, sample_weights)?;

        Ok(Box::new(TrainedELM {
            input_weights: w,
            biases: b,
            output_weights,
            activation: self.activation,
            is_classifier: true,
            n_features,
            feature_mean,
            feature_std,
        }))
    }

    fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
        self.check_n_hidden()?;
        crate::validate::check_no_nan(task.features())?;
        let features = task.features();
        let target = task.target();
        let n_features = task.n_features();
        let sample_weights = task.weights();

        let (standardized, feature_mean, feature_std) = match sample_weights {
            None => standardize_fit(features),
            Some(sw) => standardize_fit_weighted(features, sw),
        };
        let (w, b) = self.random_hidden_layer(n_features);
        let h = self.hidden_output(&standardized, &w, &b);
        let targets = Array2::from_shape_vec((target.len(), 1), target.to_vec())
            .expect("target length matches n_samples by construction");

        let output_weights = self.solve_output_weights(&h, &targets, sample_weights)?;

        Ok(Box::new(TrainedELM {
            input_weights: w,
            biases: b,
            output_weights,
            activation: self.activation,
            is_classifier: false,
            n_features,
            feature_mean,
            feature_std,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn registered_id_matches() {
        assert_eq!(ExtremeLearningMachine::new().id(), "elm");
    }

    /// Regression test (5th audit, LOW-D): `with_n_hidden(0)` used to be
    /// silently clamped to 1 by the builder — a different model than
    /// requested, with no signal. It must now be a clear train-time error
    /// on both task types.
    #[test]
    fn zero_hidden_units_is_rejected_at_train() {
        let features = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let classif =
            ClassificationTask::new("elm0_c", features.clone(), vec![0, 0, 1, 1]).unwrap();
        let regress = RegressionTask::new("elm0_r", features, vec![0.0, 1.0, 2.0, 3.0]).unwrap();

        let Err(err) = ExtremeLearningMachine::new()
            .with_n_hidden(0)
            .train_classif(&classif)
        else {
            panic!("n_hidden=0 must be rejected for classification");
        };
        assert!(
            matches!(err, SmeltError::InvalidParameter(_)) && format!("{err}").contains("n_hidden"),
            "got: {err}"
        );

        let Err(err) = ExtremeLearningMachine::new()
            .with_n_hidden(0)
            .train_regress(&regress)
        else {
            panic!("n_hidden=0 must be rejected for regression");
        };
        assert!(
            matches!(err, SmeltError::InvalidParameter(_)) && format!("{err}").contains("n_hidden"),
            "got: {err}"
        );
    }

    #[test]
    fn fits_a_linear_trend_for_regression() {
        let mut rng = StdRng::seed_from_u64(1);
        let n = 300;
        let mut feats = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);
        for _ in 0..n {
            let x: f64 = rng.random::<f64>() * 10.0;
            feats.push(x);
            target.push(2.0 * x + 1.0 + rng.random::<f64>() * 0.1);
        }
        let features = Array2::from_shape_vec((n, 1), feats).unwrap();
        let task = RegressionTask::new("elm_lin", features.clone(), target.clone()).unwrap();

        let mut elm = ExtremeLearningMachine::new().with_n_hidden(50).with_seed(2);
        let model = elm.train_regress(&task).unwrap();
        let pred = model.predict(&features).unwrap();
        let Prediction::Regression { predicted, .. } = pred else {
            panic!("expected regression");
        };
        let rmse = (predicted
            .iter()
            .zip(&target)
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / n as f64)
            .sqrt();
        assert!(
            rmse < 1.0,
            "should fit a clear linear trend well, got RMSE={rmse}"
        );
    }

    #[test]
    fn separates_a_simple_classification_boundary() {
        let mut rng = StdRng::seed_from_u64(3);
        let n = 400;
        let mut feats = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);
        for _ in 0..n {
            let x0: f64 = rng.random();
            let x1: f64 = rng.random();
            feats.push(x0);
            feats.push(x1);
            target.push(if x0 + x1 > 1.0 { 1usize } else { 0 });
        }
        let features = Array2::from_shape_vec((n, 2), feats).unwrap();
        let task =
            ClassificationTask::new("elm_classif", features.clone(), target.clone()).unwrap();

        let mut elm = ExtremeLearningMachine::new().with_n_hidden(50).with_seed(4);
        let model = elm.train_classif(&task).unwrap();
        let pred = model.predict(&features).unwrap();
        let Prediction::Classification {
            predicted,
            probabilities,
            ..
        } = pred
        else {
            panic!("expected classification");
        };
        let correct = predicted
            .iter()
            .zip(&target)
            .filter(|(p, t)| *p == *t)
            .count();
        let acc = correct as f64 / n as f64;
        assert!(
            acc > 0.85,
            "should separate a simple linear boundary well, got acc={acc}"
        );

        let probs = probabilities.unwrap();
        for row in &probs {
            let sum: f64 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "probabilities should sum to 1, got {sum}"
            );
        }
    }

    #[test]
    fn multiclass_probabilities_are_well_formed() {
        let features = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [5.0, 5.0],
            [5.1, 4.9],
            [4.9, 5.1],
            [10.0, 0.0],
            [10.1, 0.1],
            [9.9, -0.1],
        ];
        let target = vec![0usize, 0, 0, 1, 1, 1, 2, 2, 2];
        let task = ClassificationTask::new("elm_multi", features.clone(), target.clone()).unwrap();

        let mut elm = ExtremeLearningMachine::new().with_n_hidden(30).with_seed(5);
        let model = elm.train_classif(&task).unwrap();
        let pred = model.predict(&features).unwrap();
        let Prediction::Classification { predicted, .. } = pred else {
            panic!("expected classification");
        };
        let correct = predicted
            .iter()
            .zip(&target)
            .filter(|(p, t)| *p == *t)
            .count();
        assert!(
            correct as f64 / target.len() as f64 > 0.6,
            "should do reasonably on 3 well-separated clusters"
        );
    }

    #[test]
    fn activation_functions_produce_bounded_or_sensible_output() {
        assert!((Activation::Sigmoid.apply(0.0) - 0.5).abs() < 1e-9);
        assert!(Activation::Sigmoid.apply(100.0) <= 1.0);
        assert!(Activation::Tanh.apply(0.0).abs() < 1e-9);
        assert_eq!(Activation::Relu.apply(-5.0), 0.0);
        assert_eq!(Activation::Relu.apply(5.0), 5.0);
    }

    /// Regression test for the HIGH finding: ELM's `Uniform(-1,1)` random
    /// input weights implicitly assume features on an O(1) scale. On a
    /// perfectly linear relationship (the easiest possible fit) with
    /// features at a realistic GIS scale (UTM-like coordinates, ~1e4), the
    /// unstandardized model saturated the sigmoid and produced relative
    /// RMSE ~0.85 (confirmed via a probe before this fix); with internal
    /// standardization it should fit just as well as it does on [0,10]
    /// features.
    #[test]
    fn fits_a_linear_trend_at_realistic_gis_feature_scales() {
        let mut rng = StdRng::seed_from_u64(1);
        let n = 400;
        let mut feats = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);
        for _ in 0..n {
            let x: f64 = rng.random::<f64>() * 1.0e4;
            feats.push(x);
            target.push(2.0 * x + 1.0);
        }
        let features = Array2::from_shape_vec((n, 1), feats).unwrap();
        let task = RegressionTask::new("elm_gis_scale", features.clone(), target.clone()).unwrap();

        let mut elm = ExtremeLearningMachine::new()
            .with_n_hidden(100)
            .with_seed(2);
        let model = elm.train_regress(&task).unwrap();
        let pred = model.predict(&features).unwrap();
        let Prediction::Regression { predicted, .. } = pred else {
            panic!("expected regression");
        };
        let mse: f64 = predicted
            .iter()
            .zip(&target)
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / n as f64;
        let target_var: f64 = {
            let mean = target.iter().sum::<f64>() / n as f64;
            target.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / n as f64
        };
        let relative_rmse = (mse / target_var).sqrt();
        assert!(
            relative_rmse < 0.05,
            "should fit a perfect linear trend at GIS feature scales, got relative RMSE={relative_rmse}"
        );
    }

    #[test]
    fn rejects_wrong_feature_count_at_predict() {
        let features = array![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [0.0, 2.0]];
        let target = vec![2.0, 4.0, 6.0, 8.0];
        let task = RegressionTask::new("elm_dim", features, target).unwrap();
        let mut elm = ExtremeLearningMachine::new().with_n_hidden(10).with_seed(1);
        let model = elm.train_regress(&task).unwrap();

        let wrong = array![[1.0, 2.0, 3.0]];
        assert!(model.predict(&wrong).is_err());
    }
}
