//! `AutoTuner`: bundle a (learner-factory + tuner + inner resampling +
//! measure + parameter space) as a single [`Learner`], the equivalent of
//! mlr3tuning's `AutoTuner`.
//!
//! # Why this exists: nested cross-validation for free
//!
//! Tuning a learner's hyperparameters and then honestly estimating that tuned
//! learner's generalization error are two *separate* resampling loops. If you
//! tune on the same data you later score on, the reported score is optimistic:
//! the winning configuration was chosen partly by fitting the noise of that
//! particular split (selection bias). The textbook remedy is **nested CV** —
//! an *outer* resampling whose every training fold runs its *own* independent
//! inner tuning, so each outer test fold is scored by a model whose
//! hyperparameters never saw it.
//!
//! Writing that by hand is fiddly and easy to get subtly wrong (leaking the
//! outer test rows into the inner tuning is the classic mistake). Because
//! `AutoTuner` *is* a [`Learner`] — its `train_*` runs the whole inner
//! tune-then-refit on whatever task it is handed — nested CV becomes a plain
//! benchmark call:
//!
//! ```text
//! benchmark::resample_classif(&mut auto_tuner, task, &outer_cv, &measures)
//! ```
//!
//! The outer [`crate::benchmark`] loop slices `task` into outer folds and
//! calls `AutoTuner::train_classif` on each outer *training* fold only. The
//! inner tuner therefore only ever sees that fold's training rows — the outer
//! test rows are structurally unreachable, so there is no leakage to remember
//! to prevent. (This is the anti-leakage property exercised by the crate's
//! nested-CV tests.)
//!
//! # Cost
//!
//! Nested CV multiplies work: `outer_folds × trials × inner_folds` model fits
//! (where `trials` is the tuner's configuration count — grid size, `n_iter`,
//! etc.), plus one final refit per outer fold. Budget accordingly.
//!
//! # Registry
//!
//! Like `Bagging`/`Stacking`/`TargetTransformRegressor`, `AutoTuner` wraps a
//! base-learner factory with no sensible default, so it is **not**
//! constructible via [`crate::learner::registry::learner_from_id`].

use crate::Result;
use crate::learner::{Learner, LearnerProperties, TrainedModel};
use crate::measure::Measure;
use crate::prediction::Prediction;
use crate::resample::Resample;
use crate::task::{ClassificationTask, RegressionTask};
use crate::tuning::{
    BayesianOptimizer, GridSearch, Hyperband, ParamDistribution, ParamGrid, ParamSet, ParamSpace,
    ParamValue, RandomSearch, TuneResult,
};
use ndarray::Array2;
use std::sync::Arc;

/// Which inner tuner drives the search, together with its parameter
/// space/grid and tuner-specific knobs.
///
/// The four tuners ([`GridSearch`], [`RandomSearch`], [`BayesianOptimizer`],
/// [`Hyperband`]) do not share a Rust trait — their `tune_*` methods differ
/// in signature (grid vs. distribution space; `Hyperband` drives its own CV
/// budget and takes no external resampling). Rather than invent a lowest-
/// common-denominator trait, this enum carries exactly the inputs each tuner
/// needs, so a `Grid` variant can only ever be paired with a [`ParamGrid`]
/// and the distribution-based tuners with a [`ParamSpace`] — a mismatch is
/// unrepresentable.
pub enum TunerSpec {
    /// Exhaustive [`GridSearch`] over every combination in the grid. Ignores
    /// [`AutoTuner`]'s seed (grid search is deterministic).
    Grid(ParamGrid),
    /// [`RandomSearch`]: sample `n_iter` configurations from `space`.
    Random {
        /// Distributions to sample hyperparameters from.
        space: ParamSpace,
        /// Number of random configurations to draw and evaluate.
        n_iter: usize,
    },
    /// [`BayesianOptimizer`] (TPE) over `space`.
    Bayesian {
        /// Distributions to sample hyperparameters from.
        space: ParamSpace,
        /// Total optimization iterations.
        n_iter: usize,
        /// Initial random iterations before TPE-guided sampling begins.
        n_initial: usize,
    },
    /// [`Hyperband`] successive halving over `space`. Uses its own internal
    /// cross-validation budget (`max_folds`/`eta`) and therefore **ignores**
    /// the inner resampling passed to [`AutoTuner::new`].
    Hyperband {
        /// Distributions to sample hyperparameters from.
        space: ParamSpace,
        /// Maximum CV fold budget for surviving configurations.
        max_folds: usize,
        /// Halving rate: keep the top `1/eta` each round (must be >= 2).
        eta: usize,
    },
}

impl TunerSpec {
    /// A single representative `ParamSet` from this spec's space/grid, used to
    /// build a probe learner for [`AutoTuner::supports_weights`] without
    /// running the RNG: the first value of each grid axis, or the lower bound
    /// / first choice of each distribution.
    fn representative_params(&self) -> ParamSet {
        fn representative_value(dist: &ParamDistribution) -> ParamValue {
            match dist {
                ParamDistribution::Uniform(lo, _) => ParamValue::Float(*lo),
                ParamDistribution::LogUniform(lo, _) => ParamValue::Float(*lo),
                ParamDistribution::Choice(values) => values
                    .first()
                    .cloned()
                    // Empty Choice is rejected by the tuners' own
                    // validate_param_space before any real run; the probe just
                    // needs *something* it won't index-panic on.
                    .unwrap_or(ParamValue::Float(0.0)),
            }
        }
        match self {
            TunerSpec::Grid(grid) => grid
                .iter()
                .filter_map(|(k, vals)| vals.first().map(|v| (k.clone(), v.clone())))
                .collect(),
            TunerSpec::Random { space, .. }
            | TunerSpec::Bayesian { space, .. }
            | TunerSpec::Hyperband { space, .. } => space
                .iter()
                .map(|(k, dist)| (k.clone(), representative_value(dist)))
                .collect(),
        }
    }
}

/// A parameterized learner factory: builds a fresh [`Learner`] from a sampled
/// hyperparameter set. Exactly the signature the four tuners already consume.
type Factory = Arc<dyn Fn(&ParamSet) -> Box<dyn Learner> + Send + Sync>;

/// Wraps `(factory + tuner + inner resampling + measure + parameter space)` as
/// a single [`Learner`], so that training it runs the inner hyperparameter
/// search on whatever task it is given, then refits the winning configuration
/// on that whole task.
///
/// See the [module documentation](self) for how this makes nested cross-
/// validation a plain [`crate::benchmark`] call.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use smelt_ml::learner::{AutoTuner, TunerSpec};
/// use smelt_ml::tuning::{ParamGrid, ParamValue};
/// use ndarray::array;
///
/// let features = array![
///     [0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [0.0, 0.2],
///     [1.0, 1.0], [1.1, 0.9], [0.9, 1.1], [1.0, 0.9]
/// ];
/// let target = vec![0, 0, 0, 0, 1, 1, 1, 1];
/// let task = ClassificationTask::new("tune", features, target).unwrap();
///
/// let mut grid = ParamGrid::new();
/// grid.insert("max_depth".into(), vec![ParamValue::Int(1), ParamValue::Int(8)]);
///
/// let mut auto = AutoTuner::new(
///     |p| Box::new(DecisionTree::new().with_max_depth(p["max_depth"].as_usize().unwrap())),
///     TunerSpec::Grid(grid),
///     Box::new(CrossValidation::new(3).with_seed(0)),
///     Box::new(Accuracy),
/// );
///
/// // Nested CV: outer 3-fold over the AutoTuner itself.
/// let outer = CrossValidation::new(3).with_seed(1);
/// let bench = benchmark::resample_classif(&mut auto, &task, &outer, &[&Accuracy]).unwrap();
/// assert_eq!(bench.scores.len(), 3);
/// ```
pub struct AutoTuner {
    factory: Factory,
    tuner: TunerSpec,
    resampling: Box<dyn Resample>,
    measure: Box<dyn Measure>,
    seed: u64,
}

impl AutoTuner {
    /// Build an `AutoTuner` from a parameterized learner `factory`, a
    /// [`TunerSpec`] (tuner choice + parameter space/grid), the inner
    /// `resampling` used to score each candidate, and the `measure` to
    /// optimize (its [`Measure::maximize`] direction is honoured by the
    /// tuner). The RNG seed defaults to `42`; override with [`Self::with_seed`].
    ///
    /// Note: [`TunerSpec::Hyperband`] ignores `resampling` — it drives its own
    /// internal CV budget from `max_folds`/`eta`.
    pub fn new(
        factory: impl Fn(&ParamSet) -> Box<dyn Learner> + Send + Sync + 'static,
        tuner: TunerSpec,
        resampling: Box<dyn Resample>,
        measure: Box<dyn Measure>,
    ) -> Self {
        Self {
            factory: Arc::new(factory),
            tuner,
            resampling,
            measure,
            seed: 42,
        }
    }

    /// Set the RNG seed handed to the inner tuner (RandomSearch/Bayesian/
    /// Hyperband). Ignored by [`TunerSpec::Grid`], which is deterministic.
    /// Reproducible: two runs with the same seed select the same
    /// `best_params`.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// A cloned callable for the inner tuner: the four tuners take ownership of
    /// their factory (`Fn + 'static`), so each `train_*` call hands them a
    /// fresh `Arc`-backed closure over the shared factory.
    fn tuner_factory(&self) -> impl Fn(&ParamSet) -> Box<dyn Learner> + Send + Sync + 'static {
        let f = Arc::clone(&self.factory);
        // `*f` is the unsized `dyn Fn(..)` behind the `Arc`; calling it
        // auto-borrows it as `&dyn Fn` (which implements `Fn`).
        move |p: &ParamSet| (*f)(p)
    }

    /// Run the inner tuner over a classification `task`, returning the full
    /// [`TuneResult`] (best config + history).
    fn tune_classif(&self, task: &ClassificationTask) -> Result<TuneResult> {
        let measure = &*self.measure;
        let resampling = &*self.resampling;
        match &self.tuner {
            TunerSpec::Grid(grid) => GridSearch::new(self.tuner_factory(), grid.clone())
                .tune_classif(task, resampling, measure),
            TunerSpec::Random { space, n_iter } => {
                RandomSearch::new(self.tuner_factory(), space.clone())
                    .with_n_iter(*n_iter)
                    .with_seed(self.seed)
                    .tune_classif(task, resampling, measure)
            }
            TunerSpec::Bayesian {
                space,
                n_iter,
                n_initial,
            } => BayesianOptimizer::new(self.tuner_factory(), space.clone())
                .with_n_iter(*n_iter)
                .with_n_initial(*n_initial)
                .with_seed(self.seed)
                .tune_classif(task, resampling, measure),
            TunerSpec::Hyperband {
                space,
                max_folds,
                eta,
            } => Hyperband::new(self.tuner_factory(), space.clone())
                .with_max_folds(*max_folds)
                .with_eta(*eta)
                .with_seed(self.seed)
                .tune_classif(task, measure),
        }
    }

    /// Run the inner tuner over a regression `task`.
    fn tune_regress(&self, task: &RegressionTask) -> Result<TuneResult> {
        let measure = &*self.measure;
        let resampling = &*self.resampling;
        match &self.tuner {
            TunerSpec::Grid(grid) => GridSearch::new(self.tuner_factory(), grid.clone())
                .tune_regress(task, resampling, measure),
            TunerSpec::Random { space, n_iter } => {
                RandomSearch::new(self.tuner_factory(), space.clone())
                    .with_n_iter(*n_iter)
                    .with_seed(self.seed)
                    .tune_regress(task, resampling, measure)
            }
            TunerSpec::Bayesian {
                space,
                n_iter,
                n_initial,
            } => BayesianOptimizer::new(self.tuner_factory(), space.clone())
                .with_n_iter(*n_iter)
                .with_n_initial(*n_initial)
                .with_seed(self.seed)
                .tune_regress(task, resampling, measure),
            TunerSpec::Hyperband {
                space,
                max_folds,
                eta,
            } => Hyperband::new(self.tuner_factory(), space.clone())
                .with_max_folds(*max_folds)
                .with_eta(*eta)
                .with_seed(self.seed)
                .tune_regress(task, measure),
        }
    }

    /// Tune on `task`, then refit the winning configuration on the whole
    /// `task`, returning the concrete [`TrainedAutoTuner`] (which exposes
    /// [`TrainedAutoTuner::best_params`]/[`TrainedAutoTuner::history`] beyond
    /// the [`TrainedModel`] trait — the same "concrete carries more than the
    /// trait" shape as [`crate::learner::DeepForest::fit`]). [`Learner::train_classif`]
    /// just boxes this.
    pub fn fit_classif(&self, task: &ClassificationTask) -> Result<TrainedAutoTuner> {
        let result = self.tune_classif(task)?;
        let mut final_learner = (*self.factory)(&result.best_params);
        let inner = final_learner.train_classif(task)?;
        Ok(TrainedAutoTuner::from_result(inner, result))
    }

    /// Regression counterpart of [`Self::fit_classif`].
    pub fn fit_regress(&self, task: &RegressionTask) -> Result<TrainedAutoTuner> {
        let result = self.tune_regress(task)?;
        let mut final_learner = (*self.factory)(&result.best_params);
        let inner = final_learner.train_regress(task)?;
        Ok(TrainedAutoTuner::from_result(inner, result))
    }
}

impl Learner for AutoTuner {
    fn id(&self) -> &str {
        "auto_tuner"
    }

    fn properties(&self) -> LearnerProperties {
        // Delegate to a representative base learner: task support, weights,
        // proba, and importance all follow the tuned learner. The default
        // supports_weights() reads this, preserving the previous delegation.
        // The trained AutoTuner is a composite with no SerializableModel
        // variant, so serializability does not carry through.
        LearnerProperties {
            serializable: false,
            ..(*self.factory)(&self.tuner.representative_params()).properties()
        }
    }

    fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        Ok(Box::new(self.fit_classif(task)?))
    }

    fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
        Ok(Box::new(self.fit_regress(task)?))
    }
}

/// A trained [`AutoTuner`]: the refit best model, plus the tuning outcome
/// (best configuration, best score, and the full evaluation history).
/// `predict` delegates to the inner model.
pub struct TrainedAutoTuner {
    inner: Box<dyn TrainedModel>,
    best_params: ParamSet,
    best_score: f64,
    history: Vec<(ParamSet, f64)>,
}

impl TrainedAutoTuner {
    fn from_result(inner: Box<dyn TrainedModel>, result: TuneResult) -> Self {
        Self {
            inner,
            best_params: result.best_params,
            best_score: result.best_score,
            history: result.all_results,
        }
    }

    /// The hyperparameter configuration selected by the inner tuner — the one
    /// the final (returned) model was refit with.
    pub fn best_params(&self) -> &ParamSet {
        &self.best_params
    }

    /// The inner-resampling score of [`Self::best_params`] (in the measure's
    /// own direction; see [`Measure::maximize`]).
    pub fn best_score(&self) -> f64 {
        self.best_score
    }

    /// Every configuration the tuner evaluated, paired with its inner score —
    /// the full tuning history for diagnostics/plots.
    pub fn history(&self) -> &[(ParamSet, f64)] {
        &self.history
    }
}

impl TrainedModel for TrainedAutoTuner {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        self.inner.predict(features)
    }

    fn feature_importance(&self) -> Option<Vec<(String, f64)>> {
        self.inner.feature_importance()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::{DecisionTree, KNearestNeighbors, RandomForest, Ridge};
    use crate::measure::{Accuracy, Rmse};
    use crate::resample::CrossValidation;
    use crate::task::Task;
    use crate::tuning::{ParamDistribution, ParamValue};
    use ndarray::Array2;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    // ── fixtures ─────────────────────────────────────────────────────────

    /// Two well-separated blobs — trivially learnable, so a tuned tree/forest
    /// reaches high accuracy.
    fn classif_task(n: usize) -> ClassificationTask {
        let features = Array2::from_shape_fn((n, 2), |(i, j)| {
            let base = if i < n / 2 { 0.0 } else { 5.0 };
            base + (j as f64) * 0.01 + (i as f64) * 0.001
        });
        let target: Vec<usize> = (0..n).map(|i| if i < n / 2 { 0 } else { 1 }).collect();
        ClassificationTask::new("c", features, target).unwrap()
    }

    /// Smooth linear-ish regression target.
    fn regress_task(n: usize) -> RegressionTask {
        let features = Array2::from_shape_fn((n, 2), |(i, j)| (i as f64) + (j as f64) * 2.0);
        let target: Vec<f64> = (0..n).map(|i| 3.0 * i as f64 + 1.0).collect();
        RegressionTask::new("r", features, target).unwrap()
    }

    fn depth_grid(values: Vec<i64>) -> ParamGrid {
        let mut g = ParamGrid::new();
        g.insert(
            "max_depth".into(),
            values.into_iter().map(ParamValue::Int).collect(),
        );
        g
    }

    fn predicted_classif(pred: Prediction) -> Vec<usize> {
        match pred {
            Prediction::Classification { predicted, .. } => predicted,
            _ => panic!("expected classification prediction"),
        }
    }

    // ── Test 1: trains and predicts reasonably (classif + regress) ───────

    #[test]
    fn autotuner_trains_and_predicts_classif_and_regress() {
        // Classification: RandomSearch over max_depth, inner CV=3.
        let task = classif_task(40);
        let mut space = ParamSpace::new();
        space.insert(
            "max_depth".into(),
            ParamDistribution::Choice(vec![ParamValue::Int(2), ParamValue::Int(6)]),
        );
        let mut auto = AutoTuner::new(
            |p| Box::new(DecisionTree::new().with_max_depth(p["max_depth"].as_usize().unwrap())),
            TunerSpec::Random { space, n_iter: 4 },
            Box::new(CrossValidation::new(3).with_seed(0)),
            Box::new(Accuracy),
        );
        let model = auto.train_classif(&task).unwrap();
        let preds = predicted_classif(model.predict(task.features()).unwrap());
        let acc = preds
            .iter()
            .zip(task.target())
            .filter(|(p, t)| p == t)
            .count() as f64
            / task.n_samples() as f64;
        assert!(
            acc > 0.9,
            "tuned tree should separate the blobs, got acc={acc}"
        );

        // Regression: Grid over Ridge alpha, inner CV=3, minimize RMSE.
        let rtask = regress_task(30);
        let mut grid = ParamGrid::new();
        grid.insert(
            "alpha".into(),
            vec![ParamValue::Float(0.01), ParamValue::Float(1.0)],
        );
        let mut rauto = AutoTuner::new(
            |p| Box::new(Ridge::new(p["alpha"].as_f64().unwrap())),
            TunerSpec::Grid(grid),
            Box::new(CrossValidation::new(3).with_seed(0)),
            Box::new(Rmse),
        );
        let rmodel = rauto.train_regress(&rtask).unwrap();
        match rmodel.predict(rtask.features()).unwrap() {
            Prediction::Regression { predicted, .. } => {
                let rmse = (predicted
                    .iter()
                    .zip(rtask.target())
                    .map(|(p, t)| (p - t).powi(2))
                    .sum::<f64>()
                    / rtask.n_samples() as f64)
                    .sqrt();
                assert!(
                    rmse < 5.0,
                    "tuned ridge should fit the linear target, rmse={rmse}"
                );
            }
            _ => panic!("expected regression prediction"),
        }
    }

    // ── Test 2: best_params ∈ space and the model was refit with them ────

    #[test]
    fn best_params_in_space_and_deeper_wins_when_depth_is_required() {
        // A two-threshold target that a depth-1 stump provably cannot fit but a
        // depth-2+ tree can, against a grid of {1, 8}. The tuner must pick 8.
        //
        // (Deliberately NOT an XOR pattern: a *greedy* axis-aligned tree can't
        // fit XOR at any depth, because the first split has zero information
        // gain, so it never splits at all — that would be a bug in the fixture,
        // not evidence the tuner picked depth. Here the label flips at x=4.5
        // and again at x=9.5, so a single threshold caps out at ~0.67 accuracy
        // while a depth-2 tree (each split has real gain) reaches 100%.)
        let features = ndarray::array![
            [0.0],
            [1.0],
            [2.0],
            [3.0], // class 1 (x < 4.5)
            [5.0],
            [6.0],
            [7.0],
            [8.0], // class 0 (4.5 < x < 9.5)
            [10.0],
            [11.0],
            [12.0],
            [13.0] // class 1 (x > 9.5)
        ];
        let target = vec![1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1];
        let task = ClassificationTask::new("two_threshold", features, target).unwrap();

        let auto = AutoTuner::new(
            |p| Box::new(DecisionTree::new().with_max_depth(p["max_depth"].as_usize().unwrap())),
            TunerSpec::Grid(depth_grid(vec![1, 8])),
            Box::new(CrossValidation::new(3).with_seed(0)),
            Box::new(Accuracy),
        );
        let trained = auto.fit_classif(&task).unwrap();

        // best_params must come from the grid.
        let chosen = trained.best_params()["max_depth"].as_i64().unwrap();
        assert!(
            chosen == 1 || chosen == 8,
            "best_params must be a grid value, got {chosen}"
        );
        assert_eq!(
            chosen, 8,
            "the two-threshold target needs depth > 1, tuner must pick 8"
        );

        // The refit final model must actually be the deep one: it fits the
        // training data perfectly, which a depth-1 stump cannot.
        let preds = predicted_classif(trained.predict(task.features()).unwrap());
        let acc = preds
            .iter()
            .zip(task.target())
            .filter(|(p, t)| p == t)
            .count() as f64
            / task.n_samples() as f64;
        assert!(
            (acc - 1.0).abs() < 1e-9,
            "depth-8 refit should fit the target perfectly, acc={acc}"
        );

        // history carries every evaluated config (2 grid points).
        assert_eq!(trained.history().len(), 2);
    }

    // ── Test 3: structural nested CV + anti-leakage probe ────────────────

    /// Records, per training call, the exact set of feature rows (by their
    /// unique feature-0 value) the learner was trained on. Reused to prove
    /// each outer fold's inner tuning only saw that fold's training rows.
    struct RowProbe {
        seen: Arc<Mutex<Vec<Vec<u64>>>>,
    }
    struct ProbeModel;
    impl TrainedModel for ProbeModel {
        fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
            Ok(Prediction::Classification {
                predicted: vec![0; features.nrows()],
                truth: None,
                probabilities: None,
            })
        }
    }
    impl Learner for RowProbe {
        fn id(&self) -> &str {
            "row_probe"
        }
        fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
            let rows: Vec<u64> = task
                .features()
                .column(0)
                .iter()
                .map(|&v| v as u64)
                .collect();
            self.seen.lock().unwrap().push(rows);
            Ok(Box::new(ProbeModel))
        }
    }

    #[test]
    fn nested_cv_runs_clean_and_inner_tuning_never_sees_outer_test_rows() {
        // Feature 0 is a unique row id 0..n; every train task records which
        // ids it saw. Outer CV = 2 folds over the AutoTuner.
        let n = 24;
        let features = Array2::from_shape_fn((n, 1), |(i, _)| i as f64);
        let target: Vec<usize> = (0..n).map(|i| i % 2).collect();
        let task = ClassificationTask::new("probe", features, target).unwrap();

        let seen: Arc<Mutex<Vec<Vec<u64>>>> = Arc::new(Mutex::new(Vec::new()));
        let seen_factory = Arc::clone(&seen);

        // A single-point grid: the probe ignores params, but the tuner still
        // runs the inner resampling + a final refit, exercising the full path.
        let auto = AutoTuner::new(
            move |_p| {
                Box::new(RowProbe {
                    seen: Arc::clone(&seen_factory),
                })
            },
            TunerSpec::Grid({
                let mut g = ParamGrid::new();
                g.insert("dummy".into(), vec![ParamValue::Int(0)]);
                g
            }),
            Box::new(CrossValidation::new(3).with_seed(7)),
            Box::new(Accuracy),
        );

        // Outer CV: build each outer fold by hand so we know its test rows,
        // then confirm none of them leaked into that fold's inner tuning.
        let outer = CrossValidation::new(2).with_seed(99);
        let outer_splits = outer.splits(n).unwrap();

        for (outer_train, outer_test) in &outer_splits {
            seen.lock().unwrap().clear();
            let outer_test_ids: std::collections::HashSet<u64> =
                outer_test.iter().map(|&i| i as u64).collect();

            // Train the AutoTuner on the outer TRAIN fold only (this is exactly
            // what benchmark::resample_classif does internally).
            let train_features = task.features().select(ndarray::Axis(0), outer_train);
            let train_target: Vec<usize> = outer_train.iter().map(|&i| task.target()[i]).collect();
            let train_task =
                ClassificationTask::new("outer", train_features, train_target).unwrap();
            let mut auto_clone = AutoTuner {
                factory: Arc::clone(&auto.factory),
                tuner: match &auto.tuner {
                    TunerSpec::Grid(g) => TunerSpec::Grid(g.clone()),
                    _ => unreachable!(),
                },
                resampling: Box::new(CrossValidation::new(3).with_seed(7)),
                measure: Box::new(Accuracy),
                seed: auto.seed,
            };
            auto_clone.train_classif(&train_task).unwrap();

            // Every recorded training call (inner folds + final refit) must be
            // a subset of the outer training rows — no outer test row anywhere.
            let calls = seen.lock().unwrap();
            assert!(!calls.is_empty(), "the probe must have been trained");
            for rows in calls.iter() {
                for r in rows {
                    assert!(
                        !outer_test_ids.contains(r),
                        "row {r} from the outer TEST fold leaked into inner tuning"
                    );
                }
            }
        }
    }

    #[test]
    fn nested_cv_via_benchmark_runs_clean() {
        // The ergonomic path the whole feature exists for: AutoTuner straight
        // into resample_classif with an outer CV.
        let task = classif_task(40);
        let mut auto = AutoTuner::new(
            |p| Box::new(DecisionTree::new().with_max_depth(p["max_depth"].as_usize().unwrap())),
            TunerSpec::Grid(depth_grid(vec![1, 4])),
            Box::new(CrossValidation::new(3).with_seed(0)),
            Box::new(Accuracy),
        );
        let outer = CrossValidation::new(3).with_seed(1);
        let bench =
            crate::benchmark::resample_classif(&mut auto, &task, &outer, &[&Accuracy]).unwrap();
        assert_eq!(bench.scores.len(), 3, "one score row per outer fold");
        assert!(
            bench.mean_scores()[0] > 0.8,
            "nested CV accuracy on separable blobs"
        );
    }

    // ── Test 4: weights ──────────────────────────────────────────────────

    fn weighted_regress_task(n: usize) -> RegressionTask {
        let features = Array2::from_shape_fn((n, 1), |(i, _)| i as f64);
        let target: Vec<f64> = (0..n).map(|i| 3.0 * i as f64).collect();
        let weights: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
        RegressionTask::new("wr", features, target)
            .unwrap()
            .with_weights(weights)
    }

    #[test]
    fn weighted_task_flows_to_a_weight_aware_base() {
        // Ridge is weight-aware, so the AutoTuner reports supports_weights and
        // a weighted task trains cleanly end to end.
        let task = weighted_regress_task(20);
        let mut grid = ParamGrid::new();
        grid.insert(
            "alpha".into(),
            vec![ParamValue::Float(0.1), ParamValue::Float(1.0)],
        );
        let mut auto = AutoTuner::new(
            |p| Box::new(Ridge::new(p["alpha"].as_f64().unwrap())),
            TunerSpec::Grid(grid),
            Box::new(CrossValidation::new(3).with_seed(0)),
            Box::new(Rmse),
        );
        assert!(auto.supports_weights(), "Ridge base is weight-aware");
        assert!(
            auto.train_regress(&task).is_ok(),
            "weighted task must train"
        );

        // A weight-aware forest classifier too.
        let mut cspace = ParamSpace::new();
        cspace.insert(
            "max_depth".into(),
            ParamDistribution::Choice(vec![ParamValue::Int(3)]),
        );
        let rf_auto = AutoTuner::new(
            |p| Box::new(RandomForest::new().with_max_depth(p["max_depth"].as_usize().unwrap())),
            TunerSpec::Random {
                space: cspace,
                n_iter: 1,
            },
            Box::new(CrossValidation::new(2).with_seed(0)),
            Box::new(Accuracy),
        );
        assert!(
            rf_auto.supports_weights(),
            "RandomForest base is weight-aware"
        );
    }

    #[test]
    fn weighted_task_with_weight_blind_base_errors_legibly() {
        // KNN is weight-blind: supports_weights must be false, and a weighted
        // task must fail with the base learner's clear guard message.
        let task = weighted_regress_task(20);
        let mut grid = ParamGrid::new();
        grid.insert("k".into(), vec![ParamValue::Int(3), ParamValue::Int(5)]);
        let mut auto = AutoTuner::new(
            |p| Box::new(KNearestNeighbors::new(p["k"].as_usize().unwrap())),
            TunerSpec::Grid(grid),
            Box::new(CrossValidation::new(3).with_seed(0)),
            Box::new(Rmse),
        );
        assert!(!auto.supports_weights(), "KNN base is weight-blind");
        let err = auto.train_regress(&task).map(|_| ()).unwrap_err();
        assert!(
            format!("{err}").contains("does not support sample weights"),
            "weight-blind base must surface a legible guard error, got: {err}"
        );
    }

    // ── Test 5: determinism ──────────────────────────────────────────────

    #[test]
    fn same_seed_gives_same_best_params_across_runs() {
        let task = classif_task(50);
        let make = || {
            let mut space = ParamSpace::new();
            space.insert("max_depth".into(), ParamDistribution::Uniform(1.0, 12.0));
            AutoTuner::new(
                |p| {
                    Box::new(DecisionTree::new().with_max_depth(p["max_depth"].as_usize().unwrap()))
                },
                TunerSpec::Random { space, n_iter: 8 },
                Box::new(CrossValidation::new(3).with_seed(0)),
                Box::new(Accuracy),
            )
            .with_seed(1234)
        };
        let a = make().fit_classif(&task).unwrap();
        let b = make().fit_classif(&task).unwrap();
        assert_eq!(
            a.best_params()["max_depth"],
            b.best_params()["max_depth"],
            "same tuner seed must select the same best_params"
        );
        assert_eq!(a.best_score(), b.best_score(), "and the same best_score");
    }

    // ── Test 6: all four tuners smoke ────────────────────────────────────

    #[test]
    fn all_four_tuners_run() {
        let task = classif_task(40);
        let depth_space = || {
            let mut s = ParamSpace::new();
            s.insert("max_depth".into(), ParamDistribution::Uniform(1.0, 8.0));
            s
        };
        let factory = || {
            |p: &ParamSet| -> Box<dyn Learner> {
                Box::new(DecisionTree::new().with_max_depth(p["max_depth"].as_usize().unwrap()))
            }
        };

        let specs = vec![
            TunerSpec::Grid(depth_grid(vec![2, 4, 6])),
            TunerSpec::Random {
                space: depth_space(),
                n_iter: 5,
            },
            TunerSpec::Bayesian {
                space: depth_space(),
                n_iter: 8,
                n_initial: 3,
            },
            TunerSpec::Hyperband {
                space: depth_space(),
                max_folds: 4,
                eta: 2,
            },
        ];

        for spec in specs {
            let mut auto = AutoTuner::new(
                factory(),
                spec,
                Box::new(CrossValidation::new(3).with_seed(0)),
                Box::new(Accuracy),
            );
            let trained = auto.fit_classif(&task).unwrap();
            // best_params always carries the tuned parameter.
            assert!(trained.best_params().contains_key("max_depth"));
            assert!(!trained.history().is_empty(), "history must be populated");
            // and the boxed trait path works too.
            let model = auto.train_classif(&task).unwrap();
            let _ = model.predict(task.features()).unwrap();
        }
    }

    #[test]
    fn representative_params_covers_every_space_key() {
        // The probe used by supports_weights must produce a value for every
        // key a real factory might index — otherwise supports_weights panics.
        let mut space = ParamSpace::new();
        space.insert("a".into(), ParamDistribution::Uniform(2.0, 9.0));
        space.insert("b".into(), ParamDistribution::LogUniform(1e-3, 1.0));
        space.insert(
            "c".into(),
            ParamDistribution::Choice(vec![ParamValue::Int(4), ParamValue::Int(9)]),
        );
        let spec = TunerSpec::Random { space, n_iter: 1 };
        let p = spec.representative_params();
        assert_eq!(p["a"], ParamValue::Float(2.0));
        assert_eq!(p["b"], ParamValue::Float(1e-3));
        assert_eq!(p["c"], ParamValue::Int(4));

        let grid: HashMap<String, Vec<ParamValue>> = depth_grid(vec![3, 7, 9]);
        let gp = TunerSpec::Grid(grid).representative_params();
        assert_eq!(
            gp["max_depth"],
            ParamValue::Int(3),
            "grid takes the first value"
        );
    }
}
