//! Learners: algorithms that train on Tasks and produce Predictions.
//!
//! Each learner implements the `Learner` trait.

pub mod adaboost;
pub mod adaptive_rf;
pub mod auto_tuner;
pub mod bagging;
pub mod calibration;
pub mod catboost;
pub mod cost_sensitive;
pub mod deep_forest;
pub mod des;
pub mod ebm;
pub mod elm;
pub(crate) mod eval;
pub mod geo_xgboost;
pub(crate) mod hist_pool;
pub mod histogram;
pub mod hoeffding;
pub mod knn;
pub mod kriging_hybrid;
pub mod lightgbm;
pub mod linear_regression;
pub mod logistic_regression;
pub(crate) mod math;
pub mod mondrian;
pub mod naive_bayes;
pub mod oblique;
pub mod properties;
pub mod quantile;
pub mod quantile_forest;
pub mod regularized;
pub mod registry;
pub mod stacking;
pub mod svm;
pub mod target_transform;
pub mod threshold;
pub mod tree;
pub mod xgboost;

use crate::prediction::Prediction;
use crate::task::{ClassificationTask, RegressionTask};
use crate::{Result, SmeltError};
use ndarray::Array2;

pub use adaboost::AdaBoost;
pub use adaptive_rf::{AdaptiveRandomForest, Adwin};
pub use auto_tuner::{AutoTuner, TrainedAutoTuner, TunerSpec};
pub use bagging::Bagging;
pub use calibration::{CalibratedClassifier, CalibrationMethod, TrainedCalibratedClassifier};
pub use catboost::CatBoost;
pub use cost_sensitive::CostSensitiveClassifier;
pub use deep_forest::{DeepForest, TrainedDeepForest};
pub use des::DynamicEnsemble;
pub use ebm::EBM;
pub use elm::{Activation, ExtremeLearningMachine};
pub use geo_xgboost::{BandwidthSelection, GeoXGBoost, TrainedGeoXGBoost};
pub use hoeffding::HoeffdingTree;
pub use knn::KNearestNeighbors;
pub use kriging_hybrid::{KrigingHybrid, TrainedKrigingHybrid, VariogramFit, VariogramModel};
pub use lightgbm::LightGBM;
pub use linear_regression::LinearRegression;
pub use logistic_regression::LogisticRegression;
pub use mondrian::{MondrianForest, MondrianTree};
pub use naive_bayes::GaussianNB;
pub use oblique::{ObliqueForest, ObliqueTree};
pub use properties::LearnerProperties;
pub use quantile::QuantileGB;
pub use quantile_forest::{QuantileForest, TrainedQuantileForest};
pub use regularized::{ElasticNet, Lasso, Ridge};
pub use registry::{learner_from_id, learner_properties, registered_learner_ids};
pub use stacking::Stacking;
pub use svm::LinearSVM;
pub use target_transform::{TargetTransform, TargetTransformRegressor, TrainedTargetTransformRegressor};
pub use threshold::{ThresholdedClassifier, TrainedThresholdedClassifier};
pub use tree::decision_tree::DecisionTree;
pub use tree::extra_trees::ExtraTrees;
pub use tree::gradient_boosting::GradientBoosting;
pub use tree::random_forest::RandomForest;
pub use xgboost::{Objective, XGBoost};

/// Core trait for classification learners.
///
/// Most learners only implement one of `train_classif`/`train_regress` (e.g.
/// `LinearRegression` is regression-only, `GaussianNB` is classification-only);
/// the other falls back to the default, which reports unsupported via `Result`
/// rather than requiring every learner to write out an identical error stub.
pub trait Learner: Send + Sync {
    /// Unique learner identifier (e.g., "classif.decision_tree").
    fn id(&self) -> &str;

    /// Train on a classification task, returning a trained model.
    fn train_classif(&mut self, _task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        Err(SmeltError::InvalidParameter(format!(
            "{} does not support classification",
            self.id()
        )))
    }

    /// Train on a regression task, returning a trained model.
    fn train_regress(&mut self, _task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
        Err(SmeltError::InvalidParameter(format!(
            "{} does not support regression",
            self.id()
        )))
    }

    /// Declarative capability metadata for this learner (task support,
    /// weights, probabilities, NaN/categorical tolerance, feature importance,
    /// serializability). See [`LearnerProperties`] for the exact observable
    /// meaning of each field.
    ///
    /// The default is [`LearnerProperties::none`] (everything `false`): a
    /// learner that forgets to override this *under*-claims and so fails the
    /// contract autotest (`tests/contract.rs`) loudly, rather than silently
    /// advertising a capability it lacks. Every registered learner overrides
    /// it with values verified against its real behaviour.
    fn properties(&self) -> LearnerProperties {
        LearnerProperties::none()
    }

    /// Whether this learner consumes per-sample weights
    /// ([`ClassificationTask::weights`]/[`RegressionTask::weights`]) during
    /// training.
    ///
    /// **Single source of truth:** this default returns
    /// [`self.properties().supports_weights`](LearnerProperties::supports_weights),
    /// so the boolean and the metadata field can never diverge. Weight-aware
    /// learners declare `.with_weights()` in their [`properties`](Self::properties)
    /// override *and* drop the [`check_no_weights`](crate::validate::check_no_weights)
    /// guard — the two must change together. Learners generally should not
    /// override this method; override `properties()` instead. (The one
    /// exception is a composite like [`AutoTuner`] whose weight support is a
    /// runtime property of its base learner rather than a compile-time
    /// constant; it overrides `properties()` to delegate, which this method
    /// then reflects.)
    ///
    /// `false` means the learner rejects a weighted task with a clear error
    /// via [`check_no_weights`](crate::validate::check_no_weights) instead of
    /// silently ignoring the weights.
    fn supports_weights(&self) -> bool {
        self.properties().supports_weights
    }
}

/// A trained model that can make predictions.
pub trait TrainedModel: Send + Sync {
    /// Predict on new feature data.
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction>;

    /// Feature importances (if available).
    ///
    /// Contract: implementations must return exactly one entry per feature,
    /// **in training-column order** (entry `i` describes feature column `i`),
    /// not sorted by importance. Consumers rely on positional alignment --
    /// e.g. `RFE` maps each entry back to the column it eliminates purely by
    /// position.
    fn feature_importance(&self) -> Option<Vec<(String, f64)>> {
        None
    }

    /// Convert this trained model into its serializable form, if this
    /// concrete type has a `SerializableModel` variant (`src/serialize.rs`).
    /// Returns `None` for the `Box<dyn TrainedModel>`-holding composites
    /// (Bagging, Pipeline, Stacking, GeoXGBoost, DeepForest, KrigingHybrid,
    /// DynamicEnsemble, CostSensitiveClassifier, TargetTransformRegressor)
    /// that have no variant.
    fn to_serializable(&self) -> Option<crate::serialize::SerializableModel> {
        None
    }
}
