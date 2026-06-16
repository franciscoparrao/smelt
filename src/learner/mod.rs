//! Learners: algorithms that train on Tasks and produce Predictions.
//!
//! Each learner implements the `Learner` trait.

pub mod adaboost;
pub mod bagging;
pub mod catboost;
pub mod des;
pub mod ebm;
pub mod geo_xgboost;
pub(crate) mod hist_pool;
pub mod histogram;
pub mod hoeffding;
pub mod knn;
pub mod lightgbm;
pub mod linear_regression;
pub mod logistic_regression;
pub mod naive_bayes;
pub mod oblique;
pub mod quantile;
pub mod quantile_forest;
pub mod regularized;
pub mod stacking;
pub mod svm;
pub mod tree;
pub mod xgboost;

use crate::Result;
use crate::prediction::Prediction;
use crate::task::{ClassificationTask, RegressionTask};
use ndarray::Array2;

pub use adaboost::AdaBoost;
pub use bagging::Bagging;
pub use catboost::CatBoost;
pub use des::DynamicEnsemble;
pub use ebm::EBM;
pub use geo_xgboost::{BandwidthSelection, GeoXGBoost, TrainedGeoXGBoost};
pub use hoeffding::HoeffdingTree;
pub use knn::KNearestNeighbors;
pub use lightgbm::LightGBM;
pub use linear_regression::LinearRegression;
pub use logistic_regression::LogisticRegression;
pub use naive_bayes::GaussianNB;
pub use oblique::{ObliqueForest, ObliqueTree};
pub use quantile::QuantileGB;
pub use quantile_forest::QuantileForest;
pub use regularized::{ElasticNet, Lasso, Ridge};
pub use stacking::Stacking;
pub use svm::LinearSVM;
pub use tree::decision_tree::DecisionTree;
pub use tree::extra_trees::ExtraTrees;
pub use tree::gradient_boosting::GradientBoosting;
pub use tree::random_forest::RandomForest;
pub use xgboost::XGBoost;

/// Core trait for classification learners.
pub trait Learner: Send + Sync {
    /// Unique learner identifier (e.g., "classif.decision_tree").
    fn id(&self) -> &str;

    /// Train on a classification task, returning a trained model.
    fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>>;

    /// Train on a regression task, returning a trained model.
    fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>>;
}

/// A trained model that can make predictions.
pub trait TrainedModel: Send + Sync {
    /// Predict on new feature data.
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction>;

    /// Feature importances (if available).
    fn feature_importance(&self) -> Option<Vec<(String, f64)>> {
        None
    }
}
