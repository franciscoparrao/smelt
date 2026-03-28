//! Learners: algorithms that train on Tasks and produce Predictions.
//!
//! Each learner implements the `Learner` trait.

pub mod tree;
pub mod knn;
pub mod linear_regression;
pub mod logistic_regression;
pub mod bagging;

use ndarray::Array2;
use crate::task::{ClassificationTask, RegressionTask};
use crate::prediction::Prediction;
use crate::Result;

pub use tree::decision_tree::DecisionTree;
pub use tree::random_forest::RandomForest;
pub use tree::gradient_boosting::GradientBoosting;
pub use knn::KNearestNeighbors;
pub use linear_regression::LinearRegression;
pub use logistic_regression::LogisticRegression;
pub use bagging::Bagging;

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
    fn feature_importance(&self) -> Option<Vec<(String, f64)>> { None }
}
