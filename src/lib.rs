//! # Smelt
//!
//! A machine learning framework for Rust, inspired by mlr3.
//!
//! ## Pipeline
//!
//! ```text
//! Data → Task → Learner → Prediction → Measure
//!                  ↑
//!              Resampling
//!              Tuning
//!              Preprocessing
//! ```
//!
//! ## Quick Start
//!
//! ```rust
//! use smelt_ml::prelude::*;
//! use ndarray::array;
//!
//! // Define a classification task
//! let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
//! let target = vec![0, 0, 1, 1];
//! let task = ClassificationTask::new("example", features, target).unwrap();
//!
//! // Train a learner
//! let model = DecisionTree::default().train_classif(&task).unwrap();
//!
//! // Predict and evaluate
//! let pred = model.predict(task.features()).unwrap()
//!     .with_truth_classif(task.target().to_vec());
//! let acc = Accuracy.score(&pred).unwrap();
//! assert!(acc > 0.0);
//! ```

pub mod task;
pub mod learner;
pub mod prediction;
pub mod resample;
pub mod measure;
pub mod benchmark;
pub mod preprocess;
pub mod tuning;
pub mod importance;
pub mod conformal;
pub mod data;
pub mod serialize;

mod error;

pub use error::{SmeltError, Result};

/// Convenience re-exports for `use smelt_ml::prelude::*`
pub mod prelude {
    pub use crate::task::{Task, ClassificationTask, RegressionTask};
    pub use crate::learner::{Learner, DecisionTree, KNearestNeighbors, LinearRegression, LogisticRegression, RandomForest, GradientBoosting, ExtraTrees, GaussianNB, Ridge, Lasso, ElasticNet, AdaBoost, LinearSVM, XGBoost, Stacking, QuantileGB, EBM, Bagging};
    pub use crate::prediction::Prediction;
    pub use crate::resample::{Resample, CrossValidation, Holdout, SpatialBlockCV, SpatialBufferCV};
    pub use crate::measure::{Measure, Accuracy, Precision, Recall, F1Score, LogLoss, AucRoc, Rmse, Mae, RSquared, Mape};
    pub use crate::preprocess::{Transformer, StandardScaler, MinMaxScaler, Imputer, ImputeStrategy, OneHotEncoder, LabelEncoder, Smote, Pipeline};
    pub use crate::tuning::{GridSearch, RandomSearch, TuneResult, ParamDistribution};
    pub use crate::importance::{FeatureImportance, permutation_importance_classif, permutation_importance_regress};
    pub use crate::data::CsvLoader;
    pub use crate::serialize::{SerializableModel, save_json, load_json};
    pub use crate::benchmark::{self, BenchmarkResult};
    pub use crate::error::{SmeltError, Result};
}
