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
//! ```rust,no_run
//! use smelt::prelude::*;
//!
//! // Define a classification task
//! let task = ClassificationTask::new("iris", features, target);
//!
//! // Train a learner
//! let model = DecisionTree::default().train(&task).unwrap();
//!
//! // Predict
//! let pred = model.predict(&new_data);
//!
//! // Evaluate
//! let acc = Accuracy.score(&pred);
//! ```

pub mod task;
pub mod learner;
pub mod prediction;
pub mod resample;
pub mod measure;
pub mod preprocess;
pub mod tuning;

mod error;

pub use error::{SmeltError, Result};

/// Convenience re-exports for `use smelt::prelude::*`
pub mod prelude {
    pub use crate::task::{Task, ClassificationTask, RegressionTask};
    pub use crate::learner::Learner;
    pub use crate::prediction::Prediction;
    pub use crate::resample::{Resample, CrossValidation, Holdout};
    pub use crate::measure::{Measure, Accuracy, Rmse, Mae};
    pub use crate::error::{SmeltError, Result};
}
