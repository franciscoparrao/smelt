#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::type_complexity)]
#![warn(missing_docs)]

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

pub mod benchmark;
pub mod benchmark_design;
pub mod causal;
pub mod cluster;
pub mod conformal;
pub mod data;
pub mod importance;
pub mod learner;
pub mod measure;
pub mod multilabel;
pub mod multioutput;
pub mod prediction;
pub mod preprocess;
pub mod resample;
pub mod serialize;
pub mod sparse;
pub mod stats;
pub mod survival;
pub mod task;
pub mod tuning;

mod error;
pub mod validate;

pub use error::{Result, SmeltError};

/// Convenience re-exports for `use smelt_ml::prelude::*`
pub mod prelude {
    pub use crate::benchmark::{self, BenchmarkResult};
    pub use crate::causal::{
        CausalEffect, CausalForest, CausalForestResult,
        meta_learners::{DrLearner, RLearner, SLearner, TLearner, XLearner},
    };
    pub use crate::cluster::{ClusterResult, DBSCAN, IsolationForest, KMeans};
    pub use crate::data::CsvLoader;
    #[cfg(feature = "parquet")]
    pub use crate::data::ParquetLoader;
    pub use crate::error::{Result, SmeltError};
    pub use crate::importance::{
        FeatureImportance, permutation_importance_classif, permutation_importance_regress,
    };
    pub use crate::learner::{
        AdaBoost, AdaptiveRandomForest, Adwin, Bagging, CatBoost, DecisionTree, DynamicEnsemble,
        EBM, ElasticNet, ExtraTrees, BandwidthSelection, GaussianNB, GeoXGBoost, GradientBoosting,
        HoeffdingTree, KNearestNeighbors, KrigingHybrid, Lasso, Learner, TrainedGeoXGBoost,
        TrainedKrigingHybrid, LightGBM, LinearRegression, LinearSVM, LogisticRegression,
        ObliqueForest, ObliqueTree, Objective, QuantileForest, QuantileGB, RandomForest, Ridge,
        Stacking, VariogramFit, VariogramModel, XGBoost, learner_from_id, registered_learner_ids,
    };
    pub use crate::measure::{
        Accuracy, AteBias, AucRoc, BalancedAccuracy, Brier, CohensKappa, F1Score, LogLoss, Mae,
        Mape, Mcc, Measure, Pehe, Precision, RSquared, Recall, Rmse,
    };
    pub use crate::prediction::Prediction;
    pub use crate::preprocess::{
        Adasyn, FilterSelector, ImputeStrategy, Imputer, LabelEncoder, MinMaxScaler, OneHotEncoder,
        PCA, Pipeline, RFE, Smote, SpatialSmote, StandardScaler, Transformer,
    };
    pub use crate::resample::{
        CrossValidation, GroupCV, Holdout, Resample, SpatialBlockCV, SpatialBufferCV, StratifiedCV,
    };
    pub use crate::serialize::{SerializableModel, load_json, save_json};
    pub use crate::sparse::CsrMatrix;
    pub use crate::survival::{
        RandomSurvivalForest, SurvivalEvent, SurvivalPrediction, TrainedRandomSurvivalForest,
        concordance_index,
    };
    pub use crate::task::{ClassificationTask, FeatureType, RegressionTask, Task};
    pub use crate::tuning::{
        BayesianOptimizer, GridSearch, Hyperband, ParamDistribution, RandomSearch, TuneResult,
    };
}
