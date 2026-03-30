# Smelt

[![Crates.io](https://img.shields.io/crates/v/smelt-ml.svg)](https://crates.io/crates/smelt-ml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A machine learning framework for Rust, inspired by [mlr3](https://mlr3.mlr-org.com/) and [scikit-learn](https://scikit-learn.org/).

The name refers to smelting — refining raw data into useful models.

**19 learners** | **10 metrics** | **XGBoost from scratch** | **Conformal Prediction** | **Spatial ML** | **Bayesian Optimization**

## Quick Start

```toml
[dependencies]
smelt-ml = "0.3"
ndarray = "0.16"
```

```rust
use smelt_ml::prelude::*;
use ndarray::array;

// Create a classification task
let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
let target = vec![0, 0, 1, 1];
let task = ClassificationTask::new("example", features, target).unwrap();

// Train and evaluate
let mut tree = DecisionTree::default();
let model = tree.train_classif(&task).unwrap();
let pred = model.predict(task.features()).unwrap()
    .with_truth_classif(task.target().to_vec());
let acc = Accuracy.score(&pred).unwrap();
```

## Pipeline (Preprocessing + Learner)

Chain transformers with a learner. The pipeline implements `Learner`, so it works with cross-validation, tuning, and bagging automatically.

```rust
use smelt_ml::prelude::*;

let mut pipe = Pipeline::new(
    vec![
        Box::new(StandardScaler::new()),
        Box::new(Imputer::mean()),
    ],
    Box::new(RandomForest::new().with_n_estimators(100)),
);

// Evaluate with 5-fold cross-validation
let cv = CrossValidation::new(5);
let result = benchmark::resample_classif(&mut pipe, &task, &cv, &[&Accuracy, &F1Score]).unwrap();
println!("Mean accuracy: {:.3}", result.mean_scores()[0]);
```

## XGBoost

Full XGBoost implementation from scratch — Newton boosting, histogram splits, NaN handling, early stopping, parallel split finding.

```rust
use smelt_ml::prelude::*;

let mut xgb = XGBoost::new()
    .with_n_estimators(100)
    .with_max_depth(6)
    .with_learning_rate(0.3)
    .with_lambda(1.0)           // L2 regularization
    .with_subsample(0.8)        // row subsampling
    .with_colsample_bytree(0.8) // column subsampling
    .with_early_stopping_rounds(10);

let model = xgb.train_classif(&task).unwrap();
```

Competitive with the official XGBoost on datasets up to 1K samples. Handles NaN natively and auto-switches to exact greedy for small datasets.

## Bayesian Optimization (TPE)

Tree-structured Parzen Estimator for efficient hyperparameter tuning — smarter than grid or random search.

```rust
use smelt_ml::prelude::*;
use smelt_ml::tuning::{BayesianOptimizer, ParamSpace, ParamDistribution};

let mut space = ParamSpace::new();
space.insert("max_depth".into(), ParamDistribution::Uniform(1.0, 10.0));
space.insert("learning_rate".into(), ParamDistribution::LogUniform(0.01, 1.0));
space.insert("n_estimators".into(), ParamDistribution::Choice(vec![50.0, 100.0, 200.0]));

let bo = BayesianOptimizer::new(
    |params| Box::new(XGBoost::new()
        .with_max_depth(params["max_depth"] as usize)
        .with_learning_rate(params["learning_rate"])
        .with_n_estimators(params["n_estimators"] as usize)),
    space,
).with_n_iter(30);

let result = bo.tune_classif(&task, &cv, &Accuracy).unwrap();
println!("Best: {:?}, score: {:.4}", result.best_params, result.best_score);
```

## Conformal Prediction

Distribution-free prediction intervals with guaranteed coverage. Works with any trained model.

```rust
use smelt_ml::conformal::ConformalRegressor;

// Train any model, then calibrate on held-out data
let cf = ConformalRegressor::calibrate(&*model, &cal_features, &cal_targets, 0.1).unwrap();
let intervals = cf.predict(&new_features).unwrap();

for iv in &intervals {
    println!("[{:.2}, {:.2}]", iv.lower, iv.upper); // 90% coverage guaranteed
}
```

## Geographical-XGBoost (Spatial ML)

Spatially local regression for geospatial data. Implements [Grekousis (2025)](https://doi.org/10.1007/s10109-025-00465-4).

```rust
use smelt_ml::prelude::*;

let coords: Vec<(f64, f64)> = /* lat/lon per sample */;

let mut gxgb = GeoXGBoost::new(coords)
    .with_bandwidth(30)           // N nearest neighbors
    .with_n_estimators(100);

let model = gxgb.train_regress(&task).unwrap();
// Each spatial unit gets its own local model + local feature importance
```

Also includes `SpatialBlockCV` and `SpatialBufferCV` for spatially-aware cross-validation.

## Stacking (Super Learner)

Combine multiple heterogeneous learners via a meta-learner trained on out-of-fold predictions.

```rust
use smelt_ml::prelude::*;

let mut stack = Stacking::new(
    vec![
        Box::new(|| Box::new(DecisionTree::default()) as Box<dyn Learner>),
        Box::new(|| Box::new(RandomForest::new().with_n_estimators(50)) as Box<dyn Learner>),
        Box::new(|| Box::new(XGBoost::new().with_n_estimators(50)) as Box<dyn Learner>),
        Box::new(|| Box::new(KNearestNeighbors::new(5)) as Box<dyn Learner>),
    ],
    || Box::new(LogisticRegression::new()),
);
let model = stack.train_classif(&task).unwrap();
```

## Class Imbalance (SMOTE)

```rust
use smelt_ml::prelude::*;

let smote = Smote::new().with_k_neighbors(5);
let balanced_task = smote.balance(&imbalanced_task).unwrap();
// Now train on balanced data
```

## CSV Data Loading

```rust
use smelt_ml::data::CsvLoader;

// Classification with auto label encoding
let task = CsvLoader::from_path("data.csv")
    .target("species")
    .load_classif().unwrap();

// Regression
let task = CsvLoader::from_path("housing.csv")
    .target("price")
    .load_regress().unwrap();
```

## All Learners

| Algorithm | Classification | Regression | Key Feature |
|-----------|:-:|:-:|-------------|
| Decision Tree (CART) | x | x | Gini / MSE splits |
| K-Nearest Neighbors | x | x | Euclidean distance |
| Linear Regression | | x | Normal equation (OLS) |
| Logistic Regression | x | | Auto-scaling, SGD |
| Random Forest | x | x | Parallel (rayon), probability averaging |
| Gradient Boosting | x | x | MSE/log-loss, multiclass softmax |
| Extra Trees | x | x | Random thresholds, no bootstrap |
| **XGBoost** | x | x | Newton, histogram, NaN, early stopping |
| **Geographical-XGBoost** | | x | Spatial kernel, local+global ensemble |
| Gaussian Naive Bayes | x | | Probabilistic, fast baseline |
| Ridge Regression | | x | L2 regularization, closed form |
| Lasso Regression | | x | L1 regularization, coordinate descent |
| Elastic Net | | x | L1+L2, coordinate descent |
| AdaBoost | x | | SAMME with weighted stumps |
| Linear SVM | x | | SGD + hinge loss, OVR multiclass |
| **Stacking (Super Learner)** | x | x | Meta-ensemble, out-of-fold |
| **Quantile GB** | | x | Pinball loss, prediction intervals |
| **EBM** | x | x | Interpretable GAM, shape functions |
| Bagging (any learner) | x | x | Generic bootstrap wrapper |

## All Metrics

| Metric | Type | Direction |
|--------|------|:---------:|
| Accuracy | Classification | maximize |
| Precision (macro) | Classification | maximize |
| Recall (macro) | Classification | maximize |
| F1 Score (macro) | Classification | maximize |
| Log Loss | Classification | minimize |
| AUC-ROC | Classification | maximize |
| RMSE | Regression | minimize |
| MAE | Regression | minimize |
| R-squared | Regression | maximize |
| MAPE | Regression | minimize |

## Preprocessing

| Transformer | Purpose |
|------------|---------|
| StandardScaler | Zero mean, unit variance |
| MinMaxScaler | Scale to [0, 1] |
| Imputer | Fill NaN (mean, median, constant) |
| OneHotEncoder | Categorical to binary columns |
| LabelEncoder | String labels to integers |
| SMOTE | Synthetic minority oversampling |
| Pipeline | Chain transformers + learner |

## Tuning

| Method | Strategy |
|--------|----------|
| GridSearch | Exhaustive over parameter grid |
| RandomSearch | Sample from distributions |
| **BayesianOptimizer** | TPE (Tree-structured Parzen Estimator) |

All support `Uniform`, `LogUniform`, and `Choice` parameter distributions.

## Resampling

| Strategy | Use Case |
|----------|----------|
| CrossValidation | Standard K-fold |
| Holdout | Simple train/test split |
| SpatialBlockCV | Geospatial block partitioning |
| SpatialBufferCV | Geospatial with exclusion buffer |

## Additional Features

- **Conformal Prediction** — distribution-free prediction intervals/sets with coverage guarantees
- **Permutation feature importance** — model-agnostic, works with any learner
- **Model serialization** — save/load as JSON
- **CSV loading** — with auto label encoding for classification
- **Input validation** — dimension checks in predict, NaN detection

## Architecture

```
Data (CSV) -> Task -> Pipeline(Transformers -> Learner) -> Prediction -> Measure
                           |
                      Resampling (CV, Holdout, Spatial)
                      Tuning (Grid, Random, Bayesian)
                      Conformal Prediction
                      Feature Importance
```

All components are **trait-based and composable**:
- Implement `Learner` to add new algorithms
- Implement `Measure` for new metrics
- Implement `Transformer` for new preprocessing steps
- Implement `Resample` for new splitting strategies

## Performance

Compile with native CPU optimizations for 18-30% speedup:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

## License

MIT

## Citation

If you use Geographical-XGBoost, please cite:

> Grekousis, G. (2025). Geographical-XGBoost: a new ensemble model for spatially local regression based on gradient-boosted trees. *Journal of Geographical Systems*, 27(2), 169-195.
