# Smelt

[![Crates.io](https://img.shields.io/crates/v/smelt-ml.svg)](https://crates.io/crates/smelt-ml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A machine learning framework for Rust, inspired by [mlr3](https://mlr3.mlr-org.com/) and [scikit-learn](https://scikit-learn.org/).

The name refers to smelting — refining raw data into useful models.

**33 supervised learners** | **Clustering** | **Causal Inference (5 meta-learners + Causal Forest)** | **XGBoost/LightGBM/CatBoost from scratch** | **Spatial ML** | **Conformal Prediction** | **4 tuning methods** | **600+ tests**

## Quick Start

```toml
[dependencies]
smelt-ml = "3.0"
ndarray = "0.16"
```

```rust
use smelt_ml::prelude::*;
use ndarray::array;

let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
let target = vec![0, 0, 1, 1];
let task = ClassificationTask::new("example", features, target).unwrap();

let mut tree = DecisionTree::default();
let model = tree.train_classif(&task).unwrap();
let pred = model.predict(task.features()).unwrap()
    .with_truth_classif(task.target().to_vec());
let acc = Accuracy.score(&pred).unwrap();
```

## Benchmark Design (mlr3-style)

Compare multiple learners on the same task with a single call:

```rust
use smelt_ml::benchmark_design::benchmark_classif;

let mut learners: Vec<Box<dyn Learner>> = vec![
    Box::new(DecisionTree::default()),
    Box::new(RandomForest::new().with_n_estimators(100)),
    Box::new(XGBoost::new().with_n_estimators(100)),
    Box::new(GaussianNB::new()),
];

let cv = CrossValidation::new(5);
let result = benchmark_classif(&mut learners, &[&task], &cv, &[&Accuracy, &F1Score]).unwrap();
println!("{}", result.summary());
```

```
Learner              Task             Accuracy      F1Score
-----------------------------------------------------------
decision_tree        iris               0.9533       0.9521
random_forest        iris               0.9667       0.9658
xgboost              iris               0.9600       0.9591
gaussian_nb          iris               0.9533       0.9521
```

## Pipeline (Preprocessing + Feature Selection + Learner)

Chain transformers with a learner. Filters and PCA are fitted only on training data (no leakage):

```rust
use smelt_ml::preprocess::filter::FilterSelector;

let mut pipe = Pipeline::new(
    vec![
        Box::new(Imputer::mean()),                    // fill NaN
        Box::new(FilterSelector::anova_f(10)),        // top 10 features by ANOVA
        Box::new(StandardScaler::new()),              // standardize
    ],
    Box::new(XGBoost::new().with_n_estimators(100)),
);

let cv = CrossValidation::new(5);
let result = benchmark::resample_classif(&mut pipe, &task, &cv, &[&Accuracy]).unwrap();
```

## XGBoost

Full implementation from scratch — Newton boosting, histogram splits, NaN handling, early stopping, parallel split finding:

```rust
let mut xgb = XGBoost::new()
    .with_n_estimators(100)
    .with_max_depth(6)
    .with_learning_rate(0.3)
    .with_lambda(1.0)           // L2 regularization
    .with_subsample(0.8)
    .with_colsample_bytree(0.8)
    .with_early_stopping_rounds(10);
```

Competitive with the official XGBoost C++ library. Auto-switches to exact greedy for small datasets.

## Hyperparameter Tuning (4 methods)

```rust
use smelt_ml::tuning::{BayesianOptimizer, Hyperband, ParamSpace, ParamDistribution};

let mut space = ParamSpace::new();
space.insert("max_depth".into(), ParamDistribution::Uniform(1.0, 10.0));
space.insert("learning_rate".into(), ParamDistribution::LogUniform(0.01, 1.0));

// Bayesian Optimization (TPE) — intelligent sampling
let bo = BayesianOptimizer::new(factory, space.clone()).with_n_iter(30);
let result = bo.tune_classif(&task, &cv, &Accuracy)?;

// Hyperband — successive halving, efficient for many configs
let hb = Hyperband::new(factory, space).with_max_folds(5);
let result = hb.tune_classif(&task, &Accuracy)?;
```

Also available: `GridSearch` (exhaustive) and `RandomSearch` (sampling).

## Clustering (Unsupervised)

```rust
use smelt_ml::cluster::{KMeans, DBSCAN};

// K-Means
let result = KMeans::new(3).fit(&features).unwrap();
println!("Labels: {:?}", result.labels);
println!("Silhouette: {:.3}", result.silhouette_score(&features));

// DBSCAN — finds clusters of arbitrary shape, detects noise
let result = DBSCAN::new(0.5, 5).fit(&features).unwrap();
// noise points labeled as -1
```

## Causal Forest

Estimate heterogeneous treatment effects — first causal inference implementation in a Rust ML framework:

```rust
use smelt_ml::causal::CausalForest;

let cf = CausalForest::new()
    .with_n_estimators(100)
    .with_min_samples_leaf(5);

let result = cf.estimate(&features, &treatment, &outcome, &feature_names).unwrap();

println!("ATE: {:.2} +/- {:.2}", result.ate, result.ate_std_error);
for effect in &result.effects {
    println!("tau={:.2}, 95% CI=[{:.2}, {:.2}]",
        effect.estimate, effect.ci_lower, effect.ci_upper);
}
```

Uses honest splitting (separate samples for tree structure and effect estimation).

## Conformal Prediction

Distribution-free prediction intervals with guaranteed coverage:

```rust
use smelt_ml::conformal::ConformalRegressor;

let cf = ConformalRegressor::calibrate(&*model, &cal_features, &cal_targets, 0.1).unwrap();
let intervals = cf.predict(&new_features).unwrap();
// Each interval has guaranteed 90% coverage
```

## Geographical-XGBoost (Spatial ML)

Spatially local regression — [Grekousis (2025)](https://doi.org/10.1007/s10109-025-00465-4):

```rust
let mut gxgb = GeoXGBoost::new(coords.clone())
    .with_bandwidth(30)
    .with_n_estimators(100);
let model = gxgb.train_geo(&task).unwrap();
// predict_spatial finds each point's nearest local model — pass the
// training coords back to get in-sample fitted values.
let fitted = model.predict_spatial(task.features(), &coords).unwrap();
```

`train_regress` (the `Learner` trait method) also works and returns `Box<dyn TrainedModel>`, but its `predict()` is global-model-only — spatial predictions always require `predict_spatial` with explicit coordinates.

Also: `SpatialBlockCV` and `SpatialBufferCV` for spatially-aware cross-validation.

## Dimensionality Reduction (PCA)

```rust
let mut pipe = Pipeline::new(
    vec![Box::new(PCA::new(10))],  // reduce to 10 components
    Box::new(RandomForest::new()),
);
```

## Recursive Feature Elimination (RFE)

Wrapper feature selection — uses model importance to iteratively remove the weakest feature:

```rust
use smelt_ml::preprocess::RFE;

let mut rfe = RFE::classif(|| Box::new(RandomForest::new()), 5); // keep 5 features
```

## Stacking (Super Learner)

```rust
let mut stack = Stacking::new(
    vec![
        Box::new(|| Box::new(DecisionTree::default()) as Box<dyn Learner>),
        Box::new(|| Box::new(XGBoost::new()) as Box<dyn Learner>),
        Box::new(|| Box::new(KNearestNeighbors::new(5)) as Box<dyn Learner>),
    ],
    || Box::new(LogisticRegression::new()),
);
```

## CSV Data Loading

```rust
use smelt_ml::data::CsvLoader;

let task = CsvLoader::from_path("data.csv")
    .target("species")
    .load_classif().unwrap();  // auto label encoding for string targets
```

## All Supervised Learners

| Algorithm | Classification | Regression | Key Feature |
|-----------|:-:|:-:|-------------|
| Decision Tree (CART) | x | x | Gini / MSE splits |
| K-Nearest Neighbors | x | x | Euclidean distance |
| Linear Regression | | x | Normal equation (OLS) |
| Logistic Regression | x | | Auto-scaling, SGD |
| Random Forest | x | x | Parallel (rayon), soft voting, all-features regression default |
| Gradient Boosting | x | x | MSE/log-loss, multiclass softmax |
| Extra Trees | x | x | Random thresholds, no bootstrap |
| **XGBoost** | x | x | Newton, histogram, NaN, early stopping |
| **LightGBM** | x | x | GOSS sampling, row/column subsampling, histogram splits |
| **CatBoost** | x | x | Ordered target statistics for categoricals |
| **Geographical-XGBoost** | | x | Spatial kernel, local+global ensemble |
| **Kriging-ML Hybrid** | | x | Regression-kriging residual correction |
| **Oblique Tree** | x | x | Sparse projection splits |
| **Oblique Forest (SPORF)** | x | x | Ensemble of oblique trees, parallel |
| Gaussian Naive Bayes | x | | Probabilistic, fast baseline |
| Ridge Regression | | x | L2 regularization, closed form |
| Lasso Regression | | x | L1 regularization, coordinate descent |
| Elastic Net | | x | L1+L2, coordinate descent |
| AdaBoost | x | | SAMME with weighted stumps |
| Linear SVM | x | | SGD + hinge loss, OVR multiclass |
| Hoeffding Tree | x | | Incremental / streaming induction |
| **Adaptive Random Forest** | x | | Streaming ensemble, ADWIN concept-drift detection |
| **Mondrian Tree** | x | x | Online Mondrian process, batch=online consistency |
| **Mondrian Forest** | x | x | Ensemble of Mondrian trees, streaming-native |
| **Extreme Learning Machine** | x | x | Fixed random hidden layer, closed-form ridge solve |
| **Stacking (Super Learner)** | x | x | Meta-ensemble, out-of-fold |
| **Dynamic Ensemble (KNORA-E)** | x | | Per-instance competence selection, train/DSEL split |
| **Deep Forest (gcForest)** | x | | Cascade forest, out-of-fold layer stacking |
| **Cost-Sensitive Classifier** | x | | Bayes-risk decision rule over any probabilistic base |
| **Quantile GB** | | x | Pinball loss, prediction intervals |
| **Quantile Forest** | | x | Full conditional distribution per leaf |
| **EBM** | x | x | Interpretable GAM, shape functions |
| Bagging (any learner) | x | x | Generic bootstrap wrapper |

## Unsupervised Learning

| Algorithm | Key Feature |
|-----------|-------------|
| **K-Means** | k-means++ init, best-of-`n_init`, silhouette score |
| **DBSCAN** | Density-based, noise detection |
| **Isolation Forest** | Anomaly detection, golden-tested against scikit-learn |

## Causal Inference

| Algorithm | Key Feature |
|-----------|-------------|
| **Causal Forest** | Honest splitting, CATE, ATE, confidence intervals |
| **T/S/X/R/DR-Learner** | Meta-learners composing any `Learner` as the base model |

## Metrics

| Metric | Type | Direction |
|--------|------|:---------:|
| Accuracy | Classification | maximize |
| Precision (macro) | Classification | maximize |
| Recall (macro) | Classification | maximize |
| F1 Score (macro) | Classification | maximize |
| Log Loss | Classification | minimize |
| AUC-ROC | Classification | maximize |
| Balanced Accuracy | Classification | maximize |
| Cohen's Kappa | Classification | maximize |
| Matthews Correlation Coefficient | Classification | maximize |
| Brier Score | Classification | minimize |
| RMSE | Regression | minimize |
| MAE | Regression | minimize |
| R-squared | Regression | maximize |
| MAPE | Regression | minimize |
| Silhouette | Clustering | maximize |

## Preprocessing & Feature Engineering

| Transformer | Purpose |
|------------|---------|
| StandardScaler | Zero mean, unit variance |
| MinMaxScaler | Scale to [0, 1] |
| Imputer | Fill NaN (mean, median, constant) |
| OneHotEncoder | Categorical to binary columns |
| LabelEncoder | String labels to integers |
| SMOTE | Synthetic minority oversampling |
| **PCA** | Dimensionality reduction |
| **FilterSelector** | Feature selection (Variance, Correlation, ANOVA-F, Information Gain, Mutual Info) |
| **RFE** | Recursive Feature Elimination (wrapper) |
| Pipeline | Chain any transformers + learner |

## Tuning

| Method | Strategy | Best For |
|--------|----------|----------|
| GridSearch | Exhaustive | Small spaces |
| RandomSearch | Sampling | Medium spaces |
| **BayesianOptimizer** | TPE | Expensive evaluations |
| **Hyperband** | Successive halving | Many configurations |

## Resampling

| Strategy | Use Case |
|----------|----------|
| CrossValidation | Standard K-fold |
| Holdout | Simple train/test split |
| SpatialBlockCV | Geospatial block partitioning |
| SpatialBufferCV | Geospatial with exclusion buffer |
| StratifiedCV | K-fold preserving per-fold class balance |
| GroupCV | K-fold keeping each group entirely in one fold |

## Additional Features

- **Conformal Prediction** — prediction intervals/sets with coverage guarantees
- **Permutation feature importance** — model-agnostic
- **Benchmark design** — multi-learner comparison tables
- **Model serialization** — save/load as JSON
- **CSV loading** — with auto label encoding
- **Input validation** — dimension checks, NaN detection
- **Model registry** — `learner_from_id("xgboost")` constructs any of the
  27 self-contained learners by name, for data-driven experiment loops
  (excludes learners needing a base-learner factory or external
  coordinates: Bagging, Stacking, DynamicEnsemble, CostSensitiveClassifier,
  KrigingHybrid, GeoXGBoost)

## Architecture

```
Data (CSV) -> Task -> Pipeline(Filters -> PCA -> Scaler -> Learner) -> Prediction -> Measure
                           |
                      Resampling (CV, Holdout, Spatial)
                      Tuning (Grid, Random, Bayesian, Hyperband)
                      Conformal Prediction
                      Feature Importance (permutation, model-based)
                      Benchmark Design (multi-learner comparison)

Unsupervised:  Data -> KMeans / DBSCAN -> ClusterResult -> Silhouette
Causal:        Data + Treatment -> CausalForest -> CATE + ATE + CIs
```

All components are **trait-based and composable**:
- `Learner` — supervised learning algorithms
- `Measure` — evaluation metrics
- `Transformer` — preprocessing steps (with `fit_supervised` for filters)
- `Resample` — data splitting strategies

## Performance

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release  # 18-30% speedup
```

## License

MIT

## References

If you use Geographical-XGBoost, please cite:
> Grekousis, G. (2025). Geographical-XGBoost. *Journal of Geographical Systems*, 27(2), 169-195.

If you use Oblique Forest, please cite:
> Tomita, T. et al. (2020). Sparse Projection Oblique Randomer Forests. *JMLR*, 21(104), 1-39.

If you use Causal Forest, please cite:
> Wager, S. & Athey, S. (2018). Estimation and Inference of Heterogeneous Treatment Effects. *JASA*, 113(523), 1228-1242.
