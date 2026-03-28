# Smelt

A machine learning framework for Rust, inspired by [mlr3](https://mlr3.mlr-org.com/).

The name refers to smelting -- refining raw data into useful models.

## Quick Start

```rust
use smelt::prelude::*;
use ndarray::array;

// Define a classification task
let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
let target = vec![0, 0, 1, 1];
let task = ClassificationTask::new("example", features, target).unwrap();

// Train a learner
let mut tree = DecisionTree::default();
let model = tree.train_classif(&task).unwrap();

// Predict and evaluate
let pred = model.predict(task.features()).unwrap()
    .with_truth_classif(task.target().to_vec());
let acc = Accuracy.score(&pred).unwrap();
```

## Pipeline

```rust
use smelt::prelude::*;

// Chain preprocessing + learner
let mut pipe = Pipeline::new(
    vec![Box::new(StandardScaler::new())],
    Box::new(RandomForest::new().with_n_estimators(100)),
);

// Evaluate with cross-validation
let cv = CrossValidation::new(5);
let result = benchmark::resample_classif(&mut pipe, &task, &cv, &[&Accuracy, &F1Score]).unwrap();
println!("Mean accuracy: {:.3}", result.mean_scores()[0]);
```

## Hyperparameter Tuning

```rust
use smelt::prelude::*;
use smelt::tuning::ParamGrid;

let mut grid = ParamGrid::new();
grid.insert("max_depth".into(), vec![3.0, 5.0, 10.0]);
grid.insert("min_samples_split".into(), vec![2.0, 5.0]);

let gs = GridSearch::new(
    |params| Box::new(DecisionTree::new()
        .with_max_depth(params["max_depth"] as usize)
        .with_min_samples_split(params["min_samples_split"] as usize)),
    grid,
);
let result = gs.tune_classif(&task, &cv, &Accuracy).unwrap();
println!("Best params: {:?}, score: {:.3}", result.best_params, result.best_score);
```

## Features

### Learners

| Algorithm | Classification | Regression |
|-----------|:-:|:-:|
| Decision Tree (CART) | x | x |
| K-Nearest Neighbors | x | x |
| Linear Regression | | x |
| Logistic Regression | x | |
| Random Forest | x | x |
| Gradient Boosting | x | x |
| Bagging (any learner) | x | x |

### Measures

| Metric | Type | Direction |
|--------|------|-----------|
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

### Preprocessing

- StandardScaler, MinMaxScaler
- Imputer (mean, median, constant)
- OneHotEncoder, LabelEncoder
- Pipeline (chains transformers + learner)

### Resampling

- K-Fold Cross-Validation
- Holdout (train/test split)
- Spatial Block CV (for geospatial data)
- Spatial Buffer CV (with exclusion zones)

### Tuning

- Grid Search
- Random Search (Uniform, LogUniform, Choice)

### Additional

- Permutation feature importance
- CSV data loading
- Model serialization (JSON)

## Architecture

```
Data -> Task -> Learner -> Prediction -> Measure
                   ^
               Resampling
               Tuning
               Preprocessing
```

All components are trait-based and composable. Implement `Learner` to add new algorithms, `Measure` for new metrics, `Transformer` for new preprocessing steps, and `Resample` for new splitting strategies.

## License

MIT
