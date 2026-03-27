# Smelt — Machine Learning Framework for Rust

## Overview

Smelt is an ML framework inspired by [mlr3](https://mlr3.mlr-org.com/) (R), designed for Rust's performance and safety guarantees. The name refers to smelting — refining raw data into useful models.

## Architecture

```
Task → Learner → TrainedModel → Prediction → Measure
                      ↑
                  Resampling (CV, Holdout)
                  Tuning (Grid, Random, Bayesian)
                  Preprocessing (Scale, Encode, Impute)
```

### Core Abstractions (mlr3 mapping)

| Smelt | mlr3 | Purpose |
|-------|------|---------|
| `Task` | `Task` | Data container with target |
| `ClassificationTask` | `TaskClassif` | Discrete target |
| `RegressionTask` | `TaskRegr` | Continuous target |
| `Learner` | `Learner` | Algorithm that trains |
| `TrainedModel` | trained Learner | Fitted model that predicts |
| `Prediction` | `Prediction` | Output with optional truth |
| `Measure` | `Measure` | Evaluation metric |
| `Resample` | `Resampling` | Train/test splitting strategy |

### Module Structure

```
src/
├── lib.rs          # Public API + prelude
├── error.rs        # SmeltError enum (thiserror)
├── task/mod.rs     # Task, ClassificationTask, RegressionTask
├── learner/mod.rs  # Learner trait, TrainedModel trait
├── prediction/     # Prediction enum (Classification/Regression)
├── measure/        # Accuracy, RMSE, MAE (+ trait Measure)
├── resample/       # CrossValidation, Holdout (+ trait Resample)
├── preprocess/     # TODO: StandardScaler, MinMaxScaler, OneHotEncoder
└── tuning/         # TODO: GridSearch, RandomSearch, BayesianOpt
```

## Build & Test

```bash
cargo check          # Type check
cargo test           # Run tests
cargo bench          # Run benchmarks (criterion)
cargo doc --open     # Generate docs
```

## Design Principles

1. **Type safety** — Classification and Regression are separate types, not runtime tags
2. **Trait-based extensibility** — Implement `Learner` to add new algorithms
3. **Zero-copy where possible** — ndarray views, references over clones
4. **Parallel by default** — rayon for data parallelism (CV folds, ensemble training)
5. **Composable pipeline** — Task → Learner → Prediction → Measure is always the flow

## Implementation Roadmap

### Phase 1 — Core (current)
- [x] Task system (Classification + Regression)
- [x] Learner + TrainedModel traits
- [x] Prediction enum
- [x] Measures: Accuracy, RMSE, MAE
- [x] Resampling: CrossValidation, Holdout

### Phase 2 — First Learners
- [ ] Decision Tree (CART)
- [ ] K-Nearest Neighbors
- [ ] Logistic Regression
- [ ] Linear Regression (OLS)
- [ ] Benchmark pipeline (resample + measure loop)

### Phase 3 — Ensembles
- [ ] Random Forest
- [ ] Gradient Boosting (XGBoost-style)
- [ ] Bagging

### Phase 4 — Preprocessing
- [ ] StandardScaler, MinMaxScaler
- [ ] OneHotEncoder, LabelEncoder
- [ ] Missing value imputation
- [ ] Pipeline chaining (preprocess → learner)

### Phase 5 — Tuning
- [ ] GridSearch
- [ ] RandomSearch
- [ ] Bayesian Optimization

### Phase 6 — Advanced
- [ ] Feature importance (permutation, SHAP-like)
- [ ] Spatial cross-validation (for geo applications)
- [ ] CSV/Parquet data loading
- [ ] Model serialization (serde)
- [ ] Python bindings (PyO3) — expose as `smelt-py`

## Dependencies

- `ndarray` — N-dimensional arrays (feature matrices)
- `rand` — Random number generation (resampling, stochastic algorithms)
- `rayon` — Data parallelism
- `thiserror` — Error types
- `serde` — Serialization
- `criterion` — Benchmarks (dev)

## Author

Francisco Parra — francisco.parra.o@usach.cl

## Inspiration

- [mlr3](https://mlr3.mlr-org.com/) (R) — Task/Learner/Measure architecture
- [scikit-learn](https://scikit-learn.org/) (Python) — fit/predict API
- [linfa](https://github.com/rust-ml/linfa) (Rust) — Existing Rust ML, but different design philosophy
