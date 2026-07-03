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
├── learner/        # Learner trait, TrainedModel trait, learner_from_id registry,
│                   # 26 learners (tree/, xgboost, lightgbm, catboost, geo_xgboost,
│                   # oblique, stacking, bagging, des, ebm, quantile*, regularized, ...)
├── prediction/     # Prediction enum (Classification/Regression)
├── measure/        # Accuracy, F1, AUC-ROC, BalancedAccuracy, Kappa, MCC, Brier,
│                   # RMSE, MAE, R², MAPE (+ trait Measure)
├── resample/       # CrossValidation, Holdout, SpatialBlockCV, SpatialBufferCV,
│                   # StratifiedCV, GroupCV (+ trait Resample)
├── preprocess/     # StandardScaler, MinMaxScaler, Imputer, OneHot/LabelEncoder,
│                   # SMOTE, Adasyn, PCA, FilterSelector, RFE, Pipeline
├── tuning/         # GridSearch, RandomSearch, BayesianOptimizer, Hyperband
├── cluster/        # KMeans, DBSCAN, IsolationForest
├── causal/         # CausalForest (honest splitting, CATE/ATE, jackknife SE)
├── conformal/      # ConformalRegressor/Classifier, CQR
├── survival/       # RandomSurvivalForest
├── importance/     # permutation importance, permutation-SHAP
├── multilabel/, multioutput/  # ClassifierChain, RegressorChain
├── stats.rs        # Wilcoxon, sign test, Friedman, Nemenyi, McNemar, bootstrap CI
├── data/           # CsvLoader
├── serialize.rs    # SerializableModel (JSON, versioned envelope)
├── benchmark.rs, benchmark_design.rs  # resample+measure loop, multi-learner tables
└── validate.rs     # dimension/NaN checks shared across public entry points
```

`smelt-py/` (PyO3 bindings) exposes a subset of the above as `smelt` on PyPI —
see Fase 3 item 15 below for the gap between what's in Rust and what's bound.

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

The original Phase 1-6 plan below (core → first learners → ensembles →
preprocessing → tuning → advanced) is **done** — 26 learners, full
preprocessing pipeline, 4 tuning methods, spatial CV, serialization, and
Python bindings (`smelt-py`) all exist and are tested. Kept for history;
current work tracked in `docs/auditoria_motor_2026-07-01.md` (engine
audit + 4-phase remediation plan — Fase 0/1/2 done, Fase 3 "paridad
competitiva" in progress):

- [x] Phase 1 — Core: Task system, Learner/TrainedModel traits, Prediction, Measures, Resampling
- [x] Phase 2 — First learners: Decision Tree, KNN, Logistic/Linear Regression, benchmark pipeline
- [x] Phase 3 — Ensembles: Random Forest, Gradient Boosting, Bagging
- [x] Phase 4 — Preprocessing: scalers, encoders, imputation, Pipeline chaining
- [x] Phase 5 — Tuning: GridSearch, RandomSearch, Bayesian Optimization, Hyperband
- [x] Phase 6 — Advanced: permutation/SHAP importance, spatial CV, CSV loading, serde serialization, PyO3 bindings

### Fase 3 remaining (paridad competitiva, see the audit doc for full detail)
- [x] Missing measures: BalancedAccuracy, CohensKappa, MCC, Brier
- [x] Model registry (`learner_from_id`)
- [x] Consistent parallel `predict` (multiclass XGBoost/CatBoost, all of LightGBM)
- [x] Categorical features + NaN support in `Task`/splits (FeatureType metadata,
      NaN/categorical-aware CsvLoader, native Fisher categorical splits in
      XGBoost/LightGBM, CatBoost target-stats fixes M2/M3); eval-set early
      stopping in all 3 boosting engines; monotone constraints + custom
      objectives (Huber/Poisson/Custom) in XGBoost; check_no_nan in non-NaN
      learners (done 2026-07-02 — see docs/fase3_progreso.md). Not yet in
      smelt-py: cat_features/eval_set/monotone/objective bindings (item 15c/15d)
- [ ] Python: `define_learner!` macro, close the ~14 learners not yet bound,
      `get_params`/`set_params`, split `smelt-py/src/lib.rs` (1800+ lines)
- [ ] Parquet/Arrow loading, `f32` histograms, sparse data support
- [ ] `README.md`/this file kept current as features land (this section itself
      was stale for a long time — reconciled 2026-07-02)

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
