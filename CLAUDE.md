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
│                   # 28 learners (tree/, xgboost, lightgbm, catboost, geo_xgboost,
│                   # kriging_hybrid, hoeffding + adaptive_rf (streaming/online),
│                   # oblique, stacking, bagging, des, ebm, quantile*, regularized, ...)
├── prediction/     # Prediction enum (Classification/Regression)
├── measure/        # Accuracy, F1, AUC-ROC, BalancedAccuracy, Kappa, MCC, Brier,
│                   # RMSE, MAE, R², MAPE (+ trait Measure)
├── resample/       # CrossValidation, Holdout, SpatialBlockCV, SpatialBufferCV,
│                   # StratifiedCV, GroupCV (+ trait Resample)
├── preprocess/     # StandardScaler, MinMaxScaler, Imputer, OneHot/LabelEncoder,
│                   # SMOTE, SpatialSmote, Adasyn, PCA, FilterSelector, RFE, Pipeline
├── tuning/         # GridSearch, RandomSearch, BayesianOptimizer, Hyperband
├── cluster/        # KMeans, DBSCAN, IsolationForest
├── causal/         # CausalForest (honest splitting, CATE/ATE, jackknife SE);
│                   # causal/meta_learners/: T/S/X/R/DR-learner (Künzel et al.
│                   # 2019, Nie & Wager 2021, Kennedy 2020), composing ordinary
│                   # Learners via Bagging/Stacking-style factory closures
├── conformal/      # ConformalRegressor/Classifier, CQR
├── survival/       # RandomSurvivalForest
├── importance/     # permutation importance, permutation-SHAP
├── multilabel/, multioutput/  # ClassifierChain, RegressorChain
├── stats.rs        # Wilcoxon, sign test, Friedman, Nemenyi, McNemar, bootstrap CI
├── data/           # CsvLoader; ParquetLoader behind the `parquet` feature
├── sparse.rs       # CsrMatrix (hand-rolled CSR); OneHotEncoder::transform_sparse
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
      learners (done 2026-07-02 — see docs/fase3_progreso.md). Exposed in
      smelt-py 2026-07-03: `cat_features`/`eval_set`/`early_stopping_rounds`
      on XGBoost/LightGBM/CatBoost `fit()`, `monotone_constraints`/
      `objective` (squared_error/huber/poisson, not `Custom`) as XGBoost
      constructor params — see docs/fase3_progreso.md
- [x] Python: `define_learner!` macro, close the ~14 learners not yet bound
      (item 15a/15b, done earlier); split `smelt-py/src/lib.rs`
      2543→114 lines across `common.rs` + `learners/{boosting,trees,linear,
      misc,ensemble}.rs` + `preprocess/resample/measures/py_stats/tuning/
      feature_selection.rs` (item 15d, done 2026-07-02); sklearn-style
      `get_params`/`set_params` on all 26 wrappers via `define_learner!` and
      a new `declare_params!` macro (item 15c, done 2026-07-03) — see
      docs/fase3_progreso.md
- [x] Parquet loading: `ParquetLoader` (`src/data/parquet.rs`, mirrors
      `CsvLoader`'s API) behind a new optional `parquet` Cargo feature
      (`polars` dependency, ~200 transitive crates, zero-cost when the
      feature is off) — done 2026-07-03, see docs/fase3_progreso.md. Not yet
      bound in smelt-py (deliberately out of scope, analogous follow-up to
      item 14's smelt-py exposure)
- [x] `f32` histograms (item 16d part 2/3) — **CatBoost only**, done
      2026-07-03. Measured each engine's histogram-accumulation share of
      total training time before touching code (CatBoost 45.5%, XGBoost
      30.7%, LightGBM 10.1%) and scoped to the one case where the Amdahl
      ceiling (~23%) justified the numerical-drift risk; measured ~26%
      real speedup after, zero test regressions (74 lib + 272 integration).
      LightGBM/XGBoost deliberately left on f64 — evaluated and passed on,
      not merely deferred — see docs/fase3_progreso.md
- [x] Sparse data support (item 16d part 3/3) — **narrow scope**, done
      2026-07-03. Investigated first: `Task::features() -> &Array2<f64>` is
      concretely typed across 44 call sites with no trait-object seam, so a
      full `SparseTask` isn't justified by current evidence (only linear
      models would get a real algorithmic speedup; boosting would need
      `HistBins` reworked regardless). Shipped a hand-rolled `CsrMatrix`
      (`src/sparse.rs`, no `sprs` dependency) + `OneHotEncoder::transform_sparse`
      — the one confirmed genuinely-wasteful path today (dense one-hot
      output on high-cardinality columns). `SparseTask`/sparse linear-model
      math left as separate, larger follow-ups — see
      docs/sparse_data_2026-07-03.md
- [x] `README.md`/this file kept current as features land (this section itself
      was stale for a long time — reconciled 2026-07-02)
- [x] `#![warn(missing_docs)]` (item 17b) — done 2026-07-03. 330 warnings
      (grown from 308 at the last count, per new code added this session)
      closed by parallelizing across 8 agents on disjoint file sets, all
      purely additive one-line `///` docs verified against actual code
      semantics (not paraphrased from names) — see
      docs/missing_docs_2026-07-03.md. Zero regressions (101 lib + 66 doc +
      274 integration tests). **Fase 3 is now fully complete.**

### Causal meta-learners (2026-07-03, not part of Fase 3)

Separate initiative — the user asked for "SOTA algorithms" without
specifying domain; after evaluating causal meta-learners, a GeoXGBoost/MGWR
extension (rejected: needs discussion with paper collaborator George
Grekousis first, not a unilateral design), audit-gap closures (DART/EFB/
ordered boosting — 2017-18 techniques, not "SOTA" strictly), and tabular
deep learning (foundational blocker, no autodiff infra exists), causal
meta-learners was chosen. See `docs/causal_meta_learners_2026-07-03.md` for
the full design rationale.

- [x] T/S/X/R/DR-learner (`src/causal/meta_learners/`) — standalone
      `estimate(features, treatment, outcome)` API (matches `CausalForest`'s
      precedent, not a `Learner` impl — a 3-input estimator doesn't fit
      `Learner::train_regress(&RegressionTask)`'s `(X,y)` shape). Composes
      ordinary `Learner`s via the same `Fn() -> Box<dyn Learner> + Send +
      Sync` factory pattern `Bagging`/`Stacking` use. R-learner/DR-learner
      share K-fold cross-fitting helpers (`meta_learners/cross_fit.rs`,
      built on `CrossValidation::splits`). New `Prediction::CausalEffect`
      variant + `Pehe`/`AteBias` measures for evaluating against synthetic
      ground-truth CATE. 95 lib tests + 66 doctests green (up from 74/61).
- [x] Python bindings (`smelt-py/src/causal.rs`) for all 5 meta-learners —
      done 2026-07-03, same session. Same id-string base-learner pattern as
      `Bagging`/`Stacking` (not `define_learner!`/generic `declare_params!`,
      both assume the `(X,y)`-`Learner` shape); `validate_learner_id`
      promoted from private to `pub(crate)` in `learners/ensemble.rs` to
      share it instead of duplicating
- [ ] Generic per-sample-weight support on `Learner`/`RegressionTask` —
      would let R-learner use the paper's weighted R-loss instead of the
      documented unweighted simplification; cross-cutting, out of scope

### Geospatial differentiators (2026-07-04, not part of Fase 3)

Separate initiative — with Fase 3 fully closed, the user chose to open a new
phase scoped to features unique to smelt's GIS niche versus sklearn/xgboost,
pulled from `docs/roadmap_checklist.md` (Prioridad 4).

- [x] Kriging-ML Hybrid (`src/learner/kriging_hybrid.rs`) — regression-kriging:
      trains a base `Learner` via the same `Fn() -> Box<dyn Learner> + Send +
      Sync` factory pattern `Bagging`/`Stacking`/the causal meta-learners use,
      fits a semivariogram (Spherical/Exponential/Gaussian, grid-search fit —
      no nonlinear-least-squares dependency, same "hand-roll the small
      numeric routine" precedent as `CsrMatrix` in `src/sparse.rs`) to its
      residuals, and krige-interpolates them at prediction time via a
      hand-rolled Gaussian-elimination solver (local neighborhood, not a
      global n×n solve). `TrainedModel::predict` is base-model-only (the
      trait carries no coordinates); `TrainedKrigingHybrid::predict_spatial`
      does the kriging correction — same split as `TrainedGeoXGBoost`.
- [x] Spatial-SMOTE (`src/preprocess/spatial_smote.rs`) — SMOTE restricted to
      same-class neighbors within an optional `max_spatial_distance`, so it
      can't splice together feature-similar but geographically distant
      minority samples the way plain `Smote` can. Interpolates a synthetic
      coordinate alongside each synthetic sample (same lambda as the feature
      interpolation) and returns it alongside the balanced task, since `Task`
      itself carries no coordinates (same "coords passed alongside, not
      stored in `Task`" idiom as `SpatialBlockCV`/`SpatialBufferCV`/
      `GeoXGBoost`). Matches plain `Smote`'s output exactly when
      `max_spatial_distance` is unset.
- [x] Python bindings (2026-07-04, same-day fast-follow once the Rust side
      was test-hardened): `KrigingHybrid` in `smelt-py/src/learners/boosting.rs`
      (alongside `GeoXGBoost` — same "inherent `predict_spatial` beyond the
      trait" shape) selects its base learner by id string and hand-writes
      `get_params`/`set_params` (not `declare_params!`) to re-validate that id
      on `set_params`, exactly like `Bagging`/`Stacking` in `ensemble.rs` (the
      macro can't express the re-validation). `Smote` (bound for the first
      time) and `SpatialSmote` live in `smelt-py/src/preprocess.rs`, using the
      project's existing `parse_coords` convention for the `coords` param.
      Verified via `maturin develop --release` + a direct Python script
      (not just `cargo check`) — confirmed the kriging correction cuts MSE
      from 8.8 to 0.036 on synthetic spatially-structured residuals and that
      an invalid `base` id raises cleanly from both `__new__` and
      `set_params`.
- [x] Adaptive Random Forest / ADWIN (2026-07-04) — `src/learner/adaptive_rf.rs`.
      Ensemble of `HoeffdingTree`s (`src/learner/hoeffding.rs`) with online
      bagging (Poisson(λ) resampling weight per sample, hand-rolled via
      Knuth's algorithm — no `rand_distr` dependency) and two `Adwin`
      concept-drift detectors per tree (warning: starts a background tree;
      drift: swaps it in). `Adwin` is a simplified "exact scan every cut
      point" version of Bifet & Gavaldà's algorithm (not the paper's O(log n)
      exponential-histogram buckets — a deliberately smaller data structure,
      bounded instead via `with_max_window`). Required one purely-additive
      change to `HoeffdingTree` (`predict_one`, since `TrainedModel::predict`
      only existed on the post-training snapshot, not the live streaming
      tree) plus registering `"adaptive_random_forest"` in
      `src/learner/registry.rs` (self-contained, no factory/coords needed —
      matches `ObliqueForest`'s precedent, not `Bagging`/`GeoXGBoost`'s
      exclusion).
    - **Found and fixed a pre-existing bug while building on `HoeffdingTree`**
      (which had zero tests before this): `find_best_split` estimated split
      quality by comparing each class's *mean* feature value against a
      single threshold as an all-or-nothing assignment — since two classes'
      means are almost never on the exact same side of a threshold, this made
      *every* feature, including pure noise, look like a "perfect" split, so
      the Hoeffding-bound gain-difference test could never clear its
      confidence bar and the tree never split at all (confirmed via a
      diagnostic test: online accuracy stuck at ~50% — chance level — even on
      a trivial single-feature threshold rule). Fixed by estimating left/right
      counts from each class's running Gaussian (mean/variance already
      tracked in `FeatureStats`) via the normal CDF at the candidate
      threshold, instead of the single mean-point comparison; needed a
      hand-rolled `erf`/`normal_cdf` (Abramowitz & Stegun 7.1.26 approximation
      — no `f64::erf` in stable Rust, no numerics crate in this workspace).
      Added `hoeffding.rs`'s first tests as part of this fix.
    - Python bindings deferred, same reasoning as Kriging-ML Hybrid: this is a
      genuinely new (not just bound) statistical algorithm — verify Rust-side
      correctness first before locking in a pyo3-facing signature.

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
