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
│                   # 35 learners (tree/, xgboost, lightgbm, catboost, geo_xgboost,
│                   # kriging_hybrid, hoeffding + adaptive_rf + mondrian (streaming/
│                   # online), oblique, stacking, bagging, des, ebm, quantile*,
│                   # regularized, elm, gaussian_process (GP + predictive se),
│                   # kernel_svm (C-SVC via SMO), deep_forest,
│                   # cost_sensitive (wrapper), ...)
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
├── survival/       # RandomSurvivalForest, CoxPH (Cox proportional hazards)
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
- [x] Generic per-sample-weight support on `Learner`/`RegressionTask` —
      done 2026-07-18 as item 2 of the mlr3-parity roadmap
      (`docs/roadmap_checklist.md` Prioridad 6): `with_weights()` on both
      Tasks, `check_no_weights` guard on non-supporting learners (never
      silently ignored), real consumption in 13 learners (trees, boosting,
      linear family), Python `sample_weight=`, and the R-learner now uses
      the paper's exact weighted R-loss with weight-aware bases
      (row-replication fallback otherwise)

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
    - [x] Python bindings (2026-07-04, same-day fast-follow): `AdaptiveRandomForest`
      in `smelt-py/src/learners/trees.rs`, bound via the `define_learner!`
      macro alongside `HoeffdingTree` -- batch-only (`fit`/`predict` through
      `Learner::train_classif`), matching `HoeffdingTree`'s own existing
      Python binding exactly rather than introducing a new streaming
      (`partial_fit`/`predict_one`) surface unilaterally on only one of the
      two streaming learners. `lambda` renamed to `lambda_` for the Python
      constructor param (`lambda` is a Python keyword), same convention as
      XGBoost/GeoXGBoost/CatBoost's L2 term. Verified via `maturin develop
      --release` + a direct Python script: 94.2% accuracy on a fresh
      regime-2 holdout after adapting to an injected drift, versus 49.4%
      (chance) for a plain `HoeffdingTree` trained identically -- matching
      the Rust-side test's qualitative result end-to-end through the actual
      Python API. Streaming API parity (`partial_fit`/`n_drifts`/`predict_one`
      exposed to Python for both learners) is a natural follow-up, not done here.
- [x] Mondrian Forest (2026-07-04) — `src/learner/mondrian.rs`. The last open
      item in Prioridad 4 (`docs/roadmap_checklist.md`), closing that
      priority tier. Implements Lakshminarayanan, Roy & Teh (2014): splits
      come from a Mondrian process (split *time* ~ Exponential(rate = box's
      total side length), split *dimension* proportional to that dimension's
      own range, split *location* uniform in the data's range on that
      dimension) rather than greedy CART/information-gain, giving a specific
      consistency property `HoeffdingTree`/`AdaptiveRandomForest` don't have:
      a tree grown incrementally one point at a time is distributed
      identically to one grown by one-shot batch construction on the same
      points (`sample_mondrian_block`, used by `Learner::train_classif`/
      `train_regress`), regardless of arrival order. The online side
      (`extend_node`, Algorithm 3 in the paper) implements this for real: a
      point falling outside a node's current bounding box can retroactively
      introduce a new split *above* that node, reproducing exactly what
      batch construction on the enlarged point set would sample — not a
      simplified "just extend the box and keep the old structure" shortcut.
      One deliberate scope cut, documented in the module: a leaf that
      receives a point *within* its box only updates running statistics
      (class counts / Welford mean-variance) rather than re-attempting an
      internal split from remaining time budget as density grows, which
      would require storing raw per-leaf data instead of O(1) stats — same
      space/fidelity trade-off `Adwin` makes over the ADWIN paper's
      exponential-histogram buckets. No online bagging (unlike
      `AdaptiveRandomForest`): each tree sees every sample, and cross-tree
      diversity comes purely from each tree's own random stream of split
      times/dimensions/locations, matching the paper's actual ensemble
      design rather than bolting on ARF's Poisson-weighted resampling.
      Supports classification AND regression (both via `MondrianTree`/
      `MondrianForest`), unlike the classification-only Hoeffding/ARF pair —
      the first streaming *regressor* in the crate. Registered as both
      `"mondrian_tree"` and `"mondrian_forest"` in `src/learner/registry.rs`.
      11 unit tests colocated with the module (matching `adaptive_rf.rs`'s
      convention) plus 2 integration tests confirming it composes with the
      generic `Learner` trait (CV via `benchmark::resample_classif`, not
      just its own direct API); the differentiating behavior itself is
      covered by `online_extension_grows_tree_coverage_beyond_initial_range`
      (trains on a narrow initial range, then streams points far outside it
      with a different label, and confirms the tree's predictions for those
      far-outside points flip accordingly — this would fail against a tree
      that only ever grows structure downward from existing leaves, the
      Hoeffding-style behavior Mondrian Forests are meant to improve on).
    - [x] Python bindings (2026-07-04, same-day fast-follow): `MondrianForest`
      in `smelt-py/src/learners/trees.rs`, bound via `define_learner!`
      alongside `ObliqueForest` (batch-only, both classif and regress
      supported, `proba=true`) rather than exposing the streaming
      `partial_fit_classif`/`partial_fit_regress`/`predict_one_*` surface —
      same precedent as `AdaptiveRandomForest`'s Python binding. `lifetime`
      defaults to `f64::INFINITY`, which PyO3 round-trips to Python's
      `float('inf')` correctly (verified via `get_params()`). Verified via
      `maturin develop --release` + a direct Python script covering both
      task types plus a `get_params`/`set_params` round-trip: 100%
      classification accuracy and ~3e-15 regression RMSE on simple synthetic
      data (both tasks noise-free by construction, so this checks the
      binding is wired correctly rather than claiming that accuracy
      generalizes).

### Prioridad 3 quick items (2026-07-04, not part of Fase 3)

Separate initiative, chosen after Prioridad 4 (geospatial differentiators)
closed out fully -- the user asked to continue with Prioridad 3's 3
remaining items in `docs/roadmap_checklist.md` (ADASYN and CQR were already
done). All 3 done in one session; Prioridad 3 is now fully complete too.

- [x] Extreme Learning Machine (`src/learner/elm.rs`) — Huang, Zhu & Siew
      (2006). Single hidden-layer feedforward net whose input-to-hidden
      weights and biases are fixed random values (never trained); only the
      output weights are learned, via a closed-form ridge-regularized
      least-squares solve (`(HᵀH + λI)β = HᵀT`, one output column at a
      time) -- no backpropagation at all, hence "extreme"-ly fast to fit.
      Own hand-rolled SPD Gaussian-elimination solver (same shape as
      `regularized.rs::solve` for Ridge, kept separate per this crate's
      per-module numeric-routine convention). Supports both classification
      (one-hot target, softmax-normalized output for probabilities) and
      regression. Registered as `"elm"`.
- [x] Cost-Sensitive Learning (`src/learner/cost_sensitive.rs`) — Elkan
      (2001)'s Bayes-risk decision rule: given a cost matrix
      `cost[true][predicted]` and any base classifier's predicted
      probabilities, replaces `argmax_i P(i|x)` with the cost-minimizing
      `argmin_j Σ_i P(i|x)·cost[i][j]`. Needs no retraining at all -- a
      thin wrapper (`Fn() -> Box<dyn Learner>` factory, same pattern as
      `Bagging`/`Stacking`) around whatever probabilistic classifier is
      already trained. `CostSensitiveClassifier::binary(factory, fp_cost,
      fn_cost)` convenience constructor for the common 2-class case (the
      roadmap's "essential for medicine/finance" framing: e.g. a missed
      diagnosis costing far more than an unnecessary follow-up test).
      Deliberately NOT `MetaCost` (Domingos 1999, which retrains via
      bagging + cost-based relabeling) -- the decision-rule approach is
      simpler, needs no retraining, and is the more commonly used technique
      for exactly this "medicine/finance" framing; noted as a possible
      future addition, not attempted here. Not registered in
      `src/learner/registry.rs` (needs a factory, like `Bagging`/`Stacking`/
      `DynamicEnsemble` -- registry docs updated to say so).
- [x] Deep Forest / gcForest (`src/learner/deep_forest.rs`) — Zhou & Feng
      (2017), classification only, scoped to the "cascade forest" half of
      the paper (not "multi-grained scanning", which targets structured
      image/sequence inputs -- out of scope for this crate's tabular
      focus). Each layer trains `2 * n_forests_per_type` forests
      (alternating `RandomForest`/`ExtraTrees`, matching the paper's
      "two completely-random + two random forests" convention) on the
      current layer's input (original features augmented with every prior
      layer's out-of-fold class probabilities); each forest's contribution
      is itself produced via internal k-fold CV (`CrossValidation`,
      reused as-is) rather than in-sample predictions, so a forest can't
      fabricate falsely confident features for the next layer from
      overfitting its own layer's input. That same k-fold's average
      accuracy decides early stopping (`early_stopping_rounds` consecutive
      layers without improvement truncates the cascade back to its
      best-so-far depth) -- verified directly via `TrainedDeepForest::
      n_layers()`, an inherent method beyond the `TrainedModel` trait
      (same "concrete type carries more than the trait" shape as
      `TrainedKrigingHybrid`/`TrainedGeoXGBoost`; `DeepForest::fit` returns
      the concrete type, `Learner::train_classif` just boxes it). Registered
      as `"deep_forest"` (self-contained, no factory needed, matching
      `ObliqueForest`'s precedent).
    - [x] Python bindings (same-day fast-follow): `ExtremeLearningMachine`
      hand-written in `smelt-py/src/learners/misc.rs` (not via
      `define_learner!`, so `activation` can be eagerly validated against
      `"sigmoid"/"tanh"/"relu"` both in the constructor and in
      `set_params`, same `resolve_*` pattern as XGBoost's `objective` in
      `boosting.rs`) — batch-only, supports classif+regress. `DeepForest`
      via `define_learner!` in `trees.rs` (self-contained, all params have
      sensible defaults, same shape as `ObliqueForest`). `CostSensitiveClassifier`
      in `smelt-py/src/learners/ensemble.rs` alongside `Bagging`/`Stacking`
      (`base` learner selected by id string, eagerly validated; `cost_matrix`
      passed as nested Python lists, validated lazily by the wrapped
      Rust `train_classif` against the task's actual n_classes rather than
      eagerly, since -- unlike a bad learner id -- a bad cost-matrix shape
      surfaces as a clean `Result::Err`, not a `.expect()` panic). Verified
      via `maturin develop --release` + a direct Python script: ELM fits
      both task types well and rejects `activation="bogus"`;
      `CostSensitiveClassifier` with a 20x false-negative cost visibly
      flips boundary-point predictions from class 0 (plain
      `LogisticRegression`) to class 1, and rejects an unknown `base` id;
      `DeepForest` reaches 100% accuracy on a simple synthetic boundary and
      exposes `predict_proba`. All 3 round-trip `get_params`/`set_params`.

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
