# Changelog

All notable changes to `smelt-ml` (Rust crate, crates.io) and `smelt-py`
(Python bindings, published as `smelt-ml` on PyPI) are documented here.
Versions are tracked independently per this workspace's existing
convention: the Rust crate follows its own semver, the Python bindings
follow their own (currently a minor/patch cadence).

## [Unreleased]

Accumulates everything on master since 2.0.1/0.5.1: the Fase F closures
of the 3rd audit (2026-07-05 → 2026-07-09) and the Fase G remediation of
the 4th audit (`docs/auditoria_motor_2026-07-10.md`).

### Breaking changes (Rust crate — next release must be smelt-ml 3.0)

- `ParamSet` is now `HashMap<String, ParamValue>` instead of
  `HashMap<String, f64>`, and `ParamGrid`/`ParamDistribution::Choice`
  carry `ParamValue` (typed `Float`/`Int`/`Str`/`Bool`) instead of `f64`.
  User code doing `params["k"] as usize` must switch to
  `params["k"].as_usize()` (or match on the variant). This is the driver
  for the next major bump.

### Changed — numerical results

- `RandomSearch`/`Hyperband`/`BayesianOptimizer` with a fixed seed now
  assign RNG draws to parameters in sorted-key order instead of `HashMap`
  iteration order. Previously the same seed sampled **different
  configurations in different processes** (the reproducibility
  `with_seed` documents was never actually delivered); sampled
  configurations change once relative to any earlier run.
- `stats::mcnemar_test`/`friedman_test`: the chi-squared upper tail is
  now computed with the standard series/continued-fraction split in
  log-space (goldens vs scipy). Previously p collapsed to ≈1 for
  chi² ≳ 300 — "not significant" exactly when evidence was overwhelming —
  and could go slightly negative.
- `stats::sign_test`: binomial CDF in log-space; previously overflowed to
  a silent p=1.0 for n ≳ 1030.
- `survival::concordance_index`: unordered pairs counted once; tied event
  times with both events contribute 0.5. Previously ties were counted in
  both directions, dragging the C-index toward 0.5.
- `DecisionTree`/`RandomForest` regression: split search now centers its
  running sums on the node mean. Targets carrying a large additive offset
  (UTM northing ~7e6, timestamps) degraded up to ~40× in RMSE under the
  incremental sweep introduced post-2.0.1.

### Added

- `Pipeline::with_resampler(...)`: SMOTE/ADASYN as a pipeline stage,
  applied only during `train_classif` (never at predict; regression
  pipelines with a resampler are rejected).
- Model persistence for the streaming learners actually works now (see
  Fixed) and is covered by an executed save/load roundtrip test for every
  `SerializableModel` variant.
- `validate::check_coords_finite`, used by `KrigingHybrid`/`GeoXGBoost`
  train/predict entry points.
- smelt-py: model save/load (`save`/`load` methods), KMeans/DBSCAN/
  IsolationForest bindings, `CsvLoader`.

### Fixed

- Model files written by `MondrianTree`/`MondrianForest` (default
  lifetime), `HoeffdingTree`, `AdaptiveRandomForest`, and CatBoost with
  `cat_features` were **unloadable** (save succeeded, load always
  failed). Infinite `tau` now roundtrips as an explicit null (old broken
  Mondrian files become readable); integer-keyed maps serialize as pairs.
  serde_json's `float_roundtrip` feature is now enabled: without it,
  reloaded weights drifted by ulps and predictions were not bit-identical
  across a save/load cycle.
- A single non-finite coordinate made `KrigingHybrid` return all-NaN
  predictions silently and `GeoXGBoost` panic the process; both now
  return a clean error naming the offending index.
- `Pipeline::train_classif` and `Smote`/`Adasyn`/`SpatialSmote::balance`
  dropped `class_names`/`feature_names`/`feature_types`: probability rows
  narrowed when a split lost the highest class (panicking under
  `Stacking`/`DynamicEnsemble` with a `Pipeline` base), and resampled
  pipelines renamed features to `x0/x1/...` in selector output and
  importances.
- `TreeBuilder` split search is incremental (single sweep) instead of
  rescanning per threshold — ~29× faster RandomForest regression fits at
  n=5000 (superlinear before), with bit-identical classification splits.
- LightGBM `subsample` was accepted and ignored; real row bagging now
  composes with GOSS (note: the official implementation makes them
  mutually exclusive instead).
- CatBoost multiclass target statistics are computed per class
  (one-vs-rest) instead of from the raw class index; zero-gain splits
  stop tree growth; GBM classification leaves get a Newton step.

## [smelt-ml 2.0.1] / [smelt-py 0.5.1] - 2026-07-05

### Fixed

- `RandomForest` and `ExtraTrees` applied the classification-style
  `sqrt(n_features)` candidate-feature heuristic to regression too,
  unlike scikit-learn (`RandomForestRegressor`/`ExtraTreesRegressor`
  default to all features; only the `*Classifier` variants default to
  `sqrt`). Found by the empirical benchmark added in 2.0.0/0.5.0: on
  OpenML `pol` (48 features, few actually informative), this made
  `RandomForest` RMSE 111.8% worse than scikit-learn's, since many splits
  never saw an informative feature at all. Regression now uses all
  features by default, matching scikit-learn; classification is
  unchanged. RMSE on `pol` closes from +111.8% to -3.8% (now slightly
  *better* than scikit-learn). Fit time trades the other way (every split
  now searches all 48 features instead of 7) -- the same trade-off
  scikit-learn's own all-features default pays, not a regression.
  `with_max_features_sqrt()`/`with_max_features_fraction()` still
  override explicitly for both task types, unchanged for existing callers
  who already set them.

## [smelt-ml 2.0.0] / [smelt-py 0.5.0] - 2026-07-04

92 commits since the last published versions (`smelt-ml` 1.3.0, `smelt-py`
0.4.6). Driven mostly by a full engine audit
(`docs/auditoria_motor_2026-07-04.md`) plus two feature initiatives
(causal meta-learners, geospatial differentiators, Prioridad 3 quick
items — see `docs/roadmap_checklist.md`).

### Breaking changes (Rust crate only)

- `Resample::splits()` now returns `Result<Vec<(Vec<usize>, Vec<usize>)>>`
  instead of `Vec<(Vec<usize>, Vec<usize>)>` directly, surfacing malformed
  configuration (e.g. more folds than samples) as an error instead of
  panicking. Any external `Resample` implementer or direct caller needs to
  handle the `Result`.
- `stats::bootstrap_ci` now returns `Result<BootstrapCI>` (empty input /
  zero resamples are errors, not silent degenerate output).
- `stats::wilcoxon_signed_rank` was rewritten (exact test for n ≤ 100,
  tie-corrected normal approximation above that) and its module-level
  doctest example changed from 5 to 6 folds, since 5 folds can never reach
  p < 0.05 regardless of the underlying effect.
- These are why this is a major version bump rather than a minor one.
  Python users are unaffected: `smelt-py` already surfaced `Result`s as
  Python exceptions before this change.

### Added — new learners

- Causal meta-learners: `TLearner`, `SLearner`, `XLearner`, `RLearner`,
  `DrLearner` (heterogeneous treatment effect estimation), plus
  `CausalForest` OOB aggregation fix (see Fixed).
- `KrigingHybrid` — regression-kriging (base learner + kriged residual
  correction), with `predict_spatial`.
- `Smote`, `SpatialSmote` — synthetic minority oversampling, the spatial
  variant restricted to same-class neighbors within a max distance.
- `AdaptiveRandomForest` — online random forest with per-tree ADWIN
  concept-drift detection.
- `MondrianTree`, `MondrianForest` — online trees via a Mondrian process,
  consistent between batch and incremental construction; the first
  streaming *regressor* in the crate.
- `ExtremeLearningMachine` — single hidden-layer net with fixed random
  weights, closed-form ridge-regularized output layer.
- `CostSensitiveClassifier` — wraps any probabilistic classifier with a
  Bayes-risk decision rule under an explicit cost matrix.
- `DeepForest` (gcForest) — cascade of forest layers with internal k-fold
  CV and early stopping.

### Added — other

- Model registry: `learner_from_id`/`registered_learner_ids`.
- New measures: `BalancedAccuracy`, `CohensKappa`, `Mcc`, `Brier`.
- `ParquetLoader` behind a new opt-in `parquet` Cargo feature.
- Categorical feature + NaN support in `Task`/`CsvLoader`; native
  categorical splits in XGBoost/LightGBM; NaN handling fixes in CatBoost.
- `sample_weight` support in XGBoost classification (previously
  regression-only); monotone constraints and pluggable regression
  objectives (Huber, Poisson) in XGBoost.
- Held-out `eval_set` early stopping across XGBoost, LightGBM, CatBoost.
- `StratifiedCV`, `GroupCV` resampling strategies.
- Sparse one-hot encoding: `CsrMatrix`, `OneHotEncoder::transform_sparse`.
- `f32` histograms in CatBoost (~26% training speedup on the case measured).
- Rayon parallelism connected in `benchmark_design`, `GridSearch`,
  `RandomSearch`, `Hyperband` (`BayesianOptimizer` deliberately left
  sequential — TPE's sampler depends on the full accumulated history).
- Two new typed `SmeltError` variants (`IncompatiblePrediction`,
  `NumericalError`) replacing many stringly-typed `SmeltError::Other`.
- `#![warn(missing_docs)]` closed across the entire crate.
- Python: `get_params`/`set_params` on all learner wrappers; ~20 more
  learners exposed since 0.4.6 (all of the above, plus `Bagging`,
  `Stacking`, `DynamicEnsemble` via base-learner id strings); the GIL is
  now released during training/prediction for real multi-core speedup
  from Python; dead `smelt.conformal` module removed (it raised
  `ImportError` on import — the real API is `<model>.conformal_predict()`).

### Fixed

- XGBoost histogram-subtraction trick reused a never-populated pooled
  histogram when the smaller sibling was a trivial leaf, corrupting the
  larger sibling's split search (RMSE ~5.0 → ~0.29 on the affected case).
- CatBoost leaf-index bit-order and boundary-value routing bugs (pre-2026-04-20
  CatBoost benchmarks are unreliable — re-benchmark before citing).
- Macro-averaged `Precision`/`Recall`/`F1Score` divided by the count of
  classes with a defined score instead of the total class count, inflating
  scores for degenerate classifiers (verified against sklearn references).
- PCA power iteration could return a non-dominant, non-reproducible
  direction after deflating a near-singular covariance matrix; now falls
  back to a Gram-Schmidt orthogonal complement in that case.
- RReliefF (`ReliefFilter`) divided both the positive and negative
  weighted-kernel terms by the same denominator instead of their own.
- `CausalForest`'s per-tree CATE aggregation included each point's own
  in-bag trees, biasing estimates toward extreme in-sample outcomes.
- `ADASYN` could synthesize points across the gap between two separated
  minority clusters instead of only within a cluster's local neighborhood.
- `HoeffdingTree`'s split-quality estimator compared class means to a
  single threshold as an all-or-nothing assignment, making every feature
  (including pure noise) look like a perfect split — the tree never
  actually split. Fixed via each class's running Gaussian + normal CDF.
- `Stacking` and `benchmark::resample_classif/regress` could panic
  (index out of bounds) when a CV fold's training data lacked the
  dataset's maximum class label.
- `histogram.rs` boundary bug made binary/low-cardinality features
  unsplittable.
- `isolation_forest` missing-parenthesis bug in the `c(n)` normalization
  factor.
- `GeoXGBoost::predict()` could use a stale positional model instead of
  the global one; LOO bandwidth selection now compares out-of-fold errors
  on both sides of the adaptive alpha (Grekousis Eq. 11/13).
- `conformal` module: `alpha`/`n` are now validated, and intervals widen
  instead of silently clamping on underflow.
- `QuantileGB`, `KrigingHybrid` (duplicate coordinates, zero-variance
  variograms), `ObliqueTree` (wrong probability-vector width), several
  more — see `docs/auditoria_motor_2026-07-04.md` for the full list with
  before/after evidence per fix.

### Process note

Every correctness fix above shipped with a regression test verified to
fail against the reverted code and pass against the fix — see
`docs/auditoria_motor_2026-07-04.md`'s closing note on this crate's testing
policy.
