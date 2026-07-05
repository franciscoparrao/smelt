# Changelog

All notable changes to `smelt-ml` (Rust crate, crates.io) and `smelt-py`
(Python bindings, published as `smelt-ml` on PyPI) are documented here.
Versions are tracked independently per this workspace's existing
convention: the Rust crate follows its own semver, the Python bindings
follow their own (currently a minor/patch cadence).

## [smelt-ml 2.0.0] / [smelt-py 0.5.0] - unreleased

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
