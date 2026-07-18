# Changelog

All notable changes to `smelt-ml` (Rust crate, crates.io) and `smelt-py`
(Python bindings, published as `smelt-ml` on PyPI) are documented here.
Versions are tracked independently per this workspace's existing
convention: the Rust crate follows its own semver, the Python bindings
follow their own (currently a minor/patch cadence). Convention: a change
that alters numerical results or a training default ships in a MAJOR or
MINOR release with an explicit changelog entry -- never silently in a
patch (established after 2.0.1 changed the RF/ET regression default in a
patch; defensible as a bug fix, not a precedent to repeat).

## [Unreleased]

LOW-severity backlog of the 4th engine audit (2026-07-10) plus the full
remediation of the 5th engine audit
(`docs/auditoria_motor_2026-07-17.md`: 0 CRITICAL / 1 HIGH / 10 MEDIUM /
~17 LOW — the HIGH was process-side). Mostly validation, determinism,
and documentation fixes; the entries below include some that can change
numerical results, so per the convention above this section ships in a
MINOR, not a patch.

### Fixed — 5th engine audit (2026-07-17): Rust core

- CatBoost: model files written by 2.0.x–3.0.0 with the legacy
  object-form `cat_encodings` wire format **load again** — the 3.1.0
  persistence fix (`f9e028f`) silently broke them with an opaque serde
  error. The deserializer now accepts both wire forms (untagged);
  round-trips of current files stay bit-identical (M-2).
- `Pipeline` now propagates `feature_types` through every stage (new
  `Transformer::transform_types`, mirroring `transform_names`;
  `FilterSelector`/`RFE` select types, `PCA`/`OneHotEncoder` emit
  Numeric for derived columns). Previously even an empty pipeline reset
  all features to Numeric, silently disabling XGBoost/LightGBM native
  categorical splits (audit probe: accuracy 1.000 direct vs 0.623
  through a no-op pipeline). **Pipelines over categorical tasks can
  produce different — better — results** (M-3).
- GeoXGBoost: `select_bandwidth` and training now return a clear
  `InvalidParameter` when the dataset cannot satisfy the 30-neighbour
  minimum (n − 1 < 30) instead of silently clamping every candidate to
  n − 1 — which made the bandwidth sweep fictitious (identical scores,
  meaningless "best"). Erroring is the conservative choice; a
  warning-mode alternative is a question for the co-designed method's
  discussion (M-1).
- Degenerate-parameter validation at train time (`InvalidParameter`):
  `CatBoost::with_max_bins(0|1)` (was a silent constant model),
  `Bagging` with `n_estimators = 0`, ELM with `n_hidden = 0` (the old
  silent clamp-to-1 is removed), Mondrian `lifetime` negative or NaN.
  `TrainedDES::predict` validates feature count (was silent zip
  truncation in the neighbour search). `CQR::calibrate` and
  `ConformalClassifier::calibrate` reject mismatched calibration
  lengths (`DimensionMismatch`), matching `SplitConformal`.
- `ClassificationTask::with_class_names` now panics immediately with a
  clear message when given fewer names than `max(label) + 1`
  (documented under `# Panics`) — previously the mismatch surfaced as
  an index-out-of-bounds deep inside consumers such as SMOTE.
- DeepForest: per-fold and final-layer tasks propagate `class_names`
  (closes M-9 of the 4th-audit backlog — rare classes keep the full
  probability width in the cascade's OOF features).
- `QuantileForest` implements `feature_importance()` (same
  weighted-gain accounting as RandomForest; previously always `None`).
  Old serialized QRF files still load (fields are `serde(default)`).
- LightGBM: the plain-GBDT default path skips the GOSS
  sort-by-gradient entirely (it selected everything with weight 1
  after an O(n log n) sort per tree, per class). Selection, weights
  and RNG state are pinned identical to the general path; **outputs
  can differ from 3.1.0 at floating-point-reassociation (ulp) level**
  because accumulation order changes — documented in the regression
  test.
- Docs: variogram WLS attribution corrected (Cressie 1985 = gstat
  `fit.method = 2`; gstat's *default* is method 7); XGBoost
  `with_objective` publicly documents that early stopping under
  `Objective::Custom` monitors plain MSE on the raw score.

### Fixed — 5th engine audit (2026-07-17): smelt-py

- Regression `fit(X, y)` (and every other float-target entry point:
  eval sets, `GeoXGBoost`/`KrigingHybrid`/`QuantileForest` fits,
  `BayesianOptimizer.optimize`, the five causal `estimate()`s) rejects
  NaN/±inf targets with a `ValueError` naming the first bad index.
  Previously the model trained fine and predicted all-NaN with no
  error — the CSV loaders were fixed in 3.1.0 but the main API path
  was not (M-4).
- `BayesianOptimizer.optimize`: tuning `huber_delta` without also
  tuning `objective` to include `"huber"` is now a `ValueError`
  instead of a silent no-op (every trial trained the identical model)
  (M-5).
- **`SplitConformal` is now exposed in Python** (`SplitConformal(alpha)`,
  `calibrate_from_predictions(cal_pred, cal_truth)`,
  `predict_interval(test_pred)` → `(lower, upper)`), closing the gap
  that made the PM2.5 paper's conformal-over-`predict_spatial` flow
  unreproducible from the bindings; empirical coverage 92% at
  alpha=0.1 in the end-to-end KrigingHybrid probe. The
  `GeoXGBoost.conformal_predict` docstring no longer promises a
  nonexistent `coords` parameter and now points to `SplitConformal`
  for the spatial flow (M-6).
- `XGBoost` validates `objective` eagerly in the constructor and
  `set_params` (listing valid options), matching the
  KrigingHybrid/ELM convention, instead of failing at `fit()`.
- All 14 two-array measure functions (`rmse`…`brier`) validate equal
  lengths — mismatched inputs previously zip-truncated and could
  report perfect scores (`rmse` 3-vs-1 → 0.0, `accuracy` → 1.0).
- Composite learners (`DeepForest`, `Bagging`, `Stacking`,
  `DynamicEnsemble`, `CostSensitiveClassifier`) raise
  `NotImplementedError` with an explanation from `save()` (was an
  opaque `RuntimeError`); `brier_score` rejects malformed probability
  input with a clear `ValueError` (was a PyO3 `TypeError`) and accepts
  sklearn-style 1D positive-class probabilities like `auc_roc_score`.

### Fixed — 5th engine audit (2026-07-17): process

- **Publishing to PyPI is now gated on the test suite**: `release.yml`
  runs the full suite (default + `parquet` features) on the tag's ref
  and `publish` depends on it. Previously a tag pushed from a red — or
  never-CI'd — tree built and published wheels (the audit's only HIGH).
- Crate packaging switched to an `include` allow-list: internal audit
  reports, private correspondence drafts, and process docs are no
  longer published to crates.io (156 → 124 files) (M-9).
- New `python-smoke` CI job builds the real maturin wheel and runs
  `smelt-py/tests/test_smoke.py` (import/version, fit/predict,
  bit-identical save/load round-trip, a short tuner run) — the first
  permanent Python test surface in the repo (M-10).
- GitHub Actions pinned by commit SHA; tracked `__pycache__/*.pyc`
  removed from git and ignored; `docs/roadmap_checklist.md` brought up
  to date.

### Added — mlr3-parity roadmap, item 1 (2026-07-18)

- `TargetTransformRegressor`: trains any base `Learner` (factory pattern,
  like `Bagging`/`CostSensitiveClassifier`) on a transformed regression
  target — `Log`, `Log1p`, `Sqrt`, or `Standardize` (train-set mean/std,
  `StandardScaler` zero-variance convention) — and applies the inverse
  automatically at predict time, returning original-scale predictions
  (mlr3's `po("targettrafo")` / sklearn's `TransformedTargetRegressor`).
  Domain violations (`Log` with y ≤ 0, non-finite targets, …) are
  rejected naming the first offending index; feature names/types
  propagate to the base learner; the log retransformation bias
  (naive `exp` estimates the conditional median, not the mean) is
  documented, with Duan smearing noted as a possible future opt-in.
  Python: `TargetTransformRegressor(base="xgboost", transform="log")`
  with eager validation of both arguments in the constructor and
  `set_params`.

### Added — KrigingHybrid variogram upgrade (PM2.5 handoff, gap 3b)

- `VariogramModel::Matern32`/`Matern52`: Matérn ν=3/2 and ν=5/2 closed
  forms (sklearn length-scale convention: `√3h/r`, `√5h/r`). ν=1/2 is
  exactly `Exponential` and ν→∞ is `Gaussian`, both already present;
  continuous ν (Bessel K_ν) is deliberately out of scope for this
  crate's hand-rolled numerics. Python:
  `KrigingHybrid(variogram_model="matern32"/"matern52")`.
- `fit_variogram` now minimizes Cressie's (1985) WLS objective
  (`Σ N_j (γ̂_j − γ_j)²/γ_j²` -- relative misfit; gstat's
  `fit.method = 2`, though gstat's own *default* is method 7, which
  weights by `N_j/h_j²`) instead of plain pair-count-weighted SSE, with a two-stage
  grid search (coarse + ±1-step local refinement). **Changes fitted
  variogram parameters and therefore `predict_spatial` outputs** for
  existing `KrigingHybrid`/`predict_spatial` users -- the old absolute
  SSE let large-semivariance long-range bins dominate, fitting worst
  exactly the short-range structure kriging uses.

### Changed — numerical behavior (4th audit LOWs)

- LightGBM GOSS: the small-gradient amplification factor is now the exact
  finite-sample correction `|rest| / |sampled|` instead of the asymptotic
  `(1-top_rate)/other_rate` with the denominator clamped at 0.01. Opted-in
  `other_rate < 0.01` was silently under-amplified (up to 2x at 0.005);
  even standard rates drift by ceil-rounding when `n * rate` isn't
  integral. GOSS is opt-in since 3.0.0, so default models are unaffected.
- XGBoost with `Objective::Huber` + eval set: early stopping now monitors
  the Huber loss instead of plain MSE, which was dominated by exactly the
  outliers Huber is designed to resist -- the stopping round can change
  for those configurations.
- LightGBM `feature_importance`: a degenerate (reverted) split no longer
  credits its gain to the feature -- importances can differ where the
  guard fires (not observed in practice; the split finder only proposes
  two-sided splits).

### Fixed / Added (4th audit LOWs, behavior-preserving)

- Tuners (RandomSearch/BayesianOptimizer/Hyperband) validate the
  ParamSpace up front (`Uniform(lo>hi)`, `LogUniform(<=0)`, empty
  `Choice` were panics mid-loop); Hyperband validates `eta >= 2`.
- `friedman_test` rejects zero scores per model (was NaN ranks).
- `SpatialBlockCV` drops folds with an empty train or test side instead
  of emitting NaN-poisoning splits; errors if none remain.
- prelude re-exports `ParamValue`/`ParamSet`/`ParamGrid`/`ParamSpace`.
- HoeffdingTree breaks split-gain ties by feature index (was HashMap
  iteration order -- nondeterministic across processes).
- `Adasyn` rejects `k_neighbors=0` (was NaN ratios, zero synthetics).
- `DynamicEnsemble` propagates base-model predict errors and rejects
  non-classification base predictions (were silently swallowed;
  val_predictions could misalign with models).
- `silhouette_score` handles non-contiguous cluster labels.
- `CatBoost::with_max_bins` exposes the histogram resolution previously
  hardcoded at 64 (official `border_count` default is 254 -- documented
  divergence); default unchanged.
- `TrainedModel::feature_importance` documents its column-order contract.
- `TrainedCatBoost` implements `feature_importance()` (gain-based, summed
  over oblivious-tree levels, normalized like XGBoost/LightGBM here);
  `ObliviousTree` stores per-level gains (`serde(default)` -- models
  serialized before this load fine and fall back to split counting).

### smelt-py (4th audit LOWs)

- `RandomForest`/`ExtraTrees`/`DecisionTree` accept `max_depth=None`
  (unlimited, the Rust default) and reject `max_depth=0`, which used to
  train a root-only constant tree that predicted at chance silently.
- `CatBoost` exposes `feature_importances_` like XGBoost/LightGBM.
- Invalid-input errors now raise `ValueError` instead of `RuntimeError`
  across the bindings: unknown learner id / metric / activation /
  variogram model / objective, malformed or unknown `param_space`
  entries, non-integer class labels in `y_pred`, bad `coords` shape or
  length, non-finite coords. `KrigingHybrid` validates `variogram_model`
  eagerly in the constructor (it only surfaced at `fit`).
- `mape_score` and `logloss_score` exist now -- both were listed in
  `tuning._MINIMIZE_METRIC_NAMES` since 0.4.x but never actually bound,
  so using them as a tuning metric was an immediate NameError.
- `GeoXGBoost`/`KrigingHybrid` `save()`/`load()` raise a clear
  `NotImplementedError` explaining the composite-model limitation
  instead of a bare AttributeError.
- CSV/Parquet loaders return `y` as a numpy array (was a Python list,
  inconsistent with `x`) and release the GIL during file parsing;
  `save`/`load` release it during JSON (de)serialization.

## [smelt-ml 3.1.0] / [smelt-py 0.7.0] - 2026-07-16

Closes Tier 3 of the 4th engine audit (M-3, M-7, M-13, M-19). Ships as a
MINOR, not a patch, per the convention above: M-3's incremental split
sweep can flip exact ties between equal-gain candidate splits
(floating-point-rounding-level differences), and M-7 changes ANOVA
degrees of freedom when a class is absent from a fold.

### Changed — numerical results (entry added retroactively 2026-07-17)

- GeoXGBoost: `select_bandwidth` now rejects candidate bandwidths below a
  30-neighbour minimum (`MIN_BANDWIDTH`, per the co-designed GWR
  reference implementation: geographically weighted fits are unreliable
  below ~30 units), and the automatic bandwidth search therefore explores
  a restricted candidate set. **Users relying on automatic bandwidth
  selection can get different predictions than under 3.0.0.** The LOO
  criterion docstring was also corrected (skipping sparse neighbourhoods
  favors small bandwidths, not large ones). This shipped in 3.1.0 without
  a changelog entry — added retroactively after the 5th engine audit
  (2026-07-17) flagged the omission; the convention in this file's header
  requires such entries at release time.

### Fixed (M-19, 4th audit Tier 3 — Python bindings)

- `QuantileForest` in Python now exposes `predict_quantile(x, q)` and
  `predict_interval(x, alpha=0.1)` (dict with `predictions`/`lower`/
  `upper`/`alpha`, same shape as `conformal_predict`) — its entire reason
  to exist; previously the binding stored the model behind the generic
  `TrainedModel` trait and only the median was reachable, making it a
  worse `RandomForest`. The wrapper now holds the concrete
  `TrainedQuantileForest` (GeoXGBoost/KrigingHybrid pattern) with the
  full previous surface preserved (`fit`/`predict`/`get_params`/
  `set_params`/`save`/`load`/explain methods). Rust gains the matching
  concrete `QuantileForest::fit` (the `DeepForest::fit` shape) plus
  `TrainedQuantileForest` in the prelude, and `predict_quantile`/
  `predict_interval` now reject out-of-range `quantile`/`alpha` with
  `InvalidParameter` instead of silently clamping to the nearest leaf
  value.

### Fixed (M-13, 4th audit Tier 3 — Python bindings)

- `BayesianOptimizer.optimize` now validates every `param_space` name
  against the exact set its learner factory reads, raising a `ValueError`
  that lists the tunable parameters. Previously a typo'd or unwired name
  (e.g. `{"objectve": [...]}`, or `min_samples_split` for xgboost) was
  "tuned" silently: every trial trained the identical model and
  `best_params` was meaningless.
- `objective` (and `huber_delta`) are now actually tunable for
  `learner_type="xgboost"` — the string-choice use case the 3.0.0 typed
  `ParamSet` was built for was a silent no-op end-to-end. Objective
  choice values are validated eagerly, before any training.
- `XGBoost` with an invalid `objective` now raises `ValueError` from
  `fit` instead of `RuntimeError`, matching the invalid-parameter
  convention used everywhere else in the bindings.

### Fixed (M-7, 4th audit Tier 3)

- `FilterSelector::anova_f`/`information_gain` (and the Python
  `filter_anova_f`/`filter_information_gain` functions) now reject a
  continuous target with a clear `InvalidParameter`/`ValueError` instead
  of silently degenerating: `t as usize` used to make (nearly) every
  value its own class, driving the ANOVA F to ∞ for every feature —
  de-facto random selection — and `n_classes = max + 1` allocated memory
  proportional to the label magnitude (~8 GB for targets ~1e9). Both
  filters now also size their per-class buffers by the *distinct* labels
  present (never `max + 1`), and ANOVA degrees of freedom come from the
  groups actually in the data — matching scikit-learn's `f_classif` when
  a CV fold is missing a class entirely (previously a phantom empty
  group deflated F). Filters designed for continuous targets
  (`correlation`, `mutual_info`, `relief`, mRMR/JMI/JMIM/CMIM) are
  unaffected.

### Changed — performance (M-3, 4th audit Tier 3)

- `ObliqueTree`/`ObliqueForest`, `QuantileForest`, and `AdaBoost` split
  search rewritten from a per-candidate O(n) rescan to the same
  incremental sweep `TreeBuilder` got in 3.0.0, including the centered
  running sums that guard the regression paths against catastrophic
  cancellation on targets with large additive offsets (UTM coordinates,
  timestamps). Measured on n=4000, p=12: QuantileForest ~20× faster
  (951 ms → 49 ms), ObliqueTree ~80× (2.31 s → 29 ms), AdaBoost ~460×
  (23.7 s → 51 ms). Classification splits are computed from the exact
  same per-class counts as before (bit-identical); regression gains and
  AdaBoost's weighted error agree with the old rescan to floating-point
  rounding, which can flip exact ties between equal-gain candidate
  splits — hence this ships in a minor, not a patch, per this
  changelog's convention.

## [smelt-ml 3.0.0] / [smelt-py 0.6.0] - 2026-07-10

Everything on master since 2.0.1/0.5.1: the Fase F closures of the 3rd
audit (2026-07-05 → 2026-07-09), the Fase G remediation of the 4th audit
(`docs/auditoria_motor_2026-07-10.md`), its Tier 1/Tier 2 MEDIUM fixes,
and the PM2.5 case-study additions (TimeSeriesCV, SplitConformal).

### Breaking changes (Rust crate — next release must be smelt-ml 3.0)

- `ParamSet` is now `HashMap<String, ParamValue>` instead of
  `HashMap<String, f64>`, and `ParamGrid`/`ParamDistribution::Choice`
  carry `ParamValue` (typed `Float`/`Int`/`Str`/`Bool`) instead of `f64`.
  User code doing `params["k"] as usize` must switch to
  `params["k"].as_usize()` (or match on the variant). This is the driver
  for the next major bump.

### Changed — numerical results

- *(entry added retroactively 2026-07-17, flagged by the 5th engine
  audit)* `LinearSVM`: the per-sample shrink implemented an effective
  regularization of `λ = 1/C` **per sample** — a factor of *n* more than
  the standard `½‖w‖² + C·Σ hinge` objective (sklearn/Pegasos
  convention, now `λ = 1/(C·n)`) — and the learner gained internal
  feature standardization (like LogisticRegression/ELM). With defaults on
  trivially separable data, training accuracy goes from chance level
  (~0.51–0.54) to ~1.0; every LinearSVM model changes. This was HIGH-5
  of the 4th audit and the single largest behavior change in 3.0.0; it
  shipped without a changelog entry.
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
- LightGBM defaults to plain GBDT: GOSS is now opt-in
  (`top_rate`/`other_rate`, paper values 0.2/0.1), matching the official
  implementation. Previously every tree trained on ~30% of the rows by
  default — and Python had no way to disable it. The Python wrapper also
  stops forcing `max_depth=6` (now uncapped by default, like Rust and
  the official implementation) and exposes the GOSS rates.
- `DecisionTree`/`RandomForest` regression: split search now centers its
  running sums on the node mean. Targets carrying a large additive offset
  (UTM northing ~7e6, timestamps) degraded up to ~40× in RMSE under the
  incremental sweep introduced post-2.0.1.

### Added

- `TimeSeriesCV` — rolling-origin / walk-forward cross-validation for
  time-ordered data (expanding or sliding training window, forecast
  `horizon`, `step`, optional `gap` embargo). In the prelude and bound in
  smelt-py. Covers walk-forward validation (e.g. the PM2.5 case study's
  temporal track) the way SpatialBlockCV/SpatialBufferCV cover spatial
  leakage.
- `SplitConformal` — split-conformal calibration from precomputed
  predictions (`calibrate_from_predictions` + `intervals_for`), so models
  whose predictor needs more than features (KrigingHybrid/GeoXGBoost
  `predict_spatial`) get calibrated intervals end-to-end.
  `ConformalRegressor` now delegates to it and both reject
  mismatched calibration lengths instead of silently zip-truncating.
- `examples/pm25_spatial_loso.rs` + `data/pm25_santiago_spatial.csv` —
  PM2.5 Santiago case study: Regression Kriging under Leave-One-Station-Out
  CV with conformal intervals calibrated against the kriging model (see
  `docs/pm25_spatial_handoff.md`).
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

- SMOTE/ADASYN/SpatialSmote reject NaN features (they interpolated NaN
  into synthetic rows silently); CausalForest validates
  honesty_fraction/subsample_fraction instead of panicking inside rayon
  (and gains `with_subsample_fraction`); smelt-py `KMeans(k=0)` and
  mismatched silhouette labels raise ValueError instead of
  PanicException; numpy bool targets are accepted; tuner `best_params`
  reports the truncated integer value the winning model actually used.
- smelt-py version is single-sourced from `smelt-py/Cargo.toml`
  (`dynamic = ["version"]` + `importlib.metadata`) — the triple
  hand-synced copy drifted more than once.
- CSV loaders reject missing ("NaN"/"NA"/empty) and non-finite target
  values with an error naming the row, instead of silently training on
  f64::NAN (regression) or label-encoding "NA" as a class
  (classification). Missing values in features stay allowed.
- Macro Precision/Recall/F1 average over the union of observed labels
  (sklearn's convention) instead of 0..=max(label): gapped label ids no
  longer deflate the scores via phantom all-zero classes.
- `friedman_test` applies the tie-correction factor; tied scores within
  folds no longer make the test conservative (goldens vs scipy).
- smelt-py `benchmark()`: a misconfigured learner no longer aborts the
  whole run (per-fold `_error` instead); regression targets get an rmse
  default metric instead of crashing on accuracy; the classification
  heuristic matches `fit()`'s dtype dispatch; `y` may be a plain list.
- smelt-py `load()` resets hyperparameters to the constructor defaults
  instead of zeroed placeholders — refit after load no longer trains
  silently with `n_estimators=0` or fails on `objective ""`.
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
