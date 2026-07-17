# Handoff: PM2.5 spatial track (Regression Kriging + LOSO) in smelt-ml

**For:** the Smelt team.
**From:** F. Parra — spatial track of Parra & Astudillo (2025), *Machine Learning for PM2.5
Prediction in Santiago de Chile* (target: *Environmental Modelling & Software*).
**Goal:** a working, compiling prototype that expresses the paper's spatial experiment
natively in `smelt-ml`, so the team can (a) adopt it as a case-study/example and
(b) close the small gaps noted below.

## What it is

Predict daily PM2.5 at an **unmonitored** location from spatial covariates, using
**Regression Kriging** (XGBoost trend + kriged residuals), validated
**station-by-station** with **Leave-One-Station-Out CV**. This is the exact spatial
method of the paper, which in Python reported a mean LOSO R² = **−1.09** (spatial
interpolation fails at 7–10 km satellite resolution — the point of the experiment is
the honest negative result, not a high score).

## Files

| File | What |
|------|------|
| `examples/pm25_spatial_loso.rs` | The prototype. Compiles (`cargo check --example pm25_spatial_loso` → OK). |
| `data/pm25_santiago_spatial.csv` | 16,344 daily station-obs, 8 stations, 10 spatial covariates + `pm25` target + integer `station_id`. Real data from the paper. |

Run: `cargo run --release --example pm25_spatial_loso`

## Pipeline → Smelt API mapping

| Paper step | smelt-ml |
|-----------|----------|
| Load spatial table | `CsvLoader::from_path(...).target("pm25").load_regress()` |
| LOSO-CV (station = held-out unit) | `GroupCV::new(n_stations, station_ids)` with `n_folds == n_stations` → each fold leaves one station out |
| Regression Kriging (XGBoost trend + kriged residuals) | `KrigingHybrid::new(\|\| Box::new(xgb()) as Box<dyn Learner>, coords).with_variogram_model(VariogramModel::Spherical).with_n_neighbors(20)` → `train_regress_geo` → `predict_spatial(feat, coords)` |
| Metrics (Table 8) | `RSquared`, `Rmse`, `Mae` |
| Calibrated intervals (paper's undercalibration fix) | `ConformalRegressor::calibrate(&*model, cal_feat, cal_tgt, 0.1)` |

**This maps almost 1:1.** Everything the paper's spatial track needs already exists in
Smelt. The prototype runs the full LOSO loop, prints per-station R²/RMSE, and adds a
plain-XGBoost column so the residual-kriging contribution is visible.

## Gaps / requests for the Smelt team

1. **[CLOSED 2026-07-10]** ~~Conformalizing the spatial model.~~ Done via
   `SplitConformal::calibrate_from_predictions(cal_pred, cal_tgt, alpha)` +
   `intervals_for(test_pred)` (`src/conformal/mod.rs`): model-free calibration
   from precomputed predictions, so `predict_spatial` outputs conformalize
   directly. The example now calibrates against the kriging model itself
   (coverage 92% vs 90% nominal on Las Condes, half-width ±19.9 µg/m³).
   Original request:
   **Conformalizing the spatial model.** `ConformalRegressor::calibrate` takes a base
   model that predicts from `features` alone, but `TrainedKrigingHybrid` needs `coords`
   at predict time (`predict_spatial`). So the prototype's conformal block falls back to
   a plain-XGBoost base. A **CQR path for spatial learners** (calibrate against
   `predict_spatial`) would let PM2.5 prediction intervals be calibrated end-to-end —
   exactly the fix for the paper's undercalibration (61% vs 90% nominal).
   `conformal/cqr.rs` already has the machinery; it needs a spatial-aware adapter.

2. **[CLOSED 2026-07-10]** ~~Temporal resampling.~~ Done: `TimeSeriesCV`
   (`src/resample/time_series.rs`) — rolling-origin/walk-forward with
   expanding or sliding window, `horizon`, `step`, and optional `gap`
   (embargo). In the prelude and bound in smelt-py (`TimeSeriesCV(horizon,
   min_train_size=None, step=None, max_window=None, gap=0)`).
   Original request:
   **Temporal resampling (for the *other* track).** `resample/` has `CrossValidation`,
   `GroupCV`, `SpatialBlockCV`, `SpatialBufferCV`, `StratifiedCV`, `Holdout` — but no
   **rolling-origin / walk-forward** resampler. The paper's temporal track (the main
   result, R²=0.76) uses walk-forward validation. A `TimeSeriesCV` / `ForwardChainCV`
   (expanding or sliding window, forecast horizon, step size) would let Smelt cover the
   temporal track too. Small, self-contained addition.

3. **[STAGE (b) CLOSED 2026-07-17]** The Smelt team's position (2026-07-10): adopting
   `geostat-rs` as a dependency conflicts with the crate's hand-rolled-numerics
   policy (`CsrMatrix`, grid-search variogram precedents). Staged instead:
   (a) short term, the external composition in `rk_smelt_geostat/` stays the
   documented publication-grade path; (b) **[DONE 2026-07-17]** upgrade the native
   variogram fit to WLS + Matérn (self-contained, no new dependency) —
   `fit_variogram` now minimizes Cressie's (1985) WLS objective
   (`Σ N_j (γ̂_j − γ_j)²/γ_j²`, the gstat-default family) with a two-stage
   grid search (coarse + local refinement), and `VariogramModel` gains
   `Matern32`/`Matern52` closed forms (sklearn length-scale convention;
   ν=1/2 ≡ Exponential, ν→∞ ≡ Gaussian; continuous ν stays out of scope —
   that's what the external geostat-rs path is for). Python:
   `KrigingHybrid(variogram_model="matern32"/"matern52")`;
   (c) an optional feature-gated backend only if the paper requires it
   (like the `parquet`/polars precedent). Original request:
   **Publication-grade residual kriging (prototyped).** `KrigingHybrid` fits an internal
   spherical variogram with `n_lags`/`n_neighbors`. For a defensible *methods* section,
   **geostat-rs** (`crates/geostat-core`) provides WLS-fitted variograms
   (spherical / exponential / Gaussian / Matérn continuous ν), geometric anisotropy, and
   ordinary **and universal** kriging (`Kriging::predict_with_drift`), validated against
   gstat. A **working composition already exists** — Smelt XGBoost trend + geostat-rs
   `RegressionKriging` — in the standalone crate
   `PM25_Santiago/prototypes/rk_smelt_geostat/` (compiles and runs; mean LOSO
   R² = −1.04, the closest of the three implementations to the paper's −1.09). It shows
   `geostat-core::regression::RegressionKriging::{new, predict}` takes an external trend
   evaluated at the data and targets, so any Smelt regressor can drive it. Adopting this
   as an optional backend would upgrade `KrigingHybrid` from "pragmatic" to
   "publication-grade" without changing its public API.

## Result of the prototype run

Actual output (`cargo run --release --example pm25_spatial_loso`):

```
Held-out station        n      RK R2     XGB R2    RK RMSE
----------------------------------------------------------
Las Condes           2413     -9.041     -9.459      38.73
Pudahuel             2432     -1.425     -1.306      25.20
Independencia        2491     -3.516     -3.617      42.69
El Bosque             904     -1.102     -1.038      20.91
Cerro Navia          2436      0.681      0.695      12.11
Parque O'Higgins     2460      0.429      0.438      23.09
Cerrillos II         1287     -0.972     -1.165      48.16
Talagante            1921      0.358      0.362      14.14
----------------------------------------------------------
Mean LOSO R2 (Regression Kriging) = -1.824
Median LOSO R2                     = -1.037

─── Conformal intervals (90% target) ───
  Held-out Las Condes: empirical coverage 91% (target 90%), n = 2413
```

Three things to note:

- **Reproduces the paper's conclusion.** Spatial interpolation fails: most stations R² < 0,
  only 2–3 generalize (Cerro Navia, Parque O'Higgins, Talagante). The mean (−1.82 here vs
  −1.09 in the paper) differs because this prototype uses a reduced 10-covariate set (no
  MODIS AOD, which wasn't in the exported CSV) and Smelt's built-in variogram; the
  **qualitative result is identical**.
- **Residual kriging adds ~nothing** (RK R² ≈ XGB R² per row). That is itself the finding:
  at 7–10 km covariate resolution there is no exploitable spatial autocorrelation structure
  in the residuals for kriging to recover.
- **Conformal prediction is calibrated** (91% empirical vs 90% nominal) — directly fixing
  the paper's undercalibration (61% vs 90%). This works today for the plain-XGBoost base;
  see gap #1 for doing it against the kriging model.

The value of the prototype is the **pipeline** — the same honest experiment as the paper,
in a fraction of the Python code, with no environment fragility (the Python run hit an
XGBoost/OpenMP crash during the manuscript review that prompted this port).
