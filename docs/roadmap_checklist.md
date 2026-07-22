# smelt-ml Roadmap Checklist

*Basado en análisis de 23,045 papers de Web of Science (2006-2026)*
*Última actualización: 2026-07-17*

---

## Estado actual del framework

- **21 learners supervisados** + K-Means + DBSCAN + CausalForest
- **10 métricas** + Silhouette
- **9 transformers** (Scalers, Imputer, OneHot, Label, SMOTE, PCA, RFE, FilterSelector)
- **4 tuners** (Grid, Random, TPE, Hyperband)
- **4 resamplers** (CV, Holdout, SpatialBlock, SpatialBuffer)
- Conformal Prediction, Benchmark Design, Permutation Importance
- CSV loading, JSON serialization, input validation
- **202+ integration tests**, publicado en crates.io v0.6.1

---

## Prioridad 1: CRÍTICO (evidencia masiva en la literatura)

- [x] **LightGBM** — GOSS + leaf-wise growth + histogram splits. 32,467 citas. ✅ v0.8.0
- [x] **CatBoost** — Ordered target statistics + oblivious trees + native categoricals. ✅ v0.8.1
- [x] **Random Survival Forest** — Log-rank split + Nelson-Aalen + C-index. 1,000+ papers. ✅ v0.9.0
- [x] **Isolation Forest** — Anomaly detection via random path length. s(x,n) = 2^{-E[h(x)]/c(n)}. 700+ menciones. ✅ v0.6.1
- [x] **TreeSHAP** — SHAP values para cualquier modelo (sampling approach). 9,000+ refs. ✅ v0.9.1

## Prioridad 2: ALTO (dominios enteros ausentes)

- [x] **Multi-label Classification (Classifier Chains)** — Train C_j on X + previous labels. 2,000+ papers. ✅ v0.6.1
- [x] **Multi-output / Multi-target Regression** — Regressor Chains. 1,000+ papers. ✅ v0.7.0
- [x] **Quantile Regression Forest** — QRF con weighted quantiles de las hojas. 602 papers. ✅ v0.7.0
- [x] **Hoeffding Trees (Streaming)** — VFDT + Hoeffding bound. Online partial_fit(). ✅ v0.9.2
- [x] **Dynamic Ensemble Selection (KNORA-E)** — Selección dinámica por instancia. ✅ v0.9.2

## Prioridad 3: MEDIO (mejoras y extensiones)

- [x] **ADASYN** — Oversampling adaptivo por densidad local. 4,070 citas. ✅ v0.7.0
- [x] **Extreme Learning Machine (ELM)** — Random weights + pseudoinverse. Ultra-rápido. ✅ `src/learner/elm.rs` (2026-07-04)
- [x] **Deep Forest / gcForest** — Cascada de forests emulando deep learning. ✅ `src/learner/deep_forest.rs` (2026-07-04)
- [x] **Cost-Sensitive Learning** — Soporte de cost matrix en learners. Esencial para medicina/finanzas. ✅ `src/learner/cost_sensitive.rs` (2026-07-04)
- [x] **Conformalized Quantile Regression (CQR)** — CP + quantile regression. Intervalos adaptativos. ✅ v0.7.0

## Prioridad 4: DIFERENCIADORES (único en Rust)

- [x] **Kriging-ML Hybrid** — Regression-kriging con XGBoost/RF para residuos espaciales. 15+ papers en Q5_3. ✅ `src/learner/kriging_hybrid.rs` (2026-07-04)
- [x] **Spatial-SMOTE** — SMOTE geoespacialmente informado. ✅ `src/preprocess/spatial_smote.rs` (2026-07-04)
- [x] **Uplift Modeling (X/R/DR-Learners)** — Meta-learners para efectos causales. ✅ `src/causal/meta_learners/` (2026-07-03, ver también T/S-learner)
- [x] **Online Adaptive Random Forest** — RF incremental con detección de concept drift (ADWIN). ✅ `src/learner/adaptive_rf.rs` (2026-07-04)
- [x] **Mondrian Forest** — Árboles con proceso de Mondrian para partición incremental consistente (batch y online consistentes: construcción por lotes vía `sample_mondrian_block`, extensión online vía `extend_node`). ✅ `src/learner/mondrian.rs` + bindings Python (2026-07-04)

## Prioridad 5: NICE-TO-HAVE

- [ ] **WildWood** — Context tree weighting sobre todos los subárboles. Novel (2023). ~300 líneas.
- [ ] **Adversarial Random Forests** — Density estimation generativa con RF. ~250 líneas.
- [ ] **Fairness Metrics** — Disparate impact, equalized odds, demographic parity. ~150 líneas.
- [ ] **OpenML Integration** — Cargar datasets de OpenML por ID. ~100 líneas.
- [x] **Parquet Loading** — `ParquetLoader` detrás del feature opcional `parquet` (dep polars, zero-cost apagado). ✅ `src/data/parquet.rs` (2026-07-03)

## Prioridad 6: PARIDAD mlr3 (gap analysis 2026-07-18)

Gaps de infraestructura de composición frente a mlr3/mlr3pipelines/mlr3tuning,
priorizados por valor para el nicho GIS/científico (análisis conversado
2026-07-18; Smelt ya está *adelante* de mlr3 en conformal, causal, streaming
y learners espaciales — esto cubre lo inverso).

- [x] **Target-trafo con inversa automática** — `TargetTransformRegressor`
      (`src/learner/target_transform.rs`): wrapper `Learner` (patrón factory
      de Bagging/CostSensitive), transforms log/log1p/sqrt/standardize con
      validación de dominio nombrando el índice, inversa automática al
      predecir, propagación de feature_names/types (lección M-3), sesgo de
      retransformación documentado (Duan smearing = opt-in futuro). Binding
      Python con validación eager de `base`/`transform`. 13 tests
      (bit-identidad manual-vs-wrapper en los 4 transforms, composición con
      CV, log-normal RMSE < 0.8× base). ✅ (2026-07-18)
- [x] **Pesos por muestra** — `with_weights()`/`weights()` en ambas Tasks
      (validación inmediata), guard `check_no_weights` en todo learner sin
      soporte (nunca ignorados en silencio; precedente check_no_nan),
      `supports_weights()` en el trait, slicing correcto por folds de CV y
      propagación por Pipeline (resamplers los rechazan: pesos sintéticos
      indefinidos). Consumo real en 13 learners: DT/RF/ET/GBM (impureza y
      hojas ponderadas, MSE centrado — lección HIGH-1), XGBoost/LightGBM/
      CatBoost (grad/hess ponderados, ordered TS ponderados, GOSS sobre
      grads ponderados; unificado con el `with_sample_weights` previo de
      XGBoost), OLS/Ridge/Lasso/EN/LogReg/ELM (WLS, CD normalizado por Σw,
      estandarizaciones ponderadas). Oráculo peso-k ≡ fila duplicada k veces
      (bit-idéntico en DT, 1e-9 documentada donde el orden de suma difiere)
      + goldens sklearn. R-learner usa la R-loss EXACTA de Nie & Wager con
      bases ponderadas (PEHE 0.272 vs 0.318 de la aproximación; fallback
      documentado para bases sin soporte). Python: `fit(..., sample_weight=)`
      en ~30 wrappers + `supports_sample_weight` + validación pre-panic.
      LinearSVM y GeoXGBoost excluidos deliberadamente (semántica SGD por
      decidir / co-diseño con George). 66 tests nuevos (740 total).
      ✅ (2026-07-18)
- [x] **AutoTuner + nested CV** — `AutoTuner` (`src/learner/auto_tuner.rs`)
      envuelve (factory + `TunerSpec` de los 4 tuners + inner resampling +
      measure) como un `Learner`: al entrenar corre el tuner sobre la task
      recibida, elige best_params y reentrena el modelo final. Metido en
      `benchmark::resample_*` con CV externo *es* nested CV sin leakage por
      construcción — probado con un learner-sonda que verifica que el tuning
      de cada fold externo nunca vio filas del test externo.
      `TrainedAutoTuner` expone `best_params()`/`best_score()`/`history()`.
      Compone con seeds reproducibles (HIGH-6) y pesos por muestra (fluyen a
      los folds internos). Python `AutoTuner(learner, param_space, tuner=,
      cv=, metric=, seed=)` con `best_params_`/`best_score_` sklearn-style,
      reusando la allowlist del M-13. 9 tests Rust + sondas. ✅ (2026-07-18)
- [x] **Calibración de probabilidades + threshold tuning** —
      `CalibratedClassifier` (`src/learner/calibration.rs`, Platt + isotonic
      PAV hand-rolled, holdout fit/calib + refit-en-todo estilo
      `CalibratedClassifierCV(ensemble=False)`, OvR multiclase) y
      `ThresholdedClassifier` (`src/learner/threshold.rs`, umbral fijo o
      tuneado en holdout maximizando una Measure, `best_threshold()`
      inherente). Wrappers factory como CostSensitive; guard de pesos
      coherente; no registrados. Oráculos: Brier baja tras calibrar, Platt
      preserva AUC (monótona), F1 0.55→0.64 con umbral tuneado en
      desbalanceado. Python sklearn-style con `best_threshold_`. 17 tests.
      ✅ (2026-07-18). **Prioridad 6 (paridad mlr3) COMPLETA.**
- [x] **Registry con properties + autotest de contrato** — `LearnerProperties`
      (`src/learner/properties.rs`): 8 flags por learner (classif/regress/
      weights/proba/nan/categorical/feature_importance/serializable), fuente
      única = override `properties()` (el método `supports_weights()` ahora
      deriva de ahí). `learner_properties(id)` en registry + Python. El
      entregable: `tests/contract.rs` — 7 tests × 27 learners que verifican
      EMPÍRICAMENTE que cada property declarada coincide con el
      comportamiento real (train_* según task-support, check_no_nan/weights,
      proba suma≈1, feature_importance posicional Some/None, serializable,
      cero-panic universal). Pasó entero a la primera → 0 mismatches: la
      metadata es verdad verificada, no declarada. Falla ruidoso si alguien
      agrega un learner con properties mentirosas. ✅ (2026-07-18)
- [ ] **Menores / futuro**: plotting (ROC/critical-difference sobre el
      benchmark ya existente), Pipeline como DAG. Deep learning (mlr3torch)
      sigue **descartado deliberadamente** (sin autodiff en el workspace).
    - [x] **Terminators componibles** (2026-07-21) — trait `Terminator` +
      `TuningProgress` (n_evals/elapsed/best_score/maximize/
      evals_since_improvement) en `src/tuning/terminator.rs`. Criterios
      concretos `MaxEvals`/`RunTime`/`Stagnation`/`TargetScore` que componen
      vía `AnyTerminator` (OR) / `AllTerminator` (AND), p.ej. "20 evals OR 30s
      OR 5 rondas sin mejora". Cableados en `BayesianOptimizer` (el tuner
      secuencial; GridSearch/RandomSearch evalúan todo up-front, no hay corrida
      parcial que cortar) como early-stop opcional vía `with_terminator`,
      chequeado tras cada eval; sin terminator corre exactamente `n_iter` como
      antes (default sin cambios). `std::time::Instant` para wall-clock (el
      crate sí puede usarlo). 6 unit tests + 1 integración end-to-end
      (MaxEvals(4)→4 evals, TargetScore(0)→1 eval, AnyTerminator→primero que
      dispara, sin terminator→n_iter completo). Binding Python: kwargs
      opcionales `max_seconds`/`patience`/`target_score` en `optimize` que
      arman un `AnyTerminator` internamente (sin exponer los trait objects);
      verificado con `maturin develop` + script.
    - [x] **Tuning multi-objetivo (Pareto)** (2026-07-21) — `ParetoResult`
      (`front`/`all_results`/`measure_ids`/`maximize`) + `pareto_front_indices`
      (non-dominated sorting con dominancia Pareto respetando el flag
      `maximize` por objetivo; NaN tratado como el peor valor) en
      `src/tuning/mod.rs`. Métodos `tune_classif_multi`/`tune_regress_multi`
      en `GridSearch` y `RandomSearch` (mismos tuners que las dependencias;
      Bayesian/Hyperband guían por escalar → fuera de scope): evalúan cada
      config en TODAS las measures reusando `benchmark::resample_*` (que ya
      acepta `&[&dyn Measure]` y devuelve un score por measure vía
      `mean_scores()`) y devuelven el frente en vez de un único best. Reusan
      las dependencias (RandomSearch poda, GridSearch dedup). 7 unit tests
      (frente all-maximize, mixto maximize/minimize, duplicados, punto único,
      NaN, ParetoResult) + 1 integración end-to-end verificando la propiedad
      estructural del frente (ningún punto del frente dominado; todo punto
      fuera dominado por alguno del frente) con GridSearch real. Rust-only:
      RandomSearch/GridSearch no están bindeados en Python (solo Bayesian lo
      está).
    - [x] **Kernel SVM (C-SVC)** (2026-07-21) — `KernelSVM`/`TrainedKernelSVM`
      + enum `Kernel` (Linear/Poly/Rbf) en `src/learner/kernel_svm.rs`. SVM
      dual soft-margin resuelto con **SMO** (Platt 1998): sin dependencia de
      QP externo; kernel trick para fronteras no lineales genuinas (a
      diferencia del `LinearSVM` SGD+hinge preexistente). Multiclase
      one-vs-rest. `decision_function` inherente más allá del trait (shape de
      GeoXGBoost/GP; `fit`-devuelve-concreto, `train_classif` envuelve).
      `Learner` classifier-only, registrado como `"kernel_svm"` (29 en
      registry) → contract autotest (7×29). **Golden vs scikit-learn 1.8.0
      `SVC(rbf, gamma, tol)`**: decision values en test points match a 1e-2
      (ambos resuelven el mismo QP convexo → función de decisión única;
      diferencia solo por tolerancias de parada) y predicciones exactas. 5
      unit tests (golden, XOR no-lineal, OvR multiclase, kernel lineal,
      rechazo params/weights) + doctest. Binding Python en
      `smelt-py/src/learners/misc.rs` (`decision_function` 1D binario/2D
      multiclase, kernel string con validación eager en `__new__`/`set_params`
      como XGBoost/KrigingHybrid); verificado con `maturin develop` + script
      (golden sklearn vía API, poly/linear, get/set_params). README
      actualizado (35 supervised, fila, 29 registry). SVR (regresión) queda
      como extensión futura documentada.
    - [x] **GP standalone con `se`** (2026-07-21) — `GaussianProcess`/
      `TrainedGaussianProcess` en `src/learner/gaussian_process.rs`. GP
      regression con kernel RBF: posterior exacto vía Cholesky de
      `K = k(X,X)+αI`, `α_vec = K⁻¹y`, media `k*ᵀα_vec` y varianza
      `k(x*,x*) − k*ᵀK⁻¹k*`. La desviación estándar predictiva (`se`) — el
      punto de usar un GP — se expone más allá del trait vía `predict_std`/
      `predict_with_std` (mismo shape "tipo concreto lleva más que el trait"
      que `TrainedGeoXGBoost::predict_spatial`; `fit` devuelve el concreto,
      `train_regress` lo envuelve). Hiperparámetros de kernel fijos (opt por
      marginal-likelihood = extensión futura documentada). Solver Cholesky +
      forward/back-substitution hand-rolled (convención per-módulo). `Learner`
      regressor-only, registrado como `"gaussian_process"` (28 en registry).
      **Golden vs scikit-learn 1.8.0** `GaussianProcessRegressor(RBF, alpha,
      optimizer=None)`: media Y std en test points match a 1e-6. 3 unit tests
      + contract autotest (7×28) + doctest. Binding Python en
      `smelt-py/src/learners/misc.rs` (mirror de QuantileForest: `predict_std`/
      `predict_with_std` inherentes, `declare_params!`, sin save/load porque
      GP no es serializable); verificado con `maturin develop` + script Python
      (golden sklearn a 1e-6 vía API). README actualizado (34 supervised, fila
      en tabla, 28 registry).
    - [x] **CoxPH** (2026-07-21) — `CoxPH`/`TrainedCoxPH` en
      `src/survival/cox.rs`. Regresión de Cox semi-paramétrica: partial
      likelihood (Cox 1972) maximizada por Newton-Raphson (log-PL globalmente
      cóncava → converge desde β=0; con step-halving de resguardo), empates
      por aproximación de Breslow, baseline cumulative hazard de Breslow →
      curvas de supervivencia por individuo en el mismo `SurvivalPrediction`
      que produce RSF. Features centradas (β invariante; baseline = individuo
      medio, = R `basehaz(centered=TRUE)`), penalización L2 opcional (Cox
      ridge, default 0). Solver Gauss con pivoteo hand-rolled (convención
      per-módulo). API standalone `fit(features, events)` como RSF (survival
      no encaja en `Learner`'s `(X,y)`), no registrado. **Golden vs R
      `survival::coxph(ties="breslow")` 3.5.8**: coeficientes, log-PL y
      baseline hazard match a 1e-6 sobre dataset fijo de 20×2. 6 unit tests
      (golden R, score≈0 en el óptimo, ranking C-index=1 con l2, orden de
      curvas por riesgo, rechazo all-censored/shape, shrinkage por l2).
      Rust-only por consistencia: RSF tampoco está bindeado en Python
      (bindear solo Cox sería inconsistente).
    - [x] **Dependencias entre parámetros en `ParamSet`** (2026-07-21) —
      `Condition` (`Equals`/`In`) + `Dependency{child,parent,cond}` en
      `src/tuning/mod.rs` (constructores `Dependency::equals`/`in_values`).
      Un hijo solo llega a la factory cuando su padre satisface la condición;
      `prune_inactive` los poda a fixpoint (cadenas hijo-de-hijo colapsan de
      una), `validate_dependencies` rechaza padre/hijo ausente del espacio
      (la misconfig exacta del M-5) y ciclos. Cableado en `GridSearch`
      (`cartesian_product_with_deps`: poda + dedup de combos que colapsan —
      la 6→4 del M-5 estructural) y `RandomSearch` (poda tras muestrear, sin
      tocar el stream RNG → reproducibilidad intacta), ambos con
      `with_dependency`. Bayesian/Hyperband fuera de scope (usan surrogate/
      successive-halving; el guard puntual del M-5 en `smelt-py/src/tuning.rs`
      ya cubre el único caso que exponen). Es la forma general del guard
      one-off que cerró el M-5. 8 unit tests + 1 integración end-to-end
      (con-vs-sin dependencia: 6 trials con waste bit-idéntico → 4 distintos).
    - [x] **Repeated CV / bootstrap / LOO explícito** (2026-07-21) —
      `RepeatedCV`/`LeaveOneOut`/`Bootstrap` en `src/resample/mod.rs`, sobre
      el trait `Resample` existente (mismo lugar que `CrossValidation`/
      `Holdout`, los otros clásicos). `RepeatedCV` reusa `CrossValidation`
      con seed por-repeat derivada vía `wrapping_add` (partición realmente
      distinta cada vez, reproducible); `LeaveOneOut` determinista sin seed;
      `Bootstrap` muestrea train con reemplazo (tamaño original, con
      duplicados) y usa el out-of-bag como test, saltando draws con OOB
      vacío (prob. no despreciable a `n` chico — ~25% en `n=2`) con un solo
      stream RNG continuo para reproducibilidad. Componen con `benchmark::
      resample_*` sin tocar el loop (el `features.select` ya maneja índices
      duplicados del bootstrap). 12 tests unitarios + 1 integración
      end-to-end con learner real. Bindings Python (`smelt-py/src/resample.rs`
      + registro en `lib.rs`/`__init__.py`): `RepeatedCV`/`LeaveOneOut`/
      `Bootstrap` con el mismo método `splits(n_samples)` que los otros
      resamplers; verificado con `maturin develop --release` + script Python.

## Infraestructura pendiente

- [x] **Python bindings (PyO3)** — Crate separado `smelt-py`, publicado en PyPI como `smelt-ml` (v0.7.0), 30+ learners + preprocessing + tuning + stats. ✅ (bindings iniciales 2026, cierre de paridad de learners 2026-07-03)
- [ ] **Paper JOSS** — Journal of Open Source Software. Requiere tracción.
- [x] **Cargo doc completo** — `#![warn(missing_docs)]` activo en `src/lib.rs`; las 330 advertencias cerradas. ✅ (2026-07-03, ver `docs/missing_docs_2026-07-03.md`)
- [ ] **Benchmarks OpenML** — Evaluar en 30+ datasets estándar de OpenML.
- [x] **CI/CD** — GitHub Actions: `ci.yml` (cargo test + parquet + clippy -D warnings + smoke pytest de smelt-py, 2026-07-10) y `release.yml` (wheels multiplataforma + sdist + publish a PyPI, con gate de tests en el ref del tag desde 2026-07-17). ✅

---

## Ya completado ✅

### Learners supervisados
- [x] Decision Tree (CART)
- [x] K-Nearest Neighbors
- [x] Linear Regression (OLS)
- [x] Logistic Regression (auto-scaling)
- [x] Random Forest (parallel, rayon)
- [x] Gradient Boosting (MSE/log-loss/softmax)
- [x] Extra Trees (random thresholds)
- [x] XGBoost (Newton, histogram, NaN, exact greedy, early stopping, parallel)
- [x] Geographical-XGBoost (Grekousis 2025, bi-square kernel, adaptive alpha)
- [x] Oblique Tree (sparse projections)
- [x] Oblique Forest / SPORF (Tomita 2020, parallel)
- [x] Gaussian Naive Bayes
- [x] Ridge Regression (L2, closed form)
- [x] Lasso Regression (L1, coordinate descent)
- [x] Elastic Net (L1+L2)
- [x] AdaBoost (SAMME)
- [x] Linear SVM (SGD + hinge loss)
- [x] Stacking / Super Learner (OOF meta-ensemble)
- [x] Quantile GB (pinball loss)
- [x] EBM (Explainable Boosting Machine, cyclic GAM)
- [x] Bagging (generic wrapper)

### Unsupervised
- [x] K-Means (Lloyd's + silhouette)
- [x] DBSCAN (density-based, noise detection)

### Causal Inference
- [x] Causal Forest (honest splitting, CATE, ATE, CIs)

### Métricas
- [x] Accuracy, Precision, Recall, F1 Score (macro)
- [x] Log Loss, AUC-ROC
- [x] RMSE, MAE, R-squared, MAPE
- [x] Silhouette Score

### Preprocessing / Feature Engineering
- [x] StandardScaler, MinMaxScaler
- [x] Imputer (mean, median, constant)
- [x] OneHotEncoder, LabelEncoder
- [x] SMOTE (synthetic minority oversampling)
- [x] PCA (power iteration)
- [x] FilterSelector (Variance, Correlation, ANOVA-F, Information Gain, Mutual Info)
- [x] RFE (Recursive Feature Elimination)
- [x] Pipeline (chains transformers + learner, fit_supervised)

### Tuning
- [x] GridSearch
- [x] RandomSearch (Uniform, LogUniform, Choice)
- [x] BayesianOptimizer (TPE)
- [x] Hyperband (successive halving)

### Resampling
- [x] CrossValidation (K-fold)
- [x] Holdout
- [x] SpatialBlockCV
- [x] SpatialBufferCV

### Advanced
- [x] Conformal Prediction (regression + classification)
- [x] Permutation Feature Importance
- [x] Benchmark Design (multi-learner × multi-task)
- [x] CSV Data Loading (auto label encoding)
- [x] Model Serialization (JSON)
- [x] Input Validation (dimension check, NaN detection)

---

## Fórmulas clave para implementación

### LightGBM GOSS
```
1. Sort instances by |g_i|
2. Keep top a% (large gradients) -> A
3. Randomly sample b% from rest -> B
4. Amplify B's gradients by (1-a)/b
5. Train tree on A ∪ B
```

### CatBoost Ordered Target Statistics
```
x_{σ(p),k} = (Σ_{σ(j)<σ(p), x_{j,k}=v} y_{σ(j)} + a·prior) / (count + a)
```

### RSF Log-Rank Split
```
L = [Σ_j (d_{j,1} - Y_{j,1}·d_j/Y_j)]² / Var
```

### Isolation Forest Score
```
s(x,n) = 2^{-E[h(x)]/c(n)}, c(n) = 2·H(n-1) - 2·(n-1)/n
```

### Hoeffding Bound
```
Split when: G_a - G_b > √(R²·ln(1/δ)/(2n))
```

### Classifier Chains (multi-label)
```
For j=1..q: Train C_j on X ∪ {l_{π(1)},...,l_{π(j-1)}}
```

### Quantile Regression Forest
```
w_i = (1/T)·Σ_t I(x_i ∈ L_t(x)) / |L_t(x)|
F⁻¹(τ) = inf{y : Σ_{y_i≤y} w_i ≥ τ}
```

---

*Documento generado a partir de análisis de 23,045 papers WOS. smelt-ml v0.6.1.*
