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
- [ ] **AutoTuner + nested CV** — envolver (learner + tuner + resampling)
      como un `Learner`, haciendo trivial el CV anidado (reporte honesto de
      performance tuneada — rigor publicable). ~200 líneas sobre los tuners
      existentes.
- [ ] **Calibración de probabilidades + threshold tuning** — Platt/isotonic
      como wrapper (mismo patrón factory) + búsqueda de umbral óptimo por
      costo/métrica; completa la historia que `CostSensitiveClassifier`
      empezó. ~300 líneas.
- [ ] **Registry con properties + autotest de contrato** — metadata
      consultable por learner (¿NaN? ¿categóricas? ¿proba? ¿pesos?) y un
      harness que verifica el contrato de cada learner registrado
      automáticamente (la clase de hallazgos que las 5 auditorías pescaron a
      mano). ~250 líneas, alto retorno en mantenibilidad.
- [ ] **Menores / futuro**: terminators componibles (tiempo/estancamiento/
      objetivo), dependencias entre parámetros en `ParamSet` (habría
      prevenido el M-5 de la 5ª auditoría), tuning multi-objetivo (Pareto),
      repeated CV / bootstrap / LOO explícito, kernel SVM, GP standalone con
      `se`, CoxPH, plotting (ROC/critical-difference sobre el benchmark ya
      existente), Pipeline como DAG. Deep learning (mlr3torch) sigue
      **descartado deliberadamente** (sin autodiff en el workspace).

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
