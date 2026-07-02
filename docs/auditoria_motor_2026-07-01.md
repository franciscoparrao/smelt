# Auditoría del Motor — Smelt

**Fecha**: 2026-07-01
**Reviewer**: Claude Code Reviewer (4 revisores paralelos: motores boosting, arquitectura/API, bindings Python, módulos estadísticos)
**Scope**: todo `src/` (~18k líneas), `smelt-py/src/lib.rs` (1858 líneas), configuración, tests, docs
**Estado de tests al momento de la auditoría**: 255 unit + 17 integración + 61 doctests, todos verdes. Clippy: solo warnings menores + código muerto.

## Resumen

- **Critical: 6 | High: 16 | Medium: ~25 | Low: ~20**
- Calidad general: **Buena base, con bugs de correctness serios concentrados en los módulos estadísticamente sofisticados**
- Deuda técnica estimada: **Media** (duplicación estructural alta, código muerto acotado, docs desactualizadas)
- Patrón dominante: los módulos core (pipeline, scalers, encoders, regresiones, stacking OOF, bagging, KNN, NB) están metodológicamente correctos; los problemas se concentran donde **no hay tests con valores de referencia** (conformal, causal, SHAP, isolation forest, survival).

---

## Hallazgos CRITICAL

### [CRITICAL-1] Binning de histogramas mata features binarias
- **Archivo**: `src/learner/histogram.rs:55`
- **Dimensión**: Correctness
- **Descripción**: Cuando `n_unique <= n_bins`, el último boundary generado iguala al máximo y NO se agrega el boundary INFINITY → los dos valores únicos más altos de cada feature caen al mismo bin. Caso extremo: feature binaria {0,1} → 1 solo bin → el loop de splits queda vacío → **la feature nunca puede usarse como split**.
- **Alcance**: los 3 motores comparten `HistBins`. XGBoost lo enmascara solo con n≤256 (exact greedy); LightGBM y CatBoost siempre usan histograma. CatBoost con 64 bins hardcoded: cualquier feature con ≤64 valores únicos pierde su top-2.
- **Verificado empíricamente**: target determinado por una feature binaria (n=600) → los 3 motores predicen la media (RMSE ~4.7-4.9); con el fix los 3 pasan (RMSE<1).
- **Impacto colateral**: puede afectar benchmarks del paper (King County tiene `waterfront` binaria y `view`/`condition` de baja cardinalidad).
- **Fix** (1 carácter, validado): en línea 55 cambiar `< vals[n_unique - 1]` por `<= vals[n_unique - 1]`. Agregar test de regresión con feature binaria para los 3 motores.

### [CRITICAL-2] El GIL nunca se libera en smelt-py
- **Archivo**: `smelt-py/src/lib.rs` (todos los `fit`, `select_bandwidth`:987, `optimize`:1579, shap, permutation)
- **Dimensión**: Bindings / UX
- **Descripción**: 0 usos de `py.allow_threads`. Entrenamientos rayon-paralelos de minutos (p.ej. LOO bandwidth) retienen el GIL: Jupyter congelado, sin progress bars, Ctrl+C no interrumpe.
- **Fix**: extraer datos bajo GIL, luego `py.allow_threads(|| learner.train_...(&task))`. Los datos ya son `Array2` owned (`Send`) — cambio mecánico.

### [CRITICAL-3] El causal forest "honest" no es honest
- **Archivo**: `src/causal/mod.rs:311-322, 184-199`
- **Descripción**: `populate_leaf_tau()` es un no-op; los τ de hojas se estiman con los mismos índices que construyen el árbol y el sample honest se descarta. Sesgo adaptativo clásico; la doc promete Athey & Imbens / Wager & Athey y el código no lo hace.
- **Fix**: recomputar τ de cada hoja con `est_idx` tras construir el árbol; agregar por muestra solo sobre árboles donde i ∉ subsample.

### [CRITICAL-4] CIs del causal forest estadísticamente inválidos
- **Archivo**: `src/causal/mod.rs:230-255`
- **Descripción**: `se = sqrt(var_between_trees / B)` → tiende a 0 al subir `n_estimators`; no estima la varianza muestral. Los CI 95% son espurios. Igual para el ATE.
- **Fix**: infinitesimal jackknife / bootstrap of little bags (Wager & Athey 2018 §4), o renombrar/documentar que no es un SE válido.

### [CRITICAL-5] "TreeSHAP" no es SHAP
- **Archivo**: `src/importance/shap.rs:39-48, 105-141`
- **Descripción**: es una diferencia de contribución marginal de una sola feature con dos referencias distintas, sin coaliciones. No cumple eficiencia (`pred ≈ base + Σ shap`) salvo para modelos aditivos, pese a que la doc lo promete. Tampoco es path-dependent TreeSHAP.
- **Fix mínimo**: renombrar a "marginal contribution importance" y corregir docs. Fix correcto: permutation-SHAP o KernelSHAP. Agregar test de eficiencia (habría detectado esto).

### [CRITICAL-6] Isolation Forest: fórmula c(n) incorrecta
- **Archivo**: `src/cluster/isolation_forest.rs:251, 226-228`
- **Descripción**: falta un paréntesis: `2.0 * (n-1.0).ln() + γ` en vez de `2.0 * ((n-1.0).ln() + γ) - 2(n-1)/n`. Con n=2 da −0.42 (negativo); n=256 da 9.67 vs 10.24. Todos los anomaly scores sesgados vs Liu et al. 2008/sklearn. Además `path_length` trunca a `usize`, perdiendo el ajuste fraccional.
- **Fix**: corregir fórmula + `f64` de punta a punta + golden test vs sklearn.

---

## Hallazgos HIGH

### Motores de boosting
1. **GOSS descarta los pesos de amplificación** (`lightgbm.rs:866-888, 937-956, 1002-1019`): los call sites hacen `let (selected, _weights)` y pasan `1.0` → sumas de gradiente sesgadas. Es LA corrección central del paper GOSS. Fix: `goss_sample` debe devolver pesos indexados por sample-id y pasarlos.
2. **El "leaf-wise" anunciado no existe** (`lightgbm.rs:503-613`): el builder real es DFS con truncado por contador — `num_leaves` corta según orden de recorrido, no por gain. El builder leaf-wise real (`build_leaf_wise_tree`:368) existe pero nunca se llama (fósil a medio terminar, líneas 235-501). Fix: terminarlo/conectarlo o corregir docs y borrar el bloque muerto.
3. **`sample_weight` ignorado silenciosamente en clasificación** (`xgboost.rs:992-998`). Fix mínimo: `Err(InvalidParameter)` en `train_classif` si hay pesos.
4. **Early stopping monitorea el train loss** (`xgboost.rs:948-980, 1034-1055, 1107-1125`): prácticamente nunca dispara; no previene overfitting. Fix: validation set o `validation_fraction` interna.

### GeoXGBoost (relevante para el paper con Grekousis)
5. **`predict()` decide por conteo de filas** (`geo_xgboost.rs:377-418`): si `nrows == n_train` asume que es el training set y aplica modelos locales por posición → dataset nuevo del mismo tamaño recibe modelos locales incorrectos, silenciosamente. Fix: eliminar heurística; `predict` = global only, predicción local siempre vía `predict_spatial`; exponer `fitted_values()`.
6. **Alpha adaptativo compara errores no comparables** (`geo_xgboost.rs:556-576`): `e_local` es OOB pero `e_global` es in-sample (XGBoost depth 6 × 100 árboles casi interpola → e_global≈0) → α degenera hacia el global. Fix: error OOB/CV para ambos lados (coherente con Grekousis Eq. 19-20); considerar α = e_global/(e_local+e_global) en vez del salto discontinuo a 1.
7. **Criterio LOO promedia sobre subconjuntos distintos por candidato** (`geo_xgboost.rs:180-221`): puntos con vecindario <3 se omiten → scores no comparables entre bandwidths (MEDIUM-HIGH).

### Arquitectura core
8. **Serialización no cubre los buques insignia** (`src/serialize.rs:34-42`): `SerializableModel` tiene 7 variantes; TrainedXGBoost/LightGBM/CatBoost/ExtraTrees/AdaBoost/LinearSVM/GaussianNB/Regularized derivan Serialize pero no se pueden guardar. Sin versionado del formato. Fix inmediato: agregar variantes + wrapper `{format_version, smelt_version, model}`. Fix estructural: `typetag`.
9. **El trait `Learner` es un tag de runtime**: todo learner debe implementar `train_classif` Y `train_regress`; 10+ learners devuelven `Err` en runtime (`Ridge::train_classif()` compila y falla). Esto genera ~8 pares de funciones espejo (~400 líneas duplicadas) en benchmark/tuning. Fix (pre-2.0): traits separados `ClassifLearner`/`RegressLearner` o genérico sobre Task; de paso `train(&self)` en vez de `&mut self` (habilita paralelizar folds).
10. **Sin CV estratificado ni group-CV** (`resample/mod.rs`): para los dominios objetivo (landslides, prospectivity — clases desbalanceadas) el K-fold plano puede dejar folds sin la clase minoritaria. Conectado: **stacking panica** si un fold pierde la clase máxima (`stacking.rs:137-166`, index out of bounds).
11. **`SmeltError::Other(String)` usado 49 veces** vs 6 tipadas — los usuarios no pueden hacer match. Fix: `UnsupportedTask`, `IncompatiblePrediction`, `NumericalError`.

### Módulos estadísticos
12. **ADASYN interpola hacia cualquier punto de la clase, no hacia k-NN** (`adasyn.rs:152-172`): genera sintéticos que cruzan regiones de la clase mayoritaria. Fix: k-NN intra-clase (como ya hace smote.rs).
13. **DES/KNORA-E evalúa competencia sobre el training set** (`des.rs:161-190`): modelos que sobreajustan son "competentes" en todo vecindario → selección vacua. Fix: DSEL hold-out interno.
14. **EBM multiclase produce basura silenciosamente** (`ebm.rs:179-199`): residuales sin sentido para n_classes>2 y predict colapsa a binario. Fix: `Err` hasta implementar OvR.
15. **RSF no puede predecir datos nuevos y su C-index es in-sample** (`survival/mod.rs:373-452`): solo `fit_predict`, bosque descartado, sin OOB. Fix: `fit() -> TrainedRSF` + predicciones OOB.

### Bindings Python
16. **Labels negativos → wrap de usize** (`smelt-py/lib.rs:39` y 3 sitios más): `y=[-1,1]` (convención SVM) → 18446744073709551615. Fix: validar y `PyValueError`. Además `unwrap()` sobre input de usuario en `build_param_space` (1519-1529) → PanicException críptica.

---

## Hallazgos MEDIUM (selección)

| # | Hallazgo | Archivo | Fix |
|---|----------|---------|-----|
| M1 | Cuantil conformal clampado al máximo cuando n < 1/α−1 → cobertura 1−α se pierde silenciosamente; α≥1 → underflow panic | `conformal/mod.rs:96, cqr.rs:80` | validar α y n; intervalo infinito o Err |
| M2 | CatBoost: NaN excluidos del gain pero enrutados a la izquierda en las hojas — estadísticas inconsistentes | `catboost.rs:226-381` | acumular nan_g/nan_h como xgboost.rs |
| M3 | CatBoost: categorías no vistas en predicción entran crudas a thresholds en [0,1] | `catboost.rs:425-432` | fallback al prior |
| M4 | CatBoost: target statistics multiclase promedia el índice de clase (orden arbitrario) | `catboost.rs:642-651` | TS one-vs-all o documentar |
| M5 | CatBoost: acepta splits con gain ≤ 0 / features constantes → árboles degenerados | `catboost.rs:239-298` | guard `gain > 0` |
| M6 | LightGBM: lambda=0 default + leaf weight sin clamp → riesgo de divergencia (mismo patrón del incidente CatBoost 2026-04-20) | `lightgbm.rs:71-88, 706-716` | default λ>0 o clamp |
| M7 | GBM clásico sin paso Newton en hojas para log-loss → probabilidades mal calibradas | `tree/gradient_boosting.rs:317-419` | cociente Newton por hoja |
| M8 | `with_truth_classif` sobre Prediction::Regression la devuelve intacta silenciosamente | `prediction/mod.rs:52-65` | Result |
| M9 | Panics en API pública: stats.rs asserts (100,188,245,352), spatial.rs assert_eq (38,131), CrossValidation::new(0) div-by-zero, tuning select_best unwrap, geo_xgboost partial_cmp con NaN (351) | varios | validar y devolver Result |
| M10 | `ParamSet = HashMap<String, f64>` stringly-typed, solo f64, typo → panic | `tuning/mod.rs:16` | enum ParamValue + Result |
| M11 | Relief: normalización del término negativo incorrecta vs RReliefF | `filter.rs:679-757` | normalizar por n_total−n_dc |
| M12 | PCA: comentario "1/n matching sklearn" falso (sklearn usa 1/(n−1)); power iteration desde vector constante sin re-ortogonalización | `pca.rs:60-114` | init aleatoria + re-ortogonalizar |
| M13 | Filter/RFE: `fit()` no supervisado hace fallback silencioso a varianza / primeras n features | `filter.rs:212-236, rfe.rs:96` | Err o documentar |
| M14 | QRF por pooling, no Meinshausen (pese a la cita) | `quantile_forest.rs:232-252` | pesos 1/‖hoja‖ o ajustar cita |
| M15 | Hoeffding: gain aproximado degenerado + bits/nats inconsistentes en la cota | `hoeffding.rs:329-360, 157` | estimadores gaussianos por clase |
| M16 | Quantile: tau=0 → underflow panic; sin line search por hoja | `quantile.rs:127` | validar 0<q<1 |
| M17 | `Clone for RFE` instala factory que hace `panic!` | `rfe.rs:78-89` | `Arc<dyn Fn>` |
| M18 | Smote/Adasyn no son Transformer → no entran al Pipeline → invita al leakage pre-split | `preprocess/` | adaptador task-level |
| M19 | Python: dtype estricto f64, extracción de y elemento a elemento, heurística int→classif sorprendente, PyRuntimeError para todo | `smelt-py/lib.rs` | PyArrayLike + validaciones |
| M20 | Versionado: 3 fuentes de verdad para 0.4.6; core 1.3.0 sin declaración de compatibilidad (`path` sin `version`) | pyproject/Cargo | single-source + `version = "1.3"` |
| M21 | Filtro `Filter` trait público sin constructor custom → extensibilidad rota | `filter.rs:13-59` | `FilterSelector::custom(Arc<dyn Filter>)` |
| M22 | Resample trait sin task → imposible estratificar, asserts en runtime | `resample/mod.rs:12` | `splits(&self, task) -> Result` |

---

## Performance

- `TrainedLightGBM::predict` y CatBoost multiclase predict son 100% seriales (XGBoost sí paraleliza). Trivial con rayon.
- Loops de actualización de predicciones en train seriales en los 3 motores — domina el wall-time con n grande.
- `HistBins::build`: serial por feature + búsqueda lineal O(n_bins) por muestra → `partition_point` + par_iter.
- `build_recursive` de LightGBM aloca `Vec left/right` por nodo (vs particionado in-place con swap de xgboost.rs — patrón ya existente en el repo).
- Grid search y benchmark: folds/combinaciones seriales ("parallel by default" no se cumple en el nivel que importa). Bloqueado por `&mut dyn Learner` → se destraba con el rediseño del trait.
- Histogramas en f64 (oficiales usan f32) — mitad del throughput de cache.
- Oblique/AdaBoost/Lasso: recomputación O(n²) por split/candidato — actualización incremental.

## Duplicación (colapsable en ~1000+ líneas)

- `sigmoid` ×6, `softmax` ×4 → `learner::math`.
- Scanner de splits sobre histograma copy-pasteado ×5 (~350 líneas) → `scan_split()` compartido (de paso unifica la semántica L1, hoy inconsistente: `_alpha` aceptado e ignorado en `hist_pool.rs:66`).
- train_binary/train_multiclass ~90% idéntico en los 4 motores → driver genérico.
- permutation_importance classif/regress 95% copy-paste.
- ~8 pares espejo classif/regress en benchmark/tuning (raíz: diseño del trait).
- smelt-py: 11 clases × boilerplate fit/predict (~700 líneas) → macro `define_learner!`.

## Deuda técnica

- **Código muerto**: bloque leaf-wise de lightgbm.rs (235-501, grafo muerto completo con comentario "Let me use a simpler strategy"), `find_best_histogram` en xgboost.rs:528 (75 líneas superseded).
- **README desactualizado**: dice `smelt-ml = "0.6"` y "21 learners" (real: 1.3.0, 27 learners).
- **CLAUDE.md desactualizado**: roadmap declara Phase 2 pendiente cuando hay 27 learners, tuning completo, conformal, causal, survival.
- **TODOs**: 0 (limpio).
- **Dependencias**: al día en lo esencial (rand 0.9 vs 0.10 disponible, menor).
- **Tests**: amplios pero casi todo smoke tests. Faltan golden tests vs sklearn/R para conformal (cobertura empírica), SHAP (eficiencia), isolation forest, PCA, causal (DGP sintético con τ conocido), survival (C-index vs R). Cada uno de los 4 CRITICAL estadísticos habría sido detectado por su golden test.

## Gaps vs "el mejor motor ML en Rust" (mlr3/sklearn/linfa/oficiales)

1. **Features categóricas y missing values en Task** — `Array2<f64>` denso sin NaN; CatBoost sin categorías nativas es irónico.
2. **CV estratificado/agrupado** — el gap más citable por un revisor dado el dominio.
3. **Early stopping con validación** en los 3 motores (hoy ninguno lo tiene de verdad).
4. **Monotone constraints, objetivos custom, scale_pos_weight** — estándar en XGBoost oficial.
5. **Persistencia completa** (core + Python) con versionado de formato.
6. **Data loading**: solo CSV en memoria con doble copia; Parquet/Arrow (polars como feature) es lo esperable.
7. **Model registry** (`learner_from_id("xgboost")`) para experimentos data-driven; los ids no siguen la convención mlr3 declarada.
8. **Medidas faltantes**: balanced accuracy, Cohen's kappa, MCC, Brier.
9. **Python**: 14 learners no expuestos (~54%), sin get_params/set_params, sin .pyi, GridSearch/RandomSearch reimplementados en Python puro (divergencia).
10. **Sparse data** y pesos de muestra unificados a nivel de Task.

## Plan priorizado

### Fase 0 — Fixes quirúrgicos de correctness (días)
1. `histogram.rs:55` (1 carácter) + tests de regresión binarios ×3 motores. **Re-correr los benchmarks del paper después** (waterfront/view son de baja cardinalidad).
2. GeoXGBoost: eliminar heurística nrows en `predict`; arreglar comparación OOB-vs-in-sample del alpha adaptativo (relevante antes de nuevos resultados con Grekousis).
3. Isolation forest c(n); conformal: validar α/n.
4. smelt-py: `allow_threads` + validar labels negativos + quitar unwraps.

### Fase 1 — Honestidad de lo anunciado (1-2 semanas)
5. Causal forest honest de verdad + jackknife (o degradar docs/renombrar SE).
6. SHAP → renombrar o implementar permutation-SHAP con test de eficiencia.
7. GOSS con pesos; leaf-wise real o degradar docs; early stopping con validación; sample_weight en clasificación (o Err).
8. Golden tests vs sklearn/R para todos los módulos estadísticos.
9. Serialización: variantes faltantes + versionado de formato; exponer a Python.

### Fase 2 — Rediseño de API (pre-2.0)
10. Traits `ClassifLearner`/`RegressLearner` (o genérico sobre Task) con `&self` → colapsa ~400 líneas espejo y habilita paralelizar folds/grid.
11. `Resample::splits(task) -> Result` + StratifiedCV + GroupCV.
12. Errores tipados; eliminar panics de rutas públicas.
13. `learner::math` + `scan_split` compartidos; borrar código muerto.

### Fase 3 — Paridad competitiva
14. Categóricas + NaN en Task y splits; early stopping/monotone constraints/objetivos custom.
15. Python: macro `define_learner!`, cerrar los 14 learners, get_params/stubs, dividir lib.rs.
16. Parquet/Arrow; model registry; medidas faltantes; f32 en histogramas; predict paralelo consistente.
17. Docs: README y CLAUDE.md al día; `#![warn(missing_docs)]`.
