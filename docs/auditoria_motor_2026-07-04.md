# Auditoría del Motor — Smelt (2ª auditoría completa)

**Fecha**: 2026-07-04
**Reviewer**: Claude Code (5 revisores paralelos: motores boosting/árboles, módulos nuevos post-07-01, módulos estadísticos, arquitectura/API, bindings Python)
**Scope**: todo `src/` (~23k líneas, 83 archivos), todo `smelt-py/` (reestructurado), capa Python pura (`python/smelt/`), config, tests, docs
**Referencia**: auditoría anterior `docs/auditoria_motor_2026-07-01.md` (6 CRITICAL / 16 HIGH / ~25 MEDIUM) + 4 fases de remediación (`docs/fase0..3_progreso.md`)
**Estado de tests**: 481 verdes (126 lib + 281 integración + 70 doctests + 4 parquet), 0 fallos. Clippy: 11 warnings menores.
**Método**: verificación línea por línea + confirmación empírica de los hallazgos graves (proyecto sonda compilado contra el crate para boosting/kriging; ejecución contra el `.so` real para smelt-py).

## Resumen

- **Nuevos: 2 CRITICAL | 11 HIGH | ~15 MEDIUM | ~25 LOW**
- **De la auditoría anterior**: los 6 CRITICAL y la mayoría de los HIGH de boosting quedaron genuinamente corregidos; pero **3 ítems declarados cerrados no lo están del todo** (CausalForest honest a medias, GIL solo en fit, panic de Stacking) y **el ítem "golden tests para todos los módulos estadísticos" desapareció silenciosamente de la tabla de Fase 1**. Siguen abiertos sin tocar: ADASYN, DES, EBM multiclase, RSF, PCA, Relief, M4-M7/M14/M16 de boosting, errores tipados (empeoró: 49→51 `Other`), ParamSet stringly-typed, paralelismo de folds/grid (0 rayon), versionado triple de smelt-py.
- **Patrón dominante de esta ronda**: (1) los bugs nuevos más graves están en **integración entre capas** (histograma compartido entre niveles del árbol, CV que pierde metadatos, capa Python pura que reimplementa mal lo que el core hace bien), no en fórmulas; (2) los **módulos nuevos jamás auditados** (KrigingHybrid, meta-learners) traen HIGHs de estreno; (3) se repite la causa raíz de 2026-07-01: donde no hay test con valores de referencia, hay bug.

---

## Hallazgos CRITICAL (nuevos)

### [CRITICAL-1] XGBoost: histograma stale en la resta parent−child — subárboles enteros colapsan en silencio
- **Archivo**: `src/learner/xgboost.rs:537-543` + `642-651`
- **Descripción**: `build_hist_sub` solo almacena su histograma en `pool[depth]` si pasa el guard de entrada; si el hijo *menor* retorna hoja temprana (`n <= 1`) no escanea ni almacena nada, pero el padre igual ejecuta `subtract_in_place(depth, depth+1)` contra lo que hubiera en ese nivel (ceros la primera vez, el histograma de un primo en visitas posteriores). El hermano mayor se construye con `hist_ready=true` sobre masa fantasma: gains inventados (que además contaminan `feature_importances`, sumadas antes del guard) y típicamente degenera a hoja.
- **Confirmado empíricamente** (sonda, config default, 1 árbol): n=600 con un outlier que el root aísla como hijo de 1 muestra → los 599 puntos restantes, con un escalón perfectamente ajustable a depth 3, quedan en **una sola hoja** (RMSE 5.00 vs 0.29 del control sin outlier).
- **Condición de disparo**: hijo menor con n=1 en profundidad no terminal — trivial en regresión con `min_child_weight=1.0` y cualquier outlier; también alcanzable con sample weights (GeoXGBoost). **Solo el modo histograma (n > 256)**: el modelo global de GeoXGBoost en King County (n_train=800) y el LOO con bw>256 corren por esta ruta; los modelos locales (n≈bw≤256) van por exact-greedy y no.
- **Fix**: que el padre sepa si el hijo menor escaneó (calcular `smaller_n` antes de recursar y usar `hist_ready=false` para el mayor cuando el menor retorna temprano, o almacenar el histograma trivial del hijo de 1 muestra). Test de regresión con el escenario de la sonda. **Re-correr los benchmarks del paper después del fix.**

### [CRITICAL-2] Capa Python: GridSearch/RandomSearch siempre maximizan — seleccionan los PEORES hiperparámetros con métricas de error
- **Archivo**: `python/smelt/tuning.py:76, 108-110`
- **Descripción**: `if mean_score > best_score` incondicional. El core Rust decide dirección vía `Measure::maximize()`; la capa Python (publicada en PyPI) no. **Verificado empíricamente**: `GridSearch(RandomForest, {'n_estimators':[1,50]}, metric=rmse_score)` elige `n_estimators=1` (RMSE 0.615) sobre 50 (RMSE 0.339).
- **Impacto**: cualquier usuario que tunee con rmse/mae/logloss/brier obtiene silenciosamente el peor modelo.
- **Fix**: inferir dirección por métrica (o parámetro `maximize`), o exponer los tuners Rust y borrar la reimplementación.

---

## Hallazgos HIGH (nuevos)

### Motores / árboles
1. **`TrainedObliqueTree::predict` pasa `feature_names.len()` como `n_classes`** (`oblique.rs:352-357`): confirmado — 2 features/3 clases → panic OOB; 5 features/2 clases → probabilidades de largo 5. `ObliqueForest` no afectado. Expuesto en registry y smelt-py. Fix: almacenar `n_classes` como hace el forest.
2. **`LightGBM::subsample` se acepta y se ignora silenciosamente** (`lightgbm.rs:68, 90, 147-151`): builder documentado, cero usos en el entrenamiento. Mismo patrón del HIGH-3 original. Fix: implementar bagging de filas (documentando la exclusión con GOSS, como el oficial) o eliminar el builder.

### Módulos nuevos (primera auditoría)
3. **KrigingHybrid: `predict_spatial` falla siempre cuando los residuales del base son ~0** (`kriging_hybrid.rs:139-241`): variograma degenera a sill=0 → γ≡0 → matriz singular → `Err` en el 100% de las queries. Confirmado empíricamente. Mientras mejor el modelo base, más inutilizable el híbrido. Fix: si `sill − nugget < eps`, corrección 0 sin armar el sistema; regularizar diagonal.
4. **KrigingHybrid: coordenadas de entrenamiento duplicadas → sistema singular** (`kriging_hybrid.rs:263-274`): dos muestras en la misma ubicación (caso común: mismo sondaje) → filas idénticas → `Err` en cada query cercana. Confirmado con un solo par duplicado en n=12. Fix estándar: promediar residuales en ubicaciones coincidentes al entrenar y/o filtrar vecinos a distancia < eps.
5. **El "R-learner" implementado es el U-learner** (`meta_learners/r_learner.rs:19-35, 120-142`): regresión no ponderada sobre `Ỹ/T̃` con drop de filas — exactamente el U-learner que Nie & Wager muestran que rinde mal. El doc afirma que conserva "la propiedad quasi-oracle de remoción de sesgo": falso, el teorema es sobre la R-loss *ponderada por T̃²*. Con `residual_clip=1e-3` default el pseudo-target admite valores ×1000, y el clipping introduce sesgo de selección. Los tests no lo detectan (el fixture confundido solo exige `is_finite()`). Fix mínimo: renombrar/documentar como U-learner + clip ~0.05 + winsorizar; fix real: R-loss ponderada vía replicación de filas ∝ T̃² (sin tocar el trait).

### Estadísticos / arquitectura
6. **CausalForest: predicciones in-sample sin agregación OOB — residuo de CRITICAL-3 declarado cerrado** (`causal/mod.rs:214-220`): `populate_leaf_tau` honest sí existe, pero cada muestra vota con árboles donde ella misma está en `est_idx` → auto-influencia en CATE/ATE y SEs IJ contaminados. La auditoría prescribió explícitamente la agregación OOB y `fase1_progreso.md` declara el ítem completo. Fix: saltar `idx` cuando `in_bag[idx]` (el indicador ya existe).
7. **Wilcoxon: p-values inválidos en el caso de uso principal** (`stats.rs:91, 162-173`): doc promete "exact tables for n < 10" que no existen; siempre aproximación normal, sin corrección de continuidad ni de empates. Con 5 folds y todas las diferencias del mismo signo: p≈0.043 "significativo", cuando el test exacto da mínimo 0.0625 — **imposible ser significativo al 5% con 5 folds**. El propio doctest del módulo afirma esa inferencia inválida. Fix: enumeración exacta 2ⁿ para n≤12 + continuidad/empates para el resto; corregir doctest.
8. **El loop de CV destruye los metadatos categóricos del Task** (`benchmark.rs:49, 85`): los tasks de fold se reconstruyen con `::new`, que resetea `feature_types` a `Numeric` — **todo el trabajo de splits categóricos de Fase 3 queda desactivado exactamente en la ruta de CV/tuning/benchmark** que decide hiperparámetros y genera resultados de paper. También se pierden `feature_names`/`class_names`. Fix: constructor `from_parent(indices)` que propague metadatos.
9. **Stacking sigue expuesto al panic por fold sin la clase máxima** (`stacking.rs:144, 167-170`): usa `CrossValidation` plano (no el `StratifiedCV` que Fase 2 creó) e indexa `probs[j][c]` con ancho global → index out of bounds si un fold pierde la clase máxima. Era la motivación del HIGH-10 original; se arregló la infraestructura pero no el consumidor. Fix: StratifiedCV interno en `train_classif`.

### Bindings / capa Python
10. **`benchmark()` de Python clona los learners con hiperparámetros default** (`python/smelt/benchmark.py:88, 171-183`): `_get_params` itera `dir()` buscando atributos que los `#[pyclass]` no exponen → `{}` (verificado); `benchmark({"XGB": XGBoost(n_estimators=500)})` evalúa `XGBoost()` default en silencio. Fix trivial: usar el `learner.get_params()` que el item 15c ya agregó.
11. **`smelt.conformal` roto en PyPI** (`python/smelt/conformal.py:3`): importa `ConformalRegressor` de `smelt._smelt`, clase nunca registrada → `ImportError` (verificado). Módulo público muerto. Fix: exponer la clase o borrar el archivo.

---

## Verificación de hallazgos previos (consolidada)

### Corregidos de verdad (verificados en código, con tests)
CRITICAL-1 (binning binarias, `histogram.rs:96`), CRITICAL-4 (jackknife IJ real con tests de no-decaimiento), CRITICAL-5 (permutation-SHAP con tests de eficiencia), CRITICAL-6 (c(n) iForest + golden vs sklearn), HIGH-1 GOSS (pesos aplicados en los 3 call sites + test), HIGH-2 leaf-wise (arena real, código muerto eliminado), HIGH-3 sample_weight (implementado, no Err), HIGH-4 early stopping (eval-set en los 3 motores), HIGH-5 GeoXGB predict global-only (test bit-a-bit), HIGH-6 alpha OOB k-fold, HIGH-8 serialización (15 variantes + envelope versionado), M1 conformal (α validado, rank correcto, cobertura empírica testeada), M2/M3 CatBoost NaN/categorías no vistas, M9 (los panics enumerados; ver residuo abajo), M22 `Resample::splits → Result`, StratifiedCV/GroupCV reales, registry (22 ids + tests), sigmoid/softmax → `math.rs`, predict paralelo en los 3 motores, f32 CatBoost verificado sano, HIGH-16 labels negativos (en rutas de fit), unwraps de tuning.

### Declarados cerrados pero incompletos (HIGH automático por proceso)
| Ítem | Residuo | Evidencia |
|---|---|---|
| CRITICAL-3 causal honest | Falta agregación OOB in-sample (nuevo HIGH-6) | `causal/mod.rs:214-220` |
| CRITICAL-2 GIL | Solo fit/estimate/shap/BO liberan; predict/predict_spatial/conformal_predict/Smote/filtros/SpatialBufferCV retienen | `common.rs:135-171, 336-371`, `boosting.rs:507,718`, `preprocess.rs:82,148`, `feature_selection.rs:83-95` |
| HIGH-10 stacking | StratifiedCV existe pero Stacking no lo usa (nuevo HIGH-9) | `stacking.rs:144` |
| Golden tests (ítem 8 de Fase 1) | **Desapareció de la tabla de fase1_progreso.md sin mención**; solo se agregaron donde hubo fix | stats/measures/survival/PCA/ADASYN/DES/EBM siguen sin valores de referencia |
| M9 panics | `bootstrap_ci` panica con scores vacío / n_bootstrap=0 | `stats.rs:415-440` |

### Siguen abiertos (sin cambios desde 2026-07-01)
| ID | Tema | Evidencia actual |
|---|---|---|
| HIGH-7 | GeoXGB LOO omite puntos con vecindario <3; docstring nuevo además invierte el sentido del sesgo | `geo_xgboost.rs:194-237` |
| HIGH-11 | Errores tipados: `Other` subió 49→**51**; `IncompatiblePrediction`/`NumericalError` nunca creados; código nuevo (kriging, s_learner) perpetúa el patrón | `error.rs`, grep |
| HIGH-12 | ADASYN interpola hacia toda la clase, no k-NN | `adasyn.rs:158-171` |
| HIGH-13 | DES/KNORA-E competencia in-sample (comentario literal "same training set as validation") | `des.rs:165-184` |
| HIGH-14 | EBM multiclase basura silenciosa (sin Err) | `ebm.rs:122-129, 190-192` |
| HIGH-15 | RSF solo `fit_predict`, C-index in-sample, 0 tests unitarios | `survival/mod.rs:379-458` |
| M4-M7 | CatBoost TS multiclase por índice; gain≤0 aceptado; LightGBM λ=0 sin clamp; GBM sin paso Newton | `catboost.rs:820-829, 352-430`, `lightgbm.rs:85,688`, `gradient_boosting.rs:318-420` |
| M8 | `with_truth_classif` no-op (solo se documentó); `with_truth_causal` nuevo replica el anti-patrón | `prediction/mod.rs:72-124` |
| M10 | `ParamSet = HashMap<String, f64>` | `tuning/mod.rs:17` |
| M11-M13 | Relief mal normalizado; PCA init constante (ver MEDIUM nuevo con caso de fallo concreto); fallbacks silenciosos de Filter/RFE | `filter.rs:743`, `pca.rs:61,115`, `rfe.rs:98` |
| M14, M16 | QRF pooling (cita a Meinshausen intacta); QuantileGB τ=0 → **panic confirmado por sonda** (`attempt to subtract with overflow`) | `quantile_forest.rs:246`, `quantile.rs:129` |
| M17-M21 | Clone de RFE panica; Smote/Adasyn fuera del Pipeline; dtype/PyRuntimeError/heurística int→classif en Python (M19); versionado triple 0.4.6 + path dep sin `version` (M20); Filter no enchufable | sin cambios |
| Paralelismo | 0 usos de rayon en `tuning/` y `benchmark*.rs` — "parallel by default" de CLAUDE.md sigue falso en el nivel que importa | grep vacío |
| Duplicación | scanner de splits ×5 (con `_alpha` ignorado como síntoma vivo), train_binary/multiclass ×4, ~8 pares espejo classif/regress | `hist_pool.rs:73` et al. |

---

## Hallazgos MEDIUM (nuevos, selección)

| # | Hallazgo | Archivo | Fix |
|---|----------|---------|-----|
| N1 | L1 (`alpha`) afecta pesos de hoja pero no el gain en los 3 caminos de XGBoost; con monotone+alpha los dos caminos divergen | `xgboost.rs:461-466`, `hist_pool.rs:73,98` | soft-threshold en `split_gain` (como oficial) |
| N2 | AdaBoost SAMME: `lr` no multiplica `ln(K−1)`; con err==0 re-entrena el mismo stump n rondas | `adaboost.rs:234-241` | paréntesis + break en err 0 |
| N3 | `TrainedQuantileGB::predict` sin `check_n_features` (panic OOB) ni feature importance | `quantile.rs:93-110` | validar como el resto |
| N4 | Hoeffding: cota en bits, entropía en nats (M15 previo, sigue) → ε inflado ×1.44 | `hoeffding.rs:170 vs 318-346` | `.ln()` |
| N5 | Hoeffding: tie-break ε<0.01 permite splits con gain 0 → cadenas degeneradas sin cota (stack overflow eventual en streams largos con default `max_depth=None`) | `hoeffding.rs:173, 414, 430` | exigir `best_gain > 0` |
| N6 | ARF sin subespacio aleatorio de features — falta el componente "Random" del paper que cita; diversidad solo por Poisson(6) | `adaptive_rf.rs:366-397` | `max_features` por split + seed por árbol |
| N7 | ADWIN default `max_window=200` solo detecta saltos Δerror > ~0.34; el trade-off no está documentado | `adaptive_rf.rs:54-58` | default 1000-2000 o documentar |
| N8 | `oof_propensity` degrada a 0.5 silenciosamente si un fold de train tiene un solo brazo (R-learner desprotegido) | `cross_fit.rs:79` | Err como `oof_regression_by_arm` |
| N9 | Parquet: nulls en target string → clase `""` silenciosa (el caso numérico sí erra) | `parquet.rs:181` | Err simétrico |
| N10 | Macro precision/recall/F1 promedian solo clases "válidas" → clasificadores degenerados inflados (0.5 vs 0.25 sklearn) — exactamente el escenario landslides | `measure/mod.rs:162-252` | promediar sobre truth∪pred con 0 |
| N11 | C-index cuenta pares con tiempos empatados en ambas direcciones → sesgo hacia 0.5 con tiempos discretizados | `survival/mod.rs:80-93` | omitir empates (Harrell) |
| N12 | PCA: caso de fallo concreto — con 2 features anticorrelacionadas la init constante ES eigenvector (el menor), converge en 1 iteración al PC equivocado, en silencio | `pca.rs:115-135` | init aleatoria seedeada + re-ortogonalización |
| N13 | `bootstrap_ci` panica (scores vacío, n_bootstrap=0) — único sobreviviente de M9 | `stats.rs:415-440` | Result |
| N14 | Measures Python castean `y_pred` con `as usize`: −1.0 satura a 0 y cuenta como acierto | `measures.rs:13-75` | validar ≥0 y fract()==0 |
| N15 | `parse_coords` no valida NaN/inf → vecindarios de kriging/bandwidth corruptos en silencio (verificado) | `common.rs:178-211` | `is_finite()` |
| N16 | Heurística classif de `benchmark()` Python diverge de `fit()` (unique<20 vs dtype) → TypeError a mitad de loop | `benchmark.py:58` | unificar criterio |
| N17 | CV loop: versión 1.3.0 sin bump tras rupturas públicas (`Resample::splits`, stats → Result); prelude sin conformal/survival/multilabel/benchmark_design; EBM ausente del registry sin doc; README de nuevo desfasado (26 vs 28 learners, sin KrigingHybrid/ARF/causal meta-learners/Parquet) | `Cargo.toml:6`, `lib.rs:68-108`, `registry.rs`, `README.md` | bump/CHANGELOG + completar |

## Hallazgos LOW (nuevos, selección)

- Boosting: `HistBins::build` puede emitir 255 bins violando el contrato MAX_BINS=254 (`histogram.rs:89-98`); códigos categóricos ≥254 fusionados en train pero "unseen→right" en predict (`xgboost.rs:361-366`); QRF panics de borde (`min_samples_leaf=0`, `n_estimators=0`); log-odds inicial −∞ si p_pos=0.
- Streaming: ARF no re-seedea entre `train_classif` (no reproducible); HoeffdingTree ignora crecimiento de `n_classes` a mitad de stream, empates de gain no deterministas (HashMap), hojas nuevas predicen la última clase; `sample_poisson` degenera con λ≳745.
- Kriging/SMOTE: k=1 vecino asigna el residual completo sin decaimiento; jitter de duplicación ±0.01 en escala absoluta (enorme en features ~1e-6); SpatialSmote duplica la coordenada exacta (alimenta el HIGH de coords duplicadas) y el déficit de balance por `max_attempts` es silencioso.
- Estadísticos: Nemenyi fallback q=2.576 para k>10 subestima el CD; Friedman sin corrección de empates; MAPE divide por n total pese al comentario "skip zeros"; LogLoss/AUC panican con labels fuera de rango; Chains panican con lista vacía; iForest submuestrea con reemplazo y `max_samples` sin clamp; conformal no valida `nrows(cal) == len(targets)` (zip trunca); clip de propensity 1e-3 agresivo en DR/X-learner.
- API/Python: `CsrMatrix::row` panica sin doc `# Panics`; ParquetLoader sin `max_rows`; doc del trait Learner dice "classification learners" y ejemplifica ids mlr3 que nunca existieron; `SmeltError` sin `#[non_exhaustive]` con variante feature-gated; excepciones inconsistentes (PyRuntimeError vs PyValueError); `KrigingHybrid.__new__` no valida `variogram_model` (set_params sí); ExtraTrees/DecisionTree sin `predict_proba` en Python; factory de BO duplica el registry con aliases divergentes; sin `__repr__` ni `.pyi`.

---

## Performance (estado actual)

- **Sigue abierto del audit anterior**: loops de actualización de predicciones seriales en los 3 motores; `HistBins::build` serial con búsqueda lineal; folds/grid/benchmark 100% seriales (0 rayon; las fábricas `Send+Sync` por fold ya existen — es el gap más barato de cerrar); TreeBuilder O(n²) por feature/nodo (afecta DT/RF/GBM/QuantileGB/QRF/AdaBoost/Oblique); Lasso O(n·p²) por sweep.
- **Nuevo**: LightGBM leaf-wise re-escanea los candidatos de TODAS las hojas activas por iteración — O(L²·F·B) por árbol, ~15× trabajo redundante con `num_leaves=31` (fix: cachear mejor split por hoja / heap, como el oficial). Predict serial fuera de los 3 motores (ExtraTrees, GBM, DT, Oblique, QuantileGB, QRF, `predict_spatial` de GeoXGB/Kriging). KrigingHybrid: O(n²) en train, O(n log n)+O(k³) por query sin rayon. `Adwin::add` realoca prefijos en cada muestra ×2 detectores ×n_trees. SMOTE recomputa k-NN por sintético. `cross_fit` secuencial (K folds × 3 nuisance models independientes). CatBoost clona `features` completo en predict sin categóricas.

## Paridad Rust↔Python (resumen)

Sin exponer: **cluster/ completo** (KMeans/DBSCAN/iForest), **CausalForest**, **survival**, **multilabel/multioutput**, **serialize (no hay persistencia de modelos desde Python)**, **CsvLoader/ParquetLoader**, MinMaxScaler/Imputer/OneHot/LabelEncoder/Adasyn/PCA/Pipeline/FilterSelector, Holdout, GridSearch/RandomSearch/Hyperband Rust (reimplementados en Python con el CRITICAL-2), friedman/nemenyi/mcnemar, ConformalClassifier/CQR, API streaming (partial_fit/predict_one/n_drifts), mape/logloss como funciones, Pehe/AteBias, CsrMatrix.
Divergencias de semántica: tuning (CRITICAL-2), benchmark (HIGH-10), heurística classif (N16), aliases de BO.

## Estado de la cultura de tests

Los módulos que recibieron fix en Fase 0/1 ganaron tests de propiedad/referencia genuinos (conformal cobertura empírica, SHAP eficiencia, iForest golden vs sklearn, CausalForest ATE sintético, GOSS/leaf-wise/binning). Los que no se tocaron conservan la superficie sin referencia que causó los CRITICAL originales: `stats.rs` (9 tests cualitativos, cero valores vs scipy — un golden habría detectado el HIGH de Wilcoxon), `measure/` (0 tests en el módulo), `survival/` (0 unitarios), PCA (solo formas), ADASYN/DES/EBM/KMeans/DBSCAN (smoke). `tests/real_benchmark.rs` vs sklearn sigue `#[ignore]` y depende de un JSON manual. Módulos nuevos: kriging_hybrid la mejor suite del lote (pero le faltan justo los edge cases que fallan); meta-learners buenos fixtures RCT pero el fixture confundido solo exige `is_finite()`; hoeffding sigue siendo el de menor cobertura relativa a su complejidad.

---

## Plan priorizado

### Fase A — Correctness urgente (días) — **COMPLETADA 2026-07-04**
1. ✅ **CRITICAL-1** histograma stale de XGBoost — fix en `build_hist_sub`
   (predice de antemano si el hijo menor será hoja trivial y, si es así,
   escanea también el mayor en vez de restar sobre un nivel nunca
   almacenado) + test de regresión (`outlier_isolated_as_singleton_child_does_not_corrupt_sibling_histogram`).
   **Benchmarks del paper re-corridos** (stash del fix + rebuild x2,
   `demo_geoxgboost.py` sobre King County): XGBoost RMSE 0.332→0.329,
   GeoXGBoost RMSE 0.263→0.262 — diferencia dentro del ruido de RNG, el bug
   no llegó a corromper los números ya publicados en este dataset de 800
   filas. No hace falta corregir el paper.
2. ✅ **CRITICAL-2** dirección de optimización en `tuning.py` (`_resolve_maximize`,
   infiere por nombre de métrica, override explícito disponible) — verificado
   que ahora selecciona el menor RMSE en vez del mayor.
   ✅ HIGH `benchmark.py` usaba `_get_params` por `dir()`-scan (devolvía `{}`
   para todo `#[pyclass]`) → ahora usa `learner.get_params()`; verificado que
   dos instancias con distintos hiperparámetros ya no dan resultados idénticos.
   ✅ HIGH `smelt.conformal` (ImportError en PyPI, clase nunca registrada) →
   módulo muerto eliminado; la API real (`<learner>.conformal_predict(...)`)
   sigue intacta y es lo que ya usa `demo_geoxgboost.py`.
3. ✅ KrigingHybrid: variograma degenerado (sill<1e-9) → corrección 0 sin
   armar el sistema; neighbors en coordenadas (casi) coincidentes → agrupados
   y representados por su residual promedio antes de resolver. Dos tests de
   regresión nuevos (`predict_spatial_handles_degenerate_zero_variance_residuals`,
   `predict_spatial_handles_duplicate_training_coordinates`).
4. ✅ Validaciones quirúrgicas: `TrainedObliqueTree` guardaba `n_classes` como
   `feature_names.len()` → ahora almacena `n_classes` real (bug reproducido y
   corregido, con test); `QuantileGB` con τ∉(0,1) ahora `Err` en vez de panic
   por underflow, `TrainedQuantileGB::predict` valida `n_features`;
   `bootstrap_ci` ahora devuelve `Result` (rechaza `scores` vacío y
   `n_bootstrap=0`, antes paniqueaba) — 2 call sites migrados (`compare_models`,
   binding Python); `parse_coords` (smelt-py) rechaza NaN/inf; las 6 measures
   de clasificación en Python ya no truncan labels negativos a la clase 0 vía
   cast `as usize` saturante; Parquet con null en target string ya no crea una
   clase `""` fantasma (ahora `Err`, simétrico al caso numérico).

Verificado: 489 tests verdes (133 lib + 282 integración + 70 doctests + 4
parquet, +8 sobre la línea base de 481), 0 fallos, `smelt-py` recompilado y
probado end-to-end con cada fix.

### Fase B — Honestidad de lo anunciado, 2ª ronda — **COMPLETADA 2026-07-04**
5. ✅ **CausalForest OOB real** (cierra CRITICAL-3 de verdad esta vez): la
   agregación por punto ahora excluye todo árbol donde ese punto estuviera
   en la submuestra (train O est), no solo los que carecían de honestidad en
   la hoja — antes, un punto en `est_idx` de un árbol veía su propio outcome
   alimentar el τ̂ que ese mismo árbol le reportaba a él (auto-influencia).
   Test de regresión con outlier extremo (`oob_aggregation_excludes_own_outcome_from_own_estimate`):
   confirmado que sin el fix el efecto reportado se dispara a 47171 (arrastrado
   por su propio 1e6), con el fix vuelve a ~1.0 (el efecto común real).
   ✅ **R-learner real**: el "R-learner" era en realidad el U-learner de
   Künzel et al. (regresión no ponderada). Ahora aproxima la R-loss ponderada
   por T̃² real vía **replicación de filas** proporcional al peso normalizado
   (capada, sin tocar el trait `Learner`) — compone con cualquier learner base
   igual que antes. Clip por defecto subido de 1e-3 a 0.05 (rango estándar de
   trimming de propensity). De paso, `oof_propensity` en `cross_fit.rs` ya no
   defaultea a 0.5 en silencio cuando un fold pierde un brazo de tratamiento
   (ahora `Err`, igual que `oof_regression_by_arm`).
6. ✅ **Wilcoxon exacto**: reemplazado el "normal approximation siempre" por
   un test exacto vía programación dinámica sobre las 2ⁿ asignaciones de
   signo (ranks duplicados ×2 para enteros, DP de subset-sum en `f64` para
   evitar overflow más allá de n=64), con fallback a normal + corrección de
   continuidad + corrección de empates (`Σ(t³-t)/48`) para n>100. Verificado
   contra fuerza bruta 2ⁿ en 4 casos con empates. El propio doctest del
   módulo (5 folds, afirmaba p<0.05) se corrigió a 6 folds — con 5 folds el
   mínimo p exacto es 2/32=0.0625, **matemáticamente no puede ser
   significativo**, exactamente el HIGH que motivó este ítem.
   ✅ **Golden tests vs sklearn 1.8.0 / scipy 1.17.1** (retoma el ítem
   desaparecido de Fase 1): `measure/mod.rs` tenía **0 tests** — ahora 4
   golden tests (accuracy/precision/recall/f1/balanced_accuracy/mcc/kappa en
   fixture de 30 muestras 3-clases; rmse/mae/r2/mape; auc/logloss/brier en
   fixture probabilístico). Al construirlos se confirmó el MEDIUM N10 del
   propio audit (precision/recall/F1 macro promediaban solo sobre clases con
   score definido, no sobre todas las clases presentes) — **corregido**:
   ahora promedian sobre `n_classes` completo, matching sklearn
   `zero_division=0` (verificado: clasificador binario degenerado que
   siempre predice clase 0 daba precision=0.5 antes, 0.25 después —
   coincide con sklearn). `PCA` (0 tests, solo shape-checks en integration.rs)
   tenía el bug concreto que el audit predijo: con 2 features exactamente
   anticorrelacionadas, el vector constante de arranque de la power iteration
   ES un autovector exacto (del autovalor CERO), y `mat.dot(v)=[0,0]` exacto
   dispara el early-return devolviendo esa dirección de varianza cero como
   "PC1" sin iterar — confirmado con test de regresión reproduciendo
   exactamente ese caso. Fix: arranque aleatorio (seed fija) + fallback a
   Gram-Schmidt determinista cuando la matriz deflactada ya no tiene norma
   significativa (autovalores restantes ~0, sin dirección dominante que
   buscar). 2 tests nuevos en `pca.rs`, ambos confirmados como discriminantes
   (fallan con la implementación vieja, pasan con el fix).
7. ✅ **Stacking**: en vez de cambiar a `StratifiedCV` interno (se evaluó y
   se descartó — el propio doctest del módulo usa 4 muestras/clase con
   `cv_folds=5` por defecto, y `StratifiedCV` exige `n_muestras_por_clase >=
   n_folds`, así que habría roto el doctest), se propagó `class_names` (+
   `feature_names`/`feature_types`) del task padre al task de cada fold —
   igual patrón que el ítem 11. Esto garantiza que el modelo base de
   cualquier fold produzca vectores de probabilidad del ancho correcto
   (`n_classes` global) aunque ese fold no contenga la clase máxima,
   eliminando el panic sin necesidad de cambiar la estrategia de folds.
   Confirmado con test de regresión (`stacking_classif_survives_fold_missing_a_class`):
   sin el fix, panic exacto `index out of bounds: the len is 2 but the index
   is 2`; con el fix, entrena sin error.
   ⏸️ CV loop de `benchmark.rs` (propagar metadatos categóricos) y LightGBM
   `subsample` se resolvieron ya en el ítem 11/Fase A — ver arriba, no
   quedaban pendientes para Fase B.
8. ✅ **GIL 2ª mitad**: `predict`/`predict_proba` (los 28 wrappers, vía
   `common.rs`), `predict_spatial` (GeoXGBoost + KrigingHybrid),
   `conformal_predict`, `Smote`/`SpatialSmote.balance`, los 10 filtros de
   `feature_selection.rs`, y `SpatialBufferCV.splits` ahora liberan el GIL
   con `py.allow_threads`. Verificado en dos niveles: (1) corrección
   funcional de cada API tras el cambio; (2) **prueba de concurrencia real**
   — un hilo Python contando en loop mientras corre `filter_relief` (O(n²),
   ~2.2s) alcanzó 15M incrementos durante la llamada, confirmando que el GIL
   efectivamente se liberó (con el GIL retenido el contador no avanzaría).

Verificado: 490 tests verdes tras el fix de CausalForest, **145 lib + 284
integración + 70 doctests (+ los de smelt-py)** al cierre de Fase B — 0
fallos. `smelt-py` recompilado y probado end-to-end (incluida la prueba de
concurrencia) después de cada fix.

### Gaps de golden tests que quedan fuera de esta pasada
`stats.rs` ganó tests exactos (ítem 6) y `measure/`/`PCA` pasaron de 0 tests
a golden tests reales (ítem 6), pero **survival/ (0 tests unitarios), ADASYN,
DES, EBM, KMeans, DBSCAN** siguen con solo smoke tests o ninguno — ninguno
de estos se tocó en Fase A/B. `tests/real_benchmark.rs` (comparación vs
sklearn real) sigue `#[ignore]` y fuera de `cargo test`/CI. Quedan para Fase
C o una iniciativa dedicada.

### Fase C — Deudas estructurales — **correctness items COMPLETADOS 2026-07-04, refactors arquitectónicos pendientes**
9. Los abiertos crónicos, todos con test de regresión:
   - ✅ **ADASYN**: interpolaba hacia cualquier punto de la misma clase, no
     hacia los k-NN reales (He et al. 2008). Fix: restringir candidatos a los
     k vecinos más cercanos (mismo patrón que `smote.rs`). Test con 2 clusters
     minoritarios lejanos separados por mayoría: sin el fix, un sintético
     aparecía en (5.57, 5.57) — justo en territorio de la clase mayoritaria
     (distancia a ambos clusters >6.2); con el fix, siempre <3.0 de un cluster.
   - ✅ **DES/KNORA-E**: la competencia se evaluaba sobre el mismo training set
     usado para entrenar los modelos base (comentario literal "same training
     set as validation"). Fix: split interno train/DSEL (`dsel_fraction`,
     default 0.3) — los modelos base entrenan solo en `train_idx`, competencia
     y vecinos-k se calculan solo en `dsel_idx`, sin reentrenar después
     (reentrenar volvería obsoletas las estimaciones de competencia).
     Expuesto en Python (`dsel_fraction`/`seed` en el constructor).
   - ✅ **EBM multiclase**: `train_classif` trataba cualquier target como
     binario sin importar `n_classes` (docstring decía literalmente
     "Simplified: use binary for now" sin el `Err`). Fix: `Err` explícito si
     `n_classes>2`; simplificada la rama muerta duplicada en `predict()`.
   - ✅ **RandomSurvivalForest**: solo existía `fit_predict` (bosque
     descartado, sin poder predecir datos nuevos) y el C-index era in-sample.
     Fix: nuevo `fit()` devuelve `(TrainedRandomSurvivalForest, oob_c_index)`
     — el modelo persiste para predecir sobre datos nuevos, y el C-index se
     computa agregando por muestra solo los árboles OOB (mismo patrón que el
     fix de CausalForest de Fase B). `fit_predict` se reimplementó sobre
     `fit()+predict()` preservando su salida numérica exacta (verificado con
     test dedicado). 3 tests nuevos: predicción sobre datos genuinamente
     nuevos, OOB C-index estrictamente menor que in-sample con un bosque
     forzado a sobreajustar (`min_node_size=1`), y equivalencia
     `fit_predict` == `fit().predict()`. Añadido `survival` al prelude
     (no estaba, gap M2 de arquitectura).
   - ✅ **Relief (RReliefF)**: ambos términos (positivo y negativo) dividían
     por `n_dc`; Robnik-Šikonja & Kononenko exigen normalizar el término
     negativo por `N_total − N_dC`. Fix + golden test contra una
     reimplementación independiente en Python/numpy de la misma fórmula
     (con normalización correcta e incorrecta calculadas aparte para
     confirmar que difieren sustancialmente: valores correctos ~0.02/0.18
     vs. los que daría el bug, ~−0.94/−0.70).
   - 512 tests verdes acumulados al cierre de la parte de correctness de
     Fase C (153 lib + 285 integración + 4 parquet + 70 doctests), 0
     fallos; `smelt-py` recompilado tras cada fix.

### Fase C, parte 2 — refactors arquitectónicos — **COMPLETADA 2026-07-04**
   - ✅ **Errores tipados**: agregadas dos variantes nuevas a `SmeltError`
     (`IncompatiblePrediction`, `NumericalError`), sumadas a las ya
     existentes (`InvalidParameter`, `Json`, etc.). Migrados 47 de 51 usos
     de `SmeltError::Other` — la concentración principal era `measure/mod.rs`
     (19 sitios, todos "predicción no coincide con lo esperado" →
     `IncompatiblePrediction`), más `conformal/` (4), `geo_xgboost.rs`,
     `kriging_hybrid.rs`, `shap.rs` (mismatches de predicción →
     `IncompatiblePrediction`; un `target_class out of range` →
     `InvalidParameter`), matrices singulares en `regularized.rs`/
     `linear_regression.rs`/`kriging_hybrid.rs` → `NumericalError`,
     `select_bandwidth`/loaders sin columna target → `InvalidParameter`,
     versión de formato de `serialize.rs` → `Json`. Quedan 4 sin migrar y
     documentados como tal (`s_learner.rs` ×3, wrapping defensivo de un
     error de `ndarray::concatenate` con baja probabilidad real de
     disparar; `label_encoder.rs` ×1, "unknown label", caso único sin un
     patrón repetido que justifique una variante nueva).
   - ✅ **Paralelismo con rayon**: `Resample` ahora requiere `Send + Sync`
     (todos los implementadores ya lo eran — structs planos con
     `Vec`/`f64`/`usize` — así que esto es puramente habilitante, sin
     cambios de comportamiento). Paralelizados:
     - `GridSearch`/`RandomSearch`: el loop sobre combinaciones/candidatos
       (cada uno construye su propio learner vía la factory `Send+Sync`,
       sin estado compartido). En `RandomSearch` el muestreo de parámetros
       se mantiene secuencial (barato, depende de `&mut rng`) y solo la
       evaluación (entrenar+puntuar) se paraleliza.
     - `Hyperband`: el loop de evaluación de configuraciones dentro de cada
       ronda de successive halving.
     - `benchmark_design::benchmark_classif`/`benchmark_regress`: el loop
       sobre `learners` para cada `task`, vía `par_iter_mut()` (cada
       `&mut Box<dyn Learner>` del slice es una porción disjunta, segura de
       mutar en paralelo sin aliasing).
     - **Deliberadamente NO paralelizado**: `BayesianOptimizer` (TPE es
       inherentemente secuencial — cada candidato se muestrea de la
       densidad de TODO el historial previo, así que la iteración *i* no
       puede empezar antes de conocer el resultado de la *i-1*; paralelizar
       de verdad requeriría un algoritmo de BO por lotes genuinamente
       distinto, no solo "conectar" el paralelismo existente — documentado
       en el código). Tampoco se tocó el loop de folds dentro de
       `benchmark::resample_classif`/`resample_regress` (recibe
       `&mut dyn Learner`, no una factory, así que paralelizar folds
       requeriría cambiar esa firma pública — invasivo, y ya se cubre el
       nivel exterior de candidatos/learners en todos los llamadores).
   - 3 tests nuevos de determinismo (`grid_search_parallel_evaluation_is_deterministic`,
     `hyperband_parallel_evaluation_is_deterministic`, más el
     `random_search_deterministic` ya existente): dos corridas con la misma
     semilla deben dar exactamente el mismo mejor resultado, verificando que
     la paralelización no introdujo condiciones de carrera.
   - 514 tests verdes al cierre completo de Fase C (153 lib + 287
     integración + 4 parquet + 70 doctests), 0 fallos; `smelt-py`
     recompilado.
10. Python: exponer cluster/persistencia/CausalForest/loaders; streaming API; `.pyi` + `__repr__`. No abordado — README/registry/versionado (M20, bump semver 2.0) tampoco. Quedan para otra iniciativa.

**Regla de proceso que esta auditoría reafirma**: ningún fix estadístico se declara cerrado sin su golden test, y ningún ítem del plan se elimina de una tabla de progreso sin nota explícita de por qué.
