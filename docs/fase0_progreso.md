# Fase 0 — Fixes quirúrgicos de correctness

Seguimiento de avance. Referencia: `docs/auditoria_motor_2026-07-01.md`.

| # | Tarea | Estado | Commit | Notas |
|---|-------|--------|--------|-------|
| 1 | Fix histogram.rs:55 (binning binario) + tests regresión | ✅ hecho | `bc2a0e5` | 1 char: `<` → `<=`. +2 tests unitarios en histogram.rs, +3 tests de regresión (xgboost/lightgbm/catboost, n=600, target=binaria*10, RMSE<1). Suite completa verde (255+22+61). |
| 2 | Re-correr benchmark King County (post-fix) | ✅ hecho | — (sin cambios de código) | Mejora medible en XGBoost y GeoXGBoost (ver log). |
| 3 | GeoXGBoost: eliminar heurística nrows en predict() | ✅ hecho | `9daeac7` | `predict()` ahora siempre global-only. 5 call sites (tests+examples+README) migrados a `train_geo()`+`predict_spatial()`. Python API no afectada (ya usaba predict_spatial). |
| 4 | GeoXGBoost: alpha adaptativo OOB vs in-sample | ✅ hecho | `21b6947` | Nuevo método `global_oob_errors` (k-fold CV, k=min(5,n)). Solo se computa si alpha=None (adaptativo); si el usuario fija alpha, se salta (sin costo extra). El demo de George usa alpha=0.5 fijo, así que NO cambia por este fix — solo afecta a quien use el alpha adaptativo por defecto. |
| 5 | Isolation Forest: fórmula c(n) | ✅ hecho | `7c9a0a6` | Paréntesis faltante corregido + n=2 special-case (sklearn) + path_length ahora f64 (sin truncar). |
| 6 | Conformal: validar alpha/n | ✅ hecho | `325691f` | 3 módulos (ConformalRegressor, ConformalClassifier, CQR): validación de alpha∈(0,1) + calibración no vacía + widening a infinito en vez de underflow/panic. |
| 7 | smelt-py: liberar GIL | ✅ hecho | `85f1226` | `py.allow_threads` en los 13 fit(), select_bandwidth, BayesianOptimizer.optimize, shap, permutation_importance, RFE. Verificado empíricamente (hilo en background avanza ~9.8M iter durante un fit()). |
| 8 | smelt-py: validar labels negativos + unwraps | ✅ hecho | `a0cf7d4` | `extract_class_labels()` (usado en fit/shap/perm/optimize) + `build_param_space` sin unwraps + `auc_roc_score`/`predict_proba_values` sin panics con inputs vacíos. 0 `.unwrap()` restantes en lib.rs. |

**Todos los commits verificados**: cada uno compila de forma independiente y pasa la suite completa en el punto en que se creó (se construyeron mediante cirugía de reversión/restauración de ediciones en los 3 archivos con cambios mezclados: `geo_xgboost.rs` items 3+4, `tests/integration.rs` items 3+6, `smelt-py/src/lib.rs` items 7+8). Diff acumulado `HEAD~7..HEAD` idéntico al diff pre-troceo (17 archivos, 762 inserciones, 153 eliminaciones).

## Log

### 2026-07-01 — Item 1: histogram.rs boundary bug
- **Causa raíz confirmada**: en `HistBins::build`, cuando `n_unique <= n_bins`, el último boundary generado (`vals[idx]`) coincide exactamente con `vals[n_unique-1]` (el máximo). La condición `*bounds.last().unwrap() < vals[n_unique-1]` es entonces falsa por igualdad estricta, así que el boundary `INFINITY` nunca se agrega. Al asignar bin con `position(|&b| val < b)`, el valor máximo no encuentra ningún boundary estrictamente mayor y cae por `unwrap_or(bounds.len()-1)` al mismo bin que el segundo valor más alto.
- **Caso extremo verificado**: feature binaria {0,1} → boundaries=[1.0] (sin INFINITY) → 0.0 y 1.0 ambos mapean a bin 0 → la feature es invisible para el split-finder.
- **Fix**: `<` → `<=` en la comparación (histogram.rs:55).
- **Tests agregados**:
  - `histogram.rs::tests::binary_feature_gets_two_bins` — verifica 2 bins distintos para {0,1}.
  - `histogram.rs::tests::low_cardinality_feature_separates_all_values` — generaliza a valores {1..5}.
  - `xgboost.rs::weight_tests::binary_feature_is_splittable_in_histogram_mode` — n=600 (fuerza modo histograma, threshold es n<=256), target=bit*10, RMSE<1.
  - `lightgbm.rs::tests::binary_feature_is_splittable` — mismo caso (LightGBM siempre usa histograma).
  - `catboost.rs::tests::binary_feature_is_splittable` — mismo caso (CatBoost usa 64 bins hardcoded).
- **Resultado**: los 5 tests nuevos pasan; suite completa sin regresiones (255 unit + 22 integration/lib + 61 doctests, todos ok).
### 2026-07-01 — Item 2: re-run benchmark King County (post-fix)
Reconstruido smelt-py (`maturin develop --release`) con el fix y corrido `paper/replication/demo_geoxgboost.py`.

| Métrica | Antes (0.4.6, bug) | Después (fix) | Δ |
|---|---|---|---|
| XGBoost holdout RMSE / R² | 0.345 / 0.586 | **0.332 / 0.617** | mejora |
| XGBoost+X,Y holdout RMSE / R² | 0.237 / 0.804 | **0.219 / 0.833** | mejora |
| GeoXGBoost (α=0.5, bw=100) RMSE / R² | 0.269 / 0.748 | **0.263 / 0.760** | mejora |
| Bandwidth LOO CV @ bw=300 | 0.2810 | 0.2825 | cambia (bw=300>256 fuerza modo histograma) |
| Bandwidth LOO CV @ bw=100/150/200 | igual | igual | sin cambio (n<256 → exact greedy, no afectado) |
| SpatialBufferCV mean RMSE | 0.541 ± 0.059 | 0.541 ± 0.059 | idéntico |
| Top-5 feature importances | igual | igual | idéntico |
| Bandwidth seleccionado | 100 | 100 | sin cambio |

**Interpretación**: el patrón es coherente con la causa raíz. GeoXGBoost entrena modelos locales con n=bandwidth muestras; con bw∈{100,150,200} (todas <256, el umbral exact/histograma de XGBoost) los modelos locales usan exact-greedy y no estaban afectados por el bug — de ahí que el bandwidth seleccionado y buena parte de las métricas locales no cambien. El modelo XGBoost global (n_train=800, siempre en modo histograma) sí estaba afectado, y mejora visiblemente al poder usar `waterfront` (binaria) y otras features de baja cardinalidad (`view`, `condition`) como splits.

**Implicación para el paper/correspondencia con George**: los números en `reply_grekousis_eq13_2026-06-22.txt` (RMSE 0.276→0.269 tras el fix de Eq. 13) NO incluyen este segundo fix — hay una mejora adicional real (0.269→0.263) que vale la pena mencionar si esa nota aún no se envió, o como seguimiento si ya se envió. **Acción sugerida, no ejecutada**: confirmar con el usuario si el correo ya se envió antes de decidir si se agrega una nota de seguimiento.

### 2026-07-01 — Item 3: GeoXGBoost predict() heurística de nrows
- **Causa raíz**: `TrainedModel::predict()` para `TrainedGeoXGBoost` comparaba `features.nrows() == self.local_models.len()`; si coincidía, aplicaba el modelo local i-ésimo por POSICIÓN de fila, asumiendo que era el training set. Un dataset nuevo con el mismo n recibía combinaciones locales de puntos completamente ajenos, en silencio (sin error, sin warning).
- **Fix**: `predict()` ahora es siempre global-only, documentado explícitamente en el docstring. Toda predicción espacial (local + blend con alpha) exige `predict_spatial(features, coords)`, que hace nearest-neighbor real sobre coordenadas explícitas — incluyendo el caso "fitted values" (pasar `predict_spatial(train_features, model.coords())`).
- **Call sites migrados** (de `train_regress()+predict()` a `train_geo()+predict_spatial()`):
  - `tests/integration.rs`: `geo_xgboost_basic_regression`, `geo_xgboost_spatial_heterogeneity`, `geo_xgboost_fixed_alpha` — antes pasaban igual (porque ahora ambos caminos usan el mismo global model y los umbrales de assert eran laxos), pero habían perdido silenciosamente su propósito de probar el comportamiento LOCAL. Restaurados para ejercer los modelos locales de verdad.
  - `examples/spatial_ml.rs`, `examples/gis_workflow.rs` — mismo patrón (in-sample "fitted values").
  - `examples/case_study_spatial.rs`, `examples/case_study_king_county.rs` — **hallazgo colateral**: estos ejemplos evaluaban GeoXGBoost out-of-sample con `.predict(test_features)`, donde `test_features.nrows() != n_train` — es decir, por la vieja heurística, el "GeoXGBoost" del benchmark en estos dos ejemplos SIEMPRE estaba midiendo el modelo global puro (nunca usó predicción espacial real), sin que nadie lo notara. Corregido para usar `predict_spatial(test_features, test_coords)` — ahora el benchmark mide el ensemble real.
  - `examples/validate_geoxgboost.rs` — ya usaba `train_geo()` pero llamaba `.predict()` para in-sample; corregido a `predict_spatial`.
  - `README.md` — snippet actualizado para mostrar `train_geo()` + `predict_spatial()` como el patrón correcto.
- **Test de regresión agregado**: `geo_xgboost.rs::tests::predict_is_global_only_never_positional_local_models` — entrena, llama `predict()` con el mismo n que training, y verifica que el resultado sea bit-a-bit igual a `global_model.predict()` solo.
- **Resultado**: suite completa verde (255 unit + 23 lib-tests + 61 doctests); todos los examples compilan; demo King County vía Python idéntico al run del item 2 (la API Python ya usaba `predict_spatial` correctamente, no se vio afectada).
- **Nota**: `case_study_spatial.rs` y `case_study_king_county.rs` ahora reportarán números de GeoXGBoost distintos (mejores o peores) la próxima vez que corran, porque antes medían accidentalmente el modelo global. No se re-corrieron (requieren `data/meuse.csv` / `data/king_county_1k.csv` que no verifiqué que existan en este entorno).

### 2026-07-01 — Item 4: alpha adaptativo (OOB vs in-sample)
- **Causa raíz**: en `train_geo_inner`, `e_local` (error de cada modelo local) es OOB por construcción (el punto central se excluye del vecindario), pero `e_global` era el residuo IN-SAMPLE del XGBoost global entrenado sobre el 100% de los datos — con 100 árboles de profundidad 6, ese modelo casi interpola (`e_global≈0` para casi todos los puntos). La comparación `e_local <= e_global` casi siempre resultaba falsa, sesgando alpha sistemáticamente hacia el modelo global (Eq. 19-20 de Grekousis exige comparar errores del mismo tipo).
- **Fix**: nuevo método `global_oob_errors()` que calcula el error del modelo global vía k-fold CV (k=min(5,n), seed=self.seed, folds paralelizados con rayon) — igual de "fuera de muestra" que el error local. Solo se ejecuta cuando `self.alpha` es `None` (adaptativo); si el usuario fija un alpha explícito, se omite por completo (sin costo adicional).
- **Test de regresión**: `global_oob_errors_are_not_optimistic_like_in_sample_residuals` — entrena un XGBoost muy flexible (300 árboles, depth=8, lambda=0.01) sobre n=40 con ruido, que memoriza casi perfectamente (residuo in-sample bajo); verifica que el error OOB calculado por el nuevo método sea al menos 2x mayor que el residuo in-sample, confirmando que ya no es optimista.
- **Impacto en el demo de George**: `demo_geoxgboost.py` usa `alpha=0.5` FIJO (no adaptativo, por diseño — así lo pidió Grekousis en su feedback), por lo que este fix **no cambia los números ya reportados**. Solo afecta a quien use el alpha adaptativo por defecto (`alpha=None`). Verificado con smoke test en King County: adaptativo da RMSE=0.271, R²=0.744 (comparable al fijo, sin errores).
- **Resultado**: suite completa verde (255 unit + 24 lib-tests + 61 doctests); smelt-py reconstruido y probado sin errores.

### 2026-07-01 — Item 5: fórmula c(n) de Isolation Forest
- **Causa raíz**: faltaba un paréntesis. Código: `2.0*(n-1.0).ln() + 0.5772156649 - 2.0*(n-1.0)/n` (gamma sumado UNA vez fuera del *2) en vez de `2.0*((n-1.0).ln() + 0.5772156649) - 2.0*(n-1.0)/n` (gamma multiplicado por 2, como exige H(i)=ln(i)+gamma). Con n=2 el resultado era negativo (−0.42); con n=256 daba ~9.67 en vez de ~10.24 (sklearn).
- **Fix**: paréntesis agregado. Además: (a) caso especial n=2 → 1.0 exacto (así lo hace `sklearn._average_path_length`, porque ln(n-1)=ln(1)=0 subestima ese caso límite); (b) `path_length` cambiado de `usize` a `f64` de punta a punta — antes truncaba `c_factor(*size) as usize`, perdiendo el ajuste fraccional y saturando valores negativos a 0 silenciosamente.
- **Tests agregados**: `c_factor_matches_reference_values` (valores de referencia sklearn en n=2,10,256 + verificación de no-negatividad para n en 1..1000) y `outlier_gets_higher_score_than_inliers` (smoke test end-to-end).
- **Resultado**: suite completa verde (255 unit + 26 lib-tests + 61 doctests).

### 2026-07-01 — Item 6: conformal prediction, validación alpha/n
- **Causa raíz**: en los 3 módulos (`ConformalRegressor`, `ConformalClassifier`, `conformal/cqr.rs::CQR`), el rank del cuantil se calculaba como `q_idx = q_idx.min(n) - 1` (usize). Si `alpha>=1` o el set de calibración estaba vacío, `q_idx.min(n)` podía ser 0 → `0usize - 1` hace underflow: panic en debug, wraparound silencioso en release (que luego un segundo `.min(n-1)` "arregla" por accidente, clampando al residuo máximo sin ninguna garantía estadística). Con `n=0` el índice quedaba fuera de rango de todas formas → panic garantizado.
- **Fix**: validación explícita al inicio de `calibrate()` en los 3 módulos: `alpha` debe estar en (0,1) y el set de calibración no puede estar vacío (`SmeltError::InvalidParameter` / `EmptyDataset`). Cuando `ceil((n+1)(1-alpha)) > n` (calibración insuficiente para la confianza pedida), ya no se clampa silenciosamente al residuo máximo — se widening explícito a `f64::INFINITY` (regresión/CQR) o al conjunto completo de clases (`quantile_score=1.0`, clasificación), documentado como la única opción consistente con la garantía de cobertura (Vovk et al.). También se corrigió `probs.get(t)` en el clasificador para no panicar con labels de calibración fuera de rango.
- **Tests agregados** (`tests/integration.rs`): `conformal_rejects_alpha_out_of_range`, `conformal_rejects_empty_calibration_set`, `conformal_tiny_calibration_set_widens_instead_of_panicking`, y — siguiendo la recomendación explícita de la auditoría — `conformal_regression_empirical_coverage_near_target` (test de cobertura empírica real con n=200 calibración / n=500 test, verifica coverage≥1-alpha-0.05; este es el tipo de test que habría detectado el bug original, a diferencia del smoke test existente que solo comprobaba `lower<=upper`).
- **Resultado**: suite completa verde (259 integration + 26 lib-tests + 61 doctests). smelt-py reconstruido; sección conformal del demo de George sin cambios (su calibración de 160 puntos es suficiente).

### 2026-07-01 — Item 7: liberar el GIL en smelt-py
- **Causa raíz**: 0 usos de `py.allow_threads` en todo `smelt-py/src/lib.rs`. Cualquier entrenamiento rayon-paralelo (potencialmente de minutos, ej. `select_bandwidth`) retenía el GIL de punta a punta — Jupyter/otros hilos Python quedaban congelados, sin progress bars, Ctrl+C sin efecto.
- **Fix**: patrón "extraer bajo GIL → entrenar sin GIL" aplicado en:
  - Los **13 `fit()`** de learners simples (XGBoost, CatBoost, RandomForest, etc.): refactor de `fit_learner()` para que reciba `py: Python<'_>` y envuelva `learner.train_classif/train_regress` en `py.allow_threads(...)`. Los 11 sitios con firma idéntica se actualizaron con `Edit(replace_all=true)` dado que el patrón era carácter-por-carácter igual.
  - `GeoXGBoost::fit` y `select_bandwidth` (el caso que motivó el hallazgo — LOO CV de minutos).
  - `shap_impl` (classif + regress) y `perm_importance_impl` (classif + regress).
  - `BayesianOptimizer::optimize`.
  - `rfe()` (backward elimination — re-entrena el learner una vez por feature eliminada, tan costoso como varios `fit()` seguidos).
  - **Cambio estructural necesario**: el trait `Measure` (src/measure/mod.rs) no exigía `Send + Sync`, lo que impedía capturar `&dyn Measure` dentro de la clausura de `allow_threads` (requiere `Send`). Se agregó `Send + Sync` al trait, igual que ya tenían `Learner` y `TrainedModel` — cambio seguro porque las 10 implementaciones existentes (Accuracy, Rmse, etc.) son structs triviales, automáticamente Send+Sync.
- **De paso**: `extract_class_labels()` nuevo helper que valida labels no-negativos antes de castear a `usize` (usado también por shap/perm/optimize, adelantando parte del item 8); `RFE::selected_indices().unwrap()` reemplazado por `ok_or_else` con `PyRuntimeError` explícito.
- **Verificación empírica** (no solo compilación): script Python con un hilo en background incrementando un contador mientras corre `XGBoost(n_estimators=300).fit()` sobre 5000×20 — el contador avanzó ~9.8M iteraciones durante el fit (antes del fix habría avanzado ~0, todo bloqueado por el GIL).
- **Resultado**: suite Rust completa verde (259+26+61); demo King County idéntico (sin regresión funcional).

### 2026-07-01 — Item 8: smelt-py — labels negativos + unwraps de input de usuario
- **`extract_class_labels()`** (agregado en el item 7 y ahora usado también aquí): valida `v >= 0` antes de `as usize`, con `PyValueError` indicando índice y valor ofensor. Cubre `fit_learner`, `shap_impl`, `perm_importance_impl` y `BayesianOptimizer::optimize` — los 4 sitios donde `Vec<i64>` de Python se convierte a labels de clase.
- **`build_param_space`**: 5 `.unwrap()` sobre `get_item("low"/"high"/"values")` reemplazados por un closure `required(field)` que devuelve `PyRuntimeError` con el nombre del parámetro y el campo faltante, en vez de un `PanicException` con traceback de Rust.
- **`auc_roc_score`**: `.max_by(...).unwrap()` sobre una fila de probabilidades — panic si la fila está vacía. Reemplazado por `.ok_or_else(...)` + `collect::<PyResult<_>>()`.
- **`predict_proba_values`**: `probs[0].len()` panicaba con `n=0` (predicción sobre un array vacío). Reemplazado por `probs.first().map_or(0, |row| row.len())`.
- **Fuera de scope, documentado**: `accuracy_score`/`f1_score`/`precision_score`/`recall_score` castean `y_pred: Vec<f64>` (predicciones ya numéricas) a `usize` con `as` — es un cast saturante bien definido en Rust (no wraparound, no panic), a lo sumo silenciosamente incorrecto si alguien pasa predicciones negativas a mano. No es el mismo bug (el bug real era el wraparound de `i64 as usize` sobre labels de entrada), así que se dejó fuera de este fix quirúrgico.
- **Verificación funcional** (no solo compilación): `XGBoost().fit(X, y=[-1,1,-1,1])` → `ValueError` limpio; `BayesianOptimizer().optimize(..., {"max_depth": {"type": "uniform"}}, ...)` (sin "low"/"high") → `RuntimeError` limpio con mensaje útil. Ambos antes causaban un panic/abort del intérprete.
- **Resultado**: 0 `.unwrap()` restantes en `smelt-py/src/lib.rs`. Suite Rust completa verde (259 integration + 26 lib-tests + 61 doctests).

## Fase 0 — COMPLETA (8/8)

Todos los fixes quirúrgicos de correctness de la auditoría del 2026-07-01 están implementados, testeados (unit + integración + verificación funcional en Python) y documentados. **Nada de esto está commiteado todavía** — pendiente de revisión y decisión del usuario sobre cómo trocear los commits (posiblemente uno por ítem, dado que tocan módulos independientes) y sobre versión a publicar (el histograma + GeoXGBoost afectan resultados ya compartidos con George).


