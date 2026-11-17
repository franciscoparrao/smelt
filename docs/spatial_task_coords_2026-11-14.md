# Coordenadas en la Task: GeoXGBoost y Moran's I dentro del benchmark (2026-11-14)

Motivación: la opción (a) de la lista de candidatos tras cerrar Moran's I. Hasta ahora
`GeoXGBoost` y `KrigingHybrid` guardaban las coordenadas del dataset completo en el learner,
así que dentro de `benchmark::resample_regress` (que entrena sobre un subconjunto de filas)
fallaban con `DimensionMismatch`, y `TrainedModel::predict` no podía ser espacial porque el
trait no conoce coordenadas. El mismo hueco impedía que `MoranResiduals` puntuara por fold. En
Python, `smelt.benchmark.benchmark()` lo resolvía con un parche a mano (`_NEEDS_COORDS`,
`fit(X, y, coords[train_idx])`, `predict(X, coords[test_idx])`); en Rust no había nada.

## Diseño elegido: coordenadas como metadato de la Task

Dos opciones sobre la mesa:

1. **Trait `SpatialLearner`** separado, con `train_regress_spatial(task, coords)` y
   `resample_regress_spatial(...)`. Obliga a duplicar el loop de benchmark, el diseño de
   benchmark, los cuatro tuners y `AutoTuner`, o a aceptar que esos no compongan.
2. **Coordenadas en la `Task`** (`with_coords`, `coords()`), rebanadas por fold igual que los
   `weights`, más dos métodos con implementación por defecto en los traits:
   `TrainedModel::predict_spatial(features, coords)` (default: ignora coords y llama a
   `predict`) y `Measure::score_spatial(prediction, coords)` (default: `score`).

Se eligió (2). Es lo que hace mlr3 (`TaskRegrST` de mlr3spatiotempcv: columnas con rol
`coordinate`), sigue el precedente exacto de los `weights` (ítem 2 del roadmap mlr3-parity,
2026-07-18), y todo lo que ya consume `Learner`/`Task` compone sin una línea nueva: tuning de
ancho de banda de GeoXGBoost bajo `SpatialBlockCV` con `GridSearch`, `benchmark_regress` con
varios learners, `Pipeline(StandardScaler → GeoXGBoost)`, `TargetTransformRegressor`.

## Qué cambió

- `src/task/mod.rs`: campo `coords: Option<Vec<(f64, f64)>>` en ambas Tasks, `with_coords`
  (panic con mensaje claro si largo ≠ n o hay NaN/∞, mismo precedente que `with_weights`),
  `coords()`, y `Task::coords()` en el trait con default `None`.
- `src/learner/mod.rs`: `TrainedModel::predict_spatial` por defecto.
- `src/measure/mod.rs`: `Measure::score_spatial` por defecto.
- `src/benchmark.rs`: rebana coords para el fold de entrenamiento (`with_coords`) y pasa el
  rebanado del fold de test a `predict_spatial` y a `score_spatial`. Sin coords en la Task el
  camino es idéntico al anterior.
- `src/learner/geo_xgboost.rs`: constructor `coords_from_task()` (coords vacías) y helper
  privado `coords_for(task)`: si la Task trae coords se usan (están alineadas por
  construcción); si no, las del learner, con las validaciones de siempre; si no hay ninguna,
  `InvalidParameter` con el mensaje de qué hacer. Lo usan `train_geo_inner`,
  `select_bandwidth` y `search_bandwidth`. `predict_spatial` pasó a ser el método del trait;
  se dejó un alias inherente para que `model.predict_spatial(...)` siga compilando sin
  importar `TrainedModel` (los `examples/` lo llaman así). **Nada del algoritmo cambia**: el
  test de integración exige igualdad exacta con el loop manual anterior.
- `src/learner/kriging_hybrid.rs`: lo mismo (`coords_from_task(factory)`).
- `src/learner/registry.rs`: `"geo_xgboost"` registrado (30 learners; README y el test de
  conteo actualizados). `tests/contract.rs` georreferencia su task de regresión para que el
  learner espacial pase el autotest como cualquier otro.
- `src/measure/moran.rs`: `MoranResiduals::coords_from_task()`, `score_spatial`,
  `test_with_coords`. Cuando el measure tiene coords propias y además recibe coords por
  `score_spatial`, ganan las recibidas (son las alineadas con las filas).
- Propagación en wrappers que reconstruyen la Task con las mismas filas: `Pipeline`
  (`train_classif`/`train_regress`; además `TrainedPipeline::predict_spatial` transforma y
  reenvía coords al modelo interno) y `TargetTransformRegressor` (`predict_spatial`
  reenvía y des-transforma). `TrainedAutoTuner::predict_spatial` reenvía. Un `Pipeline` con
  etapa de resampler rechaza una Task con coords (las muestras sintéticas no tienen
  ubicación; `SpatialSmote` fuera del Pipeline sí las devuelve).
- No propagan coords (deliberado): `Stacking`, `DES`, `CalibratedClassifier`,
  `ThresholdedClassifier`, `Bagging`, `cross_fit`. Rebanan filas pero no hay clasificador
  espacial que las consuma; GeoXGBoost como base de Stacking sigue fallando como antes.

## Verificación (`tests/spatial_benchmark.rs`, 9 tests)

- `resample_regress(GeoXGBoost::coords_from_task(), task.with_coords, SpatialBlockCV)` da
  **exactamente** el mismo RMSE por fold que el loop manual (`GeoXGBoost::new(coords_fold)`
  + `train_geo` + `predict_spatial`), y se comprueba que difiere de la predicción global
  (`predict`) en algún fold, o sea que de verdad se usó `predict_spatial`.
- Ídem para `KrigingHybrid`.
- Las coords de la Task ganan sobre las del learner; `GeoXGBoost::new(coords completas)`
  sobre una Task sin coords sigue dando `DimensionMismatch` (comportamiento anterior intacto);
  `coords_from_task()` sin coords en ningún lado da `InvalidParameter` con "no coordinates".
- Un learner aespacial (`LinearRegression`) con y sin coords en la Task da scores idénticos.
- `MoranResiduals::coords_from_task()` por fold coincide con `morans_i` manual sobre los
  residuos del fold; el I medio de GeoXGBoost es menor que el de la regresión lineal.
- `Pipeline(StandardScaler → GeoXGBoost)`: `predict_spatial` del pipeline es igual a
  escalar a mano y entrenar GeoXGBoost con coords explícitas, y distinto de `predict`.
- `TargetTransformRegressor(Sqrt, GeoXGBoost)`: `predict_spatial` reenvía y des-transforma.
  **Este test falló la primera vez**: el wrapper reconstruía la Task sin coords. Corregido
  propagando coords junto a weights/feature_types (clase de bug M-3 de la auditoría).
- `GridSearch` sobre `bandwidth` de GeoXGBoost bajo `SpatialBlockCV`: cada score coincide
  con el `resample_regress` del candidato.
- `learner_from_id("geo_xgboost")` dentro de `benchmark_regress` junto a otro learner.

Suite completa en verde (lib, doctests, integración, contrato), clippy limpio en lo nuevo
(los únicos avisos/errores de clippy son preexistentes en `examples/` e
`isolation_forest.rs`). `smelt-py` compila sin cambios de código: sus bindings siguen
construyendo `GeoXGBoost::new(coords)` y llamando `predict_spatial` por el alias inherente.

## Pendiente natural (no hecho aquí)

- Python: `smelt.benchmark.benchmark()` conserva su parche `_NEEDS_COORDS` porque smelt-py no
  expone `Task`. Si algún día se expone, el parche sobra.
- Un clasificador espacial haría necesario propagar coords en los wrappers de clasificación.
