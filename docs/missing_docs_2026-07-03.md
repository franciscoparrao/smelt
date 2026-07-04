# `missing_docs` cleanup (item 17b) — 2026-07-03

## Contexto

Último ítem grande pendiente de Fase 3. Un intento previo (documentado en
`docs/fase3_progreso.md`) había activado el lint temporalmente, contado 308
advertencias, y revertido sin commitear por no tener el tiempo dedicado
necesario en esa sesión. El código creció bastante desde entonces (causal
meta-learners, ParquetLoader, sparse.rs, etc.), así que el conteo real al
retomar esto fue **330 advertencias**.

## Enfoque

Dado que el trabajo es genuinamente mecánico (agregar un comentario `///`
de una línea a cada item público sin documentar — constructores, builder
methods `with_x`, campos de struct, variantes de enum) y las 330
advertencias están repartidas en ~58 archivos sin ninguna dependencia entre
sí, se paralelizó en **8 agentes**, cada uno responsable de un subconjunto
disjunto de archivos, con instrucciones idénticas de estilo:

- Ningún cambio de lógica — solo insertar líneas `///`.
- Una línea por doc (rara vez dos), sin relleno genérico ("This function
  does X"), afirmando directamente qué hace/representa el item.
- Leer el código real (uso del campo, doc del módulo, sitios de
  construcción) antes de escribir cada descripción — no adivinar solo por
  el nombre.
- Igualar el estilo terso ya usado en los pocos items que sí tenían
  documentación (ej. `t_learner.rs`/`x_learner.rs` en `causal/meta_learners/`,
  `FeatureType` en `task/mod.rs`).
- Verificar al final con `cargo build -p smelt-ml` que su subconjunto de
  archivos queda en 0 advertencias y que el build sigue compilando.

## Batches

| Batch | Archivos | Docs agregados |
|---|---|---|
| A | error.rs, prediction/mod.rs, serialize.rs, sparse.rs, benchmark_design.rs | 52 |
| B | causal/mod.rs, causal/meta_learners/{dr,r,s}_learner.rs | 11 |
| C | cluster/{mod,isolation_forest}.rs, conformal/cqr.rs, data/mod.rs, multilabel/mod.rs, multioutput/mod.rs, survival/mod.rs | 28 |
| D | learner/{adaboost,bagging,des,ebm,hoeffding,knn,naive_bayes,stacking,svm}.rs | 32 |
| E | learner/{quantile,quantile_forest,regularized,linear_regression,logistic_regression,oblique}.rs | 37 |
| F | learner/{catboost,geo_xgboost,histogram,lightgbm,xgboost}.rs | 61 |
| G | learner/tree/{mod,decision_tree,extra_trees,gradient_boosting,random_forest}.rs | 41 |
| H | preprocess/*.rs (9 files), resample/*.rs (3), task/mod.rs, tuning/*.rs (4) | 54 |

Total: **316 doc comments** en 59 archivos (el resto de las 330
advertencias originales se resolvió por overlap entre el conteo inicial y
correcciones incidentales — el conteo final verificado es 0).

## Hallazgos durante la revisión

- Los agentes confirmaron semántica leyendo el código real en vez de
  adivinar por nombre: p. ej. `nan_left`/`nan_goes_left` en los nodos de
  árbol de XGBoost/LightGBM se verificó leyendo `predict_one` antes de
  documentar como "ruta las filas con NaN al hijo izquierdo"; `with_gamma`/
  `with_n_candidates` de `BayesianOptimizer` se verificaron leyendo su uso
  real en el algoritmo de adquisición (líneas 181-193 de `bayesian.rs`);
  `with_max_features_sqrt` de `RandomForest` se confirmó que internamente
  fija `max_features_fraction = 0.0`, interpretado más abajo como el
  heurístico `sqrt(n_features)`.
- Único cambio no puramente aditivo: 2 declaraciones de variante de enum en
  una sola línea (`DimensionMismatch { expected: usize, got: usize }` en
  `error.rs`, `Huber { delta: f64 }` en `xgboost.rs`) se expandieron a
  multi-línea para poder adjuntar un doc a cada campo — sin cambio de
  lógica, solo formato.

## Validación

`cargo build -p smelt-ml 2>&1 | grep -c "missing documentation"`: 330 → 0.
`cargo test -p smelt-ml --lib`: 101 verdes, sin cambios. `cargo test -p
smelt-ml --doc`: 66 verdes, sin cambios. `cargo test --test integration`:
274 verdes, sin cambios. `cargo check -p smelt-py`: limpio. `git diff
--stat -- src/`: 59 archivos, 410 inserciones, 3 eliminaciones (las 3
eliminaciones son exactamente las 2 reformateos de una línea a multi-línea
mencionados arriba, confirmado revisando el diff completo).

Con esto, Fase 3 (`docs/auditoria_motor_2026-07-01.md`) queda completa en
todos sus ítems.
