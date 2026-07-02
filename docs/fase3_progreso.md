# Fase 3 — Paridad competitiva

Seguimiento de avance. Referencia: `docs/auditoria_motor_2026-07-01.md` (sección "Fase 3").
Fase 0, 1 y 2 completas y pusheadas.

**Nota de alcance**: a diferencia de Fase 0-2, los 4 ítems de Fase 3 (14-17) tienen
tamaños muy dispares — el ítem 14 (categóricas + NaN en `Task`) es probablemente
más grande que Fase 0+1+2 juntas. Esta fase se ejecuta como una serie de
"ganancias rápidas" de bajo riesgo primero (subconjunto de los ítems 16/17),
dejando los ítems 14 y 15 completos para sesiones dedicadas con su propio
scoping.

| # | Tarea | Estado | Commit | Notas |
|---|-------|--------|--------|-------|
| 16a | Medidas faltantes: BalancedAccuracy, CohensKappa, MCC, Brier | ✅ hecho | `1731956` | Con tests de valor conocido derivados a mano (no smoke tests) en tests/integration.rs. Expuestas en prelude y como score functions en smelt-py. |
| 16b | Model registry (`learner_from_id`) | ✅ hecho | `67f45c4`, fix `8b813c7` | 21 de 26 learners registrados (todos los que tienen `Default` sensato + QuantileGB con tau=0.5). Excluidos: Bagging/Stacking/DynamicEnsemble (necesitan factory de base learner) y GeoXGBoost (necesita coords externas). **Corrección** (`8b813c7`): ObliqueForest había quedado excluida por error asumiendo que necesitaba factory como las 3 anteriores — es autocontenida (`Default` + builders escalares), se agregó al registry. |
| 16c | Predict paralelo consistente | ✅ hecho | `14f318c` | XGBoost y CatBoost tenían Regression/BinaryClassif paralelos con rayon pero MultiClassif serial; LightGBM tenía los 3 modos seriales. Ahora los 3 motores usan el mismo patrón `into_par_iter()` en los 3 modos. Sin cambios numéricos, solo el loop por fila. |
| 17a | README.md / CLAUDE.md al día | ✅ hecho | `8f36278` | Versión 0.6→1.3, conteo de learners 21→26, tabla de learners/medidas/resampling completada, roadmap de CLAUDE.md (Phase 1-6, todo marcado como pendiente pese a estar hecho hace tiempo) reconciliado con el estado real. |
| 17b | `#![warn(missing_docs)]` | ❌ evaluado, no ejecutado | | 308 advertencias al activarlo — demasiado para tratarlo como ganancia rápida. Queda como ítem grande para una sesión dedicada (escribir ~300 doc comments o decidir cuáles APIs realmente necesitan documentación pública vs volverlas `pub(crate)`). |
| 14 | Categóricas + NaN en Task/splits; early stopping real; monotone constraints; objetivos custom | pendiente | | El ítem más grande del proyecto — toca `Task`, `CsvLoader`, `histogram.rs`, y el split-finding de XGBoost/LightGBM/CatBoost a la vez. Requiere su propia sesión de scoping antes de empezar. |
| 15a | Macro `define_learner!` + exponer 11 de 14 learners faltantes | ✅ hecho | `2565c23`, fix `8b813c7` | AdaBoost, EBM, Lasso, ElasticNet, GradientBoosting, HoeffdingTree, LinearSVM, ObliqueTree, ObliqueForest (agregada en el fix), QuantileForest, QuantileGB. Reusa `add_explain_methods!`/`declare_support!` existentes (shap_values, permutation_importance, conformal_predict, supports_classification/regression) — no solo fit/predict. Verificado de punta a punta con `maturin develop --release` + smoke test real en Python (no solo `cargo build`). Bug encontrado y corregido en el desarrollo: `$has_proba:literal` no se puede reenviar/re-matchear en una macro recursiva (macro_rules! lo prohíbe para fragmentos `literal`; solución: `:tt`). |
| 15b | Exponer Bagging/Stacking/DynamicEnsemble | ✅ hecho | `3844173` | Diseño: base learners seleccionados por id de string (mismo registry que `learner_from_id`, expuesto a Python como `registered_learner_ids()`) en vez de aceptar un objeto learner Python ya construido — evita puentear una pyclass arbitraria hacia el closure `Fn() -> Box<dyn Learner>`, que requeriría reacquirir el GIL en cada bootstrap/fold. Ids validados en el constructor (no en fit), con mensaje de error que lista los ids válidos. `Bagging`/`Stacking` declaran classif+regress (genéricos según el base elegido); `DynamicEnsemble` solo classif (así es DES/KNORA-E en sí). Verificado con smoke test real: fit/predict/predict_proba en los 3, más los 2 caminos de error (id inválido, lista vacía) devolviendo `PyErr` claro en vez de panicar. Con esto los 26 learners de Rust son alcanzables desde Python. |
| 15c | get_params/set_params | pendiente | | |
| 15d | Dividir `smelt-py/src/lib.rs` (ahora 2500+ líneas) | pendiente | | |
| 16d | Parquet/Arrow, f32 en histogramas, sparse data | pendiente | | `f32` en histogramas es un cambio de precisión numérica, no una "ganancia rápida" — requiere re-validar todos los tests de referencia de los 3 motores de boosting. |

## Log

### 2026-07-02 — Ganancias rápidas (16a, 16b, 16c, 17a)

Ejecutadas en orden: medidas → registry → parallel predict → docs. Cada una
compilada y testeada en debug y release antes de commitear (patrón establecido
en Fase 0-2). Suite completa verde en todo el lote: 55 lib tests (52→55 con
el registry) + 267 integration tests (259→267 con las medidas) + 61 doctests.

`smelt-py` compiló en cada paso relevante (medidas y registry lo tocan;
parallel predict y docs no).

Evaluación de `missing_docs` (17b): se activó temporalmente el lint, se contó
el número de advertencias (308), y se revirtió sin commitear — documentado
como pendiente en vez de forzarlo en una sesión que no le puede dedicar el
tiempo necesario.

### 2026-07-02 — Ítem 15a: macro define_learner! + 10 learners

Diseñada la macro comparando el patrón manual existente (Ridge, RandomForest)
y las dos macros ya presentes en el archivo (`add_explain_methods!`,
`declare_support!`) para mantener consistencia de estilo. Los 10 candidatos
se eligieron por tener hyperparámetros escalares simples (usize/f64/u64);
los 4 restantes (Bagging/Stacking/DynamicEnsemble/ObliqueForest) quedaron
fuera por necesitar una factory de base-learner sin equivalente en Python
hoy — mismo motivo por el que quedaron fuera del model registry Rust (16b).

Validación: `cargo build` (debug, encontró el bug del `:literal`) → fix →
`cargo build`/`cargo build --release` limpios → `maturin develop --release`
→ smoke test real en Python contra los 13 casos (10 learners, 3 de doble
capacidad probados en ambos modos) cubriendo fit/predict/predict_proba/
supports_classification/supports_regression/feature_importances_, más una
verificación aparte de permutation_importance (código compartido vía
add_explain_methods!) en dos de los nuevos learners.

### 2026-07-02 — Fix: ObliqueForest mal excluida (registry Rust + 15a)

Al diseñar 15b se notó que `ObliqueForest` tenía exactamente la misma forma
que los demás candidatos de `define_learner!` (Default + builders escalares:
n_estimators, max_depth, n_projections, seed) — no necesitaba factory de
base-learner como Bagging/Stacking/DynamicEnsemble. Se había excluido tanto
del registry Rust (16b) como de los bindings Python (15a) por generalizar
mal la razón de exclusión de esos tres. Corregido en ambos lados en un
mismo commit (`8b813c7`), verificado con test de Rust actualizado
(`factory_based_ensemble_ids_are_not_registered` ya no incluye
`oblique_forest`) y smoke test en Python.

### 2026-07-02 — Ítem 15b: Bagging/Stacking/DynamicEnsemble

Con el registry Rust ya corregido y completo, la pieza que faltaba era
solo decidir cómo pasar el "base learner" desde Python. Se descartó la
opción de aceptar un objeto Python (requeriría un wrapper `Learner` que
reacquiera el GIL en cada closure invocation — mucho más lento y mucho más
código) a favor de reusar los ids de `learner_from_id`, validados en el
constructor. Verificado con `maturin develop --release` + smoke test
cubriendo los 3 wrappers, ambas capacidades de Bagging (classif/regress), y
los 2 caminos de error.

Con esto, los 3 sub-ítems de "cerrar los learners no expuestos" de la Fase
3 (15a + fix + 15b) quedan completos: **26 de 26 learners de Rust son
alcanzables desde Python** (aunque `Bagging`/`Stacking`/`DynamicEnsemble`
seleccionan su base por id en vez de por objeto). Quedan 15c
(get_params/set_params) y 15d (dividir lib.rs) del ítem 15, y el ítem 14
completo (categóricas + NaN) como los dos grandes pendientes de Fase 3.
