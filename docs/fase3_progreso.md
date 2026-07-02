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
| 16b | Model registry (`learner_from_id`) | ✅ hecho | `67f45c4` | 20 de 26 learners registrados (todos los que tienen `Default` sensato + QuantileGB con tau=0.5). Excluidos deliberadamente: Bagging/Stacking/DynamicEnsemble/ObliqueForest (necesitan factory de base learner) y GeoXGBoost (necesita coords externas). Solo Rust por ahora — exponerlo a Python requiere despachar a las pyclasses existentes, no trivial, queda pendiente. |
| 16c | Predict paralelo consistente | ✅ hecho | `14f318c` | XGBoost y CatBoost tenían Regression/BinaryClassif paralelos con rayon pero MultiClassif serial; LightGBM tenía los 3 modos seriales. Ahora los 3 motores usan el mismo patrón `into_par_iter()` en los 3 modos. Sin cambios numéricos, solo el loop por fila. |
| 17a | README.md / CLAUDE.md al día | ✅ hecho | `8f36278` | Versión 0.6→1.3, conteo de learners 21→26, tabla de learners/medidas/resampling completada, roadmap de CLAUDE.md (Phase 1-6, todo marcado como pendiente pese a estar hecho hace tiempo) reconciliado con el estado real. |
| 17b | `#![warn(missing_docs)]` | ❌ evaluado, no ejecutado | | 308 advertencias al activarlo — demasiado para tratarlo como ganancia rápida. Queda como ítem grande para una sesión dedicada (escribir ~300 doc comments o decidir cuáles APIs realmente necesitan documentación pública vs volverlas `pub(crate)`). |
| 14 | Categóricas + NaN en Task/splits; early stopping real; monotone constraints; objetivos custom | pendiente | | El ítem más grande del proyecto — toca `Task`, `CsvLoader`, `histogram.rs`, y el split-finding de XGBoost/LightGBM/CatBoost a la vez. Requiere su propia sesión de scoping antes de empezar. |
| 15 | Python: macro `define_learner!`, cerrar 14 learners no expuestos, get_params/set_params, dividir lib.rs (1800+ líneas) | pendiente | | Bien acotado pero repetitivo; buen candidato para trabajo incremental ítem por ítem como Fase 0-2. |
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
