# Fase 1 — Honestidad de lo anunciado

Seguimiento de avance. Referencia: `docs/auditoria_motor_2026-07-01.md` (sección "Fase 1").
Fase 0 (fixes quirúrgicos de correctness) completa y pusheada — ver `docs/fase0_progreso.md`.

| # | Tarea | Estado | Commit | Notas |
|---|-------|--------|--------|-------|
| 1 | Serialización: agregar XGBoost/LightGBM/CatBoost/etc + versionado de formato | ✅ hecho | `4cdc571` | 8 variantes agregadas + wrapper {format_version, smelt_version, model}. 2 tests nuevos (round-trip + rechazo de versión incompatible). |
| 2 | XGBoost: sample_weight en clasificación (implementar o Err) | ✅ hecho | `d5482af` | Implementado (no Err) para train_binary y train_multiclass: initial log-odds/log-priors ponderados, grads/hess escalados, early-stop loss ponderado. 3 tests nuevos. |
| 3 | XGBoost: early stopping con validation set (no train loss) | ✅ hecho | `4d5814d` | `with_eval_set_regress`/`with_eval_set_classif`. Sin eval_set, comportamiento igual que antes (train-loss). Test: modelo deliberadamente sobreajustable generaliza >10% mejor con eval_set real. |
| 4 | LightGBM: GOSS con pesos de amplificación reales | ✅ hecho | `6a10a3d` | goss_sample ahora scatter-ea los pesos a un array de largo n indexado por sample id original (antes se descartaban en los 3 call sites). 2 tests nuevos (suma ponderada insesgada + top-gradientes siempre seleccionados). |
| 5 | LightGBM: leaf-wise real o degradar docs + limpiar código muerto | ✅ hecho | `efea446` | Reescrito con arena: la búsqueda best-first ya no se descarta al final. `build_recursive`/`find_best_split_hist` (código muerto tras el cambio, ~200 líneas) eliminados. Test: leaf count exacto = num_leaves para varios valores. Suite completa verde (37+259+61). |
| 6 | Causal forest: honest splitting real (recompute leaf tau con est_idx) | ✅ hecho | `0603232` | `populate_leaf_tau` implementado de verdad (era no-op). `honest_valid` reemplaza campos muertos n_treated/n_control. Leaves sin tratado+control honesto se excluyen del voto de ese árbol. 3 tests nuevos. Suite completa verde (40+259+61). |
| 7 | Causal forest: CI válidos (infinitesimal jackknife) | ✅ hecho | `8c4c68f` | IJ estimator (Wager-Hastie-Efron / Wager-Athey) SIN corrección bootstrap (esa corrección asume with-replacement; con subsampling sin reemplazo colapsaba a 0). Estimador sin corregir: siempre ≥0, sesgo conservador conocido a B finito que no desaparece. 2 tests: SE no colapsa como 1/sqrt(B) al aumentar n_estimators 15x. Suite completa verde (42+259+61). |
| 8 | SHAP: renombrar o implementar permutation-SHAP + test de eficiencia | ✅ hecho | `60f356d` | Permutation-SHAP real (Štrumbelj & Kononenko 2010). Bug encontrado en desarrollo: resamplear bg por permutación rompía la eficiencia (residuo MC ~7 unidades) — corregido iterando sobre el mismo bg_indices usado para base_value. También arregla background = primeras N filas → muestra aleatoria. 3 tests (eficiencia regresión+clasificación, bg no es solo las primeras filas). Suite completa verde (45+259+61). |

## Fase 1 — COMPLETA (8/8)

Todos los ítems de "honestidad de lo anunciado" implementados, testeados y commiteados individualmente. **Nada pusheado a origin/master todavía** — pendiente de decisión del usuario.

## Log
