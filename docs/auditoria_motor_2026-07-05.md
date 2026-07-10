# Auditoría del Motor — Smelt (3ª auditoría completa)

**Fecha**: 2026-07-05
**Reviewer**: Claude Code (5 revisores paralelos: motores boosting/árboles, módulos nuevos post-07-04, módulos estadísticos, arquitectura/API, bindings Python)
**Scope**: todo `src/` (~26.8k líneas), todo `smelt-py/` (~4.3k líneas Rust + capa Python pura), config, tests, docs, coherencia post-release 2.0.1/0.5.1
**Referencia**: auditorías previas `docs/auditoria_motor_2026-07-01.md` (6 CRITICAL / 16 HIGH) y `docs/auditoria_motor_2026-07-04.md` (2 CRITICAL / 11 HIGH) + fases de remediación A/B/C completadas
**Estado de tests**: suite completa verde (514 tests: 153 lib + 287 integración + 4 parquet + 70 doctests + suite smelt-py), 0 fallos. Clippy lib: 9 warnings menores.
**Método**: lectura línea por línea + confirmación empírica de todo CRITICAL/HIGH (3 sondas Rust compiladas contra el crate en release, goldens vs scipy 1.16.3 / sklearn 1.8.0, ~50 casos ejecutados contra el `.so` real de smelt-py).

## Resumen

- **Nuevos: 0 CRITICAL | 6 HIGH | ~18 MEDIUM | ~30 LOW**
- **De la auditoría anterior**: los 10 fixes de Fases B/C verificados uno a uno están **genuinamente cerrados** (Wilcoxon exacto, golden measures, PCA, ADASYN, DES/DSEL, EBM Err, RSF fit+OOB, Relief, CausalForest OOB, R-learner ponderado) — sin "a medias" esta vez, con **una excepción grave de proceso**: LightGBM `subsample` fue declarado resuelto en el propio doc del 07-04 y nunca se tocó (HIGH-1). El fix del CRITICAL-1 (histograma stale de XGBoost) se re-verificó a fondo: cerrado de verdad.
- **Patrón dominante de esta ronda**: (1) por primera vez **cero CRITICAL** — el núcleo numérico de los motores está sano; (2) los HIGH se concentran en **funciones auxiliares sin golden test** (`chi_squared_cdf`, KMeans — exactamente donde la regla "sin test de referencia, hay bug" predijo) y en **cara pública desfasada** (README, benchmark Python roto para los learners GIS); (3) los módulos nuevos de Prioridad 3/4 salieron notablemente limpios — Mondrian sobrevivió la sonda más agresiva (online=batch verificado con 15 seeds) y los bindings post-audit cumplen el estándar GIL/validación al 100%.

---

## Hallazgos CRITICAL (nuevos)

Ninguno.

---

## Hallazgos HIGH (nuevos)

### [HIGH-1] LightGBM `subsample` sigue aceptado e ignorado — y el audit 07-04 lo declaró resuelto sin serlo
- **Archivo**: `src/learner/lightgbm.rs:68, 90, 147-151`
- El campo, su default y `with_subsample()` existen; cero usos en el entrenamiento (el único sampling de filas es GOSS). **Confirmado por sonda**: `subsample=1.0` vs `0.05` → predicciones bit a bit idénticas (`max |Δ| = 0`). `git log -S subsample` confirma que ningún commit lo tocó.
- **Agravante de proceso**: `auditoria_motor_2026-07-04.md` (Fase B, ítem 7) dice "LightGBM `subsample` se resolvieron ya en el ítem 11/Fase A". Es falso. Cae en la categoría "declarado cerrado pero no lo está = HIGH automático" que ese mismo doc estableció.
- **Fix**: bagging de filas real (mutuamente excluyente con GOSS, como el oficial) o eliminar el builder con deprecación; corregir la línea del doc 07-04.

### [HIGH-2] `chi_squared_cdf` colapsa para chi² ≳ 300 — McNemar y Friedman reportan "no significativo" justo cuando la evidencia es abrumadora
- **Archivo**: `src/stats.rs:570-593` (`lower_incomplete_gamma`, serie truncada a 200 términos)
- La serie necesita ~x términos para converger; con el cap de 200, para `x = chi2/2 ≳ 150` → cdf≈0 → p≈1. **Confirmado con sonda + scipy**: McNemar b=400,c=0 (test set ~10k, 4% discordante) → smelt p=0.467, `significant=false`; scipy = 1.5e-88. Friedman k=3, n=200 → p=0.481; true 1.4e-87. La degradación empieza en chi2≈300 (62 órdenes de magnitud de error). Bonus: p **negativos** (−8.9e-16) en chi2 intermedios, sin clamp.
- **Fix**: switch serie/fracción continua estándar (Numerical Recipes: serie si x<a+1, Lentz si x≥a+1) + `clamp(0,1)` + golden vs scipy en chi2 ∈ {1, 10, 100, 400, 1000}.

### [HIGH-3] KMeans falla en el 38% de las semillas sobre 3 blobs trivialmente separables (sklearn: 0%)
- **Archivo**: `src/cluster/mod.rs:146-209`
- Init aleatoria simple (no k-means++), `n_init=1`, y un cluster vacío conserva su centroide obsoleto para siempre. **Confirmado por sonda**: 3 blobs separados por >8σ, 50 semillas → 19/50 clusterings incorrectos, 2 de ellos con solo 2 clusters no vacíos mientras `ClusterResult.n_clusters` sigue reportando 3. KMeans tenía **0 tests** — el patrón "sin test de referencia, hay bug" otra vez.
- **Fix**: k-means++ hand-rolled (precedente `CsrMatrix`), `n_init` con mejor inercia, re-seed de clusters vacíos, `n_clusters` reales; golden test de recovery de blobs multi-seed.

### [HIGH-4] ELM: fits silenciosamente inútiles con features en escalas realistas (sin escalado interno ni advertencia)
- **Archivo**: `src/learner/elm.rs:155-160` + docs del módulo
- Pesos de entrada `U(-1,1)` presuponen inputs normalizados (Huang et al. normalizan todo); con magnitudes del nicho GIS (UTM, metros) la sigmoide satura y la capa oculta degenera. **Confirmado por sonda**: `y=2x+1` exacto con `x∈[0,10⁴]` → RMSE relativo 0.85; escalado a `[0,1]` → 0.0026 (factor ~330×). Ningún fixture del módulo ejercita features fuera de [0,10].
- **Fix**: estandarizar internamente en fit (guardar mean/std en `TrainedELM`, costo cero) — lo que hacen las implementaciones serias; test de regresión con features en escala 1e4.

### [HIGH-5] README publicado con 2.0.1 tiene cifras y catálogo falsos
- **Archivo**: `README.md:10, 229-258, 336-337`
- Dice "26 supervised learners" (hay 33), "registry con 21" (hay 26 ids), "300+ tests" (hay 514). La tabla omite los 7 learners que el propio CHANGELOG 2.0.0 anuncia (KrigingHybrid, ARF, MondrianTree/Forest, ELM, DeepForest, CostSensitiveClassifier), los 5 causal meta-learners e IsolationForest. Es la cara pública en crates.io del release mayor.
- **Fix**: actualizar tabla + cifras; idealmente test que compare `registered_learner_ids().len()` contra el número citado.

### [HIGH-6] `benchmark()` de Python crashea entero con los learners espaciales (KrigingHybrid/GeoXGBoost)
- **Archivo**: `python/smelt/benchmark.py:88-97`
- `TypeError: fit() missing 1 required positional argument: 'coords'` no capturado (el `except RuntimeError` no lo cubre) → se pierde el run completo, incluidos los demás learners. `benchmark()` acepta `coords` pero solo lo usa para el splitter, nunca lo pasa a `fit`. **Verificado contra el `.so` real.** Los dos learners diferenciadores del nicho GIS son exactamente los que no se pueden benchmarkear.
- **Fix**: pasar `coords[train_idx]`/`coords[test_idx]` a fit/predict cuando el learner lo requiere; mínimo, ampliar el except a `(RuntimeError, TypeError)` y marcar `_error`.

---

## Verificación de hallazgos previos (consolidada)

### Fixes de Fases A/B/C verificados genuinamente cerrados
CRITICAL-1 07-04 (histograma stale XGBoost — predicado `left/right_is_trivial` re-analizado camino por camino, test discriminante presente), ObliqueTree n_classes, QuantileGB τ/predict, Wilcoxon exacto (única divergencia vs scipy: tolerancia de empates 1e-15, defendible — LOW doc), golden measures + fix N10, PCA init aleatoria + Gram-Schmidt fallback (golden vs sklearn), ADASYN k-NN, DES/DSEL split real, EBM `Err` multiclase, RSF `fit()`+OOB (3 tests), Relief RReliefF (golden numpy), CausalForest OOB (`in_bag` excluido, test outlier 1e6), R-learner ponderado (replicación T̃², clip 0.05, `oof_propensity` → Err), GIL 2ª mitad (verificado también en los bindings post-audit, contador concurrente), bootstrap_ci → Result, measures Python sin cast saturante, parse_coords NaN, errores tipados (4 `Other` restantes = los documentados), rayon Fase C (determinista por diseño, tests ejecutados), fixes de KrigingHybrid Fase A (sanos, con la salvedad MEDIUM-8 de escala).

### Declarado cerrado pero no lo está
| Ítem | Residuo | Evidencia |
|---|---|---|
| LightGBM `subsample` (HIGH-2 del 07-04) | Nunca se implementó; el doc 07-04 Fase B lo da por resuelto | HIGH-1 arriba, sonda bit a bit |

### Siguen abiertos (sin cambios desde 07-04 o antes)
| ID | Tema | Evidencia actual |
|---|---|---|
| M4-M7 | CatBoost TS multiclase por índice; gain≤0 aceptado; LightGBM λ=0 sin clamp; GBM sin paso Newton | `catboost.rs:820-829, 352-430`, `lightgbm.rs:85, 688-691`, `gradient_boosting.rs:318-420` |
| M8 | `with_truth_classif` no-op (documentado, no corregido); `with_truth_causal` replica | `prediction/mod.rs:72-124` |
| M10 | `ParamSet = HashMap<String, f64>` | `tuning/mod.rs:17` |
| M13/M17/M21 | Filter/RFE fallback silencioso; Clone de RFE panica; trait Filter no enchufable | `filter.rs:220-244`, `rfe.rs:80-104` |
| M14 | QRF pooling, cita a Meinshausen intacta | `quantile_forest.rs:6, 242-258` |
| M18 | Smote/Adasyn fuera del Pipeline (requeriría etapa "resampler", no un impl trivial) | `preprocess/` |
| M19 | Python: dtype f64 críptico, heurística int→classif silenciosa, PyRuntimeError para casi todo (ahora sería trivial mapear `InvalidParameter`→ValueError con los errores tipados de Fase C) | verificado contra el `.so` |
| M20 | Versionado: triple fuente sincronizada en 0.5.1 pero sin single-source (fix: `dynamic = ["version"]` + `importlib.metadata`); path dep sin `version` | `pyproject.toml`, `smelt-py/Cargo.toml:14` |
| HIGH-7 orig | GeoXGB LOO omite vecindarios <3; docstring sigue invirtiendo el sentido del sesgo | `geo_xgboost.rs:193-237` |
| N1 | alpha (L1) no afecta gain; caminos monotone divergen con alpha>0 | `xgboost.rs:461-479`, `hist_pool.rs:73, 98-105` |
| N2 | AdaBoost lr fuera del paréntesis; err==0 re-entrena el mismo stump (elevado a MEDIUM-3 con caso concreto) | `adaboost.rs:234-241` |
| N4/N5 | Hoeffding: cota en bits vs entropía en nats (ε ×1.44); tie-break permite gain 0 → crecimiento por ruido sin cota (se activa desde n≈80k/hoja con defaults) | `hoeffding.rs:170, 173, 328-344` |
| N6/N7 | ARF sin subespacio aleatorio de features; ADWIN max_window=200 solo detecta Δerror ≳ 0.34 (el doc nuevo documenta el costo, no el piso de sensibilidad) | `adaptive_rf.rs:57, 154-162` |
| N11 | C-index cuenta empates en ambas direcciones → arrastre a 0.5 (el fix RSF de Fase C no tocó `concordance_index`) | `survival/mod.rs:80-93` |
| N16 | Heurística classif de `benchmark()` diverge de `fit()` → TypeError a mitad de loop | `benchmark.py:58` |
| N17 parcial | Prelude sin conformal/multilabel/multioutput/benchmark_design (survival sí se agregó); EBM ausente del registry sin doc | `lib.rs:68-114`, `registry.rs:16-27` |
| LOWs 07-04 | HistBins 255 bins (verificado benigno: sin colisión con NAN_BIN); categóricas ≥254 train/predict inconsistentes; log-odds −∞; QRF panics de borde; Nemenyi q fallback; MAPE/n total; LogLoss/AUC panican; Chains panican; iForest reemplazo/max_samples; conformal zip trunca; clip 1e-3 DR/X-learner (R-learner sí subió a 0.05); BO factory duplica registry **y no conoce ningún learner nuevo** (brecha creciente); sin `__repr__`/`.pyi`; ExtraTrees/DT sin predict_proba en Python; KrigingHybrid ctor no valida variogram | verificados uno a uno |

---

## Hallazgos MEDIUM (nuevos)

| # | Hallazgo | Archivo | Fix |
|---|----------|---------|-----|
| M-1 | `with_max_features_fraction(0.0)` cambió de semántica en silencio en 95a0ef0: antes "usa sqrt(p)" (documentado), ahora "1 feature por split" — confirmado por sonda (RMSE 0.613 vs 0.396); ET además no ganó `with_max_features_sqrt()` pese al commit message, así que su única vía a sqrt desapareció | `tree/mod.rs:102`, `extra_trees.rs` | validar `f∈(0,1]` o mapear ≤0→Sqrt con deprecación; añadir `ExtraTrees::with_max_features_sqrt()` |
| M-2 | QuantileForest quedó incoherente con 95a0ef0: único bosque de regresión con sqrt(p) hardcodeado, sin override | `quantile_forest.rs:142` | reutilizar `MaxFeatures` |
| M-3 | AdaBoost: umbral "can't improve" con typo (`1e-10/K` ≈ 1.0 en vez de `1−1/K`); con lr<1 multiclase acepta stumps con err≈0.95 y voto positivo | `adaboost.rs:234-241` | fórmula sklearn + break con err==0 |
| M-4 | `sign_test`: overflow de binom_coeff → NaN → `NaN.min(1.0)` = **p=1.0 silencioso** para n≳1030 (scipy: 3.5e-11); ruta realista: pérdidas pareadas por muestra en test set grande | `stats.rs:289-291, 627-637` | log-espacio (ln-gamma) o aproximación normal; golden vs scipy |
| M-5 | `silhouette_score`: singletons puntúan 1.0 (sklearn: 0) → infla el score justo al barrer k alto para seleccionarlo; sin singletons coincide con sklearn a 1e-10 | `cluster/mod.rs:51-79` | contribución 0 para a_count==0 |
| M-6 | MondrianForest: las "probas" son fracciones de voto duro (cuantizadas a 1/n_trees), no el promedio de distribuciones del paper — degrada AUC/Brier/conformal/cost-sensitive aguas abajo | `mondrian.rs:666-696, 858-879` | promediar las `probs` por árbol (el dato ya se calcula y se descarta) |
| M-7 | DeepForest: doc promete augmentación con "every previous layer"; el código (correctamente, como el paper) concatena solo la capa anterior — el doc y CLAUDE.md describen otra arquitectura | `deep_forest.rs:19-21, 292` | corregir docs |
| M-8 | KrigingHybrid: umbral absoluto `sill<1e-9` desactiva la corrección en silencio con targets pequeños (geoquímica ~1e-5 → sill ~1e-10); mismo problema de escala en el pivote 1e-10 del solver | `kriging_hybrid.rs:212, 270` | umbral relativo a la varianza del target |
| M-9 | DeepForest: fold-tasks internos con `::new` sin propagar ancho de clases — mismo patrón que Stacking arregló en Fase B; verificado que NO corrompe (índices estables), pero asimetría de criterio | `deep_forest.rs:254` | propagar `class_names` como `stacking.rs` |
| M-10 | Ningún learner nuevo es serializable y el doc de serialize.rs lo enmascara: Mondrian/ELM/ARF/Hoeffding/Oblique/EBM/QuantileGB son self-contained (serializables con derive) pero no derivan ni `Serialize`; la nota de exclusión solo menciona los de trait objects | `serialize.rs:30-80` | derivar + variantes para los concretos; ampliar nota |
| M-11 | EBM registrable pero ausente del registry sin doc (cumple el mismo criterio que ObliqueForest, citado como contraste); KrigingHybrid tampoco documentado como excluido | `registry.rs:16-27` | registrar `"ebm"` + doc |
| M-12 | Cambio de default RF/ET en release **patch** (2.0.1) — cambia resultados numéricos de usuarios existentes; defendible como bug-fix, precedente a no repetir | `95a0ef0` | anotar convención: minor + aviso |
| M-13 | Prelude incompleto: conformal, multilabel/multioutput, benchmark_design fuera (README los muestra como primera clase) | `lib.rs:68-114` | re-exports |
| M-14 | Archivos de modelo 1.3.0 fallan con `Json("missing field format_version")` opaco en vez del mensaje de versión que el envelope promete | `serialize.rs:170-188` | mensaje explícito "formato pre-2.0 no soportado" |
| M-15 | ARF/HoeffdingTree batch (la única API en Python) con n ≤ grace_period=200 → modelo a nivel de azar en silencio (verificado: 0.45 acc vs 0.833 RF en folds de 80) | `smelt-py trees.rs:174, 192` | documentar + clamp `grace_period≈n/10` en fit batch, o exponer streaming API |
| M-16 | RandomSearch Python `use_proba=True`: `except Exception` degrada a hard labels y computa AUC degenerado sin aviso (combinación realista: ExtraTrees sin predict_proba) | `tuning.py:117-123` | capturar solo AttributeError + warning |
| M-17 | `resolve_measure` no conoce balanced_accuracy/kappa/mcc/brier como strings — `optimize(metric="mcc")` falla aunque la función exista | `smelt-py common.rs:228-243` | agregar las 4 al match |
| M-18 | Causal learners Python: `treatment=-1` → `OverflowError` críptico de PyO3 antes de cualquier validación | `causal.rs:52` | `Vec<i64>` + mensaje como `extract_class_labels` |

## Hallazgos LOW (nuevos, selección)

- **Boosting**: `eval_set` sin `early_stopping_rounds>0` se evalúa y descarta en los 3 motores (costo silencioso); XGBoost multiclase reutiliza el mismo subsample para los K árboles de la ronda; smelt-py fuerza `max_depth=6` en LightGBM cuando el default Rust/oficial es sin límite (cambia el modelo por default vs API Rust); XGBoost mantiene la lógica de early stopping inline ×3 cuando `EarlyStopper` (eval.rs) ya existe y LightGBM/CatBoost lo usan.
- **Streaming/nuevos**: ARF nunca descarta el background tree si el warning es falsa alarma (el paper sí); Mondrian/ARF no re-seedean entre `train_*` repetidos; Hoeffding sigue descartando labels ≥ counts.len() (mondrian.rs sí lo resuelve con resize — patrón disponible en el crate); `best_threshold` = media global (subóptimo con clases desbalanceadas); ELM re-hace la eliminación gaussiana por columna de salida (O(k·h³)) y `solve_spd` sin unit test propio; DeepForest predice con forests re-entrenados sobre todo el train (el oficial promedia los modelos de fold — defendible, no documentado); "ExtraTrees ≠ completely-random forest" del paper (aproximación no documentada).
- **Estadísticos**: Wilcoxon tolerancia de empates 1e-15 (documentar); McNemar usa chi² incluso con b+c<25 (scipy usa binomial exacto); measures derivan n_classes de max(label)+1 → clases fantasma con labels con huecos; Chains panican con filas ragged además de lista vacía y las métricas multilabel truncan por zip; Imputer con columna 100% NaN imputa 0.0 en silencio; Relief panica con NaN (no pasa por check_no_nan); X-learner conserva `unwrap_or(0.5)` (hoy inalcanzable, anti-patrón documentado); CausalForest: puntos sin árbol OOB entran al ATE con estimate=0.0; ADASYN puede dejar la clase sin balance exacto por redondeo, en silencio.
- **Arquitectura/tuning**: error no determinista con múltiples candidatos fallidos en los tuners paralelos; `select_best` deja ganar a un NaN; falta comentario en benchmark.rs explicando los folds seriales; CHANGELOG 2.0.0 no declara `Resample: Send+Sync` como breaking; `use` colgando al pie de benchmark_design.rs; `clippy --all-targets` ~50 warnings.
- **Python/empaquetado**: `n_trees=0`/`cv_folds=0` clampean en silencio; `__pycache__/*.pyc` trackeados en git; pyproject sin `readme`/`classifiers` (página PyPI pobre); `is_classif` variable muerta en tuning.py.

---

## Calidad de suites de los módulos nuevos

- **mondrian.rs**: la mejor del lote — test central genuinamente discriminante (falla contra crecimiento solo-hacia-abajo), property tests de las distribuciones, edge cases. Falta: test online-vs-batch de la propiedad de consistencia (la sonda de esta auditoría es casi copy-paste).
- **cost_sensitive.rs**: el test estrella usa `>=` entre conteos — **pasa idéntico si el wrapper fuera un no-op**. La sonda confirmó que el comportamiento real existe (40/40 flips de frontera con costo 20×), pero la suite no detectaría una regresión silenciosa. Fix: `assert!(flips > 0)`.
- **elm.rs**: smoke; sin golden del solve, sin test de `regularization`, y todos los fixtures en [0,10] — el gap que tapó HIGH-4.
- **deep_forest.rs**: media; early stopping discriminante de verdad, pero nada verifica que las probas de augmentación sean OOF y no in-sample.
- **KMeans**: 0 tests → HIGH-3. **DBSCAN**: sin goldens en el repo pero verificado en esta auditoría vs sklearn (labels idénticos, ARI=1.0) — consolidar ese fixture como golden test. `tests/real_benchmark.rs` sigue `#[ignore]`.

## Performance (estado)

- Sin regresiones nuevas salvo la esperada: RF/ET regresión ~9× más lento en fit con el default nuevo, y como TreeBuilder sigue O(n²·log n) por feature/nodo (abierto desde 07-01), **el fix de perf de TreeBuilder subió de prioridad**.
- Siguen abiertos: LightGBM leaf-wise re-escanea O(L²·F·B); loops de actualización de predicciones seriales en los 3 motores; predict serial en DT/GBM/QuantileGB; AdaBoost O(n²·p) por stump; los tuners Python puros sin el paralelismo rayon que Fase C dio al core.

## Paridad Rust↔Python

**Cerrado por 0.5.x**: todo lo nuevo entró con binding el mismo día (KrigingHybrid, Smote/SpatialSmote, ARF, MondrianForest, DeepForest, ELM, CostSensitiveClassifier, 5 causal meta-learners, DES dsel_fraction). Bindings post-audit ejemplares: GIL en fit Y predict al 100% (verificado con contador concurrente), cero PanicException en ~50 casos de borde, round-trips correctos (`lambda_`, `float('inf')`).
**Sigue sin exponer** (por dolor de usuario): **persistencia** (pickle falla — un modelo entrenado en Python muere con el proceso) > cluster/ > CsvLoader/ParquetLoader > streaming API (agrava M-15) > survival, multilabel/multioutput, CausalForest original, resto de preprocess, tuners Rust, ConformalClassifier/CQR, friedman/nemenyi/mcnemar, Pehe/AteBias, CsrMatrix.

---

## Plan priorizado

### Fase D — Correctness quirúrgico (días) — **COMPLETADA 2026-07-05**
1. ✅ **HIGH-2** `chi_squared_cdf` reemplazada por `chi_squared_sf` (survival
   function calculada directamente vía `Q(a,x)`, no `1.0 - cdf`): serie de
   Numerical Recipes para `x<a+1`, fracción continua de Lentz para `x>=a+1`,
   ambas en log-espacio vía `ln_gamma` (Lanczos) — el intento inicial de
   mantener `chi_squared_cdf` y restar de 1 fallaba igual por cancelación
   catastrófica (`1.0 - (1-3e-67) = 0.0` exacto en f64), de ahí el cambio a
   una función de cola dedicada. Goldens vs `scipy.stats.chi2.sf` en
   x∈{1,10,300,800}; McNemar con 400 discordantes reproduce 1.499e-88 (antes
   p=0.467, "no significativo"). **M-4** `binomial_cdf`/`sign_test`
   reescritos en log-espacio (`ln_binom_coeff` + log-sum-exp) — el
   `NaN.min(1.0)=1.0` que enmascaraba el overflow para n≳1030 ya no ocurre;
   goldens vs `scipy.stats.binomtest` en n=10 y n=1100. 16 tests en
   `stats.rs`, todos verdes.
2. ✅ **HIGH-3** KMeans: init k-means++ (Arthur & Vassilvitskii 2007) +
   `n_init=10` (mejor inercia) + reseed de clusters vacíos (roba el punto
   peor servido, en vez de dejar el centroide muerto) + `n_clusters` como
   conteo real de clusters no vacíos. Test de recuperación de 3 blobs en 30
   seeds (0 fallos; antes 19/50). **M-5** silhouette: singleton ahora
   contribuye 0 (no 1.0) — golden exacto vs `sklearn.metrics.silhouette_score`
   (0.5919185734900021). 7 tests en `cluster/mod.rs` (0 antes del fix).
3. ✅ **HIGH-4** ELM: estandarización interna (`standardize_fit`/
   `standardize_apply`, mean/std guardados en `TrainedELM`, aplicados
   idénticamente en predict). Test con `y=2x+1` exacto y `x∈[0,10⁴]`: RMSE
   relativo pasa de 0.85 (sin fix, confirmado por sonda) a <0.05.
4. ✅ **HIGH-1** LightGBM `subsample`: implementado bagging de filas real
   (`LightGBM::sample_rows`, aplicado antes de que GOSS muestree top/other
   sobre la población resultante — compone en vez de competir con GOSS,
   igual que el oficial). Test que confirma que `subsample=0.05` produce
   predicciones distintas de `subsample=1.0` (antes, bit a bit idénticas).
   Corregida la línea de `auditoria_motor_2026-07-04.md` que declaraba esto
   resuelto sin estarlo.
5. ✅ **M-1** `MaxFeatures::Fraction(f<=0.0)` ahora cae a `Sqrt` (preserva la
   semántica del sentinel pre-95a0ef0) en vez de degenerar a 1 feature;
   agregado `ExtraTrees::with_max_features_sqrt()` (faltaba pese a que el
   commit lo nombraba). **M-2** QuantileForest ahora usa `MaxFeatures`
   (default `Auto` → todas las features en regresión, igual que RF/ET) en
   vez de `sqrt(p)` hardcodeado; test que reproduce el mismo patrón de
   48-features/3-informativas del test de RF. **M-3** AdaBoost: umbral
   "can't improve" corregido a `err >= 1 - 1/K` (antes `1 - 1e-10/K`, casi
   inerte en la práctica per el techo `err <= 1-1/K` que la selección de
   mayoría local ya garantiza); `learning_rate` ahora escala el alpha SAMME
   completo (`lr*(ln((1-err)/err)+ln(K-1))`), no solo el primer término;
   stump perfecto (err=0) detiene el entrenamiento en vez de re-entrenar el
   mismo stump el resto de `n_estimators`. Tests con discriminador exacto
   (`lr=0.0` → alpha debe ser exactamente 0.0) y con separación perfecta.
   **M-8** KrigingHybrid: umbral de variograma degenerado ahora relativo a
   la varianza de los residuales (no absoluto `1e-9`) — permite sill~1e-10
   genuino en targets de escala geoquímica (~1e-5); el pivote del solver
   pasó a *scaled partial pivoting* (Golub & Van Loan) en vez de un umbral
   relativo al máximo global de la matriz, que fallaba falsamente en el
   sistema de kriging (bloque de semivarianza a escala del sill, bordeado
   por la restricción de Lagrange en escala O(1)). Test con residuales
   espacialmente estructurados a escala 1e-5: MSE con corrección < 50% del
   MSE base-only (antes, corrección idéntica a 0 por el umbral absoluto). No
   se tocó **M-6** (probas de Mondrian) — queda para la siguiente iteración.

### Fase E — Cara pública y Python (días) — **COMPLETADA 2026-07-05**
6. ✅ **HIGH-5** README: cifras corregidas (26→33 learners, 21→27 ids de
   registry, 300+→570+ tests), tabla "All Supervised Learners" ampliada de
   26 a 33 filas (agregadas KrigingHybrid, Adaptive Random Forest, Mondrian
   Tree, Mondrian Forest, Extreme Learning Machine, Deep Forest,
   Cost-Sensitive Classifier), Isolation Forest agregado a Unsupervised, los
   5 causal meta-learners agregados a Causal Inference, `smelt-ml = "1.3"`
   corregido a `"2.0"` en Quick Start (estaba dos majors atrás). Test de
   regresión nuevo (`registered_learner_count_matches_readme_claim`) que
   falla si el conteo del registry cambia sin actualizar el README.
   ✅ **M-11** EBM registrado en `learner_from_id`/`registered_learner_ids`
   (cumplía el mismo criterio que ObliqueForest); doc de exclusiones del
   registry actualizado para mencionar KrigingHybrid explícitamente.
   ✅ **M-13** prelude completo: agregados `benchmark_design`, `conformal`
   (+ `cqr`), `multilabel`, `multioutput` (`survival` ya estaba desde Fase C).
   ✅ **M-14** `load_json` sobre un archivo pre-2.0 (sin envelope) ahora falla
   con un mensaje explícito nombrando el formato legacy, en vez del
   `Json("missing field \`format_version\`")` opaco que el envelope
   pretendía evitar. Test de regresión con un archivo `to_json()` crudo (sin
   wrapper).
7. ✅ **HIGH-6** `benchmark()` de Python: los learners que requieren
   coordenadas (`GeoXGBoost`, `KrigingHybrid`, detectados por nombre de
   clase) ahora reciben `coords[train_idx]`/`coords[test_idx]` en
   `fit`/`predict`; sin `coords=` en la llamada a `benchmark()` se saltan
   limpiamente (`_skipped`) en vez de tirar `TypeError` sin capturar y
   abortar el benchmark completo. Verificado contra el `.so` real: los 3
   learners (KrigingHybrid, GeoXGBoost, RandomForest) benchmarkean
   correctamente con `coords=`, y sin `coords=` los espaciales se saltan
   mientras RF sigue funcionando.
   ✅ **M-15** HoeffdingTree/AdaptiveRandomForest: agregado un parámetro
   `note = "..."` al macro `define_learner!` (opcional, backward-compatible,
   se emite como `#[doc = ...]` en el `#[pyclass]` generado → aparece como
   `__doc__` real en Python) documentando que `grace_period=200` es un
   default de streaming inadecuado para batches chicos, con la
   recomendación concreta (`max(10, n // 10)`). Verificado:
   `HoeffdingTree.__doc__`/`AdaptiveRandomForest.__doc__` muestran el aviso.
   ✅ **M-16** `RandomSearch`/`GridSearch` con `use_proba=True`: el
   `except Exception` que degradaba a hard-labels en silencio ahora captura
   solo `AttributeError` (learner sin `predict_proba`) y emite
   `warnings.warn(...)` antes de degradar. Verificado con `ExtraTrees`
   (sin `predict_proba`): 1 warning por fold disparado.
   ✅ **M-17** `resolve_measure` (smelt-py `common.rs`): agregadas
   `balanced_accuracy`/`kappa`/`mcc`/`brier` como strings de métrica
   (existían como funciones pero no eran resolubles por nombre en
   `permutation_importance`/`BayesianOptimizer.optimize`). Verificado:
   `permutation_importance(X, y, metric="mcc")` ya no falla con "Unknown
   metric".
   ✅ **M-18** Causal meta-learners (`causal.rs`): `treatment` pasó de
   `Vec<usize>` (PyO3 lo convierte en el binding de argumentos, antes de que
   el cuerpo de la función corra — ahí es donde ocurría el
   `OverflowError: can't convert negative int to unsigned` para valores
   negativos) a `Vec<i64>` + validación explícita vía la nueva
   `extract_treatment_labels` (mismo patrón que `extract_class_labels`).
   Verificado: `treatment=-1` da `ValueError: negative treatment arm -1 at
   index 0; ...` en vez de `OverflowError` sin contexto.
   ✅ **M19 parcial**: `smelt_err` ahora mapea `SmeltError::InvalidParameter`
   y `SmeltError::DimensionMismatch` a `PyValueError` (antes, todo
   `SmeltError` — incluidos estos dos, ya tipados desde Fase C del audit
   07-04 — cayía en `PyRuntimeError` sin distinción). El resto de M19
   (dtype f64 críptico, heurística int→classif silenciosa) y N16
   (heurística de `benchmark()` diverge de `fit()`) quedan abiertos, fuera
   del alcance declarado de esta fase.
8. ✅ **M-10** Serialización de los 9 learners self-contained del 2026-07-04
   que quedaron sin variante en `SerializableModel` pese a no tener
   `Box<dyn TrainedModel>` internos: Mondrian Tree/Forest, ELM,
   AdaptiveRandomForest, HoeffdingTree, Oblique Tree/Forest, EBM,
   QuantileGB — más QuantileForest, que el propio doc de `serialize.rs` ya
   señalaba como "excluido pendiente de una variante dedicada" desde antes
   del 07-04. `#[derive(Serialize, Deserialize)]` en cada struct interno
   (nodos/stats — todos tipos simples, ninguno necesitó lógica de
   serialización custom) + 10 variantes nuevas en `SerializableModel` + doc
   de exclusiones corregido para nombrar solo lo que de verdad usa trait
   objects (Bagging, Pipeline, Stacking, GeoXGBoost, DeepForest,
   KrigingHybrid, DynamicEnsemble, CostSensitiveClassifier). Un test de
   round-trip end-to-end (`elm_roundtrips_through_save_load`) verifica el
   mecanismo completo save→load→predict; los otros 8 comparten el mismo
   patrón de derive+variante, verificado estructuralmente por la
   compilación exitosa del crate (el derive macro de serde falla en tiempo
   de compilación si algún campo no es serializable).

   Verificado: suite completa verde (202 lib + 292 integración + 4 parquet +
   75 doctests) tras cada sub-ítem; `smelt-py` recompilado con
   `maturin develop --release` y probado end-to-end contra el `.so` real
   para los 5 ítems de Python (HIGH-6, M-15, M-16, M-17, M-18).

### Gaps de esta pasada
`resolve_variogram_model`/`validate_learner_id` (smelt-py) siguen usando
`PyRuntimeError` para errores de validación de input en vez de
`PyValueError` — el mismo patrón que motivó M19-parcial, pero fuera del
alcance declarado (una limpieza de consistencia más amplia, no solo el
`smelt_err`/`SmeltError` que Fase E tocó). Fase F (estructurales) sigue sin
tocar: `ParamSet` stringly-typed, duplicación (scanner ×5,
train_binary/multiclass ×4, EarlyStopper en XGBoost), TreeBuilder O(n²),
N4-N7 streaming, M4-M7 boosting, N11 C-index, single-source de versión
smelt-py, M-6 (probas de Mondrian), N16 (heurística classif de benchmark()
Python), M19 restante (dtype/heurística int→classif).

   Verificado: suite completa verde (199 lib + 292 integración + 4 parquet +
   75 doctests), 0 fallos, clippy sin warnings nuevos (los 7 preexistentes
   intactos).

### Fase E — Cara pública y Python (días)
6. **HIGH-5** README al día (+ test de conteo vs registry). **M-11** registrar EBM. **M-13** prelude completo. **M-14** mensaje legacy en load_json.
7. **HIGH-6** benchmark() con coords. **M-15** grace_period batch. **M-16** use_proba. **M-17** measures como strings. **M-18** treatment i64. M19 parcial: mapear `InvalidParameter`→ValueError (trivial post-Fase C).
8. **M-10** serialización de los self-contained nuevos (derive + variantes).

### Fase F — Estructurales (iniciativa aparte, sin cambios de la lista 07-04)
9. Persistencia/cluster/loaders en Python; M10 ParamSet tipado; M18 resampler stage en Pipeline; duplicación (scanner ×5, train_binary/multiclass ×4, EarlyStopper en XGBoost); TreeBuilder O(n²) (subió de prioridad); ~~N4-N7 streaming~~; ~~M4-M7 boosting~~; ~~N11 C-index~~; single-source de versión smelt-py.

    **✅ N4-N7 (streaming) RESUELTOS 2026-07-05**:
    - **N4** Hoeffding bound en bits vs. entropía en nats
      (`src/learner/hoeffding.rs`): `entropy`/`entropy_weighted` usan `ln`
      (nats), pero el rango `R` del bound usaba `log2(n_classes)` (bits) —
      un factor `1/ln(2) ≈ 1.44` de más en epsilon, retrasando splits
      innecesariamente. Cambiado a `r = (n_classes as f64).ln()`.
    - **N5** El fallback de tie-break (`epsilon < 0.01`) forzaba un split
      incluso con `best_gain == 0.0` (sin exigir que hubiera *algo* de
      información) — una vez que una hoja acumula suficientes muestras para
      que epsilon caiga bajo ese umbral (~80k con el delta default), el
      árbol crecía sin límite aunque cada feature tuviera ganancia cero.
      Agregado `best_gain > 0.0` como precondición de cualquier split (tanto
      la vía del bound como la del tie-break). Test discriminante con
      features de varianza cero (ganancia exactamente 0, no solo pequeña
      por ruido de muestreo) y 150k muestras: el árbol debe quedar en 1 hoja.
    - **N6** `AdaptiveRandomForest` hacía que cada `HoeffdingTree` mirara
      *todas* las features en cada split — a diferencia de
      `RandomForest`/`ExtraTrees`, que restringen cada split a un
      subconjunto aleatorio (`MaxFeatures`), la diversidad del ensamble
      dependía solo del online bagging, no del subespacio de features.
      Agregado `HoeffdingTree::with_feature_subset` (nuevo campo
      `feature_subset: Option<Vec<usize>>`, `#[serde(default)]` para
      compatibilidad hacia atrás) que restringe qué features se trackean en
      `feature_stats` durante `update_node`; `AdaptiveRandomForest` sortea un
      subconjunto de tamaño `sqrt(n_features)` (el default de clasificación
      de RF/ET) por árbol al momento de la primera muestra (n_features solo
      se conoce ahí, como en `HoeffdingTree` mismo), y lo fija para ese slot
      del ensamble incluyendo sus árboles de background/reemplazo por
      drift. Test que verifica tamaño de subconjunto correcto, sin
      duplicados, y que al menos 2 de 8 árboles terminan con subconjuntos
      distintos (diversidad real, no coincidencia).
    - **N7** Piso de sensibilidad de ADWIN no cuantificado (el doc anterior
      documentaba el costo de `max_window=200` pero no qué tan chico puede
      ser un cambio detectable). Agregada la derivación explícita al doc de
      `Adwin`: con `max_window=200` y el delta de warning default (0.01),
      `epsilon ≈ 0.34` en el corte balanceado — un cambio sostenido menor a
      eso es invisible al detector en su configuración default, sin importar
      cuánto dure. Test dorado que confirma un salto de 0.15 (bajo el piso)
      no se detecta y uno de 0.6 (sobre el piso) sí.

    Verificado: suite completa verde (213 lib + 286 integración + 74
    doctests) tras cada sub-ítem.

    **✅ N11 (C-index) RESUELTO 2026-07-05**: `concordance_index`
    (`src/survival/mod.rs`) reescrito para iterar pares no ordenados `{i,j}`
    exactamente una vez (antes: `i` sobre eventos, `j` sobre todos, sin
    deduplicar) — el bug solo se manifestaba cuando dos sujetos no
    censurados compartían el mismo tiempo de evento exacto: ninguna
    dirección se saltaba, así que ese par contaba `total += 2` en vez de 1,
    sobre-pesando exactamente los pares "empate perfecto" (que promedian
    0.5) relativo a todos los demás pares correctamente contados una vez —
    arrastrando el índice agregado hacia 0.5 cuando los empates de tiempo
    eran comunes. Nueva semántica explícita para los 3 casos (ambos
    censurados → no comparable; ambos evento con tiempo empatado →
    comparable pero sin orden temporal, crédito 0.5; tiempos distintos → el
    más temprano debe ser evento). 3 tests nuevos, uno con valor esperado
    calculado a mano (`5.5/6` vs el `6/7` que daba el código viejo en los
    mismos datos). Suite verde (205→208 lib tests durante esta sub-fase).

    **✅ M4-M7 (boosting) RESUELTOS 2026-07-05**:
    - **M4** CatBoost `train_multiclass` (`src/learner/catboost.rs`) codificaba
      las features categóricas UNA sola vez usando el índice de clase crudo
      (0,1,2,...) como si fuera un target continuo — sin sentido para clases
      nominales (el "promedio del índice de clase" de una categoría no es una
      estadística interpretable). Reescrito para computar una codificación
      de target statistics independiente por clase (indicador binario
      one-vs-rest 1{clase==c}), con sus propios bins/histogramas y mapas de
      encoding — `TrainedCatBoost.cat_encodings`/`prior` pasaron de un único
      mapa/escalar a `Vec<...>` indexado por clase (1 elemento para
      Regression/BinaryClassif, `n_classes` para MultiClassif). Test
      discriminante: una categoría 50/50 mezcla de clase 0/2 y otra 100%
      clase 1 colapsaban al mismo valor (~1.0) bajo el esquema viejo
      (indistinguibles), y son perfectamente separables (recall >95% en
      clase 1) con el esquema nuevo.
    - **M5** CatBoost `build_oblivious_tree` siempre profundizaba `depth`
      niveles sin mirar el signo del gain — ahora corta el crecimiento si el
      mejor gain disponible en un nivel es `<= 0.0` (posible específicamente
      en árboles oblivious porque el split se fuerza sobre todas las hojas
      actuales a la vez, no elegido greedy por hoja). `leaf_weights` pasa a
      dimensionarse por `partitions.len()` real, no por `2^depth` fijo. Test
      con gradientes cero: el árbol debe quedar en 0 splits.
    - **M6** LightGBM (`src/learner/lightgbm.rs`): `leaf_weight`/el gain de
      split dividían por `h + lambda` sin piso; con `lambda=0.0` (default) y
      `min_child_weight=0.0` (no-default, pero configurable) un hijo/hoja de
      hessian ~0 producía NaN/Inf. Extraído un helper `split_gain` compartido
      (usado en los 3 sitios que antes duplicaban la fórmula inline) con
      `.max(1e-12)` en cada denominador — no-op bajo los defaults reales
      (`min_child_weight=1.0` ya mantiene hl/hr ≥ 1.0).
    - **M7** GradientBoosting (`src/learner/tree/gradient_boosting.rs`):
      `train_binary`/`train_multiclass` usaban el valor de hoja que el
      `TreeBuilder` de error-cuadrático ya computa (media de los residuos),
      sin el paso de Newton estándar (Friedman 2001, sec 4.6) que corrige
      cada hoja por la curvatura real de la pérdida
      (`sum(gradiente)/sum(hessiano)`, con hessiano `p(1-p)` para log-loss/
      softmax en vez de asumir hessiano 1 como en MSE). Nueva función
      `refit_leaf_newton` que reutiliza la ESTRUCTURA del árbol (los splits
      elegidos por error cuadrático) pero recalcula cada valor de hoja con
      el paso de Newton, evaluado en la predicción actual del ensamble antes
      de sumar el árbol nuevo. Regresión (MSE, hessiano=1 constante) no
      cambia — el paso de Newton degenera exactamente a la misma media. 2
      tests directos sobre `refit_leaf_newton` (valores esperados a mano,
      y hoja de hessiano ~0 que debe quedar sin tocar).

    Verificado: suite completa verde (210 lib + 286 integración + 74
    doctests) tras cada sub-ítem, sin recompilar `smelt-py` todavía (los
    bindings de CatBoost/LightGBM/GradientBoosting no tocan los campos
    internos cambiados — `cat_encodings`/`prior` son `pub(crate)`, no
    expuestos a Python).
10. **`SpatialBlockCV` conflates block size with fold count** — encontrado
    2026-07-05 al validar `XGBoost`+CV espacial contra datos reales de
    susceptibilidad de remociones en masa (cuenca Huasco, 686 muestras, ver
    `paper/replication/huasco_validation.py`). La API original
    (`SpatialBlockCV::new(n_folds, coords)`) deriva una grilla
    `ceil(√n_folds)²` a partir de `n_folds`, así que no había forma de pedir
    "bloques de 2 km" de forma directa e independiente del número de folds —
    había que traducir tamaño de bloque → n_folds a mano desde el extent, y
    esa traducción degenera en dominios no cuadrados o cuando el bloque es
    chico relativo al extent (ej. ~5000 folds para bloques de 2 km sobre un
    extent de 147×129 km). **✅ RESUELTO 2026-07-05**: nuevo constructor
    `SpatialBlockCV::with_block_size(n_folds, coords, block_size)`
    (`src/resample/spatial.rs`) que fija el lado de celda directamente
    (`floor((x-min_x)/block_size)`), independiente de `n_folds` — el modulo
    `cell_id % n_folds` sigue repartiendo celdas en folds, pero la
    resolución de grilla ya no depende de cuántos folds se pidan. Expuesto
    en `smelt-py` como parámetro opcional `block_size=None` en
    `SpatialBlockCV.__new__` (mismo binding, sin romper la firma anterior).
    2 tests de regresión nuevos en `tests/integration.rs`
    (`spatial_block_with_block_size_uses_fixed_cell_size_not_n_folds` —
    verifica la grilla exacta contra un cálculo a mano —, y el rechazo de
    `block_size <= 0`). Verificado end-to-end contra los datos reales de
    Huasco vía `SpatialBlockCV(n_folds=5, coords, block_size=...)` en
    `paper/replication/huasco_validation.py`: reproduce el mismo patrón de
    caída de AUC con bloques crecientes (0.92 en 2 km → 0.76 en 30 km) sin
    degenerar, usando 3-5 de los 5 folds en todos los tamaños de bloque
    probados. Suite completa verde tras el cambio (202 lib + 286 integración
    + 74 doctests) y `smelt-py` recompilado/verificado contra el `.so` real.

    **✅ Persistencia/cluster/loaders en Python RESUELTOS 2026-07-09**
    (primer sub-ítem de la lista de Fase F, punto 9): cierra el dolor de
    usuario #1 de "Paridad Rust↔Python" arriba (pickle falla — un modelo
    entrenado en Python muere con el proceso).
    - **Persistencia**: `TrainedModel::to_serializable() -> Option<SerializableModel>`
      nuevo (`src/learner/mod.rs`, default `None`), implementado en los 25
      tipos `Trained*` que ya tenían variante en `SerializableModel`
      (requirió agregar `Clone` a cada uno y a sus tipos de nodo internos —
      ninguno tenía campos que lo impidieran). `SerializableModel::type_name()`
      + `impl TrainedModel for SerializableModel` (`src/serialize.rs`) para
      poder re-boxear un modelo cargado como `Box<dyn TrainedModel>` sin
      reconstruir el tipo concreto. En `smelt-py`, `save()`/`load()` en
      todos los wrappers de learners con variante: vía `define_learner!`
      extendido (nuevo parámetro `serial_as`) para los 14 macro-generados,
      y una nueva macro `add_persistence_methods!` para los ~15
      hand-written (`common.rs`). `load()` valida el `model_type` del
      archivo contra el esperado (`load_model_checked`), así que
      `RandomForest.load("catboost.json")` falla con `ValueError` claro en
      vez de envolver silenciosamente el modelo equivocado bajo el nombre
      de clase incorrecto. `KNearestNeighbors` (única clase Python con 2
      variantes posibles, `KnnClassifier`/`KnnRegressor` según
      `is_classif`) tiene `save`/`load` a mano en vez de vía macro.
      `Ridge`/`Lasso`/`ElasticNet` comparten `RegularizedRegression` (una
      sola clase Rust detrás). Los compuestos que sostienen
      `Box<dyn TrainedModel>` internamente (Bagging, Stacking,
      DynamicEnsemble, CostSensitiveClassifier, DeepForest, GeoXGBoost,
      KrigingHybrid) reciben `save`/`load` igual por consistencia de API,
      pero siempre fallan con error claro ("no soporta serialización") en
      vez de quedar ausentes del API (`AttributeError`) — GeoXGBoost/
      KrigingHybrid quedan afuera de esto porque su campo `trained` es el
      tipo concreto (`Option<TrainedGeoXGBoost>`), no
      `Option<Box<dyn TrainedModel>>` (necesario para su `predict_spatial`
      inherente).
    - **Cluster**: `KMeans`/`DBSCAN`/`IsolationForest` bindeados
      (`smelt-py/src/cluster.rs`, nuevo) — no pasan por `Learner`/
      `TrainedModel` (sin `Task`), llaman `fit`/`fit_predict` directo y
      devuelven arrays numpy planos, mismo patrón que `Smote`/`SpatialSmote`.
    - **Loaders**: `CsvLoader` (siempre disponible) y `ParquetLoader`
      (`smelt-py/src/data.rs`, nuevo) detrás de un feature Cargo `parquet`
      propio de `smelt-py` (opt-in, espeja el feature de smelt-ml — evita
      forzar polars en un `maturin develop` normal). Devuelven
      `(x, y, feature_names)` en vez de un objeto `Task` (no existe uno en
      Python; todo `fit(x, y)` ya espera exactamente esa forma).
    - Verificado con scripts Python reales contra el `.so` compilado (no
      solo `cargo check`): roundtrip save/load con predicciones idénticas
      en RandomForest/XGBoost/KNN(ambas variantes)/Ridge→Lasso(cross-load)/
      ELM/GaussianNB/ObliqueForest; rechazo de tipo cruzado
      (`CatBoost.load(rf.json)`), de modelo sin fit, y de composite
      (Bagging/DeepForest); KMeans/DBSCAN recuperan blobs separados,
      IsolationForest marca el outlier con score mayor; CsvLoader
      classif/regress; ParquetLoader compilado y confirmado bajo
      `--features parquet` (build debug con fixture generada vía Rust/
      polars directo, sin depender de pyarrow/pandas en el venv). Suite
      completa verde (213 lib + 74 doctests) durante todo el trabajo.
    - Quedan del punto 9 original: M18 resampler stage en Pipeline,
      TreeBuilder O(n²), single-source de versión smelt-py. (M10 ParamSet
      tipado y duplicación scanner/EarlyStopper resueltos más abajo.)

    **✅ Duplicación (scanner, EarlyStopper) RESUELTA 2026-07-09 — train_binary/multiclass evaluada y descartada deliberadamente**:
    - **EarlyStopper en XGBoost**: sus 3 métodos (`train_regress`,
      `train_binary`, `train_multiclass`) reimplementaban inline la misma
      bitácora `(best_loss, no_improve, best_n)` + comparación + truncate-y-
      break que `EarlyStopper` (`src/learner/eval.rs`) ya encapsula y que
      LightGBM/CatBoost ya usaban — la única razón documentada para no
      usarlo era que XGBoost también pesa la loss por `sample_weight`, pero
      eso vive en el cómputo de `loss` (sin tocar), no en `EarlyStopper`
      mismo. Reemplazadas las 3 copias por `EarlyStopper::new(...)` +
      `.update(loss, n_trees)`, sin cambiar ningún cómputo de loss.
    - **"scanner ×5"**: los 3 loops de acumulación de histograma
      (`find_best_histogram_saving` en XGBoost, `build_leaf_hist` en
      LightGBM) y las 2 funciones de escaneo-de-histograma-con-gain-
      cerrado (mismo `find_best_histogram_saving`, `find_best_from_cache`
      en LightGBM) eran estructuralmente idénticas salvo por el cierre de
      gain (XGBoost aplica `violates_monotone` antes de puntuar; LightGBM
      no tiene esa restricción) — extraídas a `accumulate_histogram`/
      `best_numeric_split` en `src/learner/histogram.rs`, junto a
      `best_categorical_split` (mismo patrón de cierre `gain_fn` ya
      establecido ahí). CatBoost's `scan_partition_hists` (bins `f32`,
      no `f64`) se dejó como copia propia deliberadamente: el `f32` fue
      una decisión de performance medida (item 16d, `docs/fase3_progreso.md`,
      45.5% del tiempo de CatBoost en acumulación de histograma), no un
      descuido — forzar un helper compartido habría revertido esa
      optimización sin motivo.
    - **"train_binary/multiclass ×4" — evaluado, NO unificado**: al
      revisar el código real (no solo el resumen de la auditoría), la
      duplicación restante entre `train_binary`/`train_multiclass` dentro
      de cada motor (LightGBM, CatBoost) resultó ser más delgada de lo que
      sugería el shorthand original — la parte genuinamente repetida (la
      bitácora de early-stopping) ya quedó resuelta arriba. Lo que queda
      no es boilerplate copy-paste: `fv` es escalar-por-muestra en binario
      vs. vector-por-muestra en multiclase (softmax uno-contra-todos, un
      árbol por clase por ronda), y CatBoost además computa una
      codificación de target-statistics y bins de histograma DISTINTOS
      por clase en multiclase (ver M4, ya resuelto) vs. una sola en
      binario. Unificar esto en una abstracción compartida exigiría
      generalizar sigmoid+logloss y softmax+cross-entropy bajo el mismo
      código, o forzar el caso escalar como "multiclase con nc=1" — el
      tipo de reescritura matemática que arriesga alterar sutilmente las
      predicciones de los 3 motores insignia sin un bug activo que lo
      motive. Mismo criterio que TreeBuilder O(n²): documentado y
      diferido, no ejecutado a la fuerza.
    - Verificado: suite completa verde (213 lib + 286 integración + 74
      doctests) tras cada sub-paso (EarlyStopper, luego scanner), 0 fallos,
      0 warnings nuevos de clippy (verificado explícitamente — la
      extracción de `accumulate_histogram` generó 3 warnings
      `needless_borrow` transitorios, corregidos antes de este commit).

**✅ M10 (ParamSet tipado) RESUELTO 2026-07-09**: `ParamSet`/`ParamGrid`
(`src/tuning/mod.rs`) eran `HashMap<String, f64>`/`HashMap<String, Vec<f64>>`
— todo hiperparámetro forzado por `f64`, sin forma alguna de representar un
valor string (ej. un `objective`/`variogram_model` choice) ni de distinguir
un entero de un flotante salvo por convención (`params["max_depth"] as
usize` en cada sitio de uso). Nuevo enum `ParamValue` (`Float`/`Int`/`Bool`/
`Str`) con accessors tipados (`as_f64`/`as_usize`/`as_i64`/`as_bool`/
`as_str`, todos devolviendo `Result` con mensaje claro en vez de un cast
silencioso) e impls `From` para construcción ergonómica; `ParamSet =
HashMap<String, ParamValue>`, `ParamGrid = HashMap<String, Vec<ParamValue>>`,
`ParamDistribution::Choice(Vec<ParamValue>)` (Uniform/LogUniform siguen
siendo solo-`f64`, ya que un rango continuo no tiene análogo string). `.
as_usize()` trunca un `Float` igual que el cast `as usize` viejo, preservando
byte a byte los resultados de tuning con seed fija ya existentes.
- **Bonus no buscado**: los 3 tuners (`RandomSearch`, `BayesianOptimizer`,
  `Hyperband`) duplicaban la misma lógica de "samplear un valor desde un
  `ParamDistribution`" — como los tres necesitaban reescribirse para emitir
  `ParamValue` en vez de `f64`, se unificó en `sample_param_space`/
  `sample_one` (`tuning/mod.rs`) en vez de triplicar la nueva lógica.
  `BayesianOptimizer`'s KDE (`sample_from_good`/`log_density`) mantiene su
  cómputo Uniform/LogUniform sin cambios (vía `.as_f64()`), y su rama
  `Choice` pasó de comparar `(c - v).abs() < f64::EPSILON` (un proxy de
  igualdad para lo que ya eran valores discretos) a igualdad real de
  `ParamValue` — estrictamente más correcto, no solo un refactor.
- **smelt-py** (`smelt-py/src/tuning.rs`): `build_param_space` ahora parsea
  cada valor de una lista `choice` de Python preservando su tipo real
  (`bool`→`Bool`, `int`→`Int`, `float`→`Float`, `str`→`Str`, chequeados en
  ese orden ya que `bool` es subclase de `int` en Python) en vez de forzar
  `Vec<f64>` (que fallaba la extracción entera ante cualquier string). El
  heurístico `is_integer_param` (allowlist de nombres) se mantiene *solo*
  como fallback para el caso `Float` continuo (Uniform/LogUniform no cargan
  "esto es un entero" en el tipo) — `Int`/`Bool`/`Str` ya no lo necesitan,
  se propagan con su tipo real sin adivinar por nombre.
- Verificado: 213 lib + 286 integración + 74 doctests (conteos idénticos a
  antes del cambio), 0 warnings nuevos de clippy, ejemplo
  `xgboost_tuning.rs` corrido end-to-end, y un script Python directo contra
  el `.so` compilado confirmando (a) paridad exacta con el comportamiento
  numérico previo (`max_depth`/`n_estimators` siguen devolviendo `int`
  Python), y (b) la capacidad nueva: un `choice` de valores string/bool
  ahora sobrevive el roundtrip completo (Python → Rust → tuning → Python)
  como `str`/`bool` reales, algo que antes fallaba en la extracción misma.

**Reglas de proceso que esta auditoría reafirma**: (1) ningún ítem se declara cerrado en un doc de progreso sin commit verificable que lo toque — el falso cierre de `subsample` sobrevivió una auditoría entera; (2) todo módulo numérico nuevo entra con al menos un golden test contra scipy/sklearn o una reimplementación independiente — los 2 HIGH estadísticos de esta ronda vivían en los únicos rincones sin referencia; (3) los tests de comportamiento deben ser discriminantes (fallar contra la implementación rota) — el test de cost_sensitive pasa con un no-op.
