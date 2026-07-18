# Auditoría del Motor — Smelt (5ª auditoría completa)

**Fecha**: 2026-07-17
**Reviewer**: Claude Code (5 revisores paralelos: A motores boosting/GIS, B árboles+streaming+SVM+serialización, C core/preprocess/stats/tuning/conformal, D bindings smelt-py, E proceso/CI/releases/docs)
**Scope**: los 24 commits desde la 4ª auditoría (`6c8f720..HEAD` = `d3a852c`), ~2.225 inserciones en 42 archivos: los 6 fixes HIGH, Tiers 1/2 MEDIUM, 4 batches LOW, releases 3.0.0/0.6.0 y 3.1.0/0.7.0 (publicadas), CI nuevo (`ci.yml`), commit clippy masivo (`1177516`), y dos features nuevas (TimeSeriesCV + SplitConformal `b3e7a4e`; variograma WLS Cressie + Matérn 3/2 y 5/2 `d3a852c`) — más re-lectura del código como está HOY y re-verificación del backlog abierto.
**Referencia**: auditorías previas `docs/auditoria_motor_2026-07-01.md`, `2026-07-04.md`, `2026-07-05.md`, `2026-07-10.md`.
**Estado de tests**: 635 verdes a HEAD, verificado empíricamente al inicio (255 lib + 299 integración + 1 serialize_all_variants + 4 vs-official + 76 doctests, 0 fallos). Nota: `cargo test --workspace` NO compila (el lib-test de smelt-py no linkea sin símbolos de Python) — la suite canónica es `cargo test` en el crate raíz, que es lo que el CI corre.
**Método**: lectura línea a línea + confirmación empírica de todo CRITICAL/HIGH. Sondas destacadas: A/B con worktrees contra commits padre (sweep incremental M-3: bit-identidad en 3/6 configuraciones, divergencias restantes a nivel de ulp con paridad de calidad confirmada; perf real medida QRF ~9.5×, ObliqueTree ~12.6×, AdaBoost ~85×), goldens vs scipy 1.16.3 / sklearn 1.8.0 (friedman con empates y macro P/R/F1 **exactos a 12 decimales**), reproducción de las sondas de la 4ª (offsets 1e6–1e8, Stacking+Pipeline 20 seeds, coords NaN), verificación de Matérn contra cómputo independiente dígito a dígito, sonda de cobertura conformal (500 repeticiones), sonda de propiedades de TimeSeriesCV (1.467 configuraciones, 0 violaciones), y **39/39 roundtrips de persistencia bit-idénticos desde el `.so` recompilado a HEAD** con `maturin develop --release`.

## Resumen

- **Nuevos: 0 CRITICAL | 1 HIGH | 10 MEDIUM | ~17 LOW**
- **Ronda histórica para la serie**: es la **primera auditoría donde TODOS los cierres HIGH declarados resistieron la confirmación empírica completa**, incluidos los casos límite que la regla "verificado estructuralmente no cuenta" exige (MondrianForest con lifetime=∞ default, ARF post-drift con 10 drifts reales, LinearSVM a escala UTM, seeds entre procesos con diff vacío, Stacking+Pipeline 0/20 panics). Los 6 HIGH de la 4ª están genuinamente cerrados; de los ~19 MEDIUM de la 4ª, los Tier 1/2 verificados cerrados uno a uno; los 4 batches LOW verificados ítem por ítem.
- **El único HIGH nuevo es de proceso, no de código**: `release.yml` sigue pudiendo publicar wheels a PyPI desde un árbol rojo (el CI nuevo no se dispara con tags y el job publish no depende de ningún test), y el header del propio `ci.yml` presenta ese riesgo como pasado — regla "declarado cerrado pero no lo está = HIGH automático". Es el mismo eslabón que la 4ª señaló como el más débil del repo, cerrado a medias.
- **Patrón dominante de la ronda: "el fix cierra la ruta probada y deja la ruta hermana"**. Aparece 5 veces: target NaN rechazado en CSV pero no en `fit(X, y)` (MEDIUM-D1); `objective` no cableado → ValueError pero `huber_delta` solo → no-op silencioso (MEDIUM-D2); SplitConformal en Rust pero no en Python — con la réplica del paper corriendo en Python (MEDIUM-D3); `class_names`/`feature_names` propagados por Pipeline pero no `feature_types` — accuracy 1.000→0.623 con categóricas (MEDIUM-C1); CI corre en push/PR pero no en tags (HIGH-1).
- **Patrón secundario: degradación silenciosa en parámetros/datos degenerados**: GeoXGBoost con n≤30 evapora el mínimo de 30 vecinos y reporta un sweep de bandwidth ficticio (MEDIUM-A1); `CatBoost::with_max_bins(0|1)` produce modelo constante; `Bagging(n_estimators=0)` y `ELM(n_hidden=0)` ídem en Python; measures Python zip-truncan largos desalineados con scores "perfectos" (rmse=0.0, accuracy=1.0).
- **Las dos features nuevas grandes son sólidas**: el variograma WLS implementa Cressie (1985) correctamente con Matérn verificado contra la literatura (único residuo: una atribución "gstat-default" imprecisa en el doc); TimeSeriesCV no tiene ninguna violación de orden temporal/gap/bordes en 1.467 configuraciones; SplitConformal usa el rank correcto de Vovk/Lei con cobertura empírica dentro de la banda teórica, y `ConformalRegressor` ahora delega en él sin duplicación.
- **El commit clippy (`1177516`, +89/−243, 15 archivos) es conductualmente neutro**: verificado hunk por hunk por E (con B y A confirmando en sus archivos); la relocación de `build_final_encodings` en catboost.rs es textualmente idéntica.
- **Proceso**: además del HIGH-1, la convención M-12 recién estrenada se violó en sus dos primeras releases (3.1.0 omite del CHANGELOG el mínimo de 30 vecinos de GeoXGBoost — que cambia predicciones; 3.0.0 omite el fix de LinearSVM), el crate publicado incluye correspondencia privada (`docs/email_grekousis*.md` con un gmail de tercero) y las auditorías internas, y no existe ni un solo test Python permanente (las ~30 sondas por auditoría son desechables).

---

## Hallazgos CRITICAL (nuevos)

Ninguno.

---

## Hallazgos HIGH (nuevos)

### [HIGH-1] `release.yml` sigue pudiendo publicar a PyPI desde un árbol rojo — y el comentario de `ci.yml` declara el riesgo como resuelto
- **Archivo**: `.github/workflows/ci.yml:3-5` + `.github/workflows/release.yml:107-110`
- `ci.yml` se dispara con `on.push.branches: [master]` / PR / dispatch — **un push de tag NO lo dispara**. `release.yml` (trigger `on.push.tags: v*`) tiene un job `publish` que solo depende de los builds de wheels (`needs: [linux, windows, macos, sdist]`); no corre tests, no depende del workflow CI (GitHub no permite `needs` cross-workflow y no hay `workflow_run` ni environment protection). `git tag v0.8.0 && git push --tags` sobre cualquier commit — incluso uno que nunca pasó por CI o con CI en rojo — construye y publica a PyPI. El header de `ci.yml` ("Before this workflow existed … a tag could publish wheels to PyPI from a red tree") presenta el riesgo como pasado; estructuralmente sigue vigente → **regla "declarado cerrado pero no lo está"**. Agravante: la publicación a crates.io es 100% manual (`cargo publish`), sin gate alguno. Es el mismo ítem que la 4ª marcó como "el eslabón más débil del repo".
- **Fix**: job de `cargo test` dentro de `release.yml` como `needs` adicional del publish (la suite corre en el ref del tag), o environment de release con required check.

---

## Verificación de cierres de la 4ª auditoría (consolidada)

### Los 6 HIGH — todos genuinamente cerrados

| Ítem 4ª | Commit | Veredicto | Evidencia empírica de esta ronda |
|---|---|---|---|
| HIGH-1 cancelación MSE (offsets grandes) | `0e20116` | ✅ **CERRADO** | Sonda de la 4ª reproducida: DT RMSE **0.1330 constante** para offsets 0/1e6/1e7/1e8 (antes: colapso ~30-40× en 1e8); RF 0.1356→0.1362 (ruido). Centrado en la media del nodo; caso offset-0 y clasificación intactos; test de invarianza pineado en el repo |
| HIGH-2 persistencia irrecuperable | `f9e028f` | ✅ **CERRADO** (residuo: MEDIUM-B1) | Rust: `tests/serialize_all_variants.rs` cubre las 25 variantes con conteo pineado; sondas de casos límite bit-idénticas (Mondrian lifetime=∞ default vía `tau_serde`, Hoeffding con `feature_stats` poblado, **ARF post-drift real con `n_drifts()=10`**, CatBoost categórico). Python: **39/39 roundtrips bit-idénticos** contra el `.so` a HEAD, incluidas las 4 variantes rotas. `float_roundtrip` activado |
| HIGH-3 coords NaN (KrigingHybrid/GeoXGBoost) | `9fd7d8d` | ✅ **CERRADO** | Los 5 entry points dan `Err` nombrando el índice (KH train/predict, GXGB train/predict/select_bandwidth); `is_finite()` cubre ±inf; verificado también desde Python. Ni all-NaN silencioso ni panic de rayon |
| HIGH-4 Pipeline no propaga class_names | `c9e4123` | ✅ **CERRADO** (residuos: MEDIUM-C1, LOW-C2) | Sondas de la 4ª reproducidas: probas 3 columnas vía Pipeline; `Stacking(base=Pipeline(GaussianNB))` clase rara, 20 seeds → **0 panics** (4ª: 10/20). Bagging también propaga. Pero `feature_types` quedó fuera (MEDIUM-C1) y DeepForest NO propaga aunque el mensaje del commit lo afirma (LOW-C2) |
| HIGH-5 LinearSVM factor n | `2020a03` | ✅ **CERRADO** | `lambda = 1/(c·n)` + estandarización interna con `serde(default)` legacy. Defaults puros: n=50/100/400 → acc 0.960/0.990/0.995, **idéntico a escala UTM** (×1000 + 7.2e6). Test no-rincón en integración (n=400, ambas escalas) |
| HIGH-6 with_seed no reproducible entre procesos | `951db6b` | ✅ **CERRADO** | Sonda obligada: mismo binario, 3 procesos separados, seed 42, espacio con inserción en desorden → RandomSearch+Hyperband+BO: 22 líneas de configuración **idénticas, diff vacío**. Keys ordenadas en `sample_param_space` y BO |

### MEDIUMs y LOWs de la 4ª — verificados uno a uno

✅ **Cerrados genuinos**: M-1/M-2 (LightGBM GBDT default bit-idéntico a top_rate=1.0, GOSS opt-in funcional, docs de divergencia corregidos — `ebe5a44`), M-3 (sweep incremental con centrado heredado, no reintroducido: sonda offset 1e8 plana en los 3 primos — `c141f55`), M-4 feature_names en resamplers, M-5 (SMOTE/ADASYN/SpatialSmote rechazan NaN con mensaje "imputar antes" — `70ac096`), M-6 (CausalForest valida honesty/subsample incl. NaN — `8d9c8f4`), M-7 (filtros rechazan target continuo, regresión legítima no bloqueada, invariancia {0,1e12}≡{0,1} — `d730afc`), M-8 (CSV rechaza target missing/no-finito nombrando fila, consistente con Parquet — `edc64a7`), M-9-4ª friedman (golden scipy exacto con y sin empates — `948a6b1`), M-10 (macro P/R/F1 = sklearn exacto con labels {0,2} — `5925ab9`), M-11 (`[Unreleased]` presente, ParamSet marcado driver del major), M-12-py (best_params trunca como la factory — `bc0adc8`), M-13 (params no cableados → ValueError con lista; objective tunable de verdad — `6fe6a5a`; residuo MEDIUM-D2), M-14/M-15/N16 (benchmark no aborta, métricas por tarea, heurística alineada con fit — `506382b`), M-16 (load() con defaults del constructor — `78be3b7`), M-17 (cluster valida — `d338ef0`), M-18 (numpy bool — `205c5f9`), M-19 (QuantileForest.predict_quantile/interval completo — `9d1db6f`; residuo LOW-D1), M20 (versión single-source verificada contra el wheel real — `d31123e`), HIGH-7-orig (docstring LOO corregido, skip ahora solo alcanzable en geometría degenerada — `c01c0d7`; residuo MEDIUM-A1), CatBoost 64 bins (`with_max_bins` cableado a los 3 caminos, default bit-idéntico), clamp goss_sample (amplificación exacta + test de insesgadez), Huber ES (monitorea Huber real, golden), bandwidth N−1 (documentado como convención de George), CI de tests (existe y es bloqueante en push/PR; residuo HIGH-1), LightGBM Python max_depth, loaders con GIL liberado, y los 4 batches LOW (`c7356a9`/`b01017e`/`53546a1`/`6726ca1`) verificados ítem por ítem por C/B/A/D respectivamente — todos genuinos.

---

## Hallazgos MEDIUM (nuevos)

| # | Hallazgo | Archivo | Fix |
|---|----------|---------|-----|
| M-1 (A) | **GeoXGBoost n≤30: el mínimo de 30 vecinos se evapora en silencio** — la validación rechaza el valor *nominal* del candidato, pero con n<31 todos los candidatos ≥30 se clampean a n−1: sonda n=25, candidatos [30,40,50] → `Ok`, 3 scores exactamente idénticos (1.514788), "best=30" reportado con bandwidth efectivo 24 — sweep ficticio y garantía documentada incumplida. El mecanismo (Err vs warning) debe confirmarse con George; lo silencioso no es discutible | `geo_xgboost.rs:219,513` vs `:187-194` | `Err` claro si `n−1 < MIN_BANDWIDTH` en `select_bandwidth`; mínimo, error si dos candidatos clampean igual |
| M-2 (B) | **f9e028f rompe archivos CatBoost válidos pre-fix dentro del mismo `format_version` 1** — un CatBoost sin cat_features guardado con 2.0.x–3.0.0 (`"cat_encodings": [{}]` forma-objeto) hoy da `JSON error: invalid type: map, expected a sequence` — el error opaco que el envelope versionado existe para evitar; modelo irrecuperable para ese usuario. Ventana acotada y falla ruidosa → MEDIUM, no HIGH | `catboost.rs` (`cat_encodings_serde::deserialize`) + `serialize.rs:302` | deserialize untagged que acepte ambas formas de wire, o subir `SERIALIZATION_FORMAT_VERSION` para que el mensaje de versión los atrape |
| M-3 (C) | **Pipeline no propaga `feature_types` — el soporte categórico se degrada en silencio incluso con cero transformers**: sonda con categórica de 8 códigos no monótona, XGBoost stumps: directo acc=1.000; vía `Pipeline::new(vec![], ...)` acc=0.623 (151/400 predicciones distintas). Anula la propagación de tipos de c9e4123 cuando Smote/Adasyn corren como stage del Pipeline (su caso de uso principal) | `pipeline.rs:134-137,168-169` | propagar `feature_types` cuando los transformers preserven columnas; `transform_types` análogo a `transform_names` para selectores |
| M-4 (D) | **`fit(X, y)` con NaN en target de regresión entrena en silencio y predice all-NaN** — XGBoost y LinearRegression retornan OK y `predict()` = `[nan…]`. La ruta CSV rechaza exactamente esto desde edc64a7; el entry point principal quedó sin el chequeo (patrón "ruta hermana") | `smelt-py/src/common.rs:151` | validar `target.iter().all(f64::is_finite)` en la rama regresión → ValueError nombrando el índice |
| M-5 (D) | **Tunear `huber_delta` sin `objective` es un no-op silencioso** — la factory solo lee `huber_delta` dentro del `if let` de `objective`, pero `factory_param_names` lo lista incondicionalmente: `optimize("xgboost", {"huber_delta": (0.5,10)})` → 6 trials con score bit-idéntico (12.447398809889 los seis). La clase exacta de M-13 sobreviviendo dentro de la allowlist | `smelt-py/src/tuning.rs:56` vs `:120` | si `huber_delta` está en el espacio, exigir `objective` con `"huber"` entre los choices, o ValueError explicativo |
| M-6 (D) | **SplitConformal no está expuesto en Python — imposible conformalizar `predict_spatial`**: KrigingHybrid no tiene superficie conformal alguna (ni shap/permutation importance); `GeoXGBoost.conformal_predict` calibra sobre el `predict` global sin corrección espacial y su docstring promete un parámetro `coords` que la firma no tiene. La réplica del paper corre en Python (`paper/replication/`) → el flujo PM2.5 publicado no es reproducible desde los bindings | `smelt-py` (ausencia) + `boosting.rs:656-658` | exponer `SplitConformal` o `conformal_predict(..., coords_cal, coords_test)` en ambos learners espaciales; corregir el docstring |
| M-7 (E) | **3.1.0 publicó el cambio de resultados de GeoXGBoost (mínimo 30 vecinos, `c01c0d7`) sin entrada de CHANGELOG** — primera violación de la convención M-12 recién escrita, en el método co-diseñado con Grekousis | `CHANGELOG.md:101-173` | entrada retroactiva con nota "added retroactively" |
| M-8 (E) | **3.0.0 omite el fix de LinearSVM** — el cambio de comportamiento más grande de la release (acc 0.51→~1.0 con defaults) sin entrada; los otros 5 HIGH sí la tienen | `CHANGELOG.md:175-303` | entrada retroactiva |
| M-9 (E) | **El crate publicado incluye correspondencia privada y strays**: `docs/email_grekousis*.md` (con gmail personal de un tercero), `docs/respuesta_equipo_*`, `docs/wos_*`, las 4 auditorías internas, `tests/{catboost,lightgbm}_perf.py` (el exclude solo cubre `tests/xgboost_*.py`). 155 archivos en `cargo package --list` | `Cargo.toml:16-27` | ampliar `exclude` o migrar a `include`-list (src/, tests/, examples/, benches/, README, CHANGELOG, LICENSE) |
| M-10 (E) | **CI no ejerce ningún comportamiento Python** — compila smelt-py vía clippy pero no hay `maturin build`, ni smoke-test de import, ni un solo `test_*.py` en el repo; la clase de bug del HIGH-2 de la 4ª sigue sin red permanente | `.github/workflows/ci.yml` | job con maturin build + pytest mínimo (import, fit/predict, save/load roundtrip, un tuner) |

---

## Hallazgos LOW (nuevos)

- **A (boosting/GIS)**: `CatBoost::with_max_bins(0|1)` produce modelo constante en silencio (sonda; validar ≥2 → `InvalidParameter`); atribución "gstat-default" incorrecta en el doc del WLS (`kriging_hybrid.rs:159-160` — el objetivo implementado es Cressie 1985 = `fit.method=2` de gstat, no el default 7; solo doc, el código está bien); el default GBDT de LightGBM paga un sort O(n log n) inútil por árbol (`lightgbm.rs:304-310`; shortcut si `top_rate≥1.0`); early stopping con `Objective::Custom` monitorea MSE sobre el score crudo, documentado solo en el método privado (`xgboost.rs:78-91` vs doc público de `with_objective`).
- **B (árboles/streaming)**: `TrainedDES::predict` no valida `n_features` — vecinos KNORA-E calculados sobre el prefijo común con dimensiones desalineadas (`des.rs:119-136`; `check_n_features` como los demás).
- **C (core)**: `CQR::calibrate` y `ConformalClassifier::calibrate` siguen zip-truncando calibración desalineada (b3e7a4e lo cerró solo en el camino regresor) (`cqr.rs:83-87`, `conformal/mod.rs:262`); el mensaje de c9e4123 afirma que DeepForest "ya propaga" class_names — falso, M-9 sigue abierto (`deep_forest.rs:254`; impacto real acotado: columnas OOF en cero, 0 panics en 10 seeds); `with_class_names` no valida largo contra labels → panic downstream (`task/mod.rs:169-172`; sonda: Smote panic index-out-of-bounds — pre-existente pero el fix de HIGH-4 amplificó la superficie).
- **D (smelt-py)**: `QuantileForest.feature_importances_` siempre `None` (`TrainedQuantileForest` no implementa `feature_importance()` en Rust); `XGBoost(objective="bogus")` se acepta hasta `fit()` (inconsistente con la convención eager de 6726ca1: KrigingHybrid/ELM validan en `__new__` y `set_params`); measures zip-truncan largos desalineados con scores "perfectos" (`rmse_score` 3-vs-1 → 0.0, `accuracy_score` → 1.0, `auc_roc` → 0.5); parámetros degenerados aceptados (`Bagging(n_estimators=0)` predice constante — el gemelo del `max_depth=0` que 6726ca1 sí validó; `ELM(n_hidden=0)`; `MondrianForest(lifetime=-1)`); taxonomía de persistencia composite inconsistente (`DeepForest/Bagging.save()` → RuntimeError vs GeoXGB/KH → NotImplementedError explicativo; `brier_score` malformado → TypeError de PyO3, no ValueError).
- **E (proceso)**: 2 `.pyc` **trackeados en git** desde v0.2.0 (`smelt-py/python/smelt/__pycache__/{__init__,stats}.cpython-312.pyc` — árbol perpetuamente sucio; `git rm --cached`); actions pinneadas por tag móvil y publicación con `PYPI_API_TOKEN` en vez de trusted publishing OIDC; `docs/roadmap_checklist.md` desactualizado en Prioridad 5/6 (Parquet/PyO3/CI/cargo-doc siguen `[ ]` estando hechos); `cargo test` de CI corre solo el paquete raíz (hoy sin efecto — smelt-py no tiene tests Rust).

---

## Verificación de las features nuevas (b3e7a4e, d3a852c)

| Feature | Veredicto | Evidencia |
|---|---|---|
| TimeSeriesCV | ✅ **sólida** | Sonda de propiedades: 1.467 configs (n∈{5..61} × horizon × min_train × step × gap × ventana) → 0 violaciones (train precede test más que el gap, horizonte exacto, ventana acotada, bordes no divisibles bien formados; error solo y siempre cuando `min_train+gap+horizon > n`). Semántica = sklearn `TimeSeriesSplit` extendido. Expuesta y funcional en Python |
| SplitConformal | ✅ en Rust / ❌ en Python (MEDIUM-6) | Rank `ceil((n+1)(1-α))` = quantile (1-α)(1+1/n) de Vovk/Lei, no el ingenuo (verificado a mano n=9; n=4/α=0.1 → ∞ conservador sin clamp). Cobertura empírica 500 reps, n_cal=50, α=0.1: **0.9031** ∈ [0.90, 0.9196] teórico. `ConformalRegressor` delega en él (ancho bit-idéntico) — sin duplicación. Residuo: CQR/Classifier siguen zip-truncando (LOW-C1) |
| Variograma WLS + Matérn | ✅ **sólida** | Pesos `N_j(γ̂−γ)²/γ(h;θ)²` = Cressie 1985; los 4 goldens de Matérn 3/2 y 5/2 reproducidos dígito a dígito contra cómputo independiente (convención sklearn `√3h/r`, `√5h/r`); γ(0)=0, límites nugget/sill correctos, `nugget≤sill` por construcción. `predict_spatial` mejora MSE en los 5 modelos (8.04 → 0.0002–0.0040). Sonda GRF: varianza del variograma empírico de una realización, no del fit. matern32/matern52 expuestos en Python con validación eager. Residuo: atribución gstat en doc (LOW-A2) |

## Commit clippy `1177516` — veredicto

**Conductualmente neutro** (E revisó el diff completo, 15 archivos +89/−243, hunk por hunk; A y B confirmaron en sus archivos): let-chains equivalentes, relocación textualmente idéntica de `build_final_encodings`, código muerto verificado muerto en examples, range-contains con precedencia correcta. Nota menor para backlog de tests: `filter_variance_selects_non_constant` construía y descartaba un Pipeline sin verificar el drop (defecto pre-existente que el lint hizo visible).

---

## Backlog abierto (sin cambios esta ronda)

| ID | Tema | Evidencia actual |
|---|---|---|
| N1 | alpha (L1) no afecta gain en XGBoost; caminos monotone divergen con alpha>0 según si el nodo vino del scan o de la resta de histogramas | `xgboost.rs:474-478,483-491`, `hist_pool.rs:73,98-105` |
| M-6 | MondrianForest probas = voto duro cuantizado (probas por árbol descartadas, ambos caminos) | `mondrian.rs:885-915`, `predict_one_classif` |
| M-7 | DeepForest doc "every previous layer" vs código (solo la anterior — el código es el correcto del paper, el bug es la doc) | `deep_forest.rs:20-21` vs `:292` |
| M-9 | DeepForest fold-tasks sin class_names (c9e4123 no lo tocó, pese a afirmarlo) | `deep_forest.rs:254,270` |
| M14 | QRF pooling crudo entre árboles sin pesos 1/\|hoja\| de Meinshausen (sesgo hacia hojas grandes), cita intacta | `quantile_forest.rs:6,310-327` |
| M13/M17/M21 | Filter fallback silencioso a varianza; Clone de RFE panica al re-fit; trait Filter no enchufable | `filter.rs:269-277`, `rfe.rs:84`, `filter.rs:47-59` |
| M19 residual | float32 → TypeError críptico sin mencionar float64; y de conteos (int) → clasificación con 65 clases en silencio | verificado contra el `.so` a HEAD |
| LOWs streaming | ARF background tree entrena para siempre en falsa alarma; Hoeffding descarta clases nuevas en hojas viejas; ARF sin re-seed entre train repetidos (`n_drifts` acumula); best_threshold media global | `adaptive_rf.rs:450-459,499-511`, `hoeffding.rs:170-172,394-407` |
| LOWs core | Wilcoxon 1e-15 sin doc; McNemar sin binomial exacto; LogLoss/AUC panican con label fuera del ancho; select_best deja ganar NaN; Imputer 100% NaN→0; Relief NaN panic (vía FilterSelector directo); MAPE/n con ceros; error no determinista en tuners paralelos; zips multilabel; CausalForest OOB→ATE 0.0; clip 1e-3 DR/X | verificados uno a uno |
| LOWs smelt-py | sin `.pyi`; sin `__repr__`; ET/DT sin predict_proba; BO factory 8 learners; paridad ausente (Adasyn, Pipeline+resampler, survival, multilabel/multioutput, CausalForest, friedman/nemenyi/mcnemar, streaming partial_fit/n_drifts, MondrianTree) | verificado `hasattr` contra el `.so` |

**Cerrados del backlog esta ronda** (además de la lista de la sección de cierres): N16, CI-de-tests (parcial: push/PR sí, tags no), M-11, M-12 (escrita, aplicada a medias), M20, LightGBM max_depth Python, loaders GIL, ADASYN k=0, silhouette labels no contiguos, DES errores tragados, contrato posicional feature_importance, mape/logloss bound.

---

## Prioridad sugerida

1. **HIGH-1** — gate de tests en `release.yml` (el mismo eslabón dos auditorías seguidas; ~20 líneas de workflow).
2. **M-2 (B)** — compat legacy CatBoost: cada día que pasa hay más archivos 3.1.0 sanos, pero los 2.0.x–3.0.0 siguen irrecuperables; decidir untagged-deserialize vs bump de format_version antes de la 3.2.0.
3. **M-3 (C) + M-4/M-5 (D)** — los tres son "ruta hermana" de fixes recientes, chicos y con sonda discriminante lista.
4. **M-6 (D)** — SplitConformal en Python: bloquea la reproducibilidad del flujo del paper desde los bindings; conviene decidir la firma junto con la subsección conformal que George tiene pendiente.
5. **M-7/M-8 (E)** — entradas retroactivas de CHANGELOG (10 minutos) y **M-9 (E)** — exclude de correspondencia privada **antes del próximo `cargo publish`**.
6. **M-1 (A)** — n≤30 en GeoXGBoost: preparar la propuesta (Err vs warning) para confirmar con George.
7. **M-10 (E)** — pytest mínimo en CI, para que la clase de bug del HIGH-2 de la 4ª tenga red permanente.
