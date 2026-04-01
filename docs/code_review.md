# Code Review — smelt-ml v1.1.1

**Fecha**: 2026-04-01
**Reviewer**: Claude Code Reviewer
**Scope**: Proyecto completo (13,377 líneas en 40 archivos .rs)

## Resumen

- Critical: 0 | High: 3 | Medium: 6 | Low: 4
- Calidad general: **Buena**
- Deuda técnica estimada: **Baja-Media**

Proyecto bien diseñado con arquitectura trait-based limpia. Los problemas principales son archivos grandes (4 archivos >500 líneas), `unwrap()` en código de producción para sorting de floats, y algunas oportunidades de reducir `.clone()` y `.to_vec()`.

**Puntos fuertes**: 0 unsafe, 0 TODOs, 0 imports circulares, 250 tests, API consistente.

---

## Hallazgos

### [HIGH] H1: Archivos > 500 líneas — 4 archivos

- **Archivos**: `xgboost.rs` (617), `oblique.rs` (602), `lightgbm.rs` (597), `catboost.rs` (537)
- **Dimensión**: D1 Complejidad
- **Descripción**: Los 3 boosting engines y oblique forest superan el umbral de 500 líneas. Cada archivo contiene binning, tree building, trained model, y learner impl — múltiples responsabilidades.
- **Impacto**: Dificulta la mantenibilidad. Un cambio en binning requiere navegar 600 líneas.
- **Fix propuesto**: Extraer binning compartido a un módulo común (`src/learner/bins.rs`), ya que XGBoost, LightGBM y CatBoost usan la misma lógica de column-major binning con variaciones menores.

### [HIGH] H2: `unwrap()` en partial_cmp para sorting (150 ocurrencias en src/)

- **Archivos**: Múltiples (sorting de f64 en histograms, splits, etc.)
- **Dimensión**: D5 Error Handling
- **Descripción**: La mayoría de los 150 `unwrap()` están en `sort_by(|a, b| a.partial_cmp(b).unwrap())`. Si un NaN escapa la validación, esto causa panic. ~80 están en doc comments (OK), ~40 en sorting de floats, ~30 son legítimos en algoritmos internos.
- **Impacto**: Un NaN inesperado en features no-XGBoost causa panic en vez de error.
- **Fix propuesto**: Reemplazar con `.unwrap_or(std::cmp::Ordering::Equal)` — ya se usa en algunos lugares pero no consistentemente.

### [HIGH] H3: `panic!` en RFE clone

- **Archivo**: `src/preprocess/rfe.rs:82`
- **Dimensión**: D5 Error Handling
- **Descripción**: `Clone` para RFE usa `panic!("cloned RFE cannot create new learners")`. Esto puede triggerearse si un Pipeline con RFE se usa en Bagging (que clona transformers via `clone_box()`).
- **Impacto**: Crash en runtime si el usuario combina RFE + Bagging + Pipeline.
- **Fix propuesto**: Hacer que `RFE::clone()` no incluya la factory, y que `fit_supervised` retorne error si la factory no está disponible. O almacenar los selected indices en el clone y skip re-fitting.

### [MEDIUM] M1: 87 `.clone()` y 90 `.to_vec()` en producción

- **Dimensión**: D1 Complejidad / Performance
- **Descripción**: Muchos clones son de `Array2<f64>` (feature matrices) y `Vec<usize>` (indices). Algunos son necesarios (pasar ownership a Tasks), otros podrían evitarse con lifetime annotations o borrowing.
- **Impacto**: Overhead de memoria en datasets grandes. No afecta correctness.
- **Fix**: Auditar caso por caso. Los más impactantes son los `features.clone()` en Pipeline y Stacking.

### [MEDIUM] M2: 5 `unreachable!()` en predict de trees

- **Archivos**: `decision_tree.rs`, `random_forest.rs`, `extra_trees.rs`
- **Dimensión**: D5 Error Handling
- **Descripción**: En predict, cuando un leaf es `LeafValue::Value` pero el modelo es clasificación (o viceversa), se usa `unreachable!()`. Esto es correcto lógicamente (un classifier nunca produce Value leaves) pero un bug en el tree builder causaría panic sin mensaje útil.
- **Fix**: Cambiar a `_ => return Err(SmeltError::Other("internal: unexpected leaf type".into()))`.

### [MEDIUM] M3: Binning duplicado en XGBoost, LightGBM, CatBoost

- **Archivos**: `xgboost.rs:87-137`, `lightgbm.rs:88-137`, `catboost.rs:149-191`
- **Dimensión**: D3 Cohesión / DRY
- **Descripción**: Los 3 boosting engines tienen su propia implementación de column-major binning. Son casi idénticas (cambia el tipo del struct y el nombre). Violación de DRY.
- **Fix**: Extraer a `src/learner/histogram.rs` con un `HistogramBins` genérico.

### [MEDIUM] M4: `serialize.rs` tiene 9 imports de `crate::`

- **Archivo**: `src/serialize.rs`
- **Dimensión**: D2 Acoplamiento
- **Descripción**: SerializableModel importa todas las trained structs directamente. Un nuevo learner requiere modificar este archivo.
- **Impacto**: Bajo — es el propósito del módulo (serialización centralizada). Pero escala mal.
- **Fix**: Aceptable por ahora. A futuro, considerar un macro que auto-registre learners.

### [MEDIUM] M5: `#[allow(dead_code)]` en 7 lugares

- **Dimensión**: D7 Deuda Técnica
- **Descripción**: 7 `allow(dead_code)` en struct fields y funciones. Algunos son campos reservados para futuro uso (como `coords` en TrainedGeoXGBoost), otros son funciones que no se llaman externamente.
- **Fix**: Revisar cada uno — si el campo no se usa, eliminarlo. Si se planea usar, documentar por qué.

### [MEDIUM] M6: Variables de una letra en contexto no-loop

- **Dimensión**: D6 Naming
- **Descripción**: ~10 variables como `r`, `t`, `x`, `b`, `d` fuera de loops. En contexto matemático (gradients, thresholds) son aceptables pero reducen legibilidad.
- **Impacto**: NIT en la mayoría de los casos — el contexto matemático justifica nombres cortos.

### [LOW] L1: No hay doc comments en módulos internos

- **Dimensión**: D6 Naming & Clarity
- **Descripción**: Los struct públicos tienen doc comments pero las funciones internas (helpers, free functions) no. Esto dificulta la contribución externa.

### [LOW] L2: Tests de integración en un solo archivo (2,700+ líneas)

- **Dimensión**: D1 Complejidad
- **Descripción**: `tests/integration.rs` tiene 250 tests en un solo archivo. Dificulta la navegación.
- **Fix**: Dividir en `tests/learners.rs`, `tests/preprocessing.rs`, `tests/tuning.rs`, etc.

### [LOW] L3: README no refleja v1.1.1

- **Dimensión**: D7 Deuda Técnica
- **Descripción**: El README muestra `smelt-ml = "0.6"` y no menciona los benchmarks de performance vs C++.
- **Fix**: Actualizar version badge y agregar sección de performance.

### [LOW] L4: Cargo.toml sin badges de CI

- **Dimensión**: D7 Deuda Técnica
- **Descripción**: No hay CI/CD configurado (GitHub Actions). No hay badge de build status.

---

## Arquitectura

### Evaluación SOLID

| Principio | Estado | Nota |
|-----------|:---:|------|
| **S** (Single Responsibility) | ✓ | Cada módulo tiene responsabilidad clara. Los boosting engines son largos pero cohesivos. |
| **O** (Open/Closed) | ✓ | Agregar un nuevo learner no requiere modificar código existente — solo implementar traits. |
| **L** (Liskov Substitution) | ✓ | Cualquier `Box<dyn Learner>` es intercambiable sin romper el sistema. |
| **I** (Interface Segregation) | ✓ | `Learner`, `TrainedModel`, `Measure`, `Transformer` son interfaces mínimas. |
| **D** (Dependency Inversion) | ✓ | Todo depende de traits, no de implementaciones concretas. |

### Fortalezas arquitectónicas

1. **Composabilidad perfecta**: Pipeline implementa Learner, Bagging wraps any Learner, Stacking combina Learners — todo se conecta sin fricciones.
2. **Zero unsafe**: 13,377 líneas sin un solo `unsafe`.
3. **Parallelism clean**: rayon usado correctamente en Random Forest, Extra Trees, Oblique Forest, XGBoost.
4. **Error handling consistente**: `Result<T>` con `SmeltError` en toda la API pública.

### Debilidades arquitectónicas

1. **Binning duplicado**: 3 implementaciones casi idénticas.
2. **Serialización frágil**: `SerializableModel` enum crece linealmente con cada learner nuevo.
3. **No hay logging**: ningún logging de progreso durante training largo.

---

## Deuda Técnica

| Tipo | Conteo | Severidad |
|------|:---:|:---:|
| TODOs/FIXMEs | 0 | Ninguna |
| Código muerto (#[allow]) | 7 | Baja |
| Archivos >500 líneas | 4 | Media |
| Tests en un solo archivo | 1 (2700+ líneas) | Baja |
| README desactualizado | Parcial | Baja |
| CI/CD ausente | Sí | Media |

---

## Recomendaciones (por prioridad)

1. **Extraer binning compartido** a `src/learner/histogram.rs` — reduce 3x duplicación, simplifica los 3 boosting engines (~200 líneas cada uno menos)
2. **Reemplazar `unwrap()` en sorting** con `.unwrap_or(Ordering::Equal)` — previene panics con NaN inesperados
3. **Fix RFE clone panic** — cambiar a error graceful
4. **Agregar CI/CD** con GitHub Actions — `cargo test` + `cargo clippy` en push
5. **Dividir tests** en múltiples archivos por dominio
6. **Actualizar README** con version 1.1.1 y benchmarks de performance
