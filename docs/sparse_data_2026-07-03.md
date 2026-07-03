# Sparse data support (item 16d parte 3/3) — 2026-07-03

## Investigación previa (antes de tocar código)

`Task::features() -> &Array2<f64>` es un método concretamente tipado usado
en **44 sitios** de `src/learner/*.rs` (`&dyn Task` aparece solo 1 vez, en
un helper interno de CatBoost) — no existe ninguna costura de trait-object
donde un `SparseTask` pudiera enchufarse sin reescribir el trait `Task` o
duplicar toda la superficie de `Learner` (`train_classif`/`train_regress`
paralelos para cada uno de los ~20 learners).

De los learners existentes, solo los modelos lineales (regresión
logística/lineal, SVM) tendrían una historia real de speedup algorítmico
con productos punto sparse — el boosting (`HistBins`) necesitaría
reescribirse igual de todos modos, sin importar el formato de `Task`.

El único caso confirmado y genuinamente desperdiciado **hoy** es
`OneHotEncoder::transform` sobre una columna de alta cardinalidad: asigna
una matriz densa `n_samples × n_categorías` que es >99% ceros para
cualquier cosa con más de un puñado de categorías.

`ndarray` no tiene soporte sparse propio; `sprs` (el crate estándar de
Rust para sparse) no está en el árbol de dependencias.

## Decisión de alcance

Dado que un `SparseTask` completo no está justificado por la evidencia
(alto riesgo de invasión en 44+ sitios, sin beneficio claro para la mayoría
de los learners), se acotó a: un tipo CSR propio (sin agregar `sprs` como
dependencia) + una salida sparse para `OneHotEncoder`, con `to_dense()`
como escape hatch. Integración de `SparseTask`/`Learner` y matemática
sparse en modelos lineales quedan como seguimientos separados, más grandes,
si alguna vez se priorizan.

## Qué se construyó

- **`src/sparse.rs`** (nuevo): `CsrMatrix`, formato CSR estándar
  (`indices`/`values`/`row_ptr`). `from_triplets(n_rows, n_cols, triplets)`
  (construcción por conteo, suma duplicados, valida bounds),
  `n_rows`/`n_cols`/`nnz`/`density`, `row(i)` (iterador `(col, value)`),
  `dot_row(i, dense)`, `to_dense()`. 6 tests unitarios.
- **`OneHotEncoder::transform_sparse`** (`src/preprocess/encoder.rs`): mismo
  algoritmo que `transform` pero emite triplets en vez de escribir en un
  `Array2` denso. Columnas passthrough se almacenan igual (rara vez son
  sparse en la práctica). 2 tests de integración: `transform_sparse(x)
  .to_dense() == transform(x)` (corrección) y densidad ~1/n para una
  columna de 200 categorías distintas (confirma el caso de uso real).
- Wiring: `pub mod sparse;` + `CsrMatrix` en el prelude de `src/lib.rs`.

## Validación

`cargo build -p smelt-ml` limpio. `cargo test -p smelt-ml --lib`: 101
verdes (95 + 6 nuevos de `sparse::tests`). `cargo test -p smelt-ml --doc`:
66 sin cambios. `cargo test --test integration`: 274 verdes (272 + 2
nuevos). `cargo check -p smelt-py`: limpio, sin cambios necesarios (no se
expone `CsrMatrix` a Python en esta pasada).

## Fuera de alcance

- `SparseTask`/integración con `Learner` — requeriría reescribir el trait
  `Task` o duplicar `train_classif`/`train_regress` en cada learner.
- Matemática sparse en modelos lineales (`logistic_regression.rs`,
  `linear_regression.rs`, `svm.rs`) — el único lugar donde sparse daría un
  speedup algorítmico real, no solo de memoria.
- Bins sparse-aware en `HistBins` (boosting) — necesitaría reescribir el
  binning column-major denso actual independientemente del formato de
  `Task`.
- Bindings Python (`smelt-py`) para `CsrMatrix`.

Con esto, ítem 16d completo: parte 1 (Parquet), parte 2 (f32 histograms,
solo CatBoost), parte 3 (CSR + OneHotEncoder sparse, alcance acotado).
Queda 17b (`missing_docs`) como único pendiente grande de Fase 3.
