# Simulación de Revisión de Pares — Journal of Statistical Software

**Manuscrito**: "smelt-ml: A Pure-Rust Machine Learning Framework with Cache-Optimized Gradient Boosting Competitive with C++ Implementations"
**Autor**: Francisco Parra
**Fecha de envío**: 2026-04-01

---

## Fase 1: Editor-in-Chief (EiC) — Desk Review

**EiC**: Jan de Leeuw

### Evaluación de admisibilidad

| Criterio | Resultado | Nota |
|----------|:---:|------|
| Scope: software estadístico/ML | ✓ | ML framework con métodos estadísticos |
| Lenguaje aceptado | ✓ | "C, C++, Fortran, **among others**" — Rust aplica |
| Código fuente incluido | ✓ | crates.io + GitHub |
| Template JSS usado | ✓ | jss.cls con markup correcto |
| Originalidad aparente | ✓ | Primer software ML en Rust para JSS |

### Decisión EiC

> **ACCEPT FOR REVIEW**
>
> This is the first submission to JSS involving a Rust implementation. The scope
> is appropriate — the manuscript presents statistical/ML software with
> benchmarks and validation. Assigning to Section Editor for ML/Software.

**Asignado a**: Section Editor para Machine Learning Software

---

## Fase 2: Section Editor — Asignación de Revisores

**Section Editor**: [Nombre ficticio]

> Selecting two reviewers:
> - **Reviewer 1**: Expert in gradient boosting implementations (XGBoost/LightGBM internals)
> - **Reviewer 2**: Expert in R/Python ML frameworks and software engineering

---

## Fase 3: Reviewer 1 — Experto en Gradient Boosting

### Report

**Recommendation**: **Major Revision**

**Summary**: The paper presents smelt-ml, a Rust-based ML framework claiming performance competitive with C++ gradient boosting libraries. The cache optimization strategy is sound and the benchmarks show impressive results. However, several methodological and presentation issues need to be addressed before publication.

---

### Strengths

1. **Novel contribution**: First comprehensive ML framework in Rust with gradient boosting from scratch. The cache-optimization via column-major storage is well-motivated and clearly explained.

2. **Impressive performance**: The XGBoost benchmarks (1.1-2.6× faster) are remarkable for a pure-Rust implementation. The CatBoost improvement (4.2×) is particularly noteworthy.

3. **Breadth of algorithms**: 27 supervised learners plus clustering, survival, causal inference — substantially more complete than linfa.

4. **Correctness validation**: Accuracy comparison against scikit-learn on standard datasets is appropriate and shows equivalent results.

---

### Major Issues

**M1. Incomplete benchmarking methodology**

The benchmarks only test N up to 10,000 samples with 20 features. This is too small to draw general conclusions. The official libraries are designed for datasets of 100K-10M rows. The paper should include:
- Benchmarks at N = 50,000 and N = 100,000
- Varying number of features (50, 100, 500)
- Memory consumption comparison
- Wall-clock time including data preprocessing

*Without larger-scale benchmarks, the claim of "beating C++" is valid only for small datasets and may be misleading.*

**M2. Single-threaded comparison only**

All benchmarks use `n_jobs=1`. The official libraries have highly optimized multi-threaded implementations. The paper should:
- Include multi-threaded benchmarks (the framework uses rayon)
- Or explicitly scope the claims to single-threaded performance

**M3. Missing statistical rigor in benchmarks**

Each benchmark is run once. There is no reporting of:
- Standard deviation across multiple runs
- Confidence intervals
- Statistical tests for significance of timing differences
- Hardware warm-up / cold-start effects

*At minimum, each benchmark should be run 10 times with mean ± std reported.*

**M4. Incomplete XGBoost implementation**

The paper does not discuss which XGBoost features are NOT implemented. Specifically:
- Monotone constraints
- Interaction constraints
- Custom loss functions (only MSE, log-loss, softmax are mentioned)
- Distributed training
- Approximate quantile sketch vs. the histogram approach used

*A clear Table comparing implemented vs. not-implemented features of each engine would strengthen the paper.*

**M5. CatBoost ordered boosting simplification**

The paper acknowledges (Section 6.2) that ordered boosting is "simplified" but does not explain the implications. The key innovation of CatBoost is the O(n²) unbiased gradient estimation. Without it, the "CatBoost" implementation is essentially a symmetric-tree gradient boosting machine with target statistics encoding — not a full CatBoost. This should be:
- Clearly stated upfront (not buried in limitations)
- The name should perhaps be "CatBoost-inspired" or "Symmetric GBM with target statistics"

---

### Minor Issues

**m1.** Section 3.2: The claim that "Rust's guarantee of no aliasing allows LLVM to apply more aggressive auto-vectorization" is plausible but not demonstrated. Please provide evidence (e.g., comparing assembly output or running with/without `target-cpu=native`).

**m2.** Table 2: LightGBM shows speedup < 1 at N=500 and N=5000 (classification). The text says "1.2-1.4× faster" but this only holds for N≥1000 regression. Please be more precise in claims.

**m3.** The accuracy comparison (Table 4) uses different seeds for cross-validation between smelt-ml and scikit-learn. Results should use the same random splits for a fair comparison.

**m4.** The paper mentions "250 tests" but does not describe test coverage methodology or what percentage of code is covered.

**m5.** No comparison with linfa is provided, even though it is the primary competitor in Rust.

**m6.** The "Unique Features" section (Section 5) is too brief. Each subsection is 2-3 sentences. Either expand with examples and validation, or move to an appendix and focus the paper on the gradient boosting contribution.

---

### Questions for the Authors

1. What happens to performance at N = 100,000? Does the Rust advantage hold or does it disappear?
2. Is the column-major optimization applicable to the official C++ libraries? If so, why haven't they adopted it?
3. What is the accuracy of the "CatBoost" implementation compared to the official CatBoost on datasets where ordered boosting matters (e.g., high-cardinality categorical features)?

---

## Fase 4: Reviewer 2 — Experto en Frameworks de Software ML

### Report

**Recommendation**: **Minor Revision**

**Summary**: This paper presents a well-designed ML framework in Rust with an impressive trait-based architecture. The software engineering is excellent — zero unsafe code, composable pipelines, comprehensive test suite. The performance claims are exciting. My concerns are primarily about presentation, reproducibility, and positioning relative to existing software.

---

### Strengths

1. **Excellent software design**: The trait-based architecture (Learner, TrainedModel, Measure, Transformer, Resample) is clean, composable, and extensible. This follows best practices from mlr3 while being idiomatic Rust.

2. **Zero unsafe code**: 13,377 lines without a single `unsafe` block is remarkable for a performance-critical numerical library. This is the strongest argument for Rust in scientific computing.

3. **Comprehensive feature set**: The framework goes beyond what linfa offers and includes unique features (GeoXGBoost, CausalForest, Conformal Prediction) not available in any other Rust framework.

4. **250 integration tests**: Good coverage for a research software project.

---

### Major Issues

**M1. Reproducibility package incomplete**

JSS requires a complete replication script. The paper references benchmarks but does not provide:
- A single script that reproduces all tables and figures
- Instructions for installing smelt-ml and running benchmarks
- Expected runtime for reproduction
- Exact hardware specification (CPU model, cache sizes — crucial for cache optimization claims)

*Please provide a `replication.sh` or equivalent that reproduces Tables 2-4.*

**M2. Missing comparison with existing Rust ML software**

The paper mentions linfa only in the introduction. A proper comparison should include:
- Feature-by-feature comparison table
- Performance comparison on shared algorithms (KNN, Decision Tree, KMeans)
- API design comparison
- This is standard for JSS software papers.

---

### Minor Issues

**m1.** The `Cargo.toml` shown should include the exact version of smelt-ml used, and the paper should note which version was benchmarked.

**m2.** The Quick Start example (Section 2.2) uses `rust>` as the prompt. JSS convention is to use the language-standard prompt. For Rust, there is no standard REPL prompt — consider removing the prompt or using `//` comments instead.

**m3.** The `Pipeline` composability is mentioned but not demonstrated with code. Please add a pipeline example:
```rust
let pipe = Pipeline::new(
    vec![Box::new(StandardScaler::new())],
    Box::new(XGBoost::new()),
);
```

**m4.** Section 5 (Unique Features) mentions Geographical-XGBoost as "first implementation outside the original Python library" — please cite the original Python package (geoxgboost on PyPI) for completeness.

**m5.** The paper does not discuss backwards compatibility or versioning strategy. For a crate published on crates.io, users need to know about semver compliance.

**m6.** Consider adding a "Comparison with scikit-learn and mlr3" table showing feature parity (which algorithms are shared, which are unique to each framework).

---

### Questions for the Authors

1. Is there a plan for Python bindings (PyO3)? This would dramatically increase adoption.
2. How does the framework handle categorical features natively (beyond CatBoost's target statistics)?
3. What is the memory footprint compared to the C++ libraries for the same datasets?

---

## Fase 5: Section Editor — Decisión

**Decision**: **MAJOR REVISION**

Dear Dr. Parra,

Thank you for your submission to the Journal of Statistical Software. Your manuscript has been reviewed by two experts in gradient boosting and ML software engineering.

Both reviewers found the work to be a novel and potentially significant contribution — the first comprehensive ML framework in Rust with performance competitive with C++ implementations. However, they raise several important concerns that need to be addressed:

### Required Changes

1. **Expand benchmarks** (Reviewer 1, M1): Include larger datasets (N ≥ 50,000), more features, and multi-run statistics with standard deviations. This is essential to support the performance claims.

2. **Clarify CatBoost implementation** (Reviewer 1, M5): Either implement full ordered boosting or clearly position the implementation as "CatBoost-inspired symmetric GBM." The current naming may confuse readers.

3. **Multi-threaded benchmarks** (Reviewer 1, M2): At minimum, acknowledge the limitation or include rayon-based parallel benchmarks.

4. **Complete reproducibility package** (Reviewer 2, M1): Provide a single script that reproduces all tables with exact hardware specifications. This is a JSS requirement.

5. **Comparison with linfa** (Reviewer 2, M2): Include a proper comparison with the primary Rust ML competitor.

6. **Feature comparison tables** (Reviewer 1, M4; Reviewer 2, m6): Add tables comparing implemented vs. not-implemented features, and comparison with scikit-learn/mlr3.

### Recommended Changes

7. Add statistical rigor to benchmarks (multiple runs, CI).
8. Expand unique features section or refocus paper on gradient boosting.
9. Fix LightGBM speedup claims to be more precise.
10. Add pipeline code example.

Please submit your revised manuscript within **6 months**. Include a point-by-point response to each reviewer comment.

Sincerely,
Section Editor

---

## Fase 6: Checklist de Respuesta del Autor

### CRÍTICOS (sin estos, rechazo seguro)

- [x] **R1-M1: Benchmarks grandes (N≥50K, 100K)** — Benchmarks ejecutados en N=500 a N=100K. Tablas expandidas en article.tex con 6 tamaños. Resultados honestos: Rust es más rápido a N≤5K-10K, C++ gana a N≥50K. Subsección "Scaling analysis" agregada.
- [x] **R1-M3: Estadísticas de benchmarks** — Cada benchmark corrido 10x. Tablas reportan mean ± std. Resultados guardados en paper/replication/benchmark_{rust,cpp}_results.json.
- [x] **R1-M5: Clarificar CatBoost** — Agregada Section 3.5 "Scope of the CatBoost implementation" en article.tex. Clarificado en abstract, Table 1 (footnote), y limitations. Doc comments en catboost.rs actualizados. Nombre struct se mantiene por compatibilidad semver pero se califica como "CatBoost-inspired symmetric GBM with ordered target statistics".
- [x] **R2-M1: Script de reproducibilidad** — Creado `paper/replication/` con: `replicate.sh` (master), `benchmark_cpp.py`, `benchmark_large.rs`, `compare_results.py`, `README.md` con hardware specs exactos (i7-1270P, 40GB RAM, L1/L2/L3). Reproduce Tables 2-3.

### IMPORTANTES (sin estos, major revision se mantiene)

- [x] **R1-M2: Multi-threaded benchmarks** — Declarado explícitamente scope = single-threaded en Section 4.1. Nota sobre qué algoritmos usan rayon (RF, ExtraTrees, ObliqueForest para árboles; boosting para features). Claims aplican solo a single-thread.
- [x] **R1-M4: Tabla de features implementadas vs no-implementadas** — Table 4 (tab:features) agregada en Section 3.7. Lista 23 features para los 3 engines: core algorithm, regularization, data handling, training control, y features no implementadas (monotone, GPU, distributed, etc.).
- [x] **R2-M2: Comparación con linfa** — Table 3 (tab:linfa) agregada en Section 2.4. Comparación de 20 categorías: smelt-ml tiene 27 vs 9 learners, más gradient boosting, causal, survival, spatial ML. linfa ventaja en kernel SVM y GMM/OPTICS.

### MENORES (resolver para aceptación final)

- [x] **R1-m1: Evidencia de auto-vectorización** — Benchmark con/sin `target-cpu=native`: XGBoost 24% más rápido (755→575ms). Agregado en Discussion.
- [x] **R1-m2: Precisar claims de LightGBM** — Claims antiguas eliminadas. Nuevas tablas con datos honestos por tamaño. "Scaling analysis" reemplaza generalizaciones.
- [x] **R1-m3: Mismos splits de CV** — Nota agregada en Table caption: "stratified 5-fold CV with seed 42".
- [x] **R2-m2: Quitar prompt `rust>` del code** — Eliminado `rust>` y `+` de todos los code chunks (Quick Start + optimization section).
- [x] **R2-m3: Agregar pipeline example** — Code chunk con Pipeline + StandardScaler + XGBoost agregado después de Quick Start.
- [x] **R2-m4: Citar geoxgboost PyPI** — Referencia `geoxgboost:2024` agregada a refs.bib y citada en Section 5.1.
- [x] **R2-m6: Tabla comparativa con scikit-learn/mlr3** — Table 3 expandida a 4 frameworks: smelt-ml, linfa, scikit-learn, mlr3. Incluye notas sobre extensiones.
- [x] **R1-m5: Comparación de cobertura de tests** — "320 integration tests" documentado en Discussion con detalle de edge cases.

### RESPUESTAS A PREGUNTAS DE REVIEWERS

- [x] **R1-Q1**: ¿Qué pasa a N=100K? — Benchmarqueado. C++ es 2-3x más rápido a N=100K. Reportado honestamente en "Scaling analysis".
- [x] **R1-Q2**: ¿Por qué C++ no usa column-major? — Respondido en Scaling analysis: legacy + GPU coalesced access + SIMD intrinsics escritos para row-major.
- [x] **R1-Q3**: Accuracy de CatBoost en categoricals de alta cardinalidad — smelt 89.7% vs oficial 89.8% en 100 categorías. Agregado en Section 3.5.
- [x] **R2-Q1**: ¿Plan de Python bindings? — Respondido en Discussion: "Python bindings via PyO3 planned as smelt-py".
- [x] **R2-Q2**: ¿Categorical features más allá de CatBoost? — Respondido en Discussion: OneHotEncoder + LabelEncoder + CatBoost ordered target stats.
- [x] **R2-Q3**: ¿Memory footprint? — Rust 5-8 MB vs C++ 176-220 MB (N=10K). Agregado en Scaling analysis.

---

### Progreso

**Estado**: En progreso
**Última actualización**: 2026-04-01

| Item | Status | Comentario |
|------|:---:|-----------|
| R1-M1 Benchmarks N≥50K | ✅ | N=500 a 100K, 6 tamaños, clasificación + regresión |
| R1-M3 Stats (10x runs) | ✅ | mean ± std, 10 runs each, resultados en JSON |
| R1-M5 CatBoost naming | ✅ | Sección 3.5 + abstract + footnote en Table 1 + doc comments |
| R2-M1 Reproducibility | ✅ | replicate.sh + benchmark scripts + README con hardware |
| R1-M2 Multi-thread | ✅ | Scope declarado single-thread + nota sobre rayon |
| R1-M4 Feature tables | ✅ | Table 4: 23 features × 3 engines |
| R2-M2 Comparar linfa | ✅ | Table 3: 20 categorías, smelt-ml 27 vs linfa 9 |
| R1-m1 Auto-vectorization | ✅ | XGBoost 24% speedup con target-cpu=native |
| R1-m2 LightGBM claims | ✅ | Claims reemplazadas por datos honestos |
| R1-m3 Same CV splits | ✅ | seed 42 documentado en tabla |
| R2-m2 Remove rust> prompt | ✅ | Prompts eliminados de todos los code chunks |
| R2-m3 Pipeline example | ✅ | Pipeline + StandardScaler + XGBoost |
| R2-m4 Cite geoxgboost | ✅ | geoxgboost:2024 en refs.bib |
| R2-m6 Framework comparison | ✅ | Tabla 4 frameworks: smelt, linfa, sklearn, mlr3 |
| R1-m5 Test coverage | ✅ | 320 tests documentados en Discussion |
| R1-Q1 N=100K perf | ✅ | C++ 2-3x más rápido; reportado en Scaling analysis |
| R1-Q2 Why not C++ col-major | ✅ | Legacy + GPU focus + SIMD row-major |
| R1-Q3 CatBoost categoricals | ✅ | 89.7% vs 89.8% oficial (100 categorías) |
| R2-Q1 Python bindings plan | ✅ | PyO3 smelt-py en Discussion |
| R2-Q2 Categorical handling | ✅ | 3 mecanismos documentados |
| R2-Q3 Memory footprint | ✅ | Rust 5-8 MB vs C++ 176-220 MB |
