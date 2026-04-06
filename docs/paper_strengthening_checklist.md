# Paper Strengthening Checklist

Análisis exhaustivo del paper JSS para smelt-ml v1.3.0 (16pp, 10 tablas, 1 figura).
Fecha: 2026-04-06

---

## Debilidades Identificadas por Prioridad

### CRÍTICAS (un reviewer rechazaría por esto)

- [ ] **D1: Caso de uso no demuestra valor** — GeoXGBoost = XGBoost en RMSE (0.152), spatial leakage solo 3%, conformal coverage 84% < 90% target. El caso de uso FALLA en demostrar la integración.
  - **Fix**: Dataset más grande (1K+ samples) con estructura espacial fuerte donde GeoXGBoost supere a XGBoost y el leakage sea >10%.
  - **Impacto**: Alto — sin esto, la contribución #1 (integrated pipeline) no tiene evidencia.

- [ ] **D2: Sin ablation study** — 3 optimizaciones (column-major, u8, hist subtraction) presentadas juntas sin cuantificar el impacto individual.
  - **Fix**: Benchmark con y sin cada optimización → tabla mostrando contribución de cada una.
  - **Impacto**: Alto — contribución empírica genuina, esperada por reviewers técnicos.

### IMPORTANTES (major revision sin estos)

- [ ] **D3: XGBoost "research-typical" es vago** — 0.7x a N=5K clasificación no es "competitive". El crossover está en N=1K.
  - **Fix**: Ser más preciso: "competitive at N≤1K for classification, N≤5K for regression". O mejorar XGBoost con data-parallel histograms.
  - **Impacto**: Medio — honestidad > marketing.

- [ ] **D4: LightGBM es peso muerto** — nunca gana, ocupa espacio en todas las tablas.
  - **Fix**: Opción A: mover a apéndice/supplementary. Opción B: mejorar LightGBM (EFB, weighted GOSS). Opción C: reducir a una mención breve.
  - **Impacto**: Medio — simplifica el paper y elimina distracción.

- [ ] **D5: 10 tablas excesivas** — para 16 páginas, demasiadas tablas diluyen el mensaje.
  - **Fix**: Consolidar Tables 5-6-7 (training time + significance) en una sola. Mover feature comparison tables a appendix.
  - **Impacto**: Medio — mejora legibilidad.

### MENORES (para aceptación final)

- [ ] **D6: Scaling analysis se repite** — lines 500-510 y 529-536 dicen lo mismo sobre SIMD.
  - **Fix**: Unificar en un solo párrafo.

- [ ] **D7: Falta benchmark de inferencia** — solo training time, no prediction time.
  - **Fix**: Agregar una tabla con prediction time (es trivial de medir).

- [ ] **D8: Solo 20 features** — ¿cómo escala con 50, 100, 500 features?
  - **Fix**: Agregar una fila de benchmark con p=100 features.

- [ ] **D9: Caption de Fig 1 desactualizada** — dice "C++ is faster at N≥50K" pero CatBoost gana a C++ en todas las escalas.
  - **Fix**: Actualizar caption.

- [ ] **D10: Líneas de código desactualizadas** — dice "13,377" en Discussion pero "13,000+" en abstract.
  - **Fix**: Contar y unificar.

---

## Mejoras de Performance (código)

### Rápidas (< 1 hora)

- [ ] **P1: LTO (Link-Time Optimization)** — `lto = true` en `[profile.release]` de Cargo.toml.
  - Impacto esperado: 5-15% en todos los engines.
  - Esfuerzo: 5 min + rebuild.

- [ ] **P2: PGO (Profile-Guided Optimization)** — compilar con profiling, re-compilar con datos de perfil.
  - Impacto esperado: 5-10% adicional.
  - Esfuerzo: 30 min.
  - Comando: `RUSTFLAGS="-Cprofile-generate=/tmp/pgo" cargo build --release && (run benchmarks) && RUSTFLAGS="-Cprofile-use=/tmp/pgo/merged.profdata" cargo build --release`

- [ ] **P3: codegen-units = 1** — mejor inlining, más tiempo de compilación.
  - Agregar a `[profile.release]`: `codegen-units = 1`
  - Impacto: 5-10% (mejor inlining cross-module).

### Medias (1 día)

- [ ] **P4: Data-parallel histogram building** — split indices entre threads, merge parcial histograms.
  - Actualmente: parallel over features (20 features, ~2 batches en 12 cores).
  - Propuesto: parallel over DATA (12 chunks de N/12, cada uno construye histograma parcial, merge al final).
  - Impacto: 1.5-2x a N grande (mejor cache utilization per thread).
  - Solo beneficia N>=10K (overhead de merge para N chico).

- [ ] **P5: Pre-allocate bin_g/bin_h en find_best_histogram_saving** — actualmente alloca Vec<f64> per feature per node.
  - Usar thread-local buffers pre-allocados.
  - Impacto: reducir allocation pressure, ~10% en hot loop.

- [ ] **P6: Sorted-index histogram** — ordenar indices por bin antes de acumular → convierte scatter-add en sequential add.
  - Impacto: potencialmente 2x en el inner loop (mejor branch prediction).
  - Riesgo: el sort es O(n log n), solo vale la pena si n es grande.

### Largas (1 semana+)

- [ ] **P7: EFB para LightGBM** — Exclusive Feature Bundling reduce features efectivos.
  - Impacto: podría hacer LightGBM competitivo.
  - Esfuerzo: alto (algoritmo complejo).

- [ ] **P8: SIMD manual** — `std::arch` para AVX2 scatter-add.
  - Impacto: 2-4x en inner loop.
  - Downside: requiere `unsafe`, rompe la claim de zero unsafe.
  - Alternativa: usar crate `packed_simd2` o esperar estabilización de `std::simd`.

- [ ] **P9: GPU via wgpu/vulkano** — compute shaders para histogram building.
  - Impacto: 10x+ a N grande.
  - Esfuerzo: muy alto. Fuera de scope para v1.x.

---

## Mejoras del Paper (contenido)

### Ablation Study (D2)

Diseño del experimento:
1. **Baseline**: row-major, u16 bins, sin histogram subtraction
2. **+Column-major**: column-major, u16 bins, sin subtraction
3. **+u8 packing**: column-major, u8 bins, sin subtraction
4. **+Hist subtraction**: column-major, u8 bins, con subtraction (actual)

Medir a N=1K, 10K, 100K para XGBoost classification.
Presentar como tabla + gráfico de barras.

Implementación: crear variantes del tree builder con flags para cada optimización.

### Caso de Uso Mejorado (D1)

Opciones de dataset:
1. **Ames Housing** (1,460 samples, 79 features, coords disponibles) — mayor que Meuse, estructura espacial
2. **King County Housing** (21,613 samples, 18 features, lat/lon) — grande, spatial heterogeneity clara
3. **Dataset propio de investigación USACH** — más auténtico, original

Criterios para un buen caso de uso:
- N ≥ 500, idealmente ≥ 1000
- GeoXGBoost RMSE < XGBoost RMSE (demostrar ventaja)
- Spatial leakage ≥ 10% (dramático)
- Conformal coverage ≥ 88% (cercano al target)

### Benchmark de Inferencia (D7)

Medir prediction time para:
- 1000 nuevas muestras
- XGBoost, CatBoost, Random Forest
- smelt-ml vs scikit-learn

Esto es fácil de medir y puede ser un punto fuerte (Rust's prediction es probablemente muy rápida por falta de Python overhead).

---

## Mejoras Estratégicas

- [ ] **E1: Contactar Grekousis** — co-autor, valida GeoXGBoost implementation.
- [ ] **E2: JOSS paper** — 2 páginas, enviar antes que JSS. Publicación rápida (80-90%).
- [ ] **E3: Cover letter JSS** — enfatizar pipeline integrado, no performance race.
- [ ] **E4: Dockerfile** — reproducibilidad exacta.
- [ ] **E5: Push a GitHub** — código actualizado con v1.3.0.
- [ ] **E6: Verificar refs con /verify-refs** — DOIs, URLs rotas.

---

## Orden de Ejecución Recomendado

### Sprint 1 (hoy): LTO + Ablation Study
1. Agregar LTO + codegen-units=1 a Cargo.toml → rebuild
2. Re-benchmark (CPU frío) → si mejora, actualizar tablas
3. Implementar ablation study → tabla nueva en paper

### Sprint 2: Caso de uso + Dataset
4. Buscar dataset con spatial heterogeneity fuerte (King County?)
5. Correr GeoXGBoost vs XGBoost con SpatialCV
6. Si GeoXGBoost gana: reemplazar Meuse. Si no: buscar otro dataset.

### Sprint 3: Limpieza del paper
7. Consolidar tablas (10 → 7-8)
8. Fix caption Fig 1
9. Unificar scaling analysis
10. Actualizar líneas de código

### Sprint 4: Performance
11. PGO benchmarks
12. Data-parallel histograms (si el gap sigue)
13. Re-benchmark final

### Sprint 5: Publicación
14. Cover letter JSS
15. JOSS paper corto
16. Contactar Grekousis
17. Push GitHub + Dockerfile

---

## Métricas de Éxito

| Métrica | Actual | Target |
|---------|--------|--------|
| CatBoost classif vs C++ | 1.0-1.4x (gana) | Mantener |
| XGBoost N=10K vs C++ | 0.7x | ≥ 0.85x con LTO/PGO |
| LightGBM N=10K vs C++ | 0.8x | ≥ 0.9x o reducir presencia en paper |
| Caso de uso: GeoXGBoost vs XGBoost | 0.0% mejora | ≥ 5% mejora |
| Spatial leakage | 3% | ≥ 10% |
| Conformal coverage | 84% | ≥ 88% |
| Tablas en paper | 10 | 7-8 |
| Páginas | 16 | 15-17 (JSS no tiene límite estricto) |
| Probabilidad publicación | ~65-70% | ≥ 80% |
