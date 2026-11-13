# Diagnóstico Moran's I de residuos (2026-11-12)

Motivación: la opción (c) de la lista de candidatos para smelt tras cerrar `search_bandwidth`.
Antes de este cambio, smelt tenía learners espaciales (`GeoXGBoost`, `KrigingHybrid`) y CV
espacial, pero ninguna forma de responder la pregunta previa: ¿quedan residuos espacialmente
autocorrelacionados después de un modelo no espacial? Es el diagnóstico estándar para
justificar un modelo espacial y, después, para verificar que el modelo espacial de verdad
eliminó la estructura.

## Qué se implementó (`src/measure/moran.rs`)

- `morans_i(values, coords, &SpatialWeights) -> MoranResult`: estadístico I con inferencia
  analítica de Cliff & Ord (1981) bajo los dos supuestos, aleatorización (campos principales
  `variance`, `z_score`, `p_value`) y normalidad (`*_normal`). p-valores bilaterales. El
  supuesto de aleatorización es el principal porque los residuos de un modelo ML rara vez
  son normales; es también el default de `spdep::moran.test` y lo que reporta `esda.Moran`
  como `z_rand`/`p_rand`.
- `morans_i_permutation(..., n_permutations, seed)`: añade un pseudo p-valor por
  permutaciones (bilateral, 2 × cola menor, con corrección +1). Determinista por semilla e
  independiente del número de hilos (un `StdRng` por permutación; test que lo verifica con
  un pool de 1 hilo).
- `SpatialWeights`: `KNearest { k }` (default k = 8, siempre conexo), `InverseDistance {
  bandwidth, power }` y `DistanceBand { bandwidth }`. Todas estandarizadas por fila.
  Islas (filas sin vecinos) quedan en cero y no contribuyen, igual que libpysal. Puntos
  coincidentes bajo distancia inversa son error explícito, no peso infinito.
- `MoranResiduals`: wrapper `Measure` (`id = "regr.moran_residuals"`) que puntúa una
  `Prediction::Regression` con verdad por el I de `truth - predicted`. Devuelve el I con
  signo; `maximize = false` solo para que las tablas de benchmark ordenen primero los
  residuos menos agrupados en el caso positivo habitual. Está documentado como diagnóstico,
  no como objetivo de tuning. `test()` y `test_permutation()` devuelven el `MoranResult`
  completo.
- Python (`smelt-py/src/measures.rs`): `morans_i(values, coords, weights="knn", k=8,
  bandwidth=None, power=1.0, n_permutations=0, seed=42)` y `moran_residuals(y_true, y_pred,
  coords, ...)`, ambas devuelven `MoranResult` (clase expuesta). Nombres de esquema
  validados eagerly con `ValueError`, mismo patrón `resolve_*` que `objective` de XGBoost.

## Limitación de diseño documentada (y su relación con la opción (a))

`MoranResiduals` exige que las filas de la predicción estén alineadas una a una con las
coordenadas con que se construyó. Sirve para un hold-out cuyas coordenadas se tienen, pero
**no** para las predicciones por fold de `benchmark::resample_regress`, porque la
`Prediction` no recuerda de qué filas de la `Task` salió. Para CV hay que tomar los splits
del `Resample`, rebanar las coordenadas con cada `test_idx` y llamar a `morans_i` sobre
los residuos del fold. Es exactamente el mismo hueco que impide componer `GeoXGBoost` con
`SpatialBlockCV` en el benchmark (candidato (a): trait `SpatialLearner` o coords en `Task`);
si se cierra ese hueco, este wrapper se beneficia sin cambios de API.

## Verificación

- Golden tests contra `esda 2.10.0` + `libpysal 4.15.1` (venv temporal, no es dependencia
  del proyecto) sobre una fixture de 20 sitios generada con un LCG fijo: kNN k = 4 y k = 1,
  distancia inversa con y sin ancho de banda, banda binaria. Se comprueban I, ambas
  varianzas y ambos z a 1e-9 / 1e-8, y el p-valor a 1e-6 (pasa por la aproximación
  Abramowitz & Stegun 7.1.26 de `erf`, la misma de `hoeffding.rs`). Más un caso de ruido
  independiente del espacio (I = −0,087, p = 0,78) y la determinación del p permutacional.
- Un defecto encontrado por los golden tests y corregido antes de terminar: S1 corre sobre
  pares **ordenados**; con pesos kNN (asimétricos) un enlace en una sola dirección debe
  aportar `(w_ij + w_ji)²` completo, no la mitad. Los casos simétricos (distancia inversa,
  banda) pasaban y ocultaban el error; solo kNN lo destapó. Razón de más para tener
  referencias con pesos asimétricos.
- Python: `maturin develop --release` + script directo que reproduce los valores de esda a
  través de la API y un caso realista (target con campo espacial oculto: regresión lineal
  sin coordenadas deja I claramente positivo y significativo; un RandomForest con las
  coordenadas como features lo reduce a menos de la mitad).

## Referencias

- Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. *Biometrika* 37, 17–23.
- Cliff, A. D. & Ord, J. K. (1981). *Spatial Processes: Models and Applications*. Pion.
- Rey, S. J. & Anselin, L. (2007). PySAL: A Python library of spatial analytical methods.
  *Review of Regional Studies* 37, 5–27 (implementación de referencia, `esda.Moran`).
