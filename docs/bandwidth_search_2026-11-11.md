# Búsqueda rápida de ancho de banda en GeoXGBoost (2026-11-10/11)

Motivación: George Grekousis preguntó (2026-11-05) cómo acelerar la selección de k en
`geoxgboost`, cuyo procedimiento es una grilla exhaustiva de paso 10 seguida de una de paso 1
alrededor del mínimo (unas 48 evaluaciones para 30 a 300, cada una con n modelos locales), con
miras al seed grant de extensión de Geo-XGBoost. Esto es la implementación en smelt y la
medición sobre los datos árticos (cationes n = 1.731, DOC n = 1.118).

## Qué se implementó (`src/learner/geo_xgboost.rs`)

- `NeighborIndex`: distancias ordenadas una vez por punto (O(n² log n) una sola vez), rebanadas
  para cualquier k (vecindarios anidados). Reproduce exactamente `spatial_weights` (test con
  coordenadas duplicadas). Lo usan `select_bandwidth`, `search_bandwidth` y `train_geo`.
- `GeoXGBoost::search_bandwidth(task, &BandwidthSearch)`:
  1. Pasada gruesa **completa** al paso configurado (defecto 20), visitada en orden de
     subdivisión binaria (extremos, centro, cuartos...) para tener pronto una referencia buena.
  2. Sección áurea entera dentro del intervalo entre los vecinos de grilla del mínimo, hasta
     `tolerance` vecinos (defecto 2), con memoización.
  3. Poda: cada candidato se evalúa en paralelo sobre los sitios en orden aleatorio fijo; cuando
     ≥ 20 % de los sitios (y ≥ 50) están evaluados, se compara pareado con los errores del mejor
     candidato en los mismos sitios; si la diferencia media es positiva con t > 3, se levanta una
     bandera y los sitios no iniciados se saltan. Los podados se reportan aparte y nunca se eligen.
- Python: `GeoXGBoost.search_bandwidth(x, y, coords, min_bw=30, max_bw=300, step=20, tol=2,
  early_stop=True, seed=42)` devuelve `best`, `best_score`, `bandwidths`, `cv`, `pruned`,
  `local_fits`, `exhaustive_local_fits`.

## Dos defectos encontrados y corregidos en el primer benchmark

1. **La curva LOO no es unimodal en rango amplio.** Cationes: mínimo local en k = 30 (0,692),
   joroba en 40-100 (0,70-0,71), mínimo global en 170 (0,669). Una salida anticipada de la pasada
   gruesa ("dos peores seguidos y paro") se quedó en 30; en DOC se quedó en 220 (óptimo 291). Por
   eso la pasada gruesa recorre toda la grilla, como hace George.
2. **La poda no actuaba en orden ascendente**: en un flanco descendente cada candidato era el
   nuevo mejor. Por eso el orden de subdivisión binaria.

## Medición (ajustes de modelos locales; los segundos no sirven: la carga del equipo varió entre 8 y 60 por otras sesiones)

| Procedimiento | Cationes: k, LOO, ajustes | DOC: k, LOO, ajustes |
|---|---|---|
| Grilla paso 10 (60-300) + refinamiento ±10 paso 2 (33 cand.) | 170, 0,6686, 57.123 | 290, 0,7142, 36.894 |
| Referencia Grekousis: paso 10 + 20 cand. paso 1 (48 cand., estimado) | 170, ≈ 83.000 | 290, ≈ 54.000 |
| `search_bandwidth`, paso 10 | 170, 0,6686, 54.822 (31 + 3 podados) | 291, 0,7106, 35.161 (25 + 8) |
| `search_bandwidth`, **paso 20 (defecto)** | 170, 0,6686, 36.371 (19 + 4) | 291, 0,7106, 24.527 (20 + 2) |
| `search_bandwidth`, paso 30 | 180, 0,6719 (pierde 0,003) | 291, 0,7106, 20.064 (17 + 1) |
| Grilla paso 1 exhaustiva (30-300) | 469.101 | 302.978 |

Lectura: con paso 20 se recupera el óptimo de paso 1 en ambos targets con 36 % menos ajustes que
la grilla de paso 10 + 2 y ~55 % menos que la referencia; la sección áurea hace el refinamiento en
6 a 9 evaluaciones en vez de 20; la poda aporta poco (2 a 8 candidatos) porque solo actúa sobre
los claramente peores. Paso 30 ya pierde el mínimo de cationes por una ondulación local: la
sección áurea es sensible a curvas no suaves dentro del intervalo.

## Para la propuesta del seed grant

- El procedimiento exhaustivo de `geoxgboost` se puede reemplazar por pasada gruesa de paso 20 +
  sección áurea + poda pareada sin cambiar el resultado, con la mitad de los modelos locales.
- Además, la implementación Python recalcula las distancias y las ordena por cada (punto, k); el
  índice de vecinos anidados elimina ese costo y es portable a Python (NumPy argsort una vez).
- Lo que no se midió limpio: el tiempo de reloj por candidato con y sin índice (hace falta una
  máquina sin carga externa); en una ventana tranquila cada candidato de cationes costó 22 s en
  smelt, pero la misma función tomó 46 a 109 s con carga de 30 a 60.

Archivos: `artic/results/bandwidth_search_benchmark.csv`, `artic/results/log.txt` (líneas
`bench`), `examples/arctic_lakes/bench_bandwidth_search.py`, `bench_bandwidth_step20.py`.
