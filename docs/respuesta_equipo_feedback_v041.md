# Respuesta al Feedback — Pipeline J. Hydrology

**Fecha**: 2026-04-17
**Versión**: smelt-py v0.4.1 (commits `8a80a99` + `8e11af9`)
**Estado**: Todos los bugs arreglados. Nuevas features para el paper agregadas.

---

## Resumen Ejecutivo

Los 9 items del feedback están resueltos. El pipeline ML completo ahora es más robusto:
filtros correctos con k=p, SHAP con nombres reales, benchmark() que no crashea con
learners incompatibles, numpy arrays aceptados en coords, BayesianOptimizer retornando
ints correctos, y ConformalRegressor + Spatial Leave-One-Out expuestos en Python.

### Para actualizar

```bash
cd smelt/smelt-py
git pull
source .venv/bin/activate
maturin develop --release
```

---

## Bugs

### 1. Filtros con k=p retornan orden de columna — RESUELTO ✅

**Causa raíz**: `FilterSelector::fit_supervised` hacía `indices.sort()` al final para
preservar orden de columnas al usarse como Transformer en pipelines. Cuando k=p, esto
retornaba todas las features en orden de columna, enmascarando los scores reales.

**Fix**: El binding Python ahora **bypassa** `FilterSelector` y llama directamente al
trait `Filter::score()`. Los scores se ordenan descendente y se retornan top-k.

```python
# Antes (bug): todos los filtros retornan el mismo orden con k=p
smelt.filter_mrmr(X, y, names, k=39)    # → [elevacion, aspect, average_normal, ...]
smelt.filter_relief(X, y, names, k=39)  # → [elevacion, aspect, average_normal, ...] ← idéntico

# Ahora: cada filtro retorna su propio ranking
smelt.filter_mrmr(X, y, names, k=5)    # → [('slope', 5.0), ('noise1', 4.0), ...]
smelt.filter_relief(X, y, names, k=5)  # → [('slope', -0.536), ('noise3', -0.559), ...]
```

**Nota**: Puedes quitar el workaround `filter_k = min(p-1, ...)` de tu código — `k=p`
ahora funciona correctamente.

---

### 2. SHAP global_importance con nombres genéricos — RESUELTO ✅

**Fix**: `shap_impl()` ahora pasa `feature_names` al Task via `with_feature_names()`.

```python
shap = model.shap_values(X, y, n_background=50, feature_names=["slope", "aspect", "hand"])
shap["global_importance"]  # → [("slope", 2.0), ("aspect", 1.5), ("hand", 1.2)]
```

---

### 3. GaussianNB no rechaza regresión gracefully — RESUELTO ✅

**Fix doble**:

1. **Flags de soporte** en cada learner (hardcoded por tipo):
```python
XGBoost().supports_classification       # True
XGBoost().supports_regression           # True
GaussianNB().supports_regression        # False
LinearRegression().supports_classification  # False
```

2. **benchmark() auto-skip**: Detecta antes de entrenar y salta con mensaje claro:
```
Learner                       rmse            r2  note
XGBoost               1.194±0.054  0.838±0.013
RandomForest          1.225±0.274  0.831±0.049
LinearRegression      0.480±0.069  0.972±0.010
GaussianNB                    n/a          n/a  (skipped: incompatible)
LogisticRegression            n/a          n/a  (skipped: incompatible)
```

El resultado incluye `results[name]["_skipped"]` con el motivo legible.
Si aún así falla por otra razón, se captura como `results[name]["_error"]` (no crashea).

---

### 4. Filtros retornan (name, column_index) — RESUELTO ✅

**Fix**: Ahora retornan `(name, score)` con el score real del filtro.

```python
# Antes: (name, column_index)
smelt.filter_mrmr(X, y, names, k=5)
# → [('aspect', 1), ('edge_density', 8), ...]    ← índice de columna

# Ahora: (name, score)
smelt.filter_mutual_information(X, y, names, k=5)
# → [('slope', 0.839), ('noise1', 0.533), ('elevation', 0.426), ...]
smelt.filter_mrmr(X, y, names, k=5)
# → [('slope', 5.0), ('noise1', 4.0), ('elevation', 3.0), ...]  ← selección greedy: p-rank
```

**Nota**: Los scores son comparables intra-filtro pero NO inter-filtro (escalas distintas).
MI usa valores absolutos de información mutua; MRMR/JMI/... usan `p - rank` por la
naturaleza secuencial de la selección greedy. Para comparar entre filtros, normaliza
o usa ranks (lo que ya hace `cumulative_ranking()` internamente).

---

## Usabilidad API

### 5. SpatialBlockCV acepta numpy array — RESUELTO ✅

Nuevo helper `parse_coords()` interno acepta **3 formatos**:

```python
# Todos funcionan ahora:
smelt.SpatialBlockCV(n_folds=5, coords=np_array)          # numpy (N, 2)
smelt.SpatialBlockCV(n_folds=5, coords=list_of_tuples)    # [(x, y), ...]
smelt.SpatialBlockCV(n_folds=5, coords=list_of_lists)     # [[x, y], ...]
```

Mismo comportamiento en **`SpatialBufferCV`** (nuevo, ver sección 8).

---

### 6. BayesianOptimizer: ints para parámetros enteros — RESUELTO ✅

Los parámetros con nombres conocidos como enteros (`n_estimators`, `max_depth`, `depth`,
`num_leaves`, `k`, `min_samples_split`, `min_samples_leaf`, `n_features`, `seed`,
`random_state`) se castean automáticamente a `int`:

```python
result = bo.optimize('rf', {'n_estimators': (50, 300), 'max_depth': (3, 15)}, X, y)
result["best_params"]  # → {"n_estimators": 164, "max_depth": 8}   ← int, no float
```

Los demás parámetros (learning_rate, lambda, etc.) se mantienen como `float`.

---

### 7. benchmark() retorna {mean, std, folds} — RESUELTO ✅

Estructura anterior reemplazada. Formato nuevo:

```python
results = benchmark({'RF': RandomForest(), 'XGB': XGBoost()}, X, y, cv=5,
                    metrics={'rmse': rmse_score})

results["RF"]["rmse"]["mean"]    # → 5.88
results["RF"]["rmse"]["std"]     # → 0.43
results["RF"]["rmse"]["folds"]   # → [5.2, 6.1, 5.8, 6.4, 5.9]
```

`benchmark_table()` sigue funcionando igual visualmente.

---

### 8. ConformalRegressor en Python — RESUELTO ✅

Expuesto como método `.conformal_predict()` en los 11 learners de regresión:

```python
from smelt import RandomForest

model = RandomForest(n_estimators=50)
model.fit(X_train, y_train)

# Split conformal prediction
cp = model.conformal_predict(X_cal, y_cal, X_test, alpha=0.1)
# cp["predictions"]     → np.array (n_test,)  ← punto estimado
# cp["lower"]           → np.array (n_test,)  ← límite inferior
# cp["upper"]           → np.array (n_test,)  ← límite superior
# cp["interval_width"]  → float               ← ancho calibrado
# cp["alpha"]           → 0.1                 ← miscoverage

# Cobertura empírica
import numpy as np
coverage = ((y_test >= cp["lower"]) & (y_test <= cp["upper"])).mean()
# Target: ≥ 1 - alpha = 0.9
```

Solo regresión por ahora. Si llamas en un modelo de clasificación, error claro:
`"conformal_predict is only available for regression models"`.

---

### 9. benchmark() acepta CV object directamente — YA FUNCIONABA ✅

Verificado — siempre funcionó, el código usa `isinstance(cv, int)`:

```python
scv = smelt.SpatialBlockCV(n_folds=5, coords=coords)
results = benchmark(learners, X, y, cv=scv, metrics={'rmse': rmse_score})
# OK, usa directamente el objeto CV
```

También acepta `cv=5` + `coords=coords` (atajo que construye SpatialBlockCV internamente).

---

## Feature Nueva (solicitada)

### Spatial Leave-One-Out con buffer — AGREGADO ✅

**Opción 1**: `SpatialBufferCV` directo (k-fold con exclusión de buffer):

```python
from smelt import SpatialBufferCV

sbc = SpatialBufferCV(
    n_folds=5,
    coords=coords,         # numpy, tuples, o listas
    buffer_distance=5000,  # metros (o la unidad de tus coords)
    seed=42,
)
splits = sbc.splits(n_samples)
```

**Opción 2**: Helper de conveniencia para SLOO (`n_folds = n_samples`):

```python
from smelt import spatial_leave_one_out

sloo = spatial_leave_one_out(coords, buffer_distance=5000)
# n=60 → 60 folds, test_size=1, train excluye vecinos <5km

# Úsalo en benchmark directamente
results = benchmark(learners, X, y, cv=sloo, metrics={...})
```

Usa esto para datasets pequeños (n<100) donde SpatialBlockCV es demasiado agresivo.

---

## Pipeline Recomendado (actualizado)

```python
import smelt_ml as sml
from smelt import cumulative_ranking, BayesianOptimizer, spatial_leave_one_out
import numpy as np

# 1. Feature selection con 10 filtros (k=p ahora funciona)
ranking = cumulative_ranking(X, y, names, top_k=15, corr_cutoff=0.9)
selected = ranking["feature"].tolist()
X_sel = X[:, [names.index(f) for f in selected]]

# 2. CV estrategia: SLOO para n pequeño
sloo = spatial_leave_one_out(coords, buffer_distance=5000)

# 3. Benchmark (learners incompatibles se skipean automáticamente)
results = sml.benchmark(
    {
        "XGB": sml.XGBoost(), "RF": sml.RandomForest(), "CB": sml.CatBoost(),
        "LGB": sml.LightGBM(), "ET": sml.ExtraTrees(), "DT": sml.DecisionTree(),
        "KNN": sml.KNearestNeighbors(), "Ridge": sml.Ridge(),
        # GaussianNB / LogReg skippean solos
    },
    X_sel, y, cv=sloo,
    metrics={"rmse": sml.rmse_score, "r2": sml.r2_score},
)
print(sml.benchmark_table(results))

# 4. Tuning del mejor
bo = BayesianOptimizer(n_iter=100, seed=42)
best = bo.optimize(
    "rf",
    {"n_estimators": (100, 1000), "max_depth": (3, 20)},  # ints auto
    X_sel, y, metric="rmse", n_folds=5,
)
# best["best_params"]["n_estimators"] es int ahora

# 5. Entrenar, SHAP, permutation, conformal
model = sml.RandomForest(**best["best_params"])
model.fit(X_train, y_train)

shap = model.shap_values(X_train, y_train, feature_names=selected)  # names reales
perm = model.permutation_importance(X_test, y_test, metric="rmse", feature_names=selected)
cp = model.conformal_predict(X_cal, y_cal, X_test, alpha=0.1)

# Mapa de incertidumbre
uncertainty = (cp["upper"] - cp["lower"]) / 2
```

---

## Referencia rápida de cambios

| API | Antes | Ahora |
|-----|-------|-------|
| `filter_*(X, y, names, k=p)` | bug: orden columna | ✅ orden por score |
| Retorno de filters | `(name, column_idx)` | `(name, score)` |
| `model.shap_values(..., feature_names=...)` | `global_importance` usa `x0, x1...` | nombres reales |
| `SpatialBlockCV(coords=np_array)` | `TypeError` | ✅ acepta numpy |
| `SpatialBufferCV` | no disponible | ✅ disponible |
| `spatial_leave_one_out()` | no disponible | ✅ disponible |
| `benchmark()` (learner incompatible) | `RuntimeError` crash | ✅ skip + `_skipped` |
| `benchmark()["RF"]["rmse"]` | `[5.2, 6.1, ...]` | `{"mean": 5.88, "std": 0.43, "folds": [...]}` |
| `BayesianOptimizer.optimize()["best_params"]` | `{"n_estimators": 164.2}` | `{"n_estimators": 164}` |
| `model.conformal_predict(...)` | no disponible | ✅ disponible |
| `learner.supports_classification/regression` | no disponible | ✅ disponible |

## Notas técnicas

- Version string aún `0.3.0` en `Cargo.toml` de smelt-py por compatibilidad durante
  desarrollo local. Subirá a `0.4.1` al publicar en PyPI.
- Tests Rust: 11/11 filter tests pasan + 6/6 GeoXGBoost + 3/3 mutual_info.
- No hay regresiones conocidas. Todos los cambios son aditivos o reparan bugs.
