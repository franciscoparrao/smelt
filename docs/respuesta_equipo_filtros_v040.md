# Respuesta al Reporte: Filtros Info-Teóricos + Extensión Python API

**Fecha**: 2026-04-16
**Commit**: `4c87318` (master, pusheado)
**Versión**: smelt-py v0.4.0

---

## Estado: TODO IMPLEMENTADO Y TESTEADO

Todas las solicitudes del reporte fueron implementadas, compiladas y validadas end-to-end.

---

## Parte 1: Filtros Info-Teóricos + Relief — COMPLETO

### Nuevo módulo compartido: `src/preprocess/mutual_info.rs`
- `discretize()` — binning uniforme en 10 bins
- `mi_from_bins()` — MI estándar desde bins pre-discretizados
- `conditional_mi()` — I(X; Y | Z) vía identidad de entropías
- `joint_mi()` — I(X, Z; Y) = I(Z; Y) + I(X; Y | Z)
- Entropías: marginal, joint (2D), triple (3D)
- 3 unit tests pasan

### 5 filtros nuevos en `src/preprocess/filter.rs`
| Filtro | Algoritmo | Tipo |
|--------|-----------|------|
| **MRMR** | Greedy: max relevancia - mean redundancia | Info-teórico |
| **JMI** | Greedy: sum I(cand, s; y) | Info-teórico |
| **JMIM** | Greedy: min I(cand, s; y) | Info-teórico |
| **CMIM** | Greedy: min I(cand; y \| s) | Info-teórico |
| **Relief** | RReliefF con kernel gaussiano adaptivo, k=10 NN | Distancia |

Todos integrados en `FilterSelector` con constructores: `.mrmr(k)`, `.jmi(k)`, `.jmim(k)`, `.cmim(k)`, `.relief(k)`.

Helper compartido `greedy_select()` para los 4 filtros info-teóricos (evita duplicación).

**5 integration tests nuevos** — todos pasan (11/11 total filter tests).

---

## Parte 2: Extensión Python API — COMPLETO

### Filtros desde Python

```python
from smelt import filter_mrmr, filter_jmi, filter_jmim, filter_cmim, filter_relief
result = filter_mrmr(X, y, feature_names, k=15)  # → [(name, idx), ...]
```

Los 10 filtros están expuestos: `filter_variance`, `filter_correlation`, `filter_anova_f`, `filter_information_gain`, `filter_mutual_information`, `filter_mrmr`, `filter_jmi`, `filter_jmim`, `filter_cmim`, `filter_relief`.

### Cumulative Ranking

```python
from smelt import cumulative_ranking
ranking = cumulative_ranking(X, y, feature_names, filters=None, top_k=15, corr_cutoff=0.9)
# → DataFrame con: feature, cumulative_rank, rank_variance, rank_mrmr, ...
```

Ubicación: `smelt-py/python/smelt/filters.py`

### SHAP Values

```python
model = RandomForest(n_estimators=50); model.fit(X, y)
shap = model.shap_values(X, y, n_background=50, feature_names=names)
# shap["values"]             → np.array (n_samples, n_features)
# shap["base_value"]         → float
# shap["global_importance"]  → [(name, mean_abs_shap), ...]
```

Disponible en los **11 learners** como método (via macro, zero duplicación).

### Permutation Importance

```python
perm = model.permutation_importance(X, y, metric="rmse", n_repeats=5, feature_names=names)
# → [{"feature": "slope", "importance": 3.32, "std_dev": 0.13}, ...]
```

Métricas soportadas: `rmse`, `mae`, `r2`, `mape`, `accuracy`, `f1`, `precision`, `recall`, `logloss`, `auc`.

### BayesianOptimizer (TPE)

```python
from smelt import BayesianOptimizer
bo = BayesianOptimizer(n_iter=100, n_initial=10, seed=42)
result = bo.optimize(
    "rf",  # o "xgboost", "catboost", "lightgbm", "dt", "ridge", "knn", "et"
    {
        "n_estimators": (50.0, 500.0),          # tuple → uniform
        "max_depth": [3.0, 6.0, 10.0, 15.0],   # list → choice
        "learning_rate": {"type": "log_uniform", "low": 0.01, "high": 0.5},  # dict → log_uniform
    },
    X, y, metric="rmse", n_folds=5,
)
# result["best_params"]  → {"n_estimators": 287.3, "max_depth": 10.0, ...}
# result["best_score"]   → 0.95
# result["all_results"]  → [(params_dict, score), ...]
```

### RFE (Recursive Feature Elimination)

```python
from smelt import rfe
selected = rfe(X, y, learner_type="decision_tree", n_features=15, feature_names=names)
# → [("slope", 0), ("elevation", 2), ...]
```

---

## Pipeline completo para el paper J.Hydrology

Todo lo necesario para ejecutar el pipeline del reporte está listo:

```python
import smelt_ml as sml
from smelt import cumulative_ranking, BayesianOptimizer, rfe
import numpy as np

# 1. Feature selection con 10 filtros
ranking = cumulative_ranking(X, y, names, top_k=15, corr_cutoff=0.9)
selected = ranking["feature"].tolist()

# 2. Benchmark con Spatial CV
spatial_cv = sml.SpatialBlockCV(n_folds=5, coords=coords)
# ... benchmark con los 12+ learners ...

# 3. Tuning bayesiano
bo = BayesianOptimizer(n_iter=100)
best = bo.optimize("rf", param_space, X_sel, y, metric="rmse", n_folds=5)

# 4. SHAP
model = sml.RandomForest(**{k: int(v) if k != "learning_rate" else v for k, v in best["best_params"].items()})
model.fit(X_sel, y)
shap = model.shap_values(X_sel, y, n_background=50, feature_names=selected)

# 5. Permutation importance
perm = model.permutation_importance(X_sel, y, metric="rmse", feature_names=selected)
```

---

## Para instalar

```bash
cd smelt/smelt-py
git pull
source .venv/bin/activate
maturin develop --release
```

O cuando se publique en PyPI: `pip install smelt-ml==0.4.0`

---

## Notas técnicas

- Los filtros greedy (MRMR, JMI, JMIM, CMIM) usan selección secuencial sobre **todas** las features, retornando scores que reflejan el orden de selección (más alto = seleccionado primero).
- Relief usa k=10 vecinos con kernel gaussiano adaptivo (sigma = distancia al k-ésimo vecino).
- MI condicional usa la identidad I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z) con bins 3D (10³ = 1000 celdas).
- BayesianOptimizer usa el TPE de Rust (no Python) — más rápido que Optuna para learners Rust.
- SHAP es model-agnostic (interventional, no TreeSHAP exacto) — funciona con cualquier learner.
