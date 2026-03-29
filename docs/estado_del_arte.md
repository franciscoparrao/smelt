# Estado del Arte ML — Revisión Rápida para smelt-ml

*Búsqueda realizada: marzo 2026*

---

## 1. Algoritmos de Mayor Impacto para Implementar

### 1.1 Conformal Prediction — **PRIORIDAD ALTA**

Framework distribution-free para cuantificar incertidumbre. Genera intervalos de predicción con garantías teóricas de cobertura. Model-agnostic: envuelve cualquier modelo existente.

**Qué es:** Dado un modelo ya entrenado, calcula un "nonconformity score" en un set de calibración, luego genera prediction sets/intervals con cobertura garantizada al nivel 1-α.

**Por qué importa:**
- Pocos frameworks lo implementan nativamente (solo MAPIE en Python, ninguno en Rust)
- Es un wrapper — funciona con todos nuestros learners sin modificarlos
- Garantías teóricas reales, no heurísticas
- Variantes: Split Conformal, CQR (Conformalized Quantile Regression)

**Complejidad de implementación:** **Baja** (~200 líneas). Es esencialmente: entrenar modelo → calibrar con holdout → computar quantile de residuos → prediction ± quantile.

**Referencia:** [Conformal Prediction: A Data Perspective | ACM Computing Surveys 2025](https://dl.acm.org/doi/10.1145/3736575)

---

### 1.2 Explainable Boosting Machine (EBM) — **PRIORIDAD ALTA**

GAM (Generalized Additive Model) entrenado con gradient boosting cíclico. Tan preciso como XGBoost pero completamente interpretable.

**Qué es:** Entrena un gradient boosting donde cada iteración ajusta UN solo feature a la vez, en round-robin. El resultado es f(x) = f₁(x₁) + f₂(x₂) + ... + fₖ(x₁,x₂) donde cada fᵢ es una función univariada visualizable. Opcionalmente detecta interacciones pairwise (GA²M).

**Por qué importa:**
- Accuracy competitiva con XGBoost (AUC 0.93 en Adult Income)
- Cada feature tiene una "shape function" que se puede graficar
- Microsoft lo posiciona como "go-to algorithm for tabular data" cuando se necesita interpretabilidad
- Implementación en InterpretML (Python), **ninguna en Rust**

**Complejidad:** **Media** (~400 líneas). Reutiliza nuestro TreeBuilder pero restringido a 1 feature por iteración.

**Referencia:** [Pushing the Boundaries of Interpretability: Incremental Enhancements to EBM (2024)](https://arxiv.org/html/2512.00528v1)

---

### 1.3 Stacking / Super Learner — **PRIORIDAD ALTA**

Meta-ensemble que combina predicciones de múltiples modelos heterogéneos usando un meta-learner.

**Qué es:** Nivel 0: entrena K modelos distintos (DT, RF, XGB, KNN, etc.) con CV. Nivel 1: las predicciones out-of-fold de nivel 0 son features para un meta-learner (típicamente Logistic Regression o Ridge). El Super Learner (van der Laan) es la versión con garantías teóricas.

**Por qué importa:**
- Consistentemente gana competiciones de ML (Kaggle)
- XStacking (2025) integra SHAP values como meta-features
- Complementa perfectamente nuestros 15 learners existentes
- Implementa `Learner` trait → funciona con benchmark/tuning

**Complejidad:** **Baja-Media** (~250 líneas). Ya tenemos todos los componentes (CV, múltiples learners, Prediction con probabilities).

**Referencia:** [XStacking: Effective and Inherently Explainable Stacked Ensemble Learning (2025)](https://www.sciencedirect.com/science/article/pii/S1566253525004312)

---

### 1.4 Quantile Regression — **PRIORIDAD MEDIA-ALTA**

Predecir quantiles de la distribución condicional, no solo la media.

**Qué es:** En vez de minimizar MSE (que da la media), minimizar la pinball loss (que da el quantile τ). Aplicable a gradient boosting y random forest. Permite construir intervalos de predicción.

**Por qué importa:**
- Predecir distribuciones es más útil que predecir puntos en geoespacial
- Se implementa cambiando la loss function — gradient/hessian diferentes
- Combinable con conformal prediction (CQR)
- scikit-learn lo tiene en HistGradientBoosting, pocos frameworks lo implementan bien

**Complejidad:** **Baja** (~150 líneas). Solo necesita una nueva loss function para nuestro GradientBoosting/XGBoost.

**Referencia:** [Prediction Intervals for Gradient Boosting Regression — scikit-learn](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html)

---

### 1.5 Oblique Decision Trees — **PRIORIDAD MEDIA**

Árboles que hacen splits usando combinaciones lineales de features, no solo un feature a la vez.

**Qué es:** En vez de split `x₃ < 0.5`, usa `0.3x₁ + 0.7x₃ < 0.5`. Captura relaciones lineales que los árboles CART necesitan muchos splits para aproximar.

**Por qué importa:**
- Probado matemáticamente consistente en 2025 (Bernoulli journal)
- Oblique Random Forests superan a RF estándar en muchos datasets
- Pocas implementaciones disponibles fuera de R (ODRF package)
- Extensión natural de nuestro TreeBuilder

**Complejidad:** **Media** (~300 líneas). Necesita optimización del hiperplano en cada split (PCA, ridge, o random projections).

**Referencia:** [Statistical Advantages of Oblique Randomized Decision Trees (2024)](https://arxiv.org/abs/2407.02458)

---

## 2. Geoespacial + ML (dominio del usuario)

### 2.1 Geographical-XGBoost (G-XGBoost) — **PRIORIDAD ALTA para GIS**

Extensión de XGBoost que crea modelos locales ponderados espacialmente.

**Qué es:** Entrena un XGBoost global + modelos locales con pesos espaciales. Combina ambos para predicción. Captura heterogeneidad espacial que un modelo global no puede.

**Publicado:** Abril 2025 en Journal of Geographical Systems.

**Complejidad:** **Media**. Extiende nuestro XGBoost con pesos por muestra y modelo local/global.

**Referencia:** [Geographical-XGBoost: Ensemble Model for Spatially Local Regression (2025)](https://link.springer.com/article/10.1007/s10109-025-00465-4)

### 2.2 Geographically Weighted Random Forest (GWRF)

Random Forest local que entrena un modelo diferente por ubicación geográfica.

**Publicado:** 2024-2025, múltiples papers. PyGRF disponible en Python.

**Referencia:** [PyGRF: Improved Python GRF (2024)](https://onlinelibrary.wiley.com/doi/10.1111/tgis.13248)

---

## 3. Técnicas Avanzadas

### 3.1 Causal Forest

Random Forest para estimar efectos heterogéneos de tratamiento (CATE). Usa "honest" splitting: un subset para splits, otro para estimación.

**Estado:** 133 papers aplicados revisados en 2025. Implementación madura en R (grf), **ninguna en Rust**.

**Referencia:** [How Do Applied Researchers Use the Causal Forest? (2025)](https://onlinelibrary.wiley.com/doi/full/10.1111/insr.12610)

### 3.2 Online/Streaming Trees (Hoeffding Trees)

Árboles que aprenden de streams infinitos de datos sin reentrenar. Usan Hoeffding bound para decidir cuándo hacer un split.

**Estado:** Avances en 2025 con mixture-of-experts de Hoeffding trees, multi-label variants.

**Referencia:** [Extremely Simple Streaming Forest (2025)](https://arxiv.org/html/2110.08483)

### 3.3 SMOTE y Data Augmentation Tabular

Oversampling sintético para clases desbalanceadas. Variantes modernas: Borderline-SMOTE, ADASYN, SMOTE-TOMEK.

**Estado:** SMOTE-TOMEK mostró 50% mejora absoluta en recall vs baseline en estudios 2025.

**Referencia:** [Strategic Application of SMOTE Variants (2025)](https://iacis.org/iis/2025/2_iis_2025_70-85.pdf)

---

## 4. Deep Learning para Tabular (contexto, no para implementar)

Los benchmarks 2024-2025 confirman que **gradient boosting sigue ganando** en la mayoría de datasets tabulares. Sin embargo:

- **TabPFN/TabICL** (~zero-shot) empatan o superan a XGBoost en datasets <10K samples
- **FT-Transformer** gana en 7/11 datasets vs XGBoost en un benchmark
- **NODE** (Neural Oblivious Decision Ensembles) combina árboles obliviosos diferenciables
- Ninguno domina universalmente → confirma que nuestro enfoque multi-algoritmo es correcto

---

## 5. Recomendación de Implementación para smelt-ml

### Fase inmediata (máximo impacto, mínimo esfuerzo)

| # | Algoritmo | Líneas est. | Diferenciador |
|---|-----------|:-----------:|---------------|
| 1 | **Conformal Prediction** | ~200 | Intervalos con garantías. Ningún framework Rust lo tiene |
| 2 | **Stacking / Super Learner** | ~250 | Combina todos nuestros learners. Ganador de competiciones |
| 3 | **Quantile Regression** | ~150 | Nueva loss para XGBoost/GB. Distribuciones, no puntos |
| 4 | **EBM** | ~400 | Interpretable + competitivo. Nicho sin competencia en Rust |
| 5 | **SMOTE** | ~150 | Class imbalance. Ubicuo en la práctica |

### Fase siguiente (mayor complejidad)

| # | Algoritmo | Complejidad | Diferenciador |
|---|-----------|:-----------:|---------------|
| 6 | **G-XGBoost** | Media | Killer feature para GIS |
| 7 | **Oblique Trees** | Media | Árboles más expresivos |
| 8 | **Causal Forest** | Alta | Inferencia causal. Sin competencia en Rust |
| 9 | **Hoeffding Trees** | Alta | Streaming/online learning |
| 10 | **Bayesian Optimization** | Media | Mejor tuning que Grid/Random |

### Lo que NO implementaría

- TabPFN/TabICL → requieren neural networks, fuera del scope de smelt-ml
- NODE/FT-Transformer → mismo motivo
- CTGAN → generación sintética compleja, mejor como crate separado

---

## Sources

- [Conformal Prediction: A Data Perspective | ACM 2025](https://dl.acm.org/doi/10.1145/3736575)
- [EBM Enhancements 2024](https://arxiv.org/html/2512.00528v1)
- [XStacking 2025](https://www.sciencedirect.com/science/article/pii/S1566253525004312)
- [TabICLv2 2026](https://arxiv.org/html/2602.11139v1)
- [Geographical-XGBoost 2025](https://link.springer.com/article/10.1007/s10109-025-00465-4)
- [Oblique Decision Trees Consistency 2025](https://arxiv.org/abs/2211.12653)
- [Causal Forest Review 2025](https://onlinelibrary.wiley.com/doi/full/10.1111/insr.12610)
- [Streaming Forest 2025](https://arxiv.org/html/2110.08483)
- [SMOTE Variants 2025](https://iacis.org/iis/2025/2_iis_2025_70-85.pdf)
- [PyGRF 2024](https://onlinelibrary.wiley.com/doi/10.1111/tgis.13248)
