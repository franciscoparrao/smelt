# Queries Web of Science — Estado del Arte ML para smelt-ml

## Objetivo

Identificar algoritmos y técnicas modernas de ML para datos tabulares que puedan incorporarse a smelt-ml, y analizar la arquitectura de otros frameworks para extraer ideas de diseño.

**Filtros sugeridos para todas las queries:**
- Años: 2022-2026
- Tipo: Article, Review, Proceedings Paper
- Ordenar por: Times Cited (descending) o Relevance

---

## Bloque 1: Algoritmos Modernos para Datos Tabulares

### Q1.1 — Gradient Boosting de nueva generación
```
TS=("gradient boosting" AND (histogram* OR "second order" OR "Newton boosting" OR "leaf-wise"))
AND TS=(tabular OR "structured data" OR classification OR regression)
NOT TS=("deep learning" OR "neural network" OR "image" OR "NLP")
```
**Busca:** Mejoras algorítmicas a XGBoost/LightGBM/CatBoost — ordered boosting, symmetric trees, GPU-efficient splits, approximate quantile splits.

### Q1.2 — Tabular Foundation Models
```
TS=("tabular foundation model" OR "TabPFN" OR "TabICL" OR "in-context learning" AND tabular)
OR TS=("prior-data fitted network" AND (classification OR regression))
```
**Busca:** Modelos pre-entrenados que predicen sin entrenar (zero-shot/few-shot para tabular). TabPFN, TabICL, CARTE.

### Q1.3 — Deep Learning vs árboles en datos tabulares
```
TS=("deep learning" AND "tabular data" AND (benchmark OR comparison OR "gradient boosting"))
AND TS=(classification OR regression)
```
**Busca:** Estudios comparativos que identifican cuándo DL supera a árboles y qué arquitecturas (FT-Transformer, SAINT, TabNet, NODE).

### Q1.4 — Extremely Randomized Trees y variantes
```
TS=("extremely randomized" OR "extra trees" OR "random rotation" OR "oblique" AND (forest OR tree*))
AND TS=(classification OR regression OR ensemble)
```
**Busca:** Variantes modernas de árboles — oblique splits, rotation forests, mondrian forests, conditional random forests.

### Q1.5 — Kernel Methods modernos y SVM escalables
```
TS=("support vector machine" OR "kernel method" OR SVM)
AND TS=(scalab* OR "random features" OR "Nystrom" OR "approximate kernel")
AND TS=(tabular OR classification OR regression)
```
**Busca:** Técnicas para hacer SVM escalable — random Fourier features, Nyström approximation, FALKON.

---

## Bloque 2: Ensemble y Meta-Learning

### Q2.1 — Stacking y Super Learner
```
TS=("stacked generalization" OR "super learner" OR "model stacking" OR "blending")
AND TS=(ensemble OR "meta-learning" OR "cross-validation")
AND TS=(tabular OR classification OR regression)
```
**Busca:** Métodos de stacking nivel 2 — cómo combinar predicciones de múltiples modelos. Super Learner (van der Laan), stacking con CV.

### Q2.2 — AutoML y pipeline optimization
```
TS=(AutoML OR "automated machine learning" OR "neural architecture search" OR CASH)
AND TS=(tabular OR pipeline OR "hyperparameter" OR "model selection")
NOT TS=(image OR "computer vision" OR NLP)
```
**Busca:** Auto-sklearn, AutoGluon, FLAML, TPOT — cómo seleccionan y combinan algoritmos automáticamente.

### Q2.3 — Bayesian Optimization para hyperparameters
```
TS=("Bayesian optimization" OR "Gaussian process" OR "surrogate model")
AND TS=("hyperparameter" OR tuning OR "model selection")
AND TS=("machine learning" OR classification OR regression)
```
**Busca:** TPE (Tree-structured Parzen Estimators), BOHB, SMAC — métodos de tuning más eficientes que grid/random.

### Q2.4 — Ensemble selection y pruning
```
TS=("ensemble selection" OR "ensemble pruning" OR "model combination")
AND TS=(diversity OR "complementary" OR "Pareto" OR "greedy forward")
```
**Busca:** Cómo elegir el subconjunto óptimo de modelos para un ensemble. Caruana ensemble selection, Pareto-optimal ensembles.

---

## Bloque 3: Interpretabilidad y Explicabilidad

### Q3.1 — SHAP y métodos post-hoc
```
TS=(SHAP OR "Shapley" OR "feature importance" OR "feature attribution")
AND TS=("machine learning" OR "gradient boosting" OR "random forest")
AND TS=(tabular OR "structured data")
```
**Busca:** Implementaciones eficientes de SHAP (TreeSHAP, KernelSHAP), LIME, Anchors, contrafactuales.

### Q3.2 — Modelos inherentemente interpretables
```
TS=("explainable boosting machine" OR "EBM" OR "generalized additive model" OR "GAM" OR "rule list" OR "scoring system")
AND TS=(interpret* OR explain* OR transparen*)
AND TS=(classification OR regression)
```
**Busca:** EBM (InterpretML), GAM2, RuleFit, FIGS — modelos que son interpretables por diseño sin sacrificar mucho accuracy.

### Q3.3 — Fairness y bias en ML
```
TS=(fairness OR bias OR "algorithmic fairness" OR "protected attribute")
AND TS=("machine learning" OR classification OR "decision making")
AND TS=(tabular OR "structured data" OR "demographic")
```
**Busca:** Métricas de fairness, debiasing constraints, fairness-aware learning.

---

## Bloque 4: Técnicas Avanzadas de Training

### Q4.1 — Semi-supervised y self-supervised para tabular
```
TS=("semi-supervised" OR "self-supervised" OR "contrastive learning" OR "pseudo-label")
AND TS=(tabular OR "structured data")
AND TS=(classification OR regression)
```
**Busca:** VIME, SubTab, SCARF — técnicas que aprovechan datos sin etiquetar para mejorar modelos tabulares.

### Q4.2 — Data augmentation para tabular
```
TS=("data augmentation" OR "synthetic data" OR SMOTE OR "oversampling")
AND TS=(tabular OR "structured data" OR "class imbalance")
NOT TS=(image OR "computer vision" OR NLP)
```
**Busca:** SMOTE variantes, mixup para tabular, CTGAN, generación de datos sintéticos.

### Q4.3 — Transfer learning para tabular
```
TS=("transfer learning" OR "domain adaptation" OR "pre-training")
AND TS=(tabular OR "structured data")
NOT TS=(image OR NLP OR "language model" OR "computer vision")
```
**Busca:** Cómo transferir conocimiento entre datasets tabulares diferentes.

### Q4.4 — Online y incremental learning
```
TS=("online learning" OR "incremental learning" OR "streaming" OR "concept drift")
AND TS=("decision tree" OR "gradient boosting" OR ensemble OR "random forest")
AND TS=(tabular OR classification OR regression)
```
**Busca:** Hoeffding trees, Mondrian forests, ARF — árboles que aprenden de streams de datos.

---

## Bloque 5: Geoespacial + ML (dominio del usuario)

### Q5.1 — Spatial ML y geographic ML
```
TS=("spatial machine learning" OR "geographically weighted" OR "spatial cross-validation" OR "spatial autocorrelation")
AND TS=(classification OR regression OR prediction)
AND TS=("random forest" OR "gradient boosting" OR "ensemble" OR "deep learning")
```
**Busca:** GWR, spatial random forests, GeoAI — técnicas que consideran la estructura espacial de los datos.

### Q5.2 — Remote sensing + ML moderno
```
TS=("remote sensing" OR "satellite imagery" OR "land cover" OR "NDVI")
AND TS=("machine learning" OR "random forest" OR "gradient boosting" OR "XGBoost")
AND TS=(classification OR "change detection" OR mapping)
```
**Busca:** Pipelines modernos para clasificación de uso de suelo, detección de cambios, predicción de variables biofísicas.

### Q5.3 — Geostatistics + ML híbrido
```
TS=(geostatistic* OR kriging OR "spatial prediction" OR "spatial interpolation")
AND TS=("machine learning" OR "random forest" OR "gradient boosting" OR ensemble)
AND TS=(hybrid OR combined OR integrated)
```
**Busca:** Regression-kriging con ML, residual kriging + XGBoost, modelos híbridos espaciales.

---

## Bloque 6: Frameworks y Arquitectura de Software

### Q6.1 — Diseño de frameworks ML
```
TS=("machine learning framework" OR "machine learning library" OR "ML toolkit")
AND TS=(design OR architecture OR API OR "software engineering")
AND TS=(Python OR Rust OR "C++" OR Julia)
```
**Busca:** Papers sobre el diseño de scikit-learn, mlr3, MLJ (Julia), linfa — decisiones arquitectónicas, patterns, extensibilidad.

### Q6.2 — Pipelines de ML reproducibles
```
TS=("ML pipeline" OR "machine learning pipeline" OR "MLOps" OR "reproducib*")
AND TS=(workflow OR orchestrat* OR automat*)
AND TS=("best practice" OR framework OR standard)
```
**Busca:** Cómo diseñar pipelines reproducibles — MLflow, DVC, feature stores, experiment tracking.

### Q6.3 — Benchmarking de algoritmos ML
```
TS=(benchmark* AND "machine learning" AND (tabular OR "structured data"))
AND TS=(comparison OR evaluation OR "empirical study")
AND TS=(classification OR regression)
```
**Busca:** OpenML, AMLB, TabZilla — benchmarks estandarizados que comparan algoritmos en muchos datasets.

### Q6.4 — ML en lenguajes de sistemas (Rust, C++, Julia)
```
TS=("machine learning" AND (Rust OR "systems programming" OR "high performance"))
AND TS=(framework OR library OR implementation)
AND TS=(performance OR "memory safe*" OR "type safe*" OR parallel*)
```
**Busca:** Ventajas/desventajas de implementar ML en lenguajes de bajo nivel. burn, candle, linfa, SmartCore.

---

## Bloque 7: Técnicas Específicas para Implementar

### Q7.1 — Conformal prediction
```
TS=("conformal prediction" OR "prediction interval" OR "coverage guarantee" OR "uncertainty quantification")
AND TS=("machine learning" OR classification OR regression)
AND TS=(tabular OR "structured data" OR "distribution-free")
```
**Busca:** Intervalos de predicción con garantías teóricas. Split conformal, CQR — técnica moderna muy relevante que pocos frameworks implementan.

### Q7.2 — Quantile regression
```
TS=("quantile regression" OR "conditional quantile" OR "distributional regression")
AND TS=("gradient boosting" OR "random forest" OR "machine learning")
AND TS=(prediction OR forecast* OR interval)
```
**Busca:** Predecir distribuciones en vez de puntos. Quantile forests, quantile GBM — aplicación directa en geoespacial.

### Q7.3 — Multi-target y multi-output learning
```
TS=("multi-target" OR "multi-output" OR "multi-label" OR "multi-task")
AND TS=(regression OR classification)
AND TS=("decision tree" OR "random forest" OR "gradient boosting" OR ensemble)
```
**Busca:** Predecir múltiples targets simultáneamente — multi-target random forests, chained regression.

### Q7.4 — Survival analysis con ML
```
TS=("survival analysis" OR "time-to-event" OR "Cox" OR "hazard")
AND TS=("machine learning" OR "random forest" OR "gradient boosting")
AND TS=(censored OR "right-censored" OR "Kaplan-Meier")
```
**Busca:** Random Survival Forests, DeepSurv, XGBoost para survival — nicho importante en salud y geociencias.

### Q7.5 — Causal inference con ML
```
TS=("causal inference" OR "treatment effect" OR "causal forest" OR "double machine learning")
AND TS=("machine learning" OR "random forest" OR "gradient boosting")
AND TS=(observational OR "heterogeneous" OR "CATE")
```
**Busca:** Causal forests (Athey & Imbens), DML, T/S/X-learners — frontera entre ML y inferencia causal.

---

## Notas de Uso

1. **Ejecutar cada query por separado** en WOS para controlar volumen
2. **Exportar como BibTeX o CSV** con abstract
3. **Priorizar**: papers con >50 citas para reviews, papers 2024-2026 para lo más nuevo
4. Las queries del **Bloque 1 y 7** son las más relevantes para algoritmos a implementar
5. Las queries del **Bloque 5** son específicas para tu dominio GIS
6. Las queries del **Bloque 6** ayudan a mejorar la arquitectura del framework

## Algoritmos con Mayor Potencial para smelt-ml

Basado en la investigación previa, estos son los que más impacto tendrían:

| Algoritmo | Novedad | Complejidad | Impacto |
|-----------|:---:|:---:|:---:|
| Conformal Prediction | Alta | Media | Intervalos con garantías |
| Stacking / Super Learner | Media | Baja | Mejor accuracy combinando modelos |
| EBM (Explainable Boosting) | Alta | Media | Interpretable + competitivo |
| Quantile Regression Forest | Media | Baja | Distribuciones, no solo puntos |
| Causal Forest | Alta | Alta | Efectos heterogéneos |
| Online/Streaming Trees | Alta | Alta | Datos en tiempo real |
| Mondrian Forest | Alta | Media | Online + conformal nativo |
| SMOTE / data augmentation | Baja | Baja | Class imbalance |
| Bayesian Optimization | Media | Media | Mejor tuning |
| Multi-target learning | Media | Media | Múltiples outputs |
