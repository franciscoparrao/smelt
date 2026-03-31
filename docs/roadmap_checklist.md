# smelt-ml Roadmap Checklist

*Basado en análisis de 23,045 papers de Web of Science (2006-2026)*
*Última actualización: 2026-03-30*

---

## Estado actual del framework

- **21 learners supervisados** + K-Means + DBSCAN + CausalForest
- **10 métricas** + Silhouette
- **9 transformers** (Scalers, Imputer, OneHot, Label, SMOTE, PCA, RFE, FilterSelector)
- **4 tuners** (Grid, Random, TPE, Hyperband)
- **4 resamplers** (CV, Holdout, SpatialBlock, SpatialBuffer)
- Conformal Prediction, Benchmark Design, Permutation Importance
- CSV loading, JSON serialization, input validation
- **202+ integration tests**, publicado en crates.io v0.6.1

---

## Prioridad 1: CRÍTICO (evidencia masiva en la literatura)

- [x] **LightGBM** — GOSS + leaf-wise growth + histogram splits. 32,467 citas. ✅ v0.8.0
- [x] **CatBoost** — Ordered target statistics + oblivious trees + native categoricals. ✅ v0.8.1
- [x] **Random Survival Forest** — Log-rank split + Nelson-Aalen + C-index. 1,000+ papers. ✅ v0.9.0
- [x] **Isolation Forest** — Anomaly detection via random path length. s(x,n) = 2^{-E[h(x)]/c(n)}. 700+ menciones. ✅ v0.6.1
- [ ] **TreeSHAP** — Explicabilidad basada en Shapley values para modelos de árboles. 9,000+ referencias. ~300 líneas.

## Prioridad 2: ALTO (dominios enteros ausentes)

- [x] **Multi-label Classification (Classifier Chains)** — Train C_j on X + previous labels. 2,000+ papers. ✅ v0.6.1
- [x] **Multi-output / Multi-target Regression** — Regressor Chains. 1,000+ papers. ✅ v0.7.0
- [x] **Quantile Regression Forest** — QRF con weighted quantiles de las hojas. 602 papers. ✅ v0.7.0
- [ ] **Hoeffding Trees (Streaming)** — VFDT + HAT adaptivo. Split con Hoeffding bound. Ni sklearn ni mlr3 lo tienen. ~300 líneas.
- [ ] **Dynamic Ensemble Selection (META-DES)** — Selección dinámica por instancia. Superior a stacking estático. 339 papers. ~250 líneas.

## Prioridad 3: MEDIO (mejoras y extensiones)

- [x] **ADASYN** — Oversampling adaptivo por densidad local. 4,070 citas. ✅ v0.7.0
- [ ] **Extreme Learning Machine (ELM)** — Random weights + pseudoinverse. Ultra-rápido. 4,620 citas. ~150 líneas.
- [ ] **Deep Forest / gcForest** — Cascada de forests emulando deep learning. ~300 líneas.
- [ ] **Cost-Sensitive Learning** — Soporte de cost matrix en learners. Esencial para medicina/finanzas. ~150 líneas.
- [x] **Conformalized Quantile Regression (CQR)** — CP + quantile regression. Intervalos adaptativos. ✅ v0.7.0

## Prioridad 4: DIFERENCIADORES (único en Rust)

- [ ] **Kriging-ML Hybrid** — Regression-kriging con XGBoost/RF para residuos espaciales. 15+ papers en Q5_3. ~300 líneas.
- [ ] **Spatial-SMOTE** — SMOTE geoespacialmente informado. ~150 líneas.
- [ ] **Uplift Modeling (X/R/DR-Learners)** — Meta-learners para efectos causales. Extiende CausalForest. ~250 líneas.
- [ ] **Online Adaptive Random Forest** — RF incremental con detección de concept drift (ADWIN). ~400 líneas.
- [ ] **Mondrian Forest** — Árboles con proceso de Mondrian para partición incremental consistente. ~300 líneas.

## Prioridad 5: NICE-TO-HAVE

- [ ] **WildWood** — Context tree weighting sobre todos los subárboles. Novel (2023). ~300 líneas.
- [ ] **Adversarial Random Forests** — Density estimation generativa con RF. ~250 líneas.
- [ ] **Fairness Metrics** — Disparate impact, equalized odds, demographic parity. ~150 líneas.
- [ ] **OpenML Integration** — Cargar datasets de OpenML por ID. ~100 líneas.
- [ ] **Parquet Loading** — Requiere dep pesada (arrow-rs). ~200 líneas.

## Infraestructura pendiente

- [ ] **Python bindings (PyO3)** — Crate separado `smelt-py`. Mayor multiplicador de usuarios.
- [ ] **Paper JOSS** — Journal of Open Source Software. Requiere tracción.
- [ ] **Cargo doc completo** — Doc comments en todos los módulos públicos.
- [ ] **Benchmarks OpenML** — Evaluar en 30+ datasets estándar de OpenML.
- [ ] **CI/CD** — GitHub Actions con tests automáticos.

---

## Ya completado ✅

### Learners supervisados
- [x] Decision Tree (CART)
- [x] K-Nearest Neighbors
- [x] Linear Regression (OLS)
- [x] Logistic Regression (auto-scaling)
- [x] Random Forest (parallel, rayon)
- [x] Gradient Boosting (MSE/log-loss/softmax)
- [x] Extra Trees (random thresholds)
- [x] XGBoost (Newton, histogram, NaN, exact greedy, early stopping, parallel)
- [x] Geographical-XGBoost (Grekousis 2025, bi-square kernel, adaptive alpha)
- [x] Oblique Tree (sparse projections)
- [x] Oblique Forest / SPORF (Tomita 2020, parallel)
- [x] Gaussian Naive Bayes
- [x] Ridge Regression (L2, closed form)
- [x] Lasso Regression (L1, coordinate descent)
- [x] Elastic Net (L1+L2)
- [x] AdaBoost (SAMME)
- [x] Linear SVM (SGD + hinge loss)
- [x] Stacking / Super Learner (OOF meta-ensemble)
- [x] Quantile GB (pinball loss)
- [x] EBM (Explainable Boosting Machine, cyclic GAM)
- [x] Bagging (generic wrapper)

### Unsupervised
- [x] K-Means (Lloyd's + silhouette)
- [x] DBSCAN (density-based, noise detection)

### Causal Inference
- [x] Causal Forest (honest splitting, CATE, ATE, CIs)

### Métricas
- [x] Accuracy, Precision, Recall, F1 Score (macro)
- [x] Log Loss, AUC-ROC
- [x] RMSE, MAE, R-squared, MAPE
- [x] Silhouette Score

### Preprocessing / Feature Engineering
- [x] StandardScaler, MinMaxScaler
- [x] Imputer (mean, median, constant)
- [x] OneHotEncoder, LabelEncoder
- [x] SMOTE (synthetic minority oversampling)
- [x] PCA (power iteration)
- [x] FilterSelector (Variance, Correlation, ANOVA-F, Information Gain, Mutual Info)
- [x] RFE (Recursive Feature Elimination)
- [x] Pipeline (chains transformers + learner, fit_supervised)

### Tuning
- [x] GridSearch
- [x] RandomSearch (Uniform, LogUniform, Choice)
- [x] BayesianOptimizer (TPE)
- [x] Hyperband (successive halving)

### Resampling
- [x] CrossValidation (K-fold)
- [x] Holdout
- [x] SpatialBlockCV
- [x] SpatialBufferCV

### Advanced
- [x] Conformal Prediction (regression + classification)
- [x] Permutation Feature Importance
- [x] Benchmark Design (multi-learner × multi-task)
- [x] CSV Data Loading (auto label encoding)
- [x] Model Serialization (JSON)
- [x] Input Validation (dimension check, NaN detection)

---

## Fórmulas clave para implementación

### LightGBM GOSS
```
1. Sort instances by |g_i|
2. Keep top a% (large gradients) -> A
3. Randomly sample b% from rest -> B
4. Amplify B's gradients by (1-a)/b
5. Train tree on A ∪ B
```

### CatBoost Ordered Target Statistics
```
x_{σ(p),k} = (Σ_{σ(j)<σ(p), x_{j,k}=v} y_{σ(j)} + a·prior) / (count + a)
```

### RSF Log-Rank Split
```
L = [Σ_j (d_{j,1} - Y_{j,1}·d_j/Y_j)]² / Var
```

### Isolation Forest Score
```
s(x,n) = 2^{-E[h(x)]/c(n)}, c(n) = 2·H(n-1) - 2·(n-1)/n
```

### Hoeffding Bound
```
Split when: G_a - G_b > √(R²·ln(1/δ)/(2n))
```

### Classifier Chains (multi-label)
```
For j=1..q: Train C_j on X ∪ {l_{π(1)},...,l_{π(j-1)}}
```

### Quantile Regression Forest
```
w_i = (1/T)·Σ_t I(x_i ∈ L_t(x)) / |L_t(x)|
F⁻¹(τ) = inf{y : Σ_{y_i≤y} w_i ≥ τ}
```

---

*Documento generado a partir de análisis de 23,045 papers WOS. smelt-ml v0.6.1.*
