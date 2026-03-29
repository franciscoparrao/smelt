"""Compare official XGBoost vs smelt-ml XGBoost on identical data."""
import json
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error

np.random.seed(42)

results = {}

# ── Test 1: Binary classification (linearly separable) ──────────────
print("=== Binary Classification (separable) ===")
X_bin = np.array([
    [0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [0.0, 0.2], [0.1, 0.0],
    [0.2, 0.1], [0.0, 0.1], [0.1, 0.2], [0.15, 0.05], [0.05, 0.15],
    [1.0, 1.0], [1.1, 0.9], [0.9, 1.1], [1.0, 0.9], [1.1, 1.0],
    [0.9, 1.0], [1.0, 1.1], [1.1, 1.1], [0.95, 0.95], [1.05, 1.05],
])
y_bin = np.array([0]*10 + [1]*10)

model = xgb.XGBClassifier(
    n_estimators=50, max_depth=3, learning_rate=0.3,
    reg_lambda=1.0, reg_alpha=0.0, gamma=0.0,
    subsample=1.0, colsample_bytree=1.0,
    min_child_weight=1.0, use_label_encoder=False,
    eval_metric='logloss', random_state=42,
)
model.fit(X_bin, y_bin)
pred_bin = model.predict(X_bin)
acc_bin = accuracy_score(y_bin, pred_bin)
print(f"  Train accuracy: {acc_bin:.4f}")
print(f"  Predictions: {pred_bin.tolist()}")
results["binary_accuracy"] = acc_bin
results["binary_predictions"] = pred_bin.tolist()

# ── Test 2: Regression (linear trend) ───────────────────────────────
print("\n=== Regression (linear) ===")
X_reg = np.array([[i] for i in range(1, 11)], dtype=float)
y_reg = np.array([2*i for i in range(1, 11)], dtype=float)

model_reg = xgb.XGBRegressor(
    n_estimators=100, max_depth=3, learning_rate=0.3,
    reg_lambda=1.0, reg_alpha=0.0, gamma=0.0,
    subsample=1.0, colsample_bytree=1.0,
    min_child_weight=1.0, random_state=42,
)
model_reg.fit(X_reg, y_reg)
pred_reg = model_reg.predict(X_reg)
rmse_reg = np.sqrt(mean_squared_error(y_reg, pred_reg))
print(f"  Train RMSE: {rmse_reg:.4f}")
print(f"  Predictions: {[round(p, 4) for p in pred_reg.tolist()]}")
print(f"  Truth:       {y_reg.tolist()}")
results["regress_rmse"] = rmse_reg
results["regress_predictions"] = [round(p, 4) for p in pred_reg.tolist()]

# ── Test 3: Multiclass (3 clusters) ─────────────────────────────────
print("\n=== Multiclass (3 classes) ===")
X_mc = np.array([
    [0.0, 0.0], [0.1, 0.1], [0.0, 0.1], [0.1, 0.0],
    [1.0, 0.0], [1.1, 0.1], [1.0, 0.1], [1.1, 0.0],
    [0.0, 1.0], [0.1, 1.1], [0.0, 1.1], [0.1, 1.0],
])
y_mc = np.array([0]*4 + [1]*4 + [2]*4)

model_mc = xgb.XGBClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.3,
    reg_lambda=0.01, reg_alpha=0.0, gamma=0.0,
    min_child_weight=0.1,
    use_label_encoder=False, eval_metric='mlogloss', random_state=42,
)
model_mc.fit(X_mc, y_mc)
pred_mc = model_mc.predict(X_mc)
acc_mc = accuracy_score(y_mc, pred_mc)
print(f"  Train accuracy: {acc_mc:.4f}")
print(f"  Predictions: {pred_mc.tolist()}")
results["multiclass_accuracy"] = acc_mc

# ── Test 4: Larger synthetic dataset with CV ────────────────────────
print("\n=== Synthetic dataset (200 samples, 5-fold CV) ===")
X_syn, y_syn = make_classification(
    n_samples=200, n_features=10, n_informative=5,
    n_redundant=2, n_classes=2, random_state=42,
)
model_syn = xgb.XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.3,
    reg_lambda=1.0, eval_metric='logloss', random_state=42,
)
cv_scores = cross_val_score(model_syn, X_syn, y_syn, cv=5, scoring='accuracy')
print(f"  5-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  Per-fold: {[round(s, 4) for s in cv_scores.tolist()]}")
results["cv_mean_accuracy"] = round(cv_scores.mean(), 4)
results["cv_std"] = round(cv_scores.std(), 4)

# ── Test 5: Feature importance ranking ──────────────────────────────
print("\n=== Feature Importance ===")
model_syn.fit(X_syn, y_syn)
imp = model_syn.feature_importances_
ranked = sorted(enumerate(imp), key=lambda x: -x[1])
print(f"  Top 5 features: {[(f'f{i}', round(v, 4)) for i, v in ranked[:5]]}")
results["top_features"] = [i for i, _ in ranked[:5]]

# Save for Rust comparison
with open("/tmp/xgboost_reference.json", "w") as f:
    json.dump(results, f, indent=2)

# Save synthetic dataset for Rust
np.savetxt("/tmp/xgb_synthetic_X.csv", X_syn, delimiter=",")
np.savetxt("/tmp/xgb_synthetic_y.csv", y_syn, delimiter=",")

print("\n✓ Reference results saved to /tmp/xgboost_reference.json")
print("✓ Synthetic dataset saved to /tmp/xgb_synthetic_{X,y}.csv")
