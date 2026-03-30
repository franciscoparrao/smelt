"""Benchmark scikit-learn on standard datasets for comparison with smelt-ml."""
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
import xgboost as xgb
import os, json

results = {}

datasets = {
    "iris": load_iris(),
    "wine": load_wine(),
    "breast_cancer": load_breast_cancer(),
}

learners = {
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "KNN(5)": KNeighborsClassifier(n_neighbors=5),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "GaussianNB": GaussianNB(),
    "XGBoost": xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss'),
}

for ds_name, ds in datasets.items():
    X, y = ds.data, ds.target
    print(f"\n=== {ds_name} ({X.shape[0]} samples, {X.shape[1]} features, {len(set(y))} classes) ===")
    print(f"{'Learner':<25} {'5-fold CV Accuracy':>20}")
    print("-" * 48)

    # Save dataset for Rust
    np.savetxt(f"/tmp/bench_{ds_name}_X.csv", X, delimiter=",")
    np.savetxt(f"/tmp/bench_{ds_name}_y.csv", y, delimiter=",")

    results[ds_name] = {"n_samples": X.shape[0], "n_features": X.shape[1], "n_classes": len(set(y))}

    for name, clf in learners.items():
        scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
        mean = scores.mean()
        std = scores.std()
        print(f"{name:<25} {mean:>10.4f} ± {std:.4f}")
        results[ds_name][name] = {"mean": round(mean, 4), "std": round(std, 4)}

with open("/tmp/sklearn_benchmark.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✓ Datasets and results saved to /tmp/")
