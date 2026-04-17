"""Benchmark utilities for comparing multiple learners with cross-validation."""

import numpy as np
from smelt._smelt import accuracy_score, CrossValidation, SpatialBlockCV


def benchmark(learners, X, y, cv=None, coords=None, metrics=None):
    """Compare multiple learners with cross-validation.

    Parameters
    ----------
    learners : dict
        Mapping of name → learner instance (e.g., {"XGBoost": XGBoost()}).
    X : numpy.ndarray
        Feature matrix (n_samples, n_features).
    y : numpy.ndarray
        Target vector.
    cv : int or CV object, optional (default: 5)
        Number of folds, or a CV object (CrossValidation, SpatialBlockCV, ...).
    coords : array-like, optional
        If provided with `cv` as int, uses SpatialBlockCV.
        Accepts numpy array (Nx2), list of (x, y) tuples, or list of [x, y] lists.
    metrics : dict, optional
        Mapping of name → metric function. Default: {"accuracy": accuracy_score}.

    Returns
    -------
    dict
        Results per learner: ``{name: {metric: {"mean", "std", "folds"}}}``.

    Example
    -------
    >>> from smelt import XGBoost, RandomForest, SpatialBlockCV
    >>> from smelt.benchmark import benchmark
    >>> results = benchmark(
    ...     {"XGB": XGBoost(), "RF": RandomForest()},
    ...     X, y, cv=SpatialBlockCV(5, coords),
    ... )
    >>> results["XGB"]["accuracy"]["mean"]
    0.87
    """
    if metrics is None:
        metrics = {"accuracy": accuracy_score}

    if cv is None:
        cv = 5

    if isinstance(cv, int):
        if coords is not None:
            splitter = SpatialBlockCV(cv, coords)
        else:
            splitter = CrossValidation(cv)
    else:
        splitter = cv

    n = len(y)
    splits = splitter.splits(n)
    is_classif = np.issubdtype(y.dtype, np.integer) or len(np.unique(y)) < 20

    results = {}
    for name, learner_template in learners.items():
        fold_scores = {m: [] for m in metrics}

        for train_idx, test_idx in splits:
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            X_tr = X[train_idx]
            y_tr = y[train_idx]
            X_te = X[test_idx]
            y_te = y[test_idx]

            # Clone learner (create fresh instance with same params)
            learner = learner_template.__class__(**_get_params(learner_template))
            try:
                learner.fit(X_tr, y_tr)
                preds = learner.predict(X_te)
            except RuntimeError as exc:
                # Incompatible learner (e.g. GaussianNB on regression target)
                for m in metrics:
                    fold_scores[m].append(float("nan"))
                fold_scores.setdefault("_error", str(exc))
                continue

            for metric_name, metric_fn in metrics.items():
                if is_classif:
                    score = metric_fn(y_te.tolist(), preds.tolist())
                else:
                    score = metric_fn(y_te.tolist(), preds.tolist())
                fold_scores[metric_name].append(score)

        # Aggregate per-metric stats
        results[name] = {}
        for metric_name in metrics:
            vals = fold_scores[metric_name]
            finite = [v for v in vals if np.isfinite(v)]
            results[name][metric_name] = {
                "mean": float(np.mean(finite)) if finite else float("nan"),
                "std": float(np.std(finite)) if finite else float("nan"),
                "folds": vals,
            }
        if "_error" in fold_scores:
            results[name]["_error"] = fold_scores["_error"]

    return results


def benchmark_table(results):
    """Format benchmark results as a printable table.

    Parameters
    ----------
    results : dict
        Output from benchmark().

    Returns
    -------
    str
        Formatted table string.
    """
    lines = []
    first = next(iter(results.values()))
    metrics = [k for k in first.keys() if not k.startswith("_")]
    header = f"{'Learner':<20}" + "".join(f"  {m:>12}" for m in metrics)
    lines.append(header)
    lines.append("-" * len(header))

    for name, scores in results.items():
        row = f"{name:<20}"
        for m in metrics:
            stats = scores[m]
            # Support both new dict format and legacy list format
            if isinstance(stats, dict):
                mean, std = stats["mean"], stats["std"]
            else:
                mean, std = float(np.mean(stats)), float(np.std(stats))
            row += f"  {mean:>5.3f}±{std:.3f}"
        lines.append(row)

    return "\n".join(lines)


def _get_params(learner):
    """Extract constructor parameters from a learner instance."""
    params = {}
    for attr in dir(learner):
        if not attr.startswith("_") and attr not in ("fit", "predict", "predict_proba",
                                                       "feature_importances_"):
            try:
                val = getattr(learner, attr)
                if isinstance(val, (int, float, bool, str)):
                    params[attr] = val
            except Exception:
                pass
    return params
