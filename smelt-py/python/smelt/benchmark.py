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
    cv : int or object, optional
        Number of folds (default 5) or a CV object (CrossValidation, SpatialBlockCV).
    coords : list of tuples, optional
        If provided with int cv, uses SpatialBlockCV instead of random CV.
    metrics : dict, optional
        Mapping of name → metric function. Default: {"accuracy": accuracy_score}.

    Returns
    -------
    dict
        Results per learner: {"name": {"metric": [fold_scores], ...}}.

    Example
    -------
    >>> from smelt import XGBoost, RandomForest, LightGBM
    >>> from smelt.benchmark import benchmark
    >>> results = benchmark(
    ...     {"XGB": XGBoost(), "RF": RandomForest(), "LGB": LightGBM()},
    ...     X, y, cv=5
    ... )
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
        results[name] = {m: [] for m in metrics}

        for train_idx, test_idx in splits:
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            X_tr = X[train_idx]
            y_tr = y[train_idx]
            X_te = X[test_idx]
            y_te = y[test_idx]

            # Clone learner (create fresh instance with same params)
            learner = learner_template.__class__(**_get_params(learner_template))
            learner.fit(X_tr, y_tr)
            preds = learner.predict(X_te)

            for metric_name, metric_fn in metrics.items():
                if is_classif:
                    score = metric_fn(y_te.tolist(), preds.tolist())
                else:
                    score = metric_fn(y_te.tolist(), preds.tolist())
                results[name][metric_name].append(score)

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
    metrics = list(next(iter(results.values())).keys())
    header = f"{'Learner':<20}" + "".join(f"  {m:>12}" for m in metrics)
    lines.append(header)
    lines.append("-" * len(header))

    for name, scores in results.items():
        row = f"{name:<20}"
        for m in metrics:
            vals = scores[m]
            mean = np.mean(vals)
            std = np.std(vals)
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
