"""Benchmark utilities for comparing multiple learners with cross-validation."""

import numpy as np
from smelt._smelt import accuracy_score, rmse_score, CrossValidation, SpatialBlockCV


_NEEDS_COORDS = {"GeoXGBoost", "KrigingHybrid"}


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
        If provided with `cv` as int, uses SpatialBlockCV. Also passed to
        the per-fold `fit()`/`predict()` calls of spatial learners
        (``GeoXGBoost``, ``KrigingHybrid``), which require it.
        Accepts numpy array (Nx2), list of (x, y) tuples, or list of [x, y] lists.
    metrics : dict, optional
        Mapping of name → metric function. Default: {"accuracy":
        accuracy_score} for classification targets (integer/bool dtype),
        {"rmse": rmse_score} for regression targets.

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
    y = np.asarray(y)
    # Same dispatch rule as fit(): integer/bool dtype => classification.
    # The old heuristic (`or len(unique) < 20`) could declare a float
    # target "classification" that fit() then trained as regression,
    # crashing mid-loop with a TypeError the per-fold handler didn't catch.
    is_classif = y.dtype.kind in "iub"

    if metrics is None:
        metrics = (
            {"accuracy": accuracy_score} if is_classif else {"rmse": rmse_score}
        )

    if cv is None:
        cv = 5

    if isinstance(cv, int):
        if coords is not None:
            splitter = SpatialBlockCV(cv, coords)
        else:
            splitter = CrossValidation(cv)
    else:
        splitter = cv

    coords_arr = np.asarray(coords) if coords is not None else None

    n = len(y)
    splits = splitter.splits(n)

    results = {}
    for name, learner_template in learners.items():
        # Skip learners incompatible with this task type
        if is_classif:
            supports = getattr(learner_template, "supports_classification", True)
        else:
            supports = getattr(learner_template, "supports_regression", True)
        if not supports:
            task_name = "classification" if is_classif else "regression"
            results[name] = {
                m: {"mean": float("nan"), "std": float("nan"), "folds": []}
                for m in metrics
            }
            results[name]["_skipped"] = f"{name} does not support {task_name}"
            continue

        needs_coords = type(learner_template).__name__ in _NEEDS_COORDS
        if needs_coords and coords_arr is None:
            results[name] = {
                m: {"mean": float("nan"), "std": float("nan"), "folds": []}
                for m in metrics
            }
            results[name]["_skipped"] = f"{name} requires coords= to be passed to benchmark()"
            continue

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
                if needs_coords:
                    learner.fit(X_tr, y_tr, coords_arr[train_idx])
                    preds = learner.predict(X_te, coords_arr[test_idx])
                else:
                    learner.fit(X_tr, y_tr)
                    preds = learner.predict(X_te)
            except (RuntimeError, ValueError, TypeError) as exc:
                # Incompatible or misconfigured learner (e.g. GaussianNB on a
                # regression target, a bad cost matrix). ValueError joined the
                # possible raises when typed errors landed (InvalidParameter
                # -> ValueError); catching only RuntimeError aborted the whole
                # run, losing every other learner's results.
                for m in metrics:
                    fold_scores[m].append(float("nan"))
                fold_scores.setdefault("_error", str(exc))
                continue

            for metric_name, metric_fn in metrics.items():
                fold_scores[metric_name].append(
                    metric_fn(y_te.tolist(), preds.tolist())
                )

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
    header = f"{'Learner':<20}" + "".join(f"  {m:>12}" for m in metrics) + "  note"
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
            if np.isnan(mean):
                row += "          n/a"
            else:
                row += f"  {mean:>5.3f}±{std:.3f}"
        if "_skipped" in scores:
            row += f"  (skipped: incompatible)"
        elif "_error" in scores:
            row += f"  (error)"
        lines.append(row)

    return "\n".join(lines)


_NON_PARAM_ATTRS = {
    "fit", "predict", "predict_proba", "feature_importances_",
    "shap_values", "permutation_importance", "conformal_predict",
    "supports_classification", "supports_regression",
}


def _get_params(learner):
    """Extract constructor parameters from a learner instance.

    Uses the estimator's own `get_params()` (sklearn-style, added to every
    wrapper). Falls back to a `dir()` scan only for objects without it — the
    scan finds nothing on `#[pyclass]` wrappers (their fields aren't `get`
    properties), which used to make `benchmark()` silently clone every
    learner with its constructor defaults instead of the params passed in.
    """
    get_params = getattr(learner, "get_params", None)
    if callable(get_params):
        return get_params()

    params = {}
    for attr in dir(learner):
        if attr.startswith("_") or attr in _NON_PARAM_ATTRS:
            continue
        try:
            val = getattr(learner, attr)
            if isinstance(val, (int, float, bool, str)):
                params[attr] = val
        except Exception:
            pass
    return params
