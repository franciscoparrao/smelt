"""Hyperparameter tuning with spatial-aware cross-validation."""

import numpy as np
from itertools import product

# Metrics where a lower score is better (error/loss metrics). Everything else
# bound in smelt.measures is a higher-is-better score (accuracy, F1, R^2, ...).
_MINIMIZE_METRIC_NAMES = frozenset({
    "rmse_score", "mae_score", "brier_score", "mape_score", "logloss_score",
})


def _resolve_maximize(metric, maximize):
    """Direction of optimization for `metric`.

    Mirrors the Rust core's `Measure::maximize()`: error metrics like RMSE/MAE
    must be minimized, not maximized. `maximize=None` (default) infers the
    direction from the metric's name; pass an explicit bool for custom
    metrics the name-based heuristic doesn't recognize.
    """
    if maximize is not None:
        return maximize
    return getattr(metric, "__name__", "") not in _MINIMIZE_METRIC_NAMES


class RandomSearch:
    """Random search hyperparameter tuning with any CV strategy.

    Parameters
    ----------
    learner_class : class
        Learner class (e.g., smelt.XGBoost).
    param_grid : dict
        Mapping of parameter name → list of values.
    cv : object
        Cross-validation splitter (CrossValidation, SpatialBlockCV).
    metric : callable
        Scoring function (e.g., auc_roc_score). Must accept (y_true, y_pred_or_proba).
    n_iter : int
        Number of random parameter combinations to try.
    use_proba : bool
        If True, pass predict_proba output to metric (for AUC-ROC).
    maximize : bool or None
        Whether a higher `metric` score is better. Defaults to `None`, which
        infers the direction from `metric`'s name (RMSE/MAE/Brier/MAPE/logloss
        are minimized, everything else — accuracy, F1, R^2, AUC, ... — is
        maximized). Pass an explicit bool for custom metrics.
    seed : int
        Random seed.

    Example
    -------
    >>> from smelt import XGBoost, SpatialBlockCV, auc_roc_score
    >>> from smelt.tuning import RandomSearch
    >>> tuner = RandomSearch(
    ...     XGBoost,
    ...     {"n_estimators": [50, 100, 200], "max_depth": [3, 6, 10]},
    ...     cv=SpatialBlockCV(5, coords),
    ...     metric=auc_roc_score,
    ...     use_proba=True,
    ...     n_iter=20,
    ... )
    >>> best = tuner.fit(X, y)
    >>> print(f"Best AUC: {best['score']:.3f}, Params: {best['params']}")
    """

    def __init__(self, learner_class, param_grid, cv, metric, n_iter=50,
                 use_proba=False, maximize=None, seed=42):
        self.learner_class = learner_class
        self.param_grid = param_grid
        self.cv = cv
        self.metric = metric
        self.n_iter = n_iter
        self.use_proba = use_proba
        self.maximize = _resolve_maximize(metric, maximize)
        self.seed = seed

    def fit(self, X, y):
        """Run random search and return best result.

        Returns
        -------
        dict with keys: 'score', 'params', 'all_results'
        """
        rng = np.random.RandomState(self.seed)
        n = len(y)
        splits = self.cv.splits(n)
        is_classif = np.issubdtype(y.dtype, np.integer)

        # Generate random param combinations
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        all_combos = list(product(*param_values))

        if self.n_iter < len(all_combos):
            indices = rng.choice(len(all_combos), size=self.n_iter, replace=False)
            combos = [all_combos[i] for i in indices]
        else:
            combos = all_combos

        maximize = self.maximize
        best_score = -np.inf if maximize else np.inf
        best_params = None
        all_results = []

        for combo in combos:
            params = dict(zip(param_names, combo))
            fold_scores = []

            for train_idx, test_idx in splits:
                if len(train_idx) == 0 or len(test_idx) == 0:
                    continue

                learner = self.learner_class(**params)
                learner.fit(X[train_idx], y[train_idx])

                if self.use_proba:
                    try:
                        proba = learner.predict_proba(X[test_idx])
                        score = self.metric(y[test_idx].tolist(), proba.tolist())
                    except Exception:
                        preds = learner.predict(X[test_idx])
                        score = self.metric(y[test_idx].tolist(), preds.tolist())
                else:
                    preds = learner.predict(X[test_idx])
                    score = self.metric(y[test_idx].tolist(), preds.tolist())

                fold_scores.append(score)

            mean_score = np.mean(fold_scores) if fold_scores else 0.0
            all_results.append({"params": params, "score": mean_score,
                                "fold_scores": fold_scores})

            better = mean_score > best_score if maximize else mean_score < best_score
            if better:
                best_score = mean_score
                best_params = params

        all_results.sort(key=lambda x: x["score"], reverse=maximize)

        return {
            "score": best_score,
            "params": best_params,
            "all_results": all_results,
        }


class GridSearch(RandomSearch):
    """Exhaustive grid search (all combinations).

    Same API as RandomSearch but tries all parameter combinations.
    """

    def __init__(self, learner_class, param_grid, cv, metric,
                 use_proba=False, maximize=None, seed=42):
        n_combos = 1
        for v in param_grid.values():
            n_combos *= len(v)
        super().__init__(learner_class, param_grid, cv, metric,
                         n_iter=n_combos, use_proba=use_proba,
                         maximize=maximize, seed=seed)
