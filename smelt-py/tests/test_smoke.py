"""Permanent smoke tests for the smelt Python bindings (audit 5a, M-10).

Before this file existed there was not a single ``test_*.py`` in the repo:
CI compiled smelt-py (clippy --workspace) but never imported the built
module, so a broken wheel or binding could only be caught by hand. These
tests exercise the already-published 0.7.0 API surface only — import +
version, fit/predict for one classifier and one regressor, a save/load
roundtrip with bit-identical predictions, and one short tuner run.

Run in CI by the ``python-smoke`` job in ``.github/workflows/ci.yml``
(maturin build → pip install wheel → pytest smelt-py/tests/).
"""

import numpy as np
import pytest

import smelt


def _classif_data(seed=0, n=120):
    """Two blobs, linearly separable: labels as int64."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(loc=-2.0, scale=0.7, size=(n // 2, 4))
    x1 = rng.normal(loc=2.0, scale=0.7, size=(n // 2, 4))
    x = np.vstack([x0, x1])
    y = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.int64)
    perm = rng.permutation(n)
    return x[perm], y[perm]


def _regress_data(seed=1, n=120):
    """y = 3*x0 - 2*x1 + noise: target as float64."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4))
    y = 3.0 * x[:, 0] - 2.0 * x[:, 1] + rng.normal(scale=0.1, size=n)
    return x, y


def test_import_and_version():
    assert isinstance(smelt.__version__, str)
    assert smelt.__version__ != ""
    # "0+unknown" is the fallback for a source tree WITHOUT an installed
    # package -- an installed wheel/maturin-develop build must carry the
    # real version stamped from smelt-py/Cargo.toml (audit M20).
    assert smelt.__version__ != "0+unknown"


def test_random_forest_classification():
    x, y = _classif_data()
    model = smelt.RandomForest(n_estimators=25, max_depth=6, seed=7)
    model.fit(x[:90], y[:90])
    preds = model.predict(x[90:])
    assert preds.shape == (30,)
    acc = smelt.accuracy_score(y[90:], preds)
    assert acc >= 0.9  # trivially separable blobs
    proba = model.predict_proba(x[90:])
    assert proba.shape == (30, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)


def test_xgboost_regression():
    x, y = _regress_data()
    model = smelt.XGBoost(n_estimators=50, max_depth=4, learning_rate=0.3)
    model.fit(x[:90], y[:90])
    preds = model.predict(x[90:])
    assert preds.shape == (30,)
    assert smelt.rmse_score(y[90:], preds) < 2.0  # sanity, not benchmark
    assert smelt.r2_score(y[90:], preds) > 0.5


def test_save_load_roundtrip(tmp_path):
    x, y = _regress_data(seed=3)
    model = smelt.XGBoost(n_estimators=30, max_depth=3)
    model.fit(x[:90], y[:90])
    before = model.predict(x[90:])

    path = str(tmp_path / "xgb_model.json")
    model.save(path)
    loaded = smelt.XGBoost.load(path, is_classif=False)
    after = loaded.predict(x[90:])

    assert np.array_equal(before, after)  # bit-identical, not just close


def test_save_load_rejects_wrong_type(tmp_path):
    x, y = _classif_data(seed=4)
    model = smelt.RandomForest(n_estimators=10)
    model.fit(x, y)
    path = str(tmp_path / "rf_model.json")
    model.save(path)
    with pytest.raises(Exception):
        smelt.XGBoost.load(path)


def test_bayesian_optimizer_short_run():
    x, y = _classif_data(seed=5)
    bo = smelt.BayesianOptimizer(n_iter=4, n_initial=2, seed=11)
    result = bo.optimize(
        "rf",
        {"n_estimators": [10, 20], "max_depth": (2, 6)},
        x,
        y,
        metric="accuracy",
        n_folds=3,
    )
    assert set(result) >= {"best_params", "best_score", "all_results"}
    assert set(result["best_params"]) == {"n_estimators", "max_depth"}
    assert 0.0 <= result["best_score"] <= 1.0
    assert len(result["all_results"]) == 4
