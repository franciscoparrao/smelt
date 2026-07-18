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


def test_sample_weight_roundtrip_random_forest():
    """fit(x, y, sample_weight=...): all-ones must match unweighted exactly
    (the Rust weighted paths are written to be bit-identical for uniform
    weights), and a skewed weighting must actually change predictions."""
    x, y = _regress_data(seed=6)
    unweighted = smelt.RandomForest(n_estimators=25, max_depth=6, seed=7)
    unweighted.fit(x[:90], y[:90])
    ones = smelt.RandomForest(n_estimators=25, max_depth=6, seed=7)
    ones.fit(x[:90], y[:90], sample_weight=np.ones(90))
    np.testing.assert_array_equal(unweighted.predict(x[90:]), ones.predict(x[90:]))

    skewed_w = np.ones(90)
    skewed_w[:30] = 50.0
    skewed = smelt.RandomForest(n_estimators=25, max_depth=6, seed=7)
    skewed.fit(x[:90], y[:90], sample_weight=skewed_w)
    assert not np.array_equal(unweighted.predict(x[90:]), skewed.predict(x[90:]))

    assert smelt.RandomForest().supports_sample_weight is True


def test_sample_weight_invalid_inputs_raise_value_error():
    """Invalid sample_weight must raise ValueError from the binding's own
    validation (never a pyo3 PanicException from Task::with_weights), and a
    weight-blind learner (KNN) must reject weights with a clear ValueError
    naming itself."""
    x, y = _regress_data(seed=7)
    model = smelt.RandomForest(n_estimators=5)
    with pytest.raises(ValueError, match="one weight per sample"):
        model.fit(x, y, sample_weight=np.ones(10))
    with pytest.raises(ValueError, match="finite"):
        model.fit(x, y, sample_weight=np.full(len(y), np.nan))
    with pytest.raises(ValueError, match=">= 0"):
        w = np.ones(len(y))
        w[3] = -1.0
        model.fit(x, y, sample_weight=w)
    with pytest.raises(ValueError, match="at least one"):
        model.fit(x, y, sample_weight=np.zeros(len(y)))

    knn = smelt.KNearestNeighbors(k=3)
    assert knn.supports_sample_weight is False
    with pytest.raises(ValueError, match="does not support sample weights"):
        knn.fit(x, y, sample_weight=np.ones(len(y)))


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


def test_auto_tuner_fit_predict_and_nested_benchmark():
    """AutoTuner tunes + refits (best_params_ ∈ space, model refit with them),
    reruns cleanly inside benchmark()'s outer CV (nested CV), and rejects a bad
    learner id / non-tunable param with ValueError. save() is unsupported."""
    from smelt.benchmark import benchmark

    x, y = _classif_data(seed=8)

    at = smelt.AutoTuner(
        learner="dt",
        param_space={"max_depth": [2, 4, 8]},
        tuner="grid",
        cv=3,
        metric="accuracy",
        seed=1,
    )
    at.fit(x, y)
    assert at.best_params_["max_depth"] in (2, 4, 8)
    assert 0.0 <= at.best_score_ <= 1.0
    assert (at.predict(x) == y).mean() > 0.9

    # Nested CV: AutoTuner (inner tuning) inside benchmark's outer CV.
    outer = smelt.CrossValidation(3, seed=2)
    res = benchmark({"auto": at}, x, y, cv=outer)
    assert res["auto"]["accuracy"]["mean"] > 0.8

    # get_params round-trips through the constructor (this is how benchmark
    # clones per fold), and set_params re-validates the learner id.
    params = at.get_params()
    assert params["learner"] == "dt" and params["tuner"] == "grid"
    smelt.AutoTuner(**params)  # must reconstruct cleanly
    with pytest.raises(ValueError):
        at.set_params(learner="bogus")

    # Bad learner id at construction, non-tunable param at fit.
    with pytest.raises(ValueError):
        smelt.AutoTuner(learner="not_a_learner", param_space={"x": [1]})
    bad = smelt.AutoTuner(
        learner="ridge", param_space={"nope": [1.0]}, tuner="grid", cv=3, metric="rmse"
    )
    xr, yr = _regress_data(seed=8)
    with pytest.raises(ValueError):
        bad.fit(xr, yr)

    # save() is unsupported for this factory-built composite.
    with pytest.raises(NotImplementedError):
        at.save("/tmp/smelt_autotuner_should_not_exist.json")
