//! Statistical test wrappers: Wilcoxon signed-rank, bootstrap CI, sign test.

use crate::common::smelt_err;
use pyo3::prelude::*;

// ── Stats ──────────────────────────────────────────────────────────────

#[pyclass]
#[derive(Clone)]
pub(crate) struct TestResult {
    #[pyo3(get)]
    test: String,
    #[pyo3(get)]
    statistic: f64,
    #[pyo3(get)]
    p_value: f64,
    #[pyo3(get)]
    significant: bool,
}

#[pymethods]
impl TestResult {
    fn __repr__(&self) -> String {
        format!(
            "TestResult(test='{}', statistic={:.4}, p_value={:.4}, significant={})",
            self.test, self.statistic, self.p_value, self.significant
        )
    }
}

#[pyfunction]
pub(crate) fn wilcoxon_signed_rank(a: Vec<f64>, b: Vec<f64>) -> PyResult<TestResult> {
    let r = smelt_ml::stats::wilcoxon_signed_rank(&a, &b).map_err(smelt_err)?;
    Ok(TestResult {
        test: r.test.to_string(),
        statistic: r.statistic,
        p_value: r.p_value,
        significant: r.significant,
    })
}

#[pyfunction]
#[pyo3(signature = (scores, confidence=0.95, n_bootstrap=10000, seed=42))]
pub(crate) fn bootstrap_ci(
    scores: Vec<f64>,
    confidence: f64,
    n_bootstrap: usize,
    seed: u64,
) -> (f64, f64, f64) {
    let r = smelt_ml::stats::bootstrap_ci(&scores, confidence, n_bootstrap, seed);
    (r.estimate, r.lower, r.upper)
}

#[pyfunction]
pub(crate) fn sign_test(a: Vec<f64>, b: Vec<f64>) -> PyResult<TestResult> {
    let r = smelt_ml::stats::sign_test(&a, &b).map_err(smelt_err)?;
    Ok(TestResult {
        test: r.test.to_string(),
        statistic: r.statistic,
        p_value: r.p_value,
        significant: r.significant,
    })
}

