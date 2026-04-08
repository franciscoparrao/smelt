# Cover Letter — Journal of Statistical Software

Dear Editors,

We submit the manuscript "smelt-ml: A Pure-Rust Machine Learning Framework Combining Spatial Modeling, Conformal Prediction, and Cache-Optimized Gradient Boosting" for consideration as a Software Article in the Journal of Statistical Software.

## Summary

smelt-ml is an open-source machine learning framework implemented entirely in Rust that provides 27 supervised learners, spatial modeling, conformal prediction, and built-in statistical testing for model comparison — all in a single, composable package with zero unsafe code. The software is published on crates.io (v1.3.0) with source code on GitHub.

## Why this paper fits JSS

We believe this manuscript is appropriate for JSS for three reasons:

**1. A genuine integration contribution, not a reimplementation.** While individual algorithms (XGBoost, Random Forest, etc.) exist in many languages, smelt-ml is the only framework — to our knowledge, in any language — that integrates Geographical-XGBoost (Grekousis, 2025), spatial cross-validation, conformal prediction with guaranteed coverage (Romano et al., 2019), and non-parametric statistical tests (Wilcoxon, Friedman) in a single composable pipeline. This integration is not merely convenient but methodologically important: it prevents the spatial leakage bugs that arise when practitioners combine multiple packages with incompatible APIs (Roberts et al., 2017). We demonstrate a 31% optimism bias from spatial leakage on a real housing dataset — a magnitude that would lead to operationally misleading conclusions.

**2. Empirically validated performance.** The gradient boosting implementations employ a cache-optimized column-major histogram accumulation strategy with u8 packing and histogram subtraction. We benchmark against the official C++ libraries (XGBoost 3.1, LightGBM 4.6, CatBoost 1.2) with 10 runs per configuration and apply paired Wilcoxon signed-rank tests with Bonferroni correction — using smelt-ml's own statistical testing module. The CatBoost-inspired implementation is significantly faster than C++ at 5 of 6 dataset sizes (p < 0.05 after correction at N ≤ 1,000), while consuming 25–40× less memory.

**3. Statistical testing as a first-class citizen.** The framework includes built-in Wilcoxon signed-rank, Friedman, Nemenyi post-hoc, McNemar, and bootstrap confidence interval tests — features typically requiring scipy.stats or R's stats package. We use these tools to validate our own benchmark claims within the paper, demonstrating the integrated approach. We believe this is directly relevant to JSS's readership.

## Addressing potential concerns

*"Why Rust instead of Python?"* — The primary use cases are: (1) Rust applications requiring embedded ML without Python runtime dependencies; (2) spatial ML workflows requiring integrated spatial CV + conformal prediction; (3) safety-critical systems where zero unsafe code and compile-time thread safety are requirements; and (4) memory-constrained environments where the 25–40× memory reduction is operationally significant.

*"Is this a complete framework?"* — smelt-ml provides 27 supervised learners (more than linfa's 9, the primary Rust ML crate), plus clustering, survival analysis, causal inference, and spatial ML. We are transparent about implementation scope: our CatBoost and LightGBM are "inspired by" rather than feature-complete reimplementations, and we provide a detailed feature comparison table (Table 4).

## Reproducibility

A complete replication package is provided in `paper/replication/` with scripts that reproduce all benchmark tables and figures. The software is available at https://crates.io/crates/smelt-ml under the MIT license.

## Conflicts of interest

The author declares no conflicts of interest.

Sincerely,

Francisco Parra
Departamento de Ingeniería Geográfica
Universidad de Santiago de Chile
francisco.parra.o@usach.cl
