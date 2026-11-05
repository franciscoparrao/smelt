# Cover letter: Journal of Statistical Software (George's version, 2026-11-05)

Dear Editors-in-Chief,

I am pleased to submit the manuscript "smelt-ml: A Pure-Rust Machine Learning Framework Combining Spatial Modeling, Conformal Prediction, and Cache-Optimized Gradient Boosting" for consideration as a software article in the Journal of Statistical Software.

smelt-ml is an open-source machine learning framework implemented entirely in Rust. It provides 27 supervised learners, clustering, causal inference, survival analysis, and the first Rust implementation of Geographical-XGBoost, using spatial cross-validation and conformal prediction. The crate is published on crates.io (smelt-ml, MIT license) and its Python bindings on PyPI (smelt-ml), so the framework is callable from Python while its core has no Python or R runtime dependencies.

The core contributions of this work are:

1. An integrated spatial ML pipeline. The framework combines Geographical-XGBoost (Grekousis, 2025), spatial block and buffer cross-validation, and split-conformal prediction in one composable trait system. To our knowledge no other library, in any language, offers the three together. The integration is methodological, not cosmetic: because the spatial resampling strategy is a first-class Resample object, it can be passed to tuning and benchmarking functions, which removes the custom loops where spatial leakage usually enters. On the King County housing data, random cross-validation understates the error of the same model by 27% relative to spatial block cross-validation, and the conformal intervals built on Geographical-XGBoost are 16 to 25% narrower than those of the non-spatial baseline at the same guaranteed coverage.

2. A native Rust framework. The 27 learners, the preprocessing pipeline, tuning (grid, random, Bayesian, Hyperband), and built-in non-parametric tests (Wilcoxon, Friedman with Nemenyi, McNemar, bootstrap intervals) share one Task, Learner, Measure, Resample architecture inspired by mlr3 and scikit-learn. The existing Rust ecosystem (linfa) covers nine algorithms and no gradient boosting.

3. High-performance, memory-safe execution. The three gradient boosting engines use column-major histogram storage with u8 bin packing and histogram subtraction. On single-threaded benchmarks against the official C++ libraries (10 runs per configuration, paired Wilcoxon tests with Bonferroni correction computed with the framework's own statistics module), the CatBoost-inspired engine is competitive at every tested size and XGBoost and LightGBM are competitive up to 10,000 samples, with 23 to 41 times lower peak memory. The codebase contains zero unsafe blocks.

The manuscript is a software article in the sense of the journal's mission: it documents the design of a statistical software system, shows its use on real data, validates it against reference implementations (the original geoxgboost package, scikit-learn), and states its limitations and scope explicitly, including a feature-by-feature comparison with the official boosting libraries. All tables and figures are reproducible from a single replication script included with the submission, with fixed seeds and the software versions recorded in the Computational details section. The paper also serves a readership the journal has not yet covered: statistical practitioners who need a memory-safe, dependency-free ML stack, and spatial analysts who need honest validation and calibrated uncertainty in one place.

The manuscript has not been published and is not under consideration elsewhere. Geographical-XGBoost was introduced by the corresponding author in the Journal of Geographical Systems (Grekousis, 2025); the present paper is the first implementation of the method outside the original Python package and extends it with spatial validation and conformal uncertainty. The benchmark and validation results have not appeared before.

The submission includes the software source code (crates.io and GitHub), the replication package (paper/replication/, single script replicate.sh, with the output files produced on our hardware), and the King County subset used in the case study.

The authors declare no conflicts of interest.

Sincerely,

George Grekousis (corresponding author, on behalf of the authors)
College of Arts and Sciences, VinUniversity, Hanoi, Vietnam
georgios.g@vinuni.edu.vn

Co-authors: Francisco Parra, Carolina Bonacic, Mauricio Marín (Universidad de Santiago de Chile); Verónica Gil-Costa (CONICET / Universidad Nacional de San Luis)
