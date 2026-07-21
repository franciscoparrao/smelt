//! Hyperparameter tuning: grid search, random search.

pub mod bayesian;
pub mod grid_search;
pub mod hyperband;
pub mod random_search;

pub use bayesian::BayesianOptimizer;
pub use grid_search::GridSearch;
pub use hyperband::Hyperband;
pub use random_search::RandomSearch;

use crate::{Result, SmeltError};
use rand::Rng;
use std::collections::HashMap;

/// A single hyperparameter value (audit issue M10: `ParamSet` used to be a
/// bare `HashMap<String, f64>`, forcing every hyperparameter through `f64` --
/// a factory closure read an integer via `params["max_depth"] as usize`, and
/// a string-valued hyperparameter like an `objective`/`variogram_model`
/// choice had no representation at all). `Float`/`Int`/`Bool` interconvert
/// via the `as_*` accessors (matching the old cast-based access pattern,
/// including `as_usize`'s truncation of a `Float` for behavioral parity with
/// existing seeded tuning runs); `Str` only accepts `as_str`.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamValue {
    /// A floating-point value (e.g. `learning_rate`).
    Float(f64),
    /// An integer value (e.g. `max_depth`, `n_estimators`).
    Int(i64),
    /// A boolean flag.
    Bool(bool),
    /// A string enum choice (e.g. `objective`, `variogram_model`).
    Str(String),
}

impl ParamValue {
    /// Read as `f64`. `Int`/`Bool` convert; `Str` errors.
    pub fn as_f64(&self) -> Result<f64> {
        match self {
            Self::Float(v) => Ok(*v),
            Self::Int(v) => Ok(*v as f64),
            Self::Bool(v) => Ok(if *v { 1.0 } else { 0.0 }),
            Self::Str(s) => Err(SmeltError::InvalidParameter(format!(
                "expected a numeric parameter value, got string \"{s}\""
            ))),
        }
    }

    /// Read as `usize`. A `Float` truncates the same way the old bare
    /// `params[key] as usize` cast did; `Int` rejects negative values
    /// instead of silently wrapping.
    pub fn as_usize(&self) -> Result<usize> {
        match self {
            Self::Int(v) => usize::try_from(*v).map_err(|_| {
                SmeltError::InvalidParameter(format!(
                    "parameter value {v} is negative, cannot convert to usize"
                ))
            }),
            Self::Float(v) => Ok(*v as usize),
            Self::Bool(v) => Ok(*v as usize),
            Self::Str(s) => Err(SmeltError::InvalidParameter(format!(
                "expected a numeric parameter value, got string \"{s}\""
            ))),
        }
    }

    /// Read as `i64`.
    pub fn as_i64(&self) -> Result<i64> {
        match self {
            Self::Int(v) => Ok(*v),
            Self::Float(v) => Ok(*v as i64),
            Self::Bool(v) => Ok(*v as i64),
            Self::Str(s) => Err(SmeltError::InvalidParameter(format!(
                "expected a numeric parameter value, got string \"{s}\""
            ))),
        }
    }

    /// Read as `bool`. `Int`/`Float` treat nonzero as `true`.
    pub fn as_bool(&self) -> Result<bool> {
        match self {
            Self::Bool(v) => Ok(*v),
            Self::Int(v) => Ok(*v != 0),
            Self::Float(v) => Ok(*v != 0.0),
            Self::Str(s) => Err(SmeltError::InvalidParameter(format!(
                "expected a boolean parameter value, got string \"{s}\""
            ))),
        }
    }

    /// Read as a string slice. Only valid for `Str`.
    pub fn as_str(&self) -> Result<&str> {
        match self {
            Self::Str(s) => Ok(s),
            other => Err(SmeltError::InvalidParameter(format!(
                "expected a string parameter value, got {other:?}"
            ))),
        }
    }
}

impl From<f64> for ParamValue {
    fn from(v: f64) -> Self {
        Self::Float(v)
    }
}
impl From<i64> for ParamValue {
    fn from(v: i64) -> Self {
        Self::Int(v)
    }
}
impl From<usize> for ParamValue {
    fn from(v: usize) -> Self {
        Self::Int(v as i64)
    }
}
impl From<bool> for ParamValue {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}
impl From<String> for ParamValue {
    fn from(v: String) -> Self {
        Self::Str(v)
    }
}
impl From<&str> for ParamValue {
    fn from(v: &str) -> Self {
        Self::Str(v.to_string())
    }
}

/// A single set of hyperparameter values.
pub type ParamSet = HashMap<String, ParamValue>;

/// A grid of hyperparameter values for exhaustive search.
pub type ParamGrid = HashMap<String, Vec<ParamValue>>;

/// Distribution for sampling hyperparameters. `Uniform`/`LogUniform` always
/// sample a `ParamValue::Float`; `Choice` can hold any mix of value types
/// (e.g. a string enum choice), sampled uniformly at random.
#[derive(Clone)]
pub enum ParamDistribution {
    /// Uniform distribution over [low, high].
    Uniform(f64, f64),
    /// Log-uniform distribution: 10^Uniform(log10(low), log10(high)).
    LogUniform(f64, f64),
    /// Choose from a fixed set of values.
    Choice(Vec<ParamValue>),
}

/// A space of hyperparameter distributions for random search.
pub type ParamSpace = HashMap<String, ParamDistribution>;

/// A condition on a parent parameter's value, used by [`Dependency`].
///
/// Equality uses `ParamValue`'s own `PartialEq`, so conditions are meant for
/// the discrete parents that actually gate other parameters (a string
/// `objective`/`kernel`, an integer `degree`, a boolean flag) — not for
/// exact-matching a continuous `Float` draw, which would essentially never
/// hold.
#[derive(Clone, Debug, PartialEq)]
pub enum Condition {
    /// The parent parameter must equal this value.
    Equals(ParamValue),
    /// The parent parameter must be one of these values.
    In(Vec<ParamValue>),
}

impl Condition {
    /// Whether `value` (a parent's sampled/gridded value) satisfies this.
    fn is_satisfied_by(&self, value: &ParamValue) -> bool {
        match self {
            Condition::Equals(target) => value == target,
            Condition::In(targets) => targets.contains(value),
        }
    }
}

/// A conditional dependency between tuning parameters: `child` is an *active*
/// hyperparameter only when `parent`'s value satisfies `cond`.
///
/// Inactive children are pruned before the learner factory ever sees them, so
/// a parameter that only matters under one setting of another — `huber_delta`
/// only when `objective == "huber"`, `degree` only when `kernel == "poly"` —
/// can't silently waste tuning trials on bit-identical scores. This is the
/// general form of the one-off guard that closed 5th-audit M-5 (tuning
/// `huber_delta` with no `"huber"` objective was a no-op where every trial
/// trained the identical model).
///
/// Registered on a tuner via `with_dependency`; grid search additionally
/// deduplicates the combinations that collapse once inactive children are
/// dropped, and both tuners reject a dependency whose `parent` isn't in the
/// space (the exact M-5 misconfiguration).
///
/// # Examples
///
/// ```
/// use smelt_ml::tuning::{Dependency, Condition, ParamValue};
///
/// // `huber_delta` is a hyperparameter only when `objective == "huber"`.
/// let dep = Dependency::equals("huber_delta", "objective", "huber");
/// assert_eq!(dep.parent, "objective");
/// assert_eq!(dep.cond, Condition::Equals(ParamValue::Str("huber".into())));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Dependency {
    /// The dependent parameter, active only when the condition holds.
    pub child: String,
    /// The parameter whose value gates `child`.
    pub parent: String,
    /// The condition on `parent` that activates `child`.
    pub cond: Condition,
}

impl Dependency {
    /// `child` is active only when `parent` equals `value`.
    pub fn equals(
        child: impl Into<String>,
        parent: impl Into<String>,
        value: impl Into<ParamValue>,
    ) -> Self {
        Self {
            child: child.into(),
            parent: parent.into(),
            cond: Condition::Equals(value.into()),
        }
    }

    /// `child` is active only when `parent` is one of `values`.
    pub fn in_values(
        child: impl Into<String>,
        parent: impl Into<String>,
        values: Vec<ParamValue>,
    ) -> Self {
        Self {
            child: child.into(),
            parent: parent.into(),
            cond: Condition::In(values),
        }
    }
}

/// Sample one `ParamSet` from `space` — shared by `RandomSearch`,
/// `BayesianOptimizer` (initial/random rounds), and `Hyperband`, which
/// previously each duplicated this same match-on-`ParamDistribution` logic.
///
/// Keys are sorted before drawing (same convention as `cartesian_product`):
/// iterating the `HashMap` directly would assign the RNG's draws to
/// parameters in `RandomState` order, which differs per process -- so the
/// same seed produced different configurations across runs, breaking the
/// reproducibility `with_seed` promises.
pub(crate) fn sample_param_space(space: &ParamSpace, rng: &mut impl Rng) -> ParamSet {
    let mut keys: Vec<&String> = space.keys().collect();
    keys.sort();
    keys.into_iter()
        .map(|name| (name.clone(), sample_one(&space[name], rng)))
        .collect()
}

/// Validate every distribution in `space` before any sampling happens --
/// shared by `RandomSearch`, `BayesianOptimizer`, and `Hyperband`. Without
/// this, `Uniform(lo > hi)` panics inside `rng.random_range` and
/// `LogUniform(<= 0)` produces NaN bounds, both deep in a tuning loop
/// instead of at the entry point.
pub(crate) fn validate_param_space(space: &ParamSpace) -> Result<()> {
    for (name, dist) in space {
        match dist {
            ParamDistribution::Uniform(lo, hi) => {
                if !lo.is_finite() || !hi.is_finite() || lo > hi {
                    return Err(SmeltError::InvalidParameter(format!(
                        "param '{name}': Uniform bounds must be finite with low <= high, got ({lo}, {hi})"
                    )));
                }
            }
            ParamDistribution::LogUniform(lo, hi) => {
                if !lo.is_finite() || !hi.is_finite() || *lo <= 0.0 || lo > hi {
                    return Err(SmeltError::InvalidParameter(format!(
                        "param '{name}': LogUniform bounds must be finite with 0 < low <= high, got ({lo}, {hi})"
                    )));
                }
            }
            ParamDistribution::Choice(values) => {
                if values.is_empty() {
                    return Err(SmeltError::InvalidParameter(format!(
                        "param '{name}': Choice requires at least one value"
                    )));
                }
            }
        }
    }
    Ok(())
}

/// Sample a single value from one `ParamDistribution`.
pub(crate) fn sample_one(dist: &ParamDistribution, rng: &mut impl Rng) -> ParamValue {
    match dist {
        ParamDistribution::Uniform(lo, hi) => ParamValue::Float(rng.random_range(*lo..=*hi)),
        ParamDistribution::LogUniform(lo, hi) => {
            let log_lo = lo.log10();
            let log_hi = hi.log10();
            ParamValue::Float(10.0f64.powf(rng.random_range(log_lo..=log_hi)))
        }
        ParamDistribution::Choice(values) => values[rng.random_range(0..values.len())].clone(),
    }
}

/// Result of a tuning run.
#[derive(Debug)]
pub struct TuneResult {
    /// Best hyperparameter configuration found.
    pub best_params: ParamSet,
    /// Score of the best configuration.
    pub best_score: f64,
    /// All evaluated configurations with their scores.
    pub all_results: Vec<(ParamSet, f64)>,
    /// Measure used for evaluation.
    pub measure_id: String,
    /// Whether higher scores are better.
    pub maximize: bool,
}

impl TuneResult {
    pub(crate) fn select_best(
        results: Vec<(ParamSet, f64)>,
        measure_id: String,
        maximize: bool,
    ) -> Result<Self> {
        if results.is_empty() {
            return Err(SmeltError::InvalidParameter(
                "tuning produced no candidates to select from (n_iter=0 or an empty grid?)".into(),
            ));
        }
        let best_idx = if maximize {
            results
                .iter()
                .enumerate()
                .max_by(|a, b| {
                    a.1.1
                        .partial_cmp(&b.1.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .expect("checked non-empty above")
                .0
        } else {
            results
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    a.1.1
                        .partial_cmp(&b.1.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .expect("checked non-empty above")
                .0
        };

        Ok(Self {
            best_params: results[best_idx].0.clone(),
            best_score: results[best_idx].1,
            all_results: results,
            measure_id,
            maximize,
        })
    }
}

/// Generate the Cartesian product of all parameter values.
pub(crate) fn cartesian_product(grid: &ParamGrid) -> Vec<ParamSet> {
    let mut keys: Vec<&String> = grid.keys().collect();
    keys.sort(); // deterministic order
    let values: Vec<&Vec<ParamValue>> = keys.iter().map(|k| &grid[*k]).collect();

    if keys.is_empty() {
        return vec![ParamSet::new()];
    }

    let mut result = Vec::new();
    let mut indices = vec![0usize; keys.len()];

    loop {
        let mut params = ParamSet::new();
        for (i, key) in keys.iter().enumerate() {
            params.insert((*key).clone(), values[i][indices[i]].clone());
        }
        result.push(params);

        let mut carry = true;
        for i in (0..keys.len()).rev() {
            if carry {
                indices[i] += 1;
                if indices[i] >= values[i].len() {
                    indices[i] = 0;
                } else {
                    carry = false;
                }
            }
        }
        if carry {
            break;
        }
    }

    result
}

/// Remove every parameter in `params` whose dependency is unsatisfied — its
/// parent is absent, or present with a value the condition rejects.
///
/// Applied to a fixpoint: dropping a child that is *itself* a parent of
/// another dependency leaves that grandchild's parent absent on the next
/// pass, so a whole dependency chain collapses in one call. Sampling happens
/// before pruning in the tuners, so the RNG stream (and thus seeded
/// reproducibility) is unaffected — pruning only decides which sampled keys
/// reach the factory.
pub(crate) fn prune_inactive(params: &mut ParamSet, deps: &[Dependency]) {
    loop {
        let inactive = deps.iter().find_map(|d| {
            if !params.contains_key(&d.child) {
                return None;
            }
            let active = params
                .get(&d.parent)
                .is_some_and(|pv| d.cond.is_satisfied_by(pv));
            (!active).then(|| d.child.clone())
        });
        match inactive {
            Some(child) => {
                params.remove(&child);
            }
            None => break,
        }
    }
}

/// Validate dependencies against the set of tunable parameter `names`.
///
/// Rejects a dependency whose `child` or `parent` isn't among `names` — the
/// M-5 misconfiguration is exactly "tune `huber_delta` while `objective`
/// isn't in the space" — and a cyclic dependency (a parameter reachable from
/// itself by following parent links), which [`prune_inactive`] would
/// otherwise resolve by arbitrary iteration order.
pub(crate) fn validate_dependencies(
    names: &std::collections::HashSet<&str>,
    deps: &[Dependency],
) -> Result<()> {
    for d in deps {
        if !names.contains(d.child.as_str()) {
            return Err(SmeltError::InvalidParameter(format!(
                "dependency references child parameter '{}', which is not in the tuning space",
                d.child
            )));
        }
        if !names.contains(d.parent.as_str()) {
            return Err(SmeltError::InvalidParameter(format!(
                "parameter '{}' depends on '{}', which is not in the tuning space; add it or \
                 remove the dependent parameter (otherwise it can never activate)",
                d.child, d.parent
            )));
        }
    }

    // Cycle check: follow parent links from every child; revisiting a node
    // means a cycle (a -> b -> a would make activation ill-defined).
    for start in deps.iter().map(|d| d.child.as_str()) {
        let mut seen = std::collections::HashSet::new();
        let mut node = start;
        loop {
            if !seen.insert(node) {
                return Err(SmeltError::InvalidParameter(format!(
                    "cyclic parameter dependency detected involving '{node}'"
                )));
            }
            // A node can appear as `child` in at most the first matching dep
            // for traversal purposes; multiple deps on one child are ANDed but
            // still form the same parent edges for cycle detection.
            match deps.iter().find(|d| d.child == node) {
                Some(d) => node = d.parent.as_str(),
                None => break,
            }
        }
    }
    Ok(())
}

/// The Cartesian product of `grid`, with `deps` applied: inactive children are
/// pruned from each combination, then the duplicates that collapse together
/// are dropped (every `huber_delta` value under `objective != "huber"` becomes
/// the same trial — the M-5 waste, removed structurally). First-seen order is
/// preserved for determinism. The `O(n^2)` dedup is fine: grids are small and
/// far outnumbered by the CV work each surviving combination triggers.
pub(crate) fn cartesian_product_with_deps(grid: &ParamGrid, deps: &[Dependency]) -> Vec<ParamSet> {
    if deps.is_empty() {
        return cartesian_product(grid);
    }
    let mut out: Vec<ParamSet> = Vec::new();
    for mut combo in cartesian_product(grid) {
        prune_inactive(&mut combo, deps);
        if !out.contains(&combo) {
            out.push(combo);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// Regression test (4th audit, HIGH-6): the draw→parameter assignment
    /// must not depend on `HashMap` iteration order (per-process
    /// `RandomState`), or the same seed samples different configurations
    /// across runs. Two spaces with identical distributions but different
    /// insertion orders and capacities must sample identically.
    #[test]
    fn sample_param_space_is_independent_of_hashmap_iteration_order() {
        let names: Vec<String> = (0..10).map(|i| format!("param_{i}")).collect();

        let mut space_a = ParamSpace::new();
        for (i, n) in names.iter().enumerate() {
            space_a.insert(n.clone(), ParamDistribution::Uniform(0.0, (i + 1) as f64));
        }
        let mut space_b = ParamSpace::with_capacity(512);
        for (i, n) in names.iter().enumerate().rev() {
            space_b.insert(n.clone(), ParamDistribution::Uniform(0.0, (i + 1) as f64));
        }

        let sampled_a = sample_param_space(&space_a, &mut StdRng::seed_from_u64(42));
        let sampled_b = sample_param_space(&space_b, &mut StdRng::seed_from_u64(42));
        for n in &names {
            assert_eq!(
                sampled_a[n], sampled_b[n],
                "{n}: same seed must assign the same draw regardless of map layout"
            );
        }
    }

    /// 4th-audit LOW: invalid distributions used to panic inside
    /// `rng.random_range` (Uniform with low > high) or produce NaN bounds
    /// (LogUniform with non-positive low) deep in the tuning loop. They must
    /// be rejected up front with `InvalidParameter` naming the parameter.
    #[test]
    fn validate_param_space_rejects_invalid_distributions() {
        let cases: Vec<(&str, ParamDistribution)> = vec![
            ("uniform_inverted", ParamDistribution::Uniform(5.0, 1.0)),
            ("uniform_nan", ParamDistribution::Uniform(f64::NAN, 1.0)),
            (
                "loguniform_zero_low",
                ParamDistribution::LogUniform(0.0, 10.0),
            ),
            (
                "loguniform_negative",
                ParamDistribution::LogUniform(-1.0, 10.0),
            ),
            (
                "loguniform_inverted",
                ParamDistribution::LogUniform(10.0, 1.0),
            ),
            ("choice_empty", ParamDistribution::Choice(vec![])),
        ];
        for (name, dist) in cases {
            let mut space = ParamSpace::new();
            space.insert(name.to_string(), dist);
            let err = validate_param_space(&space).unwrap_err();
            let msg = err.to_string();
            assert!(
                msg.contains(name),
                "{name}: error must name the offending parameter, got: {msg}"
            );
        }
    }

    #[test]
    fn validate_param_space_accepts_valid_distributions() {
        let mut space = ParamSpace::new();
        space.insert("u".into(), ParamDistribution::Uniform(0.0, 1.0));
        // Degenerate-but-harmless single-point range is allowed.
        space.insert("point".into(), ParamDistribution::Uniform(3.0, 3.0));
        space.insert("log".into(), ParamDistribution::LogUniform(1e-4, 10.0));
        space.insert(
            "c".into(),
            ParamDistribution::Choice(vec![ParamValue::Int(1), ParamValue::Str("a".into())]),
        );
        assert!(validate_param_space(&space).is_ok());
    }

    fn pset(pairs: &[(&str, ParamValue)]) -> ParamSet {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn prune_keeps_active_child_and_drops_inactive_one() {
        let dep = Dependency::equals("huber_delta", "objective", "huber");

        // objective == "huber" -> huber_delta stays.
        let mut active = pset(&[
            ("objective", ParamValue::Str("huber".into())),
            ("huber_delta", ParamValue::Float(2.0)),
        ]);
        prune_inactive(&mut active, std::slice::from_ref(&dep));
        assert!(active.contains_key("huber_delta"));

        // objective == "squared_error" -> huber_delta pruned (the M-5 no-op).
        let mut inactive = pset(&[
            ("objective", ParamValue::Str("squared_error".into())),
            ("huber_delta", ParamValue::Float(2.0)),
        ]);
        prune_inactive(&mut inactive, std::slice::from_ref(&dep));
        assert!(!inactive.contains_key("huber_delta"));
        assert!(inactive.contains_key("objective"));
    }

    #[test]
    fn prune_drops_child_when_parent_absent() {
        let dep = Dependency::equals("degree", "kernel", "poly");
        let mut p = pset(&[("degree", ParamValue::Int(3))]);
        prune_inactive(&mut p, &[dep]);
        assert!(!p.contains_key("degree"), "no parent present -> inactive");
    }

    #[test]
    fn prune_collapses_a_dependency_chain_to_fixpoint() {
        // c depends on b == true; b depends on a == "on". a == "off" makes b
        // inactive, which in turn must make c inactive in the same call.
        let deps = vec![
            Dependency::equals("b", "a", "on"),
            Dependency::equals("c", "b", true),
        ];
        let mut p = pset(&[
            ("a", ParamValue::Str("off".into())),
            ("b", ParamValue::Bool(true)),
            ("c", ParamValue::Float(1.0)),
        ]);
        prune_inactive(&mut p, &deps);
        assert_eq!(p.keys().collect::<Vec<_>>(), vec![&"a".to_string()]);
    }

    #[test]
    fn prune_supports_in_condition() {
        let dep = Dependency::in_values(
            "child",
            "mode",
            vec![ParamValue::Str("a".into()), ParamValue::Str("b".into())],
        );
        let mut keep = pset(&[
            ("mode", ParamValue::Str("b".into())),
            ("child", ParamValue::Int(1)),
        ]);
        prune_inactive(&mut keep, std::slice::from_ref(&dep));
        assert!(keep.contains_key("child"));

        let mut drop = pset(&[
            ("mode", ParamValue::Str("c".into())),
            ("child", ParamValue::Int(1)),
        ]);
        prune_inactive(&mut drop, &[dep]);
        assert!(!drop.contains_key("child"));
    }

    #[test]
    fn validate_dependencies_rejects_missing_parent_and_child() {
        let names: std::collections::HashSet<&str> = ["huber_delta"].into_iter().collect();
        // parent 'objective' not in space -> the exact M-5 misconfiguration.
        let err = validate_dependencies(
            &names,
            &[Dependency::equals("huber_delta", "objective", "huber")],
        )
        .unwrap_err();
        assert!(err.to_string().contains("objective"));

        // child not in space either.
        let names2: std::collections::HashSet<&str> = ["objective"].into_iter().collect();
        assert!(
            validate_dependencies(
                &names2,
                &[Dependency::equals("huber_delta", "objective", "huber")]
            )
            .is_err()
        );
    }

    #[test]
    fn validate_dependencies_detects_cycles() {
        let names: std::collections::HashSet<&str> = ["a", "b"].into_iter().collect();
        let deps = vec![
            Dependency::equals("a", "b", true),
            Dependency::equals("b", "a", true),
        ];
        let err = validate_dependencies(&names, &deps).unwrap_err();
        assert!(err.to_string().contains("cyclic"));
    }

    #[test]
    fn validate_dependencies_accepts_well_formed() {
        let names: std::collections::HashSet<&str> =
            ["objective", "huber_delta"].into_iter().collect();
        assert!(
            validate_dependencies(
                &names,
                &[Dependency::equals("huber_delta", "objective", "huber")]
            )
            .is_ok()
        );
    }

    #[test]
    fn cartesian_product_with_deps_dedups_collapsed_combinations() {
        // objective in {squared_error, huber} x huber_delta in {1, 2, 3}.
        // Full product = 6; but the 3 huber_delta values under squared_error
        // all collapse to one, so 3 (huber) + 1 (squared_error) = 4.
        let mut grid = ParamGrid::new();
        grid.insert(
            "objective".into(),
            vec![
                ParamValue::Str("squared_error".into()),
                ParamValue::Str("huber".into()),
            ],
        );
        grid.insert(
            "huber_delta".into(),
            vec![
                ParamValue::Float(1.0),
                ParamValue::Float(2.0),
                ParamValue::Float(3.0),
            ],
        );
        let deps = vec![Dependency::equals("huber_delta", "objective", "huber")];
        let combos = cartesian_product_with_deps(&grid, &deps);
        assert_eq!(combos.len(), 4, "6 raw combos collapse to 4");

        // Exactly one squared_error combo, and it carries no huber_delta.
        let se: Vec<_> = combos
            .iter()
            .filter(|c| c.get("objective") == Some(&ParamValue::Str("squared_error".into())))
            .collect();
        assert_eq!(se.len(), 1);
        assert!(!se[0].contains_key("huber_delta"));

        // The three huber combos keep their distinct huber_delta values.
        let huber: Vec<_> = combos
            .iter()
            .filter(|c| c.get("objective") == Some(&ParamValue::Str("huber".into())))
            .collect();
        assert_eq!(huber.len(), 3);
        assert!(huber.iter().all(|c| c.contains_key("huber_delta")));
    }

    #[test]
    fn cartesian_product_with_deps_is_identity_without_deps() {
        let mut grid = ParamGrid::new();
        grid.insert("x".into(), vec![ParamValue::Int(1), ParamValue::Int(2)]);
        assert_eq!(
            cartesian_product_with_deps(&grid, &[]),
            cartesian_product(&grid)
        );
    }
}
