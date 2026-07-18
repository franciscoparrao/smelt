//! Construct a learner by its [`Learner::id`] string.
//!
//! Useful for data-driven experiment loops (benchmark sweeps, CLI
//! `--learner` flags) that iterate over a list of names instead of
//! hardcoding a `match` at every call site.

use super::{
    AdaBoost, AdaptiveRandomForest, CatBoost, DecisionTree, DeepForest, ElasticNet, ExtraTrees,
    ExtremeLearningMachine, GaussianNB, GradientBoosting, HoeffdingTree, KNearestNeighbors, Lasso,
    Learner, LearnerProperties, LightGBM, LinearRegression, LinearSVM, LogisticRegression,
    MondrianForest, MondrianTree, ObliqueForest, ObliqueTree, QuantileForest, QuantileGB,
    RandomForest, Ridge, XGBoost, EBM,
};
use crate::{Result, SmeltError};

/// Construct a learner by its `id()` string, using default hyperparameters.
///
/// Not every learner is registered: [`super::Bagging`], [`super::Stacking`],
/// [`super::DynamicEnsemble`], [`super::CostSensitiveClassifier`],
/// [`super::AutoTuner`], [`super::TargetTransformRegressor`] and
/// [`super::KrigingHybrid`] wrap *other* learners via a base-learner factory
/// that has no sensible default (`CostSensitiveClassifier` additionally
/// needs an explicit cost matrix with no sensible default either, and
/// `AutoTuner` a tuner spec + parameter space), and
/// [`super::GeoXGBoost`] needs training coordinates supplied externally.
/// [`super::ObliqueForest`] and [`EBM`], by contrast, are self-contained
/// (no factory, no external coordinates) and *are* registered. See
/// [`registered_learner_ids`] for the full list.
pub fn learner_from_id(id: &str) -> Result<Box<dyn Learner>> {
    Ok(match id {
        "adaboost" => Box::new(AdaBoost::default()),
        "adaptive_random_forest" => Box::new(AdaptiveRandomForest::default()),
        "catboost" => Box::new(CatBoost::default()),
        "decision_tree" => Box::new(DecisionTree::default()),
        "deep_forest" => Box::new(DeepForest::default()),
        "ebm" => Box::new(EBM::default()),
        "elastic_net" => Box::new(ElasticNet::default()),
        "elm" => Box::new(ExtremeLearningMachine::default()),
        "extra_trees" => Box::new(ExtraTrees::default()),
        "gaussian_nb" => Box::new(GaussianNB),
        "gradient_boosting" => Box::new(GradientBoosting::default()),
        "hoeffding_tree" => Box::new(HoeffdingTree::default()),
        "knn" => Box::new(KNearestNeighbors::default()),
        "lasso" => Box::new(Lasso::default()),
        "lightgbm" => Box::new(LightGBM::default()),
        "linear_regression" => Box::new(LinearRegression),
        "linear_svm" => Box::new(LinearSVM::default()),
        "logistic_regression" => Box::new(LogisticRegression::default()),
        "mondrian_forest" => Box::new(MondrianForest::default()),
        "mondrian_tree" => Box::new(MondrianTree::default()),
        "oblique_forest" => Box::new(ObliqueForest::default()),
        "oblique_tree" => Box::new(ObliqueTree::default()),
        "quantile_forest" => Box::new(QuantileForest::default()),
        "quantile_gb" => Box::new(QuantileGB::new(0.5)),
        "random_forest" => Box::new(RandomForest::default()),
        "ridge" => Box::new(Ridge::default()),
        "xgboost" => Box::new(XGBoost::default()),
        other => {
            return Err(SmeltError::InvalidParameter(format!(
                "unknown learner id \"{other}\" (or not registry-constructible: \
                 ensembles like bagging/stacking/dynamic_ensemble/kriging_hybrid need a \
                 base-learner factory, and geo_xgboost needs training coordinates)"
            )));
        }
    })
}

/// Query the declared [`LearnerProperties`] of a registered learner by id,
/// without the caller having to instantiate it by hand.
///
/// Constructs the learner via [`learner_from_id`] (default hyperparameters)
/// and returns its [`Learner::properties`]. Useful for capability-driven
/// filtering — e.g. "give me every registered learner that supports sample
/// weights":
///
/// ```
/// use smelt_ml::learner::{learner_properties, registered_learner_ids};
///
/// let weighted: Vec<&str> = registered_learner_ids()
///     .iter()
///     .copied()
///     .filter(|id| learner_properties(id).unwrap().supports_weights)
///     .collect();
/// assert!(weighted.contains(&"random_forest"));
/// assert!(!weighted.contains(&"knn"));
/// ```
///
/// Returns the same "unknown learner id" error as [`learner_from_id`] for an
/// unregistered id (including the factory-based composites).
pub fn learner_properties(id: &str) -> Result<LearnerProperties> {
    Ok(learner_from_id(id)?.properties())
}

/// All learner ids constructible via [`learner_from_id`].
pub fn registered_learner_ids() -> &'static [&'static str] {
    &[
        "adaboost",
        "adaptive_random_forest",
        "catboost",
        "decision_tree",
        "deep_forest",
        "ebm",
        "elastic_net",
        "elm",
        "extra_trees",
        "gaussian_nb",
        "gradient_boosting",
        "hoeffding_tree",
        "knn",
        "lasso",
        "lightgbm",
        "linear_regression",
        "linear_svm",
        "logistic_regression",
        "mondrian_forest",
        "mondrian_tree",
        "oblique_forest",
        "oblique_tree",
        "quantile_forest",
        "quantile_gb",
        "random_forest",
        "ridge",
        "xgboost",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_registered_id_constructs_and_matches_its_own_id() {
        for &id in registered_learner_ids() {
            let learner = learner_from_id(id).unwrap_or_else(|e| panic!("{id}: {e}"));
            assert_eq!(learner.id(), id, "learner_from_id(\"{id}\") returned a learner whose id() disagrees");
        }
    }

    #[test]
    fn unknown_id_is_an_error() {
        assert!(learner_from_id("not_a_real_learner").is_err());
    }

    #[test]
    fn factory_based_ensemble_ids_are_not_registered() {
        for id in [
            "bagging",
            "stacking",
            "dynamic_ensemble",
            "cost_sensitive",
            "kriging_hybrid",
            "geo_xgboost",
        ] {
            assert!(
                learner_from_id(id).is_err(),
                "{id} should not be registry-constructible"
            );
        }
    }

    /// Regression test (HIGH-5, `docs/auditoria_motor_2026-07-05.md`): the
    /// README's "27 self-contained learners" / "33 supervised learners"
    /// claims went stale for months without anything catching it. This
    /// doesn't validate the README text itself (nothing outside a doc test
    /// can), but pins the registry count to a literal so that adding or
    /// removing a registered learner without updating README.md's "Model
    /// registry" bullet and "All Supervised Learners" table fails CI
    /// immediately instead of silently drifting.
    #[test]
    fn registered_learner_count_matches_readme_claim() {
        assert_eq!(
            registered_learner_ids().len(),
            27,
            "registry count changed -- update the \"Model registry\" bullet \
             and \"All Supervised Learners\" table in README.md to match"
        );
    }
}
