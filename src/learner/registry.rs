//! Construct a learner by its [`Learner::id`] string.
//!
//! Useful for data-driven experiment loops (benchmark sweeps, CLI
//! `--learner` flags) that iterate over a list of names instead of
//! hardcoding a `match` at every call site.

use super::{
    AdaBoost, CatBoost, DecisionTree, ElasticNet, ExtraTrees, GaussianNB, GradientBoosting,
    HoeffdingTree, KNearestNeighbors, Lasso, Learner, LightGBM, LinearRegression, LinearSVM,
    LogisticRegression, ObliqueForest, ObliqueTree, QuantileForest, QuantileGB, RandomForest,
    Ridge, XGBoost,
};
use crate::{Result, SmeltError};

/// Construct a learner by its `id()` string, using default hyperparameters.
///
/// Not every learner is registered: [`super::Bagging`], [`super::Stacking`]
/// and [`super::DynamicEnsemble`] wrap *other* learners via a base-learner
/// factory that has no sensible default, and [`super::GeoXGBoost`] needs
/// training coordinates supplied externally. [`super::ObliqueForest`], by
/// contrast, is a self-contained ensemble of its own oblique trees (not
/// pluggable) and *is* registered. See [`registered_learner_ids`] for the
/// full list.
pub fn learner_from_id(id: &str) -> Result<Box<dyn Learner>> {
    Ok(match id {
        "adaboost" => Box::new(AdaBoost::default()),
        "catboost" => Box::new(CatBoost::default()),
        "decision_tree" => Box::new(DecisionTree::default()),
        "elastic_net" => Box::new(ElasticNet::default()),
        "extra_trees" => Box::new(ExtraTrees::default()),
        "gaussian_nb" => Box::new(GaussianNB::default()),
        "gradient_boosting" => Box::new(GradientBoosting::default()),
        "hoeffding_tree" => Box::new(HoeffdingTree::default()),
        "knn" => Box::new(KNearestNeighbors::default()),
        "lasso" => Box::new(Lasso::default()),
        "lightgbm" => Box::new(LightGBM::default()),
        "linear_regression" => Box::new(LinearRegression::default()),
        "linear_svm" => Box::new(LinearSVM::default()),
        "logistic_regression" => Box::new(LogisticRegression::default()),
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
                 ensembles like bagging/stacking/dynamic_ensemble need a base-learner \
                 factory, and geo_xgboost needs training coordinates)"
            )));
        }
    })
}

/// All learner ids constructible via [`learner_from_id`].
pub fn registered_learner_ids() -> &'static [&'static str] {
    &[
        "adaboost",
        "catboost",
        "decision_tree",
        "elastic_net",
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
        for id in ["bagging", "stacking", "dynamic_ensemble", "geo_xgboost"] {
            assert!(
                learner_from_id(id).is_err(),
                "{id} should not be registry-constructible"
            );
        }
    }
}
