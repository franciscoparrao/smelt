//! XGBoost hyperparameter tuning with Bayesian Optimization (TPE).
//!
//! Run with: cargo run --example xgboost_tuning

use ndarray::array;
use smelt_ml::prelude::*;
use smelt_ml::tuning::{BayesianOptimizer, ParamDistribution, ParamSpace};

fn main() {
    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [0.0, 0.2],
        [0.1, 0.0],
        [0.2, 0.1],
        [0.0, 0.1],
        [0.1, 0.2],
        [1.0, 1.0],
        [1.1, 0.9],
        [0.9, 1.1],
        [1.0, 0.9],
        [1.1, 1.0],
        [0.9, 1.0],
        [1.0, 1.1],
        [1.1, 1.1],
    ];
    let target = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
    let task = ClassificationTask::new("tuning_demo", features, target).unwrap();
    let cv = CrossValidation::new(4).with_seed(42);

    // Define hyperparameter search space
    let mut space = ParamSpace::new();
    space.insert("max_depth".into(), ParamDistribution::Uniform(2.0, 8.0));
    space.insert(
        "learning_rate".into(),
        ParamDistribution::LogUniform(0.01, 0.5),
    );
    space.insert(
        "n_estimators".into(),
        ParamDistribution::Choice(vec![25.0.into(), 50.0.into(), 100.0.into()]),
    );

    // Bayesian Optimization (TPE) — 20 evaluations
    println!("=== Bayesian Optimization (TPE) ===");
    let bo = BayesianOptimizer::new(
        |params| {
            Box::new(
                XGBoost::new()
                    .with_max_depth(params["max_depth"].as_usize().unwrap())
                    .with_learning_rate(params["learning_rate"].as_f64().unwrap())
                    .with_n_estimators(params["n_estimators"].as_usize().unwrap()),
            )
        },
        space.clone(),
    )
    .with_n_iter(20)
    .with_seed(42);

    let bo_result = bo.tune_classif(&task, &cv, &Accuracy).unwrap();
    println!("Best score: {:.4}", bo_result.best_score);
    println!("Best params:");
    for (k, v) in &bo_result.best_params {
        println!("  {k}: {:.4}", v.as_f64().unwrap());
    }

    // Compare with Random Search — same budget
    println!("\n=== Random Search (same budget) ===");
    let rs = smelt_ml::tuning::RandomSearch::new(
        |params| {
            Box::new(
                XGBoost::new()
                    .with_max_depth(params["max_depth"].as_usize().unwrap())
                    .with_learning_rate(params["learning_rate"].as_f64().unwrap())
                    .with_n_estimators(params["n_estimators"].as_usize().unwrap()),
            )
        },
        space,
    )
    .with_n_iter(20)
    .with_seed(42);

    let rs_result = rs.tune_classif(&task, &cv, &Accuracy).unwrap();
    println!("Best score: {:.4}", rs_result.best_score);

    println!("\n=== Summary ===");
    println!(
        "Bayesian: {:.4}  |  Random: {:.4}",
        bo_result.best_score, rs_result.best_score
    );
}
