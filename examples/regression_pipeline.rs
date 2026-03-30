//! Regression pipeline: preprocessing + learner + cross-validation.
//!
//! Run with: cargo run --example regression_pipeline

use ndarray::array;
use smelt_ml::prelude::*;

fn main() {
    // Dataset: y = 2*x1 + 0.5*x2 + noise
    let features = array![
        [1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0],
        [6.0, 60.0], [7.0, 70.0], [8.0, 80.0], [9.0, 90.0], [10.0, 100.0],
    ];
    let target = vec![12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 84.0, 96.0, 108.0, 120.0];
    let task = RegressionTask::new("pipeline_demo", features, target).unwrap();

    // Pipeline: StandardScaler -> XGBoost
    let mut pipe = Pipeline::new(
        vec![Box::new(StandardScaler::new())],
        Box::new(XGBoost::new().with_n_estimators(50).with_max_depth(3)),
    );

    // Evaluate with holdout
    let ho = Holdout::new(0.8).with_seed(42);
    let result = benchmark::resample_regress(&mut pipe, &task, &ho, &[&Rmse, &Mae, &RSquared]).unwrap();

    println!("Regression Pipeline Results (StandardScaler + XGBoost)");
    println!("------------------------------------------------------");
    println!("RMSE:      {:.4}", result.mean_scores()[0]);
    println!("MAE:       {:.4}", result.mean_scores()[1]);
    println!("R-squared: {:.4}", result.mean_scores()[2]);

    // Compare: bare XGBoost without scaling
    let mut bare = XGBoost::new().with_n_estimators(50).with_max_depth(3);
    let bare_result = benchmark::resample_regress(&mut bare, &task, &ho, &[&Rmse]).unwrap();
    println!("\nBare XGBoost RMSE: {:.4}", bare_result.mean_scores()[0]);

    // Regularized regression comparison
    println!("\n--- Regularized Regression ---");
    let regs: Vec<(&str, Box<dyn Learner>)> = vec![
        ("Ridge(0.1)", Box::new(Ridge::new(0.1))),
        ("Lasso(0.01)", Box::new(Lasso::new(0.01))),
        ("ElasticNet", Box::new(ElasticNet::new(0.01, 0.5))),
    ];

    for (name, mut learner) in regs {
        let r = benchmark::resample_regress(&mut *learner, &task, &ho, &[&Rmse, &RSquared]).unwrap();
        println!("{name:<15} RMSE: {:.4}, R²: {:.4}", r.mean_scores()[0], r.mean_scores()[1]);
    }
}
