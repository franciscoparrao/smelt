//! Basic classification example: train, predict, evaluate with cross-validation.
//!
//! Run with: cargo run --example basic_classification

use ndarray::array;
use smelt_ml::prelude::*;

fn main() {
    // Create a simple binary classification dataset
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
    let task = ClassificationTask::new("demo", features, target).unwrap();

    println!(
        "Dataset: {} samples, {} features, {} classes\n",
        task.n_samples(),
        task.n_features(),
        task.n_classes()
    );

    // Compare multiple learners with 4-fold cross-validation
    let cv = CrossValidation::new(4).with_seed(42);

    let learners: Vec<(&str, Box<dyn Learner>)> = vec![
        ("DecisionTree", Box::new(DecisionTree::default())),
        (
            "RandomForest",
            Box::new(RandomForest::new().with_n_estimators(50).with_seed(42)),
        ),
        ("XGBoost", Box::new(XGBoost::new().with_n_estimators(50))),
        ("KNN(3)", Box::new(KNearestNeighbors::new(3))),
        ("GaussianNB", Box::new(GaussianNB::new())),
        ("LogisticReg", Box::new(LogisticRegression::new())),
    ];

    println!(
        "{:<15} {:>10} {:>10} {:>10}",
        "Learner", "Accuracy", "F1", "AUC-ROC"
    );
    println!("{}", "-".repeat(48));

    for (name, mut learner) in learners {
        let result =
            benchmark::resample_classif(&mut *learner, &task, &cv, &[&Accuracy, &F1Score, &AucRoc])
                .unwrap();
        let means = result.mean_scores();
        println!(
            "{name:<15} {:>10.4} {:>10.4} {:>10.4}",
            means[0], means[1], means[2]
        );
    }
}
