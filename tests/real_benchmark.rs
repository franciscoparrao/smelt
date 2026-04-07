//! Benchmark smelt-ml on real datasets: Iris, Wine, Breast Cancer.
//! Run with: cargo test --test real_benchmark --release -- --nocapture

use ndarray::Array2;
use smelt_ml::prelude::*;

fn load_dataset(name: &str) -> (Array2<f64>, Vec<usize>) {
    let x_str = std::fs::read_to_string(format!("/tmp/bench_{name}_X.csv"))
        .unwrap_or_else(|_| panic!("Run tests/real_benchmark.py first"));
    let y_str = std::fs::read_to_string(format!("/tmp/bench_{name}_y.csv")).unwrap();

    let rows: Vec<Vec<f64>> = x_str
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.split(',').map(|v| v.trim().parse().unwrap()).collect())
        .collect();
    let ns = rows.len();
    let nf = rows[0].len();
    let mut features = Array2::zeros((ns, nf));
    for (i, row) in rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            features[[i, j]] = v;
        }
    }
    let target: Vec<usize> = y_str
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.trim().parse::<f64>().unwrap() as usize)
        .collect();
    (features, target)
}

fn cv_accuracy(learner: &mut dyn Learner, task: &ClassificationTask) -> f64 {
    let cv = CrossValidation::new(5).with_seed(42);
    let result = benchmark::resample_classif(learner, task, &cv, &[&Accuracy]).unwrap();
    result.mean_scores()[0]
}

#[test]
#[ignore] // requires Python-generated CSV files
fn benchmark_real_datasets() {
    let datasets = ["iris", "wine", "breast_cancer"];

    // Load sklearn reference results
    let ref_json = std::fs::read_to_string("/tmp/sklearn_benchmark.json")
        .unwrap_or_else(|_| panic!("Run tests/real_benchmark.py first"));
    let sklearn_results: serde_json::Value = serde_json::from_str(&ref_json).unwrap();

    for ds_name in &datasets {
        let (features, target) = load_dataset(ds_name);
        let task = ClassificationTask::new(*ds_name, features, target).unwrap();

        println!(
            "\n=== {} ({} samples, {} features, {} classes) ===",
            ds_name,
            task.n_samples(),
            task.n_features(),
            task.n_classes()
        );
        println!(
            "{:<25} {:>10} {:>12} {:>8}",
            "Learner", "smelt-ml", "sklearn", "Δ"
        );
        println!("{}", "-".repeat(58));

        let learners: Vec<(&str, Box<dyn Learner>)> = vec![
            ("DecisionTree", Box::new(DecisionTree::default())),
            (
                "RandomForest",
                Box::new(RandomForest::new().with_n_estimators(100).with_seed(42)),
            ),
            (
                "GradientBoosting",
                Box::new(GradientBoosting::new().with_n_estimators(100)),
            ),
            ("KNN(5)", Box::new(KNearestNeighbors::new(5))),
            (
                "LogisticRegression",
                Box::new(LogisticRegression::new().with_max_iter(1000)),
            ),
            ("GaussianNB", Box::new(GaussianNB::new())),
            (
                "XGBoost",
                Box::new(XGBoost::new().with_n_estimators(100).with_seed(42)),
            ),
        ];

        for (name, mut learner) in learners {
            let smelt_acc = cv_accuracy(&mut *learner, &task);

            let sklearn_acc = sklearn_results[ds_name][name]["mean"]
                .as_f64()
                .unwrap_or(0.0);
            let delta = smelt_acc - sklearn_acc;
            let marker = if delta.abs() < 0.02 {
                "≈"
            } else if delta > 0.0 {
                "↑"
            } else {
                "↓"
            };

            println!("{name:<25} {smelt_acc:>10.4} {sklearn_acc:>12.4} {delta:>+8.4} {marker}");
        }
    }
}
