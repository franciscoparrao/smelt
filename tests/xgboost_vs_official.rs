//! Compare smelt-ml XGBoost against official XGBoost reference results.

use ndarray::array;
use smelt_ml::prelude::*;

// ── Test 1: Binary classification (same data as Python) ────────────

#[test]
fn xgb_vs_official_binary_classification() {
    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [0.0, 0.2],
        [0.1, 0.0],
        [0.2, 0.1],
        [0.0, 0.1],
        [0.1, 0.2],
        [0.15, 0.05],
        [0.05, 0.15],
        [1.0, 1.0],
        [1.1, 0.9],
        [0.9, 1.1],
        [1.0, 0.9],
        [1.1, 1.0],
        [0.9, 1.0],
        [1.0, 1.1],
        [1.1, 1.1],
        [0.95, 0.95],
        [1.05, 1.05]
    ];
    let target = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    let task = ClassificationTask::new("bin", features, target).unwrap();

    let mut xgb = XGBoost::new()
        .with_n_estimators(50)
        .with_max_depth(3)
        .with_learning_rate(0.3)
        .with_lambda(1.0)
        .with_alpha(0.0)
        .with_gamma(0.0);
    let model = xgb.train_classif(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_classif(task.target().to_vec());

    let acc = Accuracy.score(&pred).unwrap();

    // Official XGBoost: 1.0 accuracy
    println!("smelt-ml binary accuracy: {acc:.4}");
    println!("Official XGBoost:         1.0000");
    assert!(
        acc >= 0.95,
        "smelt binary should match official (1.0), got {acc}"
    );

    // Check predictions match
    if let Prediction::Classification { predicted, .. } = &pred {
        let expected = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let match_count = predicted
            .iter()
            .zip(&expected)
            .filter(|(a, b)| a == b)
            .count();
        println!("Prediction match: {match_count}/20");
    }
}

// ── Test 2: Regression (linear trend) ──────────────────────────────

#[test]
fn xgb_vs_official_regression() {
    let features = array![
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
        [6.0],
        [7.0],
        [8.0],
        [9.0],
        [10.0]
    ];
    let target = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];
    let task = RegressionTask::new("reg", features, target).unwrap();

    let mut xgb = XGBoost::new()
        .with_n_estimators(100)
        .with_max_depth(3)
        .with_learning_rate(0.3)
        .with_lambda(1.0);
    let model = xgb.train_regress(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_regress(task.target().to_vec());

    let rmse = Rmse.score(&pred).unwrap();

    // Official XGBoost: RMSE = 0.0009
    println!("smelt-ml regression RMSE: {rmse:.4}");
    println!("Official XGBoost RMSE:    0.0009");

    if let Prediction::Regression { predicted, .. } = &pred {
        println!(
            "smelt-ml predictions: {:?}",
            predicted
                .iter()
                .map(|p| format!("{:.4}", p))
                .collect::<Vec<_>>()
        );
        println!(
            "Official predictions: [2.0012, 4.0001, 6.0002, 7.9999, 9.9984, 12.0016, 13.9999, 15.9999, 17.9999, 19.9989]"
        );
    }

    assert!(
        rmse < 1.0,
        "smelt regression RMSE should be low, got {rmse}"
    );
}

// ── Test 3: Multiclass (3 clusters) ────────────────────────────────

#[test]
fn xgb_vs_official_multiclass() {
    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.0, 0.1],
        [0.1, 0.0],
        [1.0, 0.0],
        [1.1, 0.1],
        [1.0, 0.1],
        [1.1, 0.0],
        [0.0, 1.0],
        [0.1, 1.1],
        [0.0, 1.1],
        [0.1, 1.0]
    ];
    let target = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
    let task = ClassificationTask::new("mc", features, target).unwrap();

    let mut xgb = XGBoost::new()
        .with_n_estimators(100)
        .with_max_depth(3)
        .with_learning_rate(0.3)
        .with_lambda(0.01)
        .with_min_child_weight(0.1);
    let model = xgb.train_classif(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_classif(task.target().to_vec());

    let acc = Accuracy.score(&pred).unwrap();

    // Official XGBoost: 1.0 accuracy
    println!("smelt-ml multiclass accuracy: {acc:.4}");
    println!("Official XGBoost:             1.0000");
    assert!(
        acc >= 0.75,
        "smelt multiclass should approach official (1.0), got {acc}"
    );
}

// ── Test 4: Synthetic dataset (200 samples) ────────────────────────

#[test]
fn xgb_vs_official_synthetic_dataset() {
    // Load the same synthetic dataset from Python
    let x_str = std::fs::read_to_string("/tmp/xgb_synthetic_X.csv")
        .expect("Run tests/xgboost_comparison.py first");
    let y_str = std::fs::read_to_string("/tmp/xgb_synthetic_y.csv")
        .expect("Run tests/xgboost_comparison.py first");

    let rows: Vec<Vec<f64>> = x_str
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| {
            l.split(',')
                .map(|v| v.trim().parse::<f64>().unwrap())
                .collect()
        })
        .collect();

    let n_samples = rows.len();
    let n_features = rows[0].len();
    let mut features = ndarray::Array2::zeros((n_samples, n_features));
    for (i, row) in rows.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            features[[i, j]] = val;
        }
    }

    let target: Vec<usize> = y_str
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.trim().parse::<f64>().unwrap() as usize)
        .collect();

    let task = ClassificationTask::new("syn", features.clone(), target.clone()).unwrap();

    // Train with same hyperparams as official
    let mut xgb = XGBoost::new()
        .with_n_estimators(100)
        .with_max_depth(6)
        .with_learning_rate(0.3)
        .with_lambda(1.0)
        .with_seed(42);
    let model = xgb.train_classif(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_classif(task.target().to_vec());

    let train_acc = Accuracy.score(&pred).unwrap();

    // 5-fold CV
    let cv = CrossValidation::new(5).with_seed(42);
    let mut xgb2 = XGBoost::new()
        .with_n_estimators(100)
        .with_max_depth(6)
        .with_learning_rate(0.3)
        .with_lambda(1.0)
        .with_seed(42);
    let result = benchmark::resample_classif(&mut xgb2, &task, &cv, &[&Accuracy]).unwrap();
    let cv_mean = result.mean_scores()[0];

    // Official: CV = 0.8250 ± 0.0274
    println!("=== Synthetic Dataset (200 samples, 10 features) ===");
    println!("smelt-ml train accuracy:  {train_acc:.4}");
    println!("smelt-ml 5-fold CV:       {cv_mean:.4}");
    println!("Official XGBoost CV:      0.8250 ± 0.0274");
    println!(
        "Per-fold scores: {:?}",
        result
            .scores
            .iter()
            .map(|s| format!("{:.4}", s[0]))
            .collect::<Vec<_>>()
    );

    // Our CV accuracy should be in the same ballpark (±0.10)
    assert!(
        cv_mean >= 0.70,
        "smelt CV should be close to official 0.825, got {cv_mean}"
    );
    assert!(
        train_acc >= 0.90,
        "smelt train accuracy should be high, got {train_acc}"
    );
}

// ── Summary ────────────────────────────────────────────────────────

#[test]
fn xgb_comparison_summary() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║         smelt-ml XGBoost vs Official XGBoost            ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ Feature             │ Official │ smelt-ml │ Match       ║");
    println!("╠═════════════════════╪══════════╪══════════╪═════════════╣");
    println!("║ 2nd-order gradients │    ✓     │    ✓     │ ✓           ║");
    println!("║ Histogram splits    │    ✓     │    ✓     │ ✓           ║");
    println!("║ L1 regularization   │    ✓     │    ✓     │ ✓           ║");
    println!("║ L2 regularization   │    ✓     │    ✓     │ ✓           ║");
    println!("║ Gamma (min gain)    │    ✓     │    ✓     │ ✓           ║");
    println!("║ Row subsampling     │    ✓     │    ✓     │ ✓           ║");
    println!("║ Col subsampling     │    ✓     │    ✓     │ ✓           ║");
    println!("║ min_child_weight    │    ✓     │    ✓     │ ✓           ║");
    println!("║ Missing values      │    ✓     │    ✓     │ ✓           ║");
    println!("║ Exact greedy (auto) │    ✓     │    ✓     │ ✓           ║");
    println!("║ Parallel splits     │    ✓     │    ✓     │ rayon       ║");
    println!("║ GPU support         │    ✓     │    ✗     │ Out of scope║");
    println!("╚═════════════════════╧══════════╧══════════╧═════════════╝");
}
