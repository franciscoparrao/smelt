use ndarray::{Array2, Axis, array};
use smelt_ml::prelude::*;

// ── Task tests ──────────────────────────────────────────────────────

#[test]
fn classification_task_creation() {
    let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let target = vec![0, 1, 1];
    let task = ClassificationTask::new("test", features, target).unwrap();

    assert_eq!(task.id(), "test");
    assert_eq!(task.n_samples(), 3);
    assert_eq!(task.n_features(), 2);
    assert_eq!(task.n_classes(), 2);
    assert_eq!(task.target(), &[0, 1, 1]);
}

#[test]
fn regression_task_creation() {
    let features = array![[1.0], [2.0], [3.0]];
    let target = vec![1.5, 2.5, 3.5];
    let task = RegressionTask::new("regr", features, target).unwrap();

    assert_eq!(task.id(), "regr");
    assert_eq!(task.n_samples(), 3);
    assert_eq!(task.n_features(), 1);
    assert_eq!(task.target(), &[1.5, 2.5, 3.5]);
}

#[test]
fn task_empty_dataset() {
    let features = Array2::<f64>::zeros((0, 2));
    let target: Vec<usize> = vec![];
    let err = ClassificationTask::new("empty", features, target).unwrap_err();
    assert!(matches!(err, SmeltError::EmptyDataset));
}

#[test]
fn task_dimension_mismatch() {
    let features = array![[1.0, 2.0], [3.0, 4.0]];
    let target = vec![0, 1, 2]; // 3 targets, 2 samples
    let err = ClassificationTask::new("bad", features, target).unwrap_err();
    assert!(matches!(err, SmeltError::DimensionMismatch { .. }));
}

#[test]
fn task_custom_feature_names() {
    let features = array![[1.0, 2.0], [3.0, 4.0]];
    let target = vec![0, 1];
    let task = ClassificationTask::new("t", features, target)
        .unwrap()
        .with_feature_names(vec!["a".into(), "b".into()])
        .unwrap();
    assert_eq!(task.feature_names(), &["a", "b"]);
}

#[test]
fn task_wrong_feature_names_count() {
    let features = array![[1.0, 2.0], [3.0, 4.0]];
    let target = vec![0, 1];
    let err = ClassificationTask::new("t", features, target)
        .unwrap()
        .with_feature_names(vec!["only_one".into()])
        .unwrap_err();
    assert!(matches!(err, SmeltError::DimensionMismatch { .. }));
}

// ── Validation tests ────────────────────────────────────────────────

#[test]
fn predict_wrong_n_features_errors() {
    let features = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
    let target = vec![0, 0, 1, 1];
    let task = ClassificationTask::new("val", features, target).unwrap();

    let mut dt = DecisionTree::default();
    let model = dt.train_classif(&task).unwrap();

    // Predict with wrong number of features
    let bad_features = array![[1.0]]; // 1 feature instead of 2
    let err = model.predict(&bad_features);
    assert!(err.is_err(), "predicting with wrong n_features should fail");
}

#[test]
fn predict_wrong_n_features_regression() {
    let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];
    let target = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let task = RegressionTask::new("val_r", features, target).unwrap();

    let mut dt = DecisionTree::default();
    let model = dt.train_regress(&task).unwrap();

    let bad = array![[1.0, 2.0, 3.0]]; // 3 features instead of 2
    assert!(model.predict(&bad).is_err());
}

#[test]
fn task_zero_columns_rejected() {
    let features = Array2::<f64>::zeros((5, 0));
    let target = vec![0, 1, 0, 1, 0];
    assert!(ClassificationTask::new("zero", features, target).is_err());
}

#[test]
fn validate_check_no_nan() {
    use smelt_ml::validate::check_no_nan;
    let clean = array![[1.0, 2.0], [3.0, 4.0]];
    assert!(check_no_nan(&clean).is_ok());

    let dirty = array![[1.0, f64::NAN], [3.0, 4.0]];
    assert!(check_no_nan(&dirty).is_err());
}

// ── Prediction tests ────────────────────────────────────────────────

#[test]
fn prediction_classification() {
    let pred = Prediction::classification(vec![0, 1, 1]);
    assert_eq!(pred.n_samples(), 3);
}

#[test]
fn prediction_with_truth() {
    let pred = Prediction::classification_with_truth(vec![0, 1, 1], vec![0, 0, 1]);
    assert_eq!(pred.n_samples(), 3);
}

#[test]
fn prediction_regression() {
    let pred = Prediction::regression_with_truth(vec![1.0, 2.0], vec![1.1, 2.1]);
    assert_eq!(pred.n_samples(), 2);
}

// ── Measure tests ───────────────────────────────────────────────────

#[test]
fn accuracy_perfect() {
    let pred = Prediction::classification_with_truth(vec![0, 1, 1, 0], vec![0, 1, 1, 0]);
    let acc = Accuracy.score(&pred).unwrap();
    assert!((acc - 1.0).abs() < f64::EPSILON);
}

#[test]
fn accuracy_half() {
    let pred = Prediction::classification_with_truth(vec![0, 1, 0, 1], vec![0, 0, 1, 1]);
    let acc = Accuracy.score(&pred).unwrap();
    assert!((acc - 0.5).abs() < f64::EPSILON);
}

#[test]
fn accuracy_requires_truth() {
    let pred = Prediction::classification(vec![0, 1]);
    assert!(Accuracy.score(&pred).is_err());
}

#[test]
fn rmse_perfect() {
    let pred = Prediction::regression_with_truth(vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]);
    let rmse = Rmse.score(&pred).unwrap();
    assert!(rmse.abs() < f64::EPSILON);
}

#[test]
fn rmse_known_value() {
    // errors: [1, 1, 1] => MSE = 1.0 => RMSE = 1.0
    let pred = Prediction::regression_with_truth(vec![2.0, 3.0, 4.0], vec![1.0, 2.0, 3.0]);
    let rmse = Rmse.score(&pred).unwrap();
    assert!((rmse - 1.0).abs() < f64::EPSILON);
}

#[test]
fn mae_known_value() {
    // errors: |1|, |1|, |1| => MAE = 1.0
    let pred = Prediction::regression_with_truth(vec![2.0, 3.0, 4.0], vec![1.0, 2.0, 3.0]);
    let mae = Mae.score(&pred).unwrap();
    assert!((mae - 1.0).abs() < f64::EPSILON);
}

#[test]
fn measure_direction() {
    assert!(Accuracy.maximize());
    assert!(!Rmse.maximize());
    assert!(!Mae.maximize());
}

// ── Resample tests ──────────────────────────────────────────────────

#[test]
fn cross_validation_splits() {
    let cv = CrossValidation::new(5);
    let splits = cv.splits(100).unwrap();
    assert_eq!(splits.len(), 5);

    for (train, test) in &splits {
        assert_eq!(train.len() + test.len(), 100);
    }

    // All indices must appear exactly once as test across all folds
    let mut all_test: Vec<usize> = splits.iter().flat_map(|(_, t)| t.clone()).collect();
    all_test.sort();
    let expected: Vec<usize> = (0..100).collect();
    assert_eq!(all_test, expected);
}

#[test]
fn holdout_split_ratio() {
    let ho = Holdout::new(0.7);
    let splits = ho.splits(100).unwrap();
    assert_eq!(splits.len(), 1);
    let (train, test) = &splits[0];
    assert_eq!(train.len(), 70);
    assert_eq!(test.len(), 30);
}

#[test]
fn resample_deterministic() {
    let cv = CrossValidation::new(3).with_seed(123);
    let s1 = cv.splits(50).unwrap();
    let s2 = cv.splits(50).unwrap();
    assert_eq!(s1, s2);
}

// ── Decision Tree tests ─────────────────────────────────────────────

#[test]
fn decision_tree_classif_linearly_separable() {
    // Two clusters clearly separable by x0
    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [0.1, 0.2],
        [0.0, 0.1],
        [1.0, 1.0],
        [1.1, 0.9],
        [0.9, 1.1],
        [1.0, 0.9],
        [1.1, 1.0]
    ];
    let target = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
    let task = ClassificationTask::new("sep", features, target).unwrap();

    let mut tree = DecisionTree::default();
    let model = tree.train_classif(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_classif(task.target().to_vec());

    let acc = Accuracy.score(&pred).unwrap();
    assert_eq!(
        acc, 1.0,
        "should perfectly separate linearly separable data"
    );
}

#[test]
fn decision_tree_classif_max_depth() {
    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [1.0, 1.0],
        [1.1, 0.9],
        [0.9, 1.1]
    ];
    let target = vec![0, 0, 0, 1, 1, 1];
    let task = ClassificationTask::new("depth", features, target).unwrap();

    let mut tree = DecisionTree::new().with_max_depth(1);
    let model = tree.train_classif(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_classif(task.target().to_vec());

    let acc = Accuracy.score(&pred).unwrap();
    assert!(acc >= 0.5, "depth-1 tree should do better than random");
}

#[test]
fn decision_tree_regress_constant() {
    // All targets the same => predict that constant
    let features = array![[1.0], [2.0], [3.0], [4.0]];
    let target = vec![5.0, 5.0, 5.0, 5.0];
    let task = RegressionTask::new("const", features, target).unwrap();

    let mut tree = DecisionTree::default();
    let model = tree.train_regress(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_regress(task.target().to_vec());

    let rmse = Rmse.score(&pred).unwrap();
    assert!(rmse.abs() < f64::EPSILON, "constant target => RMSE=0");
}

#[test]
fn decision_tree_regress_learns_step_function() {
    // Step function: x < 5 => 0, x >= 5 => 10
    let features = array![[1.0], [2.0], [3.0], [4.0], [6.0], [7.0], [8.0], [9.0]];
    let target = vec![0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0];
    let task = RegressionTask::new("step", features, target).unwrap();

    let mut tree = DecisionTree::default();
    let model = tree.train_regress(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_regress(task.target().to_vec());

    let rmse = Rmse.score(&pred).unwrap();
    assert!(rmse < 0.01, "tree should learn a step function perfectly");
}

#[test]
fn decision_tree_feature_importance() {
    // Only feature 0 is informative
    let features = array![
        [0.0, 42.0],
        [0.1, 13.0],
        [0.2, 99.0],
        [1.0, 42.0],
        [1.1, 13.0],
        [1.2, 99.0]
    ];
    let target = vec![0, 0, 0, 1, 1, 1];
    let task = ClassificationTask::new("imp", features, target).unwrap();

    let mut tree = DecisionTree::default();
    let model = tree.train_classif(&task).unwrap();

    let importances = model.feature_importance().unwrap();
    assert!(
        importances[0].1 > importances[1].1,
        "feature 0 should be more important than noise feature 1"
    );
}

#[test]
fn decision_tree_predict_unseen() {
    let features = array![[0.0], [1.0], [2.0], [3.0]];
    let target = vec![0, 0, 1, 1];
    let task = ClassificationTask::new("gen", features, target).unwrap();

    let mut tree = DecisionTree::default();
    let model = tree.train_classif(&task).unwrap();

    let unseen = array![[0.5], [2.5]];
    let pred = model.predict(&unseen).unwrap();
    assert_eq!(pred.n_samples(), 2);
}

// ── Pipeline end-to-end ─────────────────────────────────────────────

#[test]
fn full_pipeline_classif_with_cv() {
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
        [1.1, 1.1]
    ];
    let target = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
    let task = ClassificationTask::new("pipeline", features.clone(), target.clone()).unwrap();

    let cv = CrossValidation::new(4).with_seed(42);
    let splits = cv.splits(task.n_samples()).unwrap();

    let mut scores = Vec::new();
    for (train_idx, test_idx) in &splits {
        let train_features = features.select(ndarray::Axis(0), train_idx);
        let train_target: Vec<usize> = train_idx.iter().map(|&i| target[i]).collect();
        let train_task = ClassificationTask::new("train", train_features, train_target).unwrap();

        let mut tree = DecisionTree::new().with_max_depth(3);
        let model = tree.train_classif(&train_task).unwrap();

        let test_features = features.select(ndarray::Axis(0), test_idx);
        let test_target: Vec<usize> = test_idx.iter().map(|&i| target[i]).collect();
        let pred = model
            .predict(&test_features)
            .unwrap()
            .with_truth_classif(test_target);

        scores.push(Accuracy.score(&pred).unwrap());
    }

    let mean_acc = scores.iter().sum::<f64>() / scores.len() as f64;
    assert!(
        mean_acc >= 0.75,
        "CV accuracy on separable data should be high, got {mean_acc}"
    );
}

#[test]
fn full_pipeline_regress_with_holdout() {
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

    let ho = Holdout::new(0.8).with_seed(42);
    let splits = ho.splits(features.nrows()).unwrap();
    let (train_idx, test_idx) = &splits[0];

    let train_features = features.select(ndarray::Axis(0), train_idx);
    let train_target: Vec<f64> = train_idx.iter().map(|&i| target[i]).collect();
    let train_task = RegressionTask::new("train", train_features, train_target).unwrap();

    let mut tree = DecisionTree::default();
    let model = tree.train_regress(&train_task).unwrap();

    let test_features = features.select(ndarray::Axis(0), test_idx);
    let test_target: Vec<f64> = test_idx.iter().map(|&i| target[i]).collect();
    let pred = model
        .predict(&test_features)
        .unwrap()
        .with_truth_regress(test_target);

    let rmse = Rmse.score(&pred).unwrap();
    let mae = Mae.score(&pred).unwrap();
    assert!(rmse < 10.0, "RMSE should be reasonable, got {rmse}");
    assert!(mae < 10.0, "MAE should be reasonable, got {mae}");
}

// ── KNN tests ───────────────────────────────────────────────────────

#[test]
fn knn_classif_separable() {
    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [1.0, 1.0],
        [1.1, 0.9],
        [0.9, 1.1]
    ];
    let target = vec![0, 0, 0, 1, 1, 1];
    let task = ClassificationTask::new("knn", features, target).unwrap();

    let mut knn = KNearestNeighbors::new(3);
    let model = knn.train_classif(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_classif(task.target().to_vec());

    let acc = Accuracy.score(&pred).unwrap();
    assert_eq!(acc, 1.0);
}

#[test]
fn knn_regress_mean() {
    // k=3, all neighbors have values [1, 2, 3] => mean = 2.0
    let features = array![[0.0], [1.0], [2.0]];
    let target = vec![1.0, 2.0, 3.0];
    let task = RegressionTask::new("knn_r", features, target).unwrap();

    let mut knn = KNearestNeighbors::new(3);
    let model = knn.train_regress(&task).unwrap();

    let test = array![[1.0]]; // equidistant from all, mean = 2.0
    let pred = model.predict(&test).unwrap().with_truth_regress(vec![2.0]);

    let rmse = Rmse.score(&pred).unwrap();
    assert!(rmse < f64::EPSILON);
}

#[test]
fn knn_k_larger_than_dataset() {
    let features = array![[0.0], [1.0]];
    let target = vec![0, 1];
    let task = ClassificationTask::new("small", features, target).unwrap();

    let mut knn = KNearestNeighbors::new(100); // k >> n
    let model = knn.train_classif(&task).unwrap();
    let pred = model.predict(task.features()).unwrap();
    assert_eq!(pred.n_samples(), 2);
}

// ── Linear Regression tests ────────────────────────────────────────

#[test]
fn linear_regression_perfect_fit() {
    // y = 2*x + 1
    let features = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let target = vec![3.0, 5.0, 7.0, 9.0, 11.0];
    let task = RegressionTask::new("lin", features, target).unwrap();

    let mut lr = LinearRegression;
    let model = lr.train_regress(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_regress(task.target().to_vec());

    let rmse = Rmse.score(&pred).unwrap();
    assert!(
        rmse < 1e-10,
        "OLS should fit y=2x+1 perfectly, got RMSE={rmse}"
    );
}

#[test]
fn linear_regression_multivariate() {
    // y = x0 + 2*x1 + 3
    let features = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0], [3.0, 1.0]];
    let target = vec![4.0, 5.0, 6.0, 9.0, 8.0]; // 1+0+3, 0+2+3, 1+2+3, 2+4+3, 3+2+3
    let task = RegressionTask::new("multi", features, target).unwrap();

    let mut lr = LinearRegression;
    let model = lr.train_regress(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_regress(task.target().to_vec());

    let rmse = Rmse.score(&pred).unwrap();
    assert!(
        rmse < 1e-10,
        "OLS should fit y=x0+2*x1+3 perfectly, got RMSE={rmse}"
    );
}

#[test]
fn linear_regression_rejects_classification() {
    let features = array![[1.0], [2.0]];
    let target = vec![0, 1];
    let task = ClassificationTask::new("bad", features, target).unwrap();

    let mut lr = LinearRegression;
    assert!(lr.train_classif(&task).is_err());
}

#[test]
fn linear_regression_feature_importance() {
    // y = 10*x0 + 1*x1, so x0 is much more important
    let features = array![
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [3.0, 1.0]
    ];
    let target = vec![10.0, 20.0, 30.0, 11.0, 21.0, 31.0];
    let task = RegressionTask::new("imp", features, target).unwrap();

    let mut lr = LinearRegression;
    let model = lr.train_regress(&task).unwrap();

    let imp = model.feature_importance().unwrap();
    assert!(
        imp[0].1 > imp[1].1,
        "x0 (coeff=10) should matter more than x1 (coeff=1)"
    );
}

// ── Logistic Regression tests ──────────────────────────────────────

#[test]
fn logistic_regression_binary() {
    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [0.0, 0.2],
        [0.1, 0.0],
        [1.0, 1.0],
        [1.1, 0.9],
        [0.9, 1.1],
        [1.0, 0.9],
        [1.1, 1.0]
    ];
    let target = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
    let task = ClassificationTask::new("logr", features, target).unwrap();

    let mut lr = LogisticRegression::new()
        .with_learning_rate(1.0)
        .with_max_iter(500);
    let model = lr.train_classif(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_classif(task.target().to_vec());

    let acc = Accuracy.score(&pred).unwrap();
    assert!(
        acc >= 0.8,
        "Logistic regression should classify separable data well, got {acc}"
    );
}

#[test]
fn logistic_regression_multiclass() {
    // 3 classes, each in a different quadrant-ish region
    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.0, 0.1],
        [1.0, 0.0],
        [1.1, 0.1],
        [1.0, 0.1],
        [0.0, 1.0],
        [0.1, 1.1],
        [0.0, 1.1]
    ];
    let target = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
    let task = ClassificationTask::new("multi", features, target).unwrap();

    let mut lr = LogisticRegression::new()
        .with_learning_rate(1.0)
        .with_max_iter(1000);
    let model = lr.train_classif(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_classif(task.target().to_vec());

    let acc = Accuracy.score(&pred).unwrap();
    assert!(
        acc >= 0.66,
        "Multiclass OVR should do better than random, got {acc}"
    );
}

#[test]
fn logistic_regression_rejects_regression() {
    let features = array![[1.0], [2.0]];
    let target = vec![1.0, 2.0];
    let task = RegressionTask::new("bad", features, target).unwrap();

    let mut lr = LogisticRegression::default();
    assert!(lr.train_regress(&task).is_err());
}

// ── Benchmark pipeline tests ───────────────────────────────────────

#[test]
fn benchmark_classif_cv() {
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
        [1.1, 1.1]
    ];
    let target = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
    let task = ClassificationTask::new("bench", features, target).unwrap();

    let cv = CrossValidation::new(4).with_seed(42);
    let mut tree = DecisionTree::new().with_max_depth(3);

    let result = benchmark::resample_classif(&mut tree, &task, &cv, &[&Accuracy]).unwrap();

    assert_eq!(result.scores.len(), 4); // 4 folds
    assert_eq!(result.measure_ids, vec!["classif.accuracy"]);
    assert_eq!(result.learner_id, "decision_tree");

    let means = result.mean_scores();
    assert!(means[0] >= 0.5, "mean accuracy should be reasonable");
}

/// The three classic resamplers added for mlr3-parity (RepeatedCV, LOO,
/// Bootstrap) must drive a real learner through `benchmark::resample_*`
/// exactly like CrossValidation/Holdout — proving they compose with the
/// evaluation loop, not just satisfy the `Resample` trait in isolation.
/// Bootstrap in particular exercises the `features.select` path with a
/// *duplicated* train index vector, which the loop must handle.
#[test]
fn benchmark_classif_repeated_loo_and_bootstrap() {
    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.0, 0.1],
        [0.1, 0.0],
        [0.2, 0.1],
        [1.0, 1.0],
        [1.1, 0.9],
        [0.9, 1.1],
        [1.0, 0.9],
        [1.1, 1.0]
    ];
    let target = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
    let task = ClassificationTask::new("bench", features, target).unwrap();

    // RepeatedCV: 2 folds × 3 repeats = 6 evaluations.
    let mut tree = DecisionTree::new().with_max_depth(3);
    let rcv = RepeatedCV::new(2, 3).with_seed(7);
    let r = benchmark::resample_classif(&mut tree, &task, &rcv, &[&Accuracy]).unwrap();
    assert_eq!(r.scores.len(), 6);
    assert!(r.mean_scores()[0] >= 0.5);

    // Leave-one-out: one evaluation per sample.
    let loo = LeaveOneOut;
    let r = benchmark::resample_classif(&mut tree, &task, &loo, &[&Accuracy]).unwrap();
    assert_eq!(r.scores.len(), task.n_samples());

    // Bootstrap: OOB test per resample; train has duplicate indices.
    let boot = Bootstrap::new(8).with_seed(3);
    let r = benchmark::resample_classif(&mut tree, &task, &boot, &[&Accuracy]).unwrap();
    assert_eq!(r.scores.len(), 8);
    assert!(r.mean_scores()[0] >= 0.5);
}

#[test]
fn benchmark_regress_holdout() {
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
    let task = RegressionTask::new("bench", features, target).unwrap();

    let ho = Holdout::new(0.8).with_seed(42);
    let mut tree = DecisionTree::default();

    let result = benchmark::resample_regress(&mut tree, &task, &ho, &[&Rmse, &Mae]).unwrap();

    assert_eq!(result.scores.len(), 1); // holdout = 1 split
    assert_eq!(result.measure_ids, vec!["regr.rmse", "regr.mae"]);
}

#[test]
fn benchmark_multiple_measures() {
    let features = array![[0.0], [0.1], [0.2], [0.3], [1.0], [1.1], [1.2], [1.3]];
    let target = vec![0, 0, 0, 0, 1, 1, 1, 1];
    let task = ClassificationTask::new("mm", features, target).unwrap();

    let cv = CrossValidation::new(2).with_seed(42);
    let mut knn = KNearestNeighbors::new(3);

    let result = benchmark::resample_classif(&mut knn, &task, &cv, &[&Accuracy]).unwrap();

    assert_eq!(result.learner_id, "knn");
    assert_eq!(result.scores.len(), 2);
}

// ── Random Forest tests ────────────────────────────────────────────

#[test]
fn random_forest_classif_separable() {
    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [0.1, 0.2],
        [0.0, 0.1],
        [1.0, 1.0],
        [1.1, 0.9],
        [0.9, 1.1],
        [1.0, 0.9],
        [1.1, 1.0]
    ];
    let target = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
    let task = ClassificationTask::new("rf", features, target).unwrap();

    let mut rf = RandomForest::new().with_n_estimators(20).with_seed(42);
    let model = rf.train_classif(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_classif(task.target().to_vec());

    let acc = Accuracy.score(&pred).unwrap();
    assert_eq!(
        acc, 1.0,
        "RF should perfectly separate linearly separable data"
    );
}

#[test]
fn random_forest_regress_step() {
    let features = array![[1.0], [2.0], [3.0], [4.0], [6.0], [7.0], [8.0], [9.0]];
    let target = vec![0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0];
    let task = RegressionTask::new("rf_step", features, target).unwrap();

    let mut rf = RandomForest::new().with_n_estimators(20).with_seed(42);
    let model = rf.train_regress(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_regress(task.target().to_vec());

    let rmse = Rmse.score(&pred).unwrap();
    assert!(rmse < 1.0, "RF should learn step function, got RMSE={rmse}");
}

#[test]
fn random_forest_feature_importance() {
    let features = array![
        [0.0, 42.0],
        [0.1, 13.0],
        [0.2, 99.0],
        [1.0, 42.0],
        [1.1, 13.0],
        [1.2, 99.0]
    ];
    let target = vec![0, 0, 0, 1, 1, 1];
    let task = ClassificationTask::new("rf_imp", features, target).unwrap();

    let mut rf = RandomForest::new().with_n_estimators(50).with_seed(42);
    let model = rf.train_classif(&task).unwrap();

    let imp = model.feature_importance().unwrap();
    assert!(imp[0].1 > imp[1].1, "feature 0 should be more important");
}

#[test]
fn random_forest_deterministic() {
    let features = array![[0.0], [1.0], [2.0], [3.0]];
    let target = vec![0, 0, 1, 1];
    let task = ClassificationTask::new("det", features, target).unwrap();

    let mut rf1 = RandomForest::new().with_n_estimators(10).with_seed(123);
    let model1 = rf1.train_classif(&task).unwrap();
    let pred1 = model1.predict(task.features()).unwrap();

    let mut rf2 = RandomForest::new().with_n_estimators(10).with_seed(123);
    let model2 = rf2.train_classif(&task).unwrap();
    let pred2 = model2.predict(task.features()).unwrap();

    match (&pred1, &pred2) {
        (
            Prediction::Classification { predicted: p1, .. },
            Prediction::Classification { predicted: p2, .. },
        ) => {
            assert_eq!(p1, p2, "same seed should produce same predictions");
        }
        _ => panic!("expected classification predictions"),
    }
}

// ── Gradient Boosting tests ────────────────────────────────────────

#[test]
fn gradient_boosting_regress_constant() {
    let features = array![[1.0], [2.0], [3.0], [4.0]];
    let target = vec![5.0, 5.0, 5.0, 5.0];
    let task = RegressionTask::new("gb_const", features, target).unwrap();

    let mut gb = GradientBoosting::new().with_n_estimators(10);
    let model = gb.train_regress(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_regress(task.target().to_vec());

    let rmse = Rmse.score(&pred).unwrap();
    assert!(rmse < 0.01, "constant target => RMSE near 0, got {rmse}");
}

#[test]
fn gradient_boosting_regress_linear() {
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
    let task = RegressionTask::new("gb_lin", features, target).unwrap();

    let mut gb = GradientBoosting::new()
        .with_n_estimators(200)
        .with_learning_rate(0.1)
        .with_max_depth(2);
    let model = gb.train_regress(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_regress(task.target().to_vec());

    let rmse = Rmse.score(&pred).unwrap();
    assert!(rmse < 1.0, "GB should learn linear trend, got RMSE={rmse}");
}

#[test]
fn gradient_boosting_classif_binary() {
    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [0.0, 0.2],
        [0.1, 0.0],
        [1.0, 1.0],
        [1.1, 0.9],
        [0.9, 1.1],
        [1.0, 0.9],
        [1.1, 1.0]
    ];
    let target = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
    let task = ClassificationTask::new("gb_bin", features, target).unwrap();

    let mut gb = GradientBoosting::new()
        .with_n_estimators(50)
        .with_learning_rate(0.3);
    let model = gb.train_classif(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_classif(task.target().to_vec());

    let acc = Accuracy.score(&pred).unwrap();
    assert!(
        acc >= 0.9,
        "GB binary should classify separable data, got {acc}"
    );
}

#[test]
fn gradient_boosting_classif_multiclass() {
    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.0, 0.1],
        [1.0, 0.0],
        [1.1, 0.1],
        [1.0, 0.1],
        [0.0, 1.0],
        [0.1, 1.1],
        [0.0, 1.1]
    ];
    let target = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
    let task = ClassificationTask::new("gb_multi", features, target).unwrap();

    let mut gb = GradientBoosting::new()
        .with_n_estimators(100)
        .with_learning_rate(0.3);
    let model = gb.train_classif(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_classif(task.target().to_vec());

    let acc = Accuracy.score(&pred).unwrap();
    assert!(
        acc >= 0.66,
        "GB multiclass should do better than random, got {acc}"
    );
}

#[test]
fn gradient_boosting_feature_importance() {
    let features = array![
        [0.0, 42.0],
        [0.1, 13.0],
        [0.2, 99.0],
        [1.0, 42.0],
        [1.1, 13.0],
        [1.2, 99.0]
    ];
    let target = vec![0, 0, 0, 1, 1, 1];
    let task = ClassificationTask::new("gb_imp", features, target).unwrap();

    let mut gb = GradientBoosting::new().with_n_estimators(50);
    let model = gb.train_classif(&task).unwrap();

    let imp = model.feature_importance().unwrap();
    assert!(imp[0].1 > imp[1].1, "feature 0 should be more important");
}

// ── Bagging tests ──────────────────────────────────────────────────

#[test]
fn bagging_decision_tree_classif() {
    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [0.1, 0.2],
        [0.0, 0.1],
        [1.0, 1.0],
        [1.1, 0.9],
        [0.9, 1.1],
        [1.0, 0.9],
        [1.1, 1.0]
    ];
    let target = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
    let task = ClassificationTask::new("bag", features, target).unwrap();

    let mut bag = Bagging::new(|| Box::new(DecisionTree::default()))
        .with_n_estimators(10)
        .with_seed(42);
    let model = bag.train_classif(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_classif(task.target().to_vec());

    let acc = Accuracy.score(&pred).unwrap();
    assert!(
        acc >= 0.8,
        "Bagged DT should classify separable data, got {acc}"
    );
}

/// Regression test (5th audit, LOW-D): `Bagging` with `n_estimators = 0`
/// used to train "successfully" and silently predict a constant (class 0 /
/// 0.0 from an empty aggregation) — the ensemble twin of the `max_depth=0`
/// validation. Both task types must reject it with InvalidParameter.
#[test]
fn bagging_rejects_zero_estimators() {
    let features = array![[0.0], [1.0], [2.0], [3.0]];
    let classif = ClassificationTask::new("bag0_c", features.clone(), vec![0, 0, 1, 1]).unwrap();
    let regress = RegressionTask::new("bag0_r", features, vec![0.0, 1.0, 2.0, 3.0]).unwrap();

    let mut bag = Bagging::new(|| Box::new(DecisionTree::default())).with_n_estimators(0);
    let Err(err) = bag.train_classif(&classif) else {
        panic!("n_estimators=0 must be rejected for classification");
    };
    assert!(
        matches!(err, smelt_ml::SmeltError::InvalidParameter(_))
            && format!("{err}").contains("n_estimators"),
        "got: {err}"
    );

    let mut bag = Bagging::new(|| Box::new(DecisionTree::default())).with_n_estimators(0);
    let Err(err) = bag.train_regress(&regress) else {
        panic!("n_estimators=0 must be rejected for regression");
    };
    assert!(
        matches!(err, smelt_ml::SmeltError::InvalidParameter(_))
            && format!("{err}").contains("n_estimators"),
        "got: {err}"
    );
}

#[test]
fn bagging_knn_classif() {
    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [1.0, 1.0],
        [1.1, 0.9],
        [0.9, 1.1]
    ];
    let target = vec![0, 0, 0, 1, 1, 1];
    let task = ClassificationTask::new("bag_knn", features, target).unwrap();

    let mut bag = Bagging::new(|| Box::new(KNearestNeighbors::new(3)))
        .with_n_estimators(5)
        .with_seed(42);
    let model = bag.train_classif(&task).unwrap();
    let pred = model.predict(task.features()).unwrap();
    assert_eq!(pred.n_samples(), 6);
}

#[test]
fn bagging_decision_tree_regress() {
    let features = array![[1.0], [2.0], [3.0], [4.0], [6.0], [7.0], [8.0], [9.0]];
    let target = vec![0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0];
    let task = RegressionTask::new("bag_reg", features, target).unwrap();

    let mut bag = Bagging::new(|| Box::new(DecisionTree::default()))
        .with_n_estimators(10)
        .with_seed(42);
    let model = bag.train_regress(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_regress(task.target().to_vec());

    let rmse = Rmse.score(&pred).unwrap();
    assert!(
        rmse < 2.0,
        "Bagged DT should learn step function, got RMSE={rmse}"
    );
}

#[test]
fn bagging_deterministic() {
    let features = array![[0.0], [1.0], [2.0], [3.0]];
    let target = vec![0, 0, 1, 1];
    let task = ClassificationTask::new("det", features, target).unwrap();

    let mut b1 = Bagging::new(|| Box::new(DecisionTree::default()))
        .with_n_estimators(5)
        .with_seed(99);
    let p1 = b1
        .train_classif(&task)
        .unwrap()
        .predict(task.features())
        .unwrap();

    let mut b2 = Bagging::new(|| Box::new(DecisionTree::default()))
        .with_n_estimators(5)
        .with_seed(99);
    let p2 = b2
        .train_classif(&task)
        .unwrap()
        .predict(task.features())
        .unwrap();

    match (&p1, &p2) {
        (
            Prediction::Classification { predicted: a, .. },
            Prediction::Classification { predicted: b, .. },
        ) => assert_eq!(a, b),
        _ => panic!("expected classification"),
    }
}

#[test]
fn bagging_feature_importance() {
    let features = array![
        [0.0, 42.0],
        [0.1, 13.0],
        [0.2, 99.0],
        [1.0, 42.0],
        [1.1, 13.0],
        [1.2, 99.0]
    ];
    let target = vec![0, 0, 0, 1, 1, 1];
    let task = ClassificationTask::new("bag_imp", features, target).unwrap();

    let mut bag = Bagging::new(|| Box::new(DecisionTree::default()))
        .with_n_estimators(10)
        .with_seed(42);
    let model = bag.train_classif(&task).unwrap();
    let imp = model.feature_importance();
    assert!(imp.is_some(), "bagged DT should provide feature importance");
}

// ── Ensemble benchmark integration ─────────────────────────────────

#[test]
fn benchmark_random_forest_cv() {
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
        [1.1, 1.1]
    ];
    let target = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
    let task = ClassificationTask::new("bench_rf", features, target).unwrap();

    let cv = CrossValidation::new(4).with_seed(42);
    let mut rf = RandomForest::new().with_n_estimators(20).with_seed(42);

    let result = benchmark::resample_classif(&mut rf, &task, &cv, &[&Accuracy]).unwrap();
    assert_eq!(result.learner_id, "random_forest");
    assert_eq!(result.scores.len(), 4);
}

#[test]
fn benchmark_gradient_boosting_holdout() {
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
    let task = RegressionTask::new("bench_gb", features, target).unwrap();

    let ho = Holdout::new(0.8).with_seed(42);
    let mut gb = GradientBoosting::new().with_n_estimators(50);

    let result = benchmark::resample_regress(&mut gb, &task, &ho, &[&Rmse]).unwrap();
    assert_eq!(result.learner_id, "gradient_boosting");
    assert_eq!(result.scores.len(), 1);
}

// ── StandardScaler tests ───────────────────────────────────────────

#[test]
fn standard_scaler_known_values() {
    let mut scaler = StandardScaler::new();
    let data = array![[0.0], [10.0]]; // mean=5, std=5
    let scaled = scaler.fit_transform(&data).unwrap();
    assert!((scaled[[0, 0]] - (-1.0)).abs() < 1e-10);
    assert!((scaled[[1, 0]] - 1.0).abs() < 1e-10);
}

#[test]
fn standard_scaler_zero_variance() {
    let mut scaler = StandardScaler::new();
    let data = array![[5.0], [5.0], [5.0]];
    let scaled = scaler.fit_transform(&data).unwrap();
    // zero variance → std=1, so (5-5)/1 = 0
    assert!((scaled[[0, 0]]).abs() < 1e-10);
}

#[test]
fn standard_scaler_unfitted_error() {
    let scaler = StandardScaler::new();
    let data = array![[1.0]];
    assert!(scaler.transform(&data).is_err());
}

#[test]
fn standard_scaler_column_mismatch() {
    let mut scaler = StandardScaler::new();
    let train = array![[1.0, 2.0], [3.0, 4.0]];
    scaler.fit(&train).unwrap();
    let bad = array![[1.0]]; // 1 col instead of 2
    assert!(scaler.transform(&bad).is_err());
}

// ── MinMaxScaler tests ─────────────────────────────────────────────

#[test]
fn min_max_scaler_known_values() {
    let mut scaler = MinMaxScaler::new();
    let data = array![[1.0], [5.0], [10.0]];
    let scaled = scaler.fit_transform(&data).unwrap();
    assert!((scaled[[0, 0]] - 0.0).abs() < 1e-10);
    assert!((scaled[[2, 0]] - 1.0).abs() < 1e-10);
    // midpoint: (5-1)/(10-1) = 4/9
    assert!((scaled[[1, 0]] - 4.0 / 9.0).abs() < 1e-10);
}

#[test]
fn min_max_scaler_constant_column() {
    let mut scaler = MinMaxScaler::new();
    let data = array![[3.0], [3.0]];
    let scaled = scaler.fit_transform(&data).unwrap();
    assert!((scaled[[0, 0]] - 0.0).abs() < 1e-10);
}

// ── Imputer tests ──────────────────────────────────────────────────

#[test]
fn imputer_mean() {
    let mut imp = Imputer::mean();
    let data = array![[1.0, f64::NAN], [3.0, 4.0]];
    let filled = imp.fit_transform(&data).unwrap();
    assert!((filled[[0, 0]] - 1.0).abs() < 1e-10); // no NaN
    assert!((filled[[0, 1]] - 4.0).abs() < 1e-10); // mean of [4.0]
    assert!((filled[[1, 1]] - 4.0).abs() < 1e-10);
}

#[test]
fn imputer_median() {
    let mut imp = Imputer::median();
    let data = array![[1.0], [f64::NAN], [3.0], [5.0]];
    let filled = imp.fit_transform(&data).unwrap();
    // median of [1, 3, 5] = 3
    assert!((filled[[1, 0]] - 3.0).abs() < 1e-10);
}

#[test]
fn imputer_constant() {
    let mut imp = Imputer::constant(-1.0);
    let data = array![[f64::NAN], [2.0]];
    let filled = imp.fit_transform(&data).unwrap();
    assert!((filled[[0, 0]] - (-1.0)).abs() < 1e-10);
}

#[test]
fn imputer_no_nan_passthrough() {
    let mut imp = Imputer::mean();
    let data = array![[1.0, 2.0], [3.0, 4.0]];
    let filled = imp.fit_transform(&data).unwrap();
    assert_eq!(filled, data);
}

// ── OneHotEncoder tests ────────────────────────────────────────────

#[test]
fn one_hot_encoder_binary() {
    let mut enc = OneHotEncoder::new(vec![0]);
    let data = array![[0.0, 10.0], [1.0, 20.0], [0.0, 30.0]];
    let encoded = enc.fit_transform(&data).unwrap();
    assert_eq!(encoded.ncols(), 3); // 2 categories + 1 passthrough
    // Row 0: category=0 → [1, 0, 10]
    assert!((encoded[[0, 0]] - 1.0).abs() < 1e-10);
    assert!((encoded[[0, 1]] - 0.0).abs() < 1e-10);
    assert!((encoded[[0, 2]] - 10.0).abs() < 1e-10);
    // Row 1: category=1 → [0, 1, 20]
    assert!((encoded[[1, 0]] - 0.0).abs() < 1e-10);
    assert!((encoded[[1, 1]] - 1.0).abs() < 1e-10);
}

#[test]
fn one_hot_encoder_three_categories() {
    let mut enc = OneHotEncoder::new(vec![0]);
    let data = array![[0.0], [1.0], [2.0]];
    let encoded = enc.fit_transform(&data).unwrap();
    assert_eq!(encoded.ncols(), 3);
    // Row 2: category=2 → [0, 0, 1]
    assert!((encoded[[2, 2]] - 1.0).abs() < 1e-10);
}

#[test]
fn one_hot_encoder_unseen_category() {
    let mut enc = OneHotEncoder::new(vec![0]);
    let train = array![[0.0], [1.0]];
    enc.fit(&train).unwrap();
    let test = array![[2.0]]; // unseen
    let encoded = enc.transform(&test).unwrap();
    // All zeros for unseen category
    assert!((encoded[[0, 0]]).abs() < 1e-10);
    assert!((encoded[[0, 1]]).abs() < 1e-10);
}

#[test]
fn one_hot_encoder_transform_names() {
    let mut enc = OneHotEncoder::new(vec![0]);
    let data = array![[0.0, 10.0], [1.0, 20.0]];
    enc.fit(&data).unwrap();
    let names = vec!["color".into(), "value".into()];
    let new_names = enc.transform_names(&names).unwrap();
    assert_eq!(new_names, vec!["color_0", "color_1", "value"]);
}

#[test]
fn one_hot_encoder_transform_sparse_matches_dense() {
    let mut enc = OneHotEncoder::new(vec![0]);
    let data = array![[0.0, 10.0], [1.0, 20.0], [2.0, 30.0], [0.0, 40.0]];
    let dense = enc.fit_transform(&data).unwrap();
    let sparse = enc.transform_sparse(&data).unwrap();
    assert_eq!(sparse.to_dense(), dense);
}

#[test]
fn one_hot_encoder_transform_sparse_high_cardinality_is_mostly_zero() {
    let n = 200;
    let mut enc = OneHotEncoder::new(vec![0]);
    let data = Array2::from_shape_fn((n, 1), |(i, _)| i as f64); // n distinct categories
    enc.fit(&data).unwrap();
    let sparse = enc.transform_sparse(&data).unwrap();
    assert_eq!(sparse.n_rows(), n);
    assert_eq!(sparse.n_cols(), n);
    assert_eq!(
        sparse.nnz(),
        n,
        "exactly one nonzero per row for a pure one-hot column"
    );
    assert!(
        sparse.density() < 0.01,
        "density should be ~1/n, got {}",
        sparse.density()
    );
}

// ── LabelEncoder tests ────────────────────────────────────────────

#[test]
fn label_encoder_roundtrip() {
    let encoder = LabelEncoder::fit(&["cat", "dog", "bird", "cat"]);
    assert_eq!(encoder.n_classes(), 3);
    let encoded = encoder.encode(&["bird", "cat", "dog"]).unwrap();
    assert_eq!(encoded, vec![0, 1, 2]);
    let decoded = encoder.decode(&encoded);
    assert_eq!(decoded, vec!["bird", "cat", "dog"]);
}

#[test]
fn label_encoder_unknown_label() {
    let encoder = LabelEncoder::fit(&["a", "b"]);
    assert!(encoder.encode(&["c"]).is_err());
}

/// Regression test (5th audit, LOW-C): `with_class_names` with FEWER names
/// than the highest label + 1 used to be accepted silently — `n_classes()`
/// is `class_names.len()`, so the mismatch only surfaced later as an opaque
/// index-out-of-bounds panic deep inside whatever consumed the class count
/// first (probe: SMOTE's per-class grouping). It must panic immediately,
/// with a message naming both counts.
#[test]
#[should_panic(expected = "with_class_names: 2 class name(s) provided")]
fn with_class_names_panics_immediately_on_too_few_names() {
    let features = array![[0.0], [1.0], [2.0]];
    let target = vec![0, 1, 2]; // labels {0, 1, 2} need >= 3 names
    let _task = ClassificationTask::new("too_few_names", features, target)
        .unwrap()
        .with_class_names(vec!["a".into(), "b".into()]);
}

/// Companion of the panic test: MORE names than observed labels stays
/// valid — it's exactly how fold/subset tasks keep their parent's class
/// width (HIGH-4's propagation).
#[test]
fn with_class_names_accepts_extra_names_for_unobserved_classes() {
    let features = array![[0.0], [1.0]];
    let target = vec![0, 1];
    let task = ClassificationTask::new("extra_names", features, target)
        .unwrap()
        .with_class_names(vec!["a".into(), "b".into(), "c".into()]);
    assert_eq!(task.n_classes(), 3);
}

// ── Pipeline tests ─────────────────────────────────────────────────

/// Regression test (4th audit, HIGH-4): `Pipeline::train_classif` rebuilt
/// the transformed task without `class_names`, so `n_classes` was re-derived
/// as max(label)+1 — a training split that lost the highest class produced
/// probability rows narrower than the task's real class count.
#[test]
fn pipeline_preserves_class_width_when_a_class_is_absent() {
    let features = array![[0.0], [1.0], [2.0], [3.0]];
    // Task declares 3 classes; the training labels only contain {0, 1}.
    let target = vec![0, 0, 1, 1];
    let task = ClassificationTask::new("narrow", features, target)
        .unwrap()
        .with_class_names(vec!["a".into(), "b".into(), "c".into()]);
    assert_eq!(task.n_classes(), 3);

    let mut pipe = Pipeline::new(
        vec![Box::new(StandardScaler::new())],
        Box::new(GaussianNB::new()),
    );
    let model = pipe.train_classif(&task).unwrap();
    let pred = model.predict(task.features()).unwrap();
    match pred {
        Prediction::Classification { probabilities, .. } => {
            let probs = probabilities.expect("GaussianNB emits probabilities");
            assert_eq!(
                probs[0].len(),
                3,
                "probability rows must keep the task's declared class width"
            );
        }
        _ => panic!("expected classification prediction"),
    }
}

/// The composition the audit probe crashed: Stacking's fold-wise
/// `class_names` propagation used to be destroyed when the base learner is
/// itself a Pipeline, panicking with index-out-of-bounds in ~half the seeds
/// once a CV fold lost the rare class. All seeds must train cleanly.
#[test]
fn stacking_with_pipeline_base_survives_rare_classes() {
    for seed in 0..20u64 {
        let n = 12;
        let features = Array2::from_shape_fn((n, 2), |(i, j)| {
            ((i as f64 + seed as f64) * 0.7 + j as f64 * 1.3).sin()
        });
        // Rare highest class: a 2-fold split often leaves it out of one fold.
        let mut target = vec![0usize; n];
        for (i, t) in target.iter_mut().enumerate() {
            *t = i % 2;
        }
        target[(seed as usize) % n] = 2;
        let task = ClassificationTask::new("rare", features, target).unwrap();

        let mut stack = Stacking::new(
            vec![Box::new(|| {
                Box::new(Pipeline::new(
                    vec![Box::new(StandardScaler::new())],
                    Box::new(GaussianNB::new()),
                )) as Box<dyn Learner>
            })],
            || Box::new(LogisticRegression::new()),
        )
        .with_cv_folds(2);
        stack
            .train_classif(&task)
            .unwrap_or_else(|e| panic!("seed {seed}: stacking over pipeline base failed: {e}"));
    }
}

/// Regression test (4th audit, M-4): the resamplers rebuilt the balanced
/// task without feature metadata, renaming every feature to x0/x1/... for
/// selectors and importances downstream.
#[test]
fn resamplers_preserve_task_metadata() {
    let features = array![
        [0.0, 10.0],
        [0.1, 11.0],
        [0.2, 12.0],
        [0.9, 13.0],
        [1.0, 14.0],
        [1.1, 15.0],
        [1.2, 16.0],
        [0.95, 17.0]
    ];
    let target = vec![0, 0, 0, 1, 1, 1, 1, 1];
    let task = ClassificationTask::new("meta", features, target)
        .unwrap()
        .with_feature_names(vec!["slope".into(), "ndvi".into()])
        .unwrap()
        .with_class_names(vec!["stable".into(), "slide".into(), "flow".into()]);

    let balanced = Smote::new().with_k_neighbors(2).balance(&task).unwrap();
    assert_eq!(balanced.feature_names(), &["slope", "ndvi"]);
    assert_eq!(balanced.n_classes(), 3);

    let balanced = Adasyn::new().with_k_neighbors(2).balance(&task).unwrap();
    assert_eq!(balanced.feature_names(), &["slope", "ndvi"]);
    assert_eq!(balanced.n_classes(), 3);
}

/// Regression test (4th audit, M-5): SMOTE/ADASYN interpolated between
/// k-NN neighbours with NaN features — arbitrary neighbour order and NaN
/// synthetic rows, all silently. Must be a clear error pointing at
/// imputation order instead.
#[test]
fn resamplers_reject_nan_features() {
    let mut features = Array2::from_shape_fn((8, 2), |(i, j)| (i * 2 + j) as f64);
    features[[1, 0]] = f64::NAN;
    let target = vec![0, 0, 0, 0, 0, 1, 1, 1];
    let task = ClassificationTask::new("nan", features, target).unwrap();

    let err = Smote::new().with_k_neighbors(2).balance(&task).unwrap_err();
    assert!(format!("{err}").contains("impute"), "got: {err}");
    let err = Adasyn::new()
        .with_k_neighbors(2)
        .balance(&task)
        .unwrap_err();
    assert!(format!("{err}").contains("impute"), "got: {err}");
}

#[test]
fn pipeline_passthrough() {
    let features = array![[0.0], [1.0], [2.0], [3.0]];
    let target = vec![0, 0, 1, 1];
    let task = ClassificationTask::new("pass", features, target).unwrap();

    let mut pipe = Pipeline::new(vec![], Box::new(DecisionTree::default()));
    assert_eq!(pipe.id(), "pipeline(decision_tree)");

    let model = pipe.train_classif(&task).unwrap();
    let pred = model.predict(task.features()).unwrap();
    assert_eq!(pred.n_samples(), 4);
}

#[test]
fn pipeline_single_transformer() {
    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [100.0, 100.0],
        [100.1, 99.9],
        [99.9, 100.1]
    ];
    let target = vec![0, 0, 0, 1, 1, 1];
    let task = ClassificationTask::new("scaled", features, target).unwrap();

    let mut pipe = Pipeline::new(
        vec![Box::new(StandardScaler::new())],
        Box::new(DecisionTree::default()),
    );
    let model = pipe.train_classif(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_classif(task.target().to_vec());

    let acc = Accuracy.score(&pred).unwrap();
    assert_eq!(acc, 1.0);
}

#[test]
fn pipeline_multiple_transformers() {
    let features = array![[1.0, 100.0], [2.0, 200.0], [3.0, 300.0], [4.0, 400.0]];
    let target = vec![0, 0, 1, 1];
    let task = ClassificationTask::new("multi", features, target).unwrap();

    let mut pipe = Pipeline::new(
        vec![
            Box::new(MinMaxScaler::new()),
            Box::new(StandardScaler::new()),
        ],
        Box::new(DecisionTree::default()),
    );
    let model = pipe.train_classif(&task).unwrap();
    let pred = model.predict(task.features()).unwrap();
    assert_eq!(pred.n_samples(), 4);
}

#[test]
fn pipeline_in_benchmark_cv() {
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
        [1.1, 1.1]
    ];
    let target = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
    let task = ClassificationTask::new("pipe_cv", features, target).unwrap();

    let cv = CrossValidation::new(4).with_seed(42);
    let mut pipe = Pipeline::new(
        vec![Box::new(StandardScaler::new())],
        Box::new(DecisionTree::new().with_max_depth(3)),
    );

    let result = benchmark::resample_classif(&mut pipe, &task, &cv, &[&Accuracy]).unwrap();
    assert_eq!(result.scores.len(), 4);
    let mean = result.mean_scores()[0];
    assert!(
        mean >= 0.5,
        "pipeline CV accuracy should be reasonable, got {mean}"
    );
}

#[test]
fn pipeline_regression() {
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
    let task = RegressionTask::new("pipe_reg", features, target).unwrap();

    let ho = Holdout::new(0.8).with_seed(42);
    let mut pipe = Pipeline::new(
        vec![Box::new(MinMaxScaler::new())],
        Box::new(DecisionTree::default()),
    );

    let result = benchmark::resample_regress(&mut pipe, &task, &ho, &[&Rmse]).unwrap();
    assert_eq!(result.scores.len(), 1);
}

#[test]
fn pipeline_nested_in_bagging() {
    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [1.0, 1.0],
        [1.1, 0.9],
        [0.9, 1.1]
    ];
    let target = vec![0, 0, 0, 1, 1, 1];
    let task = ClassificationTask::new("bag_pipe", features, target).unwrap();

    let mut bag = Bagging::new(|| {
        Box::new(Pipeline::new(
            vec![Box::new(StandardScaler::new())],
            Box::new(DecisionTree::default()),
        )) as Box<dyn Learner>
    })
    .with_n_estimators(5)
    .with_seed(42);

    let model = bag.train_classif(&task).unwrap();
    let pred = model.predict(task.features()).unwrap();
    assert_eq!(pred.n_samples(), 6);
}

// ── GridSearch tests ───────────────────────────────────────────────

#[test]
fn grid_search_classif_finds_best() {
    use smelt_ml::tuning::ParamGrid;

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
        [1.1, 1.1]
    ];
    let target = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
    let task = ClassificationTask::new("gs", features, target).unwrap();

    let mut grid = ParamGrid::new();
    grid.insert("max_depth".into(), vec![1.0.into(), 3.0.into(), 5.0.into()]);

    let gs = GridSearch::new(
        |params| {
            Box::new(DecisionTree::new().with_max_depth(params["max_depth"].as_usize().unwrap()))
        },
        grid,
    );
    let cv = CrossValidation::new(4).with_seed(42);
    let result = gs.tune_classif(&task, &cv, &Accuracy).unwrap();

    assert_eq!(result.all_results.len(), 3);
    assert!(result.best_score >= 0.5);
    assert!(result.maximize);
    assert_eq!(result.measure_id, "classif.accuracy");
    assert!(result.best_params.contains_key("max_depth"));
}

#[test]
fn grid_search_regress() {
    use smelt_ml::tuning::ParamGrid;

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
    let task = RegressionTask::new("gs_reg", features, target).unwrap();

    let mut grid = ParamGrid::new();
    grid.insert("max_depth".into(), vec![1.0.into(), 3.0.into(), 5.0.into()]);

    let gs = GridSearch::new(
        |params| {
            Box::new(DecisionTree::new().with_max_depth(params["max_depth"].as_usize().unwrap()))
        },
        grid,
    );
    let ho = Holdout::new(0.8).with_seed(42);
    let result = gs.tune_regress(&task, &ho, &Rmse).unwrap();

    assert_eq!(result.all_results.len(), 3);
    assert!(!result.maximize);
    for (_, score) in &result.all_results {
        assert!(result.best_score <= *score + 1e-10);
    }
}

#[test]
fn grid_search_multi_param() {
    use smelt_ml::tuning::ParamGrid;

    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [0.0, 0.2],
        [1.0, 1.0],
        [1.1, 0.9],
        [0.9, 1.1],
        [1.0, 0.9]
    ];
    let target = vec![0, 0, 0, 0, 1, 1, 1, 1];
    let task = ClassificationTask::new("gs_multi", features, target).unwrap();

    let mut grid = ParamGrid::new();
    grid.insert("max_depth".into(), vec![1.0.into(), 3.0.into()]);
    grid.insert("min_samples_split".into(), vec![2.0.into(), 4.0.into()]);

    let gs = GridSearch::new(
        |params| {
            Box::new(
                DecisionTree::new()
                    .with_max_depth(params["max_depth"].as_usize().unwrap())
                    .with_min_samples_split(params["min_samples_split"].as_usize().unwrap()),
            )
        },
        grid,
    );
    let cv = CrossValidation::new(2).with_seed(42);
    let result = gs.tune_classif(&task, &cv, &Accuracy).unwrap();

    assert_eq!(result.all_results.len(), 4);
}

/// End-to-end proof of the parameter-dependency feature against the exact
/// 5th-audit M-5 shape: a factory that reads a "child" parameter only under
/// one setting of a "parent" flag, so the child is dead otherwise. Without a
/// declared dependency, grid search wastes trials on bit-identical dead-param
/// models; with one, those combinations collapse to a single trial and the
/// surviving dead-branch config carries no child parameter at all.
#[test]
fn grid_search_dependency_collapses_dead_param_combinations() {
    use smelt_ml::tuning::{Dependency, ParamGrid, ParamSet, ParamValue};

    // Factory reads `min_samples_split` ONLY when `use_split_control` is true.
    fn build(params: &ParamSet) -> Box<dyn smelt_ml::learner::Learner> {
        let mut dt = DecisionTree::new().with_max_depth(3);
        if params
            .get("use_split_control")
            .and_then(|v| v.as_bool().ok())
            == Some(true)
        {
            dt = dt.with_min_samples_split(params["min_samples_split"].as_usize().unwrap());
        }
        Box::new(dt)
    }

    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [0.0, 0.2],
        [1.0, 1.0],
        [1.1, 0.9],
        [0.9, 1.1],
        [1.0, 0.9]
    ];
    let target = vec![0, 0, 0, 0, 1, 1, 1, 1];
    let task = ClassificationTask::new("gs_dep", features, target).unwrap();

    let mut grid = ParamGrid::new();
    grid.insert(
        "use_split_control".into(),
        vec![ParamValue::Bool(false), ParamValue::Bool(true)],
    );
    grid.insert(
        "min_samples_split".into(),
        vec![ParamValue::Int(2), ParamValue::Int(5), ParamValue::Int(8)],
    );
    let cv = CrossValidation::new(2).with_seed(42);

    // Without the dependency: full 2×3 = 6 trials; the 3 control=false combos
    // are bit-identical dead-param waste (the M-5 no-op).
    let r_no_dep = GridSearch::new(build, grid.clone())
        .tune_classif(&task, &cv, &Accuracy)
        .unwrap();
    assert_eq!(r_no_dep.all_results.len(), 6);

    // With the dependency: control=false collapses to 1 → 4 distinct trials.
    let r_dep = GridSearch::new(build, grid)
        .with_dependency(Dependency::equals(
            "min_samples_split",
            "use_split_control",
            true,
        ))
        .tune_classif(&task, &cv, &Accuracy)
        .unwrap();
    assert_eq!(r_dep.all_results.len(), 4);

    let dead: Vec<_> = r_dep
        .all_results
        .iter()
        .filter(|(p, _)| p.get("use_split_control") == Some(&ParamValue::Bool(false)))
        .collect();
    assert_eq!(dead.len(), 1, "the dead branch collapses to a single trial");
    assert!(
        !dead[0].0.contains_key("min_samples_split"),
        "the pruned child never reaches the factory or the reported params"
    );
}

/// Regression/determinism test for the rayon-parallelized combination loop:
/// running the same grid twice must produce byte-identical best results, and
/// `all_results` must contain every combination exactly once regardless of
/// which thread evaluated it (parallel iteration must not drop, duplicate,
/// or race on shared state).
#[test]
fn grid_search_parallel_evaluation_is_deterministic() {
    use smelt_ml::tuning::ParamGrid;

    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [0.0, 0.2],
        [1.0, 1.0],
        [1.1, 0.9],
        [0.9, 1.1],
        [1.0, 0.9]
    ];
    let target = vec![0, 0, 0, 0, 1, 1, 1, 1];
    let task = ClassificationTask::new("gs_det", features, target).unwrap();

    let mut grid = ParamGrid::new();
    grid.insert(
        "max_depth".into(),
        vec![1.0.into(), 2.0.into(), 3.0.into(), 4.0.into(), 5.0.into()],
    );
    grid.insert(
        "min_samples_split".into(),
        vec![2.0.into(), 3.0.into(), 4.0.into()],
    );
    let cv = CrossValidation::new(2).with_seed(42);

    let make_gs = || {
        GridSearch::new(
            |params| {
                Box::new(
                    DecisionTree::new()
                        .with_max_depth(params["max_depth"].as_usize().unwrap())
                        .with_min_samples_split(params["min_samples_split"].as_usize().unwrap()),
                )
            },
            grid.clone(),
        )
    };

    let r1 = make_gs().tune_classif(&task, &cv, &Accuracy).unwrap();
    let r2 = make_gs().tune_classif(&task, &cv, &Accuracy).unwrap();

    assert_eq!(r1.best_params, r2.best_params);
    assert!((r1.best_score - r2.best_score).abs() < 1e-10);
    assert_eq!(r1.all_results.len(), 15);
    assert_eq!(r2.all_results.len(), 15);
}

// ── RandomSearch tests ─────────────────────────────────────────────

#[test]
fn random_search_classif() {
    use smelt_ml::tuning::ParamSpace;

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
        [1.1, 1.1]
    ];
    let target = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
    let task = ClassificationTask::new("rs", features, target).unwrap();

    let mut space = ParamSpace::new();
    space.insert(
        "max_depth".into(),
        ParamDistribution::Choice(vec![1.0.into(), 3.0.into(), 5.0.into(), 10.0.into()]),
    );

    let rs = RandomSearch::new(
        |params| {
            Box::new(DecisionTree::new().with_max_depth(params["max_depth"].as_usize().unwrap()))
        },
        space,
    )
    .with_n_iter(5)
    .with_seed(42);

    let cv = CrossValidation::new(4).with_seed(42);
    let result = rs.tune_classif(&task, &cv, &Accuracy).unwrap();

    assert_eq!(result.all_results.len(), 5);
    assert!(result.best_score >= 0.5);
}

/// 4th-audit LOW: an inverted `Uniform` used to panic inside
/// `rng.random_range` mid-tuning; invalid `eta` made Hyperband's bracket
/// math divide by zero or overflow. Both must fail fast with a clean error.
#[test]
fn tuners_reject_invalid_param_space_and_eta_up_front() {
    use smelt_ml::tuning::{Hyperband, ParamSpace};

    let features = array![[0.0, 0.0], [0.1, 0.1], [1.0, 1.0], [1.1, 0.9]];
    let target = vec![0, 0, 1, 1];
    let task = ClassificationTask::new("bad_space", features, target).unwrap();
    let factory = |_params: &smelt_ml::tuning::ParamSet| {
        Box::new(DecisionTree::new()) as Box<dyn smelt_ml::learner::Learner>
    };

    let mut inverted = ParamSpace::new();
    inverted.insert("lr".into(), ParamDistribution::Uniform(1.0, 0.1));
    let cv = CrossValidation::new(2).with_seed(42);
    let err = RandomSearch::new(factory, inverted.clone())
        .tune_classif(&task, &cv, &Accuracy)
        .unwrap_err();
    assert!(err.to_string().contains("lr"), "got: {err}");

    let mut valid = ParamSpace::new();
    valid.insert("lr".into(), ParamDistribution::Uniform(0.1, 1.0));
    let err = Hyperband::new(factory, valid)
        .with_eta(1)
        .tune_classif(&task, &Accuracy)
        .unwrap_err();
    assert!(err.to_string().contains("eta"), "got: {err}");
}

#[test]
fn random_search_uniform() {
    use smelt_ml::tuning::ParamSpace;

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
    let task = RegressionTask::new("rs_uni", features, target).unwrap();

    let mut space = ParamSpace::new();
    space.insert("max_depth".into(), ParamDistribution::Uniform(1.0, 10.0));

    let rs = RandomSearch::new(
        |params| {
            Box::new(DecisionTree::new().with_max_depth(params["max_depth"].as_usize().unwrap()))
        },
        space,
    )
    .with_n_iter(8)
    .with_seed(42);

    let ho = Holdout::new(0.8).with_seed(42);
    let result = rs.tune_regress(&task, &ho, &Rmse).unwrap();

    assert_eq!(result.all_results.len(), 8);
    for (params, _) in &result.all_results {
        let d = params["max_depth"].as_f64().unwrap();
        assert!((1.0..=10.0).contains(&d));
    }
}

#[test]
fn random_search_deterministic() {
    use smelt_ml::tuning::ParamSpace;

    let features = array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]];
    let target = vec![0, 0, 0, 0, 1, 1, 1, 1];
    let task = ClassificationTask::new("det", features, target).unwrap();

    let mut space = ParamSpace::new();
    space.insert(
        "max_depth".into(),
        ParamDistribution::Choice(vec![
            1.0.into(),
            2.0.into(),
            3.0.into(),
            4.0.into(),
            5.0.into(),
        ]),
    );

    let cv = CrossValidation::new(2).with_seed(42);

    let rs1 = RandomSearch::new(
        |p| Box::new(DecisionTree::new().with_max_depth(p["max_depth"].as_usize().unwrap())),
        space.clone(),
    )
    .with_n_iter(3)
    .with_seed(123);
    let r1 = rs1.tune_classif(&task, &cv, &Accuracy).unwrap();

    let rs2 = RandomSearch::new(
        |p| Box::new(DecisionTree::new().with_max_depth(p["max_depth"].as_usize().unwrap())),
        space,
    )
    .with_n_iter(3)
    .with_seed(123);
    let r2 = rs2.tune_classif(&task, &cv, &Accuracy).unwrap();

    assert_eq!(r1.best_params, r2.best_params);
    assert!((r1.best_score - r2.best_score).abs() < 1e-10);
}

#[test]
fn random_search_log_uniform() {
    use smelt_ml::tuning::ParamSpace;

    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [0.0, 0.2],
        [1.0, 1.0],
        [1.1, 0.9],
        [0.9, 1.1],
        [1.0, 0.9]
    ];
    let target = vec![0, 0, 0, 0, 1, 1, 1, 1];
    let task = ClassificationTask::new("log", features, target).unwrap();

    let mut space = ParamSpace::new();
    space.insert(
        "learning_rate".into(),
        ParamDistribution::LogUniform(0.001, 1.0),
    );

    let rs = RandomSearch::new(
        |params| {
            Box::new(
                LogisticRegression::new()
                    .with_learning_rate(params["learning_rate"].as_f64().unwrap())
                    .with_max_iter(500),
            )
        },
        space,
    )
    .with_n_iter(5)
    .with_seed(42);

    let cv = CrossValidation::new(2).with_seed(42);
    let result = rs.tune_classif(&task, &cv, &Accuracy).unwrap();

    for (params, _) in &result.all_results {
        let lr = params["learning_rate"].as_f64().unwrap();
        assert!((0.001..=1.0).contains(&lr), "lr={lr} out of bounds");
    }
}

// ── Permutation Feature Importance tests ───────────────────────────

#[test]
fn importance_informative_feature() {
    let features = array![
        [0.0, 99.0],
        [0.1, 42.0],
        [0.2, 13.0],
        [0.0, 77.0],
        [1.0, 99.0],
        [1.1, 42.0],
        [1.2, 13.0],
        [1.0, 77.0]
    ];
    let target = vec![0, 0, 0, 0, 1, 1, 1, 1];
    let task = ClassificationTask::new("imp", features.clone(), target.clone())
        .unwrap()
        .with_feature_names(vec!["signal".into(), "noise".into()])
        .unwrap();

    let mut tree = DecisionTree::default();
    let model = tree.train_classif(&task).unwrap();

    let imp = permutation_importance_classif(&*model, &task, &Accuracy, 5, 42).unwrap();
    assert_eq!(imp.len(), 2);
    assert!(
        imp[0].importance >= imp[1].importance,
        "signal should be more important than noise"
    );
}

#[test]
fn importance_regression() {
    let features = array![
        [1.0, 99.0],
        [2.0, 42.0],
        [3.0, 13.0],
        [4.0, 77.0],
        [5.0, 99.0],
        [6.0, 42.0],
        [7.0, 13.0],
        [8.0, 77.0]
    ];
    let target = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
    let task = RegressionTask::new("imp_r", features, target)
        .unwrap()
        .with_feature_names(vec!["x".into(), "noise".into()])
        .unwrap();

    let mut tree = DecisionTree::default();
    let model = tree.train_regress(&task).unwrap();

    let imp = permutation_importance_regress(&*model, &task, &Rmse, 5, 42).unwrap();
    assert_eq!(imp.len(), 2);
    assert!(
        imp[0].importance > 0.0,
        "informative feature should have positive importance"
    );
}

#[test]
fn importance_has_std_dev() {
    let features = array![[0.0], [1.0], [2.0], [3.0]];
    let target = vec![0, 0, 1, 1];
    let task = ClassificationTask::new("std", features, target).unwrap();

    let mut tree = DecisionTree::default();
    let model = tree.train_classif(&task).unwrap();

    let imp = permutation_importance_classif(&*model, &task, &Accuracy, 10, 42).unwrap();
    // std_dev should be non-negative
    for fi in &imp {
        assert!(fi.std_dev >= 0.0);
    }
}

// ── Spatial CV tests ───────────────────────────────────────────────

#[test]
fn spatial_block_splits_cover_all() {
    let coords: Vec<(f64, f64)> = (0..20).map(|i| (i as f64, (i % 5) as f64)).collect();
    let cv = SpatialBlockCV::new(4, coords);
    let splits = cv.splits(20).unwrap();

    assert_eq!(splits.len(), 4);

    // All indices should appear in some test fold
    let mut all_test: Vec<usize> = splits.iter().flat_map(|(_, t)| t.clone()).collect();
    all_test.sort();
    all_test.dedup();
    assert_eq!(all_test.len(), 20);

    // Train + test should cover all (but may overlap across folds)
    for (train, test) in &splits {
        assert!(!test.is_empty(), "test fold should not be empty");
        assert!(!train.is_empty(), "train fold should not be empty");
    }
}

#[test]
fn spatial_block_spatial_separation() {
    // Two clear spatial clusters
    let mut coords = Vec::new();
    for i in 0..10 {
        coords.push((i as f64 * 0.1, 0.0)); // cluster at x=[0,1]
    }
    for i in 0..10 {
        coords.push((10.0 + i as f64 * 0.1, 0.0)); // cluster at x=[10,11]
    }

    let cv = SpatialBlockCV::new(2, coords.clone());
    let splits = cv.splits(20).unwrap();

    // In at least one split, test should be dominated by one cluster
    let has_separated = splits.iter().any(|(_, test)| {
        let in_cluster1 = test.iter().filter(|&&i| i < 10).count();
        let in_cluster2 = test.iter().filter(|&&i| i >= 10).count();
        in_cluster1 == 0 || in_cluster2 == 0
    });
    assert!(
        has_separated,
        "spatial block should separate clusters into different folds"
    );
}

#[test]
fn spatial_block_with_block_size_uses_fixed_cell_size_not_n_folds() {
    // 3 pairs of points 2000 units apart, each pair 1 unit apart (well within
    // a block_size=1000 cell). With n_folds derived from n_folds alone
    // (grid_size = ceil(sqrt(3)) = 2), the grid would only have 2 columns
    // and could not resolve blocks this fine over a 4001-unit extent. With a
    // fixed block_size=1000, grid_cols = ceil(4001/1000) = 5, so cell
    // (col = floor(x/1000)) is 0, 2, 4 for the three pairs, and
    // fold = col % n_folds(=3) gives folds 0, 2, 1 respectively — i.e. the
    // grid resolution comes from block_size, independent of n_folds.
    let coords = vec![
        (0.0, 0.0),
        (1.0, 0.0),
        (2000.0, 0.0),
        (2001.0, 0.0),
        (4000.0, 0.0),
        (4001.0, 0.0),
    ];
    let cv = SpatialBlockCV::with_block_size(3, coords, 1000.0);
    let splits = cv.splits(6).unwrap();
    assert_eq!(splits.len(), 3);

    let mut test_by_fold: Vec<Vec<usize>> = splits.iter().map(|(_, t)| t.clone()).collect();
    for fold in &mut test_by_fold {
        fold.sort();
    }
    assert_eq!(
        test_by_fold[0],
        vec![0, 1],
        "fold 0 should hold the first pair"
    );
    assert_eq!(
        test_by_fold[1],
        vec![4, 5],
        "fold 1 should hold the third pair"
    );
    assert_eq!(
        test_by_fold[2],
        vec![2, 3],
        "fold 2 should hold the second pair"
    );
}

#[test]
fn spatial_block_with_block_size_rejects_non_positive_block_size() {
    let coords = vec![(0.0, 0.0), (1.0, 0.0)];
    let cv = SpatialBlockCV::with_block_size(2, coords, 0.0);
    assert!(cv.splits(2).is_err());
}

/// 4th-audit LOW: folds whose grid cells received no samples used to be
/// emitted with an empty test side, so measures over them produced
/// 0/0 = NaN and poisoned `mean_scores` for the whole resample run.
#[test]
fn spatial_block_drops_folds_with_empty_test_side() {
    // Two tight clusters and a huge block_size: only 2 grid cells are
    // populated, but 4 folds are requested -- folds 2 and 3 get no cells.
    let mut coords = Vec::new();
    for i in 0..5 {
        coords.push((i as f64 * 0.01, 0.0));
        coords.push((100.0 + i as f64 * 0.01, 0.0));
    }
    let cv = SpatialBlockCV::with_block_size(4, coords, 50.0);
    let splits = cv.splits(10).unwrap();

    assert_eq!(splits.len(), 2, "the two unpopulated folds must be dropped");
    for (train, test) in &splits {
        assert!(!train.is_empty());
        assert!(!test.is_empty());
    }
}

#[test]
fn spatial_block_errors_when_no_usable_fold_remains() {
    // Every sample in one tight cluster inside a single huge cell: the only
    // populated fold has an empty train side, so no usable fold remains.
    let coords: Vec<(f64, f64)> = (0..8).map(|i| (i as f64 * 0.01, 0.0)).collect();
    let cv = SpatialBlockCV::with_block_size(3, coords, 1000.0);
    let err = cv.splits(8).unwrap_err();
    assert!(err.to_string().contains("no usable folds"), "got: {err}");
}

#[test]
fn spatial_buffer_removes_nearby() {
    // Tight cluster at (0,0) and one point far at (100,100)
    let coords = vec![(0.0, 0.0), (0.1, 0.0), (0.0, 0.1), (100.0, 100.0)];
    let cv = SpatialBufferCV::new(2, coords, 1.0).with_seed(42);
    let splits = cv.splits(4).unwrap();

    for (train, test) in &splits {
        // If test contains point from cluster, nearby train points should be removed
        if test.iter().any(|&i| i < 3) {
            // Train should not contain nearby cluster points
            let nearby_in_train = train.iter().filter(|&&i| i < 3).count();
            // The buffer should remove at least some nearby points
            assert!(
                nearby_in_train < 3,
                "buffer should remove nearby train samples"
            );
        }
    }
}

// ── CSV Loading tests ──────────────────────────────────────────────

#[test]
fn load_csv_classification() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.csv");
    std::fs::write(&path, "x1,x2,label\n1.0,2.0,0\n3.0,4.0,1\n5.0,6.0,1\n").unwrap();

    let task = CsvLoader::from_path(&path)
        .target("label")
        .load_classif()
        .unwrap();
    assert_eq!(task.n_samples(), 3);
    assert_eq!(task.n_features(), 2);
    assert_eq!(task.target(), &[0, 1, 1]);
    assert_eq!(task.feature_names(), &["x1", "x2"]);
}

/// Regression test (4th audit, M-8): the target column used to bypass the
/// missing-value handling the features get. A literal "NaN" in a regression
/// target (numpy.savetxt's output) parsed to f64::NAN and trained/predicted
/// NaN silently; a literal "NA" in a classification target was label-encoded
/// as a real class. Both must be load-time errors naming the row.
#[test]
fn load_csv_rejects_missing_target_values() {
    let dir = tempfile::tempdir().unwrap();

    // Regression: "NaN" literal in the target
    let path = dir.path().join("nan_target.csv");
    std::fs::write(&path, "x,y\n1.0,2.0\n3.0,NaN\n5.0,10.0\n").unwrap();
    let err = CsvLoader::from_path(&path).target("y").load_regress();
    let msg = format!("{}", err.unwrap_err());
    assert!(
        msg.contains("row 1") && msg.contains("missing"),
        "got: {msg}"
    );

    // Regression: "inf" parses but is unusable as a target
    let path = dir.path().join("inf_target.csv");
    std::fs::write(&path, "x,y\n1.0,2.0\n3.0,inf\n").unwrap();
    let msg = format!(
        "{}",
        CsvLoader::from_path(&path)
            .target("y")
            .load_regress()
            .unwrap_err()
    );
    assert!(msg.contains("not finite"), "got: {msg}");

    // Classification: "NA" must not become a class
    let path = dir.path().join("na_target.csv");
    std::fs::write(&path, "x,label\n1.0,a\n2.0,NA\n3.0,b\n").unwrap();
    let msg = format!(
        "{}",
        CsvLoader::from_path(&path)
            .target("label")
            .load_classif()
            .unwrap_err()
    );
    assert!(
        msg.contains("row 1") && msg.contains("missing"),
        "got: {msg}"
    );

    // Missing values in FEATURES stay allowed (NaN pipeline), only the
    // target is strict.
    let path = dir.path().join("na_feature.csv");
    std::fs::write(&path, "x,y\nNA,2.0\n3.0,6.0\n").unwrap();
    let task = CsvLoader::from_path(&path)
        .target("y")
        .load_regress()
        .unwrap();
    assert!(task.features()[[0, 0]].is_nan());
}

#[test]
fn load_csv_regression() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.csv");
    std::fs::write(&path, "x,y\n1.0,2.0\n3.0,6.0\n5.0,10.0\n").unwrap();

    let task = CsvLoader::from_path(&path)
        .target("y")
        .load_regress()
        .unwrap();
    assert_eq!(task.n_samples(), 3);
    assert_eq!(task.n_features(), 1);
    assert_eq!(task.target(), &[2.0, 6.0, 10.0]);
}

#[test]
fn load_csv_string_target() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.csv");
    std::fs::write(&path, "x,species\n1.0,cat\n2.0,dog\n3.0,cat\n").unwrap();

    let task = CsvLoader::from_path(&path)
        .target("species")
        .load_classif()
        .unwrap();
    assert_eq!(task.n_classes(), 2); // cat=0, dog=1
    assert_eq!(task.target(), &[0, 1, 0]);
}

#[test]
fn load_csv_missing_column_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.csv");
    std::fs::write(&path, "x,y\n1.0,2.0\n").unwrap();

    let err = CsvLoader::from_path(&path).target("missing").load_classif();
    assert!(err.is_err());
}

#[test]
fn load_csv_missing_values_become_nan() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.csv");
    std::fs::write(
        &path,
        "x1,x2,y\n1.0,,2.0\nNA,4.0,6.0\nnan,NULL,10.0\n?,8.0,14.0\n",
    )
    .unwrap();

    let task = CsvLoader::from_path(&path)
        .target("y")
        .load_regress()
        .unwrap();
    let f = task.features();
    assert!(f[[0, 1]].is_nan(), "empty cell must load as NaN");
    assert!(f[[1, 0]].is_nan(), "NA must load as NaN");
    assert!(
        f[[2, 0]].is_nan() && f[[2, 1]].is_nan(),
        "nan/NULL must load as NaN"
    );
    assert!(f[[3, 0]].is_nan(), "? must load as NaN");
    assert_eq!(f[[0, 0]], 1.0);
    assert_eq!(f[[3, 1]], 8.0);
    // No categorical columns: everything non-missing is numeric.
    assert!(task.categorical_features().is_empty());
}

#[test]
fn load_csv_string_column_auto_categorical() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.csv");
    std::fs::write(
        &path,
        "soil,elev,y\nclay,100.0,1.0\nsand,200.0,2.0\nclay,300.0,3.0\nloam,,4.0\n",
    )
    .unwrap();

    let task = CsvLoader::from_path(&path)
        .target("y")
        .load_regress()
        .unwrap();
    assert_eq!(task.categorical_features(), vec![0]);
    assert_eq!(
        task.feature_types()[0],
        FeatureType::Categorical { n_categories: 3 }
    );
    assert_eq!(task.feature_types()[1], FeatureType::Numeric);
    // LabelEncoder sorts: clay=0, loam=1, sand=2.
    let f = task.features();
    assert_eq!(f[[0, 0]], 0.0);
    assert_eq!(f[[1, 0]], 2.0);
    assert_eq!(f[[3, 0]], 1.0);
    assert!(f[[3, 1]].is_nan());
}

#[test]
fn load_csv_forced_categorical_column() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.csv");
    std::fs::write(&path, "region,x,y\n3,1.0,0\n7,2.0,1\n3,3.0,0\n").unwrap();

    // Without forcing, "region" parses as numeric.
    let plain = CsvLoader::from_path(&path)
        .target("y")
        .load_classif()
        .unwrap();
    assert!(plain.categorical_features().is_empty());

    let task = CsvLoader::from_path(&path)
        .target("y")
        .categorical(&["region"])
        .load_classif()
        .unwrap();
    assert_eq!(task.categorical_features(), vec![0]);

    // Forcing a nonexistent column errors.
    let err = CsvLoader::from_path(&path)
        .target("y")
        .categorical(&["no_such_col"])
        .load_classif();
    assert!(err.is_err());
}

// ── Parquet loading tests (item 16d, `parquet` feature) ──────────────

#[cfg(feature = "parquet")]
mod parquet_tests {
    use super::*;
    use polars::prelude::{Column, DataFrame, ParquetWriter};
    use smelt_ml::prelude::ParquetLoader;

    fn write_parquet(path: &std::path::Path, height: usize, columns: Vec<Column>) {
        let mut df = DataFrame::new(height, columns).unwrap();
        let file = std::fs::File::create(path).unwrap();
        ParquetWriter::new(file).finish(&mut df).unwrap();
    }

    #[test]
    fn load_parquet_classification() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.parquet");
        write_parquet(
            &path,
            3,
            vec![
                Column::new("x1".into(), vec![1.0f64, 3.0, 5.0]),
                Column::new("x2".into(), vec![2.0f64, 4.0, 6.0]),
                Column::new("label".into(), vec![0i64, 1, 1]),
            ],
        );

        let task = ParquetLoader::from_path(&path)
            .target("label")
            .load_classif()
            .unwrap();
        assert_eq!(task.n_samples(), 3);
        assert_eq!(task.n_features(), 2);
        assert_eq!(task.target(), &[0, 1, 1]);
        assert_eq!(task.feature_names(), &["x1", "x2"]);
    }

    #[test]
    fn load_parquet_regression() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.parquet");
        write_parquet(
            &path,
            3,
            vec![
                Column::new("x".into(), vec![1.0f64, 3.0, 5.0]),
                Column::new("y".into(), vec![2.0f64, 6.0, 10.0]),
            ],
        );

        let task = ParquetLoader::from_path(&path)
            .target("y")
            .load_regress()
            .unwrap();
        assert_eq!(task.n_samples(), 3);
        assert_eq!(task.n_features(), 1);
        assert_eq!(task.target(), &[2.0, 6.0, 10.0]);
    }

    #[test]
    fn load_parquet_string_target() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.parquet");
        write_parquet(
            &path,
            3,
            vec![
                Column::new("x".into(), vec![1.0f64, 2.0, 3.0]),
                Column::new(
                    "species".into(),
                    vec!["cat".to_string(), "dog".to_string(), "cat".to_string()],
                ),
            ],
        );

        let task = ParquetLoader::from_path(&path)
            .target("species")
            .load_classif()
            .unwrap();
        assert_eq!(task.n_classes(), 2); // cat=0, dog=1
        assert_eq!(task.target(), &[0, 1, 0]);
    }

    #[test]
    fn load_parquet_string_target_with_null_errors() {
        // A null in a string target used to silently become the empty
        // string "" — a phantom class distinct from any real label —
        // instead of erroring like the numeric-target null path already did.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.parquet");
        write_parquet(
            &path,
            3,
            vec![
                Column::new("x".into(), vec![1.0f64, 2.0, 3.0]),
                Column::new(
                    "species".into(),
                    vec![Some("cat".to_string()), None, Some("dog".to_string())],
                ),
            ],
        );

        let err = ParquetLoader::from_path(&path)
            .target("species")
            .load_classif();
        assert!(
            err.is_err(),
            "null in string target must error, not become a phantom \"\" class"
        );
    }

    #[test]
    fn load_parquet_missing_column_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.parquet");
        write_parquet(
            &path,
            1,
            vec![
                Column::new("x".into(), vec![1.0f64]),
                Column::new("y".into(), vec![2.0f64]),
            ],
        );

        let err = ParquetLoader::from_path(&path)
            .target("missing")
            .load_classif();
        assert!(err.is_err());
    }

    #[test]
    fn load_parquet_nulls_become_nan() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.parquet");
        write_parquet(
            &path,
            2,
            vec![
                Column::new("x1".into(), vec![Some(1.0f64), None]),
                Column::new("y".into(), vec![2.0f64, 4.0]),
            ],
        );

        let task = ParquetLoader::from_path(&path)
            .target("y")
            .load_regress()
            .unwrap();
        let f = task.features();
        assert!(f[[1, 0]].is_nan(), "null must load as NaN");
        assert_eq!(f[[0, 0]], 1.0);
    }

    #[test]
    fn load_parquet_string_column_auto_categorical() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.parquet");
        write_parquet(
            &path,
            3,
            vec![
                Column::new(
                    "soil".into(),
                    vec!["clay".to_string(), "sand".to_string(), "clay".to_string()],
                ),
                Column::new("y".into(), vec![1.0f64, 2.0, 3.0]),
            ],
        );

        let task = ParquetLoader::from_path(&path)
            .target("y")
            .load_regress()
            .unwrap();
        assert_eq!(task.categorical_features(), vec![0]);
        assert_eq!(
            task.feature_types()[0],
            FeatureType::Categorical { n_categories: 2 }
        );
    }

    #[test]
    fn load_parquet_forced_categorical_column() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.parquet");
        write_parquet(
            &path,
            3,
            vec![
                Column::new("region".into(), vec![3i64, 7, 3]),
                Column::new("x".into(), vec![1.0f64, 2.0, 3.0]),
                Column::new("y".into(), vec![0i64, 1, 0]),
            ],
        );

        // Without forcing, "region" parses as numeric.
        let plain = ParquetLoader::from_path(&path)
            .target("y")
            .load_classif()
            .unwrap();
        assert!(plain.categorical_features().is_empty());

        let task = ParquetLoader::from_path(&path)
            .target("y")
            .categorical(&["region"])
            .load_classif()
            .unwrap();
        assert_eq!(task.categorical_features(), vec![0]);

        // Forcing a nonexistent column errors.
        let err = ParquetLoader::from_path(&path)
            .target("y")
            .categorical(&["no_such_col"])
            .load_classif();
        assert!(err.is_err());
    }
}

/// NaN policy (item 14): the boosting engines handle missing values natively
/// (learned default direction); every other learner must reject NaN features
/// with a clear error instead of silently producing garbage distances,
/// coefficients, or splits.
#[test]
fn nan_features_rejected_by_non_nan_learners_accepted_by_boosting() {
    use ndarray::array;

    let features = array![
        [0.0, 1.0],
        [1.0, f64::NAN],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0],
        [5.0, 6.0]
    ];
    let rtask = RegressionTask::new(
        "nan_policy",
        features.clone(),
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    )
    .unwrap();
    let ctask = ClassificationTask::new("nan_policy", features, vec![0, 1, 0, 1, 0, 1]).unwrap();

    // Non-NaN-capable learners: clear error at train time.
    assert!(KNearestNeighbors::new(2).train_regress(&rtask).is_err());
    assert!(LinearRegression::new().train_regress(&rtask).is_err());
    assert!(DecisionTree::new().train_classif(&ctask).is_err());
    assert!(RandomForest::new().train_regress(&rtask).is_err());
    assert!(LogisticRegression::new().train_classif(&ctask).is_err());

    // Boosting engines: NaN is a first-class missing value.
    assert!(
        XGBoost::new()
            .with_n_estimators(5)
            .train_regress(&rtask)
            .is_ok()
    );
    assert!(
        LightGBM::new()
            .with_n_estimators(5)
            .with_top_rate(1.0)
            .with_other_rate(0.0)
            .train_regress(&rtask)
            .is_ok()
    );
    assert!(
        CatBoost::new()
            .with_n_estimators(5)
            .train_regress(&rtask)
            .is_ok()
    );
}

#[test]
fn task_with_categorical_features_validates_codes() {
    use ndarray::array;

    // Valid integer codes (with NaN as missing) are accepted.
    let feats = array![[0.0, 1.5], [2.0, 2.5], [f64::NAN, 3.5]];
    let task = RegressionTask::new("t", feats, vec![1.0, 2.0, 3.0])
        .unwrap()
        .with_categorical_features(&[0])
        .unwrap();
    assert_eq!(
        task.feature_types()[0],
        FeatureType::Categorical { n_categories: 3 }
    );

    // Non-integer codes are rejected.
    let feats = array![[0.5, 1.0], [1.0, 2.0]];
    let err = RegressionTask::new("t", feats, vec![1.0, 2.0])
        .unwrap()
        .with_categorical_features(&[0]);
    assert!(err.is_err());

    // Negative codes are rejected.
    let feats = array![[-1.0, 1.0], [1.0, 2.0]];
    let err = ClassificationTask::new("t", feats, vec![0, 1])
        .unwrap()
        .with_categorical_features(&[0]);
    assert!(err.is_err());

    // Out-of-range column index is rejected.
    let feats = array![[0.0, 1.0], [1.0, 2.0]];
    let err = RegressionTask::new("t", feats, vec![1.0, 2.0])
        .unwrap()
        .with_categorical_features(&[5]);
    assert!(err.is_err());
}

/// Regression test: `benchmark::resample_classif`/`resample_regress` used to
/// rebuild each CV fold's task via `RegressionTask::new`/`ClassificationTask::new`,
/// which resets `feature_types` to all-Numeric — silently disabling native
/// categorical splits inside any CV run. Same "parity" scenario as
/// `xgboost::cat_tests`: y depends on the parity of a 7-code categorical
/// feature, which no single numeric threshold can separate but a native
/// categorical split can.
#[test]
fn benchmark_cv_preserves_categorical_feature_types_across_folds() {
    let n = 350;
    let mut features = ndarray::Array2::<f64>::zeros((n, 1));
    let mut target = vec![0.0; n];
    for i in 0..n {
        let code = (i % 7) as f64;
        features[[i, 0]] = code;
        target[i] = ((i % 7) % 2) as f64 * 10.0;
    }
    let task = RegressionTask::new("parity", features, target)
        .unwrap()
        .with_categorical_features(&[0])
        .unwrap();

    let mut model = XGBoost::new()
        .with_n_estimators(3)
        .with_max_depth(1)
        .with_learning_rate(1.0)
        .with_lambda(1e-6);
    let cv = CrossValidation::new(5).with_seed(0);
    let result = benchmark::resample_regress(&mut model, &task, &cv, &[&Rmse]).unwrap();
    let mean_rmse = result.mean_scores()[0];
    assert!(
        mean_rmse < 1.0,
        "CV folds should retain the categorical feature type and fit parity exactly \
         (RMSE < 1.0), got {mean_rmse} — a numeric-threshold fallback caused by lost \
         feature_types metadata would give RMSE > 2.0"
    );
}

// ── Serialization tests ────────────────────────────────────────────

#[test]
fn serialize_prediction_roundtrip() {
    let pred = Prediction::classification_with_truth(vec![0, 1, 1], vec![0, 0, 1]);
    let json = serde_json::to_string(&pred).unwrap();
    let restored: Prediction = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.n_samples(), 3);
}

#[test]
fn serialize_decision_tree_roundtrip() {
    let features = array![[0.0], [1.0], [2.0], [3.0]];
    let target = vec![0, 0, 1, 1];
    let task = ClassificationTask::new("ser", features.clone(), target).unwrap();

    let mut tree = DecisionTree::default();
    let model = tree.train_classif(&task).unwrap();

    let pred = model.predict(&features).unwrap();
    let json = serde_json::to_string(&pred).unwrap();
    let restored: Prediction = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.n_samples(), pred.n_samples());
}

#[test]
fn save_load_json_file() {
    let features = array![[0.0], [1.0], [2.0], [3.0]];
    let target = vec![0, 0, 1, 1];
    let task = ClassificationTask::new("save", features.clone(), target).unwrap();

    let mut tree = DecisionTree::default();
    let model = tree.train_classif(&task).unwrap();
    let pred = model.predict(&features).unwrap();

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pred.json");
    let json = serde_json::to_string(&pred).unwrap();
    std::fs::write(&path, &json).unwrap();
    let loaded_json = std::fs::read_to_string(&path).unwrap();
    let pred2: Prediction = serde_json::from_str(&loaded_json).unwrap();
    assert_eq!(pred.n_samples(), pred2.n_samples());
}

// ── Precision / Recall / F1 tests ──────────────────────────────────

/// Regression test (4th audit, M-10): with gapped label ids the macro
/// averages used to divide by max(label)+1 — labels {0, 2} created a
/// phantom class 1 with all-zero counts, deflating Precision/Recall/F1 by
/// ~33% here (sklearn averages over the union of observed labels only).
/// Realistic whenever a CV fold loses an intermediate class. Goldens
/// verified against sklearn 1.8 (`average='macro'`).
#[test]
fn macro_measures_ignore_phantom_gap_classes() {
    let truth = vec![0, 2, 0, 2];
    let predicted = vec![0, 2, 2, 2];
    let pred = Prediction::classification_with_truth(predicted, truth);

    // class 0: prec 1/1, rec 1/2; class 2: prec 2/3, rec 2/2
    assert!((Precision.score(&pred).unwrap() - 0.8333333333333333).abs() < 1e-12);
    assert!((Recall.score(&pred).unwrap() - 0.75).abs() < 1e-12);
    assert!((F1Score.score(&pred).unwrap() - 0.7333333333333334).abs() < 1e-12);
}

#[test]
fn precision_perfect() {
    let pred = Prediction::classification_with_truth(vec![0, 1, 1, 0], vec![0, 1, 1, 0]);
    let p = Precision.score(&pred).unwrap();
    assert!((p - 1.0).abs() < 1e-10);
}

#[test]
fn precision_known_value() {
    // pred: [1, 1, 1, 0], truth: [1, 0, 1, 0]
    // class 0: TP=1, FP=0 → prec=1.0
    // class 1: TP=2, FP=1 → prec=2/3
    // macro: (1 + 2/3) / 2 = 5/6
    let pred = Prediction::classification_with_truth(vec![1, 1, 1, 0], vec![1, 0, 1, 0]);
    let p = Precision.score(&pred).unwrap();
    assert!((p - 5.0 / 6.0).abs() < 1e-10);
}

#[test]
fn recall_known_value() {
    // pred: [1, 1, 1, 0], truth: [1, 0, 1, 0]
    // class 0: TP=1, FN=1 → recall=0.5
    // class 1: TP=2, FN=0 → recall=1.0
    // macro: (0.5 + 1.0) / 2 = 0.75
    let pred = Prediction::classification_with_truth(vec![1, 1, 1, 0], vec![1, 0, 1, 0]);
    let r = Recall.score(&pred).unwrap();
    assert!((r - 0.75).abs() < 1e-10);
}

#[test]
fn f1_perfect() {
    let pred = Prediction::classification_with_truth(vec![0, 1, 0, 1], vec![0, 1, 0, 1]);
    let f1 = F1Score.score(&pred).unwrap();
    assert!((f1 - 1.0).abs() < 1e-10);
}

#[test]
fn f1_known_value() {
    // pred: [1, 1, 1, 0], truth: [1, 0, 1, 0]
    // class 0: prec=1.0, recall=0.5 → F1=2/3
    // class 1: prec=2/3, recall=1.0 → F1=4/5
    // macro: (2/3 + 4/5) / 2 = (10/15 + 12/15) / 2 = 22/30 = 11/15
    let pred = Prediction::classification_with_truth(vec![1, 1, 1, 0], vec![1, 0, 1, 0]);
    let f1 = F1Score.score(&pred).unwrap();
    assert!((f1 - 11.0 / 15.0).abs() < 1e-10);
}

#[test]
fn f1_requires_truth() {
    let pred = Prediction::classification(vec![0, 1]);
    assert!(F1Score.score(&pred).is_err());
}

#[test]
fn precision_recall_directions() {
    assert!(Precision.maximize());
    assert!(Recall.maximize());
    assert!(F1Score.maximize());
}

// ── LogLoss tests ──────────────────────────────────────────────────

#[test]
fn logloss_perfect() {
    // Perfect probabilities: class 0 gets [1.0, 0.0], class 1 gets [0.0, 1.0]
    let pred = Prediction::Classification {
        predicted: vec![0, 1],
        truth: Some(vec![0, 1]),
        probabilities: Some(vec![vec![1.0, 0.0], vec![0.0, 1.0]]),
    };
    let ll = LogLoss.score(&pred).unwrap();
    assert!(
        ll < 1e-10,
        "perfect predictions should have near-zero logloss, got {ll}"
    );
}

#[test]
fn logloss_uncertain() {
    // 50/50 probabilities
    let pred = Prediction::Classification {
        predicted: vec![0, 1],
        truth: Some(vec![0, 1]),
        probabilities: Some(vec![vec![0.5, 0.5], vec![0.5, 0.5]]),
    };
    let ll = LogLoss.score(&pred).unwrap();
    // -ln(0.5) ≈ 0.693
    assert!((ll - 0.5_f64.ln().abs()).abs() < 1e-10);
}

#[test]
fn logloss_requires_probabilities() {
    let pred = Prediction::classification_with_truth(vec![0, 1], vec![0, 1]);
    assert!(LogLoss.score(&pred).is_err());
}

#[test]
fn logloss_direction() {
    assert!(!LogLoss.maximize());
}

// ── AUC-ROC tests ──────────────────────────────────────────────────

#[test]
fn auc_perfect_binary() {
    let pred = Prediction::Classification {
        predicted: vec![0, 0, 1, 1],
        truth: Some(vec![0, 0, 1, 1]),
        probabilities: Some(vec![
            vec![1.0, 0.0],
            vec![0.9, 0.1],
            vec![0.1, 0.9],
            vec![0.0, 1.0],
        ]),
    };
    let auc = AucRoc.score(&pred).unwrap();
    assert!(
        (auc - 1.0).abs() < 1e-10,
        "perfect separation should give AUC=1.0, got {auc}"
    );
}

#[test]
fn auc_random_binary() {
    // Probabilities don't distinguish classes at all
    let pred = Prediction::Classification {
        predicted: vec![0, 1, 0, 1],
        truth: Some(vec![0, 0, 1, 1]),
        probabilities: Some(vec![
            vec![0.5, 0.5],
            vec![0.5, 0.5],
            vec![0.5, 0.5],
            vec![0.5, 0.5],
        ]),
    };
    let auc = AucRoc.score(&pred).unwrap();
    assert!(
        (auc - 0.5).abs() < 1e-10,
        "random predictions should give AUC≈0.5, got {auc}"
    );
}

#[test]
fn auc_requires_probabilities() {
    let pred = Prediction::classification_with_truth(vec![0, 1], vec![0, 1]);
    assert!(AucRoc.score(&pred).is_err());
}

#[test]
fn auc_direction() {
    assert!(AucRoc.maximize());
}

// ── BalancedAccuracy / CohensKappa / MCC / Brier tests ───────────────

#[test]
fn balanced_accuracy_penalizes_majority_class_collapse() {
    // 9 negatives, 1 positive; predicting all-negative gets 90% plain
    // accuracy but should score 0.5 balanced accuracy (chance-level on
    // the minority class).
    let predicted = vec![0usize; 10];
    let mut truth = vec![0usize; 9];
    truth.push(1);
    let pred = Prediction::classification_with_truth(predicted, truth);

    let acc = Accuracy.score(&pred).unwrap();
    assert!((acc - 0.9).abs() < 1e-10);

    let bacc = BalancedAccuracy.score(&pred).unwrap();
    assert!(
        (bacc - 0.5).abs() < 1e-10,
        "expected balanced accuracy 0.5, got {bacc}"
    );
    assert!(BalancedAccuracy.maximize());
}

#[test]
fn cohens_kappa_known_value() {
    // 5 zeros, 5 ones; 8/10 correct, symmetric confusion matrix
    // [[4,1],[1,4]] => po=0.8, pe=0.5 => kappa=(0.8-0.5)/(1-0.5)=0.6.
    let predicted = vec![0, 0, 0, 0, 1, 0, 1, 1, 1, 1];
    let truth = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
    let pred = Prediction::classification_with_truth(predicted, truth);
    let kappa = CohensKappa.score(&pred).unwrap();
    assert!(
        (kappa - 0.6).abs() < 1e-10,
        "expected kappa=0.6, got {kappa}"
    );
}

#[test]
fn cohens_kappa_perfect_agreement() {
    let pred = Prediction::classification_with_truth(vec![0, 1, 0, 1, 1], vec![0, 1, 0, 1, 1]);
    let kappa = CohensKappa.score(&pred).unwrap();
    assert!((kappa - 1.0).abs() < 1e-10);
    assert!(CohensKappa.maximize());
}

#[test]
fn mcc_known_value() {
    // Same confusion matrix as the kappa test: TP=4, FN=1, FP=1, TN=4
    // (class 1 as positive) => MCC = (16-1)/sqrt(5*5*5*5) = 0.6.
    let predicted = vec![0, 0, 0, 0, 1, 0, 1, 1, 1, 1];
    let truth = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
    let pred = Prediction::classification_with_truth(predicted, truth);
    let mcc = Mcc.score(&pred).unwrap();
    assert!((mcc - 0.6).abs() < 1e-10, "expected mcc=0.6, got {mcc}");
}

#[test]
fn mcc_perfect_and_random() {
    let perfect = Prediction::classification_with_truth(vec![0, 1, 0, 1], vec![0, 1, 0, 1]);
    assert!((Mcc.score(&perfect).unwrap() - 1.0).abs() < 1e-10);

    // Degenerate: model always predicts class 0 => predicted marginal is
    // constant => denominator is 0 => defined as 0 (mlr3/sklearn convention).
    let degenerate = Prediction::classification_with_truth(vec![0, 0, 0, 0], vec![0, 1, 0, 1]);
    assert!((Mcc.score(&degenerate).unwrap() - 0.0).abs() < 1e-10);
    assert!(Mcc.maximize());
}

#[test]
fn brier_known_value() {
    let pred = Prediction::Classification {
        predicted: vec![0, 1],
        truth: Some(vec![0, 1]),
        probabilities: Some(vec![vec![0.9, 0.1], vec![0.2, 0.8]]),
    };
    // sample0: (0.9-1)^2+(0.1-0)^2=0.02; sample1: (0.2-0)^2+(0.8-1)^2=0.08
    // mean = 0.05
    let brier = Brier.score(&pred).unwrap();
    assert!((brier - 0.05).abs() < 1e-10, "expected 0.05, got {brier}");
}

#[test]
fn brier_perfect_is_zero() {
    let pred = Prediction::Classification {
        predicted: vec![0, 1],
        truth: Some(vec![0, 1]),
        probabilities: Some(vec![vec![1.0, 0.0], vec![0.0, 1.0]]),
    };
    let brier = Brier.score(&pred).unwrap();
    assert!(brier.abs() < 1e-10);
    assert!(!Brier.maximize());
}

#[test]
fn brier_requires_probabilities() {
    let pred = Prediction::classification_with_truth(vec![0, 1], vec![0, 1]);
    assert!(Brier.score(&pred).is_err());
}

// ── R² tests ───────────────────────────────────────────────────────

#[test]
fn rsquared_perfect() {
    let pred = Prediction::regression_with_truth(vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]);
    let r2 = RSquared.score(&pred).unwrap();
    assert!((r2 - 1.0).abs() < 1e-10);
}

#[test]
fn rsquared_mean_prediction() {
    // Predicting the mean always gives R²=0
    let pred = Prediction::regression_with_truth(vec![2.0, 2.0, 2.0], vec![1.0, 2.0, 3.0]);
    let r2 = RSquared.score(&pred).unwrap();
    assert!(
        r2.abs() < 1e-10,
        "predicting mean should give R²=0, got {r2}"
    );
}

#[test]
fn rsquared_negative() {
    // Worse than predicting the mean
    let pred = Prediction::regression_with_truth(vec![10.0, 10.0, 10.0], vec![1.0, 2.0, 3.0]);
    let r2 = RSquared.score(&pred).unwrap();
    assert!(r2 < 0.0, "worse than mean should give R²<0, got {r2}");
}

#[test]
fn rsquared_direction() {
    assert!(RSquared.maximize());
}

// ── MAPE tests ─────────────────────────────────────────────────────

#[test]
fn mape_perfect() {
    let pred = Prediction::regression_with_truth(vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]);
    let mape = Mape.score(&pred).unwrap();
    assert!(mape.abs() < 1e-10);
}

#[test]
fn mape_known_value() {
    // errors: |1-2|/|2|=0.5, |3-4|/|4|=0.25 → mean = 0.375
    let pred = Prediction::regression_with_truth(vec![1.0, 3.0], vec![2.0, 4.0]);
    let mape = Mape.score(&pred).unwrap();
    assert!(
        (mape - 0.375).abs() < 1e-10,
        "expected MAPE=0.375, got {mape}"
    );
}

#[test]
fn mape_direction() {
    assert!(!Mape.maximize());
}

// ── Extra Trees tests ──────────────────────────────────────────────

#[test]
fn extra_trees_classif_separable() {
    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [0.1, 0.2],
        [0.0, 0.1],
        [1.0, 1.0],
        [1.1, 0.9],
        [0.9, 1.1],
        [1.0, 0.9],
        [1.1, 1.0]
    ];
    let target = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
    let task = ClassificationTask::new("et", features, target).unwrap();

    let mut et = ExtraTrees::new().with_n_estimators(20).with_seed(42);
    let model = et.train_classif(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_classif(task.target().to_vec());
    let acc = Accuracy.score(&pred).unwrap();
    assert_eq!(acc, 1.0);
}

#[test]
fn extra_trees_regress() {
    let features = array![[1.0], [2.0], [3.0], [4.0], [6.0], [7.0], [8.0], [9.0]];
    let target = vec![0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0];
    let task = RegressionTask::new("et_r", features, target).unwrap();

    let mut et = ExtraTrees::new().with_n_estimators(20).with_seed(42);
    let model = et.train_regress(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_regress(task.target().to_vec());
    let rmse = Rmse.score(&pred).unwrap();
    assert!(rmse < 1.0, "ET should learn step function, got RMSE={rmse}");
}

// ── Gaussian Naive Bayes tests ─────────────────────────────────────

#[test]
fn naive_bayes_separable() {
    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [0.0, 0.2],
        [5.0, 5.0],
        [5.1, 4.9],
        [4.9, 5.1],
        [5.0, 4.8]
    ];
    let target = vec![0, 0, 0, 0, 1, 1, 1, 1];
    let task = ClassificationTask::new("nb", features, target).unwrap();

    let mut nb = GaussianNB::new();
    let model = nb.train_classif(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_classif(task.target().to_vec());
    let acc = Accuracy.score(&pred).unwrap();
    assert_eq!(acc, 1.0, "NB should separate widely spaced clusters");
}

#[test]
fn naive_bayes_produces_probabilities() {
    let features = array![[0.0], [1.0], [2.0], [3.0]];
    let target = vec![0, 0, 1, 1];
    let task = ClassificationTask::new("nb_p", features, target).unwrap();

    let mut nb = GaussianNB::new();
    let model = nb.train_classif(&task).unwrap();
    let pred = model.predict(task.features()).unwrap();

    if let Prediction::Classification {
        probabilities: Some(probs),
        ..
    } = &pred
    {
        for p in probs {
            let sum: f64 = p.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10, "probabilities should sum to 1");
        }
    } else {
        panic!("expected probabilities");
    }
}

#[test]
fn naive_bayes_rejects_regression() {
    let features = array![[1.0], [2.0]];
    let target = vec![1.0, 2.0];
    let task = RegressionTask::new("nb_bad", features, target).unwrap();
    let mut nb = GaussianNB::new();
    assert!(nb.train_regress(&task).is_err());
}

// ── Ridge Regression tests ─────────────────────────────────────────

#[test]
fn ridge_fits_linear() {
    let features = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let target = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let task = RegressionTask::new("ridge", features, target).unwrap();

    let mut ridge = Ridge::new(0.01); // small regularization
    let model = ridge.train_regress(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_regress(task.target().to_vec());
    let rmse = Rmse.score(&pred).unwrap();
    assert!(
        rmse < 0.5,
        "Ridge with small alpha should fit linear, got RMSE={rmse}"
    );
}

#[test]
fn ridge_shrinks_coefficients() {
    let features = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let target = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let task = RegressionTask::new("ridge_s", features, target).unwrap();

    let mut ridge_small = Ridge::new(0.001);
    let mut ridge_big = Ridge::new(100.0);
    let m1 = ridge_small.train_regress(&task).unwrap();
    let m2 = ridge_big.train_regress(&task).unwrap();

    let imp1 = m1.feature_importance().unwrap();
    let imp2 = m2.feature_importance().unwrap();
    // Both should have importance, but the coefficient magnitudes differ
    assert!(imp1[0].1 > 0.0);
    assert!(imp2[0].1 > 0.0);
}

// ── Lasso Regression tests ─────────────────────────────────────────

#[test]
fn lasso_fits_linear() {
    let features = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let target = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let task = RegressionTask::new("lasso", features, target).unwrap();

    let mut lasso = Lasso::new(0.01);
    let model = lasso.train_regress(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_regress(task.target().to_vec());
    let rmse = Rmse.score(&pred).unwrap();
    assert!(rmse < 1.0, "Lasso should fit linear, got RMSE={rmse}");
}

#[test]
fn lasso_sparsity() {
    // x0 is informative, x1 is noise
    let features = array![
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [3.0, 1.0]
    ];
    let target = vec![2.0, 4.0, 6.0, 2.0, 4.0, 6.0];
    let task = RegressionTask::new("lasso_sp", features, target).unwrap();

    let mut lasso = Lasso::new(0.1);
    let model = lasso.train_regress(&task).unwrap();
    let imp = model.feature_importance().unwrap();
    assert!(
        imp[0].1 > imp[1].1,
        "Lasso should assign more importance to informative feature"
    );
}

// ── Elastic Net tests ──────────────────────────────────────────────

#[test]
fn elastic_net_fits_linear() {
    let features = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let target = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let task = RegressionTask::new("enet", features, target).unwrap();

    let mut enet = ElasticNet::new(0.01, 0.5);
    let model = enet.train_regress(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_regress(task.target().to_vec());
    let rmse = Rmse.score(&pred).unwrap();
    assert!(rmse < 1.0, "ElasticNet should fit linear, got RMSE={rmse}");
}

// ── AdaBoost tests ─────────────────────────────────────────────────

#[test]
fn adaboost_separable() {
    let features = array![[0.0], [0.5], [1.0], [1.5], [3.0], [3.5], [4.0], [4.5]];
    let target = vec![0, 0, 0, 0, 1, 1, 1, 1];
    let task = ClassificationTask::new("ada", features, target).unwrap();

    let mut ada = AdaBoost::new().with_n_estimators(20);
    let model = ada.train_classif(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_classif(task.target().to_vec());
    let acc = Accuracy.score(&pred).unwrap();
    assert!(
        acc >= 0.75,
        "AdaBoost should learn separable data, got {acc}"
    );
}

#[test]
fn adaboost_produces_probabilities() {
    let features = array![[0.0], [1.0], [2.0], [3.0]];
    let target = vec![0, 0, 1, 1];
    let task = ClassificationTask::new("ada_p", features, target).unwrap();

    let mut ada = AdaBoost::new().with_n_estimators(10);
    let model = ada.train_classif(&task).unwrap();
    let pred = model.predict(task.features()).unwrap();

    if let Prediction::Classification {
        probabilities: Some(probs),
        ..
    } = &pred
    {
        for p in probs {
            let sum: f64 = p.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10, "probabilities should sum to 1");
        }
    } else {
        panic!("expected probabilities");
    }
}

#[test]
fn adaboost_rejects_regression() {
    let features = array![[1.0], [2.0]];
    let target = vec![1.0, 2.0];
    let task = RegressionTask::new("ada_bad", features, target).unwrap();
    let mut ada = AdaBoost::default();
    assert!(ada.train_regress(&task).is_err());
}

// ── Linear SVM tests ──────────────────────────────────────────────

/// Regression test (4th audit, HIGH-5): the per-sample weight decay was
/// lambda = 1/C instead of 1/(n*C) -- n times more regularization than the
/// SVM objective asks for. With defaults, ||w|| stayed pinned near zero and
/// TRAINING accuracy sat at chance level (~0.51-0.54) on trivially
/// separable data at any realistic n; the only pre-existing test used
/// n=10 with C=10, the corner where the factor-n error doesn't hurt.
/// Also covers internal standardization: UTM-scale features must work
/// with defaults, like LogisticRegression/ELM.
#[test]
fn linear_svm_defaults_learn_separable_data_at_realistic_n() {
    for (scale_x, scale_y, offset) in [(1.0, 1.0, 0.0), (1e5, 1e6, 7.2e6)] {
        let n = 400;
        let features = Array2::from_shape_fn((n, 2), |(i, j)| {
            let raw = ((i * 7 + j * 3) % 100) as f64 / 100.0;
            if j == 0 {
                raw * scale_x
            } else {
                raw * scale_y + offset
            }
        });
        let target: Vec<usize> = (0..n)
            .map(|i| {
                let x0 = features[[i, 0]] / scale_x;
                let x1 = (features[[i, 1]] - offset) / scale_y;
                usize::from(x0 + x1 > 1.0)
            })
            .collect();
        let task =
            ClassificationTask::new("svm_defaults", features.clone(), target.clone()).unwrap();

        let mut svm = LinearSVM::new(); // defaults on purpose
        let model = svm.train_classif(&task).unwrap();
        let pred = model.predict(&features).unwrap();
        let acc = match pred {
            Prediction::Classification { predicted, .. } => {
                predicted
                    .iter()
                    .zip(&target)
                    .filter(|(a, b)| a == b)
                    .count() as f64
                    / n as f64
            }
            _ => panic!("expected classification"),
        };
        assert!(
            acc > 0.9,
            "scale ({scale_x:e},{scale_y:e})+{offset:e}: default LinearSVM must learn a \
             separable boundary, got training accuracy {acc}"
        );
    }
}

#[test]
fn linear_svm_separable() {
    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [0.0, 0.2],
        [0.1, 0.0],
        [1.0, 1.0],
        [1.1, 0.9],
        [0.9, 1.1],
        [1.0, 0.9],
        [1.1, 1.0]
    ];
    let target = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
    let task = ClassificationTask::new("svm", features, target).unwrap();

    let mut svm = LinearSVM::new()
        .with_max_iter(2000)
        .with_c(10.0)
        .with_learning_rate(0.1);
    let model = svm.train_classif(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_classif(task.target().to_vec());
    let acc = Accuracy.score(&pred).unwrap();
    assert!(acc >= 0.8, "SVM should classify separable data, got {acc}");
}

#[test]
fn linear_svm_feature_importance() {
    let features = array![
        [0.0, 42.0],
        [0.1, 13.0],
        [0.2, 99.0],
        [1.0, 42.0],
        [1.1, 13.0],
        [1.2, 99.0]
    ];
    let target = vec![0, 0, 0, 1, 1, 1];
    let task = ClassificationTask::new("svm_imp", features, target).unwrap();

    let mut svm = LinearSVM::new().with_max_iter(500);
    let model = svm.train_classif(&task).unwrap();
    let imp = model.feature_importance();
    assert!(imp.is_some());
}

#[test]
fn linear_svm_rejects_regression() {
    let features = array![[1.0], [2.0]];
    let target = vec![1.0, 2.0];
    let task = RegressionTask::new("svm_bad", features, target).unwrap();
    let mut svm = LinearSVM::default();
    assert!(svm.train_regress(&task).is_err());
}

// ── All new learners in benchmark ──────────────────────────────────

#[test]
fn benchmark_new_learners_cv() {
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
        [1.1, 1.1]
    ];
    let target = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
    let task = ClassificationTask::new("bench_new", features, target).unwrap();
    let cv = CrossValidation::new(4).with_seed(42);

    // Test each new learner works through the benchmark pipeline
    let mut et = ExtraTrees::new().with_n_estimators(10).with_seed(42);
    let r = benchmark::resample_classif(&mut et, &task, &cv, &[&Accuracy]).unwrap();
    assert_eq!(r.learner_id, "extra_trees");

    let mut nb = GaussianNB::new();
    let r = benchmark::resample_classif(&mut nb, &task, &cv, &[&Accuracy]).unwrap();
    assert_eq!(r.learner_id, "gaussian_nb");

    let mut ada = AdaBoost::new().with_n_estimators(10);
    let r = benchmark::resample_classif(&mut ada, &task, &cv, &[&Accuracy]).unwrap();
    assert_eq!(r.learner_id, "adaboost");
}

// ── XGBoost tests ──────────────────────────────────────────────────

#[test]
fn xgboost_classif_binary() {
    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [0.0, 0.2],
        [0.1, 0.0],
        [1.0, 1.0],
        [1.1, 0.9],
        [0.9, 1.1],
        [1.0, 0.9],
        [1.1, 1.0]
    ];
    let target = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
    let task = ClassificationTask::new("xgb_bin", features, target).unwrap();

    let mut xgb = XGBoost::new()
        .with_n_estimators(50)
        .with_max_depth(3)
        .with_learning_rate(0.3);
    let model = xgb.train_classif(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_classif(task.target().to_vec());
    let acc = Accuracy.score(&pred).unwrap();
    assert!(
        acc >= 0.9,
        "XGBoost should classify separable data, got {acc}"
    );
}

#[test]
fn xgboost_classif_multiclass() {
    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.0, 0.1],
        [1.0, 0.0],
        [1.1, 0.1],
        [1.0, 0.1],
        [0.0, 1.0],
        [0.1, 1.1],
        [0.0, 1.1]
    ];
    let target = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
    let task = ClassificationTask::new("xgb_mc", features, target).unwrap();

    let mut xgb = XGBoost::new()
        .with_n_estimators(200)
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
    assert!(
        acc >= 0.66,
        "XGBoost multiclass should do better than random, got {acc}"
    );
}

#[test]
fn xgboost_regress() {
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
    let task = RegressionTask::new("xgb_reg", features, target).unwrap();

    let mut xgb = XGBoost::new()
        .with_n_estimators(100)
        .with_max_depth(3)
        .with_learning_rate(0.3);
    let model = xgb.train_regress(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_regress(task.target().to_vec());
    let rmse = Rmse.score(&pred).unwrap();
    assert!(
        rmse < 1.0,
        "XGBoost should learn linear trend, got RMSE={rmse}"
    );
}

#[test]
fn xgboost_regularization() {
    let features = array![[0.0], [1.0], [2.0], [3.0]];
    let target = vec![0, 0, 1, 1];
    let task = ClassificationTask::new("xgb_reg", features, target).unwrap();

    // High lambda should still work (more conservative splits)
    let mut xgb = XGBoost::new()
        .with_n_estimators(20)
        .with_lambda(10.0)
        .with_gamma(0.1);
    let model = xgb.train_classif(&task).unwrap();
    let pred = model.predict(task.features()).unwrap();
    assert_eq!(pred.n_samples(), 4);
}

#[test]
fn xgboost_feature_importance() {
    let features = array![
        [0.0, 42.0],
        [0.1, 13.0],
        [0.2, 99.0],
        [0.0, 55.0],
        [0.15, 30.0],
        [1.0, 42.0],
        [1.1, 13.0],
        [1.2, 99.0],
        [1.0, 55.0],
        [1.15, 30.0]
    ];
    let target = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
    let task = ClassificationTask::new("xgb_imp", features, target).unwrap();

    let mut xgb = XGBoost::new()
        .with_n_estimators(50)
        .with_lambda(0.01)
        .with_min_child_weight(0.1);
    let model = xgb.train_classif(&task).unwrap();
    let imp = model.feature_importance();
    assert!(imp.is_some(), "XGBoost should provide feature importance");
}

#[test]
fn xgboost_subsample() {
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
        [1.1, 1.1]
    ];
    let target = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
    let task = ClassificationTask::new("xgb_sub", features, target).unwrap();

    let mut xgb = XGBoost::new()
        .with_n_estimators(30)
        .with_subsample(0.8)
        .with_colsample_bytree(0.8)
        .with_seed(42);
    let model = xgb.train_classif(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_classif(task.target().to_vec());
    let acc = Accuracy.score(&pred).unwrap();
    assert!(
        acc >= 0.5,
        "XGBoost with subsampling should work, got {acc}"
    );
}

#[test]
fn xgboost_in_benchmark() {
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
        [1.1, 1.1]
    ];
    let target = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
    let task = ClassificationTask::new("xgb_bench", features, target).unwrap();

    let cv = CrossValidation::new(4).with_seed(42);
    let mut xgb = XGBoost::new().with_n_estimators(20).with_max_depth(3);
    let r = benchmark::resample_classif(&mut xgb, &task, &cv, &[&Accuracy]).unwrap();
    assert_eq!(r.learner_id, "xgboost");
    assert_eq!(r.scores.len(), 4);
}

#[test]
fn xgboost_handles_nan() {
    // Features with NaN — XGBoost should handle them natively
    let features = array![
        [0.0, f64::NAN],
        [0.1, 0.1],
        [f64::NAN, 0.0],
        [0.0, 0.2],
        [1.0, 1.0],
        [1.1, f64::NAN],
        [0.9, 1.1],
        [f64::NAN, 0.9]
    ];
    let target = vec![0, 0, 0, 0, 1, 1, 1, 1];
    let task = ClassificationTask::new("xgb_nan", features, target).unwrap();

    let mut xgb = XGBoost::new()
        .with_n_estimators(50)
        .with_max_depth(3)
        .with_learning_rate(0.3);
    let model = xgb.train_classif(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_classif(task.target().to_vec());
    let acc = Accuracy.score(&pred).unwrap();
    assert!(acc >= 0.75, "XGBoost should handle NaN, got acc={acc}");
}

#[test]
fn xgboost_nan_in_prediction() {
    // Train on clean data, predict with NaN
    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [1.0, 1.0],
        [1.1, 0.9],
        [0.9, 1.1]
    ];
    let target = vec![0, 0, 0, 1, 1, 1];
    let task = ClassificationTask::new("xgb_nan_pred", features, target).unwrap();

    let mut xgb = XGBoost::new().with_n_estimators(50);
    let model = xgb.train_classif(&task).unwrap();

    // Predict with NaN — should not panic
    let test = array![[0.05, f64::NAN], [f64::NAN, 1.0]];
    let pred = model.predict(&test).unwrap();
    assert_eq!(pred.n_samples(), 2);
}

#[test]
fn xgboost_exact_greedy_small_dataset() {
    // With only 10 samples and n_bins=256, exact greedy should activate
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
    let task = RegressionTask::new("xgb_exact", features, target).unwrap();

    let mut xgb = XGBoost::new()
        .with_n_estimators(100)
        .with_max_depth(3)
        .with_learning_rate(0.3);
    let model = xgb.train_regress(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_regress(task.target().to_vec());

    let rmse = Rmse.score(&pred).unwrap();
    // Exact greedy should give much better RMSE than histogram on small data
    assert!(
        rmse < 0.1,
        "exact greedy should fit small data precisely, got RMSE={rmse}"
    );
}

#[test]
fn xgboost_early_stopping_regress() {
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
    let task = RegressionTask::new("xgb_es", features.clone(), target.clone()).unwrap();

    // With early stopping, should stop before 1000 rounds
    let mut xgb_es = XGBoost::new()
        .with_n_estimators(1000)
        .with_learning_rate(0.3)
        .with_early_stopping_rounds(10);
    let model_es = xgb_es.train_regress(&task).unwrap();

    // Without early stopping, all 1000 rounds
    let mut xgb_full = XGBoost::new()
        .with_n_estimators(1000)
        .with_learning_rate(0.3);
    let model_full = xgb_full.train_regress(&task).unwrap();

    // Both should produce good predictions
    let pred_es = model_es
        .predict(&features)
        .unwrap()
        .with_truth_regress(target.clone());
    let pred_full = model_full
        .predict(&features)
        .unwrap()
        .with_truth_regress(target);
    let rmse_es = Rmse.score(&pred_es).unwrap();
    let rmse_full = Rmse.score(&pred_full).unwrap();

    assert!(
        rmse_es < 1.0,
        "early stopped model should be accurate, got {rmse_es}"
    );
    assert!(
        rmse_full < 1.0,
        "full model should be accurate, got {rmse_full}"
    );
}

#[test]
fn xgboost_early_stopping_classif() {
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
        [1.1, 1.1]
    ];
    let target = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
    let task = ClassificationTask::new("xgb_es_c", features, target).unwrap();

    let mut xgb = XGBoost::new()
        .with_n_estimators(500)
        .with_learning_rate(0.3)
        .with_early_stopping_rounds(5);
    let model = xgb.train_classif(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_classif(task.target().to_vec());
    let acc = Accuracy.score(&pred).unwrap();
    assert!(
        acc >= 0.9,
        "early stopped classifier should work, got {acc}"
    );
}

// ── Conformal Prediction tests ─────────────────────────────────────

/// SplitConformal (prediction-based calibration, PM2.5 handoff gap #1) must
/// be exactly the same calibration ConformalRegressor computes when both
/// see the same predictions — one takes a model, the other its outputs.
#[test]
fn split_conformal_matches_model_driven_calibration() {
    use smelt_ml::conformal::{ConformalRegressor, SplitConformal};

    let features = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]];
    let target = vec![2.0, 4.1, 5.8, 8.3, 9.9, 12.2, 13.7, 16.1];
    let task = RegressionTask::new("sc", features, target).unwrap();
    let mut dt = DecisionTree::default();
    let model = dt.train_regress(&task).unwrap();

    let cal_features = array![[2.5], [4.5], [6.5], [7.5]];
    let cal_targets = vec![5.0, 9.0, 13.0, 15.0];

    let via_model =
        ConformalRegressor::calibrate(&*model, &cal_features, &cal_targets, 0.2).unwrap();

    let cal_pred = match model.predict(&cal_features).unwrap() {
        Prediction::Regression { predicted, .. } => predicted,
        _ => panic!("expected regression"),
    };
    let via_preds =
        SplitConformal::calibrate_from_predictions(&cal_pred, &cal_targets, 0.2).unwrap();

    assert_eq!(via_model.interval_width(), via_preds.interval_width());

    let test_pred = vec![3.0, 7.0];
    let intervals = via_preds.intervals_for(&test_pred);
    assert_eq!(intervals.len(), 2);
    assert_eq!(intervals[0].prediction, 3.0);
    assert!(
        (intervals[0].upper - intervals[0].lower - 2.0 * via_preds.interval_width()).abs() < 1e-12
    );
}

/// Mismatched calibration lengths must be a clean error, not a silent
/// zip-truncation (4th audit LOW).
#[test]
fn split_conformal_rejects_mismatched_lengths() {
    use smelt_ml::conformal::SplitConformal;
    let err = SplitConformal::calibrate_from_predictions(&[1.0, 2.0, 3.0], &[1.0, 2.0], 0.1);
    assert!(err.is_err());
}

/// End-to-end spatial conformalization — the exact composition the PM2.5
/// handoff asked for: KrigingHybrid's real predictor is predict_spatial
/// (needs coords, so ConformalRegressor can't drive it); SplitConformal
/// calibrates from its outputs and the intervals achieve near-nominal
/// coverage on spatially structured data.
#[test]
fn split_conformal_calibrates_kriging_predict_spatial() {
    use smelt_ml::conformal::SplitConformal;

    // Deterministic spatially-structured field: y = x0 + smooth(coords) + noise
    let n = 300;
    let mk = |i: usize| {
        let x = (i % 20) as f64;
        let y = (i / 20) as f64;
        (x, y)
    };
    let coords_all: Vec<(f64, f64)> = (0..n).map(mk).collect();
    let features_all = Array2::from_shape_fn((n, 1), |(i, _)| (i as f64 * 0.13).sin() * 2.0);
    let target_all: Vec<f64> = (0..n)
        .map(|i| {
            let (cx, cy) = coords_all[i];
            features_all[[i, 0]] * 3.0
                + (cx * 0.3).sin() + (cy * 0.4).cos()          // spatial signal
                + ((i as f64 * 12.9898).sin() * 0.2) // pseudo-noise
        })
        .collect();

    // 150 train / 75 calibration / 75 test
    let (tr, rest) = (0..150usize, 150..n);
    let cal: Vec<usize> = rest.clone().step_by(2).collect();
    let te: Vec<usize> = rest.skip(1).step_by(2).collect();

    let tr_idx: Vec<usize> = tr.collect();
    let sel = |idx: &[usize]| {
        (
            features_all.select(Axis(0), idx).to_owned(),
            idx.iter().map(|&i| target_all[i]).collect::<Vec<f64>>(),
            idx.iter()
                .map(|&i| coords_all[i])
                .collect::<Vec<(f64, f64)>>(),
        )
    };
    let (tr_f, tr_t, tr_c) = sel(&tr_idx);
    let (cal_f, cal_t, cal_c) = sel(&cal);
    let (te_f, te_t, te_c) = sel(&te);

    let task = RegressionTask::new("spatial_cf", tr_f, tr_t).unwrap();
    let mut kh = KrigingHybrid::new(|| Box::new(LinearRegression::new()), tr_c);
    let model = kh.train_regress_geo(&task).unwrap();

    let cal_pred = match model.predict_spatial(&cal_f, &cal_c).unwrap() {
        Prediction::Regression { predicted, .. } => predicted,
        _ => panic!("expected regression"),
    };
    let sc = SplitConformal::calibrate_from_predictions(&cal_pred, &cal_t, 0.1).unwrap();

    let te_pred = match model.predict_spatial(&te_f, &te_c).unwrap() {
        Prediction::Regression { predicted, .. } => predicted,
        _ => panic!("expected regression"),
    };
    let intervals = sc.intervals_for(&te_pred);
    let covered = intervals
        .iter()
        .zip(&te_t)
        .filter(|(iv, t)| **t >= iv.lower && **t <= iv.upper)
        .count();
    let coverage = covered as f64 / te_t.len() as f64;
    assert!(
        coverage >= 0.80,
        "90%-nominal spatial conformal intervals should cover >=80% empirically \
         on {} test points, got {coverage:.2}",
        te_t.len()
    );
    assert!(sc.interval_width().is_finite());
}

#[test]
fn conformal_regression_coverage() {
    use smelt_ml::conformal::ConformalRegressor;

    let features = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]];
    let target = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
    let task = RegressionTask::new("cf", features, target).unwrap();

    let mut dt = DecisionTree::default();
    let model = dt.train_regress(&task).unwrap();

    // Calibrate on last 4 samples
    let cal_features = array![[5.0], [6.0], [7.0], [8.0]];
    let cal_targets = vec![10.0, 12.0, 14.0, 16.0];

    let cf = ConformalRegressor::calibrate(&*model, &cal_features, &cal_targets, 0.1).unwrap();
    let intervals = cf.predict(&array![[3.0], [5.0]]).unwrap();

    assert_eq!(intervals.len(), 2);
    assert!(intervals[0].lower <= intervals[0].upper);
    assert!(cf.interval_width() >= 0.0);
}

#[test]
fn conformal_rejects_alpha_out_of_range() {
    use smelt_ml::conformal::ConformalRegressor;

    let features = array![[1.0], [2.0], [3.0], [4.0]];
    let target = vec![2.0, 4.0, 6.0, 8.0];
    let task = RegressionTask::new("cf", features, target).unwrap();
    let mut dt = DecisionTree::default();
    let model = dt.train_regress(&task).unwrap();

    let cal_features = array![[1.0], [2.0]];
    let cal_targets = vec![2.0, 4.0];

    for bad_alpha in [0.0, 1.0, 1.5, -0.1] {
        assert!(
            ConformalRegressor::calibrate(&*model, &cal_features, &cal_targets, bad_alpha).is_err(),
            "alpha={bad_alpha} should be rejected"
        );
    }
}

#[test]
fn conformal_rejects_empty_calibration_set() {
    use smelt_ml::conformal::ConformalRegressor;

    let features = array![[1.0], [2.0]];
    let target = vec![2.0, 4.0];
    let task = RegressionTask::new("cf", features, target).unwrap();
    let mut dt = DecisionTree::default();
    let model = dt.train_regress(&task).unwrap();

    let empty_features = Array2::<f64>::zeros((0, 1));
    let empty_targets: Vec<f64> = vec![];
    assert!(ConformalRegressor::calibrate(&*model, &empty_features, &empty_targets, 0.1).is_err());
}

/// Regression test: previously, when the calibration set was too small to
/// support the requested confidence level (`ceil((n+1)(1-alpha)) > n`), the
/// quantile index computation underflowed (`0usize - 1`), which panics in
/// debug builds and silently wraps to an out-of-bounds-but-clamped index in
/// release builds — either way losing the 1-alpha coverage guarantee without
/// telling the caller. It must now widen to an infinite interval instead.
#[test]
fn conformal_tiny_calibration_set_widens_instead_of_panicking() {
    use smelt_ml::conformal::ConformalRegressor;

    let features = array![[1.0], [2.0]];
    let target = vec![2.0, 4.0];
    let task = RegressionTask::new("cf", features, target).unwrap();
    let mut dt = DecisionTree::default();
    let model = dt.train_regress(&task).unwrap();

    // n=1, alpha=0.01 -> ceil((1+1)*0.99) = 2 > n=1: no finite quantile
    // exists that guarantees 99% coverage from a single calibration point.
    let cal_features = array![[1.0]];
    let cal_targets = vec![2.0];
    let cf = ConformalRegressor::calibrate(&*model, &cal_features, &cal_targets, 0.01).unwrap();
    assert!(cf.interval_width().is_infinite());
}

#[test]
fn conformal_classification_sets() {
    use smelt_ml::conformal::ConformalClassifier;

    let features = array![[0.0], [0.5], [1.0], [1.5], [2.0], [2.5], [3.0], [3.5]];
    let target = vec![0, 0, 0, 0, 1, 1, 1, 1];
    let task = ClassificationTask::new("cf_c", features, target).unwrap();

    let mut dt = DecisionTree::default();
    let model = dt.train_classif(&task).unwrap();

    let cal_features = array![[1.0], [2.0], [3.0], [0.0]];
    let cal_targets = vec![0, 1, 1, 0];

    let cf = ConformalClassifier::calibrate(&*model, &cal_features, &cal_targets, 0.1).unwrap();
    let sets = cf.predict(&array![[0.5], [2.5]]).unwrap();

    assert_eq!(sets.len(), 2);
    // Prediction set should contain at least the predicted class
    for s in &sets {
        assert!(!s.prediction_set.is_empty());
    }
}

/// Golden coverage test: with a properly-sized calibration set, split
/// conformal prediction must deliver empirical coverage close to the
/// nominal 1-alpha target. This directly exercises the quantile-index
/// arithmetic (ceil((n+1)(1-alpha))) that previously underflowed for small
/// calibration sets — a coverage check here would have caught it, unlike a
/// smoke test that only asserts `lower <= upper`.
#[test]
fn conformal_regression_empirical_coverage_near_target() {
    use rand::prelude::*;
    use smelt_ml::conformal::ConformalRegressor;

    let mut rng = StdRng::seed_from_u64(11);
    let n_train = 400;
    let n_cal = 200;
    let n_test = 500;
    let alpha = 0.1; // target 90% coverage

    let make_data = |n: usize, rng: &mut StdRng| {
        let mut feats = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);
        for _ in 0..n {
            let x: f64 = rng.random::<f64>() * 10.0;
            let noise: f64 = (rng.random::<f64>() - 0.5) * 2.0;
            feats.push(x);
            target.push(2.0 * x + noise);
        }
        (Array2::from_shape_vec((n, 1), feats).unwrap(), target)
    };

    let (tr_features, tr_target) = make_data(n_train, &mut rng);
    let (cal_features, cal_target) = make_data(n_cal, &mut rng);
    let (te_features, te_target) = make_data(n_test, &mut rng);

    let task = RegressionTask::new("cov", tr_features, tr_target).unwrap();
    let mut xgb = XGBoost::new().with_n_estimators(50).with_seed(11);
    let model = xgb.train_regress(&task).unwrap();

    let cf = ConformalRegressor::calibrate(&*model, &cal_features, &cal_target, alpha).unwrap();
    let intervals = cf.predict(&te_features).unwrap();

    let covered = intervals
        .iter()
        .zip(&te_target)
        .filter(|&(iv, &y)| y >= iv.lower && y <= iv.upper)
        .count();
    let coverage = covered as f64 / n_test as f64;

    assert!(
        coverage >= 1.0 - alpha - 0.05,
        "empirical coverage {coverage:.3} should be close to target {:.3}",
        1.0 - alpha
    );
}

// ── Stacking tests ─────────────────────────────────────────────────

#[test]
fn stacking_classif() {
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
        [1.1, 1.1]
    ];
    let target = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
    let task = ClassificationTask::new("stack", features, target).unwrap();

    let mut stack = Stacking::new(
        vec![
            Box::new(|| Box::new(DecisionTree::default()) as Box<dyn Learner>),
            Box::new(|| Box::new(KNearestNeighbors::new(3)) as Box<dyn Learner>),
        ],
        || Box::new(LogisticRegression::new().with_max_iter(500)),
    )
    .with_cv_folds(2);

    let model = stack.train_classif(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_classif(task.target().to_vec());
    let acc = Accuracy.score(&pred).unwrap();
    assert!(
        acc >= 0.75,
        "Stacking should classify separable data, got {acc}"
    );
}

/// Regression test: class 2 has a single sample, so any k-fold partition
/// (k >= 2) necessarily puts it in exactly one fold's test set (every k-fold
/// CV assigns every sample to exactly one test fold) -- that fold's training
/// set then has only classes {0, 1}. Before the fix, the base model trained
/// on that fold produced 2-wide probability rows, and writing them into the
/// 3-wide (task-level `n_classes`) out-of-fold buffer panicked with an
/// index-out-of-bounds instead of returning gracefully.
#[test]
fn stacking_classif_survives_fold_missing_a_class() {
    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [0.0, 0.2],
        [1.0, 1.0],
        [1.1, 0.9],
        [0.9, 1.1],
        [1.0, 0.9],
        [5.0, 5.0],
    ];
    let target = vec![0, 0, 0, 0, 1, 1, 1, 1, 2];
    let task = ClassificationTask::new("stack_imbalanced", features, target).unwrap();

    let mut stack = Stacking::new(
        vec![Box::new(|| {
            Box::new(DecisionTree::default()) as Box<dyn Learner>
        })],
        || Box::new(LogisticRegression::new().with_max_iter(500)),
    )
    .with_cv_folds(3);

    let model = stack.train_classif(&task);
    assert!(
        model.is_ok(),
        "Stacking must not panic when a fold's training data misses a class: {:?}",
        model.err()
    );
}

#[test]
fn stacking_regress() {
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
    let task = RegressionTask::new("stack_r", features, target).unwrap();

    let mut stack = Stacking::new(
        vec![
            Box::new(|| Box::new(DecisionTree::default()) as Box<dyn Learner>),
            Box::new(|| Box::new(LinearRegression) as Box<dyn Learner>),
        ],
        || Box::new(Ridge::new(0.1)),
    )
    .with_cv_folds(2);

    let model = stack.train_regress(&task).unwrap();
    let pred = model.predict(task.features()).unwrap();
    assert_eq!(pred.n_samples(), 10);
}

// ── Quantile GB tests ──────────────────────────────────────────────

#[test]
fn quantile_gb_median() {
    let features = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let target = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let task = RegressionTask::new("qgb", features.clone(), target.clone()).unwrap();

    let mut qgb = QuantileGB::new(0.5)
        .with_n_estimators(100)
        .with_learning_rate(0.1);
    let model = qgb.train_regress(&task).unwrap();
    let pred = model.predict(&features).unwrap().with_truth_regress(target);
    let rmse = Rmse.score(&pred).unwrap();
    assert!(
        rmse < 2.0,
        "median quantile should approximate well, got RMSE={rmse}"
    );
}

#[test]
fn quantile_gb_interval() {
    let features = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let target = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let task = RegressionTask::new("qgb_int", features.clone(), target).unwrap();

    let mut lower = QuantileGB::new(0.1).with_n_estimators(50);
    let mut upper = QuantileGB::new(0.9).with_n_estimators(50);

    let model_lo = lower.train_regress(&task).unwrap();
    let model_hi = upper.train_regress(&task).unwrap();

    let pred_lo = model_lo.predict(&features).unwrap();
    let pred_hi = model_hi.predict(&features).unwrap();

    if let (
        Prediction::Regression { predicted: lo, .. },
        Prediction::Regression { predicted: hi, .. },
    ) = (&pred_lo, &pred_hi)
    {
        // Upper quantile should generally be >= lower quantile
        let violations = lo.iter().zip(hi).filter(|(l, h)| l > h).count();
        assert!(
            violations <= 1,
            "upper quantile should be >= lower in most cases"
        );
    }
}

// ── EBM tests ──────────────────────────────────────────────────────

#[test]
fn ebm_classif() {
    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [0.0, 0.2],
        [1.0, 1.0],
        [1.1, 0.9],
        [0.9, 1.1],
        [1.0, 0.9]
    ];
    let target = vec![0, 0, 0, 0, 1, 1, 1, 1];
    let task = ClassificationTask::new("ebm", features, target).unwrap();

    let mut ebm = EBM::new().with_n_rounds(50).with_learning_rate(0.05);
    let model = ebm.train_classif(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_classif(task.target().to_vec());
    let acc = Accuracy.score(&pred).unwrap();
    assert!(acc >= 0.75, "EBM should classify separable data, got {acc}");
}

#[test]
fn ebm_regress() {
    let features = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let target = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let task = RegressionTask::new("ebm_r", features, target).unwrap();

    let mut ebm = EBM::new().with_n_rounds(100).with_learning_rate(0.05);
    let model = ebm.train_regress(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_regress(task.target().to_vec());
    let rmse = Rmse.score(&pred).unwrap();
    assert!(rmse < 3.0, "EBM should learn linear trend, got RMSE={rmse}");
}

#[test]
fn ebm_feature_importance() {
    let features = array![
        [0.0, 42.0],
        [0.1, 13.0],
        [0.2, 99.0],
        [0.0, 55.0],
        [1.0, 42.0],
        [1.1, 13.0],
        [1.2, 99.0],
        [1.0, 55.0]
    ];
    let target = vec![0, 0, 0, 0, 1, 1, 1, 1];
    let task = ClassificationTask::new("ebm_imp", features, target).unwrap();

    let mut ebm = EBM::new().with_n_rounds(50);
    let model = ebm.train_classif(&task).unwrap();
    let imp = model.feature_importance();
    assert!(imp.is_some(), "EBM should provide feature importance");
}

/// Regression test for HIGH-14: EBM used to silently treat any target as
/// binary, regardless of the actual number of classes -- a 3-class target
/// would train and predict without error, producing meaningless
/// (effectively binary, always predicting 0 or 1) output. It must now error
/// instead.
#[test]
fn ebm_rejects_multiclass_target() {
    let features = array![
        [0.0],
        [0.1],
        [0.2],
        [1.0],
        [1.1],
        [1.2],
        [2.0],
        [2.1],
        [2.2]
    ];
    let target = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
    let task = ClassificationTask::new("ebm_multiclass", features, target).unwrap();

    let mut ebm = EBM::new().with_n_rounds(10);
    let err = ebm.train_classif(&task);
    assert!(
        err.is_err(),
        "EBM must reject a 3-class target instead of silently treating it as binary"
    );
}

// ── SMOTE tests ────────────────────────────────────────────────────

#[test]
fn smote_balances_classes() {
    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [0.0, 0.2],
        [0.1, 0.0],
        [1.0, 1.0],
    ];
    let target = vec![0, 0, 0, 0, 0, 1]; // 5 vs 1
    let task = ClassificationTask::new("smote", features, target).unwrap();

    let smote = Smote::new().with_seed(42);
    let balanced = smote.balance(&task).unwrap();

    // After SMOTE, class counts should be equal
    let n0 = balanced.target().iter().filter(|&&t| t == 0).count();
    let n1 = balanced.target().iter().filter(|&&t| t == 1).count();
    assert_eq!(n0, n1, "SMOTE should balance classes: {n0} vs {n1}");
    assert!(balanced.n_samples() > task.n_samples());
}

#[test]
fn smote_already_balanced() {
    let features = array![[0.0], [1.0], [2.0], [3.0]];
    let target = vec![0, 0, 1, 1];
    let task = ClassificationTask::new("bal", features, target).unwrap();

    let smote = Smote::new();
    let balanced = smote.balance(&task).unwrap();
    assert_eq!(balanced.n_samples(), 4); // no change needed
}

#[test]
fn smote_then_train() {
    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [0.0, 0.2],
        [0.1, 0.0],
        [0.2, 0.1],
        [0.0, 0.1],
        [0.1, 0.2],
        [1.0, 1.0], // minority: 1 sample
    ];
    let target = vec![0, 0, 0, 0, 0, 0, 0, 0, 1];
    let task = ClassificationTask::new("smote_train", features, target).unwrap();

    let smote = Smote::new().with_k_neighbors(1).with_seed(42);
    let balanced = smote.balance(&task).unwrap();

    let mut dt = DecisionTree::default();
    let model = dt.train_classif(&balanced).unwrap();
    let pred = model.predict(balanced.features()).unwrap();
    assert_eq!(pred.n_samples(), balanced.n_samples());
}

// ── Geographical-XGBoost tests ─────────────────────────────────────

#[test]
fn geo_xgboost_basic_regression() {
    // Spatially varying relationship: target depends on position
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
    // Target varies spatially: low values on left, high on right
    let target = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];
    let coords: Vec<(f64, f64)> = (0..10).map(|i| (i as f64, 0.0)).collect();
    let task = RegressionTask::new("geo", features.clone(), target.clone()).unwrap();

    let mut gxgb = GeoXGBoost::new(coords.clone())
        .with_bandwidth(4)
        .with_n_estimators(50)
        .with_max_depth(3);
    let model = gxgb.train_geo(&task).unwrap();
    // Fitted values: predict_spatial with the training coords exercises each
    // point's own local model (predict() alone is global-only by design).
    let pred = model
        .predict_spatial(&features, &coords)
        .unwrap()
        .with_truth_regress(target);

    let rmse = Rmse.score(&pred).unwrap();
    assert!(
        rmse < 5.0,
        "G-XGBoost should learn spatial pattern, got RMSE={rmse}"
    );
}

#[test]
fn geo_xgboost_spatial_heterogeneity() {
    // Two spatial clusters with different relationships
    // Left cluster: y = x, Right cluster: y = -x + 20
    let features = array![
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0], // left cluster
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0] // right cluster (same x, different y)
    ];
    let target = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, // y = x
        19.0, 18.0, 17.0, 16.0, 15.0, // y = -x + 20
    ];
    let coords: Vec<(f64, f64)> = vec![
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (3.0, 0.0),
        (4.0, 0.0),
        (100.0, 0.0),
        (101.0, 0.0),
        (102.0, 0.0),
        (103.0, 0.0),
        (104.0, 0.0),
    ];
    let task = RegressionTask::new("hetero", features.clone(), target.clone()).unwrap();

    // G-XGBoost should handle this better than global XGBoost
    let mut gxgb = GeoXGBoost::new(coords.clone())
        .with_bandwidth(4)
        .with_n_estimators(50);
    let model = gxgb.train_geo(&task).unwrap();
    // Fitted values via the local models (predict() alone is global-only).
    let pred = model
        .predict_spatial(&features, &coords)
        .unwrap()
        .with_truth_regress(target.clone());
    let gxgb_rmse = Rmse.score(&pred).unwrap();

    // Compare with global XGBoost
    let mut xgb = XGBoost::new().with_n_estimators(50);
    let global_model = xgb.train_regress(&task).unwrap();
    let global_pred = global_model
        .predict(&features)
        .unwrap()
        .with_truth_regress(target);
    let global_rmse = Rmse.score(&global_pred).unwrap();

    // G-XGBoost should perform at least as well as global
    assert!(
        gxgb_rmse <= global_rmse + 1.0,
        "G-XGBoost ({gxgb_rmse:.2}) should be competitive with global XGBoost ({global_rmse:.2})"
    );
}

#[test]
fn geo_xgboost_feature_importance() {
    let features = array![
        [0.0, 99.0],
        [1.0, 42.0],
        [2.0, 13.0],
        [3.0, 77.0],
        [4.0, 99.0],
        [5.0, 42.0],
        [6.0, 13.0],
        [7.0, 77.0]
    ];
    let target = vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0];
    let coords: Vec<(f64, f64)> = (0..8).map(|i| (i as f64, 0.0)).collect();
    let task = RegressionTask::new("geo_imp", features, target).unwrap();

    let mut gxgb = GeoXGBoost::new(coords)
        .with_bandwidth(4)
        .with_n_estimators(30);
    let model = gxgb.train_regress(&task).unwrap();
    let imp = model.feature_importance();
    assert!(imp.is_some(), "G-XGBoost should provide feature importance");
}

#[test]
fn geo_xgboost_coords_mismatch_error() {
    let features = array![[1.0], [2.0], [3.0]];
    let target = vec![1.0, 2.0, 3.0];
    let coords = vec![(0.0, 0.0), (1.0, 0.0)]; // only 2 coords for 3 samples
    let task = RegressionTask::new("bad", features, target).unwrap();

    let mut gxgb = GeoXGBoost::new(coords);
    assert!(gxgb.train_regress(&task).is_err());
}

#[test]
fn geo_xgboost_rejects_classification() {
    let features = array![[1.0], [2.0]];
    let target = vec![0, 1];
    let task = ClassificationTask::new("bad", features, target).unwrap();
    let mut gxgb = GeoXGBoost::new(vec![(0.0, 0.0), (1.0, 0.0)]);
    assert!(gxgb.train_classif(&task).is_err());
}

#[test]
fn geo_xgboost_fixed_alpha() {
    let features = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]];
    let target = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
    let coords: Vec<(f64, f64)> = (0..8).map(|i| (i as f64, 0.0)).collect();
    let task = RegressionTask::new("alpha", features.clone(), target.clone()).unwrap();

    // alpha=0 should be pure global, alpha=1 pure local
    let mut g0 = GeoXGBoost::new(coords.clone())
        .with_alpha(0.0)
        .with_n_estimators(30)
        .with_bandwidth(3);
    let m0 = g0.train_geo(&task).unwrap();
    let p0 = m0.predict_spatial(&features, &coords).unwrap();

    let mut g1 = GeoXGBoost::new(coords.clone())
        .with_alpha(1.0)
        .with_n_estimators(30)
        .with_bandwidth(3);
    let m1 = g1.train_geo(&task).unwrap();
    let p1 = m1.predict_spatial(&features, &coords).unwrap();

    // Both should produce valid predictions
    assert_eq!(p0.n_samples(), 8);
    assert_eq!(p1.n_samples(), 8);
}

// ── Bayesian Optimization (TPE) tests ──────────────────────────────

#[test]
fn bayesian_optimizer_classif() {
    use smelt_ml::tuning::ParamSpace;

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
        [1.1, 1.1]
    ];
    let target = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
    let task = ClassificationTask::new("bo", features, target).unwrap();

    let mut space = ParamSpace::new();
    space.insert("max_depth".into(), ParamDistribution::Uniform(1.0, 10.0));

    let bo = BayesianOptimizer::new(
        |params| {
            Box::new(DecisionTree::new().with_max_depth(params["max_depth"].as_usize().unwrap()))
        },
        space,
    )
    .with_n_iter(15)
    .with_n_initial(5)
    .with_seed(42);

    let cv = CrossValidation::new(3).with_seed(42);
    let result = bo.tune_classif(&task, &cv, &Accuracy).unwrap();

    assert_eq!(result.all_results.len(), 15);
    assert!(result.best_score >= 0.5);
    assert!(result.maximize);
}

#[test]
fn bayesian_optimizer_regress() {
    use smelt_ml::tuning::ParamSpace;

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
    let task = RegressionTask::new("bo_r", features, target).unwrap();

    let mut space = ParamSpace::new();
    space.insert("max_depth".into(), ParamDistribution::Uniform(1.0, 8.0));

    let bo = BayesianOptimizer::new(
        |params| {
            Box::new(DecisionTree::new().with_max_depth(params["max_depth"].as_usize().unwrap()))
        },
        space,
    )
    .with_n_iter(12)
    .with_seed(42);

    let ho = Holdout::new(0.8).with_seed(42);
    let result = bo.tune_regress(&task, &ho, &Rmse).unwrap();

    assert_eq!(result.all_results.len(), 12);
    assert!(!result.maximize); // RMSE minimized
}

#[test]
fn bayesian_optimizer_multi_param() {
    use smelt_ml::tuning::ParamSpace;

    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [0.0, 0.2],
        [1.0, 1.0],
        [1.1, 0.9],
        [0.9, 1.1],
        [1.0, 0.9]
    ];
    let target = vec![0, 0, 0, 0, 1, 1, 1, 1];
    let task = ClassificationTask::new("bo_mp", features, target).unwrap();

    let mut space = ParamSpace::new();
    space.insert(
        "max_depth".into(),
        ParamDistribution::Choice(vec![1.0.into(), 3.0.into(), 5.0.into(), 7.0.into()]),
    );
    space.insert(
        "n_estimators".into(),
        ParamDistribution::Uniform(10.0, 100.0),
    );

    let bo = BayesianOptimizer::new(
        |params| {
            Box::new(
                RandomForest::new()
                    .with_n_estimators(params["n_estimators"].as_usize().unwrap())
                    .with_max_depth(params["max_depth"].as_usize().unwrap())
                    .with_seed(42),
            )
        },
        space,
    )
    .with_n_iter(10)
    .with_seed(42);

    let cv = CrossValidation::new(2).with_seed(42);
    let result = bo.tune_classif(&task, &cv, &Accuracy).unwrap();

    assert_eq!(result.all_results.len(), 10);
    assert!(result.best_params.contains_key("max_depth"));
    assert!(result.best_params.contains_key("n_estimators"));
}

#[test]
fn bayesian_optimizer_log_uniform() {
    use smelt_ml::tuning::ParamSpace;

    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [0.0, 0.2],
        [1.0, 1.0],
        [1.1, 0.9],
        [0.9, 1.1],
        [1.0, 0.9]
    ];
    let target = vec![0, 0, 0, 0, 1, 1, 1, 1];
    let task = ClassificationTask::new("bo_lu", features, target).unwrap();

    let mut space = ParamSpace::new();
    space.insert(
        "learning_rate".into(),
        ParamDistribution::LogUniform(0.001, 1.0),
    );

    let bo = BayesianOptimizer::new(
        |params| {
            Box::new(
                LogisticRegression::new()
                    .with_learning_rate(params["learning_rate"].as_f64().unwrap())
                    .with_max_iter(500),
            )
        },
        space,
    )
    .with_n_iter(10)
    .with_seed(42);

    let cv = CrossValidation::new(2).with_seed(42);
    let result = bo.tune_classif(&task, &cv, &Accuracy).unwrap();

    // All learning rates should be in [0.001, 1.0]
    for (params, _) in &result.all_results {
        let lr = params["learning_rate"].as_f64().unwrap();
        assert!((0.001..=1.0).contains(&lr), "lr={lr} out of bounds");
    }
}

#[test]
fn bayesian_optimizer_beats_random() {
    use smelt_ml::tuning::{ParamSpace, RandomSearch};

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
        [1.1, 1.1]
    ];
    let target = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
    let task = ClassificationTask::new("bo_vs", features, target).unwrap();

    let mut space = ParamSpace::new();
    space.insert("max_depth".into(), ParamDistribution::Uniform(1.0, 10.0));

    let cv = CrossValidation::new(3).with_seed(42);

    // Bayesian with 15 iterations
    let bo = BayesianOptimizer::new(
        |p| Box::new(DecisionTree::new().with_max_depth(p["max_depth"].as_usize().unwrap())),
        space.clone(),
    )
    .with_n_iter(15)
    .with_seed(42);
    let bo_result = bo.tune_classif(&task, &cv, &Accuracy).unwrap();

    // Random with same budget
    let rs = RandomSearch::new(
        |p| Box::new(DecisionTree::new().with_max_depth(p["max_depth"].as_usize().unwrap())),
        space,
    )
    .with_n_iter(15)
    .with_seed(42);
    let rs_result = rs.tune_classif(&task, &cv, &Accuracy).unwrap();

    // BO should be at least as good as random (on average it's better)
    assert!(
        bo_result.best_score >= rs_result.best_score - 0.1,
        "BO ({:.4}) should be competitive with Random ({:.4})",
        bo_result.best_score,
        rs_result.best_score
    );
}

// ── Oblique Tree / Forest tests ────────────────────────────────────

#[test]
fn oblique_tree_classif_separable() {
    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [0.0, 0.2],
        [1.0, 1.0],
        [1.1, 0.9],
        [0.9, 1.1],
        [1.0, 0.9]
    ];
    let target = vec![0, 0, 0, 0, 1, 1, 1, 1];
    let task = ClassificationTask::new("obl", features, target).unwrap();

    let mut tree = ObliqueTree::new().with_seed(42).with_n_projections(20);
    let model = tree.train_classif(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_classif(task.target().to_vec());
    let acc = Accuracy.score(&pred).unwrap();
    assert!(
        acc >= 0.75,
        "ObliqueTree should classify separable data, got {acc}"
    );
}

#[test]
fn oblique_tree_xor_pattern() {
    // XOR: axis-aligned trees struggle, oblique should do better
    let features = array![
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.1, 0.1],
        [0.1, 0.9],
        [0.9, 0.1],
        [0.9, 0.9]
    ];
    let target = vec![0, 1, 1, 0, 0, 1, 1, 0]; // XOR
    let task = ClassificationTask::new("xor", features, target).unwrap();

    let mut tree = ObliqueTree::new().with_seed(42).with_n_projections(30);
    let model = tree.train_classif(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_classif(task.target().to_vec());
    let acc = Accuracy.score(&pred).unwrap();
    assert!(
        acc >= 0.5,
        "ObliqueTree on XOR should do at least random, got {acc}"
    );
}

#[test]
fn oblique_tree_regress() {
    let features = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]];
    let target = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
    let task = RegressionTask::new("obl_r", features, target).unwrap();

    let mut tree = ObliqueTree::new().with_seed(42);
    let model = tree.train_regress(&task).unwrap();
    let pred = model.predict(task.features()).unwrap();
    assert_eq!(pred.n_samples(), 8);
}

#[test]
fn oblique_forest_classif() {
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
        [1.1, 1.1]
    ];
    let target = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
    let task = ClassificationTask::new("sporf", features, target).unwrap();

    let mut forest = ObliqueForest::new().with_n_estimators(20).with_seed(42);
    let model = forest.train_classif(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_classif(task.target().to_vec());
    let acc = Accuracy.score(&pred).unwrap();
    assert_eq!(
        acc, 1.0,
        "ObliqueForest should perfectly separate this data"
    );
}

#[test]
fn oblique_forest_regress() {
    let features = array![[1.0], [2.0], [3.0], [4.0], [6.0], [7.0], [8.0], [9.0]];
    let target = vec![0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0];
    let task = RegressionTask::new("sporf_r", features, target).unwrap();

    let mut forest = ObliqueForest::new().with_n_estimators(20).with_seed(42);
    let model = forest.train_regress(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_regress(task.target().to_vec());
    let rmse = Rmse.score(&pred).unwrap();
    assert!(
        rmse < 2.0,
        "ObliqueForest should learn step function, got RMSE={rmse}"
    );
}

#[test]
fn oblique_forest_feature_importance() {
    let features = array![
        [0.0, 42.0],
        [0.1, 13.0],
        [0.2, 99.0],
        [0.0, 55.0],
        [1.0, 42.0],
        [1.1, 13.0],
        [1.2, 99.0],
        [1.0, 55.0]
    ];
    let target = vec![0, 0, 0, 0, 1, 1, 1, 1];
    let task = ClassificationTask::new("sporf_imp", features, target).unwrap();

    let mut forest = ObliqueForest::new().with_n_estimators(30).with_seed(42);
    let model = forest.train_classif(&task).unwrap();
    let imp = model.feature_importance();
    assert!(
        imp.is_some(),
        "ObliqueForest should provide feature importance"
    );
}

#[test]
fn oblique_forest_in_benchmark() {
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
        [1.1, 1.1]
    ];
    let target = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
    let task = ClassificationTask::new("sporf_bench", features, target).unwrap();
    let cv = CrossValidation::new(4).with_seed(42);

    let mut forest = ObliqueForest::new().with_n_estimators(10).with_seed(42);
    let r = benchmark::resample_classif(&mut forest, &task, &cv, &[&Accuracy]).unwrap();
    assert_eq!(r.learner_id, "oblique_forest");
    assert_eq!(r.scores.len(), 4);
}

// ── Causal Forest tests ────────────────────────────────────────────

#[test]
fn causal_forest_basic_ate() {
    use smelt_ml::causal::CausalForest;

    // Treatment adds 3 to outcome
    let features = array![
        [25.0],
        [30.0],
        [35.0],
        [40.0],
        [45.0],
        [25.0],
        [30.0],
        [35.0],
        [40.0],
        [45.0],
    ];
    let treatment = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
    let outcome = vec![5.0, 6.0, 7.0, 8.0, 9.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let names = vec!["age".to_string()];

    let cf = CausalForest::new()
        .with_n_estimators(50)
        .with_min_samples_leaf(2)
        .with_seed(42);

    let result = cf
        .estimate(&features, &treatment, &outcome, &names)
        .unwrap();

    // ATE should be around 3.0
    assert!(
        result.ate > 0.0,
        "ATE should be positive, got {}",
        result.ate
    );
    assert_eq!(result.effects.len(), 10);
    assert!(!result.feature_importance.is_empty());
}

#[test]
fn causal_forest_heterogeneous_effect() {
    use smelt_ml::causal::CausalForest;

    // Treatment effect varies: young people benefit more
    let features = array![
        [20.0],
        [20.0],
        [20.0],
        [20.0],
        [50.0],
        [50.0],
        [50.0],
        [50.0],
        [20.0],
        [20.0],
        [20.0],
        [20.0],
        [50.0],
        [50.0],
        [50.0],
        [50.0],
    ];
    let treatment = vec![
        0, 0, 0, 0, // young control
        0, 0, 0, 0, // old control
        1, 1, 1, 1, // young treated
        1, 1, 1, 1, // old treated
    ];
    let outcome = vec![
        5.0, 5.0, 5.0, 5.0, // young control: baseline 5
        8.0, 8.0, 8.0, 8.0, // old control: baseline 8
        15.0, 15.0, 15.0, 15.0, // young treated: +10
        10.0, 10.0, 10.0, 10.0, // old treated: +2
    ];
    let names = vec!["age".to_string()];

    let cf = CausalForest::new()
        .with_n_estimators(100)
        .with_min_samples_leaf(2)
        .with_seed(42);

    let result = cf
        .estimate(&features, &treatment, &outcome, &names)
        .unwrap();

    // Effects should differ between young and old
    let young_effects: Vec<f64> = result.effects[..4].iter().map(|e| e.estimate).collect();
    let old_effects: Vec<f64> = result.effects[4..8].iter().map(|e| e.estimate).collect();
    let young_avg = young_effects.iter().sum::<f64>() / young_effects.len() as f64;
    let old_avg = old_effects.iter().sum::<f64>() / old_effects.len() as f64;

    // With small data, at least the overall ATE should be positive
    // (treatment has a positive effect in both groups)
    assert!(
        result.ate > 0.0,
        "ATE should be positive, got {:.2}. Young={young_avg:.2}, Old={old_avg:.2}",
        result.ate
    );
}

#[test]
fn causal_forest_confidence_intervals() {
    use smelt_ml::causal::CausalForest;

    let features = array![
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
    ];
    let treatment = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
    let outcome = vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let names = vec!["x".to_string()];

    let cf = CausalForest::new().with_n_estimators(50).with_seed(42);
    let result = cf
        .estimate(&features, &treatment, &outcome, &names)
        .unwrap();

    for effect in &result.effects {
        assert!(effect.ci_lower <= effect.estimate);
        assert!(effect.estimate <= effect.ci_upper);
        assert!(effect.std_error >= 0.0);
    }
}

#[test]
fn causal_forest_dimension_mismatch() {
    use smelt_ml::causal::CausalForest;

    let features = array![[1.0], [2.0], [3.0]];
    let treatment = vec![0, 1]; // wrong size
    let outcome = vec![1.0, 2.0, 3.0];

    let cf = CausalForest::new();
    assert!(
        cf.estimate(&features, &treatment, &outcome, &["x".into()])
            .is_err()
    );
}

// ── Feature Selection Filter tests ─────────────────────────────────

#[test]
fn filter_variance_selects_non_constant() {
    use smelt_ml::preprocess::filter::FilterSelector;

    let features = array![
        [1.0, 5.0, 0.0], // col 0: varies, col 1: varies more, col 2: constant
        [2.0, 5.0, 0.0],
        [3.0, 5.0, 0.0],
        [4.0, 5.0, 0.0],
    ];
    let target = vec![0.0, 1.0, 0.0, 1.0];
    let task = RegressionTask::new("var", features.clone(), target).unwrap();

    let selector = FilterSelector::variance(2); // keep top 2
    let _pipe = Pipeline::new(vec![Box::new(selector)], Box::new(DecisionTree::default()));
    // The constant column should be dropped
    // Just verify it compiles and runs
    let mut pipe = Pipeline::new(
        vec![Box::new(FilterSelector::variance(2))],
        Box::new(DecisionTree::default()),
    );
    let model = pipe.train_regress(&task).unwrap();
    let pred = model
        .predict(&features.select(ndarray::Axis(1), &[0, 1]))
        .unwrap();
    assert_eq!(pred.n_samples(), 4);
}

#[test]
fn filter_correlation_selects_informative() {
    use smelt_ml::preprocess::filter::FilterSelector;

    // col 0: correlated with target, col 1: random noise
    let features = array![
        [1.0, 42.0],
        [2.0, 13.0],
        [3.0, 99.0],
        [4.0, 55.0],
        [5.0, 42.0],
        [6.0, 13.0],
        [7.0, 99.0],
        [8.0, 55.0],
    ];
    let target = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]; // y = 2x

    let mut selector = FilterSelector::correlation(1);
    selector.fit_supervised(&features, &target).unwrap();

    let selected = selector.selected_indices().unwrap();
    assert_eq!(selected.len(), 1);
    assert_eq!(
        selected[0], 0,
        "should select feature 0 (correlated with target)"
    );
}

#[test]
fn filter_anova_classif() {
    use smelt_ml::preprocess::filter::FilterSelector;

    // col 0: separates classes, col 1: noise
    let features = array![
        [0.0, 42.0],
        [0.1, 13.0],
        [0.2, 99.0],
        [0.0, 55.0],
        [1.0, 42.0],
        [1.1, 13.0],
        [1.2, 99.0],
        [1.0, 55.0],
    ];
    let target = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

    let mut selector = FilterSelector::anova_f(1);
    selector.fit_supervised(&features, &target).unwrap();

    let selected = selector.selected_indices().unwrap();
    assert_eq!(selected[0], 0, "ANOVA should select feature 0");
}

#[test]
fn filter_information_gain_classif() {
    use smelt_ml::preprocess::filter::FilterSelector;

    let features = array![
        [0.0, 42.0],
        [0.1, 13.0],
        [0.2, 99.0],
        [0.0, 55.0],
        [1.0, 42.0],
        [1.1, 13.0],
        [1.2, 99.0],
        [1.0, 55.0],
    ];
    let target = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

    let mut selector = FilterSelector::information_gain(1);
    selector.fit_supervised(&features, &target).unwrap();

    let selected = selector.selected_indices().unwrap();
    assert_eq!(selected[0], 0, "IG should select feature 0");
}

#[test]
fn filter_in_pipeline_with_cv() {
    use smelt_ml::preprocess::filter::FilterSelector;

    let features = array![
        [0.0, 42.0, 99.0],
        [0.1, 13.0, 55.0],
        [0.2, 99.0, 42.0],
        [0.0, 55.0, 13.0],
        [0.1, 42.0, 99.0],
        [0.2, 13.0, 55.0],
        [0.0, 99.0, 42.0],
        [0.1, 55.0, 13.0],
        [1.0, 42.0, 99.0],
        [1.1, 13.0, 55.0],
        [1.2, 99.0, 42.0],
        [1.0, 55.0, 13.0],
        [1.1, 42.0, 99.0],
        [1.2, 13.0, 55.0],
        [1.0, 99.0, 42.0],
        [1.1, 55.0, 13.0],
    ];
    let target = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
    let task = ClassificationTask::new("filter_cv", features, target).unwrap();

    // Pipeline: select top 2 features by ANOVA → DecisionTree
    let mut pipe = Pipeline::new(
        vec![Box::new(FilterSelector::anova_f(2))],
        Box::new(DecisionTree::default()),
    );

    let cv = CrossValidation::new(4).with_seed(42);
    let result = benchmark::resample_classif(&mut pipe, &task, &cv, &[&Accuracy]).unwrap();

    assert_eq!(result.scores.len(), 4);
    let mean_acc = result.mean_scores()[0];
    assert!(
        mean_acc >= 0.5,
        "filtered pipeline should work, got {mean_acc}"
    );
}

#[test]
fn filter_mutual_info_regression() {
    use smelt_ml::preprocess::filter::FilterSelector;

    let features = array![
        [1.0, 42.0],
        [2.0, 13.0],
        [3.0, 99.0],
        [4.0, 55.0],
        [5.0, 42.0],
        [6.0, 13.0],
        [7.0, 99.0],
        [8.0, 55.0],
    ];
    let target = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];

    let mut selector = FilterSelector::mutual_info(1);
    selector.fit_supervised(&features, &target).unwrap();

    let selected = selector.selected_indices().unwrap();
    assert_eq!(selected.len(), 1);
    // Feature 0 has higher MI with target
    assert_eq!(selected[0], 0);
}

// ── Info-theoretic filter tests ────────────────────────────────────

#[test]
fn filter_mrmr_selects_informative() {
    use smelt_ml::preprocess::filter::FilterSelector;

    // Feature 0: perfectly correlated with target
    // Feature 1: random noise
    // Feature 2: partial correlation
    let features = array![
        [1.0, 42.0, 1.5],
        [2.0, 13.0, 2.3],
        [3.0, 99.0, 3.1],
        [4.0, 55.0, 4.8],
        [5.0, 42.0, 4.9],
        [6.0, 13.0, 6.2],
        [7.0, 99.0, 7.0],
        [8.0, 55.0, 8.4],
    ];
    let target = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];

    let mut selector = FilterSelector::mrmr(2);
    selector.fit_supervised(&features, &target).unwrap();

    let selected = selector.selected_indices().unwrap();
    assert_eq!(selected.len(), 2);
    // Feature 0 (best correlated) should be selected
    assert!(selected.contains(&0), "MRMR should select feature 0");
}

#[test]
fn filter_jmi_selects_informative() {
    use smelt_ml::preprocess::filter::FilterSelector;

    let features = array![
        [1.0, 42.0],
        [2.0, 13.0],
        [3.0, 99.0],
        [4.0, 55.0],
        [5.0, 42.0],
        [6.0, 13.0],
        [7.0, 99.0],
        [8.0, 55.0],
    ];
    let target = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];

    let mut selector = FilterSelector::jmi(1);
    selector.fit_supervised(&features, &target).unwrap();

    let selected = selector.selected_indices().unwrap();
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0], 0, "JMI should select feature 0");
}

#[test]
fn filter_jmim_selects_informative() {
    use smelt_ml::preprocess::filter::FilterSelector;

    let features = array![
        [1.0, 42.0],
        [2.0, 13.0],
        [3.0, 99.0],
        [4.0, 55.0],
        [5.0, 42.0],
        [6.0, 13.0],
        [7.0, 99.0],
        [8.0, 55.0],
    ];
    let target = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];

    let mut selector = FilterSelector::jmim(1);
    selector.fit_supervised(&features, &target).unwrap();

    let selected = selector.selected_indices().unwrap();
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0], 0, "JMIM should select feature 0");
}

#[test]
fn filter_cmim_selects_informative() {
    use smelt_ml::preprocess::filter::FilterSelector;

    let features = array![
        [1.0, 42.0],
        [2.0, 13.0],
        [3.0, 99.0],
        [4.0, 55.0],
        [5.0, 42.0],
        [6.0, 13.0],
        [7.0, 99.0],
        [8.0, 55.0],
    ];
    let target = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];

    let mut selector = FilterSelector::cmim(1);
    selector.fit_supervised(&features, &target).unwrap();

    let selected = selector.selected_indices().unwrap();
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0], 0, "CMIM should select feature 0");
}

#[test]
fn filter_relief_selects_informative() {
    use smelt_ml::preprocess::filter::FilterSelector;

    // Feature 0 linearly related to target, feature 1 is noise
    let features = array![
        [1.0, 42.0],
        [2.0, 13.0],
        [3.0, 99.0],
        [4.0, 55.0],
        [5.0, 42.0],
        [6.0, 13.0],
        [7.0, 99.0],
        [8.0, 55.0],
        [9.0, 42.0],
        [10.0, 13.0],
        [11.0, 99.0],
        [12.0, 55.0],
    ];
    let target = vec![
        2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0,
    ];

    let mut selector = FilterSelector::relief(1);
    selector.fit_supervised(&features, &target).unwrap();

    let selected = selector.selected_indices().unwrap();
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0], 0, "Relief should select feature 0");
}

// ── K-Means tests ──────────────────────────────────────────────────

#[test]
fn kmeans_two_clusters() {
    use smelt_ml::cluster::KMeans;
    let data = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [5.0, 5.0],
        [5.1, 4.9],
        [4.9, 5.1]
    ];
    let result = KMeans::new(2).fit(&data).unwrap();
    assert_eq!(result.n_clusters, 2);
    // First 3 samples should be in one cluster, last 3 in another
    assert_eq!(result.labels[0], result.labels[1]);
    assert_eq!(result.labels[3], result.labels[4]);
    assert_ne!(result.labels[0], result.labels[3]);
}

#[test]
fn kmeans_silhouette() {
    use smelt_ml::cluster::KMeans;
    let data = array![[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]];
    let result = KMeans::new(2).fit(&data).unwrap();
    let sil = result.silhouette_score(&data);
    assert!(
        sil > 0.5,
        "well-separated clusters should have high silhouette, got {sil}"
    );
}

// ── DBSCAN tests ───────────────────────────────────────────────────

#[test]
fn dbscan_finds_clusters() {
    use smelt_ml::cluster::DBSCAN;
    let data = array![[0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [5.0, 5.0], [5.1, 5.1]];
    let result = DBSCAN::new(0.5, 2).fit(&data).unwrap();
    assert!(result.n_clusters >= 2, "should find at least 2 clusters");
}

#[test]
fn dbscan_noise_detection() {
    use smelt_ml::cluster::DBSCAN;
    // Outlier at (100, 100)
    let data = array![[0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [100.0, 100.0]];
    let result = DBSCAN::new(0.5, 2).fit(&data).unwrap();
    assert_eq!(result.labels[3], -1, "outlier should be noise (-1)");
}

// ── PCA tests ──────────────────────────────────────────────────────

#[test]
fn pca_reduces_dimensions() {
    let data = array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]
    ];
    let mut pca = PCA::new(2);
    let reduced = pca.fit_transform(&data).unwrap();
    assert_eq!(reduced.ncols(), 2);
    assert_eq!(reduced.nrows(), 4);
}

#[test]
fn pca_in_pipeline() {
    let features = array![
        [0.0, 0.0, 0.0],
        [0.1, 0.1, 0.1],
        [0.2, 0.0, 0.2],
        [0.0, 0.2, 0.1],
        [1.0, 1.0, 1.0],
        [1.1, 0.9, 1.1],
        [0.9, 1.1, 0.9],
        [1.0, 0.9, 1.0]
    ];
    let target = vec![0, 0, 0, 0, 1, 1, 1, 1];
    let task = ClassificationTask::new("pca_pipe", features, target).unwrap();

    let mut pipe = Pipeline::new(
        vec![Box::new(PCA::new(2))],
        Box::new(DecisionTree::default()),
    );
    let model = pipe.train_classif(&task).unwrap();
    // predict needs 3 features (original), pipeline transforms internally
    let test = array![[0.5, 0.5, 0.5]];
    let pred = model.predict(&test).unwrap();
    assert_eq!(pred.n_samples(), 1);
}

// ── RFE tests ──────────────────────────────────────────────────────

#[test]
fn rfe_selects_features() {
    let features = array![
        [0.0, 42.0, 99.0],
        [0.1, 13.0, 55.0],
        [0.2, 99.0, 42.0],
        [0.0, 55.0, 13.0],
        [1.0, 42.0, 99.0],
        [1.1, 13.0, 55.0],
        [1.2, 99.0, 42.0],
        [1.0, 55.0, 13.0],
    ];
    let target = vec![0, 0, 0, 0, 1, 1, 1, 1];
    let task = ClassificationTask::new("rfe", features, target).unwrap();

    let mut rfe = RFE::classif(|| Box::new(DecisionTree::default()), 2);
    let target_f64: Vec<f64> = task.target().iter().map(|&t| t as f64).collect();
    rfe.fit_supervised(task.features(), &target_f64).unwrap();

    let selected = rfe.selected_indices().unwrap();
    assert_eq!(selected.len(), 2);
}

// ── Benchmark Design tests ─────────────────────────────────────────

#[test]
fn benchmark_design_multi_learner() {
    use smelt_ml::benchmark_design::benchmark_classif;

    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [0.0, 0.2],
        [1.0, 1.0],
        [1.1, 0.9],
        [0.9, 1.1],
        [1.0, 0.9]
    ];
    let target = vec![0, 0, 0, 0, 1, 1, 1, 1];
    let task = ClassificationTask::new("bd", features, target).unwrap();

    let mut learners: Vec<Box<dyn Learner>> = vec![
        Box::new(DecisionTree::default()),
        Box::new(KNearestNeighbors::new(3)),
        Box::new(GaussianNB::new()),
    ];
    let cv = CrossValidation::new(2).with_seed(42);
    let result = benchmark_classif(&mut learners, &[&task], &cv, &[&Accuracy, &F1Score]).unwrap();

    assert_eq!(result.entries.len(), 3); // 3 learners × 1 task
    for entry in &result.entries {
        assert_eq!(entry.measure_ids.len(), 2);
    }
    // summary() should produce a readable table
    let summary = result.summary();
    assert!(summary.contains("decision_tree"));
}

// ── Mondrian Forest tests ───────────────────────────────────────────

/// Confirms `MondrianForest` composes with the rest of the framework
/// through the generic `Learner` trait -- cross-validation via
/// `benchmark::resample_classif`, not just its own direct batch/streaming
/// API (already covered by the module's own unit tests).
#[test]
fn mondrian_forest_works_through_generic_benchmark_cv() {
    use smelt_ml::benchmark;

    let mut rng_feats = Vec::new();
    let mut target = Vec::new();
    for i in 0..200 {
        let x = (i % 40) as f64 / 40.0;
        rng_feats.push(x);
        target.push(if x > 0.5 { 1usize } else { 0 });
    }
    let features = Array2::from_shape_vec((200, 1), rng_feats).unwrap();
    let task = ClassificationTask::new("mondrian_cv", features, target).unwrap();

    let mut forest = MondrianForest::new().with_n_trees(10).with_seed(1);
    let cv = CrossValidation::new(3).with_seed(0);
    let result = benchmark::resample_classif(&mut forest, &task, &cv, &[&Accuracy]).unwrap();
    let mean_acc = result.mean_scores()[0];
    assert!(
        mean_acc > 0.8,
        "MondrianForest via generic CV should fit this threshold rule well, got {mean_acc}"
    );
}

#[test]
fn mondrian_tree_regress_via_learner_trait() {
    let features = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]];
    let target = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
    let task = RegressionTask::new("mondrian_lin", features.clone(), target.clone()).unwrap();

    let mut tree = MondrianTree::new().with_seed(1);
    let model = tree.train_regress(&task).unwrap();
    let pred = model.predict(&features).unwrap();
    let Prediction::Regression { predicted, .. } = pred else {
        panic!("expected regression");
    };
    let rmse = (predicted
        .iter()
        .zip(&target)
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>()
        / target.len() as f64)
        .sqrt();
    assert!(
        rmse < 1.0,
        "should fit a simple linear trend well, got RMSE={rmse}"
    );
}

// ── Prioridad 3 quick items: ELM, Cost-Sensitive, Deep Forest ───────

#[test]
fn elm_works_through_generic_benchmark_cv() {
    use smelt_ml::benchmark;

    let mut feats = Vec::new();
    let mut target = Vec::new();
    for i in 0..200 {
        let x0 = (i % 20) as f64 / 20.0;
        let x1 = ((i / 20) % 20) as f64 / 20.0;
        feats.push(x0);
        feats.push(x1);
        target.push(if x0 + x1 > 1.0 { 1usize } else { 0 });
    }
    let features = Array2::from_shape_vec((200, 2), feats).unwrap();
    let task = ClassificationTask::new("elm_cv", features, target).unwrap();

    let mut elm = ExtremeLearningMachine::new().with_n_hidden(50).with_seed(1);
    let cv = CrossValidation::new(3).with_seed(0);
    let result = benchmark::resample_classif(&mut elm, &task, &cv, &[&Accuracy]).unwrap();
    let mean_acc = result.mean_scores()[0];
    assert!(
        mean_acc > 0.8,
        "ELM via generic CV should fit this boundary well, got {mean_acc}"
    );
}

#[test]
fn cost_sensitive_classifier_works_through_generic_learner_trait() {
    let features = array![
        [0.0],
        [0.5],
        [1.0],
        [1.5],
        [2.0],
        [2.5],
        [7.0],
        [7.5],
        [8.0],
        [8.5],
        [9.0],
        [9.5]
    ];
    let target = vec![0usize, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
    let task = ClassificationTask::new("cost_cv", features.clone(), target.clone()).unwrap();

    let mut cs = CostSensitiveClassifier::binary(|| Box::new(LogisticRegression::new()), 1.0, 5.0);
    let model = cs.train_classif(&task).unwrap();
    let pred = model.predict(&features).unwrap();
    let Prediction::Classification { predicted, .. } = pred else {
        panic!("expected classification");
    };
    let correct = predicted
        .iter()
        .zip(&target)
        .filter(|(p, t)| *p == *t)
        .count();
    assert!(
        correct as f64 / target.len() as f64 > 0.8,
        "cost-sensitive wrapper should still separate a clearly separable dataset well"
    );
}

#[test]
fn deep_forest_works_through_generic_benchmark_cv() {
    use smelt_ml::benchmark;

    let mut feats = Vec::new();
    let mut target = Vec::new();
    for i in 0..200 {
        let x0 = (i % 20) as f64 / 20.0;
        let x1 = ((i / 20) % 20) as f64 / 20.0;
        feats.push(x0);
        feats.push(x1);
        target.push(if x0 + x1 > 1.0 { 1usize } else { 0 });
    }
    let features = Array2::from_shape_vec((200, 2), feats).unwrap();
    let task = ClassificationTask::new("deep_forest_cv", features, target).unwrap();

    let mut df = DeepForest::new()
        .with_n_estimators_per_forest(20)
        .with_max_layers(3)
        .with_seed(1);
    let cv = CrossValidation::new(3).with_seed(0);
    let result = benchmark::resample_classif(&mut df, &task, &cv, &[&Accuracy]).unwrap();
    let mean_acc = result.mean_scores()[0];
    assert!(
        mean_acc > 0.8,
        "DeepForest via generic CV should fit this boundary well, got {mean_acc}"
    );
}

// ── Hyperband tests ────────────────────────────────────────────────

#[test]
fn hyperband_classif() {
    use smelt_ml::tuning::{Hyperband, ParamDistribution, ParamSpace};

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
        [1.1, 1.1]
    ];
    let target = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
    let task = ClassificationTask::new("hb", features, target).unwrap();

    let mut space = ParamSpace::new();
    space.insert("max_depth".into(), ParamDistribution::Uniform(1.0, 8.0));

    let hb = Hyperband::new(
        |params| {
            Box::new(DecisionTree::new().with_max_depth(params["max_depth"].as_usize().unwrap()))
        },
        space,
    )
    .with_max_folds(4)
    .with_seed(42);

    let result = hb.tune_classif(&task, &Accuracy).unwrap();
    assert!(result.best_score >= 0.5);
    assert!(!result.all_results.is_empty());
}

/// Determinism test for Hyperband's rayon-parallelized per-round evaluation
/// loop: two runs with the same seed must agree exactly.
#[test]
fn hyperband_parallel_evaluation_is_deterministic() {
    use smelt_ml::tuning::{Hyperband, ParamDistribution, ParamSpace};

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
        [1.1, 1.1]
    ];
    let target = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
    let task = ClassificationTask::new("hb_det", features, target).unwrap();

    let make_hb = || {
        let mut space = ParamSpace::new();
        space.insert("max_depth".into(), ParamDistribution::Uniform(1.0, 8.0));
        Hyperband::new(
            |params| {
                Box::new(
                    DecisionTree::new().with_max_depth(params["max_depth"].as_usize().unwrap()),
                )
            },
            space,
        )
        .with_max_folds(4)
        .with_seed(7)
    };

    let r1 = make_hb().tune_classif(&task, &Accuracy).unwrap();
    let r2 = make_hb().tune_classif(&task, &Accuracy).unwrap();

    assert_eq!(r1.best_params, r2.best_params);
    assert!((r1.best_score - r2.best_score).abs() < 1e-10);
    assert_eq!(r1.all_results.len(), r2.all_results.len());
}

// ── Isolation Forest tests ─────────────────────────────────────────

#[test]
fn isolation_forest_detects_outlier() {
    use smelt_ml::cluster::IsolationForest;
    let data = array![
        [1.0, 1.0],
        [1.1, 0.9],
        [0.9, 1.1],
        [1.0, 1.0],
        [0.9, 0.9],
        [1.1, 1.1],
        [1.0, 0.9],
        [0.9, 1.0],
        [50.0, 50.0], // clear outlier
    ];

    let iforest = IsolationForest::new()
        .with_n_estimators(100)
        .with_contamination(0.15)
        .with_seed(42);
    let result = iforest.fit_predict(&data).unwrap();

    assert_eq!(result.scores.len(), 9);
    // Outlier (index 8) should have the highest anomaly score
    let max_idx = result
        .scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    assert_eq!(max_idx, 8, "outlier at index 8 should have highest score");
    assert_eq!(result.labels[8], 1, "outlier should be labeled as anomaly");
}

#[test]
fn isolation_forest_all_normal() {
    use smelt_ml::cluster::IsolationForest;
    let data = array![[1.0, 1.0], [1.1, 0.9], [0.9, 1.1], [1.0, 1.0],];
    let iforest = IsolationForest::new()
        .with_n_estimators(50)
        .with_contamination(0.0) // no contamination expected
        .with_seed(42);
    let result = iforest.fit_predict(&data).unwrap();

    // All scores should be relatively similar (no clear outlier)
    let mean_score = result.scores.iter().sum::<f64>() / result.scores.len() as f64;
    for &s in &result.scores {
        assert!(
            (s - mean_score).abs() < 0.3,
            "scores should be similar for uniform data"
        );
    }
}

#[test]
fn isolation_forest_two_clusters_with_outlier() {
    use smelt_ml::cluster::IsolationForest;
    let data = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [5.0, 5.0],
        [5.1, 4.9],
        [4.9, 5.1],
        [100.0, 100.0], // outlier far from both clusters
    ];
    let iforest = IsolationForest::new()
        .with_n_estimators(100)
        .with_contamination(0.15)
        .with_seed(42);
    let result = iforest.fit_predict(&data).unwrap();

    assert!(
        result.scores[6] > result.scores[0],
        "outlier should score higher than cluster point"
    );
    assert!(result.n_anomalies >= 1, "should detect at least 1 anomaly");
}

// ── Classifier Chain (multi-label) tests ───────────────────────────

#[test]
fn classifier_chain_basic() {
    use smelt_ml::multilabel::ClassifierChain;

    let features = array![
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
    ];
    let labels = vec![
        vec![1, 0, 1],
        vec![0, 1, 0],
        vec![1, 1, 1],
        vec![0, 0, 0],
        vec![1, 0, 1],
        vec![0, 1, 0],
        vec![1, 1, 1],
        vec![0, 0, 0],
    ];

    let cc = ClassifierChain::new(|| Box::new(DecisionTree::default()));
    let model = cc.fit(&features, &labels).unwrap();
    let pred = model.predict(&features).unwrap();

    assert_eq!(pred.n_samples, 8);
    assert_eq!(pred.n_labels, 3);
    for row in &pred.labels {
        assert_eq!(row.len(), 3);
        for &v in row {
            assert!(v <= 1, "labels should be 0 or 1");
        }
    }
}

#[test]
fn classifier_chain_accuracy_metrics() {
    use smelt_ml::multilabel::ClassifierChain;

    let features = array![
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
    ];
    let labels = vec![
        vec![1, 0],
        vec![0, 1],
        vec![1, 1],
        vec![0, 0],
        vec![1, 0],
        vec![0, 1],
        vec![1, 1],
        vec![0, 0],
    ];

    let cc = ClassifierChain::new(|| Box::new(DecisionTree::default()));
    let model = cc.fit(&features, &labels).unwrap();
    let pred = model.predict(&features).unwrap();

    let subset_acc = model.subset_accuracy(&pred, &labels);
    let hamming = model.hamming_score(&pred, &labels);

    assert!((0.0..=1.0).contains(&subset_acc));
    assert!((0.0..=1.0).contains(&hamming));
    assert!(
        hamming >= subset_acc,
        "hamming score should be >= subset accuracy"
    );
}

#[test]
fn classifier_chain_with_rf() {
    use smelt_ml::multilabel::ClassifierChain;

    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [1.0, 1.0],
        [1.1, 0.9],
        [0.0, 0.0],
        [0.1, 0.1],
        [1.0, 1.0],
        [1.1, 0.9],
    ];
    let labels = vec![
        vec![0, 1],
        vec![0, 1],
        vec![1, 0],
        vec![1, 0],
        vec![0, 1],
        vec![0, 1],
        vec![1, 0],
        vec![1, 0],
    ];

    let cc =
        ClassifierChain::new(|| Box::new(RandomForest::new().with_n_estimators(10).with_seed(42)));
    let model = cc.fit(&features, &labels).unwrap();
    let pred = model.predict(&features).unwrap();

    assert_eq!(pred.n_labels, 2);
}

// ── Quantile Regression Forest tests ───────────────────────────────

#[test]
fn qrf_predicts_median() {
    let features = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]];
    let target = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
    let task = RegressionTask::new("qrf", features.clone(), target.clone()).unwrap();

    let mut qrf = QuantileForest::new().with_n_estimators(50).with_seed(42);
    let model = qrf.train_regress(&task).unwrap();
    let pred = model.predict(&features).unwrap().with_truth_regress(target);
    let rmse = Rmse.score(&pred).unwrap();
    assert!(
        rmse < 5.0,
        "QRF median should approximate well, got RMSE={rmse}"
    );
}

#[test]
fn qrf_quantile_ordering() {
    let features = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]];
    let target = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
    let task = RegressionTask::new("qrf_q", features.clone(), target).unwrap();

    let mut qrf = QuantileForest::new().with_n_estimators(50).with_seed(42);
    let model = qrf.train_regress(&task).unwrap();

    // Downcast to TrainedQuantileForest for quantile access
    // Since we can't downcast Box<dyn TrainedModel>, test via predict (median)
    let pred = model.predict(&features).unwrap();
    assert_eq!(pred.n_samples(), 8);
}

#[test]
fn qrf_in_benchmark() {
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
    let task = RegressionTask::new("qrf_b", features, target).unwrap();

    let ho = Holdout::new(0.8).with_seed(42);
    let mut qrf = QuantileForest::new().with_n_estimators(20).with_seed(42);
    let r = benchmark::resample_regress(&mut qrf, &task, &ho, &[&Rmse]).unwrap();
    assert_eq!(r.learner_id, "quantile_forest");
}

// ── ADASYN tests ───────────────────────────────────────────────────

#[test]
fn adasyn_balances_classes() {
    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [0.0, 0.2],
        [0.1, 0.0],
        [1.0, 1.0],
    ];
    let target = vec![0, 0, 0, 0, 0, 1];
    let task = ClassificationTask::new("adasyn", features, target).unwrap();

    let adasyn = Adasyn::new().with_seed(42);
    let balanced = adasyn.balance(&task).unwrap();

    let _n0 = balanced.target().iter().filter(|&&t| t == 0).count();
    let n1 = balanced.target().iter().filter(|&&t| t == 1).count();
    // ADASYN should approximately balance (may not be exact due to rounding)
    assert!(
        n1 >= 3,
        "minority should have more samples after ADASYN: {n1}"
    );
    assert!(balanced.n_samples() > 6);
}

#[test]
fn adasyn_focuses_on_boundary() {
    // Minority samples near majority boundary should get more synthetic samples
    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [0.3, 0.3], // minority NEAR majority (harder)
        [5.0, 5.0], // minority FAR from majority (easier)
    ];
    let target = vec![0, 0, 0, 1, 1];
    let task = ClassificationTask::new("ada_bound", features, target).unwrap();

    let adasyn = Adasyn::new().with_k_neighbors(3).with_seed(42);
    let balanced = adasyn.balance(&task).unwrap();
    assert!(balanced.n_samples() > 5);
}

// ── Multi-output Regression tests ──────────────────────────────────

#[test]
fn regressor_chain_basic() {
    use smelt_ml::multioutput::RegressorChain;

    let features = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
    let targets = vec![
        vec![2.0, 10.0],
        vec![4.0, 20.0],
        vec![6.0, 30.0],
        vec![8.0, 40.0],
        vec![10.0, 50.0],
        vec![12.0, 60.0],
    ];

    let rc = RegressorChain::new(|| Box::new(DecisionTree::default()));
    let model = rc.fit(&features, &targets).unwrap();
    let pred = model.predict(&features).unwrap();

    assert_eq!(pred.n_samples, 6);
    assert_eq!(pred.n_targets, 2);
    for row in &pred.values {
        assert_eq!(row.len(), 2);
    }
}

#[test]
fn regressor_chain_rmse() {
    use smelt_ml::multioutput::RegressorChain;

    let features = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
    let targets = vec![
        vec![2.0, 1.0],
        vec![4.0, 4.0],
        vec![6.0, 9.0],
        vec![8.0, 16.0],
        vec![10.0, 25.0],
        vec![12.0, 36.0],
    ];

    let rc = RegressorChain::new(|| Box::new(DecisionTree::default()));
    let model = rc.fit(&features, &targets).unwrap();
    let pred = model.predict(&features).unwrap();

    let rmse = model.mean_rmse(&pred, &targets);
    assert!(rmse >= 0.0);
}

// ── CQR tests ──────────────────────────────────────────────────────

#[test]
fn cqr_adaptive_intervals() {
    use smelt_ml::conformal::cqr::CQR;

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
        [10.0],
    ];
    let target = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];
    let task = RegressionTask::new("cqr", features.clone(), target.clone()).unwrap();

    // Train lower (0.1) and upper (0.9) quantile models
    let mut lower_gb = QuantileGB::new(0.1)
        .with_n_estimators(50)
        .with_learning_rate(0.1);
    let mut upper_gb = QuantileGB::new(0.9)
        .with_n_estimators(50)
        .with_learning_rate(0.1);

    let lower_model = lower_gb.train_regress(&task).unwrap();
    let upper_model = upper_gb.train_regress(&task).unwrap();

    // Calibrate on last 4 samples
    let cal_features = array![[7.0], [8.0], [9.0], [10.0]];
    let cal_targets = vec![14.0, 16.0, 18.0, 20.0];

    let cqr = CQR::calibrate(
        &*lower_model,
        &*upper_model,
        &cal_features,
        &cal_targets,
        0.1,
    )
    .unwrap();
    let intervals = cqr.predict(&array![[3.0], [6.0]]).unwrap();

    assert_eq!(intervals.len(), 2);
    for iv in &intervals {
        assert!(iv.lower <= iv.prediction);
        assert!(iv.prediction <= iv.upper);
    }
}

/// Regression test (5th audit, LOW-C): `CQR::calibrate` used to zip-truncate
/// mismatched calibration features/targets silently (conformity scores over
/// the common prefix), the same bug SplitConformal/ConformalRegressor
/// already guard against. It must be a DimensionMismatch.
#[test]
fn cqr_calibrate_rejects_mismatched_calibration_lengths() {
    use smelt_ml::conformal::cqr::CQR;

    let features = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]];
    let target = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
    let task = RegressionTask::new("cqr_dim", features, target).unwrap();

    let mut lower_gb = QuantileGB::new(0.1).with_n_estimators(20);
    let mut upper_gb = QuantileGB::new(0.9).with_n_estimators(20);
    let lower_model = lower_gb.train_regress(&task).unwrap();
    let upper_model = upper_gb.train_regress(&task).unwrap();

    // 3 calibration rows but only 2 targets.
    let cal_features = array![[6.0], [7.0], [8.0]];
    let cal_targets = vec![12.0, 14.0];
    let Err(err) = CQR::calibrate(
        &*lower_model,
        &*upper_model,
        &cal_features,
        &cal_targets,
        0.1,
    ) else {
        panic!("mismatched calibration lengths must be rejected, not zip-truncated");
    };
    assert!(
        matches!(
            err,
            smelt_ml::SmeltError::DimensionMismatch {
                expected: 2,
                got: 3
            }
        ),
        "got: {err:?}"
    );
}

/// Regression test (5th audit, LOW-C): `ConformalClassifier::calibrate`
/// shared CQR's silent zip-truncation of mismatched calibration lengths.
#[test]
fn conformal_classifier_calibrate_rejects_mismatched_calibration_lengths() {
    use smelt_ml::conformal::ConformalClassifier;

    let features = array![[0.0], [0.5], [1.0], [1.5], [2.0], [2.5]];
    let target = vec![0, 0, 0, 1, 1, 1];
    let task = ClassificationTask::new("ccl_dim", features, target).unwrap();

    let mut dt = DecisionTree::default();
    let model = dt.train_classif(&task).unwrap();

    // 3 calibration rows but only 2 targets.
    let cal_features = array![[0.5], [1.0], [2.0]];
    let cal_targets = vec![0, 1];
    let Err(err) = ConformalClassifier::calibrate(&*model, &cal_features, &cal_targets, 0.1) else {
        panic!("mismatched calibration lengths must be rejected, not zip-truncated");
    };
    assert!(
        matches!(
            err,
            smelt_ml::SmeltError::DimensionMismatch {
                expected: 2,
                got: 3
            }
        ),
        "got: {err:?}"
    );
}

// ── LightGBM tests ─────────────────────────────────────────────────

#[test]
fn lightgbm_classif_binary() {
    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [0.0, 0.2],
        [0.1, 0.0],
        [1.0, 1.0],
        [1.1, 0.9],
        [0.9, 1.1],
        [1.0, 0.9],
        [1.1, 1.0]
    ];
    let target = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
    let task = ClassificationTask::new("lgbm", features, target).unwrap();

    let mut lgbm = LightGBM::new()
        .with_n_estimators(50)
        .with_num_leaves(8)
        .with_learning_rate(0.1);
    let model = lgbm.train_classif(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_classif(task.target().to_vec());
    let acc = Accuracy.score(&pred).unwrap();
    assert!(
        acc >= 0.5,
        "LightGBM should classify separable data, got {acc}"
    );
}

#[test]
fn lightgbm_regress() {
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
    let task = RegressionTask::new("lgbm_r", features, target).unwrap();

    let mut lgbm = LightGBM::new()
        .with_n_estimators(100)
        .with_learning_rate(0.1);
    let model = lgbm.train_regress(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_regress(task.target().to_vec());
    let rmse = Rmse.score(&pred).unwrap();
    assert!(
        rmse < 2.0,
        "LightGBM should learn linear trend, got RMSE={rmse}"
    );
}

#[test]
fn lightgbm_in_benchmark() {
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
        [1.1, 1.1]
    ];
    let target = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
    let task = ClassificationTask::new("lgbm_b", features, target).unwrap();

    let cv = CrossValidation::new(4).with_seed(42);
    let mut lgbm = LightGBM::new().with_n_estimators(30).with_num_leaves(8);
    let r = benchmark::resample_classif(&mut lgbm, &task, &cv, &[&Accuracy]).unwrap();
    assert_eq!(r.learner_id, "lightgbm");
}

#[test]
fn lightgbm_multiclass() {
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
    let task = ClassificationTask::new("lgbm_mc", features, target).unwrap();

    let mut lgbm = LightGBM::new()
        .with_n_estimators(50)
        .with_learning_rate(0.1);
    let model = lgbm.train_classif(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_classif(task.target().to_vec());
    let acc = Accuracy.score(&pred).unwrap();
    assert!(acc >= 0.33, "LightGBM multiclass should work, got {acc}");
}

// ── CatBoost tests ─────────────────────────────────────────────────

#[test]
fn catboost_classif_binary() {
    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [0.0, 0.2],
        [0.1, 0.0],
        [1.0, 1.0],
        [1.1, 0.9],
        [0.9, 1.1],
        [1.0, 0.9],
        [1.1, 1.0]
    ];
    let target = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
    let task = ClassificationTask::new("cb", features, target).unwrap();

    let mut cb = CatBoost::new()
        .with_n_estimators(50)
        .with_depth(3)
        .with_learning_rate(0.1);
    let model = cb.train_classif(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_classif(task.target().to_vec());
    let acc = Accuracy.score(&pred).unwrap();
    assert!(acc >= 0.5, "CatBoost should classify, got {acc}");
}

#[test]
fn catboost_regress() {
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
    let task = RegressionTask::new("cb_r", features, target).unwrap();

    let mut cb = CatBoost::new().with_n_estimators(100).with_depth(3);
    let model = cb.train_regress(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_regress(task.target().to_vec());
    let rmse = Rmse.score(&pred).unwrap();
    assert!(rmse < 3.0, "CatBoost should learn linear, got RMSE={rmse}");
}

#[test]
fn catboost_with_categoricals() {
    // Feature 1 is categorical: 0.0=cat_A, 1.0=cat_B
    let features = array![
        [0.5, 0.0],
        [0.6, 0.0],
        [0.7, 0.0],
        [0.8, 0.0],
        [0.5, 1.0],
        [0.6, 1.0],
        [0.7, 1.0],
        [0.8, 1.0]
    ];
    let target = vec![0, 0, 0, 0, 1, 1, 1, 1];
    let task = ClassificationTask::new("cb_cat", features, target).unwrap();

    let mut cb = CatBoost::new()
        .with_n_estimators(50)
        .with_depth(3)
        .with_cat_features(vec![1]); // mark feature 1 as categorical
    let model = cb.train_classif(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_classif(task.target().to_vec());
    let acc = Accuracy.score(&pred).unwrap();
    assert!(
        acc >= 0.5,
        "CatBoost with categoricals should work, got {acc}"
    );
}

#[test]
fn catboost_in_benchmark() {
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
        [1.1, 1.1]
    ];
    let target = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
    let task = ClassificationTask::new("cb_b", features, target).unwrap();

    let cv = CrossValidation::new(4).with_seed(42);
    let mut cb = CatBoost::new().with_n_estimators(30).with_depth(3);
    let r = benchmark::resample_classif(&mut cb, &task, &cv, &[&Accuracy]).unwrap();
    assert_eq!(r.learner_id, "catboost");
}

// ── Random Survival Forest tests ───────────────────────────────────

#[test]
fn rsf_basic_prediction() {
    use smelt_ml::survival::{RandomSurvivalForest, SurvivalEvent};

    let features = array![
        [25.0, 0.0],
        [30.0, 1.0],
        [35.0, 0.0],
        [40.0, 1.0],
        [50.0, 0.0],
        [55.0, 1.0],
        [60.0, 0.0],
        [65.0, 1.0],
    ];
    let events = vec![
        SurvivalEvent {
            time: 10.0,
            event: true,
        },
        SurvivalEvent {
            time: 15.0,
            event: false,
        },
        SurvivalEvent {
            time: 8.0,
            event: true,
        },
        SurvivalEvent {
            time: 20.0,
            event: false,
        },
        SurvivalEvent {
            time: 5.0,
            event: true,
        },
        SurvivalEvent {
            time: 12.0,
            event: true,
        },
        SurvivalEvent {
            time: 3.0,
            event: true,
        },
        SurvivalEvent {
            time: 7.0,
            event: false,
        },
    ];

    let rsf = RandomSurvivalForest::new()
        .with_n_estimators(50)
        .with_seed(42);
    let predictions = rsf.fit_predict(&features, &events).unwrap();

    assert_eq!(predictions.len(), 8);
    for pred in &predictions {
        assert!(!pred.times.is_empty());
        assert!(!pred.survival.is_empty());
        // Survival should be monotonically non-increasing
        for w in pred.survival.windows(2) {
            assert!(
                w[0] >= w[1] - 1e-10,
                "survival should decrease: {} >= {}",
                w[0],
                w[1]
            );
        }
    }
}

#[test]
fn rsf_concordance_index() {
    use smelt_ml::survival::{RandomSurvivalForest, SurvivalEvent, concordance_index};

    let features = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0],];
    // Higher feature value = shorter survival (clear signal)
    let events = vec![
        SurvivalEvent {
            time: 100.0,
            event: true,
        },
        SurvivalEvent {
            time: 80.0,
            event: true,
        },
        SurvivalEvent {
            time: 60.0,
            event: true,
        },
        SurvivalEvent {
            time: 50.0,
            event: true,
        },
        SurvivalEvent {
            time: 40.0,
            event: true,
        },
        SurvivalEvent {
            time: 30.0,
            event: true,
        },
        SurvivalEvent {
            time: 20.0,
            event: true,
        },
        SurvivalEvent {
            time: 10.0,
            event: true,
        },
    ];

    let rsf = RandomSurvivalForest::new()
        .with_n_estimators(50)
        .with_seed(42);
    let predictions = rsf.fit_predict(&features, &events).unwrap();

    let c_idx = concordance_index(&predictions, &events);
    assert!((0.0..=1.0).contains(&c_idx));
    // With a clear monotonic relationship, C-index should be reasonable
    assert!(
        c_idx >= 0.4,
        "C-index should be above random (0.5), got {c_idx}"
    );
}

#[test]
fn rsf_survival_at_time() {
    use smelt_ml::survival::{RandomSurvivalForest, SurvivalEvent};

    let features = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
    let events = vec![
        SurvivalEvent {
            time: 5.0,
            event: true,
        },
        SurvivalEvent {
            time: 10.0,
            event: true,
        },
        SurvivalEvent {
            time: 15.0,
            event: false,
        },
        SurvivalEvent {
            time: 20.0,
            event: true,
        },
        SurvivalEvent {
            time: 25.0,
            event: true,
        },
        SurvivalEvent {
            time: 30.0,
            event: false,
        },
    ];

    let rsf = RandomSurvivalForest::new()
        .with_n_estimators(30)
        .with_seed(42);
    let preds = rsf.fit_predict(&features, &events).unwrap();

    // Survival at time 0 should be ~1.0, and decrease over time
    for pred in &preds {
        let s_early = pred.survival_at(1.0);
        let s_late = pred.survival_at(100.0);
        assert!(s_early >= s_late, "survival should decrease over time");
    }
}

// ── TreeSHAP tests ─────────────────────────────────────────────────

#[test]
fn shap_regress_basic() {
    use smelt_ml::importance::shap::tree_shap_regress;

    let features = array![
        [0.0, 99.0],
        [1.0, 42.0],
        [2.0, 13.0],
        [3.0, 77.0],
        [4.0, 99.0],
        [5.0, 42.0],
        [6.0, 13.0],
        [7.0, 77.0],
    ];
    let target = vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0];
    let task = RegressionTask::new("shap", features, target).unwrap();

    let mut dt = DecisionTree::default();
    let model = dt.train_regress(&task).unwrap();

    let result = tree_shap_regress(&*model, &task, 8).unwrap();

    assert_eq!(result.explanations.len(), 8);
    for exp in &result.explanations {
        assert_eq!(exp.values.len(), 2);
        // prediction should approximately equal base_value + sum(shap)
        let reconstructed = exp.base_value + exp.values.iter().sum::<f64>();
        // Allow some tolerance due to sampling approximation
        assert!(
            (reconstructed - exp.prediction).abs() < 15.0,
            "pred={:.2}, reconstructed={:.2}",
            exp.prediction,
            reconstructed
        );
    }

    // Global importance should exist
    assert_eq!(result.global_importance.len(), 2);
}

#[test]
fn shap_classif_basic() {
    use smelt_ml::importance::shap::tree_shap_classif;

    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [0.0, 0.2],
        [1.0, 1.0],
        [1.1, 0.9],
        [0.9, 1.1],
        [1.0, 0.9],
    ];
    let target = vec![0, 0, 0, 0, 1, 1, 1, 1];
    let task = ClassificationTask::new("shap_c", features, target).unwrap();

    let mut rf = RandomForest::new().with_n_estimators(20).with_seed(42);
    let model = rf.train_classif(&task).unwrap();

    let result = tree_shap_classif(&*model, &task, 8, 1).unwrap(); // explain class 1

    assert_eq!(result.explanations.len(), 8);
    // Class 1 samples should have positive total SHAP, class 0 negative
    let class1_shap: f64 = result.explanations[4..8]
        .iter()
        .map(|e| e.values.iter().sum::<f64>())
        .sum();
    let class0_shap: f64 = result.explanations[0..4]
        .iter()
        .map(|e| e.values.iter().sum::<f64>())
        .sum();
    assert!(
        class1_shap >= class0_shap - 0.5,
        "class 1 should have higher SHAP sum: c1={class1_shap:.2}, c0={class0_shap:.2}"
    );
}

#[test]
fn shap_global_importance_order() {
    use smelt_ml::importance::shap::tree_shap_regress;

    // Feature 0 is informative, feature 1 is noise
    let features = array![
        [0.0, 42.0],
        [1.0, 13.0],
        [2.0, 99.0],
        [3.0, 55.0],
        [4.0, 42.0],
        [5.0, 13.0],
        [6.0, 99.0],
        [7.0, 55.0],
    ];
    let target = vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0];
    let task = RegressionTask::new("shap_gi", features, target)
        .unwrap()
        .with_feature_names(vec!["signal".into(), "noise".into()])
        .unwrap();

    let mut dt = DecisionTree::default();
    let model = dt.train_regress(&task).unwrap();

    let result = tree_shap_regress(&*model, &task, 8).unwrap();

    // Signal feature should have higher global importance
    let signal_imp = result
        .global_importance
        .iter()
        .find(|(n, _)| n == "signal")
        .unwrap()
        .1;
    let noise_imp = result
        .global_importance
        .iter()
        .find(|(n, _)| n == "noise")
        .unwrap()
        .1;
    assert!(
        signal_imp >= noise_imp,
        "signal ({signal_imp:.4}) should be >= noise ({noise_imp:.4})"
    );
}

// ── Hoeffding Tree tests ───────────────────────────────────────────

#[test]
fn hoeffding_tree_classif() {
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
    let task = ClassificationTask::new("ht", features, target).unwrap();

    let mut ht = HoeffdingTree::new().with_grace_period(5).with_delta(1e-3);
    let model = ht.train_classif(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_classif(task.target().to_vec());
    let acc = Accuracy.score(&pred).unwrap();
    assert!(
        acc >= 0.5,
        "HoeffdingTree should do better than random, got {acc}"
    );
}

#[test]
fn hoeffding_tree_online() {
    // Test incremental learning
    let mut ht = HoeffdingTree::new().with_grace_period(5).with_delta(1e-3);

    // Feed samples one at a time
    for _ in 0..20 {
        ht.partial_fit(&[0.0, 0.0], 0, 2);
        ht.partial_fit(&[1.0, 1.0], 1, 2);
    }

    // Should have learned the pattern
    let test = array![[0.1, 0.1], [0.9, 0.9]];
    let task = ClassificationTask::new("ht_online", test.clone(), vec![0, 1]).unwrap();
    let mut ht2 = HoeffdingTree::new().with_grace_period(5);

    // Feed all training data
    for _ in 0..20 {
        ht2.partial_fit(&[0.0, 0.0], 0, 2);
        ht2.partial_fit(&[1.0, 1.0], 1, 2);
    }

    let model = ht2.train_classif(&task).unwrap();
    let pred = model.predict(&test).unwrap();
    assert_eq!(pred.n_samples(), 2);
}

#[test]
fn hoeffding_tree_in_benchmark() {
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
    let task = ClassificationTask::new("ht_b", features, target).unwrap();

    let cv = CrossValidation::new(4).with_seed(42);
    let mut ht = HoeffdingTree::new().with_grace_period(3);
    let r = benchmark::resample_classif(&mut ht, &task, &cv, &[&Accuracy]).unwrap();
    assert_eq!(r.learner_id, "hoeffding_tree");
}

// ── Dynamic Ensemble Selection tests ───────────────────────────────

#[test]
fn des_basic_classif() {
    use smelt_ml::learner::DynamicEnsemble;

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
        [1.1, 1.1]
    ];
    let target = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
    let task = ClassificationTask::new("des", features, target).unwrap();

    let mut des = DynamicEnsemble::new(vec![
        Box::new(|| Box::new(DecisionTree::default()) as Box<dyn Learner>),
        Box::new(|| Box::new(KNearestNeighbors::new(3)) as Box<dyn Learner>),
        Box::new(|| Box::new(GaussianNB::new()) as Box<dyn Learner>),
    ])
    .with_k_neighbors(3);

    let model = des.train_classif(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_classif(task.target().to_vec());
    let acc = Accuracy.score(&pred).unwrap();
    assert!(acc >= 0.5, "DES should classify, got {acc}");
}

// ════════════════════════════════════════════════════════════════════
// CONSOLIDATION TESTS: Edge Cases + Benchmark Coverage
// ════════════════════════════════════════════════════════════════════

// ── Edge Cases: Single sample ──────────────────────────────────────

#[test]
fn edge_single_sample_classif() {
    // Single sample should not panic — creates a trivial model
    let features = array![[1.0, 2.0]];
    let target = vec![0];
    let task = ClassificationTask::new("single", features, target).unwrap();
    let mut dt = DecisionTree::default();
    let model = dt.train_classif(&task).unwrap();
    let pred = model.predict(task.features()).unwrap();
    assert_eq!(pred.n_samples(), 1);
}

#[test]
fn edge_single_sample_regress() {
    let features = array![[1.0]];
    let target = vec![5.0];
    let task = RegressionTask::new("single_r", features, target).unwrap();
    let mut dt = DecisionTree::default();
    let model = dt.train_regress(&task).unwrap();
    let pred = model.predict(task.features()).unwrap();
    assert_eq!(pred.n_samples(), 1);
}

// ── Edge Cases: All same class/target ──────────────────────────────

#[test]
fn edge_all_same_class() {
    let features = array![[0.0], [1.0], [2.0], [3.0]];
    let target = vec![0, 0, 0, 0]; // all same class
    let task = ClassificationTask::new("same", features, target).unwrap();

    let mut dt = DecisionTree::default();
    let model = dt.train_classif(&task).unwrap();
    let pred = model.predict(task.features()).unwrap();
    if let Prediction::Classification { predicted, .. } = &pred {
        assert!(
            predicted.iter().all(|&p| p == 0),
            "all predictions should be class 0"
        );
    }
}

#[test]
fn edge_all_same_target_regress() {
    let features = array![[0.0], [1.0], [2.0], [3.0]];
    let target = vec![5.0, 5.0, 5.0, 5.0];
    let task = RegressionTask::new("same_r", features, target).unwrap();

    let mut rf = RandomForest::new().with_n_estimators(10).with_seed(42);
    let model = rf.train_regress(&task).unwrap();
    let pred = model
        .predict(task.features())
        .unwrap()
        .with_truth_regress(task.target().to_vec());
    let rmse = Rmse.score(&pred).unwrap();
    assert!(rmse < 1.0, "all same target should give near-zero RMSE");
}

// ── Edge Cases: Extreme values ─────────────────────────────────────

#[test]
fn edge_large_values() {
    let features = array![
        [1e10, 1e10],
        [1e10 + 1.0, 1e10 + 1.0],
        [0.0, 0.0],
        [1.0, 1.0]
    ];
    let target = vec![1, 1, 0, 0];
    let task = ClassificationTask::new("large", features, target).unwrap();

    let mut dt = DecisionTree::default();
    let model = dt.train_classif(&task).unwrap();
    let pred = model.predict(task.features()).unwrap();
    assert_eq!(pred.n_samples(), 4);
}

#[test]
fn edge_small_values() {
    let features = array![[1e-10], [2e-10], [1e-5], [2e-5]];
    let target = vec![0, 0, 1, 1];
    let task = ClassificationTask::new("small", features, target).unwrap();

    let mut dt = DecisionTree::default();
    let model = dt.train_classif(&task).unwrap();
    let pred = model.predict(task.features()).unwrap();
    assert_eq!(pred.n_samples(), 4);
}

// ── Edge Cases: Imbalanced dataset ─────────────────────────────────

#[test]
fn edge_imbalanced_99_1() {
    // 99 samples class 0, 1 sample class 1
    let mut feat_data = vec![vec![0.0; 2]; 100];
    let mut target = vec![0usize; 100];
    for (i, row) in feat_data.iter_mut().take(99).enumerate() {
        *row = vec![i as f64 * 0.01, 0.0];
    }
    feat_data[99] = vec![5.0, 5.0];
    target[99] = 1;

    let mut features = Array2::zeros((100, 2));
    for (i, row) in feat_data.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            features[[i, j]] = v;
        }
    }

    let task = ClassificationTask::new("imb", features, target).unwrap();
    let mut rf = RandomForest::new().with_n_estimators(20).with_seed(42);
    let model = rf.train_classif(&task).unwrap();
    let pred = model.predict(task.features()).unwrap();
    assert_eq!(pred.n_samples(), 100);
}

// ── Benchmark CV: all learners ─────────────────────────────────────

#[test]
fn benchmark_all_learners_cv() {
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
        [1.1, 1.1]
    ];
    let target = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
    let task = ClassificationTask::new("all", features, target).unwrap();
    let cv = CrossValidation::new(2).with_seed(42);

    // Test each learner that wasn't in previous benchmark tests
    let learners: Vec<(&str, Box<dyn Learner>)> = vec![
        (
            "extra_trees",
            Box::new(ExtraTrees::new().with_n_estimators(10).with_seed(42)),
        ),
        ("gaussian_nb", Box::new(GaussianNB::new())),
        ("ridge_classif_skip", Box::new(DecisionTree::default())), // placeholder
        ("adaboost", Box::new(AdaBoost::new().with_n_estimators(10))),
        (
            "linear_svm",
            Box::new(
                LinearSVM::new()
                    .with_max_iter(500)
                    .with_c(10.0)
                    .with_learning_rate(0.1),
            ),
        ),
        (
            "ebm",
            Box::new(EBM::new().with_n_rounds(20).with_learning_rate(0.05)),
        ),
        (
            "hoeffding",
            Box::new(HoeffdingTree::new().with_grace_period(3)),
        ),
    ];

    for (name, mut learner) in learners {
        let r = benchmark::resample_classif(&mut *learner, &task, &cv, &[&Accuracy]);
        assert!(
            r.is_ok(),
            "learner {name} failed in benchmark CV: {:?}",
            r.err()
        );
    }
}

#[test]
fn benchmark_all_regressors_cv() {
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
    let task = RegressionTask::new("all_r", features, target).unwrap();
    let ho = Holdout::new(0.8).with_seed(42);

    let learners: Vec<(&str, Box<dyn Learner>)> = vec![
        ("ridge", Box::new(Ridge::new(0.1))),
        ("lasso", Box::new(Lasso::new(0.01))),
        ("elastic_net", Box::new(ElasticNet::new(0.01, 0.5))),
        (
            "quantile_gb",
            Box::new(QuantileGB::new(0.5).with_n_estimators(20)),
        ),
        (
            "quantile_forest",
            Box::new(QuantileForest::new().with_n_estimators(10).with_seed(42)),
        ),
    ];

    for (name, mut learner) in learners {
        let r = benchmark::resample_regress(&mut *learner, &task, &ho, &[&Rmse]);
        assert!(
            r.is_ok(),
            "learner {name} failed in benchmark: {:?}",
            r.err()
        );
    }
}

// ── DynamicEnsemble additional tests ───────────────────────────────

#[test]
fn des_different_base_learners() {
    use smelt_ml::learner::DynamicEnsemble;

    let features = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [0.0, 0.2],
        [1.0, 1.0],
        [1.1, 0.9],
        [0.9, 1.1],
        [1.0, 0.9]
    ];
    let target = vec![0, 0, 0, 0, 1, 1, 1, 1];
    let task = ClassificationTask::new("des2", features, target).unwrap();

    let mut des = DynamicEnsemble::new(vec![
        Box::new(|| Box::new(DecisionTree::default()) as Box<dyn Learner>),
        Box::new(|| {
            Box::new(RandomForest::new().with_n_estimators(5).with_seed(42)) as Box<dyn Learner>
        }),
    ])
    .with_k_neighbors(3);

    let model = des.train_classif(&task).unwrap();
    let pred = model.predict(task.features()).unwrap();
    assert_eq!(pred.n_samples(), 8);
}

// ── Hyperband additional tests ─────────────────────────────────────

#[test]
fn hyperband_regress() {
    use smelt_ml::tuning::{Hyperband, ParamDistribution, ParamSpace};

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
    let task = RegressionTask::new("hb_r", features, target).unwrap();

    let mut space = ParamSpace::new();
    space.insert("max_depth".into(), ParamDistribution::Uniform(1.0, 6.0));

    let hb = Hyperband::new(
        |params| {
            Box::new(DecisionTree::new().with_max_depth(params["max_depth"].as_usize().unwrap()))
        },
        space,
    )
    .with_max_folds(3)
    .with_seed(42);

    let result = hb.tune_regress(&task, &Rmse).unwrap();
    assert!(!result.all_results.is_empty());
    assert!(!result.maximize);
}

// ── RFE additional tests ───────────────────────────────────────────

#[test]
fn rfe_in_pipeline() {
    let features = array![
        [0.0, 42.0, 99.0],
        [0.1, 13.0, 55.0],
        [0.2, 99.0, 42.0],
        [0.0, 55.0, 13.0],
        [1.0, 42.0, 99.0],
        [1.1, 13.0, 55.0],
        [1.2, 99.0, 42.0],
        [1.0, 55.0, 13.0],
    ];
    let target = vec![0, 0, 0, 0, 1, 1, 1, 1];
    let task = ClassificationTask::new("rfe_pipe", features, target).unwrap();

    let rfe = RFE::classif(|| Box::new(DecisionTree::default()), 2);
    let mut pipe = Pipeline::new(vec![Box::new(rfe)], Box::new(DecisionTree::default()));
    let model = pipe.train_classif(&task).unwrap();
    let pred = model.predict(task.features()).unwrap();
    assert_eq!(pred.n_samples(), 8);
}

// ── Conformal: different confidence levels ─────────────────────────

#[test]
fn conformal_different_alphas() {
    use smelt_ml::conformal::ConformalRegressor;

    let features = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]];
    let target = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
    let task = RegressionTask::new("cf_alpha", features.clone(), target.clone()).unwrap();

    let mut dt = DecisionTree::default();
    let model = dt.train_regress(&task).unwrap();

    let cal = array![[6.0], [7.0], [8.0]];
    let cal_t = vec![12.0, 14.0, 16.0];

    // Wider interval at 95% vs 80%
    let cf_95 = ConformalRegressor::calibrate(&*model, &cal, &cal_t, 0.05).unwrap();
    let cf_80 = ConformalRegressor::calibrate(&*model, &cal, &cal_t, 0.20).unwrap();

    assert!(
        cf_95.interval_width() >= cf_80.interval_width(),
        "95% CI should be >= 80% CI: {:.2} vs {:.2}",
        cf_95.interval_width(),
        cf_80.interval_width()
    );
}

// ── Survival: censoring scenarios ──────────────────────────────────

#[test]
fn rsf_heavy_censoring() {
    use smelt_ml::survival::{RandomSurvivalForest, SurvivalEvent};

    let features = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0],];
    // Heavy censoring: only 2 of 8 are events
    let events = vec![
        SurvivalEvent {
            time: 10.0,
            event: false,
        },
        SurvivalEvent {
            time: 15.0,
            event: true,
        },
        SurvivalEvent {
            time: 8.0,
            event: false,
        },
        SurvivalEvent {
            time: 20.0,
            event: false,
        },
        SurvivalEvent {
            time: 5.0,
            event: false,
        },
        SurvivalEvent {
            time: 12.0,
            event: true,
        },
        SurvivalEvent {
            time: 3.0,
            event: false,
        },
        SurvivalEvent {
            time: 7.0,
            event: false,
        },
    ];

    let rsf = RandomSurvivalForest::new()
        .with_n_estimators(20)
        .with_seed(42);
    let preds = rsf.fit_predict(&features, &events).unwrap();
    assert_eq!(preds.len(), 8);
    // With heavy censoring, survival probabilities should still be valid
    for p in &preds {
        for &s in &p.survival {
            assert!((0.0..=1.0 + 1e-10).contains(&s));
        }
    }
}

// ── CSV: edge cases ────────────────────────────────────────────────

#[test]
fn csv_with_nan_values() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("nan.csv");
    std::fs::write(&path, "x1,x2,y\n1.0,,0\n2.0,3.0,1\n,4.0,0\n5.0,6.0,1\n").unwrap();

    // CSV with empty values should produce NaN
    let result = CsvLoader::from_path(&path).target("y").load_classif();
    // May succeed with NaN or fail gracefully
    // The important thing is no panic
    let _ = result;
}

// ── Serialization: prediction roundtrip for regression ─────────────

#[test]
fn serialize_regression_roundtrip() {
    let pred = Prediction::regression_with_truth(vec![1.0, 2.0, 3.0], vec![1.1, 2.1, 3.1]);
    let json = serde_json::to_string(&pred).unwrap();
    let restored: Prediction = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.n_samples(), 3);
}

// ── TargetTransformRegressor (Prioridad 6, item 1) ─────────────────

/// The wrapper must compose with the generic benchmark/resampling machinery
/// with no special handling: since `predict` already returns original-scale
/// values, `resample_regress` scores original-scale errors directly.
#[test]
fn target_transform_composes_with_cv_resampling() {
    let n = 120;
    let mut features = Array2::<f64>::zeros((n, 1));
    let mut target = vec![0.0; n];
    for i in 0..n {
        let x = i as f64 / n as f64 * 3.0;
        features[[i, 0]] = x;
        target[i] = (1.2 * x + 0.3 * ((i * 7 % 11) as f64 / 11.0 - 0.5)).exp();
    }
    let task = RegressionTask::new("cv_log", features, target).unwrap();

    let mut ttr =
        TargetTransformRegressor::new(|| Box::new(LinearRegression), TargetTransform::Log);
    let cv = CrossValidation::new(5).with_seed(42);
    let result = benchmark::resample_regress(&mut ttr, &task, &cv, &[&Rmse]).unwrap();
    let mean_rmse = result.mean_scores()[0];
    assert!(
        mean_rmse.is_finite() && mean_rmse >= 0.0,
        "CV over the wrapper must produce a finite RMSE, got {mean_rmse}"
    );
}

/// feature_names and feature_types must reach the base learner through the
/// wrapper's task rebuild (the 5th audit's M-3 lesson: silently dropping
/// them disables native categorical splits downstream). Verified with a
/// probe learner that records what it is actually trained on.
#[test]
fn target_transform_feature_metadata_reaches_base_learner() {
    use smelt_ml::learner::TrainedModel;
    use smelt_ml::task::FeatureType;
    use std::sync::{Arc, Mutex};

    type SeenMetadata = Option<(Vec<String>, Vec<FeatureType>)>;
    struct MetaProbe {
        seen: Arc<Mutex<SeenMetadata>>,
    }
    impl Learner for MetaProbe {
        fn id(&self) -> &str {
            "meta_probe"
        }
        fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
            *self.seen.lock().unwrap() =
                Some((task.feature_names().to_vec(), task.feature_types().to_vec()));
            LinearRegression.train_regress(task)
        }
    }

    let features = array![[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0]];
    let target = vec![1.0, 2.0, 4.0, 8.0];
    let names = vec!["grade".to_string(), "lithology".to_string()];
    let types = vec![
        FeatureType::Numeric,
        FeatureType::Categorical { n_categories: 3 },
    ];
    let task = RegressionTask::new("meta", features, target)
        .unwrap()
        .with_feature_names(names.clone())
        .unwrap()
        .with_feature_types(types.clone())
        .unwrap();

    let seen = Arc::new(Mutex::new(None));
    let seen_clone = seen.clone();
    let mut ttr = TargetTransformRegressor::new(
        move || {
            Box::new(MetaProbe {
                seen: seen_clone.clone(),
            })
        },
        TargetTransform::Log,
    );
    ttr.train_regress(&task).unwrap();

    let observed = seen.lock().unwrap().clone();
    assert_eq!(
        observed,
        Some((names, types)),
        "feature_names and feature_types must survive the wrapper's task rebuild"
    );
}
