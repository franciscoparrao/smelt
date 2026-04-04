//! Accuracy validation on medium-scale datasets.
//!
//! Compares smelt-ml against scikit-learn reference values using 5-fold CV
//! with seed 42 on Wine, Breast Cancer, Digits, and California Housing.
//!
//! Run with: cargo run --release --example accuracy_validation

use ndarray::Axis;
use smelt_ml::prelude::*;
use smelt_ml::data::CsvLoader;

fn mean(v: &[f64]) -> f64 { v.iter().sum::<f64>() / v.len() as f64 }
fn std_dev(v: &[f64]) -> f64 {
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}

fn cv_classif(
    features: &ndarray::Array2<f64>,
    target: &[usize],
    learner_fn: impl Fn() -> Box<dyn smelt_ml::learner::Learner>,
    name: &str,
) -> f64 {
    let cv = CrossValidation::new(5).with_seed(42);
    let task = ClassificationTask::new("cv", features.clone(), target.to_vec()).unwrap();
    let splits = cv.splits(task.n_samples());

    let mut scores = Vec::new();
    for (train_idx, test_idx) in &splits {
        let tr_feat = features.select(Axis(0), train_idx).to_owned();
        let tr_tgt: Vec<usize> = train_idx.iter().map(|&i| target[i]).collect();
        let te_feat = features.select(Axis(0), test_idx).to_owned();
        let te_tgt: Vec<usize> = test_idx.iter().map(|&i| target[i]).collect();

        let tr_task = ClassificationTask::new("fold", tr_feat, tr_tgt).unwrap();
        let mut learner = learner_fn();
        let model = learner.train_classif(&tr_task).unwrap();
        let pred = model.predict(&te_feat).unwrap();
        let acc = Accuracy.score(&pred.with_truth_classif(te_tgt)).unwrap();
        scores.push(acc);
    }
    let m = mean(&scores);
    let s = std_dev(&scores);
    println!("  {:<25} {:.3} +/- {:.3}", name, m, s);
    m
}

fn cv_regress(
    features: &ndarray::Array2<f64>,
    target: &[f64],
    learner_fn: impl Fn() -> Box<dyn smelt_ml::learner::Learner>,
    name: &str,
) -> f64 {
    let cv = CrossValidation::new(5).with_seed(42);
    let splits = cv.splits(features.nrows());

    let mut scores = Vec::new();
    for (train_idx, test_idx) in &splits {
        let tr_feat = features.select(Axis(0), train_idx).to_owned();
        let tr_tgt: Vec<f64> = train_idx.iter().map(|&i| target[i]).collect();
        let te_feat = features.select(Axis(0), test_idx).to_owned();
        let te_tgt: Vec<f64> = test_idx.iter().map(|&i| target[i]).collect();

        let tr_task = RegressionTask::new("fold", tr_feat, tr_tgt).unwrap();
        let mut learner = learner_fn();
        let model = learner.train_regress(&tr_task).unwrap();
        let pred = model.predict(&te_feat).unwrap();
        let rmse = Rmse.score(&pred.with_truth_regress(te_tgt)).unwrap();
        scores.push(rmse);
    }
    let m = mean(&scores);
    let s = std_dev(&scores);
    println!("  {:<25} {:.4} +/- {:.4}", name, m, s);
    m
}

fn main() {
    println!("==========================================================");
    println!("  smelt-ml Accuracy Validation (5-fold CV, seed=42)");
    println!("==========================================================");

    // ── Classification: Wine (178 samples, 13 features, 3 classes) ──
    let wine = CsvLoader::from_path("data/wine.csv")
        .target("target")
        .load_classif()
        .unwrap();
    println!("\nWine ({} samples, {} features)", wine.n_samples(), wine.features().ncols());
    let wf = wine.features().to_owned();
    let wt = wine.target().to_vec();
    cv_classif(&wf, &wt, || Box::new(DecisionTree::default()), "Decision Tree");
    cv_classif(&wf, &wt, || Box::new(RandomForest::new().with_n_estimators(100).with_seed(42)), "Random Forest");
    cv_classif(&wf, &wt, || Box::new(LogisticRegression::new()), "Logistic Regression");

    // ── Classification: Breast Cancer (569 samples, 30 features, 2 classes) ──
    let bc = CsvLoader::from_path("data/breast_cancer.csv")
        .target("target")
        .load_classif()
        .unwrap();
    println!("\nBreast Cancer ({} samples, {} features)", bc.n_samples(), bc.features().ncols());
    let bf = bc.features().to_owned();
    let bt = bc.target().to_vec();
    cv_classif(&bf, &bt, || Box::new(DecisionTree::default()), "Decision Tree");
    cv_classif(&bf, &bt, || Box::new(RandomForest::new().with_n_estimators(100).with_seed(42)), "Random Forest");
    cv_classif(&bf, &bt, || Box::new(LogisticRegression::new()), "Logistic Regression");

    // ── Classification: Digits (1,797 samples, 64 features, 10 classes) ──
    let digits = CsvLoader::from_path("data/digits.csv")
        .target("target")
        .load_classif()
        .unwrap();
    println!("\nDigits ({} samples, {} features, 10 classes)", digits.n_samples(), digits.features().ncols());
    let df = digits.features().to_owned();
    let dt = digits.target().to_vec();
    cv_classif(&df, &dt, || Box::new(DecisionTree::default()), "Decision Tree");
    cv_classif(&df, &dt, || Box::new(RandomForest::new().with_n_estimators(100).with_seed(42)), "Random Forest");

    // ── Regression: California Housing (20,640 samples, 8 features) ──
    let cal = CsvLoader::from_path("data/california_housing.csv")
        .target("target")
        .load_regress()
        .unwrap();
    println!("\nCalifornia Housing ({} samples, {} features)", cal.n_samples(), cal.features().ncols());
    let cf = cal.features().to_owned();
    let ct = cal.target().to_vec();
    cv_regress(&cf, &ct, || Box::new(DecisionTree::default()), "Decision Tree");
    cv_regress(&cf, &ct, || Box::new(RandomForest::new().with_n_estimators(100).with_seed(42)), "Random Forest");
    cv_regress(&cf, &ct, || Box::new(Ridge::new(1.0)), "Ridge Regression");

    println!();
}
