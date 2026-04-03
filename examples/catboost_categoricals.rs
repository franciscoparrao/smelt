//! Test CatBoost accuracy on high-cardinality categoricals
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::Rng;
use smelt_ml::prelude::*;

fn main() {
    let n = 2000;
    let mut rng = StdRng::seed_from_u64(42);

    // 5 numeric features + 1 categorical (encoded as float 0-99)
    let mut features = Array2::zeros((n, 6));
    let mut target = vec![0usize; n];

    for i in 0..n {
        let cat_val = (rng.random::<f64>() * 100.0).floor();
        features[[i, 5]] = cat_val;
        for j in 0..5 {
            let u1: f64 = rng.random::<f64>().max(1e-15);
            let u2: f64 = rng.random::<f64>();
            features[[i, j]] = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        }
        target[i] = if cat_val > 50.0 { 1 } else { 0 };
    }
    // Add 10% noise
    for _ in 0..(n / 10) {
        let idx = (rng.random::<f64>() * n as f64) as usize % n;
        target[idx] = 1 - target[idx];
    }

    // Split 80/20
    let holdout = Holdout::new(0.8).with_seed(42);
    let splits = holdout.splits(n);
    let (train_idx, test_idx) = &splits[0];

    let train_features = features.select(ndarray::Axis(0), train_idx);
    let train_target: Vec<usize> = train_idx.iter().map(|&i| target[i]).collect();
    let test_features = features.select(ndarray::Axis(0), test_idx);
    let test_target: Vec<usize> = test_idx.iter().map(|&i| target[i]).collect();

    let train_task = ClassificationTask::new("train", train_features, train_target).unwrap();

    // With cat_features
    let mut cb = CatBoost::new()
        .with_n_estimators(100)
        .with_depth(6)
        .with_learning_rate(0.3)
        .with_cat_features(vec![5]);
    let model = cb.train_classif(&train_task).unwrap();
    let pred = model.predict(&test_features).unwrap()
        .with_truth_classif(test_target.clone());
    let acc = Accuracy.score(&pred).unwrap();
    println!("smelt CatBoost (cat_features=[5]): {:.3}", acc);

    // Without cat_features
    let mut cb2 = CatBoost::new()
        .with_n_estimators(100)
        .with_depth(6)
        .with_learning_rate(0.3);
    let model2 = cb2.train_classif(&train_task).unwrap();
    let pred2 = model2.predict(&test_features).unwrap()
        .with_truth_classif(test_target.clone());
    let acc2 = Accuracy.score(&pred2).unwrap();
    println!("smelt CatBoost (numeric only):    {:.3}", acc2);
}
