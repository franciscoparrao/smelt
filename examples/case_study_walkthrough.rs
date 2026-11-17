//! King County walkthrough: the complete spatial pipeline in one listing.
//!
//! This is the code shown in the manuscript's case-study section, verbatim:
//! it loads the data, georeferences the task, compares Geo-XGBoost with a
//! global XGBoost under a hold-out and under random vs spatial-block CV
//! (with the residual Moran's I next to the RMSE), and wraps the spatial
//! predictions in split-conformal intervals. Every number printed here is a
//! number quoted in the paper.
//!
//! Run with: cargo run --release --example case_study_walkthrough

use ndarray::Axis;
use smelt_ml::prelude::*;

fn main() -> Result<()> {
    // 1. Data: 1,000 King County sales; coordinates are projected metres
    //    (UTM 10N), kept as task metadata, NOT as model features.
    let raw = CsvLoader::from_path("data/king_county_1k_utm.csv")
        .target("log_price")
        .load_regress()?;
    let names = raw.feature_names();
    let col = |n: &str| names.iter().position(|c| c == n).unwrap();
    let (ix, iy) = (col("X"), col("Y"));
    let coords: Vec<(f64, f64)> = (0..raw.n_samples())
        .map(|i| (raw.features()[[i, ix]], raw.features()[[i, iy]]))
        .collect();
    let keep: Vec<usize> = (0..raw.n_features())
        .filter(|&j| ![ix, iy, col("lat"), col("long")].contains(&j))
        .collect();
    let task = RegressionTask::new("king_county", raw.features().select(Axis(1), &keep), raw.target().to_vec())?
        .with_feature_names(keep.iter().map(|&j| names[j].clone()).collect())?
        .with_coords(coords.clone());
    println!("{} samples, {} features, coordinates attached to the task", task.n_samples(), task.n_features());

    // 2. Learners. Geo-XGBoost reads the coordinates from the task.
    let xgb = || XGBoost::new().with_n_estimators(100).with_max_depth(4).with_learning_rate(0.1);
    let geo = || {
        GeoXGBoost::coords_from_task()
            .with_bandwidth(50)
            .with_alpha(0.5)
            .with_n_estimators(100)
            .with_max_depth(4)
            .with_learning_rate(0.1)
    };

    // 3. Hold-out 80/20. The same slicing serves every learner; resample_regress
    //    does it internally, here it is written out once for the conformal step.
    let (train_idx, test_idx) = &Holdout::new(0.8).with_seed(42).splits(task.n_samples())?[0];
    let slice = |idx: &[usize]| -> Result<RegressionTask> {
        Ok(RegressionTask::new("split", task.features().select(Axis(0), idx), idx.iter().map(|&i| task.target()[i]).collect())?
            .with_coords(idx.iter().map(|&i| coords[i]).collect()))
    };
    let (train, test) = (slice(train_idx)?, slice(test_idx)?);
    let test_coords = test.coords().unwrap();
    let rmse_of = |pred: Prediction| Rmse.score(&pred.with_truth_regress(test.target().to_vec()));

    let xgb_model = xgb().train_regress(&train)?;
    let geo_model = geo().train_regress(&train)?;
    let geo_pred = geo_model.predict_spatial(test.features(), test_coords)?;
    println!("\nHold-out RMSE (log price)");
    println!("  XGBoost, no coordinates      {:.3}", rmse_of(xgb_model.predict(test.features())?)?);
    println!("  Geo-XGBoost, k = 50, alpha 0.5 {:.3}", rmse_of(geo_pred.clone())?);
    println!("  Geo-XGBoost global part only {:.3}", rmse_of(geo_model.predict(test.features())?)?);

    // 4. Random vs spatial-block CV, same call, same measures. The residual
    //    Moran's I says how much spatial structure each model leaves behind.
    let moran = MoranResiduals::coords_from_task();
    let measures: [&dyn Measure; 2] = [&Rmse, &moran];
    let random_cv = CrossValidation::new(5).with_seed(42);
    let block_cv = SpatialBlockCV::new(5, coords.clone());
    println!("\n5-fold CV: RMSE / residual Moran's I");
    for (name, cv) in [("random folds ", &random_cv as &dyn Resample), ("spatial blocks", &block_cv)] {
        let x = benchmark::resample_regress(&mut xgb(), &task, cv, &measures)?.mean_scores();
        let g = benchmark::resample_regress(&mut geo(), &task, cv, &measures)?.mean_scores();
        println!("  {name}   XGBoost {:.3} / {:+.3}   Geo-XGBoost {:.3} / {:+.3}", x[0], x[1], g[0], g[1]);
    }

    // 5. Split-conformal intervals around the spatial predictions: calibrate on
    //    a quarter of the training set, evaluate on the hold-out.
    let (fit_idx, cal_idx) = &Holdout::new(0.75).with_seed(7).splits(train.n_samples())?[0];
    let to_global = |local: &[usize]| local.iter().map(|&i| train_idx[i]).collect::<Vec<_>>();
    let (fit, cal) = (slice(&to_global(fit_idx))?, slice(&to_global(cal_idx))?);
    let model = geo().train_regress(&fit)?;
    let values = |p: Prediction| match p {
        Prediction::Regression { predicted, .. } => predicted,
        _ => unreachable!(),
    };
    let cal_pred = values(model.predict_spatial(cal.features(), cal.coords().unwrap())?);
    let conformal = SplitConformal::calibrate_from_predictions(&cal_pred, cal.target(), 0.10)?;
    let intervals = conformal.intervals_for(&values(model.predict_spatial(test.features(), test_coords)?));
    let covered = intervals.iter().zip(test.target()).filter(|(iv, y)| iv.lower <= **y && **y <= iv.upper).count();
    let width = intervals.iter().map(|iv| iv.upper - iv.lower).sum::<f64>() / intervals.len() as f64;
    println!("\nSplit-conformal 90% intervals on Geo-XGBoost: coverage {:.1}% ({}/{}), mean width {:.2} log($)",
        100.0 * covered as f64 / intervals.len() as f64, covered, intervals.len(), width);
    Ok(())
}
