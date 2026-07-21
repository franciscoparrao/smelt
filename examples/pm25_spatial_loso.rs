//! PM2.5 Spatial Interpolation — Leave-One-Station-Out CV (Regression Kriging)
//!
//! Case-study handoff for the Smelt team. Reproduces the *spatial track* of
//! Parra & Astudillo (2025), "Machine Learning for PM2.5 Prediction in Santiago
//! de Chile" (target: Environmental Modelling & Software):
//!
//!   Predict daily PM2.5 at an *unmonitored* location from spatial covariates,
//!   using Regression Kriging (XGBoost trend + kriged residuals), validated
//!   station-by-station with Leave-One-Station-Out CV (LOSO).
//!
//! The original paper implements this in Python (XGBoost + a hand-rolled
//! variogram) and reports a mean LOSO R2 = -1.09 (spatial interpolation fails
//! at 7-10 km satellite resolution). This prototype shows the *same experiment*
//! expressed natively in smelt-ml:
//!
//!   CsvLoader -> KrigingHybrid (XGBoost trend + spherical variogram)
//!             -> GroupCV (station = group, n_folds = n_stations => LOSO)
//!             -> RSquared / Rmse / Mae
//!             -> SplitConformal for calibrated intervals over the kriging model
//!
//! Data: data/pm25_santiago_spatial.csv  (16,344 daily station-obs, 8 stations,
//! 10 spatial covariates + target `pm25`, integer `station_id`).
//!
//! Run: cargo run --release --example pm25_spatial_loso

use ndarray::Axis;
use smelt_ml::conformal::SplitConformal;
use smelt_ml::data::CsvLoader;
use smelt_ml::prelude::*;

/// Station id -> name (order matches the `station_id` column in the CSV).
const STATIONS: [&str; 8] = [
    "Cerrillos II",
    "Cerro Navia",
    "El Bosque",
    "Independencia",
    "Las Condes",
    "Parque O'Higgins",
    "Pudahuel",
    "Talagante",
];

/// Build an XGBoost base learner with the paper's hyperparameters.
fn xgb() -> XGBoost {
    XGBoost::new()
        .with_n_estimators(200)
        .with_max_depth(6)
        .with_learning_rate(0.05)
}

fn main() {
    // ── Load ────────────────────────────────────────────────────────────────
    let task = CsvLoader::from_path("data/pm25_santiago_spatial.csv")
        .target("pm25")
        .load_regress()
        .expect("load data/pm25_santiago_spatial.csv");

    let names = task.feature_names();
    let sid = names.iter().position(|n| n == "station_id").unwrap();
    let lon_i = names.iter().position(|n| n == "lon").unwrap();
    let lat_i = names.iter().position(|n| n == "lat").unwrap();

    let n = task.n_samples();
    let feats_all = task.features();
    let target = task.target().to_vec();

    // groups = station id per row (drives LOSO); coords = (x=lon, y=lat) for kriging
    let groups: Vec<usize> = (0..n).map(|i| feats_all[[i, sid]] as usize).collect();
    let coords: Vec<(f64, f64)> = (0..n)
        .map(|i| (feats_all[[i, lon_i]], feats_all[[i, lat_i]]))
        .collect();

    // Feature matrix WITHOUT the `station_id` identifier (keep lat/lon as covariates,
    // exactly as the paper's XGBoost trend model does).
    let keep: Vec<usize> = (0..feats_all.ncols()).filter(|&j| j != sid).collect();
    let features = feats_all.select(Axis(1), &keep);

    let n_stations = groups.iter().copied().max().unwrap() + 1;

    // ── LOSO-CV: GroupCV with n_folds == n_stations => each fold leaves one station out ──
    let loso = GroupCV::new(n_stations, groups.clone());
    let splits = loso.splits(n).unwrap();

    println!("Leave-One-Station-Out CV — PM2.5 spatial interpolation, Santiago");
    println!("Regression Kriging (XGBoost trend + spherical variogram) vs plain XGBoost\n");
    println!(
        "{:<18} {:>6} {:>10} {:>10} {:>10}",
        "Held-out station", "n", "RK R2", "XGB R2", "RK RMSE"
    );
    println!("{}", "-".repeat(58));

    let mut rk_r2s = Vec::new();
    for (tr, te) in &splits {
        let tr_feat = features.select(Axis(0), tr).to_owned();
        let tr_tgt: Vec<f64> = tr.iter().map(|&i| target[i]).collect();
        let tr_coords: Vec<(f64, f64)> = tr.iter().map(|&i| coords[i]).collect();

        let te_feat = features.select(Axis(0), te).to_owned();
        let te_tgt: Vec<f64> = te.iter().map(|&i| target[i]).collect();
        let te_coords: Vec<(f64, f64)> = te.iter().map(|&i| coords[i]).collect();

        let tr_task = RegressionTask::new("loso_train", tr_feat, tr_tgt).unwrap();

        // (a) Regression Kriging = XGBoost trend + kriged residuals (the paper's method)
        let mut rk = KrigingHybrid::new(|| Box::new(xgb()) as Box<dyn Learner>, tr_coords)
            .with_variogram_model(VariogramModel::Spherical)
            .with_n_neighbors(20);
        let rk_model = rk.train_regress_geo(&tr_task).unwrap();
        let rk_pred = rk_model.predict_spatial(&te_feat, &te_coords).unwrap();
        let rk_truth = rk_pred.with_truth_regress(te_tgt.clone());
        let rk_r2 = RSquared.score(&rk_truth).unwrap();
        let rk_rmse = Rmse.score(&rk_truth).unwrap();

        // (b) Plain XGBoost baseline (no residual kriging) for contrast
        let mut base = xgb();
        let base_model = base.train_regress(&tr_task).unwrap();
        let base_pred = base_model.predict(&te_feat).unwrap();
        let base_r2 = RSquared
            .score(&base_pred.with_truth_regress(te_tgt.clone()))
            .unwrap();

        let st = STATIONS[groups[te[0]]];
        println!(
            "{:<18} {:>6} {:>10.3} {:>10.3} {:>10.2}",
            st,
            te.len(),
            rk_r2,
            base_r2,
            rk_rmse
        );
        rk_r2s.push(rk_r2);
    }
    let mean_r2 = rk_r2s.iter().sum::<f64>() / rk_r2s.len() as f64;
    let median = {
        let mut v = rk_r2s.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.0
    };
    println!("{}", "-".repeat(58));
    println!("Mean LOSO R2 (Regression Kriging) = {:.3}", mean_r2);
    println!("Median LOSO R2                     = {:.3}", median);
    println!("(Paper reports mean LOSO R2 = -1.09: spatial interpolation fails at 7-10 km.)\n");

    // ── Conformal prediction over the KRIGING model itself ──────────────────
    // The paper's quantile intervals were undercalibrated (61% coverage vs 90%
    // nominal). TrainedKrigingHybrid's real predictor is predict_spatial
    // (needs coords), which ConformalRegressor can't drive — SplitConformal
    // calibrates from precomputed predictions instead, closing handoff gap #1:
    // the intervals below are calibrated against the actual spatial model.
    println!("─── Conformal intervals (90% target) over Regression Kriging ───");
    let (tr, te) = &splits[0];
    // Split the training rows into fit (75%) + calibration (25%). The CSV is
    // station-ordered, so this index cut holds out whole calibration
    // stations — matching the LOSO regime the intervals will face.
    let cut = tr.len() * 3 / 4;
    let (fit_idx, cal_idx) = tr.split_at(cut);

    let fit_task = RegressionTask::new(
        "fit",
        features.select(Axis(0), fit_idx).to_owned(),
        fit_idx.iter().map(|&i| target[i]).collect(),
    )
    .unwrap();
    let fit_coords: Vec<(f64, f64)> = fit_idx.iter().map(|&i| coords[i]).collect();
    let mut rk = KrigingHybrid::new(|| Box::new(xgb()) as Box<dyn Learner>, fit_coords)
        .with_variogram_model(VariogramModel::Spherical)
        .with_n_neighbors(20);
    let rk_model = rk.train_regress_geo(&fit_task).unwrap();

    let cal_feat = features.select(Axis(0), cal_idx).to_owned();
    let cal_tgt: Vec<f64> = cal_idx.iter().map(|&i| target[i]).collect();
    let cal_coords: Vec<(f64, f64)> = cal_idx.iter().map(|&i| coords[i]).collect();
    let cal_pred = match rk_model.predict_spatial(&cal_feat, &cal_coords).unwrap() {
        Prediction::Regression { predicted, .. } => predicted,
        _ => unreachable!("kriging predicts regression"),
    };
    let sc = SplitConformal::calibrate_from_predictions(&cal_pred, &cal_tgt, 0.1).unwrap();

    let te_feat = features.select(Axis(0), te).to_owned();
    let te_tgt: Vec<f64> = te.iter().map(|&i| target[i]).collect();
    let te_coords_cf: Vec<(f64, f64)> = te.iter().map(|&i| coords[i]).collect();
    let te_pred = match rk_model.predict_spatial(&te_feat, &te_coords_cf).unwrap() {
        Prediction::Regression { predicted, .. } => predicted,
        _ => unreachable!("kriging predicts regression"),
    };
    let intervals = sc.intervals_for(&te_pred);
    let mut covered = 0usize;
    for (iv, &t) in intervals.iter().zip(&te_tgt) {
        if t >= iv.lower && t <= iv.upper {
            covered += 1;
        }
    }
    println!(
        "  Held-out {}: empirical coverage {:.0}% (target 90%), n = {}, half-width ±{:.1} µg/m³",
        STATIONS[groups[te[0]]],
        100.0 * covered as f64 / te_tgt.len() as f64,
        te_tgt.len(),
        sc.interval_width(),
    );
}
