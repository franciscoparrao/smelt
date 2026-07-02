//! Small numeric helpers shared across the gradient boosting engines
//! (logistic link / softmax for binary and multiclass objectives).
//!
//! Previously duplicated verbatim in xgboost.rs, lightgbm.rs, catboost.rs,
//! gradient_boosting.rs, ebm.rs and logistic_regression.rs.

/// Logistic sigmoid: maps a real-valued score to (0, 1).
#[inline]
pub(crate) fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Numerically-stable softmax (subtracts the max score before exponentiating).
pub(crate) fn softmax(scores: &[f64]) -> Vec<f64> {
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|&s| (s - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_bounds_and_midpoint() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-12);
        assert!(sigmoid(100.0) > 0.999);
        assert!(sigmoid(-100.0) < 0.001);
    }

    #[test]
    fn softmax_sums_to_one_and_matches_ratios() {
        let probs = softmax(&[1.0, 2.0, 3.0]);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
        assert!(probs[2] > probs[1] && probs[1] > probs[0]);
    }

    #[test]
    fn softmax_is_shift_invariant() {
        let a = softmax(&[1.0, 2.0, 3.0]);
        let b = softmax(&[1001.0, 1002.0, 1003.0]);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-9);
        }
    }
}
