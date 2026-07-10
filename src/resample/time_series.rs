//! Rolling-origin (walk-forward) cross-validation for temporal data.

use super::Resample;
use crate::{Result, SmeltError};

/// Rolling-origin / walk-forward cross-validation for time-ordered data.
///
/// Samples are assumed to be **sorted by time** (index 0 = oldest); every
/// split trains strictly on the past and tests on the future, so there is
/// no look-ahead leakage — the temporal analogue of what `SpatialBlockCV`/
/// `SpatialBufferCV` do for spatial leakage. This is scikit-learn's
/// `TimeSeriesSplit` / the forecasting literature's "rolling origin"
/// scheme (Tashman 2000).
///
/// Each split's test window is `horizon` consecutive samples; the origin
/// (first test index) starts at `min_train_size` and advances by `step`
/// per split. The training window is everything before the origin
/// (expanding, the default) or the most recent `max_window` samples
/// (sliding, via [`with_sliding_window`](TimeSeriesCV::with_sliding_window)).
/// An optional `gap` leaves an embargo of samples between the end of the
/// training window and the start of the test window (useful when the
/// target is a lagged/rolling aggregate that would otherwise leak across
/// the boundary).
///
/// ```text
/// n = 10, horizon = 2, min_train_size = 4, step = 2 (expanding):
///   split 0: train [0..4)  test [4..6)
///   split 1: train [0..6)  test [6..8)
///   split 2: train [0..8)  test [8..10)
/// ```
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
///
/// let cv = TimeSeriesCV::new(2).with_min_train_size(4);
/// let splits = cv.splits(10).unwrap();
/// assert_eq!(splits.len(), 3);
/// assert_eq!(splits[0].0, vec![0, 1, 2, 3]); // train: all of the past
/// assert_eq!(splits[0].1, vec![4, 5]);       // test: the next horizon
/// ```
pub struct TimeSeriesCV {
    /// Length of each test window (forecast horizon), in samples.
    pub horizon: usize,
    /// Minimum training-window length before the first test window.
    pub min_train_size: usize,
    /// How far the origin advances between consecutive splits.
    pub step: usize,
    /// `Some(w)`: sliding training window of at most `w` samples;
    /// `None`: expanding window (train on everything before the origin).
    pub max_window: Option<usize>,
    /// Embargo: samples excluded between train end and test start.
    pub gap: usize,
}

impl TimeSeriesCV {
    /// Walk-forward CV with the given forecast `horizon`. Defaults:
    /// `min_train_size = horizon`, `step = horizon` (contiguous,
    /// non-overlapping test windows), expanding window, no gap.
    pub fn new(horizon: usize) -> Self {
        Self {
            horizon,
            min_train_size: horizon,
            step: horizon,
            max_window: None,
            gap: 0,
        }
    }

    /// Sets the minimum training-window length before the first test window.
    pub fn with_min_train_size(mut self, n: usize) -> Self {
        self.min_train_size = n;
        self
    }

    /// Sets how far the origin advances between consecutive splits
    /// (`step < horizon` gives overlapping test windows; `step > horizon`
    /// skips samples between them).
    pub fn with_step(mut self, step: usize) -> Self {
        self.step = step;
        self
    }

    /// Switches from an expanding to a sliding training window of at most
    /// `window` samples (the most recent ones before the origin).
    pub fn with_sliding_window(mut self, window: usize) -> Self {
        self.max_window = Some(window);
        self
    }

    /// Sets an embargo of `gap` samples between the end of the training
    /// window and the start of the test window.
    pub fn with_gap(mut self, gap: usize) -> Self {
        self.gap = gap;
        self
    }
}

impl Resample for TimeSeriesCV {
    fn splits(&self, n_samples: usize) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        if self.horizon == 0 {
            return Err(SmeltError::InvalidParameter(
                "TimeSeriesCV horizon must be at least 1".into(),
            ));
        }
        if self.step == 0 {
            return Err(SmeltError::InvalidParameter(
                "TimeSeriesCV step must be at least 1".into(),
            ));
        }
        if self.min_train_size == 0 {
            return Err(SmeltError::InvalidParameter(
                "TimeSeriesCV min_train_size must be at least 1".into(),
            ));
        }
        if self.max_window == Some(0) {
            return Err(SmeltError::InvalidParameter(
                "TimeSeriesCV sliding window must be at least 1 sample".into(),
            ));
        }

        let needed = self.min_train_size + self.gap + self.horizon;
        if n_samples < needed {
            return Err(SmeltError::InvalidParameter(format!(
                "TimeSeriesCV needs at least min_train_size + gap + horizon = {needed} \
                 samples for one split, got {n_samples}"
            )));
        }

        let mut splits = Vec::new();
        // `origin` = index one past the end of the training window.
        let mut origin = self.min_train_size;
        while origin + self.gap + self.horizon <= n_samples {
            let train_start = match self.max_window {
                Some(w) => origin.saturating_sub(w),
                None => 0,
            };
            let test_start = origin + self.gap;
            splits.push((
                (train_start..origin).collect(),
                (test_start..test_start + self.horizon).collect(),
            ));
            origin += self.step;
        }
        Ok(splits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expanding_window_matches_hand_computed_splits() {
        let cv = TimeSeriesCV::new(2).with_min_train_size(4);
        let splits = cv.splits(10).unwrap();
        assert_eq!(
            splits,
            vec![
                ((0..4).collect::<Vec<_>>(), vec![4, 5]),
                ((0..6).collect(), vec![6, 7]),
                ((0..8).collect(), vec![8, 9]),
            ]
        );
    }

    /// The whole point of walk-forward CV: no test index may ever precede
    /// (or touch, given a gap) a training index of its own split.
    #[test]
    fn train_always_strictly_precedes_test() {
        for gap in [0usize, 3] {
            let cv = TimeSeriesCV::new(5)
                .with_min_train_size(10)
                .with_step(3)
                .with_gap(gap);
            let splits = cv.splits(60).unwrap();
            assert!(!splits.is_empty());
            for (train, test) in &splits {
                let train_max = *train.last().unwrap();
                let test_min = test[0];
                assert!(
                    train_max + gap < test_min,
                    "gap={gap}: train end {train_max} must precede test start {test_min} \
                     by more than the embargo"
                );
                assert_eq!(test.len(), 5);
            }
        }
    }

    #[test]
    fn sliding_window_caps_train_length() {
        let cv = TimeSeriesCV::new(2)
            .with_min_train_size(4)
            .with_sliding_window(3);
        let splits = cv.splits(10).unwrap();
        // First origin is 4: window of 3 -> train [1..4)
        assert_eq!(splits[0].0, vec![1, 2, 3]);
        for (train, _) in &splits {
            assert!(train.len() <= 3);
        }
    }

    #[test]
    fn overlapping_and_skipping_steps() {
        // step < horizon: overlapping test windows
        let cv = TimeSeriesCV::new(4).with_min_train_size(4).with_step(2);
        let splits = cv.splits(12).unwrap();
        assert_eq!(splits[0].1, vec![4, 5, 6, 7]);
        assert_eq!(splits[1].1, vec![6, 7, 8, 9]);
        // step > horizon: samples skipped between test windows
        let cv = TimeSeriesCV::new(2).with_min_train_size(4).with_step(4);
        let splits = cv.splits(12).unwrap();
        assert_eq!(splits[0].1, vec![4, 5]);
        assert_eq!(splits[1].1, vec![8, 9]);
    }

    #[test]
    fn rejects_degenerate_configs_and_too_few_samples() {
        assert!(TimeSeriesCV::new(0).splits(10).is_err());
        assert!(TimeSeriesCV::new(2).with_step(0).splits(10).is_err());
        assert!(TimeSeriesCV::new(2).with_min_train_size(0).splits(10).is_err());
        assert!(TimeSeriesCV::new(2).with_sliding_window(0).splits(10).is_err());
        // min_train(4) + gap(2) + horizon(2) = 8 > 7
        assert!(
            TimeSeriesCV::new(2)
                .with_min_train_size(4)
                .with_gap(2)
                .splits(7)
                .is_err()
        );
    }

    /// A model evaluated with walk-forward CV must never see the future:
    /// with a target that leaks trivially from a future-looking feature,
    /// expanding-window CV keeps the test truly out-of-sample.
    #[test]
    fn composes_with_the_resample_trait_object() {
        let cv: Box<dyn Resample> = Box::new(TimeSeriesCV::new(3).with_min_train_size(6));
        let splits = cv.splits(15).unwrap();
        assert_eq!(splits.len(), 3);
    }
}
