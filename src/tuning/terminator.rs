//! Composable early-stopping criteria for sequential tuning.
//!
//! A [`Terminator`] decides, after each evaluation, whether a tuning run
//! should stop before exhausting its iteration budget — the mlr3 `Terminator`
//! concept. Concrete criteria ([`MaxEvals`], [`RunTime`], [`Stagnation`],
//! [`TargetScore`]) compose through [`AnyTerminator`] (stop if *any* fires)
//! and [`AllTerminator`] (stop only when *all* fire), so a run can be bounded
//! by e.g. "20 evaluations OR 30 seconds OR no improvement for 5 rounds" at
//! once.
//!
//! Attached to a sequential optimizer via
//! [`BayesianOptimizer::with_terminator`](crate::tuning::BayesianOptimizer::with_terminator);
//! the parallel batch tuners (`GridSearch`/`RandomSearch`) evaluate every
//! candidate up front, so there is no partial run for a terminator to cut
//! short there.

use std::time::Duration;

/// Snapshot of a tuning run's progress, passed to a [`Terminator`] after each
/// evaluation.
#[derive(Debug, Clone, Copy)]
pub struct TuningProgress {
    /// Number of configurations evaluated so far (>= 1 when checked).
    pub n_evals: usize,
    /// Wall-clock time elapsed since the run started.
    pub elapsed: Duration,
    /// Best score seen so far (direction given by `maximize`).
    pub best_score: f64,
    /// Whether higher scores are better.
    pub maximize: bool,
    /// Consecutive evaluations since the best score last improved.
    pub evals_since_improvement: usize,
}

/// A stopping criterion for a sequential tuning run.
///
/// `Send + Sync` so an optimizer holding a `Box<dyn Terminator>` stays shareable
/// across threads, matching the rest of the tuning types.
///
/// # Examples
///
/// ```
/// use smelt_ml::tuning::{AnyTerminator, MaxEvals, RunTime, Stagnation, Terminator};
///
/// // Stop at 50 evals, OR 30 seconds, OR 5 rounds without improvement.
/// let stop: Box<dyn Terminator> = Box::new(AnyTerminator::new(vec![
///     Box::new(MaxEvals::new(50)),
///     Box::new(RunTime::seconds(30.0)),
///     Box::new(Stagnation::new(5)),
/// ]));
/// // Attach to a BayesianOptimizer via `.with_terminator(stop)`.
/// let _ = stop;
/// ```
pub trait Terminator: Send + Sync {
    /// Whether the run should stop now, given the progress so far.
    fn should_terminate(&self, progress: &TuningProgress) -> bool;
}

/// Stop after a fixed number of evaluations. (An early stop independent of the
/// optimizer's own `n_iter`, and the natural budget to combine with others.)
pub struct MaxEvals {
    /// Evaluation count at which to stop.
    pub max_evals: usize,
}

impl MaxEvals {
    /// Stop once `max_evals` configurations have been evaluated.
    pub fn new(max_evals: usize) -> Self {
        Self { max_evals }
    }
}

impl Terminator for MaxEvals {
    fn should_terminate(&self, progress: &TuningProgress) -> bool {
        progress.n_evals >= self.max_evals
    }
}

/// Stop once a wall-clock time budget is exceeded.
pub struct RunTime {
    /// Maximum elapsed time before stopping.
    pub max: Duration,
}

impl RunTime {
    /// Stop after `seconds` of wall-clock time.
    pub fn seconds(seconds: f64) -> Self {
        Self {
            max: Duration::from_secs_f64(seconds),
        }
    }

    /// Stop after a given [`Duration`].
    pub fn new(max: Duration) -> Self {
        Self { max }
    }
}

impl Terminator for RunTime {
    fn should_terminate(&self, progress: &TuningProgress) -> bool {
        progress.elapsed >= self.max
    }
}

/// Stop when the best score hasn't improved for `patience` consecutive
/// evaluations (a convergence / diminishing-returns cutoff).
pub struct Stagnation {
    /// Evaluations-without-improvement threshold.
    pub patience: usize,
}

impl Stagnation {
    /// Stop after `patience` consecutive non-improving evaluations.
    pub fn new(patience: usize) -> Self {
        Self { patience }
    }
}

impl Terminator for Stagnation {
    fn should_terminate(&self, progress: &TuningProgress) -> bool {
        progress.evals_since_improvement >= self.patience
    }
}

/// Stop as soon as the best score reaches a target (good enough — no need to
/// keep searching). Direction follows `progress.maximize`.
pub struct TargetScore {
    /// Score at (or beyond) which to stop.
    pub target: f64,
}

impl TargetScore {
    /// Stop once the best score reaches `target`.
    pub fn new(target: f64) -> Self {
        Self { target }
    }
}

impl Terminator for TargetScore {
    fn should_terminate(&self, progress: &TuningProgress) -> bool {
        if progress.maximize {
            progress.best_score >= self.target
        } else {
            progress.best_score <= self.target
        }
    }
}

/// Stop when **any** of the wrapped terminators fires (logical OR) — the usual
/// way to combine budgets, e.g. "N evaluations or T seconds, whichever first".
pub struct AnyTerminator {
    /// The wrapped criteria; the run stops if any returns `true`.
    pub terminators: Vec<Box<dyn Terminator>>,
}

impl AnyTerminator {
    /// Combine terminators so the run stops when any one fires.
    pub fn new(terminators: Vec<Box<dyn Terminator>>) -> Self {
        Self { terminators }
    }
}

impl Terminator for AnyTerminator {
    fn should_terminate(&self, progress: &TuningProgress) -> bool {
        self.terminators
            .iter()
            .any(|t| t.should_terminate(progress))
    }
}

/// Stop only when **all** of the wrapped terminators fire (logical AND). Empty
/// never terminates (there is nothing all-true to satisfy).
pub struct AllTerminator {
    /// The wrapped criteria; the run stops only if all return `true`.
    pub terminators: Vec<Box<dyn Terminator>>,
}

impl AllTerminator {
    /// Combine terminators so the run stops only when every one fires.
    pub fn new(terminators: Vec<Box<dyn Terminator>>) -> Self {
        Self { terminators }
    }
}

impl Terminator for AllTerminator {
    fn should_terminate(&self, progress: &TuningProgress) -> bool {
        !self.terminators.is_empty()
            && self
                .terminators
                .iter()
                .all(|t| t.should_terminate(progress))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn progress(
        n_evals: usize,
        secs: f64,
        best: f64,
        maximize: bool,
        stall: usize,
    ) -> TuningProgress {
        TuningProgress {
            n_evals,
            elapsed: Duration::from_secs_f64(secs),
            best_score: best,
            maximize,
            evals_since_improvement: stall,
        }
    }

    #[test]
    fn max_evals_fires_at_threshold() {
        let t = MaxEvals::new(5);
        assert!(!t.should_terminate(&progress(4, 0.0, 0.0, true, 0)));
        assert!(t.should_terminate(&progress(5, 0.0, 0.0, true, 0)));
        assert!(t.should_terminate(&progress(6, 0.0, 0.0, true, 0)));
    }

    #[test]
    fn runtime_fires_after_budget() {
        let t = RunTime::seconds(2.0);
        assert!(!t.should_terminate(&progress(1, 1.9, 0.0, true, 0)));
        assert!(t.should_terminate(&progress(1, 2.0, 0.0, true, 0)));
    }

    #[test]
    fn stagnation_fires_on_patience() {
        let t = Stagnation::new(3);
        assert!(!t.should_terminate(&progress(10, 0.0, 0.0, true, 2)));
        assert!(t.should_terminate(&progress(10, 0.0, 0.0, true, 3)));
    }

    #[test]
    fn target_score_respects_direction() {
        let maxi = TargetScore::new(0.95);
        assert!(maxi.should_terminate(&progress(1, 0.0, 0.96, true, 0)));
        assert!(!maxi.should_terminate(&progress(1, 0.0, 0.94, true, 0)));
        // Minimizing: stop when best drops to/below the target.
        let mini = TargetScore::new(0.1);
        assert!(mini.should_terminate(&progress(1, 0.0, 0.08, false, 0)));
        assert!(!mini.should_terminate(&progress(1, 0.0, 0.2, false, 0)));
    }

    #[test]
    fn any_terminator_is_or() {
        let t = AnyTerminator::new(vec![
            Box::new(MaxEvals::new(100)),
            Box::new(TargetScore::new(0.9)),
        ]);
        // Neither the eval budget nor... wait, target IS met -> fires.
        assert!(t.should_terminate(&progress(1, 0.0, 0.95, true, 0)));
        // Neither fires.
        assert!(!t.should_terminate(&progress(1, 0.0, 0.5, true, 0)));
    }

    #[test]
    fn all_terminator_is_and_and_empty_never_fires() {
        let t = AllTerminator::new(vec![
            Box::new(MaxEvals::new(5)),
            Box::new(Stagnation::new(3)),
        ]);
        // Only evals threshold met -> not all -> continue.
        assert!(!t.should_terminate(&progress(5, 0.0, 0.0, true, 1)));
        // Both met -> stop.
        assert!(t.should_terminate(&progress(5, 0.0, 0.0, true, 3)));
        // Empty AllTerminator never terminates.
        let empty = AllTerminator::new(vec![]);
        assert!(!empty.should_terminate(&progress(1000, 1e9, 0.0, true, 1000)));
    }
}
