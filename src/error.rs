use thiserror::Error;

/// Shorthand for `Result<T, SmeltError>`, used throughout the crate's public API.
pub type Result<T> = std::result::Result<T, SmeltError>;

/// Errors that can occur while building tasks, training learners, or predicting.
#[derive(Debug, Error)]
pub enum SmeltError {
    /// A feature matrix, target vector, or array pair had mismatched lengths.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// The length that was required.
        expected: usize,
        /// The length that was actually found.
        got: usize,
    },

    /// A task or dataset had zero rows.
    #[error("empty dataset")]
    EmptyDataset,

    /// A referenced target column name does not exist.
    #[error("unknown target column: {0}")]
    UnknownTarget(String),

    /// A model's `predict` was called before it was fitted.
    #[error("model not trained")]
    NotTrained,

    /// A learner or resampling strategy received an out-of-range or inconsistent hyperparameter.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),

    /// A referenced feature column name does not exist in the task.
    #[error("feature '{0}' not found")]
    FeatureNotFound(String),

    /// Wraps an underlying I/O failure (e.g. reading a file).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A CSV file could not be parsed or was malformed.
    #[error("CSV error: {0}")]
    Csv(String),

    /// A Parquet file could not be read or was malformed.
    #[cfg(feature = "parquet")]
    #[error("Parquet error: {0}")]
    Parquet(String),

    /// A JSON payload (e.g. a serialized model) could not be parsed or was malformed.
    #[error("JSON error: {0}")]
    Json(String),

    /// A `Prediction` was the wrong variant (or missing a required field,
    /// e.g. probabilities) for the operation that received it -- a
    /// regression measure given a `Classification` prediction, a measure
    /// that needs probabilities given `probabilities: None`, etc. Lets
    /// callers `match` on this specific failure mode instead of pattern-
    /// matching an opaque string.
    #[error("incompatible prediction: {0}")]
    IncompatiblePrediction(String),

    /// A numerical computation failed in a way callers may want to detect
    /// specifically (e.g. a singular matrix in a linear solve), rather than
    /// an invalid input parameter or a malformed file.
    #[error("numerical error: {0}")]
    NumericalError(String),

    /// Catch-all for errors that don't fit the other variants.
    #[error("{0}")]
    Other(String),
}
