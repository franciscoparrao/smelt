use thiserror::Error;

pub type Result<T> = std::result::Result<T, SmeltError>;

#[derive(Debug, Error)]
pub enum SmeltError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("empty dataset")]
    EmptyDataset,

    #[error("unknown target column: {0}")]
    UnknownTarget(String),

    #[error("model not trained")]
    NotTrained,

    #[error("invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("feature '{0}' not found")]
    FeatureNotFound(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("CSV error: {0}")]
    Csv(String),

    #[error("JSON error: {0}")]
    Json(String),

    #[error("{0}")]
    Other(String),
}
