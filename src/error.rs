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

    #[error("{0}")]
    Other(String),
}
