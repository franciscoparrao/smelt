//! Preprocessing: feature scaling, encoding, imputation, and pipeline composition.

pub mod scaler;
pub mod imputer;
pub mod encoder;
pub mod label_encoder;
pub mod smote;
pub mod pipeline;

use ndarray::Array2;
use crate::Result;

pub use scaler::{StandardScaler, MinMaxScaler};
pub use imputer::{Imputer, ImputeStrategy};
pub use encoder::OneHotEncoder;
pub use label_encoder::LabelEncoder;
pub use smote::Smote;
pub use pipeline::Pipeline;

/// Trait for feature transformers (scalers, encoders, imputers).
///
/// Follows a fit/transform pattern: `fit` learns parameters from training data,
/// `transform` applies the learned transformation to any data.
pub trait Transformer: Send + Sync {
    /// Transformer identifier.
    fn id(&self) -> &str;

    /// Learn transformation parameters from training data.
    fn fit(&mut self, features: &Array2<f64>) -> Result<()>;

    /// Apply the learned transformation. Fails if not fitted.
    fn transform(&self, features: &Array2<f64>) -> Result<Array2<f64>>;

    /// Convenience: fit on data and immediately transform it.
    fn fit_transform(&mut self, features: &Array2<f64>) -> Result<Array2<f64>> {
        self.fit(features)?;
        self.transform(features)
    }

    /// Transform feature names (for transformers that change column count).
    /// Default: pass through unchanged.
    fn transform_names(&self, names: &[String]) -> Result<Vec<String>> {
        Ok(names.to_vec())
    }

    /// Clone this transformer into a boxed trait object.
    fn clone_box(&self) -> Box<dyn Transformer>;
}
