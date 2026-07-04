//! Preprocessing: feature scaling, encoding, imputation, and pipeline composition.

pub mod adasyn;
pub mod encoder;
pub mod filter;
pub mod imputer;
pub mod mutual_info;
pub mod label_encoder;
pub mod pca;
pub mod pipeline;
pub mod rfe;
pub mod scaler;
pub mod smote;
pub mod spatial_smote;

use crate::Result;
use ndarray::Array2;

pub use adasyn::Adasyn;
pub use encoder::OneHotEncoder;
pub use filter::FilterSelector;
pub use imputer::{ImputeStrategy, Imputer};
pub use label_encoder::LabelEncoder;
pub use pca::PCA;
pub use pipeline::Pipeline;
pub use rfe::RFE;
pub use scaler::{MinMaxScaler, StandardScaler};
pub use smote::Smote;
pub use spatial_smote::SpatialSmote;

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

    /// Fit with access to target values (for supervised filters like information gain).
    /// Default: ignores target and delegates to `fit`.
    fn fit_supervised(&mut self, features: &Array2<f64>, _target: &[f64]) -> Result<()> {
        self.fit(features)
    }

    /// Convenience: fit on data and immediately transform it.
    fn fit_transform(&mut self, features: &Array2<f64>) -> Result<Array2<f64>> {
        self.fit(features)?;
        self.transform(features)
    }

    /// Supervised fit + transform.
    fn fit_transform_supervised(
        &mut self,
        features: &Array2<f64>,
        target: &[f64],
    ) -> Result<Array2<f64>> {
        self.fit_supervised(features, target)?;
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
