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
use crate::task::{ClassificationTask, FeatureType};
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

    /// Transform per-column [`FeatureType`]s, mirroring [`Self::transform_names`]
    /// (5th audit, M-3: `Pipeline` rebuilt the transformed task without its
    /// feature types, silently degrading the boosting engines' categorical
    /// splits to numeric even with zero transformers). Default: pass through
    /// unchanged (correct for column-preserving transformers). Column
    /// *selectors* (FilterSelector/RFE) keep the selected columns' types;
    /// transformers whose outputs mix or re-derive columns (PCA, one-hot
    /// expansion) return `Numeric` for those output columns, since the
    /// integer-code invariant no longer holds there.
    fn transform_types(&self, types: &[FeatureType]) -> Result<Vec<FeatureType>> {
        Ok(types.to_vec())
    }

    /// Clone this transformer into a boxed trait object.
    fn clone_box(&self) -> Box<dyn Transformer>;
}

/// Trait for training-only resamplers (`Smote`, `Adasyn`) that rebalance a
/// classification task by adding synthetic samples.
///
/// Unlike [`Transformer`], a resampler changes the sample count (and the
/// target, not just the features), so it can't fit that trait's
/// `transform(&Array2<f64>) -> Array2<f64>` signature (audit issue M18) --
/// and critically, it must run only once, on the training set, never at
/// predict time (synthesizing samples for held-out data would be
/// meaningless). [`Pipeline`] keeps a resampler structurally separate from
/// its `transformers` for exactly that reason: applied once at the start
/// of `train_classif`, and never stored in the trained model.
pub trait Resampler: Send + Sync {
    /// Resampler identifier.
    fn id(&self) -> &str;

    /// Rebalance a classification task, returning a new task with a
    /// (typically larger) balanced sample set.
    fn resample(&self, task: &ClassificationTask) -> Result<ClassificationTask>;
}
