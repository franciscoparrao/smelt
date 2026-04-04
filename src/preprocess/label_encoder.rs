//! Label encoding: maps string labels to integer indices.

use crate::{Result, SmeltError};
use std::collections::HashMap;

/// Maps string labels to integer indices and back.
///
/// This is a standalone utility, not a `Transformer`, because it operates
/// on classification targets rather than feature matrices.
///
/// # Examples
///
/// ```
/// use smelt_ml::preprocess::LabelEncoder;
///
/// let encoder = LabelEncoder::fit(&["cat", "dog", "cat", "bird"]);
/// let encoded = encoder.encode(&["bird", "cat", "dog"]).unwrap();
/// assert_eq!(encoded, vec![0, 1, 2]); // alphabetical order
///
/// let decoded = encoder.decode(&encoded);
/// assert_eq!(decoded, vec!["bird", "cat", "dog"]);
/// ```
pub struct LabelEncoder {
    classes: Vec<String>,
    class_to_index: HashMap<String, usize>,
}

impl LabelEncoder {
    /// Fit the encoder on a set of labels.
    pub fn fit(labels: &[impl AsRef<str>]) -> Self {
        let mut unique: Vec<String> = labels
            .iter()
            .map(|l| l.as_ref().to_string())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        unique.sort();
        let class_to_index = unique
            .iter()
            .enumerate()
            .map(|(i, c)| (c.clone(), i))
            .collect();
        Self {
            classes: unique,
            class_to_index,
        }
    }

    /// Encode labels to integer indices.
    pub fn encode(&self, labels: &[impl AsRef<str>]) -> Result<Vec<usize>> {
        labels
            .iter()
            .map(|l| {
                self.class_to_index
                    .get(l.as_ref())
                    .copied()
                    .ok_or_else(|| SmeltError::Other(format!("unknown label: {}", l.as_ref())))
            })
            .collect()
    }

    /// Decode integer indices back to labels.
    pub fn decode(&self, indices: &[usize]) -> Vec<String> {
        indices.iter().map(|&i| self.classes[i].clone()).collect()
    }

    /// Get the list of classes in order.
    pub fn classes(&self) -> &[String] {
        &self.classes
    }

    /// Number of classes.
    pub fn n_classes(&self) -> usize {
        self.classes.len()
    }
}
