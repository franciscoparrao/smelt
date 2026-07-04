//! Spatial-SMOTE: SMOTE that respects spatial proximity, not just
//! feature-space proximity.
//!
//! Plain SMOTE (`crate::preprocess::Smote`) can splice together two
//! minority-class samples that are feature-similar but geographically
//! distant (e.g. across two different regions), producing synthetic points
//! that don't correspond to any real spatial location. `SpatialSmote`
//! restricts the candidate-neighbor pool to same-class points within an
//! optional `max_spatial_distance`, and additionally interpolates a
//! synthetic coordinate for each synthetic sample (using the same lambda as
//! the feature interpolation), so downstream spatially-aware code
//! (`SpatialBlockCV`, `GeoXGBoost`, ...) can consume the balanced dataset.

use crate::Result;
use crate::SmeltError;
use crate::task::{ClassificationTask, Task};
use ndarray::Array2;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Euclidean distance between two coordinates.
#[inline]
fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

/// Spatially-informed SMOTE oversampler.
///
/// Generates synthetic minority-class samples by interpolating between a
/// minority instance and one of its k-nearest same-class neighbors,
/// restricted to neighbors within `max_spatial_distance` (when set), and
/// interpolates a synthetic coordinate alongside each synthetic sample.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use ndarray::array;
///
/// let features = array![
///     [0.0, 0.0], [0.1, 0.1], [0.2, 0.0],  // class 0 (majority: 3)
///     [1.0, 1.0],                             // class 1 (minority: 1)
/// ];
/// let target = vec![0, 0, 0, 1];
/// let coords = vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (5.0, 5.0)];
/// let task = ClassificationTask::new("imb", features, target).unwrap();
///
/// let smote = SpatialSmote::new().with_seed(42);
/// let (balanced, new_coords) = smote.balance(&task, &coords).unwrap();
/// assert_eq!(new_coords.len(), balanced.n_samples());
/// ```
pub struct SpatialSmote {
    k_neighbors: usize,
    seed: u64,
    max_spatial_distance: Option<f64>,
}

impl Default for SpatialSmote {
    fn default() -> Self {
        Self {
            k_neighbors: 5,
            seed: 42,
            max_spatial_distance: None,
        }
    }
}

impl SpatialSmote {
    /// Create a Spatial-SMOTE oversampler with default parameters (no
    /// spatial distance cutoff -- equivalent to plain `Smote` plus
    /// coordinate interpolation).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of nearest same-class neighbors used to synthesize
    /// new samples.
    pub fn with_k_neighbors(mut self, k: usize) -> Self {
        self.k_neighbors = k;
        self
    }

    /// Set the RNG seed for reproducible synthetic sample generation.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Restrict candidate neighbors to those within this spatial distance.
    /// A minority sample with no same-class neighbor within the cutoff
    /// contributes no synthetic samples for that draw (graceful
    /// degradation, not an error).
    pub fn with_max_spatial_distance(mut self, d: f64) -> Self {
        self.max_spatial_distance = Some(d);
        self
    }

    /// Balance the task by oversampling minority classes, using `coords`
    /// (one `(x, y)` per sample, same order as `task`'s rows) to restrict
    /// and locate synthetic samples spatially.
    ///
    /// Returns the balanced task alongside an updated coordinate vector
    /// (original coordinates followed by each synthetic sample's
    /// interpolated coordinate), since `Task` itself carries no notion of
    /// spatial location.
    pub fn balance(
        &self,
        task: &ClassificationTask,
        coords: &[(f64, f64)],
    ) -> Result<(ClassificationTask, Vec<(f64, f64)>)> {
        let features = task.features();
        let target = task.target();
        let n_classes = task.n_classes();
        let n_features = task.n_features();
        let n_samples = task.n_samples();

        if coords.len() != n_samples {
            return Err(SmeltError::DimensionMismatch {
                expected: n_samples,
                got: coords.len(),
            });
        }

        let mut class_counts = vec![0usize; n_classes];
        for &t in target {
            class_counts[t] += 1;
        }
        let max_count = *class_counts.iter().max().unwrap();

        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut new_features: Vec<Vec<f64>> =
            features.rows().into_iter().map(|r| r.to_vec()).collect();
        let mut new_target: Vec<usize> = target.to_vec();
        let mut new_coords: Vec<(f64, f64)> = coords.to_vec();

        for class in 0..n_classes {
            let n_to_generate = max_count - class_counts[class];
            if n_to_generate == 0 {
                continue;
            }

            let class_indices: Vec<usize> = target
                .iter()
                .enumerate()
                .filter(|&(_, &t)| t == class)
                .map(|(i, _)| i)
                .collect();
            let n_class = class_indices.len();
            if n_class == 0 {
                continue;
            }

            let k = self.k_neighbors.min(n_class.saturating_sub(1)).max(1);

            // A same-class point beyond `max_spatial_distance` from every
            // other same-class point contributes nothing on the draws that
            // pick it; bound the retry budget rather than looping forever.
            let max_attempts = 10 * n_to_generate.max(1);
            let mut attempts = 0usize;
            let mut generated = 0usize;

            while generated < n_to_generate && attempts < max_attempts {
                attempts += 1;

                let idx = class_indices[rng.random_range(0..n_class)];
                let sample = features.row(idx);
                let sample_coord = coords[idx];

                let mut dists: Vec<(usize, f64)> = class_indices
                    .iter()
                    .filter(|&&i| i != idx)
                    .filter(|&&i| {
                        self.max_spatial_distance
                            .is_none_or(|d| dist(sample_coord, coords[i]) <= d)
                    })
                    .map(|&i| {
                        let d: f64 = features
                            .row(i)
                            .iter()
                            .zip(sample.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>()
                            .sqrt();
                        (i, d)
                    })
                    .collect();

                if dists.is_empty() {
                    if n_class == 1 {
                        // Only 1 sample in the class overall: duplicate with
                        // small noise (matches plain `Smote`'s fallback;
                        // there is no spatial constraint to apply here).
                        let synthetic: Vec<f64> = sample
                            .iter()
                            .map(|&s| s + rng.random_range(-0.01..0.01))
                            .collect();
                        new_features.push(synthetic);
                        new_target.push(class);
                        new_coords.push(sample_coord);
                        generated += 1;
                    }
                    // Else: no same-class neighbor within max_spatial_distance
                    // for this draw -- skip, bounded by max_attempts above.
                    continue;
                }

                dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                let nn_idx = dists[rng.random_range(0..k.min(dists.len()))].0;
                let neighbor = features.row(nn_idx);
                let lambda: f64 = rng.random_range(0.0..1.0);

                let synthetic: Vec<f64> = sample
                    .iter()
                    .zip(neighbor.iter())
                    .map(|(&s, &n)| s + lambda * (n - s))
                    .collect();
                let neighbor_coord = coords[nn_idx];
                let synthetic_coord = (
                    sample_coord.0 + lambda * (neighbor_coord.0 - sample_coord.0),
                    sample_coord.1 + lambda * (neighbor_coord.1 - sample_coord.1),
                );

                new_features.push(synthetic);
                new_target.push(class);
                new_coords.push(synthetic_coord);
                generated += 1;
            }
        }

        let n_total = new_features.len();
        let mut result = Array2::zeros((n_total, n_features));
        for (i, row) in new_features.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                result[[i, j]] = val;
            }
        }

        let balanced_task = ClassificationTask::new(task.id(), result, new_target)?;
        Ok((balanced_task, new_coords))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocess::Smote;

    #[test]
    fn rejects_coord_mismatch() {
        let features = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let task = ClassificationTask::new("t", features, vec![0, 0, 1]).unwrap();
        let coords = vec![(0.0, 0.0), (1.0, 0.0)]; // wrong length
        let smote = SpatialSmote::new();
        assert!(smote.balance(&task, &coords).is_err());
    }

    #[test]
    fn achieves_class_parity_without_spatial_constraint() {
        let features = Array2::from_shape_vec(
            (6, 1),
            vec![0.0, 0.1, 0.2, 0.3, 0.4, 1.0],
        )
        .unwrap();
        let target = vec![0, 0, 0, 0, 0, 1];
        let coords: Vec<(f64, f64)> = (0..6).map(|i| (i as f64, 0.0)).collect();
        let task = ClassificationTask::new("t", features, target).unwrap();

        let smote = SpatialSmote::new().with_seed(42);
        let (balanced, new_coords) = smote.balance(&task, &coords).unwrap();

        let mut counts = vec![0usize; balanced.n_classes()];
        for &t in balanced.target() {
            counts[t] += 1;
        }
        assert_eq!(counts[0], counts[1], "classes should be balanced");
        assert_eq!(new_coords.len(), balanced.n_samples());
    }

    #[test]
    fn matches_plain_smote_when_unconstrained() {
        let features = Array2::from_shape_vec(
            (6, 2),
            vec![
                0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 0.3, 0.2, 0.4, 0.1, 1.0, 1.0,
            ],
        )
        .unwrap();
        let target = vec![0, 0, 0, 0, 0, 1];
        let coords: Vec<(f64, f64)> = (0..6).map(|i| (i as f64, 0.0)).collect();

        let task_a = ClassificationTask::new("t", features.clone(), target.clone()).unwrap();
        let task_b = ClassificationTask::new("t", features, target).unwrap();

        let spatial = SpatialSmote::new().with_seed(7).balance(&task_a, &coords).unwrap().0;
        let plain = Smote::new().with_seed(7).balance(&task_b).unwrap();

        assert_eq!(spatial.target(), plain.target());
        assert_eq!(spatial.features(), plain.features());
    }

    #[test]
    fn spatial_constraint_keeps_synthetics_within_one_cluster() {
        // Two same-class clusters far apart; a handful of majority-class
        // points elsewhere keep the minority class the one being oversampled.
        let mut feats = Vec::new();
        let mut target = Vec::new();
        let mut coords = Vec::new();

        // Minority class 1: cluster A near (0,0), cluster B near (100,100).
        for &(x, y) in &[(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)] {
            feats.push(x);
            feats.push(y);
            coords.push((x, y));
            target.push(1usize);
        }
        for &(x, y) in &[(100.0, 100.0), (100.0, 101.0), (101.0, 100.0), (101.0, 101.0)] {
            feats.push(x);
            feats.push(y);
            coords.push((x, y));
            target.push(1usize);
        }
        // Majority class 0: enough points to force oversampling of class 1.
        for i in 0..20 {
            let x = 50.0 + i as f64;
            let y = 50.0;
            feats.push(x);
            feats.push(y);
            coords.push((x, y));
            target.push(0usize);
        }

        let features = Array2::from_shape_vec((feats.len() / 2, 2), feats).unwrap();
        let task = ClassificationTask::new("clusters", features, target).unwrap();

        let smote = SpatialSmote::new().with_seed(3).with_max_spatial_distance(3.0);
        let (balanced, new_coords) = smote.balance(&task, &coords).unwrap();

        // Every synthetic coordinate (appended after the original 28 rows)
        // must sit within one cluster's neighborhood, never near the
        // midpoint of the ~140-unit gap between clusters.
        for &(x, _y) in &new_coords[28..] {
            assert!(
                x < 10.0 || x > 90.0,
                "synthetic point at x={x} straddles the inter-cluster gap"
            );
        }
        assert_eq!(new_coords.len(), balanced.n_samples());
    }

    #[test]
    fn isolated_minority_point_degrades_gracefully() {
        // Minority class has exactly 2 members, spatially far apart: no
        // draw ever finds a same-class neighbor within the cutoff, so no
        // synthetic samples can be generated -- must return Ok, not panic
        // or loop forever, and simply fall short of full balance.
        let features = Array2::from_shape_vec(
            (10, 1),
            vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1000.0, 1001.0],
        )
        .unwrap();
        let mut target = vec![0usize; 8];
        target.push(1);
        target.push(1);
        let coords: Vec<(f64, f64)> = (0..8)
            .map(|i| (i as f64, 0.0))
            .chain([(1000.0, 1000.0), (1001.0, 1001.0)])
            .collect();
        let task = ClassificationTask::new("isolated", features, target).unwrap();

        let smote = SpatialSmote::new().with_seed(5).with_max_spatial_distance(1.0);
        let (balanced, new_coords) = smote.balance(&task, &coords).unwrap();

        let mut counts = vec![0usize; balanced.n_classes()];
        for &t in balanced.target() {
            counts[t] += 1;
        }
        assert_eq!(counts[1], 2, "no synthetic minority samples should have been generated");
        assert_eq!(new_coords.len(), balanced.n_samples());
    }
}
