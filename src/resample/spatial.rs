//! Spatial cross-validation strategies for geospatial data.

use rand::seq::SliceRandom;
use rand::rngs::StdRng;
use rand::SeedableRng;
use super::Resample;

/// Spatial block cross-validation.
///
/// Divides the spatial extent into a grid of cells and assigns cells to folds,
/// ensuring spatial separation between train and test sets. Prevents spatial
/// autocorrelation leakage.
///
/// # Examples
///
/// ```
/// use smelt::resample::{Resample, SpatialBlockCV};
///
/// let coords = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0),
///                   (2.0, 2.0), (3.0, 2.0), (2.0, 3.0), (3.0, 3.0)];
/// let cv = SpatialBlockCV::new(2, coords);
/// let splits = cv.splits(8);
/// assert_eq!(splits.len(), 2);
/// ```
pub struct SpatialBlockCV {
    n_folds: usize,
    coords: Vec<(f64, f64)>,
}

impl SpatialBlockCV {
    pub fn new(n_folds: usize, coords: Vec<(f64, f64)>) -> Self {
        Self { n_folds, coords }
    }
}

impl Resample for SpatialBlockCV {
    fn splits(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        assert_eq!(
            self.coords.len(), n_samples,
            "coords length ({}) must match n_samples ({})",
            self.coords.len(), n_samples
        );

        // Compute bounding box
        let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
        let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        for &(x, y) in &self.coords {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }

        // Grid dimensions: ceil(sqrt(n_folds)) × ceil(sqrt(n_folds))
        let grid_size = (self.n_folds as f64).sqrt().ceil() as usize;
        let cell_w = (max_x - min_x + f64::EPSILON) / grid_size as f64;
        let cell_h = (max_y - min_y + f64::EPSILON) / grid_size as f64;

        // Assign each sample to a cell, then cell to fold
        let cell_assignments: Vec<usize> = self.coords.iter()
            .map(|&(x, y)| {
                let col = ((x - min_x) / cell_w).floor() as usize;
                let row = ((y - min_y) / cell_h).floor() as usize;
                let cell_id = row * grid_size + col;
                cell_id % self.n_folds
            })
            .collect();

        (0..self.n_folds)
            .map(|fold| {
                let test: Vec<usize> = (0..n_samples)
                    .filter(|&i| cell_assignments[i] == fold)
                    .collect();
                let train: Vec<usize> = (0..n_samples)
                    .filter(|&i| cell_assignments[i] != fold)
                    .collect();
                (train, test)
            })
            .collect()
    }
}

/// Spatial buffer cross-validation.
///
/// Performs k-fold splitting, then removes training samples within
/// `buffer_distance` of any test sample. This creates a spatial gap
/// between train and test sets, reducing autocorrelation leakage.
///
/// # Examples
///
/// ```
/// use smelt::resample::{Resample, SpatialBufferCV};
///
/// let coords = vec![(0.0, 0.0), (0.1, 0.0), (10.0, 10.0), (10.1, 10.0)];
/// let cv = SpatialBufferCV::new(2, coords, 1.0).with_seed(42);
/// let splits = cv.splits(4);
/// ```
pub struct SpatialBufferCV {
    n_folds: usize,
    coords: Vec<(f64, f64)>,
    buffer_distance: f64,
    seed: u64,
}

impl SpatialBufferCV {
    pub fn new(n_folds: usize, coords: Vec<(f64, f64)>, buffer_distance: f64) -> Self {
        Self { n_folds, coords, buffer_distance, seed: 42 }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

fn euclidean_dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

impl Resample for SpatialBufferCV {
    fn splits(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        assert_eq!(
            self.coords.len(), n_samples,
            "coords length ({}) must match n_samples ({})",
            self.coords.len(), n_samples
        );

        // Standard k-fold shuffle
        let mut indices: Vec<usize> = (0..n_samples).collect();
        let mut rng = StdRng::seed_from_u64(self.seed);
        indices.shuffle(&mut rng);

        let fold_size = n_samples / self.n_folds;

        (0..self.n_folds)
            .map(|fold| {
                let test_start = fold * fold_size;
                let test_end = if fold == self.n_folds - 1 { n_samples } else { test_start + fold_size };
                let test: Vec<usize> = indices[test_start..test_end].to_vec();

                // Remove train samples within buffer_distance of any test sample
                let train: Vec<usize> = indices[..test_start].iter()
                    .chain(indices[test_end..].iter())
                    .copied()
                    .filter(|&train_idx| {
                        !test.iter().any(|&test_idx| {
                            euclidean_dist(self.coords[train_idx], self.coords[test_idx])
                                < self.buffer_distance
                        })
                    })
                    .collect();

                (train, test)
            })
            .collect()
    }
}
