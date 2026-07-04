//! Unsupervised clustering and anomaly detection.

pub mod isolation_forest;

pub use isolation_forest::IsolationForest;

use crate::Result;
use crate::SmeltError;
use ndarray::{Array2, ArrayView1};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Clustering result.
#[derive(Debug, Clone)]
pub struct ClusterResult {
    /// Cluster assignment per sample (-1 = noise for DBSCAN).
    pub labels: Vec<i32>,
    /// Number of clusters found.
    pub n_clusters: usize,
    /// Cluster centroids (for K-Means). `n_clusters × n_features`.
    pub centroids: Option<Array2<f64>>,
}

impl ClusterResult {
    /// Silhouette score: measures how well samples are clustered. Range [-1, 1].
    pub fn silhouette_score(&self, features: &Array2<f64>) -> f64 {
        let n = features.nrows();
        if self.n_clusters < 2 {
            return 0.0;
        }

        let mut total = 0.0;
        let mut count = 0;

        for i in 0..n {
            if self.labels[i] < 0 {
                continue;
            } // skip noise
            let ci = self.labels[i];

            // a(i) = mean distance to samples in same cluster
            let mut a_sum = 0.0;
            let mut a_count = 0;
            for j in 0..n {
                if j != i && self.labels[j] == ci {
                    a_sum += euclidean(features.row(i), features.row(j));
                    a_count += 1;
                }
            }
            let a = if a_count > 0 {
                a_sum / a_count as f64
            } else {
                0.0
            };

            // b(i) = min mean distance to samples in nearest other cluster
            let mut b = f64::INFINITY;
            for c in 0..self.n_clusters as i32 {
                if c == ci {
                    continue;
                }
                let mut b_sum = 0.0;
                let mut b_count = 0;
                for j in 0..n {
                    if self.labels[j] == c {
                        b_sum += euclidean(features.row(i), features.row(j));
                        b_count += 1;
                    }
                }
                if b_count > 0 {
                    b = b.min(b_sum / b_count as f64);
                }
            }

            if a.max(b) > 0.0 {
                total += (b - a) / a.max(b);
                count += 1;
            }
        }

        if count > 0 { total / count as f64 } else { 0.0 }
    }
}

#[inline]
fn euclidean(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

// ── K-Means ─────────────────────────────────────────────────────────

/// K-Means clustering (Lloyd's algorithm).
///
/// # Examples
///
/// ```
/// use smelt_ml::cluster::KMeans;
/// use ndarray::array;
///
/// let data = array![[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 4.9]];
/// let result = KMeans::new(2).fit(&data).unwrap();
/// assert_eq!(result.n_clusters, 2);
/// ```
pub struct KMeans {
    k: usize,
    max_iter: usize,
    seed: u64,
}

impl KMeans {
    /// Creates a K-Means clusterer for `k` clusters, with defaults `max_iter=300`, `seed=42`.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            max_iter: 300,
            seed: 42,
        }
    }
    /// Sets the maximum number of Lloyd's algorithm iterations before stopping.
    pub fn with_max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }
    /// Sets the RNG seed used to initialize centroids.
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Runs Lloyd's algorithm and returns the resulting cluster assignment and centroids.
    pub fn fit(&self, features: &Array2<f64>) -> Result<ClusterResult> {
        let n = features.nrows();
        let p = features.ncols();
        if n < self.k {
            return Err(SmeltError::InvalidParameter(format!(
                "k={} > n_samples={}",
                self.k, n
            )));
        }

        let mut rng = StdRng::seed_from_u64(self.seed);

        // Initialize centroids: random samples
        let mut centroid_idx: Vec<usize> = Vec::with_capacity(self.k);
        while centroid_idx.len() < self.k {
            let idx = rng.random_range(0..n);
            if !centroid_idx.contains(&idx) {
                centroid_idx.push(idx);
            }
        }
        let mut centroids = Array2::zeros((self.k, p));
        for (c, &idx) in centroid_idx.iter().enumerate() {
            centroids.row_mut(c).assign(&features.row(idx));
        }

        let mut labels = vec![0i32; n];

        for _ in 0..self.max_iter {
            // Assign each point to nearest centroid
            let mut changed = false;
            for i in 0..n {
                let mut best_c = 0;
                let mut best_d = f64::INFINITY;
                for c in 0..self.k {
                    let d = euclidean(features.row(i), centroids.row(c));
                    if d < best_d {
                        best_d = d;
                        best_c = c;
                    }
                }
                if labels[i] != best_c as i32 {
                    changed = true;
                    labels[i] = best_c as i32;
                }
            }

            if !changed {
                break;
            }

            // Update centroids
            let mut sums: Array2<f64> = Array2::zeros((self.k, p));
            let mut counts = vec![0usize; self.k];
            for i in 0..n {
                let c = labels[i] as usize;
                for j in 0..p {
                    sums[[c, j]] += features[[i, j]];
                }
                counts[c] += 1;
            }
            for c in 0..self.k {
                if counts[c] > 0 {
                    for j in 0..p {
                        centroids[[c, j]] = sums[[c, j]] / counts[c] as f64;
                    }
                }
            }
        }

        Ok(ClusterResult {
            labels,
            n_clusters: self.k,
            centroids: Some(centroids),
        })
    }
}

// ── DBSCAN ──────────────────────────────────────────────────────────

/// DBSCAN density-based clustering.
///
/// Finds clusters of arbitrary shape. Points in low-density regions are
/// labeled as noise (-1).
///
/// # Examples
///
/// ```
/// use smelt_ml::cluster::DBSCAN;
/// use ndarray::array;
///
/// let data = array![[0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [5.0, 5.0], [5.1, 5.1]];
/// let result = DBSCAN::new(0.5, 2).fit(&data).unwrap();
/// assert!(result.n_clusters >= 2);
/// ```
pub struct DBSCAN {
    eps: f64,
    min_pts: usize,
}

impl DBSCAN {
    /// Creates a DBSCAN clusterer with neighborhood radius `eps` and minimum points `min_pts`
    /// required to form a dense region.
    pub fn new(eps: f64, min_pts: usize) -> Self {
        Self { eps, min_pts }
    }

    /// Runs density-based clustering and returns the resulting cluster assignment.
    pub fn fit(&self, features: &Array2<f64>) -> Result<ClusterResult> {
        let n = features.nrows();
        let mut labels = vec![-1i32; n]; // -1 = unvisited/noise
        let mut cluster_id = 0i32;

        for i in 0..n {
            if labels[i] != -1 {
                continue;
            } // already assigned

            let neighbors = self.range_query(features, i);
            if neighbors.len() < self.min_pts {
                // labels[i] remains -1 (noise, may be reassigned later)
                continue;
            }

            // Start new cluster
            labels[i] = cluster_id;
            let mut seed_set: Vec<usize> = neighbors.into_iter().filter(|&j| j != i).collect();
            let mut idx = 0;

            while idx < seed_set.len() {
                let q = seed_set[idx];
                if labels[q] == -1 {
                    labels[q] = cluster_id; // was noise, now border point
                }
                if labels[q] != -1 && labels[q] != cluster_id {
                    idx += 1;
                    continue; // already in another cluster
                }
                labels[q] = cluster_id;

                let q_neighbors = self.range_query(features, q);
                if q_neighbors.len() >= self.min_pts {
                    for &nn in &q_neighbors {
                        if labels[nn] == -1 && !seed_set.contains(&nn) {
                            seed_set.push(nn);
                        }
                    }
                }
                idx += 1;
            }

            cluster_id += 1;
        }

        let n_clusters = if cluster_id > 0 {
            cluster_id as usize
        } else {
            0
        };
        Ok(ClusterResult {
            labels,
            n_clusters,
            centroids: None,
        })
    }

    fn range_query(&self, features: &Array2<f64>, point: usize) -> Vec<usize> {
        let n = features.nrows();
        (0..n)
            .filter(|&j| euclidean(features.row(point), features.row(j)) <= self.eps)
            .collect()
    }
}
