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
    ///
    /// Points in a singleton cluster (no other member to average `a(i)`
    /// over) get silhouette `0`, matching the sklearn/Rousseeuw convention,
    /// rather than being scored as if `a(i) = 0` (which previously inflated
    /// their contribution to the maximum possible value of `1.0` -- exactly
    /// the regime this score is meant to penalize when sweeping `k` for
    /// model selection, since larger `k` produces more singletons).
    pub fn silhouette_score(&self, features: &Array2<f64>) -> f64 {
        let n = features.nrows();
        if self.n_clusters < 2 {
            return 0.0;
        }

        // Distinct non-noise labels actually present: hand-built results may
        // carry non-contiguous labels (e.g. {0, 5}), which the previous
        // `0..n_clusters` loop silently excluded from every b(i).
        let mut present_labels: Vec<i32> =
            self.labels.iter().copied().filter(|&l| l >= 0).collect();
        present_labels.sort_unstable();
        present_labels.dedup();

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

            if a_count == 0 {
                // Singleton cluster: silhouette is defined as 0, not scored
                // via a(i)=0 against whatever b(i) happens to be.
                count += 1;
                continue;
            }
            let a = a_sum / a_count as f64;

            // b(i) = min mean distance to samples in nearest other cluster
            let mut b = f64::INFINITY;
            for &c in &present_labels {
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
            }
            count += 1;
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
    n_init: usize,
    seed: u64,
}

impl KMeans {
    /// Creates a K-Means clusterer for `k` clusters, with defaults
    /// `max_iter=300`, `n_init=10`, `seed=42`.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            max_iter: 300,
            n_init: 10,
            seed: 42,
        }
    }
    /// Sets the maximum number of Lloyd's algorithm iterations before stopping.
    pub fn with_max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }
    /// Sets the number of independent random (k-means++) initializations to
    /// run; the run with the lowest final inertia is kept. Matches
    /// sklearn's default of 10 -- a single run of plain Lloyd's algorithm
    /// (the previous default and only option) converges to a local optimum
    /// that depends heavily on the initial centroids, and can merge or
    /// split visually obvious clusters depending on the seed.
    pub fn with_n_init(mut self, n: usize) -> Self {
        self.n_init = n.max(1);
        self
    }
    /// Sets the RNG seed used to initialize centroids.
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Runs Lloyd's algorithm `n_init` times (k-means++ initialization,
    /// independently seeded) and returns the run with the lowest inertia.
    pub fn fit(&self, features: &Array2<f64>) -> Result<ClusterResult> {
        let n = features.nrows();
        if n < self.k {
            return Err(SmeltError::InvalidParameter(format!(
                "k={} > n_samples={}",
                self.k, n
            )));
        }

        let mut best: Option<(f64, Vec<i32>, Array2<f64>)> = None;
        for run in 0..self.n_init {
            let run_seed = self.seed.wrapping_add(run as u64);
            let (labels, centroids, inertia) = self.run_once(features, run_seed);
            let is_better = match &best {
                Some((best_inertia, _, _)) => inertia < *best_inertia,
                None => true,
            };
            if is_better {
                best = Some((inertia, labels, centroids));
            }
        }

        let (_, labels, centroids) =
            best.expect("n_init is always clamped to >= 1 in with_n_init/new");

        // Defense in depth: the empty-cluster reseeding in `run_once` is
        // expected to always leave exactly `k` non-empty clusters, but
        // report the count actually observed in `labels` rather than
        // assuming it, so a caller never sees `n_clusters` claim more
        // clusters than actually have members.
        let mut present = vec![false; self.k];
        for &l in &labels {
            present[l as usize] = true;
        }
        let n_clusters = present.iter().filter(|&&p| p).count();

        Ok(ClusterResult {
            labels,
            n_clusters,
            centroids: Some(centroids),
        })
    }

    /// One full run of k-means++ initialization + Lloyd's algorithm (with
    /// empty-cluster reseeding). Returns `(labels, centroids, inertia)`.
    fn run_once(&self, features: &Array2<f64>, seed: u64) -> (Vec<i32>, Array2<f64>, f64) {
        let n = features.nrows();
        let p = features.ncols();
        let mut rng = StdRng::seed_from_u64(seed);

        let centroid_idx = kmeans_plus_plus_init(features, self.k, &mut rng);
        let mut centroids = Array2::zeros((self.k, p));
        for (c, &idx) in centroid_idx.iter().enumerate() {
            centroids.row_mut(c).assign(&features.row(idx));
        }

        let mut labels = vec![0i32; n];

        for _ in 0..self.max_iter {
            // Assign each point to nearest centroid, tracking the distance
            // to its assigned centroid (needed below to reseed empty
            // clusters with the currently-worst-served point).
            let mut changed = false;
            let mut dists = vec![0.0f64; n];
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
                dists[i] = best_d;
                if labels[i] != best_c as i32 {
                    changed = true;
                    labels[i] = best_c as i32;
                }
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

            // Reseed any empty cluster by stealing the point currently
            // farthest from its own centroid (the point Lloyd's algorithm
            // is serving worst) and making it that cluster's sole member.
            // Without this, an empty cluster's centroid never updates
            // (there's nothing to average), so it stays wherever
            // initialization or a bad update left it -- permanently dead
            // for the rest of the run, silently reducing the effective k.
            // Looping (bounded by k) handles the rare cascade where
            // stealing a cluster's only point empties *that* cluster too.
            for _ in 0..self.k {
                let empty_cluster = (0..self.k).find(|&c| counts[c] == 0);
                let Some(c) = empty_cluster else { break };
                let farthest = (0..n)
                    .max_by(|&a, &b| dists[a].partial_cmp(&dists[b]).unwrap())
                    .expect("n >= k >= 1, so there is at least one point");
                let old_c = labels[farthest] as usize;
                labels[farthest] = c as i32;
                counts[old_c] -= 1;
                counts[c] = 1;
                for j in 0..p {
                    sums[[c, j]] = features[[farthest, j]];
                    sums[[old_c, j]] -= features[[farthest, j]];
                }
                dists[farthest] = 0.0;
                changed = true;
            }

            for c in 0..self.k {
                if counts[c] > 0 {
                    for j in 0..p {
                        centroids[[c, j]] = sums[[c, j]] / counts[c] as f64;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        let inertia: f64 = (0..n)
            .map(|i| {
                let c = labels[i] as usize;
                euclidean(features.row(i), centroids.row(c)).powi(2)
            })
            .sum();

        (labels, centroids, inertia)
    }
}

/// k-means++ initialization (Arthur & Vassilvitskii, 2007): picks the first
/// center uniformly at random, then each subsequent center with probability
/// proportional to its squared distance to the nearest already-chosen
/// center. Spreads initial centroids across the data instead of the
/// previous plain-random-sample init, which could (and empirically did,
/// ~38% of seeds on trivially-separated synthetic blobs) place two initial
/// centroids in the same true cluster, merging or splitting obvious
/// clusters for the rest of the run.
fn kmeans_plus_plus_init(features: &Array2<f64>, k: usize, rng: &mut StdRng) -> Vec<usize> {
    let n = features.nrows();
    let mut chosen: Vec<usize> = Vec::with_capacity(k);
    let first = rng.random_range(0..n);
    chosen.push(first);

    let mut dist_sq: Vec<f64> = (0..n)
        .map(|i| euclidean(features.row(i), features.row(first)).powi(2))
        .collect();

    while chosen.len() < k {
        let total: f64 = dist_sq.iter().sum();
        let next = if total <= 0.0 {
            // All remaining points coincide with an already-chosen center;
            // any unchosen index is as good as any other.
            (0..n).find(|i| !chosen.contains(i)).unwrap_or(0)
        } else {
            let target = rng.random::<f64>() * total;
            let mut acc = 0.0;
            let mut selected = n - 1;
            for (i, &d) in dist_sq.iter().enumerate() {
                acc += d;
                if acc >= target {
                    selected = i;
                    break;
                }
            }
            selected
        };
        chosen.push(next);
        for i in 0..n {
            let d = euclidean(features.row(i), features.row(next)).powi(2);
            if d < dist_sq[i] {
                dist_sq[i] = d;
            }
        }
    }

    chosen
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

#[cfg(test)]
mod tests {
    use super::*;

    /// 4th-audit LOW: b(i) used to iterate `0..n_clusters`, so a hand-built
    /// result with non-contiguous labels (e.g. {0, 5}) had its high labels
    /// silently excluded from every neighbour-cluster distance. The score
    /// must be invariant to relabeling.
    #[test]
    fn silhouette_is_invariant_to_non_contiguous_labels() {
        let data = ndarray::array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 9.9],
            [9.9, 10.1]
        ];
        let contiguous = ClusterResult {
            labels: vec![0, 0, 0, 1, 1, 1],
            n_clusters: 2,
            centroids: None,
        };
        let relabeled = ClusterResult {
            labels: vec![0, 0, 0, 5, 5, 5],
            n_clusters: 2,
            centroids: None,
        };
        let s_contiguous = contiguous.silhouette_score(&data);
        let s_relabeled = relabeled.silhouette_score(&data);
        assert!(
            (s_contiguous - s_relabeled).abs() < 1e-12,
            "relabeling {{0,1}} -> {{0,5}} changed the score: {s_contiguous} vs {s_relabeled}"
        );
        assert!(
            s_relabeled > 0.9,
            "two tight, far-apart blobs must score near 1, got {s_relabeled}"
        );
    }

    /// 3 well-separated 2D blobs (>8 sigma apart), 10 points each, with a
    /// small fixed jitter pattern -- deterministic, no RNG needed for a
    /// golden fixture.
    fn three_blobs() -> Array2<f64> {
        let centers = [[0.0, 0.0], [20.0, 0.0], [0.0, 20.0]];
        let jitter = [
            [0.0, 0.0],
            [0.3, -0.2],
            [-0.2, 0.3],
            [0.1, 0.1],
            [-0.1, -0.1],
            [0.2, 0.2],
            [-0.3, 0.1],
            [0.1, -0.3],
            [0.0, 0.2],
            [-0.2, 0.0],
        ];
        let mut flat = Vec::with_capacity(centers.len() * jitter.len() * 2);
        for c in &centers {
            for j in &jitter {
                flat.push(c[0] + j[0]);
                flat.push(c[1] + j[1]);
            }
        }
        Array2::from_shape_vec((centers.len() * jitter.len(), 2), flat).unwrap()
    }

    /// True iff every contiguous `points_per_blob`-sized segment of `labels`
    /// (one per blob, in fixture order) got a single label, and the blobs'
    /// labels are pairwise distinct -- i.e. no blob was split across
    /// clusters and no two blobs were merged into one.
    fn recovers_blobs(labels: &[i32], points_per_blob: usize) -> bool {
        let n_blobs = labels.len() / points_per_blob;
        let mut blob_labels = Vec::with_capacity(n_blobs);
        for b in 0..n_blobs {
            let seg = &labels[b * points_per_blob..(b + 1) * points_per_blob];
            if !seg.windows(2).all(|w| w[0] == w[1]) {
                return false;
            }
            blob_labels.push(seg[0]);
        }
        blob_labels.sort_unstable();
        blob_labels.dedup();
        blob_labels.len() == n_blobs
    }

    /// Regression test for the KMeans HIGH finding: plain single-run
    /// Lloyd's algorithm with random-sample init (the previous
    /// implementation) merged/split these trivially-separated blobs on
    /// ~38% of seeds (19/50 in the audit's sonda). k-means++ init +
    /// best-of-`n_init` should recover all 3 blobs on every seed tried.
    #[test]
    fn kmeans_recovers_well_separated_blobs_across_seeds() {
        let data = three_blobs();
        for seed in 0..30u64 {
            let result = KMeans::new(3).with_seed(seed).fit(&data).unwrap();
            assert_eq!(
                result.n_clusters, 3,
                "seed {seed}: expected 3 non-empty clusters, got {}",
                result.n_clusters
            );
            assert!(
                recovers_blobs(&result.labels, 10),
                "seed {seed}: blobs were merged or split, labels={:?}",
                result.labels
            );
        }
    }

    /// A single k-means++ run (`n_init=1`) can still occasionally land on a
    /// bad local optimum on adversarial data; `n_init`'s default of 10
    /// (best-of-10 by inertia) is what the HIGH fix actually relies on.
    /// This just checks the builder plumbs through and a low n_init still
    /// produces a valid (if not necessarily perfect) clustering.
    #[test]
    fn kmeans_with_n_init_one_still_produces_valid_clustering() {
        let data = three_blobs();
        let result = KMeans::new(3).with_seed(7).with_n_init(1).fit(&data).unwrap();
        assert!(result.n_clusters >= 1 && result.n_clusters <= 3);
        assert_eq!(result.labels.len(), data.nrows());
    }

    /// Regression test: an empty cluster used to keep its stale centroid
    /// forever (nothing to average) and `n_clusters` still reported `k`
    /// even when fewer clusters actually had members. With k deliberately
    /// larger than the number of natural groups, reseeding should still
    /// leave every cluster non-empty.
    #[test]
    fn kmeans_reseeds_empty_clusters_instead_of_leaving_them_dead() {
        let data = three_blobs(); // 3 natural groups, 30 points
        let result = KMeans::new(5).with_seed(3).fit(&data).unwrap();
        assert_eq!(
            result.n_clusters, 5,
            "all 5 clusters should end up with members via reseeding"
        );
        let mut counts = vec![0usize; 5];
        for &l in &result.labels {
            counts[l as usize] += 1;
        }
        assert!(counts.iter().all(|&c| c > 0), "counts={counts:?}");
    }

    /// Regression test for the silhouette singleton bug: a singleton
    /// cluster's `a(i)` fallback to `0.0` made `(b-0)/max(0,b) = 1.0`, the
    /// maximum possible score, regardless of how well- or poorly-separated
    /// that lone point actually was. Golden value from
    /// `sklearn.metrics.silhouette_score` on the same fixture and labels.
    #[test]
    fn silhouette_singleton_cluster_matches_sklearn() {
        let features = Array2::from_shape_vec((5, 1), vec![0.0, 0.1, 0.2, 10.0, 20.0]).unwrap();
        let result = ClusterResult {
            labels: vec![2, 2, 2, 1, 0], // cluster 0 = singleton {20.0}
            n_clusters: 3,
            centroids: None,
        };
        let score = result.silhouette_score(&features);
        let expected = 0.5919185734900021;
        assert!(
            (score - expected).abs() < 1e-9,
            "got {score}, expected {expected} (sklearn)"
        );
    }

    /// Non-singleton case should be unaffected by the fix: matches sklearn
    /// to high precision on a fixture with no singleton clusters.
    #[test]
    fn silhouette_matches_sklearn_without_singletons() {
        let data = three_blobs();
        let mut labels = vec![0i32; 30];
        for (b, l) in labels.chunks_mut(10).enumerate() {
            l.fill(b as i32);
        }
        let result = ClusterResult {
            labels,
            n_clusters: 3,
            centroids: None,
        };
        let score = result.silhouette_score(&data);
        assert!(score > 0.9, "well-separated blobs should score high: {score}");
    }
}
