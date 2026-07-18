//! Mondrian Forest: an ensemble of online random trees grown via a Mondrian
//! process, giving partition-consistent structure across batch and
//! incremental (streaming) training.
//!
//! Lakshminarayanan, B., Roy, D. M., & Teh, Y. W. (2014). "Mondrian Forests:
//! Efficient Online Random Forests." NeurIPS 2014.
//!
//! # Why this differs from `HoeffdingTree`/`AdaptiveRandomForest`
//!
//! Hoeffding trees choose splits by information gain and only ever add
//! structure (a leaf becomes a split once enough evidence accumulates) --
//! there's no principled way to say whether the *resulting tree* from an
//! online stream would statistically resemble one built by batch CART on
//! the same data. A Mondrian tree's splits instead come from a continuous-
//! time stochastic process (a Mondrian process) with a specific,
//! self-consistent property: a Mondrian tree grown incrementally, one point
//! at a time, has exactly the same distribution as one grown by
//! [`sample_mondrian_block`] on the same points in one batch, regardless of
//! arrival order. Concretely:
//!
//! - Each node's split *time* is drawn from an exponential distribution
//!   whose rate is the node's bounding box's total side length (its "linear
//!   dimension") -- bigger boxes split sooner in expectation.
//! - The split *dimension* is chosen with probability proportional to that
//!   dimension's own side length, and the split *location* uniformly within
//!   the data's range on that dimension.
//! - Online, a new point that falls outside a node's existing bounding box
//!   can retroactively introduce a split *above* that node (see
//!   [`extend_node`]), exactly reproducing what batch construction on the
//!   enlarged point set would have sampled.
//!
//! # Scope
//!
//! This implements the core partition process (batch construction and
//! online extension) faithfully, plus simple per-leaf running statistics
//! (class counts / Welford mean-variance) for point predictions -- not the
//! paper's optional hierarchical Gaussian smoothing across the node
//! hierarchy (a Bayesian refinement layered on top of the partition
//! structure, not the partition-consistency property itself). One further
//! simplification: a leaf that receives a point *within* its current
//! bounding box only updates its running statistics; it does not re-attempt
//! `sample_mondrian_block` on its accumulated points to spontaneously split
//! from remaining time budget as density increases (which would require
//! storing raw per-leaf data rather than O(1) running statistics, the same
//! space/fidelity trade-off `Adwin`'s scan-based window makes over the
//! paper's exponential-histogram buckets). The genuinely distinguishing
//! online behavior -- new points outside current coverage introduce
//! consistent new splits -- is implemented in full.

use crate::learner::{Learner, TrainedModel};
use crate::prediction::Prediction;
use crate::task::{ClassificationTask, RegressionTask, Task};
use crate::Result;
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// Samples from `Exponential(rate)` via inverse-CDF (`-ln(U) / rate`). No
/// `rand_distr` dependency for a single distribution, consistent with this
/// crate's preference for hand-rolling small numeric routines (see
/// `src/sparse.rs`, `adaptive_rf.rs::sample_poisson`).
fn sample_exponential(rng: &mut impl Rng, rate: f64) -> f64 {
    if rate <= 0.0 {
        return f64::INFINITY;
    }
    let u: f64 = rng.random::<f64>().max(1e-300); // guard ln(0)
    -u.ln() / rate
}

/// Picks an index with probability proportional to `weights` (all `>= 0`,
/// sum `> 0`).
fn weighted_choice(rng: &mut impl Rng, weights: &[f64]) -> usize {
    let total: f64 = weights.iter().sum();
    let mut r = rng.random::<f64>() * total;
    for (i, &w) in weights.iter().enumerate() {
        r -= w;
        if r <= 0.0 {
            return i;
        }
    }
    weights.len() - 1 // floating point rounding fallback
}

/// Per-node running statistics: class counts for classification, or
/// Welford's online mean/variance for regression. `Vec`-backed and O(1)
/// update/space per point, like `HoeffdingTree`'s `FeatureStats`.
#[derive(Clone, Debug)]
#[derive(Serialize, Deserialize)]
enum NodeStats {
    Classification { counts: Vec<usize> },
    Regression { n: usize, mean: f64, m2: f64 },
}

impl NodeStats {
    fn new_classif(n_classes: usize) -> Self {
        NodeStats::Classification {
            counts: vec![0; n_classes.max(1)],
        }
    }

    fn new_regress() -> Self {
        NodeStats::Regression {
            n: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    fn update_classif(&mut self, label: usize) {
        if let NodeStats::Classification { counts } = self {
            if label >= counts.len() {
                counts.resize(label + 1, 0);
            }
            counts[label] += 1;
        }
    }

    fn update_regress(&mut self, y: f64) {
        if let NodeStats::Regression { n, mean, m2 } = self {
            *n += 1;
            let delta = y - *mean;
            *mean += delta / *n as f64;
            let delta2 = y - *mean;
            *m2 += delta * delta2;
        }
    }

    /// `(predicted_class, probabilities)`, padded to `n_classes` wide (a
    /// class seen only later in the stream can make some leaves' `counts`
    /// shorter than the tree's current `n_classes`).
    fn predict_classif(&self, n_classes: usize) -> (usize, Vec<f64>) {
        match self {
            NodeStats::Classification { counts } => {
                let total: usize = counts.iter().sum();
                let mut probs = vec![0.0; n_classes.max(counts.len())];
                if total > 0 {
                    for (c, &cnt) in counts.iter().enumerate() {
                        probs[c] = cnt as f64 / total as f64;
                    }
                } else {
                    let k = probs.len().max(1);
                    probs.iter_mut().for_each(|p| *p = 1.0 / k as f64);
                }
                let pred = probs
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                (pred, probs)
            }
            NodeStats::Regression { .. } => (0, vec![]),
        }
    }

    fn predict_regress(&self) -> f64 {
        match self {
            NodeStats::Regression { mean, .. } => *mean,
            NodeStats::Classification { .. } => 0.0,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
enum MondrianNodeKind {
    Leaf {
        stats: NodeStats,
    },
    Split {
        feature: usize,
        value: f64,
        left: Box<MondrianNode>,
        right: Box<MondrianNode>,
    },
}

/// A node's `min_d`/`max_d` bounding box grows over the node's lifetime as
/// points pass through it (whether it ends up absorbed here or in a
/// descendant); `tau` is this node's OWN split time if it's a `Split` node,
/// or the time it would need to exceed to split (capped at the tree's
/// `lifetime`) if it's a `Leaf`.
#[derive(Clone, Serialize, Deserialize)]
struct MondrianNode {
    min_d: Vec<f64>,
    max_d: Vec<f64>,
    #[serde(with = "tau_serde")]
    tau: f64,
    kind: MondrianNodeKind,
}

/// serde adapter for `tau`, which is `f64::INFINITY` at every leaf when the
/// tree uses the default unlimited `lifetime`. JSON has no infinity:
/// serde_json silently writes non-finite floats as `null`, which the plain
/// `f64` deserializer then rejects — so without this adapter every model
/// saved with the default lifetime produced a file `load_json` could never
/// read back. Encodes non-finite as an explicit `null` and decodes `null`
/// back to `INFINITY`, which also recovers files written before the adapter
/// existed.
mod tau_serde {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &f64, s: S) -> Result<S::Ok, S::Error> {
        if v.is_finite() { s.serialize_f64(*v) } else { s.serialize_none() }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
        Ok(Option::<f64>::deserialize(d)?.unwrap_or(f64::INFINITY))
    }
}

impl MondrianNode {
    fn new_leaf_singleton(x: &[f64], stats: NodeStats, lifetime: f64) -> Box<Self> {
        Box::new(MondrianNode {
            min_d: x.to_vec(),
            max_d: x.to_vec(),
            tau: lifetime,
            kind: MondrianNodeKind::Leaf { stats },
        })
    }
}

/// Batch construction (Lakshminarayanan et al. 2014, Algorithm 1,
/// `SampleMondrianBlock`): recursively partitions `indices` by sampling a
/// split time, dimension, and location from the data's own bounding box,
/// stopping once the sampled time exceeds `lifetime` or a single point
/// remains.
#[allow(clippy::too_many_arguments)]
fn sample_mondrian_block(
    indices: &[usize],
    features: &Array2<f64>,
    labels: Option<&[usize]>,
    targets: Option<&[f64]>,
    n_classes: usize,
    tau_parent: f64,
    lifetime: f64,
    rng: &mut StdRng,
) -> Box<MondrianNode> {
    let p = features.ncols();
    let mut min_d = vec![f64::INFINITY; p];
    let mut max_d = vec![f64::NEG_INFINITY; p];
    for &i in indices {
        for d in 0..p {
            let v = features[[i, d]];
            min_d[d] = min_d[d].min(v);
            max_d[d] = max_d[d].max(v);
        }
    }

    let mut stats = if labels.is_some() {
        NodeStats::new_classif(n_classes)
    } else {
        NodeStats::new_regress()
    };
    for &i in indices {
        if let Some(labels) = labels {
            stats.update_classif(labels[i]);
        }
        if let Some(targets) = targets {
            stats.update_regress(targets[i]);
        }
    }

    let box_size: f64 = (0..p).map(|d| max_d[d] - min_d[d]).sum();
    let e = sample_exponential(rng, box_size);
    let tau = tau_parent + e;

    if tau >= lifetime || indices.len() <= 1 {
        return Box::new(MondrianNode {
            min_d,
            max_d,
            tau: lifetime,
            kind: MondrianNodeKind::Leaf { stats },
        });
    }

    let ranges: Vec<f64> = (0..p).map(|d| max_d[d] - min_d[d]).collect();
    let dim = weighted_choice(rng, &ranges);
    let split_value = min_d[dim] + rng.random::<f64>() * ranges[dim];

    let (left_idx, right_idx): (Vec<usize>, Vec<usize>) = indices
        .iter()
        .copied()
        .partition(|&i| features[[i, dim]] <= split_value);

    if left_idx.is_empty() || right_idx.is_empty() {
        // Degenerate split (shouldn't normally happen since split_value is
        // strictly inside (min_d[dim], max_d[dim]) whenever ranges[dim] >
        // 0, but float edge cases are cheap to guard against).
        return Box::new(MondrianNode {
            min_d,
            max_d,
            tau: lifetime,
            kind: MondrianNodeKind::Leaf { stats },
        });
    }

    let left = sample_mondrian_block(&left_idx, features, labels, targets, n_classes, tau, lifetime, rng);
    let right = sample_mondrian_block(&right_idx, features, labels, targets, n_classes, tau, lifetime, rng);

    Box::new(MondrianNode {
        min_d,
        max_d,
        tau,
        kind: MondrianNodeKind::Split {
            feature: dim,
            value: split_value,
            left,
            right,
        },
    })
}

/// One data point to extend the tree with.
enum Sample {
    Classif { label: usize },
    Regress { y: f64 },
}

/// Online extension (Algorithm 3, `ExtendMondrianBlock`): grows `node`'s
/// bounding box to include `x`, introducing a new split *above* `node`
/// (splitting off a fresh singleton leaf for `x`) whenever the sampled
/// expansion time would have occurred before `node`'s own recorded split
/// time -- exactly reproducing what batch construction on the enlarged
/// point set would have sampled at this point in the tree. Otherwise grows
/// `node`'s box in place and recurses into whichever child `x` belongs to
/// (or updates the leaf's running statistics).
fn extend_node(
    mut node: Box<MondrianNode>,
    tau_parent: f64,
    x: &[f64],
    sample: &Sample,
    n_classes: usize,
    lifetime: f64,
    rng: &mut StdRng,
) -> Box<MondrianNode> {
    let p = x.len();
    let mut expand: Vec<f64> = Vec::with_capacity(p);
    let mut expand_size = 0.0;
    for d in 0..p {
        let el = (node.min_d[d] - x[d]).max(0.0);
        let eu = (x[d] - node.max_d[d]).max(0.0);
        expand.push(el + eu);
        expand_size += el + eu;
    }

    let e = sample_exponential(rng, expand_size);

    if tau_parent + e < node.tau {
        let split_time = tau_parent + e;
        let dim = weighted_choice(rng, &expand);
        let (lo, hi) = if x[dim] < node.min_d[dim] {
            (x[dim], node.min_d[dim])
        } else {
            (node.max_d[dim], x[dim])
        };
        let split_value = lo + rng.random::<f64>() * (hi - lo);

        let mut new_stats = match sample {
            Sample::Classif { .. } => NodeStats::new_classif(n_classes),
            Sample::Regress { .. } => NodeStats::new_regress(),
            
        };
        match sample {
            Sample::Classif { label } => new_stats.update_classif(*label),
            Sample::Regress { y } => new_stats.update_regress(*y),
            
        }
        let new_leaf = MondrianNode::new_leaf_singleton(x, new_stats, lifetime);

        let old_min = node.min_d.clone();
        let old_max = node.max_d.clone();
        let (left, right) = if x[dim] <= split_value {
            (new_leaf, node)
        } else {
            (node, new_leaf)
        };
        let min_d: Vec<f64> = (0..p).map(|d| old_min[d].min(x[d])).collect();
        let max_d: Vec<f64> = (0..p).map(|d| old_max[d].max(x[d])).collect();

        return Box::new(MondrianNode {
            min_d,
            max_d,
            tau: split_time,
            kind: MondrianNodeKind::Split {
                feature: dim,
                value: split_value,
                left,
                right,
            },
        });
    }

    for d in 0..p {
        node.min_d[d] = node.min_d[d].min(x[d]);
        node.max_d[d] = node.max_d[d].max(x[d]);
    }

    match &mut node.kind {
        MondrianNodeKind::Leaf { stats } => match sample {
            Sample::Classif { label } => stats.update_classif(*label),
            Sample::Regress { y } => stats.update_regress(*y),
            
        },
        MondrianNodeKind::Split { feature, value, left, right } => {
            let go_left = x[*feature] <= *value;
            let tau_here = node.tau;
            if go_left {
                let child = std::mem::replace(left, MondrianNode::new_leaf_singleton(x, NodeStats::new_regress(), lifetime));
                *left = extend_node(child, tau_here, x, sample, n_classes, lifetime, rng);
            } else {
                let child = std::mem::replace(right, MondrianNode::new_leaf_singleton(x, NodeStats::new_regress(), lifetime));
                *right = extend_node(child, tau_here, x, sample, n_classes, lifetime, rng);
            }
        }
    }
    node
}

fn find_leaf_stats<'a>(node: &'a MondrianNode, x: &[f64]) -> &'a NodeStats {
    match &node.kind {
        MondrianNodeKind::Leaf { stats } => stats,
        MondrianNodeKind::Split { feature, value, left, right } => {
            if x[*feature] <= *value {
                find_leaf_stats(left, x)
            } else {
                find_leaf_stats(right, x)
            }
        }
    }
}

/// A single Mondrian tree. Use [`MondrianForest`] for an ensemble (an
/// ensemble of independently-seeded trees, since the stochastic partition
/// process itself -- not bagging -- is what gives each tree its diversity,
/// matching the original paper: every tree sees the same stream).
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
///
/// let mut tree = MondrianTree::new().with_seed(1);
/// for i in 0..200 {
///     let x = (i % 20) as f64 / 20.0;
///     let y = if x > 0.5 { 1 } else { 0 };
///     tree.partial_fit_classif(&[x], y, 2);
/// }
/// let (pred, _probs) = tree.predict_one_classif(&[0.9]).unwrap();
/// assert_eq!(pred, 1);
/// ```
pub struct MondrianTree {
    lifetime: f64,
    seed: u64,
    rng: StdRng,
    root: Option<Box<MondrianNode>>,
    n_classes: usize,
    n_features: usize,
    is_classifier: bool,
}

impl Default for MondrianTree {
    fn default() -> Self {
        Self::new()
    }
}

impl MondrianTree {
    /// Creates a `MondrianTree` with unlimited lifetime (splits until a
    /// leaf's bounding box has zero extent or a single point remains --
    /// like this crate's other trees defaulting `max_depth` to unlimited)
    /// and seed 42.
    pub fn new() -> Self {
        Self {
            lifetime: f64::INFINITY,
            seed: 42,
            rng: StdRng::seed_from_u64(42),
            root: None,
            n_classes: 0,
            n_features: 0,
            is_classifier: false,
        }
    }

    /// Sets the lifetime (Mondrian process time budget): smaller values
    /// stop splitting sooner, regularizing the tree similarly to a
    /// continuous analog of `max_depth`. Defaults to unlimited.
    pub fn with_lifetime(mut self, lifetime: f64) -> Self {
        self.lifetime = lifetime;
        self
    }

    /// Sets the RNG seed controlling the Mondrian process's random split
    /// times, dimensions, and locations.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self.rng = StdRng::seed_from_u64(seed);
        self
    }

    /// Online update: extend the tree with a single labeled point.
    pub fn partial_fit_classif(&mut self, x: &[f64], label: usize, n_classes: usize) {
        self.is_classifier = true;
        self.n_features = self.n_features.max(x.len());
        self.n_classes = self.n_classes.max(n_classes).max(label + 1);
        let sample = Sample::Classif { label };
        self.root = Some(match self.root.take() {
            None => {
                let mut stats = NodeStats::new_classif(self.n_classes);
                stats.update_classif(label);
                MondrianNode::new_leaf_singleton(x, stats, self.lifetime)
            }
            Some(root) => extend_node(root, 0.0, x, &sample, self.n_classes, self.lifetime, &mut self.rng),
        });
    }

    /// Online update: extend the tree with a single (features, target) pair.
    pub fn partial_fit_regress(&mut self, x: &[f64], y: f64) {
        self.is_classifier = false;
        self.n_features = self.n_features.max(x.len());
        let sample = Sample::Regress { y };
        self.root = Some(match self.root.take() {
            None => {
                let mut stats = NodeStats::new_regress();
                stats.update_regress(y);
                MondrianNode::new_leaf_singleton(x, stats, self.lifetime)
            }
            Some(root) => extend_node(root, 0.0, x, &sample, self.n_classes, self.lifetime, &mut self.rng),
        });
    }

    /// Predicts from the tree's current (possibly still-streaming) state.
    /// Returns `None` before it has seen any sample.
    pub fn predict_one_classif(&self, x: &[f64]) -> Option<(usize, Vec<f64>)> {
        let root = self.root.as_ref()?;
        Some(find_leaf_stats(root, x).predict_classif(self.n_classes))
    }

    /// Predicts a continuous target from the tree's current state. Returns
    /// `None` before it has seen any sample.
    pub fn predict_one_regress(&self, x: &[f64]) -> Option<f64> {
        let root = self.root.as_ref()?;
        Some(find_leaf_stats(root, x).predict_regress())
    }

    /// Rebuilds the tree from scratch via batch construction
    /// ([`sample_mondrian_block`]) instead of replaying points one at a
    /// time -- identical in distribution, and what [`Learner::train_classif`]
    /// uses under the hood.
    pub fn fit_batch_classif(&mut self, features: &Array2<f64>, labels: &[usize], n_classes: usize) {
        self.is_classifier = true;
        self.n_features = features.ncols();
        self.n_classes = n_classes;
        let indices: Vec<usize> = (0..features.nrows()).collect();
        self.root = Some(sample_mondrian_block(
            &indices,
            features,
            Some(labels),
            None,
            n_classes,
            0.0,
            self.lifetime,
            &mut self.rng,
        ));
    }

    /// Batch-constructs the tree for regression.
    pub fn fit_batch_regress(&mut self, features: &Array2<f64>, targets: &[f64]) {
        self.is_classifier = false;
        self.n_features = features.ncols();
        let indices: Vec<usize> = (0..features.nrows()).collect();
        self.root = Some(sample_mondrian_block(
            &indices,
            features,
            None,
            Some(targets),
            0,
            0.0,
            self.lifetime,
            &mut self.rng,
        ));
    }
}

/// Ensemble of independently-seeded [`MondrianTree`]s. Unlike
/// [`crate::learner::AdaptiveRandomForest`], there is no online bagging:
/// every tree sees every sample, and diversity across trees comes purely
/// from each tree's own random stream of split times/dimensions/locations
/// (matching Lakshminarayanan et al. 2014's actual ensemble design, not an
/// added embellishment).
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
///
/// let mut forest = MondrianForest::new().with_n_trees(10).with_seed(7);
/// for i in 0..300 {
///     let x = (i % 30) as f64 / 30.0;
///     let y = if x > 0.5 { 1 } else { 0 };
///     forest.partial_fit_classif(&[x], y, 2);
/// }
/// let (pred, _probs) = forest.predict_one_classif(&[0.9]).unwrap();
/// assert_eq!(pred, 1);
/// ```
pub struct MondrianForest {
    n_trees: usize,
    lifetime: f64,
    seed: u64,
    trees: Vec<MondrianTree>,
    n_classes: usize,
    n_features: usize,
    is_classifier: bool,
}

impl Default for MondrianForest {
    fn default() -> Self {
        Self::new()
    }
}

impl MondrianForest {
    /// Creates a `MondrianForest` with 10 trees and unlimited lifetime.
    pub fn new() -> Self {
        Self {
            n_trees: 10,
            lifetime: f64::INFINITY,
            seed: 42,
            trees: Vec::new(),
            n_classes: 0,
            n_features: 0,
            is_classifier: false,
        }
    }

    /// Sets the number of trees in the ensemble.
    pub fn with_n_trees(mut self, n: usize) -> Self {
        self.n_trees = n.max(1);
        self
    }

    /// Sets each member tree's lifetime (see [`MondrianTree::with_lifetime`]).
    pub fn with_lifetime(mut self, lifetime: f64) -> Self {
        self.lifetime = lifetime;
        self
    }

    /// Sets the RNG seed; member tree `i` is seeded with `seed + i` so the
    /// whole forest is reproducible from one value.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    fn ensure_trees(&mut self) {
        if self.trees.is_empty() {
            self.trees = (0..self.n_trees)
                .map(|i| {
                    MondrianTree::new()
                        .with_lifetime(self.lifetime)
                        .with_seed(self.seed.wrapping_add(i as u64))
                })
                .collect();
        }
    }

    /// Online update across every member tree.
    pub fn partial_fit_classif(&mut self, x: &[f64], label: usize, n_classes: usize) {
        self.is_classifier = true;
        self.n_features = self.n_features.max(x.len());
        self.n_classes = self.n_classes.max(n_classes).max(label + 1);
        self.ensure_trees();
        for tree in &mut self.trees {
            tree.partial_fit_classif(x, label, self.n_classes);
        }
    }

    /// Online update across every member tree.
    pub fn partial_fit_regress(&mut self, x: &[f64], y: f64) {
        self.is_classifier = false;
        self.n_features = self.n_features.max(x.len());
        self.ensure_trees();
        for tree in &mut self.trees {
            tree.partial_fit_regress(x, y);
        }
    }

    /// Majority-vote prediction across member trees. `None` before any
    /// tree has seen a sample.
    pub fn predict_one_classif(&self, x: &[f64]) -> Option<(usize, Vec<f64>)> {
        if self.trees.is_empty() {
            return None;
        }
        let mut votes = vec![0usize; self.n_classes.max(1)];
        let mut any = false;
        for tree in &self.trees {
            if let Some((pred, _)) = tree.predict_one_classif(x) {
                any = true;
                if pred < votes.len() {
                    votes[pred] += 1;
                }
            }
        }
        if !any {
            return None;
        }
        let total: usize = votes.iter().sum();
        let probs: Vec<f64> = if total > 0 {
            votes.iter().map(|&v| v as f64 / total as f64).collect()
        } else {
            vec![1.0 / votes.len() as f64; votes.len()]
        };
        let pred = votes
            .iter()
            .enumerate()
            .max_by_key(|&(_, &c)| c)
            .map(|(i, _)| i)
            .unwrap_or(0);
        Some((pred, probs))
    }

    /// Mean prediction across member trees. `None` before any tree has seen
    /// a sample.
    pub fn predict_one_regress(&self, x: &[f64]) -> Option<f64> {
        if self.trees.is_empty() {
            return None;
        }
        let preds: Vec<f64> = self.trees.iter().filter_map(|t| t.predict_one_regress(x)).collect();
        if preds.is_empty() {
            None
        } else {
            Some(preds.iter().sum::<f64>() / preds.len() as f64)
        }
    }
}

/// A negative or NaN lifetime silently degenerates the Mondrian process:
/// every sampled split time exceeds the budget immediately (NaN poisons the
/// `tau >= lifetime` comparison the same way), so no tree ever splits and
/// the model is a single running-stats leaf. `f64::INFINITY` — the default
/// — remains valid (unlimited budget), as does `0.0` (the limiting valid
/// budget). The builders don't return `Result`, so the check runs at train
/// time (5th audit, LOW-D).
fn check_lifetime(lifetime: f64) -> Result<()> {
    if lifetime.is_nan() || lifetime < 0.0 {
        return Err(crate::SmeltError::InvalidParameter(format!(
            "mondrian lifetime must be non-negative (infinity, the default, is valid), got {lifetime}"
        )));
    }
    Ok(())
}

impl Learner for MondrianTree {
    fn id(&self) -> &str {
        "mondrian_tree"
    }

    fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        crate::validate::check_no_weights(task.weights(), "MondrianTree")?;
        check_lifetime(self.lifetime)?;
        crate::validate::check_no_nan(task.features())?;
        self.fit_batch_classif(task.features(), task.target(), task.n_classes());
        Ok(Box::new(TrainedMondrianTree {
            root: self.root.take(),
            n_classes: self.n_classes,
            n_features: self.n_features,
            is_classifier: true,
        }))
    }

    fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
        crate::validate::check_no_weights(task.weights(), "MondrianTree")?;
        check_lifetime(self.lifetime)?;
        crate::validate::check_no_nan(task.features())?;
        self.fit_batch_regress(task.features(), task.target());
        Ok(Box::new(TrainedMondrianTree {
            root: self.root.take(),
            n_classes: 0,
            n_features: self.n_features,
            is_classifier: false,
        }))
    }
}

/// A trained Mondrian tree.
#[derive(Clone, Serialize, Deserialize)]
pub struct TrainedMondrianTree {
    root: Option<Box<MondrianNode>>,
    n_classes: usize,
    n_features: usize,
    is_classifier: bool,
}

impl TrainedModel for TrainedMondrianTree {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        crate::validate::check_n_features(features, self.n_features)?;
        let Some(root) = &self.root else {
            return Err(crate::SmeltError::NotTrained);
        };
        if self.is_classifier {
            let mut predicted = Vec::with_capacity(features.nrows());
            let mut probabilities = Vec::with_capacity(features.nrows());
            for row in features.rows() {
                let row_vec: Vec<f64> = row.to_vec();
                let (pred, probs) = find_leaf_stats(root, &row_vec).predict_classif(self.n_classes);
                predicted.push(pred);
                probabilities.push(probs);
            }
            Ok(Prediction::Classification {
                predicted,
                truth: None,
                probabilities: Some(probabilities),
            })
        } else {
            let predicted: Vec<f64> = features
                .rows()
                .into_iter()
                .map(|row| {
                    let row_vec: Vec<f64> = row.to_vec();
                    find_leaf_stats(root, &row_vec).predict_regress()
                })
                .collect();
            Ok(Prediction::regression(predicted))
        }
    }

    fn to_serializable(&self) -> Option<crate::serialize::SerializableModel> {
        Some(crate::serialize::SerializableModel::MondrianTree(
            self.clone(),
        ))
    }
}

impl Learner for MondrianForest {
    fn id(&self) -> &str {
        "mondrian_forest"
    }

    fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        crate::validate::check_no_weights(task.weights(), "MondrianForest")?;
        check_lifetime(self.lifetime)?;
        crate::validate::check_no_nan(task.features())?;
        let features = task.features();
        let target = task.target();
        let n_classes = task.n_classes();

        self.trees.clear();
        self.n_classes = 0;
        self.ensure_trees();
        self.n_classes = n_classes;

        for i in 0..task.n_samples() {
            let row: Vec<f64> = features.row(i).to_vec();
            for tree in &mut self.trees {
                tree.partial_fit_classif(&row, target[i], n_classes);
            }
        }

        Ok(Box::new(TrainedMondrianForest {
            trees: std::mem::take(&mut self.trees)
                .into_iter()
                .filter_map(|t| t.root.map(|root| (root, true)))
                .collect(),
            n_features: task.n_features(),
            n_classes: self.n_classes,
            is_classifier: true,
        }))
    }

    fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
        crate::validate::check_no_weights(task.weights(), "MondrianForest")?;
        check_lifetime(self.lifetime)?;
        crate::validate::check_no_nan(task.features())?;
        let features = task.features();
        let target = task.target();

        self.trees.clear();
        self.ensure_trees();

        for i in 0..task.n_samples() {
            let row: Vec<f64> = features.row(i).to_vec();
            for tree in &mut self.trees {
                tree.partial_fit_regress(&row, target[i]);
            }
        }

        Ok(Box::new(TrainedMondrianForest {
            trees: std::mem::take(&mut self.trees)
                .into_iter()
                .filter_map(|t| t.root.map(|root| (root, false)))
                .collect(),
            n_features: task.n_features(),
            n_classes: 0,
            is_classifier: false,
        }))
    }
}

/// A trained Mondrian forest.
#[derive(Clone, Serialize, Deserialize)]
pub struct TrainedMondrianForest {
    trees: Vec<(Box<MondrianNode>, bool)>,
    n_features: usize,
    n_classes: usize,
    is_classifier: bool,
}

impl TrainedModel for TrainedMondrianForest {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        crate::validate::check_n_features(features, self.n_features)?;
        if self.is_classifier {
            let mut predicted = Vec::with_capacity(features.nrows());
            let mut probabilities = Vec::with_capacity(features.nrows());
            for row in features.rows() {
                let row_vec: Vec<f64> = row.to_vec();
                let mut votes = vec![0usize; self.n_classes.max(1)];
                for (root, _) in &self.trees {
                    let (pred, _) = find_leaf_stats(root, &row_vec).predict_classif(self.n_classes);
                    if pred < votes.len() {
                        votes[pred] += 1;
                    }
                }
                let total: usize = votes.iter().sum();
                let probs: Vec<f64> = if total > 0 {
                    votes.iter().map(|&v| v as f64 / total as f64).collect()
                } else {
                    vec![1.0 / votes.len() as f64; votes.len()]
                };
                let pred = votes
                    .iter()
                    .enumerate()
                    .max_by_key(|&(_, &c)| c)
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                predicted.push(pred);
                probabilities.push(probs);
            }
            Ok(Prediction::Classification {
                predicted,
                truth: None,
                probabilities: Some(probabilities),
            })
        } else {
            let predicted: Vec<f64> = features
                .rows()
                .into_iter()
                .map(|row| {
                    let row_vec: Vec<f64> = row.to_vec();
                    let preds: Vec<f64> = self
                        .trees
                        .iter()
                        .map(|(root, _)| find_leaf_stats(root, &row_vec).predict_regress())
                        .collect();
                    preds.iter().sum::<f64>() / preds.len().max(1) as f64
                })
                .collect();
            Ok(Prediction::regression(predicted))
        }
    }

    fn to_serializable(&self) -> Option<crate::serialize::SerializableModel> {
        Some(crate::serialize::SerializableModel::MondrianForest(
            self.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// Regression test (5th audit, LOW-D): a negative or NaN lifetime used
    /// to be accepted and silently degenerated every tree to a single
    /// running-stats leaf (any sampled split time exceeds the budget; NaN
    /// poisons the comparison). Both learners must reject it at train time,
    /// while the infinite default stays valid.
    #[test]
    fn negative_or_nan_lifetime_is_rejected_at_train() {
        let features = Array2::from_shape_fn((6, 1), |(i, _)| i as f64);
        let classif =
            ClassificationTask::new("mt_life_c", features.clone(), vec![0, 0, 0, 1, 1, 1]).unwrap();
        let regress = RegressionTask::new(
            "mt_life_r",
            features,
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        )
        .unwrap();

        for bad in [-1.0, f64::NAN, f64::NEG_INFINITY] {
            let Err(err) = MondrianTree::new().with_lifetime(bad).train_classif(&classif)
            else {
                panic!("bad lifetime must be rejected (tree, classif)");
            };
            assert!(
                matches!(err, crate::SmeltError::InvalidParameter(_))
                    && format!("{err}").contains("lifetime"),
                "got: {err}"
            );
            assert!(
                MondrianTree::new().with_lifetime(bad).train_regress(&regress).is_err(),
                "bad lifetime must be rejected (tree, regress)"
            );
            assert!(
                MondrianForest::new().with_lifetime(bad).train_classif(&classif).is_err(),
                "bad lifetime must be rejected (forest, classif)"
            );
            assert!(
                MondrianForest::new().with_lifetime(bad).train_regress(&regress).is_err(),
                "bad lifetime must be rejected (forest, regress)"
            );
        }

        // The infinite default (and an explicit finite budget) remain valid.
        MondrianTree::new().train_classif(&classif).unwrap();
        MondrianForest::new().with_lifetime(5.0).train_regress(&regress).unwrap();
    }

    #[test]
    fn sample_exponential_mean_matches_rate() {
        let mut rng = StdRng::seed_from_u64(1);
        let rate = 2.0;
        let n = 20_000;
        let total: f64 = (0..n).map(|_| sample_exponential(&mut rng, rate)).sum();
        let mean = total / n as f64;
        assert!(
            (mean - 1.0 / rate).abs() < 0.05,
            "empirical mean {mean} should be close to 1/rate = {}",
            1.0 / rate
        );
    }

    #[test]
    fn sample_exponential_zero_rate_is_infinite() {
        let mut rng = StdRng::seed_from_u64(1);
        assert_eq!(sample_exponential(&mut rng, 0.0), f64::INFINITY);
    }

    #[test]
    fn weighted_choice_favors_larger_weights() {
        let mut rng = StdRng::seed_from_u64(2);
        let mut counts = [0usize; 3];
        for _ in 0..10_000 {
            counts[weighted_choice(&mut rng, &[1.0, 0.0, 9.0])] += 1;
        }
        assert_eq!(counts[1], 0, "zero-weight option should never be picked");
        assert!(counts[2] > counts[0] * 5, "index 2 (weight 9) should dominate index 0 (weight 1)");
    }

    #[test]
    fn registered_id_matches() {
        assert_eq!(MondrianForest::new().id(), "mondrian_forest");
        assert_eq!(MondrianTree::new().id(), "mondrian_tree");
    }

    #[test]
    fn batch_fit_classif_separates_a_simple_threshold_rule() {
        let mut rng = StdRng::seed_from_u64(3);
        let n = 400;
        let mut feats = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);
        for _ in 0..n {
            let x: f64 = rng.random();
            feats.push(x);
            target.push(if x > 0.5 { 1usize } else { 0 });
        }
        let features = Array2::from_shape_vec((n, 1), feats).unwrap();
        let task = ClassificationTask::new("mondrian", features.clone(), target.clone()).unwrap();

        let mut forest = MondrianForest::new().with_n_trees(10).with_seed(1);
        let model = forest.train_classif(&task).unwrap();
        let pred = model.predict(&features).unwrap();
        let Prediction::Classification { predicted, .. } = pred else {
            panic!("expected classification");
        };
        let correct = predicted.iter().zip(&target).filter(|(p, t)| *p == *t).count();
        let acc = correct as f64 / n as f64;
        assert!(acc > 0.85, "should fit a simple threshold rule well, got acc={acc}");
    }

    #[test]
    fn batch_fit_regress_fits_a_linear_trend() {
        let mut rng = StdRng::seed_from_u64(4);
        let n = 300;
        let mut feats = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);
        for _ in 0..n {
            let x: f64 = rng.random::<f64>() * 10.0;
            feats.push(x);
            target.push(2.0 * x + rng.random::<f64>() * 0.1);
        }
        let features = Array2::from_shape_vec((n, 1), feats).unwrap();
        let task = RegressionTask::new("mondrian_r", features.clone(), target.clone()).unwrap();

        let mut forest = MondrianForest::new().with_n_trees(15).with_seed(2);
        let model = forest.train_regress(&task).unwrap();
        let pred = model.predict(&features).unwrap();
        let Prediction::Regression { predicted, .. } = pred else {
            panic!("expected regression");
        };
        let rmse = (predicted
            .iter()
            .zip(&target)
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / n as f64)
            .sqrt();
        assert!(rmse < 2.0, "should fit a clear linear trend reasonably well, got RMSE={rmse}");
    }

    /// Regression test for the actual differentiator: a point arriving
    /// *outside* the tree's current bounding box must expand the tree's
    /// coverage (via `extend_node`'s new-split-above-node branch), not
    /// silently get absorbed by whatever leaf currently sits nearest in
    /// threshold space. Train on a narrow initial range, then check the
    /// tree can distinguish a far-outside point pair it never directly saw
    /// bracketed by training data on both sides.
    #[test]
    fn online_extension_grows_tree_coverage_beyond_initial_range() {
        let mut tree = MondrianTree::new().with_seed(5);
        // Initial narrow regime: all points in [0, 1), label always 0.
        let mut rng = StdRng::seed_from_u64(6);
        for _ in 0..200 {
            let x: f64 = rng.random::<f64>();
            tree.partial_fit_classif(&[x], 0, 2);
        }
        // Now extend far outside the original box on both sides with a
        // different label, which can only be captured by new splits
        // introduced above existing nodes.
        for _ in 0..200 {
            tree.partial_fit_classif(&[-5.0 - rng.random::<f64>()], 1, 2);
            tree.partial_fit_classif(&[10.0 + rng.random::<f64>()], 1, 2);
        }

        let (pred_low, _) = tree.predict_one_classif(&[-6.0]).unwrap();
        let (pred_high, _) = tree.predict_one_classif(&[15.0]).unwrap();
        let (pred_mid, _) = tree.predict_one_classif(&[0.5]).unwrap();
        assert_eq!(pred_low, 1, "far-below-original-range point should be captured by the extended tree");
        assert_eq!(pred_high, 1, "far-above-original-range point should be captured by the extended tree");
        assert_eq!(pred_mid, 0, "original-range point should still be classified as before");
    }

    #[test]
    fn ensemble_is_at_least_as_accurate_as_a_single_tree_on_noisy_data() {
        let mut rng = StdRng::seed_from_u64(7);
        let n = 500;
        let mut feats = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);
        for _ in 0..n {
            let x0: f64 = rng.random();
            let x1: f64 = rng.random();
            feats.push(x0);
            feats.push(x1);
            let noisy_label = if rng.random::<f64>() < 0.1 {
                if x0 > 0.5 { 0 } else { 1 } // flipped label (label noise)
            } else if x0 > 0.5 {
                1
            } else {
                0
            };
            target.push(noisy_label);
        }
        let features = Array2::from_shape_vec((n, 2), feats).unwrap();
        let task = ClassificationTask::new("mondrian_noisy", features.clone(), target.clone()).unwrap();

        let mut single = MondrianTree::new().with_seed(1);
        let single_model = single.train_classif(&task).unwrap();
        let mut forest = MondrianForest::new().with_n_trees(20).with_seed(1);
        let forest_model = forest.train_classif(&task).unwrap();

        let acc = |model: &dyn TrainedModel| {
            let Prediction::Classification { predicted, .. } = model.predict(&features).unwrap() else {
                panic!("expected classification");
            };
            predicted.iter().zip(&target).filter(|(p, t)| *p == *t).count() as f64 / n as f64
        };
        let single_acc = acc(&*single_model);
        let forest_acc = acc(&*forest_model);
        assert!(
            forest_acc >= single_acc - 0.05,
            "forest ({forest_acc}) shouldn't be meaningfully worse than a single tree ({single_acc})"
        );
    }

    #[test]
    fn handles_duplicate_points_without_panicking() {
        let features = Array2::from_shape_vec((4, 1), vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let target = vec![0usize, 1, 0, 1];
        let task = ClassificationTask::new("dup", features.clone(), target).unwrap();
        let mut forest = MondrianForest::new().with_n_trees(3).with_seed(1);
        let model = forest.train_classif(&task).unwrap();
        let pred = model.predict(&features);
        assert!(pred.is_ok(), "zero-variance (duplicate) points must not panic");
    }

    #[test]
    fn handles_single_sample() {
        let features = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let target = vec![0usize];
        let task = ClassificationTask::new("single", features.clone(), target).unwrap();
        let mut forest = MondrianForest::new().with_n_trees(3).with_seed(1);
        let model = forest.train_classif(&task).unwrap();
        let pred = model.predict(&features).unwrap();
        let Prediction::Classification { predicted, .. } = pred else {
            panic!("expected classification");
        };
        assert_eq!(predicted, vec![0]);
    }

    #[test]
    fn handles_class_seen_only_later_in_the_stream() {
        // n_classes grows mid-stream (class 2 appears only after many
        // class-0/1 samples) -- must not panic on the width mismatch
        // between older leaves' shorter `counts` and the tree's current
        // n_classes.
        let mut tree = MondrianTree::new().with_seed(9);
        for i in 0..100 {
            tree.partial_fit_classif(&[i as f64], i % 2, 2);
        }
        tree.partial_fit_classif(&[1000.0], 2, 3);
        let pred = tree.predict_one_classif(&[1000.0]);
        assert!(pred.is_some());
    }
}
