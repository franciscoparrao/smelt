//! Minimal Compressed Sparse Row (CSR) matrix.
//!
//! # Scope (item 16d part 3/3)
//!
//! An investigation before writing any code (see
//! `docs/sparse_data_2026-07-03.md`) found `Task::features() -> &Array2<f64>`
//! is a concretely-typed return used at 44 call sites across every learner,
//! with no existing trait-object seam a sparse `Task` could plug into --
//! threading sparsity through `Task`/`Learner` end-to-end would mean either
//! rewriting the `Task` trait's core method or duplicating the entire
//! `Learner` surface, neither justified by current evidence. Of the
//! learners, only linear models (logistic/linear regression, SVM) have a
//! real "sparse dot product is algorithmically faster" story; boosting
//! would need `HistBins` reworked regardless of `Task`'s storage format.
//!
//! The one genuinely wasteful path *today* is [`crate::preprocess::OneHotEncoder`]
//! on a high-cardinality column: `transform` always allocates a dense
//! `n_samples x n_categories` matrix, which is >99% zero for anything with
//! more than a handful of categories. So the scope here is deliberately
//! narrow: a hand-rolled CSR type (no `sprs` dependency -- `ndarray` has no
//! sparse support of its own, and pulling in a whole sparse-linalg crate for
//! "materialize one-hot output without wasting memory" would be a poor
//! trade) plus [`OneHotEncoder::transform_sparse`], with [`CsrMatrix::to_dense`]
//! as the escape hatch for every learner that doesn't (yet) have a sparse
//! code path. Sparse `Task`/`Learner` integration and sparse linear-model
//! math are left as separate, larger follow-ups if ever prioritized.

use crate::{Result, SmeltError};
use ndarray::Array2;

/// A sparse matrix in Compressed Sparse Row format: `indices`/`values` hold
/// the nonzero entries in row-major order, and `row_ptr[i]..row_ptr[i+1]`
/// indexes into them for row `i` (the standard CSR layout).
#[derive(Debug, Clone)]
pub struct CsrMatrix {
    n_rows: usize,
    n_cols: usize,
    indices: Vec<usize>,
    values: Vec<f64>,
    row_ptr: Vec<usize>,
}

impl CsrMatrix {
    /// Build from `(row, col, value)` triplets in any order. Triplets with
    /// the same `(row, col)` are summed, matching the usual sparse-matrix
    /// assembly convention (e.g. duplicate feature hits).
    pub fn from_triplets(
        n_rows: usize,
        n_cols: usize,
        mut triplets: Vec<(usize, usize, f64)>,
    ) -> Result<Self> {
        for &(r, c, _) in &triplets {
            if r >= n_rows || c >= n_cols {
                return Err(SmeltError::InvalidParameter(format!(
                    "triplet ({r}, {c}) out of bounds for a {n_rows}x{n_cols} matrix"
                )));
            }
        }
        triplets.sort_by_key(|&(r, c, _)| (r, c));

        let mut indices = Vec::with_capacity(triplets.len());
        let mut values = Vec::with_capacity(triplets.len());
        let mut row_ptr = vec![0usize; n_rows + 1];
        let mut iter = triplets.into_iter().peekable();
        while let Some((r, c, mut v)) = iter.next() {
            while let Some(&(r2, c2, v2)) = iter.peek() {
                if r2 == r && c2 == c {
                    v += v2;
                    iter.next();
                } else {
                    break;
                }
            }
            indices.push(c);
            values.push(v);
            row_ptr[r + 1] += 1;
        }
        for i in 0..n_rows {
            row_ptr[i + 1] += row_ptr[i];
        }

        Ok(Self {
            n_rows,
            n_cols,
            indices,
            values,
            row_ptr,
        })
    }

    pub fn n_rows(&self) -> usize {
        self.n_rows
    }
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }
    /// Number of stored (nonzero) entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
    /// Fraction of entries that are stored (nonzero), in `[0, 1]`.
    pub fn density(&self) -> f64 {
        if self.n_rows == 0 || self.n_cols == 0 {
            0.0
        } else {
            self.nnz() as f64 / (self.n_rows * self.n_cols) as f64
        }
    }

    /// `(column, value)` pairs for row `i`, in increasing column order.
    pub fn row(&self, i: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let start = self.row_ptr[i];
        let end = self.row_ptr[i + 1];
        self.indices[start..end]
            .iter()
            .copied()
            .zip(self.values[start..end].iter().copied())
    }

    /// Dot product of row `i` with a dense vector of length `n_cols`.
    pub fn dot_row(&self, i: usize, dense: &[f64]) -> f64 {
        self.row(i).map(|(j, v)| v * dense[j]).sum()
    }

    /// Materialize as a dense matrix. The escape hatch for any learner
    /// without a sparse-aware code path.
    pub fn to_dense(&self) -> Array2<f64> {
        let mut out = Array2::zeros((self.n_rows, self.n_cols));
        for i in 0..self.n_rows {
            for (j, v) in self.row(i) {
                out[[i, j]] = v;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_dense_round_trips() {
        let csr = CsrMatrix::from_triplets(2, 3, vec![(0, 0, 1.0), (0, 2, 3.0), (1, 1, 2.0)]).unwrap();
        let dense = csr.to_dense();
        assert_eq!(dense, ndarray::array![[1.0, 0.0, 3.0], [0.0, 2.0, 0.0]]);
    }

    #[test]
    fn duplicate_triplets_sum() {
        let csr = CsrMatrix::from_triplets(1, 1, vec![(0, 0, 1.0), (0, 0, 2.0)]).unwrap();
        assert_eq!(csr.to_dense()[[0, 0]], 3.0);
        assert_eq!(csr.nnz(), 1);
    }

    #[test]
    fn dot_row_matches_dense() {
        let csr = CsrMatrix::from_triplets(1, 3, vec![(0, 0, 1.0), (0, 2, 4.0)]).unwrap();
        let dense = vec![2.0, 100.0, 5.0];
        assert_eq!(csr.dot_row(0, &dense), 1.0 * 2.0 + 4.0 * 5.0);
    }

    #[test]
    fn density_reflects_sparsity() {
        let csr = CsrMatrix::from_triplets(10, 10, vec![(0, 0, 1.0)]).unwrap();
        assert!((csr.density() - 0.01).abs() < 1e-9);
    }

    #[test]
    fn out_of_bounds_triplet_errors() {
        let err = CsrMatrix::from_triplets(2, 2, vec![(5, 0, 1.0)]);
        assert!(err.is_err());
    }

    #[test]
    fn empty_matrix() {
        let csr = CsrMatrix::from_triplets(3, 3, vec![]).unwrap();
        assert_eq!(csr.nnz(), 0);
        assert_eq!(csr.to_dense(), Array2::<f64>::zeros((3, 3)));
    }
}
