"""Spatial ML components: spatial cross-validation strategies."""

from smelt._smelt import SpatialBlockCV, SpatialBufferCV


def spatial_leave_one_out(coords, buffer_distance, seed=42):
    """Spatial Leave-One-Out with a buffer exclusion zone.

    Each fold tests a single sample; training samples within ``buffer_distance``
    of the test sample are excluded, reducing spatial autocorrelation leakage.

    Appropriate for small datasets (n < 100) where SpatialBlockCV removes
    too much training data per fold.

    Parameters
    ----------
    coords : array-like (n_samples, 2)
        (x, y) coordinates for each sample. Accepts numpy array, list of
        tuples, or list of [x, y] lists.
    buffer_distance : float
        Euclidean distance threshold (in same units as coords).
    seed : int, optional (default 42)
        Seed for internal shuffling.

    Returns
    -------
    SpatialBufferCV
        A CV splitter with ``n_folds = n_samples``.
    """
    try:
        n = len(coords)
    except TypeError:
        n = coords.shape[0]
    return SpatialBufferCV(n, coords, buffer_distance, seed)


__all__ = ["SpatialBlockCV", "SpatialBufferCV", "spatial_leave_one_out"]
