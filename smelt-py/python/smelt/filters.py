"""Feature selection utilities with cumulative ranking across multiple filters."""

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from smelt._smelt import (
    filter_variance,
    filter_correlation,
    filter_anova_f,
    filter_information_gain,
    filter_mutual_information,
    filter_mrmr,
    filter_jmi,
    filter_jmim,
    filter_cmim,
    filter_relief,
)

ALL_FILTERS = [
    "variance", "correlation", "anova_f", "information_gain",
    "mutual_information", "mrmr", "jmi", "jmim", "cmim", "relief",
]

_FILTER_FUNCS = {
    "variance": filter_variance,
    "correlation": filter_correlation,
    "anova_f": filter_anova_f,
    "information_gain": filter_information_gain,
    "mutual_information": filter_mutual_information,
    "mrmr": filter_mrmr,
    "jmi": filter_jmi,
    "jmim": filter_jmim,
    "cmim": filter_cmim,
    "relief": filter_relief,
}


def cumulative_ranking(X, y, feature_names, filters=None, top_k=15, corr_cutoff=0.9):
    """
    Compute cumulative ranking across multiple filter methods.

    Algorithm:
    1. Run each filter on (X, y) -> selected features in order
    2. Assign ordinal rank per filter (1 = best)
    3. Sum ranks across all filters per feature
    4. Select top_k features with lowest cumulative rank
    5. Remove features with inter-correlation > corr_cutoff

    Parameters
    ----------
    X : np.ndarray (n_samples, n_features)
    y : np.ndarray (n_samples,)
    feature_names : list[str]
    filters : list[str], optional (default: all 10 filters)
    top_k : int (default: 15)
    corr_cutoff : float (default: 0.9)

    Returns
    -------
    pandas.DataFrame (if pandas available) or list[dict] with columns:
        feature, cumulative_rank, rank_{filter_name} for each filter
    """
    if filters is None:
        filters = ALL_FILTERS

    p = len(feature_names)
    all_ranks = {}

    for fname in filters:
        func = _FILTER_FUNCS[fname]
        # Run filter selecting ALL features to get full ranking
        result = func(X, y, feature_names, k=p)
        # result is list of (name, index) in selection order
        rank_map = {}
        for rank, (name, _idx) in enumerate(result):
            rank_map[name] = rank + 1  # 1 = best
        # Features not selected get worst rank
        for name in feature_names:
            if name not in rank_map:
                rank_map[name] = p + 1
        all_ranks[fname] = rank_map

    # Cumulative rank (lower = better)
    cumulative = {}
    for feat in feature_names:
        cumulative[feat] = sum(
            all_ranks.get(f, {}).get(feat, p + 1) for f in filters
        )

    # Sort by cumulative rank
    sorted_feats = sorted(feature_names, key=lambda f: cumulative[f])

    # Select top_k
    selected = sorted_feats[:top_k]

    # Remove high inter-correlation
    if corr_cutoff < 1.0 and len(selected) > 1:
        sel_idx = [feature_names.index(f) for f in selected]
        corr_matrix = np.corrcoef(X[:, sel_idx].T)
        to_remove = set()
        for i in range(len(selected)):
            if selected[i] in to_remove:
                continue
            for j in range(i + 1, len(selected)):
                if selected[j] in to_remove:
                    continue
                if abs(corr_matrix[i, j]) > corr_cutoff:
                    # Remove the one with worse cumulative rank
                    if cumulative[selected[j]] >= cumulative[selected[i]]:
                        to_remove.add(selected[j])
                    else:
                        to_remove.add(selected[i])
        selected = [f for f in selected if f not in to_remove]

    # Build result
    rows = []
    for feat in selected:
        row = {"feature": feat, "cumulative_rank": cumulative[feat]}
        for fname in filters:
            row[f"rank_{fname}"] = all_ranks.get(fname, {}).get(feat, p + 1)
        rows.append(row)

    if HAS_PANDAS:
        return pd.DataFrame(rows)
    return rows
