from typing import List, Tuple


def linear_sum_assignment(cost_matrix: List[List[float]]) -> Tuple[List[int], List[int]]:
    """Hungarian (linear sum assignment) algorithm wrapper.
    Requires `scipy`; calls `scipy.optimize.linear_sum_assignment`.

    Input:
    - `cost_matrix`: cost matrix of shape `n x m`; lower values indicate better matches.

    Output:
    - `(rows, cols)`: two equal-length index lists representing the selected row-column match pairs.

    Notes:
    - When using a similarity matrix, convert to cost first: `cost = 1 - similarity`.
    - No sub-optimal fallback is provided; if `scipy` is unavailable, an exception is raised directly.
    """
    try:
        from scipy.optimize import linear_sum_assignment as lsa
        import numpy as np
    except Exception as e:
        raise RuntimeError("scipy is not installed or unavailable; Hungarian algorithm requires scipy") from e
    cm = np.array(cost_matrix, dtype=float)
    row_ind, col_ind = lsa(cm)
    return row_ind.tolist(), col_ind.tolist()
