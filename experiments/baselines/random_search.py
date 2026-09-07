"""Full library random search baseline for one inverse design target.

Author:  Hooman Danesh <hooman.danesh@tu-braunschweig.de>, TU Braunschweig
License: MIT (see LICENSE). Released September 2026.
"""
from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np


OracleMetrics = Callable[
    [np.ndarray, np.ndarray, np.ndarray, Sequence[str], np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]


def run_for_target(
    target_index: int,
    target_vec: np.ndarray,
    n_candidates: int,
    targets: np.ndarray,
    row_labels: Sequence[str],
    loss_weights: np.ndarray,
    oracle_metrics: OracleMetrics,
    *,
    budget: int,
    eta: float,
    base_seed: int,
) -> dict[str, np.ndarray | int | bool]:
    """Evaluate one deterministic random sequence without replacement."""
    rng = np.random.default_rng(base_seed + int(target_index))
    indices = rng.choice(
        n_candidates, size=budget, replace=False,
    ).astype(np.int64)
    _, nmae = oracle_metrics(
        target_vec, indices, targets, row_labels, loss_weights,
    )
    feasible = np.flatnonzero(np.asarray(nmae) <= eta)
    threshold_met = bool(feasible.size)
    eval_count = int(feasible[0]) + 1 if threshold_met else budget + 1
    return {
        "indices": indices,
        "nmae": np.asarray(nmae, dtype=np.float64),
        "eval_count": eval_count,
        "threshold_met": threshold_met,
    }
