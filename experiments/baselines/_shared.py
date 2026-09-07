"""Shared configuration and metrics for full library search baselines.

Author:  Hooman Danesh <hooman.danesh@tu-braunschweig.de>, TU Braunschweig
License: MIT (see LICENSE). Released September 2026.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str((ROOT / "src").resolve()))

import inverse_design as inv  # noqa: E402
from active_learning import _lhs_select_indices  # noqa: E402


DATASET_PATH = ROOT / "data/oracle/oracle_45deg.npz"
PCA_PATH = ROOT / "data/pca/pc_scores.npz"
_SURROGATE_DIR = ROOT / "output/surrogate/active_learning"
CHECKPOINT_PATH = _SURROGATE_DIR / "al_model_selected.pt"
if not CHECKPOINT_PATH.exists():
    CHECKPOINT_PATH = _SURROGATE_DIR / "al_model_ntrain_200.pt"
TRAIN_CONFIG_PATH = ROOT / "output/surrogate/active_learning/al_config.json"
SWEEP_MANIFEST_PATH = ROOT / "output/inverse_design/sweep_settings.json"

OMEGA_P = {"P11": 1.0, "P22": 1.0, "P12": 1.0}   # omega_p, component weights
COMPONENTS = ("P11", "P22", "P12")
ETA = 5.0               # eta, acceptance tolerance on the weighted nMAE [%]
N_TARGETS = 1000        # N_tar, held-out targets
ORACLE_BUDGET = 50      # largest E_hit reported by the comparison
BO_INITIAL_POINTS = 200
BO_LHS_SEED = 271828
SETUP_SEED = 777
EI_XI = 0.01
GP_ALPHA = 1e-6
GP_LENGTH_SCALE_BOUNDS = (1e-2, 1e2)
GP_AMPLITUDE_BOUNDS = (1e-3, 1e3)
SAVE_EVERY = 10


def path_identity(path: Path) -> Dict[str, Any]:
    stat = path.resolve().stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.int64).ravel())
    return hashlib.sha256(array.tobytes()).hexdigest()


def config_signature(config: Dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_target_indices(
    ctx: inv.InverseDesignSetup,
    max_targets: int | None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    if not SWEEP_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            "The fixed inverse design target manifest is required: "
            f"{SWEEP_MANIFEST_PATH}"
        )
    manifest = json.loads(SWEEP_MANIFEST_PATH.read_text(encoding="utf-8"))
    all_indices = np.asarray(
        manifest.get("random_target_indices", []), dtype=np.int64,
    )
    if all_indices.shape != (N_TARGETS,):
        raise ValueError(
            f"Expected {N_TARGETS} target indices in the sweep manifest; "
            f"found shape {all_indices.shape}."
        )
    if all_indices.min() < 0 or all_indices.max() >= ctx.targets_full.shape[0]:
        raise ValueError("The sweep manifest contains out-of-range target indices.")
    indices = all_indices
    if max_targets is not None:
        if not 1 <= max_targets <= N_TARGETS:
            raise ValueError(f"max_targets must lie in [1, {N_TARGETS}].")
        indices = indices[:max_targets]
    return indices, all_indices, "output/inverse_design/sweep_settings.json"


def shared_lhs_initial_indices(
    x_all: np.ndarray,
    all_target_indices: np.ndarray,
) -> np.ndarray:
    """Select one deterministic LHS initialization disjoint from all targets."""
    available = np.setdiff1d(
        np.arange(x_all.shape[0], dtype=np.int64),
        np.asarray(all_target_indices, dtype=np.int64),
        assume_unique=False,
    )
    selected, _ = _lhs_select_indices(
        available,
        np.asarray(x_all, dtype=np.float64),
        BO_INITIAL_POINTS,
        np.random.default_rng(BO_LHS_SEED),
    )
    selected = np.asarray(selected, dtype=np.int64)
    if selected.shape != (BO_INITIAL_POINTS,):
        raise RuntimeError(
            f"Expected {BO_INITIAL_POINTS} LHS indices; found {selected.shape}."
        )
    if np.unique(selected).size != BO_INITIAL_POINTS:
        raise RuntimeError("The shared LHS initialization contains duplicates.")
    if np.intersect1d(selected, all_target_indices).size:
        raise RuntimeError("A fixed inverse design target leaked into the LHS set.")
    return selected


def weighted_nmae_for_pool(
    target_vec: np.ndarray,
    oracle_pool: np.ndarray,
    row_labels: Sequence[str],
) -> np.ndarray:
    labels = np.asarray(row_labels)
    component_values = []
    component_weights = []
    for component in COMPONENTS:
        mask = labels == component
        component_weight = float(OMEGA_P.get(component, 0.0))
        if not np.any(mask) or component_weight <= 0.0:
            continue
        scale = float(np.mean(np.abs(target_vec[mask])))
        if scale <= 1e-12:
            values = np.full(oracle_pool.shape[0], np.inf, dtype=np.float64)
        else:
            values = (
                100.0
                * np.mean(
                    np.abs(oracle_pool[:, mask] - target_vec[mask]), axis=1,
                )
                / scale
            )
        component_values.append(values)
        component_weights.append(component_weight)
    if not component_values:
        return np.full(oracle_pool.shape[0], np.nan, dtype=np.float64)
    return np.average(
        np.stack(component_values, axis=1),
        axis=1,
        weights=np.asarray(component_weights, dtype=np.float64),
    )


def oracle_metrics(
    target_vec: np.ndarray,
    candidate_indices: np.ndarray,
    targets: np.ndarray,
    row_labels: Sequence[str],
    loss_weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(candidate_indices, dtype=np.int64).ravel()
    oracle_pool = targets[indices]
    squared_mismatch = np.average(
        (oracle_pool - target_vec[np.newaxis, :]) ** 2,
        axis=1,
        weights=loss_weights,
    )
    nmae = weighted_nmae_for_pool(target_vec, oracle_pool, row_labels)
    return np.asarray(squared_mismatch), np.asarray(nmae)


def hit_rate_curve(eval_counts: np.ndarray) -> np.ndarray:
    return np.asarray([
        100.0 * np.mean(eval_counts <= step)
        for step in range(1, ORACLE_BUDGET + 1)
    ])


def best_nmae_percentiles(
    nmae_by_method: np.ndarray,
    eval_counts: np.ndarray,
    percentile_levels: np.ndarray,
) -> np.ndarray:
    """Aggregate best-so-far online nMAE, carrying early successes forward."""
    nmae_by_method = np.asarray(nmae_by_method, dtype=np.float64)
    eval_counts = np.asarray(eval_counts, dtype=np.int64)
    percentile_levels = np.asarray(percentile_levels, dtype=np.float64).ravel()
    if nmae_by_method.ndim != 3:
        raise ValueError("nmae_by_method must have shape (method, target, E).")
    n_methods, n_targets, n_steps = nmae_by_method.shape
    if n_steps != ORACLE_BUDGET:
        raise ValueError(
            f"Expected {ORACLE_BUDGET} online nMAE columns; found {n_steps}."
        )
    if eval_counts.shape != (n_methods, n_targets):
        raise ValueError(
            "eval_counts must match the method and target axes of nmae_by_method."
        )
    if percentile_levels.size == 0 or np.any(
        (percentile_levels < 0.0) | (percentile_levels > 100.0)
    ):
        raise ValueError("percentile_levels must lie in [0, 100].")

    summaries = np.empty(
        (n_methods, percentile_levels.size, ORACLE_BUDGET),
        dtype=np.float64,
    )
    for method_idx in range(n_methods):
        trajectories = np.empty(
            (n_targets, ORACLE_BUDGET), dtype=np.float64,
        )
        for target_idx in range(n_targets):
            count = int(eval_counts[method_idx, target_idx])
            n_evaluated = ORACLE_BUDGET if count > ORACLE_BUDGET else count
            if not 1 <= n_evaluated <= ORACLE_BUDGET:
                raise ValueError(
                    "Every target must have at least one online oracle evaluation."
                )
            values = nmae_by_method[method_idx, target_idx, :n_evaluated]
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    "An evaluated online nMAE sequence contains non-finite values."
                )
            best = np.minimum.accumulate(values)
            trajectories[target_idx, :n_evaluated] = best
            trajectories[target_idx, n_evaluated:] = best[-1]
        summaries[method_idx] = np.percentile(
            trajectories, percentile_levels, axis=0,
        )
    return summaries
