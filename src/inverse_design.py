"""Inverse design utilities for candidate screening and oracle validation.

Purpose:
    This module uses the trained surrogate to rank candidate structures for a
    target response, evaluates a shortlist with the oracle under a fixed
    budget, and returns the best validated design.

Main parameters:
    - DATASET_PATH, PCA_PATH, CHECKPOINT_PATH, TRAIN_CONFIG_PATH: required
      inputs for a reusable inverse design context.
    - ROTATIONS and INCLUDE_PATHS: loading-path subset used in the design
      problem.
    - MC_SAMPLES and MC_BATCH: Monte Carlo controls for uncertainty-aware
      ranking.
    - omega_p, eta, E_max, and components in design(): target weighting,
      acceptance threshold, oracle budget, and selected stress channels.

Author:  Hooman Danesh <hooman.danesh@tu-braunschweig.de>, TU Braunschweig
License: MIT (see LICENSE). Released September 2026.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from data_utils import (
    build_design_matrix,
    load_dataset,
    load_pca,
    select_fit_indices,
    stack_stress_targets,
)
from oracle import call_oracle
from gp_core import (
    CorrelatedLatentSVGP,
    build_svgp_from_config,
    normalize_z,
)

# === Default paths ===
DATASET_PATH = Path("data/oracle/oracle_45deg.npz")
PCA_PATH = Path("data/pca/pc_scores.npz")
_SELECTED_CHECKPOINT = Path("output/surrogate/active_learning/al_model_selected.pt")
_LEGACY_CHECKPOINT = Path("output/surrogate/active_learning/al_model_ntrain_200.pt")
CHECKPOINT_PATH = (
    _SELECTED_CHECKPOINT if _SELECTED_CHECKPOINT.exists() else _LEGACY_CHECKPOINT
)
TRAIN_CONFIG_PATH = Path("output/surrogate/active_learning/al_config.json")

SEED = 777
ROTATIONS = (45,)
INCLUDE_PATHS = ("tension_x", "tension_y", "equibiaxial", "off_x", "off_y")
MC_SAMPLES = 64   # S, Monte Carlo samples for the predictive moments
MC_BATCH = 100


@dataclass
class InverseDesignSetup:
    """Precomputed surrogate + oracle data. Created by setup(), passed to design()."""
    model: CorrelatedLatentSVGP
    z_norm: torch.Tensor
    gp_mean_params: np.ndarray
    targets_full: np.ndarray
    row_labels_full: List[str]
    design_full: np.ndarray
    train_idx: np.ndarray
    tags_subset: np.ndarray
    F_grid_subset: np.ndarray
    fit_idx: np.ndarray
    include_shear: bool
    shear_rotations: Tuple[int, ...]
    device: torch.device


@dataclass
class PreparedDesignProblem:
    """Combo-specific arrays and tensors reused across many target solves."""
    row_mask: np.ndarray
    row_labels: Tuple[str, ...]
    active_comps: Tuple[str, ...]
    pred_np: np.ndarray
    design_t: torch.Tensor


@dataclass
class InverseDesignResult:
    """Result of one inverse design call."""
    best_idx: int
    oracle_stress: np.ndarray
    E_used: int
    threshold_met: bool
    eval_indices: np.ndarray
    oracle_mismatches: np.ndarray
    nmae_by_component: Dict[str, float]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.benchmark = False


def _normalize_component_names(names: Sequence[str]) -> List[str]:
    """Normalize component names to P11, P22, and P12."""
    normalized = []
    for name in names:
        key = name.strip().upper()
        if key in ("P11", "PXX"):
            normalized.append("P11")
        elif key in ("P22", "PYY"):
            normalized.append("P22")
        elif key in ("P12", "P21", "PXY"):
            normalized.append("P12")
        else:
            raise ValueError(f"Unknown stress component: {name}")
    return list(dict.fromkeys(normalized))


def _component_mask(row_labels: Sequence[str], components: Sequence[str]) -> List[int]:
    wanted = set(_normalize_component_names(components))
    mask = [i for i, lbl in enumerate(row_labels) if lbl in wanted]
    if not mask:
        raise ValueError(f"No rows match requested components {components}")
    return mask


def _auto_shear_rotations(rotations: Sequence[int]) -> Tuple[int, ...]:
    return tuple(r for r in rotations if r != 0)


def _load_training_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_train_indices(
    checkpoint_path: Path, train_config_path: Path | None, expected_size: int | None = None,
) -> np.ndarray:
    def _matches(idx: np.ndarray) -> bool:
        return expected_size is None or int(idx.size) == int(expected_size)

    def _from_history(hist: np.ndarray) -> np.ndarray | None:
        if expected_size is None:
            return np.asarray(hist[-1], dtype=np.int64) if len(hist) else None
        for item in reversed(hist):
            idx = np.asarray(item, dtype=np.int64)
            if _matches(idx):
                return idx
        return None

    ckpt_dir = checkpoint_path.parent
    # New active learning runs keep one compact state file alongside the
    # selected checkpoint. It contains the exact labeled set used by that
    # checkpoint, so no per-iteration index files are required.
    if checkpoint_path.name == "al_model_selected.pt":
        state_path = ckpt_dir / "al_state.npz"
        if state_path.exists():
            data = np.load(state_path, allow_pickle=True)
            for key in ("selected_train_idx", "final_train_idx", "current_train_idx"):
                if key in data:
                    idx = np.asarray(data[key], dtype=np.int64).ravel()
                    if idx.size and _matches(idx):
                        return idx

    ntrain_match = re.search(r"al_model_ntrain_(\d+)\.pt$", checkpoint_path.name)
    if ntrain_match:
        n_train = int(ntrain_match.group(1))
        per_train = ckpt_dir / f"train_idx_ntrain_{n_train}.npy"
        if per_train.exists():
            idx = np.asarray(np.load(per_train), dtype=np.int64)
            if _matches(idx):
                return idx
        for history in (
            ckpt_dir / f"al_history_ntrain_{n_train}.npz",
            ckpt_dir / f"al_history_ntrain_{n_train:04d}.npz",
        ):
            if history.exists():
                data = np.load(history, allow_pickle=True)
                if "final_train_idx" in data:
                    idx = np.asarray(data["final_train_idx"], dtype=np.int64)
                    if _matches(idx):
                        return idx
                for key in ("train_idx_hist", "train_idx_history"):
                    if key in data:
                        idx = _from_history(data[key])
                        if idx is not None:
                            return idx

    match = re.search(r"al_model_iter_(\d+)\.pt$", checkpoint_path.name)
    if match:
        iter_id = int(match.group(1))
        per_iter = ckpt_dir / f"train_idx_iter_{iter_id:04d}.npy"
        if per_iter.exists():
            idx = np.asarray(np.load(per_iter), dtype=np.int64)
            if _matches(idx):
                return idx
        history = ckpt_dir / f"al_history_iter_{iter_id:04d}.npz"
        if history.exists():
            data = np.load(history, allow_pickle=True)
            for key in ("train_idx_hist", "train_idx_history"):
                if key in data:
                    idx = _from_history(data[key])
                    if idx is not None:
                        return idx
            if "final_train_idx" in data:
                idx = np.asarray(data["final_train_idx"], dtype=np.int64)
                if _matches(idx):
                    return idx
                if expected_size is not None and idx.size > expected_size:
                    return idx[:expected_size]
        splits = ckpt_dir / "al_splits.npz"
        if splits.exists():
            data = np.load(splits, allow_pickle=True)
            if "train_idx_history" in data:
                idx = _from_history(data["train_idx_history"])
                if idx is not None:
                    return idx
            if "initial_train_idx" in data and "added_train_idx" in data:
                initial = np.asarray(data["initial_train_idx"], dtype=np.int64)
                added = np.asarray(data["added_train_idx"], dtype=np.int64)
                if expected_size is not None:
                    n_added = expected_size - initial.size
                    if 0 <= n_added <= added.size:
                        return np.concatenate([initial, added[:n_added]])
                elif added.size >= iter_id:
                    return np.concatenate([initial, added[:iter_id]])
    latest = ckpt_dir / "train_idx.npy"
    if latest.exists():
        idx = np.asarray(np.load(latest), dtype=np.int64)
        if _matches(idx):
            return idx
    if train_config_path is not None:
        fallback = train_config_path.parent / "train_idx.npy"
        if fallback.exists():
            idx = np.asarray(np.load(fallback), dtype=np.int64)
            if _matches(idx):
                return idx
    return np.array([], dtype=np.int64)


def _infer_latent_rank_and_tasks(state_dict: dict) -> Tuple[int, int]:
    coeff = state_dict.get("variational_strategy.lmc_coefficients")
    if coeff is None:
        raise ValueError("lmc_coefficients not found in checkpoint")
    if coeff.ndim != 2:
        raise ValueError(f"Unexpected lmc_coefficients shape {coeff.shape}")
    num_latents, num_tasks = coeff.shape
    rank = int(num_latents - num_tasks)
    return rank, int(num_tasks)


def _build_loss_weights(
    row_labels: List[str], omega_p: Dict[str, float],
) -> np.ndarray:
    """Per-row component weights omega_p for the weighted aggregate nMAE."""
    weights = np.ones(len(row_labels), dtype=np.float64)
    for i, lbl in enumerate(row_labels):
        weights[i] = float(omega_p.get(lbl, 1.0))
    return weights


def _target_norm_weights(
    target_vec: np.ndarray,
    row_labels: List[str],
    base_weights: np.ndarray,
) -> np.ndarray:
    """Scale weights by 1/mean(|target|)^2 per component (MSE target-mean normalization)."""
    eps = 1e-12
    scales = np.ones(len(row_labels), dtype=np.float64)
    for comp in ("P11", "P22", "P12"):
        idxs = [i for i, lbl in enumerate(row_labels) if lbl == comp]
        if not idxs:
            continue
        mean_abs = float(np.mean(np.abs(target_vec[idxs])))
        scale = mean_abs if mean_abs > eps else 1.0
        scales[idxs] = scale
    return base_weights / (scales * scales)


def _component_nmae(
    target_vec: np.ndarray,
    oracle_vec: np.ndarray,
    row_labels: List[str],
) -> Dict[str, float]:
    """Per-component normalized MAE (nMAE_p)."""
    eps = 1e-12
    nmae: Dict[str, float] = {}
    for comp in ("P11", "P22", "P12"):
        idxs = [i for i, lbl in enumerate(row_labels) if lbl == comp]
        if not idxs:
            continue
        target_vals = target_vec[idxs]
        oracle_vals = oracle_vec[idxs]
        mae = float(np.mean(np.abs(target_vals - oracle_vals)))
        mean_target = float(np.mean(np.abs(target_vals)))
        nmae[comp] = float("inf") if mean_target <= eps else 100.0 * mae / mean_target
    return nmae


def _weighted_nmae(
    nmae_map: Dict[str, float],
    omega_p: Dict[str, float],
    active_comps: List[str],
) -> float:
    """Weighted aggregate nMAE (nMAE_bar)."""
    total = 0.0
    denom = 0.0
    for comp in active_comps:
        if comp not in nmae_map:
            continue
        w = float(omega_p.get(comp, 0.0))
        if w <= 0.0:
            continue
        total += w * float(nmae_map[comp])
        denom += w
    return float("nan") if denom <= 0.0 else total / denom


def candidate_score_terms(
    model: CorrelatedLatentSVGP,
    z_norm: torch.Tensor,
    design_t: torch.Tensor,
    target_vec: np.ndarray,
    loss_weights: np.ndarray,
    idxs: np.ndarray,
    mc_samples: int,
    mc_batch: int,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Return the predictive mismatch terms used for candidate ranking.

    The returned tensors contain the Monte Carlo mean and standard deviation
    of each candidate's weighted squared mismatch. The scalar ``uncertainty_weight``
    is the automatically calibrated value

        mean(mismatch_mean) / mean(mismatch_std).

    Keeping these terms separate allows sensitivity analyses to vary the
    uncertainty multiplier without drawing a different Monte Carlo sample
    for every value.
    """
    if idxs.size == 0:
        empty = torch.empty(0, dtype=z_norm.dtype, device=z_norm.device)
        return empty, empty, 0.0
    weights_sum = float(loss_weights.sum())
    if weights_sum <= 0.0:
        weights_sum = float(loss_weights.size)
    loss_weights_t = torch.from_numpy(
        loss_weights.reshape(1, -1).astype(np.float32),
    ).to(z_norm.device)
    target_t = torch.from_numpy(target_vec.astype(np.float32)).to(z_norm.device)
    n = idxs.shape[0]
    mismatch_mean = torch.empty(n, device=z_norm.device)
    mismatch_var = torch.empty(n, device=z_norm.device)
    with torch.no_grad():
        for start in range(0, n, mc_batch):
            stop = min(start + mc_batch, n)
            idx_slice = idxs[start:stop]
            out = model(z_norm[idx_slice])
            samples = out.rsample(torch.Size([mc_samples]))
            params_samples = F.softplus(samples) + 1e-6
            pred_samples = torch.matmul(params_samples, design_t)
            diff = pred_samples - target_t
            errs = (diff * diff * loss_weights_t).sum(dim=2) / weights_sum
            mismatch_mean[start:stop] = errs.mean(dim=0)
            mismatch_var[start:stop] = errs.var(dim=0, unbiased=False)
    mismatch_std = torch.sqrt(mismatch_var.clamp_min(0.0) + 1e-12)
    uncertainty_weight = float(
        torch.mean(mismatch_mean).abs().item()
        / torch.mean(mismatch_std).abs().clamp_min(1e-12).item()
    )
    return mismatch_mean, mismatch_std, uncertainty_weight


def _candidate_scores(
    model: CorrelatedLatentSVGP,
    z_norm: torch.Tensor,
    design_t: torch.Tensor,
    target_vec: np.ndarray,
    loss_weights: np.ndarray,
    idxs: np.ndarray,
    mc_samples: int,
    mc_batch: int,
) -> np.ndarray:
    """Score candidates for oracle ranking.

    Combines expected surrogate mismatch and predictive uncertainty:
        score_i = mean(mismatch_i) + uncertainty_weight * std(mismatch_i)
    where uncertainty_weight is auto-calibrated as mean(means) / mean(stds).
    """
    mismatch_mean, mismatch_std, uncertainty_weight = candidate_score_terms(
        model=model,
        z_norm=z_norm,
        design_t=design_t,
        target_vec=target_vec,
        loss_weights=loss_weights,
        idxs=idxs,
        mc_samples=mc_samples,
        mc_batch=mc_batch,
    )
    scores = mismatch_mean + uncertainty_weight * mismatch_std
    return scores.cpu().numpy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare_design_problem(
    ctx: InverseDesignSetup,
    omega_p: Dict[str, float],
    *,
    components: Tuple[str, ...] = ("P11", "P22", "P12"),
    pred_np: np.ndarray | None = None,
) -> PreparedDesignProblem:
    """Build combo-specific state once and reuse it across many targets."""
    row_mask = np.array(
        _component_mask(ctx.row_labels_full, components), dtype=np.int64,
    )
    row_labels = tuple(ctx.row_labels_full[i] for i in row_mask)
    active_comps = tuple(
        comp for comp in ("P11", "P22", "P12")
        if comp in row_labels and omega_p.get(comp, 0.0) > 0.0
    )

    design_mat = np.asarray(ctx.design_full[row_mask], dtype=np.float32)
    if pred_np is None:
        pred_np = ctx.gp_mean_params @ design_mat.T
    else:
        pred_np = np.asarray(pred_np)
        n_s, n_r = int(ctx.targets_full.shape[0]), int(design_mat.shape[0])
        if pred_np.shape != (n_s, n_r):
            raise ValueError(
                f"pred_np shape {pred_np.shape} != ({n_s}, {n_r}) for components {components}"
            )

    design_t = torch.from_numpy(design_mat).to(ctx.device).t().contiguous()
    return PreparedDesignProblem(
        row_mask=row_mask,
        row_labels=row_labels,
        active_comps=active_comps,
        pred_np=np.asarray(pred_np),
        design_t=design_t,
    )


def setup(
    dataset_path: Path = DATASET_PATH,
    pca_path: Path = PCA_PATH,
    checkpoint_path: Path = CHECKPOINT_PATH,
    train_config_path: Path = TRAIN_CONFIG_PATH,
    *,
    rotations: Tuple[int, ...] = ROTATIONS,
    include_paths: Tuple[str, ...] = INCLUDE_PATHS,
    seed: int = SEED,
    device: str = "cpu",
) -> InverseDesignSetup:
    """Load GP surrogate, dataset, PCA. Call once, pass result to design()."""
    _set_deterministic(seed)
    dev = torch.device(device)

    train_cfg = _load_training_config(train_config_path)

    shear_rots = _auto_shear_rotations(rotations)
    include_shear = bool(shear_rots)

    data = load_dataset(dataset_path)
    tags = data["deformation_tags"]
    F_grid = data["deformation_grid"]
    stresses = data["stresses"]

    fit_idx, _ = select_fit_indices(
        tags, F_grid, rotations, include_paths=include_paths,
    )
    design_full, row_labels_full = build_design_matrix(
        F_grid[fit_idx], tags[fit_idx], include_shear, shear_rots,
    )
    targets_full = stack_stress_targets(
        stresses[:, fit_idx], tags[fit_idx], include_shear, shear_rots,
    )

    n_pca = train_cfg.get("n_pca_components")
    if n_pca is None:
        pcs_all = load_pca(pca_path, n_components=None)
        n_pca = pcs_all.shape[1]
    pcs = load_pca(pca_path, int(n_pca))
    if pcs.shape[0] != targets_full.shape[0]:
        raise ValueError(
            f"PC score rows ({pcs.shape[0]}) != oracle rows ({targets_full.shape[0]})"
        )

    z_raw = torch.from_numpy(pcs.astype(np.float32)).to(dev)
    z_mean = z_raw.mean(0, keepdim=True)
    z_std = z_raw.std(0, keepdim=True).clamp_min(1e-8)
    z_norm = normalize_z(z_raw, z_mean, z_std)

    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    state_dict = ckpt["model_state"]
    inducing_shape = state_dict[
        "variational_strategy.base_variational_strategy.inducing_points"
    ].shape
    n_inducing = int(inducing_shape[-2])
    train_idx = _resolve_train_indices(
        checkpoint_path, train_config_path, expected_size=n_inducing,
    )
    if train_idx.size == 0:
        raise FileNotFoundError(
            f"No train_idx with {n_inducing} entries found for {checkpoint_path.name}."
        )
    if train_idx.size != n_inducing:
        raise ValueError(
            f"Resolved train_idx has {train_idx.size} entries, but "
            f"{checkpoint_path.name} has {n_inducing} inducing points. "
            "Use al_state.npz for the selected checkpoint or the matching "
            "legacy active learning history/index file."
        )
    inducing_points = torch.zeros(inducing_shape, device=dev)
    rank, num_tasks = _infer_latent_rank_and_tasks(state_dict)
    model = build_svgp_from_config(
        inducing_points, num_tasks, rank, train_cfg,
    ).to(dev)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        output = model(z_norm)
        params_pos = F.softplus(output.mean) + 1e-6
        gp_mean_params = params_pos.cpu().numpy()

    print(
        f"[setup] structures={targets_full.shape[0]}  "
        f"rows={len(row_labels_full)}  "
        f"train={train_idx.size}  "
        f"pca_dim={pcs.shape[1]}  "
        f"checkpoint={checkpoint_path.name}"
    )

    return InverseDesignSetup(
        model=model,
        z_norm=z_norm,
        gp_mean_params=np.asarray(gp_mean_params, dtype=np.float32),
        targets_full=targets_full,
        row_labels_full=row_labels_full,
        design_full=design_full,
        train_idx=train_idx,
        tags_subset=tags[fit_idx],
        F_grid_subset=F_grid[fit_idx],
        fit_idx=fit_idx,
        include_shear=include_shear,
        shear_rotations=shear_rots,
        device=dev,
    )


def design(
    target_vec: np.ndarray,
    omega_p: Dict[str, float],
    eta: float,
    E_max: int,
    ctx: InverseDesignSetup,
    *,
    components: Tuple[str, ...] = ("P11", "P22", "P12"),
    pred_np: np.ndarray | None = None,
    prepared: PreparedDesignProblem | None = None,
    mc_samples: int = MC_SAMPLES,
    mc_batch: int = MC_BATCH,
) -> InverseDesignResult:
    """Run Bayesian-guided inverse design for one target.

    Given target stress P*, component weights omega_p, error threshold eta,
    and oracle budget E_max, returns the optimal microstructure M*.

    If ``pred_np`` is provided (shape ``(n_structures, n_active_rows)``), it must
    be the surrogate mean stress for the same ``components``; pass it when
    evaluating many targets for one combo to avoid recomputing
    ``gp_mean_params @ Q^T`` every call.
    """
    target_vec = np.asarray(target_vec)
    if prepared is None:
        prepared = prepare_design_problem(
            ctx, omega_p, components=components, pred_np=pred_np,
        )
    elif pred_np is not None:
        pred_np = np.asarray(pred_np)
        if pred_np.shape != prepared.pred_np.shape or not np.array_equal(
            pred_np, prepared.pred_np,
        ):
            raise ValueError(
                "When 'prepared' is supplied, 'pred_np' must be omitted or match "
                "prepared.pred_np exactly."
            )

    row_mask = prepared.row_mask
    row_labels = list(prepared.row_labels)
    active_comps = list(prepared.active_comps)
    pred_np = prepared.pred_np
    design_t = prepared.design_t

    base_weights = _build_loss_weights(row_labels, omega_p)
    loss_weights = _target_norm_weights(target_vec, row_labels, base_weights)

    # Surrogate-predicted mismatch for all candidates
    surrogate_mismatch = np.average(
        (pred_np - target_vec[np.newaxis, :]) ** 2,
        axis=1,
        weights=loss_weights,
    )

    # Pre-filter to E_max best candidates by surrogate mismatch.
    #
    # np.argpartition guarantees the partition but not the order within the
    # selected block, and _candidate_scores draws the Monte Carlo samples for
    # the whole pool in one batch, assigning them by position rather than by
    # candidate. Sorting by structure index pins that assignment to a canonical
    # order, so an equivalent pool arriving in a different order cannot change
    # the scores, the ranking, or the number of oracle calls spent.
    pool_size = min(E_max, pred_np.shape[0])
    candidate_pool = np.sort(
        np.argpartition(surrogate_mismatch, pool_size - 1)[:pool_size],
    )

    # Score candidates by uncertainty-aware acquisition (phi_i)
    candidate_scores = _candidate_scores(
        model=ctx.model,
        z_norm=ctx.z_norm,
        design_t=design_t,
        target_vec=target_vec,
        loss_weights=loss_weights,
        idxs=candidate_pool,
        mc_samples=mc_samples,
        mc_batch=mc_batch,
    )

    order = np.argsort(candidate_scores)
    ranked_candidates = candidate_pool[order]

    def _oracle_mismatch(oracle_vec: np.ndarray) -> float:
        return float(
            np.average(
                (oracle_vec - target_vec) ** 2, weights=loss_weights,
            )
        )

    explored_indices: List[int] = []
    oracle_mismatches: List[float] = []
    best_mismatch: float | None = None
    best_idx: int | None = None
    threshold_met = False

    for ptr in range(min(E_max, ranked_candidates.size)):
        candidate_idx = int(ranked_candidates[ptr])
        oracle_vec = np.asarray(call_oracle(ctx.targets_full, candidate_idx))[row_mask]
        mismatch = _oracle_mismatch(oracle_vec)
        explored_indices.append(candidate_idx)
        oracle_mismatches.append(mismatch)

        if best_mismatch is None or mismatch < best_mismatch:
            best_mismatch = mismatch
            best_idx = candidate_idx

        nmae_map = _component_nmae(target_vec, oracle_vec, row_labels)
        nmae_bar = _weighted_nmae(nmae_map, omega_p, active_comps)
        if np.isfinite(nmae_bar) and nmae_bar <= eta:
            threshold_met = True
            break

    if best_idx is None:
        best_idx = int(np.argmin(surrogate_mismatch))

    best_oracle = np.asarray(call_oracle(ctx.targets_full, best_idx))[row_mask]
    nmae_final = _component_nmae(target_vec, best_oracle, row_labels)

    return InverseDesignResult(
        best_idx=best_idx,
        oracle_stress=best_oracle,
        E_used=len(explored_indices),
        threshold_met=threshold_met,
        eval_indices=np.array(explored_indices, dtype=np.int64),
        oracle_mismatches=np.array(oracle_mismatches, dtype=np.float64),
        nmae_by_component=nmae_final,
    )
