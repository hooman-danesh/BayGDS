#!/usr/bin/env python3
"""Run the inverse design sweep across targets, budgets, and components.

Purpose:
    This script evaluates many target responses, ranks candidate structures
    with the trained surrogate, spends a limited oracle budget on shortlisted
    candidates, and stores cached results for later analysis and plotting.

Main parameters:
    - DATASET_PATH: oracle response database used for validation.
    - PCA_PATH: PC score file used as GP input.
    - CHECKPOINT_PATH: trained active learning surrogate checkpoint.
    - TRAIN_CONFIG_PATH: active learning training configuration.
    - OMEGA_P: relative weights of the target stress components.
    - ETA: acceptance threshold for the weighted normalized error.
    - N_TARGETS: number of target responses sampled from the design pool.
    - SEED: reproducible target sampling seed.
    - COMPONENT_COMBOS: stress-component subsets to be studied.
    - EVAL_STEPS: oracle budgets used in the sweep.
    - RUN_DIR: cache location for the stored inverse design results.

Outputs:
    Produces cached .npz result files and a sweep manifest for downstream
    plotting and analysis.

Usage:
    python -m experiments.baygds.run_inverse_design

Author:  Hooman Danesh <hooman.danesh@tu-braunschweig.de>, TU Braunschweig
License: MIT (see LICENSE). Released September 2026.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str((ROOT / "src").resolve()))

import numpy as np
import inverse_design as inv

# ---------------------------------------------------------------------------
# Control parameters and paths
# ---------------------------------------------------------------------------
DATASET_PATH = ROOT / "data/oracle/oracle_45deg.npz"
PCA_PATH = ROOT / "data/pca/pc_scores.npz"
# Active learning retains the checkpoint selected by its stopping criterion.
_SURROGATE_DIR = ROOT / "output/surrogate/active_learning"
CHECKPOINT_PATH = _SURROGATE_DIR / "al_model_selected.pt"
if not CHECKPOINT_PATH.exists():
    CHECKPOINT_PATH = _SURROGATE_DIR / "al_model_ntrain_200.pt"
TRAIN_CONFIG_PATH = ROOT / "output/surrogate/active_learning/al_config.json"
RUN_DIR = ROOT / "output/inverse_design"

OMEGA_P = {"P11": 1.0, "P22": 1.0, "P12": 1.0}   # omega_p, component weights
ETA = 5.0               # eta, acceptance tolerance on the weighted nMAE [%]
N_TARGETS = 1000        # N_tar, held-out targets
SEED = 42

COMPONENT_COMBOS = [
    ("P11",), ("P22",), ("P12",),
    ("P11", "P22"),
    ("P11", "P12"),
    ("P22", "P12"),
    ("P11", "P22", "P12"),
]

EVAL_STEPS = [1] + list(range(10, 210, 10))      # E_hit, reported oracle budgets

MC_SAMPLES = 64         # S, Monte Carlo samples for the predictive moments
MC_BATCH = 100          # candidates scored per Monte Carlo batch
SETUP_SEED = 777        # seed for the candidate-set construction

# Bump if fingerprint fields change (forces one-time recompute for old caches)

# Keep inverse_design module globals aligned because cache fingerprints and
# manifests include them.
inv.DATASET_PATH = DATASET_PATH
inv.PCA_PATH = PCA_PATH
inv.CHECKPOINT_PATH = CHECKPOINT_PATH
inv.TRAIN_CONFIG_PATH = TRAIN_CONFIG_PATH
# design() and setup() bind these as default arguments, so they are also
# passed explicitly at the call sites below; assigning them here keeps the
# recorded sweep manifest in step with the values actually used.
inv.MC_SAMPLES = MC_SAMPLES
inv.MC_BATCH = MC_BATCH
inv.SEED = SETUP_SEED


def _path_stat(path: Path) -> Tuple[int, int] | None:
    """Return (mtime_ns, size) for cache invalidation when files are replaced."""
    try:
        st = path.resolve().stat()
        return (int(st.st_mtime_ns), int(st.st_size))
    except OSError:
        return None


def _train_idx_digest(train_idx: np.ndarray) -> str:
    """Stable id for the training index set used in setup()."""
    arr = np.ascontiguousarray(np.asarray(train_idx, dtype=np.int64).ravel())
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _fingerprint_dict(
    ctx: inv.InverseDesignSetup,
    *,
    target_sample_seed: int,
    target_indices: np.ndarray,
    omega_p: Dict[str, float],
    eta: float,
    e_max: int,
    components: Tuple[str, ...],
) -> Dict[str, Any]:
    """All settings that affect inverse_design outputs for this eval folder."""
    dataset = Path(inv.DATASET_PATH).resolve()
    pca = Path(inv.PCA_PATH).resolve()
    ckpt = Path(inv.CHECKPOINT_PATH).resolve()
    tcfg = Path(inv.TRAIN_CONFIG_PATH).resolve()
    ck_stat = _path_stat(ckpt)
    tc_stat = _path_stat(tcfg)
    return {
        "target_sample_seed": int(target_sample_seed),
        "n_targets": int(target_indices.size),
        "eta": float(eta),
        "omega_p": {k: float(omega_p.get(k, 1.0)) for k in ("P11", "P22", "P12")},
        "e_max": int(e_max),
        "components": list(components),
        "dataset_path": str(dataset),
        "pca_path": str(pca),
        "checkpoint_path": str(ckpt),
        "checkpoint_mtime_ns": ck_stat[0] if ck_stat else -1,
        "checkpoint_size": ck_stat[1] if ck_stat else -1,
        "train_config_path": str(tcfg),
        "train_config_mtime_ns": tc_stat[0] if tc_stat else -1,
        "train_config_size": tc_stat[1] if tc_stat else -1,
        "setup_seed": int(inv.SEED),
        "rotations": list(inv.ROTATIONS),
        "include_paths": list(inv.INCLUDE_PATHS),
        "mc_samples": int(inv.MC_SAMPLES),
        "mc_batch": int(inv.MC_BATCH),
        "train_idx_digest": _train_idx_digest(ctx.train_idx),
        "n_structures": int(ctx.targets_full.shape[0]),
        "notation": {
            "n_targets": "N_tar, held-out targets",
            "eta": "eta, acceptance tolerance [%]",
            "omega_p": "omega_p, component weights",
            "eval_steps": "E_hit, reported oracle budgets",
            "mc_samples": "S, Monte Carlo samples",
            "pca_dim": "n_z, descriptor dimension",
        },
        "n_stress_rows": int(len(ctx.row_labels_full)),
        "pca_dim": int(ctx.z_norm.shape[1]),
    }


def _fingerprints_equal(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Stable comparison (avoids JSON float formatting drift)."""
    return json.dumps(a, sort_keys=True, separators=(",", ":")) == json.dumps(
        b, sort_keys=True, separators=(",", ":"),
    )


def _cache_is_valid(
    out_path: Path,
    ctx: inv.InverseDesignSetup,
    target_indices: np.ndarray,
    omega_p: Dict[str, float],
    eta: float,
    e_max: int,
    components: Tuple[str, ...],
    target_sample_seed: int,
) -> bool:
    if not out_path.is_file():
        return False
    try:
        data = np.load(out_path, allow_pickle=True)
    except Exception:
        return False

    stored_raw = data.get("cache_fingerprint_json")
    if stored_raw is None:
        return False
    try:
        s = stored_raw.item() if isinstance(stored_raw, np.ndarray) else stored_raw
        if isinstance(s, bytes):
            s = s.decode("utf-8")
        stored_fp = json.loads(str(s))
    except Exception:
        return False

    current_fp = _fingerprint_dict(
        ctx,
        target_sample_seed=target_sample_seed,
        target_indices=target_indices,
        omega_p=omega_p,
        eta=eta,
        e_max=e_max,
        components=components,
    )
    if not _fingerprints_equal(stored_fp, current_fp):
        return False

    try:
        prev_idx = np.asarray(data["random_target_indices"], dtype=np.int64)
    except Exception:
        return False
    if prev_idx.shape != target_indices.shape:
        return False
    if not np.array_equal(prev_idx, target_indices):
        return False

    return True


def _sweep_manifest(
    ctx: inv.InverseDesignSetup,
    target_indices: np.ndarray,
    target_sample_seed: int,
) -> Dict[str, Any]:
    """Settings shared across all combo / E_max caches (plus target list for the notebook)."""
    dataset = Path(inv.DATASET_PATH).resolve()
    pca = Path(inv.PCA_PATH).resolve()
    ckpt = Path(inv.CHECKPOINT_PATH).resolve()
    tcfg = Path(inv.TRAIN_CONFIG_PATH).resolve()
    ck_stat = _path_stat(ckpt)
    tc_stat = _path_stat(tcfg)
    return {
        "target_sample_seed": int(target_sample_seed),
        "n_targets": int(target_indices.size),
        "eta": float(ETA),
        "omega_p": {k: float(OMEGA_P.get(k, 1.0)) for k in ("P11", "P22", "P12")},
        "dataset_path": str(dataset),
        "pca_path": str(pca),
        "checkpoint_path": str(ckpt),
        "checkpoint_mtime_ns": ck_stat[0] if ck_stat else -1,
        "checkpoint_size": ck_stat[1] if ck_stat else -1,
        "train_config_path": str(tcfg),
        "train_config_mtime_ns": tc_stat[0] if tc_stat else -1,
        "train_config_size": tc_stat[1] if tc_stat else -1,
        "setup_seed": int(inv.SEED),
        "rotations": list(inv.ROTATIONS),
        "include_paths": list(inv.INCLUDE_PATHS),
        "mc_samples": int(inv.MC_SAMPLES),
        "mc_batch": int(inv.MC_BATCH),
        "train_idx_digest": _train_idx_digest(ctx.train_idx),
        "n_structures": int(ctx.targets_full.shape[0]),
        "n_stress_rows": int(len(ctx.row_labels_full)),
        "pca_dim": int(ctx.z_norm.shape[1]),
        "eval_steps": [int(x) for x in EVAL_STEPS],
        "component_combos": [list(c) for c in COMPONENT_COMBOS],
        "random_target_indices": target_indices.astype(np.int64).tolist(),
    }


def _write_sweep_settings(
    ctx: inv.InverseDesignSetup,
    target_indices: np.ndarray,
    target_sample_seed: int,
) -> None:
    """Shared settings for all eval_* folders; notebook can compare to its expectations."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _sweep_manifest(ctx, target_indices, target_sample_seed)
    out = RUN_DIR / "sweep_settings.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Batch runner with disk caching
# ---------------------------------------------------------------------------
def run_batch(
    ctx: inv.InverseDesignSetup,
    target_indices: np.ndarray,
    omega_p: Dict[str, float],
    eta: float,
    E_max: int,
    components: Tuple[str, ...],
    run_dir: Path,
    *,
    target_sample_seed: int,
) -> bool:
    """Run inverse design for a batch of targets; cache results to disk.

    Returns True if an existing cache was reused, False if freshly computed.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "inverse_design.npz"
    if _cache_is_valid(
        out_path, ctx, target_indices, omega_p, eta, E_max, components,
        target_sample_seed,
    ):
        return True

    prepared = inv.prepare_design_problem(ctx, omega_p, components=components)
    row_mask = prepared.row_mask
    row_labels = list(prepared.row_labels)

    n_targets = int(target_indices.size)
    best_indices = np.empty(n_targets, dtype=np.int64)
    eval_counts = np.empty(n_targets, dtype=np.int64)
    threshold_met = np.empty(n_targets, dtype=bool)
    eval_indices_list: list[np.ndarray] = []
    oracle_mismatch_list: list[np.ndarray] = []

    for i, idx in enumerate(target_indices):
        result = inv.design(
            target_vec=ctx.targets_full[int(idx), row_mask],
            omega_p=omega_p,
            eta=eta,
            E_max=E_max,
            ctx=ctx,
            components=components,
            prepared=prepared,
            mc_samples=MC_SAMPLES,
            mc_batch=MC_BATCH,
        )
        best_indices[i] = result.best_idx
        eval_counts[i] = result.E_used
        threshold_met[i] = result.threshold_met
        eval_indices_list.append(result.eval_indices)
        oracle_mismatch_list.append(result.oracle_mismatches)

    fp = _fingerprint_dict(
        ctx,
        target_sample_seed=target_sample_seed,
        target_indices=target_indices,
        omega_p=omega_p,
        eta=eta,
        e_max=E_max,
        components=components,
    )
    fp_json = json.dumps(fp, sort_keys=True, separators=(",", ":"))

    data = {
        "random_target_indices": target_indices.astype(np.int64),
        "random_best_indices": best_indices,
        "random_oracle_eval_counts": eval_counts,
        "random_oracle_eval_threshold_met": threshold_met,
        "random_oracle_eval_indices": np.array(eval_indices_list, dtype=object),
        "random_oracle_eval_mismatches": np.array(oracle_mismatch_list, dtype=object),
        "row_labels": np.array(row_labels),
        "components": np.array(components),
        "oracle_stop_mode": np.array(["weighted_mean"]),
        "oracle_stop_weight_p11": np.array([omega_p.get("P11", 1.0)]),
        "oracle_stop_weight_p22": np.array([omega_p.get("P22", 1.0)]),
        "oracle_stop_weight_p12": np.array([omega_p.get("P12", 1.0)]),
        "run_eta": np.array([float(eta)], dtype=np.float64),
        "run_e_max": np.array([int(E_max)], dtype=np.int64),
        "run_sample_seed": np.array([int(target_sample_seed)], dtype=np.int64),
        "cache_fingerprint_json": np.array(fp_json, dtype=object),
    }
    np.savez_compressed(out_path, **data)
    return False


def hit_rate(path: Path) -> float:
    """Compute hit rate (%) from a cached .npz file."""
    data = np.load(path, allow_pickle=True)
    met = data.get("random_oracle_eval_threshold_met",
                   data.get("random_oracle_eval_success", np.array([])))
    met = np.asarray(met, dtype=bool)
    if met.size == 0:
        return float("nan")
    return float(np.mean(met) * 100.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ctx = inv.setup(
        dataset_path=DATASET_PATH,
        pca_path=PCA_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        train_config_path=TRAIN_CONFIG_PATH,
        seed=SETUP_SEED,
    )

    rng = np.random.default_rng(SEED)
    candidates = np.setdiff1d(np.arange(ctx.targets_full.shape[0]), ctx.train_idx)
    target_indices = rng.choice(candidates, size=N_TARGETS, replace=False)
    print(f"Targets: {N_TARGETS} from {candidates.size} candidates\n")

    _write_sweep_settings(ctx, target_indices, SEED)

    combo_key_str = lambda c: "_".join(s.lower() for s in c)

    header = f"{'combo':<16s} {'E_max':>5s} {'R_eta':>8s}  {'status'}"
    print(header)
    print("-" * len(header))

    for combo in COMPONENT_COMBOS:
        combo_key = tuple(combo)
        label = combo_key_str(combo_key)
        for max_ev in EVAL_STEPS:
            run_dir = RUN_DIR / "runs" / label / f"eval_{max_ev:04d}"
            cached = run_batch(
                ctx, target_indices, OMEGA_P, ETA,
                int(max_ev), combo_key, run_dir,
                target_sample_seed=SEED,
            )
            r = hit_rate(run_dir / "inverse_design.npz")
            status = "cached" if cached else "done"
            print(f"{label:<16s} {max_ev:>5d} {r:>7.1f}%  {status}")

    print("\nAll runs complete.")
    print(f"Sweep manifest: {RUN_DIR / 'sweep_settings.json'}")


if __name__ == "__main__":
    main()
