#!/usr/bin/env python3
"""Evaluate inverse design efficiency across uncertainty weights.

This diagnostic is separate from the main inverse design sweep. It uses the
same 1,000 targets and the full three-component objective. For each target,
50 candidates are preselected by surrogate mean mismatch; the
uncertainty-adjusted score changes only the order in which this same fixed pool
receives oracle evaluation.
The predictive mismatch mean and standard deviation are sampled once and
reused for every multiplier in

    score_i(gamma) = mean_i + gamma * mean(mean_i) / mean(std_i) * std_i,

where gamma=1 reproduces the automatically balanced score used by the
inverse design workflow.

No GP retraining, active learning, or new oracle simulations are performed.
The stored oracle labels are used only to evaluate ranking efficiency.

Author:  Hooman Danesh <hooman.danesh@tu-braunschweig.de>, TU Braunschweig
License: MIT (see LICENSE). Released September 2026.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.resolve()))
sys.path.insert(0, str((ROOT / "src").resolve()))

import inverse_design as inv  # noqa: E402


# ---------------------------------------------------------------------------
# Fixed sensitivity-study configuration
# ---------------------------------------------------------------------------
DATASET_PATH = ROOT / "data/oracle/oracle_45deg.npz"
PCA_PATH = ROOT / "data/pca/pc_scores.npz"
_SURROGATE_DIR = ROOT / "output/surrogate/active_learning"
CHECKPOINT_PATH = _SURROGATE_DIR / "al_model_selected.pt"
if not CHECKPOINT_PATH.exists():
    CHECKPOINT_PATH = _SURROGATE_DIR / "al_model_ntrain_200.pt"
TRAIN_CONFIG_PATH = ROOT / "output/surrogate/active_learning/al_config.json"
SWEEP_MANIFEST_PATH = ROOT / "output/inverse_design/sweep_settings.json"

OUTPUT_DIR = ROOT / "output/studies/lambda_sensitivity"

OMEGA_P = {"P11": 1.0, "P22": 1.0, "P12": 1.0}   # omega_p, component weights
COMPONENTS = ("P11", "P22", "P12")
ETA = 5.0               # eta, acceptance tolerance on the weighted nMAE [%]
N_TARGETS = 1000        # N_tar, held-out targets
TARGET_SEED = 42
PREFILTER_SIZE = 50
ORACLE_BUDGET = 50      # largest E_hit reported by this study
GAMMA_VALUES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)   # gamma in lambda = gamma * lambda_0
MC_SAMPLES = 64         # S, Monte Carlo samples for the predictive moments
MC_BATCH = 100
MC_SEED = 777
SAVE_EVERY = 25


def _path_identity(path: Path) -> Dict[str, Any]:
    """Return a compact identity for an input file used by the study."""
    stat = path.resolve().stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _target_digest(indices: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(indices, dtype=np.int64).ravel())
    return hashlib.sha256(values.tobytes()).hexdigest()


def _load_target_indices(ctx: inv.InverseDesignSetup) -> Tuple[np.ndarray, str]:
    """Reuse the exact main-sweep targets, with a deterministic fallback."""
    if SWEEP_MANIFEST_PATH.is_file():
        manifest = json.loads(SWEEP_MANIFEST_PATH.read_text(encoding="utf-8"))
        stored = np.asarray(manifest.get("random_target_indices", []), dtype=np.int64)
        if stored.shape == (N_TARGETS,):
            if stored.min() < 0 or stored.max() >= ctx.targets_full.shape[0]:
                raise ValueError("The sweep manifest contains out-of-range target indices.")
            if np.intersect1d(stored, ctx.train_idx).size:
                raise ValueError("The sweep manifest target set overlaps the GP training set.")
            return stored, "output/inverse_design/sweep_settings.json"

    rng = np.random.default_rng(TARGET_SEED)
    candidates = np.setdiff1d(np.arange(ctx.targets_full.shape[0]), ctx.train_idx)
    selected = rng.choice(candidates, size=N_TARGETS, replace=False)
    return np.asarray(selected, dtype=np.int64), "deterministic_seed_42_fallback"


def _configuration(
    ctx: inv.InverseDesignSetup,
    target_indices: np.ndarray,
    target_source: str,
) -> Dict[str, Any]:
    return {
        "dataset": _path_identity(DATASET_PATH),
        "pca": _path_identity(PCA_PATH),
        "checkpoint": _path_identity(CHECKPOINT_PATH),
        "train_config": _path_identity(TRAIN_CONFIG_PATH),
        "target_source": target_source,
        "target_seed": TARGET_SEED,
        "target_indices_sha256": _target_digest(target_indices),
        "n_targets": int(target_indices.size),
        "components": list(COMPONENTS),
        "omega_p": {key: float(value) for key, value in OMEGA_P.items()},
        "eta_percent": ETA,
        "prefilter_size": PREFILTER_SIZE,
        "oracle_budget": ORACLE_BUDGET,
        "prefilter_rule": "lowest_surrogate_mean_squared_mismatch",
        "oracle_ranking_rule": "uncertainty_adjusted_mismatch",
        "gamma_values": list(GAMMA_VALUES),
        "lambda_0_definition": "mean(mismatch_mean) / mean(mismatch_std)",
        "candidate_score": "mismatch_mean + gamma * lambda_0 * mismatch_std",
        "mc_samples": MC_SAMPLES,
        "mc_batch": MC_BATCH,
        "mc_seed": MC_SEED,
        "mc_seed_rule": "mc_seed + target_structure_index",
        "pca_dim": int(ctx.z_norm.shape[1]),
        "n_gp_training_structures": int(ctx.train_idx.size),
        # JSON has no comments; this maps the keys above to the symbols
        # used in the manuscript.
        "notation": {
            "n_targets": "N_tar, held-out targets",
            "eta_percent": "eta, acceptance tolerance [%]",
            "omega_p": "omega_p, component weights",
            "oracle_budget": "largest E_hit reported",
            "gamma_values": "gamma in lambda = gamma * lambda_0",
            "lambda_0_definition": "lambda_0, auto-balanced uncertainty weight",
            "mc_samples": "S, Monte Carlo samples",
            "pca_dim": "n_z, descriptor dimension",
        },
    }


def _config_signature(config: Dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _weighted_nmae_for_pool(
    target_vec: np.ndarray,
    oracle_pool: np.ndarray,
    row_labels: Sequence[str],
) -> np.ndarray:
    """Vectorized counterpart of the inverse design stopping criterion."""
    component_values = []
    component_weights = []
    labels = np.asarray(row_labels)
    for component in COMPONENTS:
        mask = labels == component
        weight = float(OMEGA_P.get(component, 0.0))
        if not np.any(mask) or weight <= 0.0:
            continue
        scale = float(np.mean(np.abs(target_vec[mask])))
        if scale <= 1e-12:
            values = np.full(oracle_pool.shape[0], np.inf, dtype=np.float64)
        else:
            values = (
                100.0
                * np.mean(np.abs(oracle_pool[:, mask] - target_vec[mask]), axis=1)
                / scale
            )
        component_values.append(values)
        component_weights.append(weight)
    if not component_values:
        return np.full(oracle_pool.shape[0], np.nan, dtype=np.float64)
    return np.average(
        np.stack(component_values, axis=1),
        axis=1,
        weights=np.asarray(component_weights, dtype=np.float64),
    )


def _empty_progress(n_targets: int, n_gamma: int) -> Dict[str, np.ndarray]:
    return {
        "completed": np.zeros(n_targets, dtype=bool),
        "candidate_indices": np.full(
            (n_targets, PREFILTER_SIZE), -1, dtype=np.int64,
        ),
        "mismatch_mean": np.full(
            (n_targets, PREFILTER_SIZE), np.nan, dtype=np.float32,
        ),
        "mismatch_std": np.full(
            (n_targets, PREFILTER_SIZE), np.nan, dtype=np.float32,
        ),
        "lambda_0": np.full(n_targets, np.nan, dtype=np.float64),
        "oracle_weighted_nmae": np.full(
            (n_targets, PREFILTER_SIZE), np.nan, dtype=np.float32,
        ),
        "ranking_local_indices": np.full(
            (n_gamma, n_targets, ORACLE_BUDGET), -1, dtype=np.int16,
        ),
        "eval_counts": np.full(
            (n_gamma, n_targets), ORACLE_BUDGET + 1, dtype=np.int16,
        ),
        "threshold_met": np.zeros((n_gamma, n_targets), dtype=bool),
    }


def _save_progress(
    path: Path,
    signature: str,
    target_indices: np.ndarray,
    progress: Dict[str, np.ndarray],
) -> None:
    np.savez_compressed(
        path,
        config_signature=np.asarray(signature),
        target_indices=target_indices,
        **progress,
    )


def _load_progress(
    path: Path,
    signature: str,
    target_indices: np.ndarray,
    n_gamma: int,
) -> Dict[str, np.ndarray]:
    if not path.is_file():
        return _empty_progress(target_indices.size, n_gamma)
    cached = np.load(path, allow_pickle=False)
    stored_signature = str(np.asarray(cached["config_signature"]).item())
    stored_targets = np.asarray(cached["target_indices"], dtype=np.int64)
    if stored_signature != signature or not np.array_equal(stored_targets, target_indices):
        return _empty_progress(target_indices.size, n_gamma)
    progress = _empty_progress(target_indices.size, n_gamma)
    for key in progress:
        progress[key] = np.asarray(cached[key])
    return progress


def _write_summary(
    output_dir: Path,
    gamma_values: np.ndarray,
    eval_counts: np.ndarray,
    lambda_0: np.ndarray,
) -> np.ndarray:
    eval_steps = np.arange(1, ORACLE_BUDGET + 1, dtype=np.int64)
    hit_rate = np.stack(
        [100.0 * np.mean(eval_counts <= step, axis=1) for step in eval_steps],
        axis=1,
    )
    fieldnames = [
        "gamma_multiplier",
        "hit_rate_e1_percent",
        "hit_rate_e10_percent",
        "hit_rate_e20_percent",
        "hit_rate_e50_percent",
        "median_e_eta_successful",
        "mean_evaluations_capped_at_51",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for i, gamma in enumerate(gamma_values):
            successful = eval_counts[i] <= ORACLE_BUDGET
            median_success = (
                float(np.median(eval_counts[i, successful]))
                if np.any(successful)
                else float("nan")
            )
            writer.writerow({
                "gamma_multiplier": float(gamma),
                "hit_rate_e1_percent": round(float(hit_rate[i, 0]), 1),
                "hit_rate_e10_percent": round(float(hit_rate[i, 9]), 1),
                "hit_rate_e20_percent": round(float(hit_rate[i, 19]), 1),
                "hit_rate_e50_percent": round(float(hit_rate[i, 49]), 1),
                "median_e_eta_successful": median_success,
                "mean_evaluations_capped_at_51": round(
                    float(np.mean(eval_counts[i])), 3,
                ),
            })

    lambda_summary = {
        "mean": float(np.mean(lambda_0)),
        "std": float(np.std(lambda_0)),
        "minimum": float(np.min(lambda_0)),
        "q05": float(np.quantile(lambda_0, 0.05)),
        "q25": float(np.quantile(lambda_0, 0.25)),
        "median": float(np.median(lambda_0)),
        "q75": float(np.quantile(lambda_0, 0.75)),
        "q95": float(np.quantile(lambda_0, 0.95)),
        "maximum": float(np.max(lambda_0)),
    }
    (output_dir / "lambda_0_summary.json").write_text(
        json.dumps(lambda_summary, indent=2), encoding="utf-8",
    )
    return hit_rate


def run_study() -> None:
    """Run or resume the fixed-target uncertainty weight sensitivity study."""
    start_time = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ctx = inv.setup(
        dataset_path=DATASET_PATH,
        pca_path=PCA_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        train_config_path=TRAIN_CONFIG_PATH,
        seed=MC_SEED,
        device="cpu",
    )
    target_indices, target_source = _load_target_indices(ctx)
    config = _configuration(ctx, target_indices, target_source)
    signature = _config_signature(config)
    config["config_signature"] = signature
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8",
    )

    gamma_values = np.asarray(GAMMA_VALUES, dtype=np.float64)
    progress_path = OUTPUT_DIR / "progress.npz"
    results_path = OUTPUT_DIR / "results.npz"
    if results_path.is_file():
        cached = np.load(results_path, allow_pickle=False)
        cached_signature = str(np.asarray(cached["config_signature"]).item())
        cached_targets = np.asarray(cached["target_indices"], dtype=np.int64)
        if cached_signature == signature and np.array_equal(cached_targets, target_indices):
            eval_steps = np.asarray(cached["eval_steps"], dtype=np.int64)
            cached_gamma = np.asarray(cached["gamma_values"], dtype=np.float64)
            hit_rate = np.asarray(cached["hit_rate_percent"], dtype=np.float64)
            if not np.allclose(
                hit_rate[:, -1], hit_rate[0, -1], rtol=0.0, atol=0.0,
            ):
                raise RuntimeError(
                    "Cached E=50 hit rates are not identical for the fixed pool."
                )
            _write_summary(
                OUTPUT_DIR,
                cached_gamma,
                np.asarray(cached["eval_counts"], dtype=np.int16),
                np.asarray(cached["lambda_0"], dtype=np.float64),
            )
            print(f"Reused complete cached study: {results_path}")
            return

    progress = _load_progress(
        progress_path, signature, target_indices, gamma_values.size,
    )

    prepared = inv.prepare_design_problem(
        ctx, OMEGA_P, components=COMPONENTS,
    )
    row_labels = list(prepared.row_labels)
    targets = np.asarray(ctx.targets_full[:, prepared.row_mask], dtype=np.float64)
    base_weights = inv._build_loss_weights(row_labels, OMEGA_P)

    completed_before = int(np.sum(progress["completed"]))
    if completed_before:
        print(f"Resuming from {completed_before}/{target_indices.size} completed targets.")

    for position, target_idx in enumerate(target_indices):
        if progress["completed"][position]:
            continue

        target_vec = targets[int(target_idx)]
        loss_weights = inv._target_norm_weights(
            target_vec, row_labels, base_weights,
        )
        surrogate_mismatch = np.average(
            (prepared.pred_np - target_vec[np.newaxis, :]) ** 2,
            axis=1,
            weights=loss_weights,
        )
        # np.argpartition guarantees the partition but not the order within the
        # selected block, and candidate_score_terms draws the Monte Carlo
        # samples for the whole pool in one batch, assigning them by position
        # rather than by candidate. Sorting by structure index pins that
        # assignment to a canonical order. Without it, the same pool arriving
        # in a different order shifts every mismatch estimate, and with them
        # lambda_0 and the ranking.
        pool = np.sort(
            np.argpartition(
                surrogate_mismatch, PREFILTER_SIZE - 1,
            )[:PREFILTER_SIZE],
        ).astype(np.int64)

        # Per-target seeding makes the result independent of interruption and
        # gives every gamma multiplier exactly the same Monte Carlo terms.
        torch.manual_seed(MC_SEED + int(target_idx))
        mismatch_mean_t, mismatch_std_t, lambda_0 = inv.candidate_score_terms(
            model=ctx.model,
            z_norm=ctx.z_norm,
            design_t=prepared.design_t,
            target_vec=target_vec,
            loss_weights=loss_weights,
            idxs=pool,
            mc_samples=MC_SAMPLES,
            mc_batch=MC_BATCH,
        )
        mismatch_mean = mismatch_mean_t.cpu().numpy().astype(np.float64)
        mismatch_std = mismatch_std_t.cpu().numpy().astype(np.float64)
        oracle_pool = targets[pool]
        oracle_nmae = _weighted_nmae_for_pool(
            target_vec, oracle_pool, row_labels,
        )

        progress["candidate_indices"][position] = pool
        progress["mismatch_mean"][position] = mismatch_mean
        progress["mismatch_std"][position] = mismatch_std
        progress["lambda_0"][position] = lambda_0
        progress["oracle_weighted_nmae"][position] = oracle_nmae

        for gamma_index, gamma in enumerate(gamma_values):
            scores = mismatch_mean + float(gamma) * lambda_0 * mismatch_std
            order = np.argsort(scores, kind="stable")[:ORACLE_BUDGET]
            progress["ranking_local_indices"][gamma_index, position] = order
            feasible = np.flatnonzero(oracle_nmae[order] <= ETA)
            if feasible.size:
                progress["eval_counts"][gamma_index, position] = int(feasible[0]) + 1
                progress["threshold_met"][gamma_index, position] = True

        progress["completed"][position] = True
        n_done = int(np.sum(progress["completed"]))
        if n_done % SAVE_EVERY == 0 or n_done == target_indices.size:
            _save_progress(
                progress_path, signature, target_indices, progress,
            )
            elapsed = time.perf_counter() - start_time
            print(
                f"Completed {n_done:4d}/{target_indices.size} targets "
                f"({elapsed:.1f} s)."
            )

    if not np.all(progress["completed"]):
        raise RuntimeError("Sensitivity study ended with incomplete targets.")
    if not np.all(np.isfinite(progress["lambda_0"])):
        raise RuntimeError("Non-finite automatic uncertainty weights were produced.")

    hit_rate = _write_summary(
        OUTPUT_DIR,
        gamma_values,
        progress["eval_counts"],
        progress["lambda_0"],
    )
    if not np.allclose(hit_rate[:, -1], hit_rate[0, -1], rtol=0.0, atol=0.0):
        raise RuntimeError(
            "E=50 hit rate must be identical because every multiplier "
            "reorders the same fixed 50-candidate pool."
        )
    np.savez_compressed(
        results_path,
        config_signature=np.asarray(signature),
        target_indices=target_indices,
        gamma_values=gamma_values,
        eval_steps=np.arange(1, ORACLE_BUDGET + 1, dtype=np.int64),
        hit_rate_percent=hit_rate,
        **progress,
    )
    if progress_path.is_file():
        progress_path.unlink()

    print("\nUncertainty weight sensitivity summary")
    print("gamma   R<=1   R<=10  R<=20  R<=50")
    for i, gamma in enumerate(gamma_values):
        print(
            f"{gamma:>5g}  {hit_rate[i, 0]:6.1f}  {hit_rate[i, 9]:6.1f}  "
            f"{hit_rate[i, 19]:6.1f}  {hit_rate[i, 49]:6.1f}"
        )
    print(f"\nResults: {results_path}")


if __name__ == "__main__":
    run_study()
