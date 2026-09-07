#!/usr/bin/env python3
"""Benchmark inverse design against full library random search and BO--EI.

Author:  Hooman Danesh <hooman.danesh@tu-braunschweig.de>, TU Braunschweig
License: MIT (see LICENSE). Released September 2026.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.resolve()))
sys.path.insert(0, str((ROOT / "src").resolve()))

import inverse_design as inv  # noqa: E402
from experiments.baselines import _bo_ei, _shared, random_search  # noqa: E402


# ---------------------------------------------------------------------------
# Fixed benchmark configuration
# ---------------------------------------------------------------------------
DATASET_PATH = _shared.DATASET_PATH
PCA_PATH = _shared.PCA_PATH
CHECKPOINT_PATH = _shared.CHECKPOINT_PATH
TRAIN_CONFIG_PATH = _shared.TRAIN_CONFIG_PATH
SWEEP_MANIFEST_PATH = _shared.SWEEP_MANIFEST_PATH
PROPOSED_RESULTS_PATH = (
    ROOT
    / "output/inverse_design/runs/p11_p22_p12/eval_0050/inverse_design.npz"
)
LEGACY_PROPOSED_RESULTS_PATH = (
    ROOT / "output/studies/lambda_sensitivity/results.npz"
)

BASELINE_ROOT = ROOT / "output/studies/baseline_comparison"

#: One directory per method, so a folder name always matches its contents.
#: The three methods below are computed in a single pass because they share the
#: target loop and the LHS initialisation, but each is published on its own.
METHOD_DIRS = {
    "Proposed": "proposed",
    "BO--EI": "bo_ei_engineered_features",
    "Random": "random_search",
}

#: Canonical stacking order for the per-method rows of the combined arrays.
METHOD_ORDER = ("Proposed", "BO--EI", "Random")

#: Arrays shared by every method, repeated in each per-method file so that a
#: single file is self-describing.
SHARED_KEYS = (
    "config_signature",
    "target_indices",
    "eval_steps",
    "best_nmae_percentile_levels",
    "completed",
)

#: Prefixes that assign a method-specific array to its owner.
METHOD_PREFIXES = {
    "Proposed": ("proposed_",),
    "BO--EI": ("bo_", "lhs_"),
    "Random": ("random_",),
}

#: The resumable progress file lives with the stage that computes the batch.
DEFAULT_OUTPUT_DIR = BASELINE_ROOT / METHOD_DIRS["BO--EI"]

OMEGA_P = _shared.OMEGA_P
COMPONENTS = _shared.COMPONENTS
ETA = _shared.ETA
N_TARGETS = _shared.N_TARGETS
ORACLE_BUDGET = _shared.ORACLE_BUDGET
BO_INITIAL_POINTS = _shared.BO_INITIAL_POINTS
BO_LHS_SEED = _shared.BO_LHS_SEED
RANDOM_SEED = 1701
SETUP_SEED = _shared.SETUP_SEED
EI_XI = _shared.EI_XI
GP_ALPHA = _shared.GP_ALPHA
GP_LENGTH_SCALE_BOUNDS = _shared.GP_LENGTH_SCALE_BOUNDS
GP_AMPLITUDE_BOUNDS = _shared.GP_AMPLITUDE_BOUNDS
SAVE_EVERY = _shared.SAVE_EVERY

_path_identity = _shared.path_identity
_digest = _shared.digest
_config_signature = _shared.config_signature
_load_target_indices = _shared.load_target_indices
_shared_lhs_initial_indices = _shared.shared_lhs_initial_indices
_weighted_nmae_for_pool = _shared.weighted_nmae_for_pool
_oracle_metrics = _shared.oracle_metrics
_hit_rate_curve = _shared.hit_rate_curve
_best_nmae_percentiles = _shared.best_nmae_percentiles


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-targets",
        type=int,
        default=None,
        help="Use only the first N fixed targets (intended for timing pilots).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the resumable cache and numerical summaries.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=SAVE_EVERY,
        help="Save resumable progress after this many completed targets.",
    )
    return parser.parse_args()


def _load_original_proposed_results(
    target_indices: np.ndarray,
    targets: np.ndarray,
    row_labels: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct online nMAE from the original E_max=50 evaluation order."""
    with np.load(PROPOSED_RESULTS_PATH, allow_pickle=True) as proposed:
        proposed_targets = np.asarray(
            proposed["random_target_indices"], dtype=np.int64,
        )
        if not np.array_equal(
            proposed_targets[:target_indices.size], target_indices,
        ):
            raise ValueError("Proposed-method results use a different target order.")
        run_budget = int(np.asarray(proposed["run_e_max"]).item())
        run_eta = float(np.asarray(proposed["run_eta"]).item())
        run_components = tuple(str(value) for value in proposed["components"])
        if run_budget != ORACLE_BUDGET:
            raise ValueError(
                f"Proposed results use E_max={run_budget}, not {ORACLE_BUDGET}."
            )
        if not np.isclose(run_eta, ETA):
            raise ValueError(f"Proposed results use eta={run_eta}, not {ETA}.")
        if run_components != COMPONENTS:
            raise ValueError(
                f"Proposed results use components {run_components}, not {COMPONENTS}."
            )
        raw_counts = np.asarray(
            proposed["random_oracle_eval_counts"][:target_indices.size],
            dtype=np.int64,
        )
        threshold_met = np.asarray(
            proposed["random_oracle_eval_threshold_met"][:target_indices.size],
            dtype=bool,
        )
        evaluated_indices = np.asarray(
            proposed["random_oracle_eval_indices"][:target_indices.size],
            dtype=object,
        )

    nmae = np.full(
        (target_indices.size, ORACLE_BUDGET), np.nan, dtype=np.float64,
    )
    for position, target_idx in enumerate(target_indices):
        sequence = np.asarray(
            evaluated_indices[position], dtype=np.int64,
        ).ravel()
        if sequence.size != int(raw_counts[position]):
            raise ValueError(
                "A proposed-method evaluation sequence disagrees with its count."
            )
        if not 1 <= sequence.size <= ORACLE_BUDGET:
            raise ValueError("A proposed-method evaluation sequence has invalid length.")
        values = _weighted_nmae_for_pool(
            targets[int(target_idx)], targets[sequence], row_labels,
        )
        nmae[position, :sequence.size] = values
        feasible = np.flatnonzero(values <= ETA)
        if threshold_met[position]:
            if feasible.size == 0 or int(feasible[0]) != sequence.size - 1:
                raise ValueError(
                    "A successful proposed sequence does not terminate at its "
                    "first threshold hit."
                )
        elif feasible.size:
            raise ValueError(
                "A failed proposed sequence contains an unreported threshold hit."
            )

    eval_counts = np.where(
        threshold_met, raw_counts, ORACLE_BUDGET + 1,
    ).astype(np.int16)
    return eval_counts, nmae


def _configuration(
    ctx: inv.InverseDesignSetup,
    target_indices: np.ndarray,
    all_target_indices: np.ndarray,
    target_source: str,
    lhs_initial_indices: np.ndarray,
) -> Dict[str, Any]:
    return {
        "dataset": _path_identity(DATASET_PATH),
        "pca": _path_identity(PCA_PATH),
        "checkpoint": _path_identity(CHECKPOINT_PATH),
        "train_config": _path_identity(TRAIN_CONFIG_PATH),
        "proposed_results": _path_identity(PROPOSED_RESULTS_PATH),
        "target_source": target_source,
        "target_indices_sha256": _digest(target_indices),
        "all_fixed_target_indices_sha256": _digest(all_target_indices),
        "n_targets": int(target_indices.size),
        "n_fixed_targets_excluded_from_lhs": int(all_target_indices.size),
        "n_candidates": int(ctx.targets_full.shape[0]),
        "candidate_domain": "entire_structure_library",
        "components": list(COMPONENTS),
        "omega_p": {key: float(value) for key, value in OMEGA_P.items()},
        "eta_percent": ETA,
        "oracle_budget": ORACLE_BUDGET,
        "oracle_budget_definition": "post_initialization_online_evaluations",
        "oracle_budget_includes_initial_points": False,
        "online_hit_rate_excludes_initial_points": True,
        "random_realizations": 1,
        "random_seed": RANDOM_SEED,
        "random_seed_rule": "random_seed + target_structure_index",
        "bo_initial_points": BO_INITIAL_POINTS,
        "bo_initial_seed": BO_LHS_SEED,
        "bo_initial_rule": "one_shared_LHS_in_six_PC_space",
        "bo_initial_reused_for_all_targets": True,
        "bo_initial_target_overlap": int(
            np.intersect1d(lhs_initial_indices, all_target_indices).size
        ),
        "bo_initial_indices_sha256": _digest(lhs_initial_indices),
        "bo_initial_success_used_for_reporting": False,
        "bo_initial_success_used_for_EI_incumbent": True,
        "bo_objective": "log(weighted_oracle_squared_mismatch)",
        "bo_model": "exact_scalar_gaussian_process",
        "bo_kernel": "ConstantKernel_times_RBF_ARD",
        "bo_kernel_lengthscale_bounds": list(GP_LENGTH_SCALE_BOUNDS),
        "bo_kernel_amplitude_bounds": list(GP_AMPLITUDE_BOUNDS),
        "bo_kernel_hyperparameter_source": (
            "optimized_per_target_on_shared_LHS_observations"
        ),
        "bo_kernel_hyperparameters_after_initial_fit": "fixed",
        "bo_posterior_update": "after_every_new_online_oracle_observation",
        "bo_expected_improvement_xi": EI_XI,
        "bo_gp_alpha": GP_ALPHA,
        "bo_normalize_y": True,
        "pca_dim": int(ctx.z_norm.shape[1]),
    }


def _empty_progress(n_targets: int) -> Dict[str, np.ndarray]:
    shape = (n_targets, ORACLE_BUDGET)
    return {
        "completed": np.zeros(n_targets, dtype=bool),
        "random_indices": np.full(shape, -1, dtype=np.int64),
        "random_nmae": np.full(shape, np.nan, dtype=np.float32),
        "random_eval_counts": np.full(
            n_targets, ORACLE_BUDGET + 1, dtype=np.int16,
        ),
        "random_threshold_met": np.zeros(n_targets, dtype=bool),
        "bo_indices": np.full(shape, -1, dtype=np.int64),
        "bo_nmae": np.full(shape, np.nan, dtype=np.float32),
        "bo_squared_mismatch": np.full(shape, np.nan, dtype=np.float32),
        "bo_eval_counts": np.full(
            n_targets, ORACLE_BUDGET + 1, dtype=np.int16,
        ),
        "bo_threshold_met": np.zeros(n_targets, dtype=bool),
        "bo_n_evaluated": np.zeros(n_targets, dtype=np.int16),
        "bo_max_alpha": np.full(n_targets, GP_ALPHA, dtype=np.float64),
        "bo_seconds": np.full(n_targets, np.nan, dtype=np.float64),
        "bo_initial_best_nmae": np.full(n_targets, np.nan, dtype=np.float32),
        "bo_initial_threshold_met_ignored": np.zeros(n_targets, dtype=bool),
        "bo_kernel_amplitude": np.full(n_targets, np.nan, dtype=np.float64),
        "bo_kernel_lengthscale": np.full(
            (n_targets, 6), np.nan, dtype=np.float64,
        ),
    }


def _save_progress(
    path: Path,
    signature: str,
    target_indices: np.ndarray,
    progress: Dict[str, np.ndarray],
) -> None:
    temporary = path.with_name(f"{path.stem}.tmp.npz")
    np.savez_compressed(
        temporary,
        config_signature=np.asarray(signature),
        target_indices=target_indices,
        **progress,
    )
    temporary.replace(path)


def _load_progress(
    path: Path,
    signature: str,
    target_indices: np.ndarray,
) -> Dict[str, np.ndarray]:
    progress = _empty_progress(target_indices.size)
    if not path.is_file():
        return progress
    try:
        cached = np.load(path, allow_pickle=False)
        stored_signature = str(np.asarray(cached["config_signature"]).item())
        stored_targets = np.asarray(cached["target_indices"], dtype=np.int64)
        if stored_signature != signature or not np.array_equal(
            stored_targets, target_indices,
        ):
            return progress
        for key in progress:
            progress[key] = np.asarray(cached[key])
    except (OSError, ValueError, KeyError):
        return _empty_progress(target_indices.size)
    return progress


def _owner_of(key: str) -> str | None:
    """Return the method that owns ``key``, or None when it is shared."""
    if key in SHARED_KEYS or key in ("method_labels", "hit_rate_percent",
                                     "eval_counts", "best_nmae_percentiles"):
        return None
    for method, prefixes in METHOD_PREFIXES.items():
        if any(key.startswith(prefix) for prefix in prefixes):
            return method
    return None


def _write_split_results(root: Path, payload: Dict[str, np.ndarray]) -> List[Path]:
    """Write one ``results.npz`` per method under ``root``.

    The combined payload carries per-method rows stacked in ``METHOD_ORDER``;
    each file receives its own row plus the arrays it owns, so no file repeats
    another method's numbers.
    """
    labels = [str(value) for value in payload["method_labels"]]
    written: List[Path] = []
    for method in METHOD_ORDER:
        row = labels.index(method)
        entry = {key: np.asarray(payload[key]) for key in SHARED_KEYS if key in payload}
        entry["method_label"] = np.asarray(method)
        entry["eval_counts"] = np.asarray(payload["eval_counts"])[row]
        entry["hit_rate_percent"] = np.asarray(payload["hit_rate_percent"])[row]
        entry["best_nmae_percentiles"] = np.asarray(payload["best_nmae_percentiles"])[row]
        for key, value in payload.items():
            if _owner_of(key) == method:
                entry[key] = np.asarray(value)

        directory = root / METHOD_DIRS[method]
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / "results.npz"
        temporary = destination.with_name(f"{destination.stem}.tmp.npz")
        np.savez_compressed(temporary, **entry)
        temporary.replace(destination)
        written.append(destination)
    return written


def load_split_results(root: Path) -> Dict[str, np.ndarray] | None:
    """Reassemble the combined payload from the per-method files.

    Returns None when any method is missing, so callers can fall back to
    recomputing. The per-method rows are restacked in ``METHOD_ORDER``.
    """
    entries = {}
    for method in METHOD_ORDER:
        path = Path(root) / METHOD_DIRS[method] / "results.npz"
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as data:
            entries[method] = {key: np.asarray(data[key]) for key in data.files}

    signatures = {str(entries[m]["config_signature"].item()) for m in METHOD_ORDER}
    if len(signatures) != 1:
        raise ValueError(
            f"Per-method baseline results under {root} disagree on config_signature."
        )
    reference = entries[METHOD_ORDER[0]]["target_indices"]
    for method in METHOD_ORDER[1:]:
        if not np.array_equal(entries[method]["target_indices"], reference):
            raise ValueError(
                f"Per-method baseline results under {root} use different targets."
            )

    payload: Dict[str, np.ndarray] = {
        key: entries[METHOD_ORDER[0]][key]
        for key in SHARED_KEYS
        if key in entries[METHOD_ORDER[0]]
    }
    payload["method_labels"] = np.asarray(list(METHOD_ORDER))
    for stacked in ("eval_counts", "hit_rate_percent", "best_nmae_percentiles"):
        payload[stacked] = np.stack(
            [entries[method][stacked] for method in METHOD_ORDER]
        )
    for method in METHOD_ORDER:
        for key, value in entries[method].items():
            if _owner_of(key) == method:
                payload[key] = value
    return payload


def _write_summary(
    output_dir: Path,
    method_labels: Sequence[str],
    eval_counts: np.ndarray,
    hit_rate_percent: np.ndarray,
    best_nmae_percentiles: np.ndarray,
    percentile_levels: np.ndarray,
) -> None:
    median_rows = np.flatnonzero(
        np.isclose(np.asarray(percentile_levels, dtype=float), 50.0)
    )
    if median_rows.size != 1:
        raise ValueError("Accuracy summary requires exactly one 50th percentile.")
    median_best_nmae = np.asarray(
        best_nmae_percentiles[:, int(median_rows[0]), :], dtype=float,
    )
    fieldnames = [
        "method",
        "hit_rate_e1_percent",
        "hit_rate_e10_percent",
        "hit_rate_e20_percent",
        "hit_rate_e50_percent",
        "median_best_nmae_e1_percent",
        "median_best_nmae_e10_percent",
        "median_best_nmae_e20_percent",
        "median_best_nmae_e50_percent",
        "median_e_eta_successful",
        "mean_evaluations_capped_at_51",
    ]
    with (output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for i, label in enumerate(method_labels):
            successful = eval_counts[i] <= ORACLE_BUDGET
            median_success = (
                float(np.median(eval_counts[i, successful]))
                if np.any(successful)
                else float("nan")
            )
            writer.writerow({
                "method": label,
                "hit_rate_e1_percent": round(float(hit_rate_percent[i, 0]), 1),
                "hit_rate_e10_percent": round(float(hit_rate_percent[i, 9]), 1),
                "hit_rate_e20_percent": round(float(hit_rate_percent[i, 19]), 1),
                "hit_rate_e50_percent": round(float(hit_rate_percent[i, 49]), 1),
                "median_best_nmae_e1_percent": round(
                    float(median_best_nmae[i, 0]), 3,
                ),
                "median_best_nmae_e10_percent": round(
                    float(median_best_nmae[i, 9]), 3,
                ),
                "median_best_nmae_e20_percent": round(
                    float(median_best_nmae[i, 19]), 3,
                ),
                "median_best_nmae_e50_percent": round(
                    float(median_best_nmae[i, 49]), 3,
                ),
                "median_e_eta_successful": median_success,
                "mean_evaluations_capped_at_51": round(
                    float(np.mean(eval_counts[i])), 3,
                ),
            })


def run_benchmark(args: argparse.Namespace) -> None:
    if args.save_every < 1:
        raise ValueError("save_every must be positive.")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ctx = inv.setup(
        dataset_path=DATASET_PATH,
        pca_path=PCA_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        train_config_path=TRAIN_CONFIG_PATH,
        seed=SETUP_SEED,
        device="cpu",
    )
    target_indices, all_target_indices, target_source = _load_target_indices(
        ctx, args.max_targets,
    )
    prepared = inv.prepare_design_problem(
        ctx, OMEGA_P, components=COMPONENTS,
    )
    row_labels = list(prepared.row_labels)
    targets = np.asarray(
        ctx.targets_full[:, prepared.row_mask], dtype=np.float64,
    )
    base_weights = inv._build_loss_weights(row_labels, OMEGA_P)
    x_all = ctx.z_norm.detach().cpu().numpy().astype(np.float64)
    if x_all.shape[1] != 6:
        raise ValueError(f"Expected six PC dimensions; found {x_all.shape[1]}.")
    lhs_initial_indices = _shared_lhs_initial_indices(
        x_all, all_target_indices,
    )

    config = _configuration(
        ctx,
        target_indices,
        all_target_indices,
        target_source,
        lhs_initial_indices,
    )
    signature = _config_signature(config)
    legacy_config = dict(config)
    legacy_config["proposed_results"] = _path_identity(
        LEGACY_PROPOSED_RESULTS_PATH
    )
    legacy_signature = _config_signature(legacy_config)
    config["config_signature"] = signature
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8",
    )

    proposed_eval_counts, proposed_nmae = _load_original_proposed_results(
        target_indices, targets, row_labels,
    )

    methods_root = output_dir.parent
    progress_path = output_dir / "progress.npz"
    cached_payload = load_split_results(methods_root)
    if cached_payload is not None:
        cached_signature = str(
            np.asarray(cached_payload["config_signature"]).item()
        )
        if cached_signature not in {signature, legacy_signature}:
            cached_payload = None
    if cached_payload is not None:
        eval_steps = np.asarray(
            cached_payload["eval_steps"], dtype=np.int64,
        )
        hit_rate = np.asarray(
            cached_payload["hit_rate_percent"], dtype=np.float64,
        ).copy()
        labels = [
            str(value) for value in cached_payload["method_labels"]
        ]
        eval_counts = np.asarray(
            cached_payload["eval_counts"], dtype=np.int64,
        ).copy()
        if set(labels) != {"Proposed", "BO--EI", "Random"}:
            raise ValueError(f"Unexpected cached method labels: {labels}.")
        proposed_row = labels.index("Proposed")
        proposed_rows_current = (
            np.array_equal(
                eval_counts[proposed_row], proposed_eval_counts,
            )
            and np.array_equal(
                hit_rate[proposed_row],
                _hit_rate_curve(proposed_eval_counts),
            )
            and "proposed_nmae" in cached_payload
            and np.allclose(
                cached_payload["proposed_nmae"],
                proposed_nmae,
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )
        )
        eval_counts[proposed_row] = proposed_eval_counts
        hit_rate[proposed_row] = _hit_rate_curve(proposed_eval_counts)
        nmae_lookup = {
            "Proposed": proposed_nmae,
            "BO--EI": np.asarray(
                cached_payload["bo_nmae"], dtype=np.float64,
            ),
            "Random": np.asarray(
                cached_payload["random_nmae"], dtype=np.float64,
            ),
        }
        nmae_by_method = np.stack([
            nmae_lookup[label] for label in labels
        ])
        percentile_levels = np.asarray([25.0, 50.0, 75.0])
        best_nmae_percentiles = _best_nmae_percentiles(
            nmae_by_method, eval_counts, percentile_levels,
        )
        accuracy_keys_current = (
            "best_nmae_percentile_levels" in cached_payload
            and "best_nmae_percentiles" in cached_payload
            and np.array_equal(
                cached_payload["best_nmae_percentile_levels"],
                percentile_levels,
            )
            and np.allclose(
                cached_payload["best_nmae_percentiles"],
                best_nmae_percentiles,
                rtol=0.0,
                atol=0.0,
            )
        )
        cache_needs_update = (
            cached_signature != signature
            or not proposed_rows_current
            or not accuracy_keys_current
        )
        cached_payload["config_signature"] = np.asarray(signature)
        cached_payload["eval_counts"] = eval_counts
        cached_payload["hit_rate_percent"] = hit_rate
        cached_payload["proposed_eval_counts"] = proposed_eval_counts
        cached_payload["proposed_nmae"] = proposed_nmae
        cached_payload["best_nmae_percentile_levels"] = percentile_levels
        cached_payload["best_nmae_percentiles"] = best_nmae_percentiles
        if cache_needs_update:
            _write_split_results(methods_root, cached_payload)
        _write_summary(
            methods_root,
            labels,
            eval_counts,
            hit_rate,
            best_nmae_percentiles,
            percentile_levels,
        )
        print(f"Reused complete cached benchmark under: {methods_root}")
        return

    progress = _load_progress(progress_path, signature, target_indices)
    completed_before = int(np.sum(progress["completed"]))
    if completed_before:
        print(
            f"Resuming from {completed_before}/{target_indices.size} "
            "completed targets."
        )
    benchmark_start = time.perf_counter()

    for position, target_idx in enumerate(target_indices):
        if progress["completed"][position]:
            continue
        target_vec = targets[int(target_idx)]
        loss_weights = inv._target_norm_weights(
            target_vec, row_labels, base_weights,
        )

        random_result = random_search.run_for_target(
            int(target_idx),
            target_vec,
            x_all.shape[0],
            targets,
            row_labels,
            loss_weights,
            _oracle_metrics,
            budget=ORACLE_BUDGET,
            eta=ETA,
            base_seed=RANDOM_SEED,
        )

        bo_result = _bo_ei.run_for_target(
            target_vec,
            lhs_initial_indices,
            x_all,
            targets,
            row_labels,
            loss_weights,
        )

        progress["random_indices"][position] = random_result["indices"]
        progress["random_nmae"][position] = random_result["nmae"]
        progress["random_eval_counts"][position] = random_result["eval_count"]
        progress["random_threshold_met"][position] = random_result[
            "threshold_met"
        ]
        n_bo = int(bo_result["n_evaluated"])
        progress["bo_indices"][position, :n_bo] = bo_result["indices"]
        progress["bo_nmae"][position, :n_bo] = bo_result["nmae"]
        progress["bo_squared_mismatch"][position, :n_bo] = (
            bo_result["squared_mismatch"]
        )
        progress["bo_eval_counts"][position] = bo_result["eval_count"]
        progress["bo_threshold_met"][position] = bo_result["threshold_met"]
        progress["bo_n_evaluated"][position] = n_bo
        progress["bo_max_alpha"][position] = bo_result["max_alpha"]
        progress["bo_seconds"][position] = bo_result["seconds"]
        progress["bo_initial_best_nmae"][position] = (
            bo_result["initial_best_nmae"]
        )
        progress["bo_initial_threshold_met_ignored"][position] = (
            bo_result["initial_threshold_met_ignored"]
        )
        progress["bo_kernel_amplitude"][position] = (
            bo_result["kernel_amplitude"]
        )
        progress["bo_kernel_lengthscale"][position] = (
            bo_result["kernel_lengthscale"]
        )
        progress["completed"][position] = True

        n_done = int(np.sum(progress["completed"]))
        if n_done % args.save_every == 0 or n_done == target_indices.size:
            _save_progress(
                progress_path, signature, target_indices, progress,
            )
            elapsed = time.perf_counter() - benchmark_start
            mean_bo_seconds = float(
                np.nanmean(progress["bo_seconds"][progress["completed"]])
            )
            print(
                f"Completed {n_done:4d}/{target_indices.size} targets "
                f"({elapsed:.1f} s; mean BO {mean_bo_seconds:.3f} s/target)."
            )

    if not np.all(progress["completed"]):
        raise RuntimeError("Full library baseline benchmark is incomplete.")

    method_labels = np.asarray(["Proposed", "BO--EI", "Random"])
    eval_counts = np.stack([
        proposed_eval_counts,
        progress["bo_eval_counts"],
        progress["random_eval_counts"],
    ])
    hit_rate = np.stack([
        _hit_rate_curve(eval_counts[i]) for i in range(eval_counts.shape[0])
    ])
    eval_steps = np.arange(1, ORACLE_BUDGET + 1, dtype=np.int64)
    percentile_levels = np.asarray([25.0, 50.0, 75.0])
    nmae_by_method = np.stack([
        proposed_nmae,
        progress["bo_nmae"],
        progress["random_nmae"],
    ])
    best_nmae_percentiles = _best_nmae_percentiles(
        nmae_by_method, eval_counts, percentile_levels,
    )
    _write_summary(
        methods_root,
        method_labels,
        eval_counts,
        hit_rate,
        best_nmae_percentiles,
        percentile_levels,
    )

    combined_payload = {
        "config_signature": np.asarray(signature),
        "target_indices": target_indices,
        "lhs_initial_indices": lhs_initial_indices,
        "method_labels": method_labels,
        "eval_steps": eval_steps,
        "eval_counts": eval_counts,
        "hit_rate_percent": hit_rate,
        "best_nmae_percentile_levels": percentile_levels,
        "best_nmae_percentiles": best_nmae_percentiles,
        "proposed_eval_counts": proposed_eval_counts,
        "proposed_nmae": proposed_nmae,
        **progress,
    }
    written_paths = _write_split_results(methods_root, combined_payload)
    if progress_path.is_file():
        progress_path.unlink()

    print("\nFull library baseline summary")
    print("method       R<=1   R<=10  R<=20  R<=50  mean(E capped at 51)")
    for i, label in enumerate(method_labels):
        print(
            f"{label:<11} {hit_rate[i, 0]:6.1f}  {hit_rate[i, 9]:6.1f}  "
            f"{hit_rate[i, 19]:6.1f}  {hit_rate[i, 49]:6.1f}  "
            f"{np.mean(eval_counts[i]):8.3f}"
        )
    print("\nResults:")
    for path in written_paths:
        print(f"  {path}")


if __name__ == "__main__":
    run_benchmark(_parse_args())
