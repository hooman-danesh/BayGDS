#!/usr/bin/env python3
"""Run an isolated BO--EI descriptor ablation using PCA of raw images.

This diagnostic keeps the existing proposed-framework, proposed-feature
BO--EI, and random search results untouched. It flattens each 96 x 96 binary
structure, fits a
deterministic six-component PCA without using oracle labels, and runs the same
full library BO--EI procedure.

Author:  Hooman Danesh <hooman.danesh@tu-braunschweig.de>, TU Braunschweig
License: MIT (see LICENSE). Released September 2026.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.resolve()))
sys.path.insert(0, str((ROOT / "src").resolve()))

import inverse_design as inv  # noqa: E402
from experiments.baselines import (  # noqa: E402
    _bo_ei,
    _shared as base,
    bo_ei_proposed_features,
)


STRUCTURES_PATH = ROOT / "data/structures/structures.npz"
DEFAULT_OUTPUT_DIR = (
    bo_ei_proposed_features.BASELINE_ROOT / "bo_ei_raw_image_pcs"
)
DEFAULT_FEATURE_PATH = DEFAULT_OUTPUT_DIR / "raw_structure_pca_06.npz"


PCA_COMPONENTS = 6
PCA_RANDOM_SEED = 314159
PCA_ITERATED_POWER = 4

warnings.filterwarnings("ignore", category=ConvergenceWarning)


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
        help="Directory for the isolated resumable raw-PCA BO result.",
    )
    parser.add_argument(
        "--feature-path",
        type=Path,
        default=DEFAULT_FEATURE_PATH,
        help="Cached deterministic raw-image PCA artifact.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=base.SAVE_EVERY,
        help="Save resumable progress after this many completed targets.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Independent targets to process in parallel (results are unchanged).",
    )
    return parser.parse_args()


def _feature_configuration() -> Dict[str, Any]:
    return {
        "source": base.path_identity(STRUCTURES_PATH),
        "representation": "PCA_of_flattened_raw_binary_structure_images",
        "input_dtype": "float64",
        "uses_oracle_labels": False,
        "n_components": PCA_COMPONENTS,
        "svd_solver": "randomized",
        "random_state": PCA_RANDOM_SEED,
        "iterated_power": PCA_ITERATED_POWER,
        "copy": True,
        "power_iteration_normalizer": "QR",
        "whiten": False,
    }


def _build_raw_pca_features(feature_path: Path) -> np.ndarray:
    """Create or load the deterministic raw-image PCA score matrix."""
    expected_config = _feature_configuration()
    expected_json = json.dumps(expected_config, sort_keys=True)
    if feature_path.is_file():
        with np.load(feature_path, allow_pickle=False) as cached:
            stored_json = str(np.asarray(cached["pca_config_json"]).item())
            scores = np.asarray(cached["scores"], dtype=np.float32)
        if stored_json == expected_json and scores.shape == (
            50000, PCA_COMPONENTS,
        ):
            print(f"Reused raw-image PCA artifact: {feature_path}")
            return scores
        raise ValueError(
            f"Existing raw-PCA artifact has incompatible metadata: {feature_path}"
        )

    print(
        "Fitting deterministic PCA to 50,000 flattened 96 x 96 raw binary "
        "structure images...",
        flush=True,
    )
    start = time.perf_counter()
    with np.load(STRUCTURES_PATH, allow_pickle=False) as data:
        structures = np.asarray(data["structures"])
    if structures.shape != (50000, 96, 96):
        raise ValueError(
            "Expected raw structures with shape (50000, 96, 96); "
            f"found {structures.shape}."
        )
    if structures.dtype != np.uint8:
        raise ValueError(f"Expected uint8 raw structures; found {structures.dtype}.")

    flat = structures.reshape(structures.shape[0], -1).astype(
        np.float64, copy=True,
    )
    del structures
    pca = PCA(
        n_components=PCA_COMPONENTS,
        svd_solver="randomized",
        whiten=False,
        copy=True,
        iterated_power=PCA_ITERATED_POWER,
        power_iteration_normalizer="QR",
        random_state=PCA_RANDOM_SEED,
    )
    scores = np.asarray(pca.fit_transform(flat), dtype=np.float32)
    cumulative_variance = float(np.sum(pca.explained_variance_ratio_))
    score_scale = np.std(scores, axis=0, ddof=1)
    if (
        scores.shape != (50000, PCA_COMPONENTS)
        or not np.all(np.isfinite(scores))
        or not np.all(np.isfinite(score_scale))
        or np.any(score_scale <= 1e-6)
        or cumulative_variance <= 1e-3
    ):
        raise RuntimeError("Raw-image PCA returned invalid scores.")

    feature_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = feature_path.with_name(f"{feature_path.stem}.tmp.npz")
    np.savez_compressed(
        temporary,
        pca_config_json=np.asarray(expected_json),
        scores=scores,
        components=np.asarray(pca.components_, dtype=np.float32),
        mean=np.asarray(pca.mean_, dtype=np.float32),
        explained_variance=np.asarray(pca.explained_variance_, dtype=np.float64),
        explained_variance_ratio=np.asarray(
            pca.explained_variance_ratio_, dtype=np.float64,
        ),
        singular_values=np.asarray(pca.singular_values_, dtype=np.float64),
        image_shape=np.asarray([96, 96], dtype=np.int64),
    )
    temporary.replace(feature_path)
    elapsed = time.perf_counter() - start
    print(
        f"Saved raw-image PCA artifact in {elapsed:.1f} s "
        f"({PCA_COMPONENTS} PCs): {feature_path}",
        flush=True,
    )
    return scores


def _standardize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    mean = np.mean(scores, axis=0)
    scale = np.std(scores, axis=0, ddof=1)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("Raw PCA contains a constant or non-finite component.")
    standardized = (scores - mean) / scale
    if not np.all(np.isfinite(standardized)):
        raise ValueError("Standardized raw PCA scores contain non-finite values.")
    return standardized


def _configuration(
    ctx: inv.InverseDesignSetup,
    target_indices: np.ndarray,
    all_target_indices: np.ndarray,
    target_source: str,
    lhs_initial_indices: np.ndarray,
    feature_path: Path,
) -> Dict[str, Any]:
    return {
        "dataset": base.path_identity(base.DATASET_PATH),
        "structures": base.path_identity(STRUCTURES_PATH),
        "raw_pca_artifact": base.path_identity(feature_path),
        "target_source": target_source,
        "target_indices_sha256": base.digest(target_indices),
        "all_fixed_target_indices_sha256": base.digest(all_target_indices),
        "n_targets": int(target_indices.size),
        "n_fixed_targets_excluded_from_lhs": int(all_target_indices.size),
        "n_candidates": int(ctx.targets_full.shape[0]),
        "candidate_domain": "entire_structure_library",
        "components": list(base.COMPONENTS),
        "omega_p": {key: float(value) for key, value in base.OMEGA_P.items()},
        "eta_percent": base.ETA,
        "oracle_budget": base.ORACLE_BUDGET,
        "oracle_budget_definition": "post_initialization_online_evaluations",
        "oracle_budget_includes_initial_points": False,
        "online_hit_rate_excludes_initial_points": True,
        "bo_feature_representation": "16_PC_scores_of_flattened_raw_images",
        "bo_feature_dimension": PCA_COMPONENTS,
        "bo_feature_standardization": "full_library_zero_mean_unit_sample_std",
        "bo_feature_uses_oracle_labels": False,
        "bo_initial_points": base.BO_INITIAL_POINTS,
        "bo_initial_seed": base.BO_LHS_SEED,
        "bo_initial_rule": "one_shared_LHS_in_16_raw_image_PC_space",
        "bo_initial_reused_for_all_targets": True,
        "bo_initial_target_overlap": int(
            np.intersect1d(lhs_initial_indices, all_target_indices).size
        ),
        "bo_initial_indices_sha256": base.digest(lhs_initial_indices),
        "bo_initial_success_used_for_reporting": False,
        "bo_initial_success_used_for_EI_incumbent": True,
        "bo_objective": "log(weighted_oracle_squared_mismatch)",
        "bo_model": "exact_scalar_gaussian_process",
        "bo_kernel": "ConstantKernel_times_RBF_ARD",
        "bo_kernel_lengthscale_bounds": list(base.GP_LENGTH_SCALE_BOUNDS),
        "bo_kernel_amplitude_bounds": list(base.GP_AMPLITUDE_BOUNDS),
        "bo_kernel_hyperparameter_source": (
            "optimized_per_target_on_shared_raw_PCA_LHS_observations"
        ),
        "bo_kernel_hyperparameters_after_initial_fit": "fixed",
        "bo_posterior_update": "after_every_new_online_oracle_observation",
        "bo_expected_improvement_xi": base.EI_XI,
        "bo_gp_alpha": base.GP_ALPHA,
        "bo_normalize_y": True,
        "numerical_inner_threads": 1,
    }


def _empty_progress(n_targets: int) -> Dict[str, np.ndarray]:
    sequence_shape = (n_targets, base.ORACLE_BUDGET)
    return {
        "completed": np.zeros(n_targets, dtype=bool),
        "bo_indices": np.full(sequence_shape, -1, dtype=np.int64),
        "bo_nmae": np.full(sequence_shape, np.nan, dtype=np.float32),
        "bo_squared_mismatch": np.full(
            sequence_shape, np.nan, dtype=np.float32,
        ),
        "bo_eval_counts": np.full(
            n_targets, base.ORACLE_BUDGET + 1, dtype=np.int16,
        ),
        "bo_threshold_met": np.zeros(n_targets, dtype=bool),
        "bo_n_evaluated": np.zeros(n_targets, dtype=np.int16),
        "bo_max_alpha": np.full(n_targets, base.GP_ALPHA, dtype=np.float64),
        "bo_seconds": np.full(n_targets, np.nan, dtype=np.float64),
        "bo_initial_best_nmae": np.full(n_targets, np.nan, dtype=np.float32),
        "bo_initial_threshold_met_ignored": np.zeros(n_targets, dtype=bool),
        "bo_kernel_amplitude": np.full(n_targets, np.nan, dtype=np.float64),
        "bo_kernel_lengthscale": np.full(
            (n_targets, PCA_COMPONENTS), np.nan, dtype=np.float64,
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
    empty = _empty_progress(target_indices.size)
    if not path.is_file():
        return empty
    try:
        with np.load(path, allow_pickle=False) as cached:
            if str(np.asarray(cached["config_signature"]).item()) != signature:
                return empty
            if not np.array_equal(cached["target_indices"], target_indices):
                return empty
            loaded = {key: np.asarray(cached[key]) for key in empty}
        if any(loaded[key].shape != empty[key].shape for key in empty):
            return empty
        return loaded
    except (OSError, ValueError, KeyError):
        return empty


def _write_summary(
    output_dir: Path,
    eval_counts: np.ndarray,
    hit_rate: np.ndarray,
    best_percentiles: np.ndarray,
    percentile_levels: np.ndarray,
) -> None:
    median_row = int(np.flatnonzero(np.isclose(percentile_levels, 50.0))[0])
    median = best_percentiles[median_row]
    successful = eval_counts <= base.ORACLE_BUDGET
    fields = [
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
        "std_evaluations_capped_at_51",
    ]
    with (output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "method": "BO--EI (raw--16)",
            "hit_rate_e1_percent": round(float(hit_rate[0]), 1),
            "hit_rate_e10_percent": round(float(hit_rate[9]), 1),
            "hit_rate_e20_percent": round(float(hit_rate[19]), 1),
            "hit_rate_e50_percent": round(float(hit_rate[49]), 1),
            "median_best_nmae_e1_percent": round(float(median[0]), 3),
            "median_best_nmae_e10_percent": round(float(median[9]), 3),
            "median_best_nmae_e20_percent": round(float(median[19]), 3),
            "median_best_nmae_e50_percent": round(float(median[49]), 3),
            "median_e_eta_successful": (
                float(np.median(eval_counts[successful]))
                if np.any(successful) else float("nan")
            ),
            "mean_evaluations_capped_at_51": round(
                float(np.mean(eval_counts)), 3,
            ),
            "std_evaluations_capped_at_51": round(
                float(np.std(eval_counts, ddof=1)), 3,
            ),
        })


def _load_complete_result(
    results_path: Path,
    signature: str,
    target_indices: np.ndarray,
) -> Dict[str, np.ndarray] | None:
    if not results_path.is_file():
        return None
    try:
        with np.load(results_path, allow_pickle=False) as cached:
            if str(np.asarray(cached["config_signature"]).item()) != signature:
                return None
            if not np.array_equal(cached["target_indices"], target_indices):
                return None
            return {key: np.asarray(cached[key]) for key in cached.files}
    except (OSError, ValueError, KeyError):
        return None


def _run_target(
    position: int,
    target_idx: int,
    lhs_initial_indices: np.ndarray,
    x_all: np.ndarray,
    targets: np.ndarray,
    row_labels: Sequence[str],
    base_weights: np.ndarray,
) -> tuple[int, Dict[str, Any]]:
    target_vec = targets[int(target_idx)]
    loss_weights = inv._target_norm_weights(
        target_vec, row_labels, base_weights,
    )
    result = _bo_ei.run_for_target(
        target_vec,
        lhs_initial_indices,
        x_all,
        targets,
        row_labels,
        loss_weights,
    )
    return position, result


def _store_target_result(
    progress: Dict[str, np.ndarray],
    position: int,
    result: Dict[str, Any],
) -> None:
    n_evaluated = int(result["n_evaluated"])
    progress["bo_indices"][position, :n_evaluated] = result["indices"]
    progress["bo_nmae"][position, :n_evaluated] = result["nmae"]
    progress["bo_squared_mismatch"][position, :n_evaluated] = result[
        "squared_mismatch"
    ]
    progress["bo_eval_counts"][position] = result["eval_count"]
    progress["bo_threshold_met"][position] = result["threshold_met"]
    progress["bo_n_evaluated"][position] = n_evaluated
    progress["bo_max_alpha"][position] = result["max_alpha"]
    progress["bo_seconds"][position] = result["seconds"]
    progress["bo_initial_best_nmae"][position] = result[
        "initial_best_nmae"
    ]
    progress["bo_initial_threshold_met_ignored"][position] = result[
        "initial_threshold_met_ignored"
    ]
    progress["bo_kernel_amplitude"][position] = result["kernel_amplitude"]
    lengthscale = np.asarray(result["kernel_lengthscale"])
    if lengthscale.shape != (PCA_COMPONENTS,):
        raise RuntimeError(
            "Raw-PCA BO fitted an unexpected ARD length-scale vector: "
            f"{lengthscale.shape}."
        )
    progress["bo_kernel_lengthscale"][position] = lengthscale
    progress["completed"][position] = True


def run_bo_ei_raw_image_pcs(args: argparse.Namespace) -> None:
    if args.save_every < 1:
        raise ValueError("save_every must be positive.")
    if args.n_jobs < 1:
        raise ValueError("n_jobs must be positive.")
    output_dir = args.output_dir.resolve()
    feature_path = args.feature_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_scores = _build_raw_pca_features(feature_path)
    x_all = _standardize_scores(raw_scores)
    del raw_scores

    ctx = inv.setup(
        dataset_path=base.DATASET_PATH,
        pca_path=base.PCA_PATH,
        checkpoint_path=base.CHECKPOINT_PATH,
        train_config_path=base.TRAIN_CONFIG_PATH,
        seed=base.SETUP_SEED,
        device="cpu",
    )
    target_indices, all_target_indices, target_source = base.load_target_indices(
        ctx, args.max_targets,
    )
    prepared = inv.prepare_design_problem(
        ctx, base.OMEGA_P, components=base.COMPONENTS,
    )
    row_labels: Sequence[str] = list(prepared.row_labels)
    targets = np.asarray(
        ctx.targets_full[:, prepared.row_mask], dtype=np.float64,
    )
    if x_all.shape != (targets.shape[0], PCA_COMPONENTS):
        raise ValueError(
            "Raw-PCA score rows do not match the oracle library: "
            f"{x_all.shape} versus {targets.shape[0]} candidates."
        )
    base_weights = inv._build_loss_weights(row_labels, base.OMEGA_P)
    lhs_initial_indices = base.shared_lhs_initial_indices(
        x_all, all_target_indices,
    )

    config = _configuration(
        ctx,
        target_indices,
        all_target_indices,
        target_source,
        lhs_initial_indices,
        feature_path,
    )
    signature = base.config_signature(config)
    config["execution_n_jobs"] = int(args.n_jobs)
    config["config_signature"] = signature
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8",
    )

    results_path = output_dir / "results.npz"
    progress_path = output_dir / "progress.npz"
    cached_result = _load_complete_result(
        results_path, signature, target_indices,
    )
    if cached_result is not None:
        _write_summary(
            output_dir,
            cached_result["eval_counts"],
            cached_result["hit_rate_percent"],
            cached_result["best_nmae_percentiles"],
            cached_result["best_nmae_percentile_levels"],
        )
        print(f"Reused complete raw-PCA BO result: {results_path}")
        return

    progress = _load_progress(progress_path, signature, target_indices)
    n_completed = int(np.sum(progress["completed"]))
    if n_completed:
        print(
            f"Resuming raw-PCA BO from {n_completed}/{target_indices.size} "
            "completed targets.",
            flush=True,
        )
    run_start = time.perf_counter()
    pending = [
        (position, int(target_idx))
        for position, target_idx in enumerate(target_indices)
        if not progress["completed"][position]
    ]
    # Fix the numerical thread count so target results do not depend on the
    # chosen amount of target-level parallelism.
    with threadpool_limits(limits=1):
        if args.n_jobs == 1:
            result_stream = (
                _run_target(
                    position,
                    target_idx,
                    lhs_initial_indices,
                    x_all,
                    targets,
                    row_labels,
                    base_weights,
                )
                for position, target_idx in pending
            )
        else:
            result_stream = Parallel(
                n_jobs=args.n_jobs,
                backend="threading",
                return_as="generator_unordered",
                batch_size=1,
            )(
                delayed(_run_target)(
                    position,
                    target_idx,
                    lhs_initial_indices,
                    x_all,
                    targets,
                    row_labels,
                    base_weights,
                )
                for position, target_idx in pending
            )
        for position, result in result_stream:
            _store_target_result(progress, position, result)
            n_done = int(np.sum(progress["completed"]))
            if n_done % args.save_every == 0 or n_done == target_indices.size:
                _save_progress(
                    progress_path, signature, target_indices, progress,
                )
                elapsed = time.perf_counter() - run_start
                mean_seconds = float(
                    np.nanmean(progress["bo_seconds"][progress["completed"]])
                )
                print(
                    f"Completed {n_done:4d}/{target_indices.size} raw-PCA BO "
                    f"targets ({elapsed:.1f} s; mean {mean_seconds:.3f} "
                    "s/target).",
                    flush=True,
                )

    if not np.all(progress["completed"]):
        raise RuntimeError("Raw-PCA BO benchmark is incomplete.")

    eval_steps = np.arange(1, base.ORACLE_BUDGET + 1, dtype=np.int64)
    eval_counts = np.asarray(progress["bo_eval_counts"], dtype=np.int64)
    hit_rate = base.hit_rate_curve(eval_counts)
    percentile_levels = np.asarray([25.0, 50.0, 75.0])
    best_percentiles = base.best_nmae_percentiles(
        progress["bo_nmae"][np.newaxis, :, :],
        eval_counts[np.newaxis, :],
        percentile_levels,
    )[0]
    _write_summary(
        output_dir,
        eval_counts,
        hit_rate,
        best_percentiles,
        percentile_levels,
    )

    raw_results: Dict[str, np.ndarray] = {
        "config_signature": np.asarray(signature),
        "target_indices": target_indices,
        "lhs_initial_indices": lhs_initial_indices,
        "method_label": np.asarray("BO--EI (raw--16)"),
        "eval_steps": eval_steps,
        "eval_counts": eval_counts,
        "hit_rate_percent": hit_rate,
        "best_nmae_percentile_levels": percentile_levels,
        "best_nmae_percentiles": best_percentiles,
        **progress,
    }
    temporary = results_path.with_name(f"{results_path.stem}.tmp.npz")
    np.savez_compressed(temporary, **raw_results)
    temporary.replace(results_path)
    if progress_path.is_file():
        progress_path.unlink()


    print("\nRaw-image PCA BO summary")
    print("R<=1   R<=10  R<=20  R<=50  mean(E capped at 51)")
    print(
        f"{hit_rate[0]:6.1f}  {hit_rate[9]:6.1f}  "
        f"{hit_rate[19]:6.1f}  {hit_rate[49]:6.1f}  "
        f"{np.mean(eval_counts):8.3f}"
    )
    print(f"\nResults: {results_path}")


if __name__ == "__main__":
    run_bo_ei_raw_image_pcs(_parse_args())
