#!/usr/bin/env python3
"""Study GP prediction error as a function of retained PC score dimension.

This is a fixed-split supervised experiment, separate from the active learning
workflow. A single random subset of structures is divided into training and
test sets once, and the same split is reused for every retained PC dimension.

Outputs are written to ``output/studies/pc_dimension``.

Author:  Hooman Danesh <hooman.danesh@tu-braunschweig.de>, TU Braunschweig
License: MIT (see LICENSE). Released September 2026.
"""
from __future__ import annotations

import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.resolve()))
sys.path.insert(0, str((ROOT / "src").resolve()))

from data_utils import (  # noqa: E402
    build_design_matrix_nominal,
    load_dataset,
    load_pca,
    select_fit_indices_0deg,
    stack_stress_targets_nominal,
)
from gp_core import (  # noqa: E402
    CorrelatedLatentSVGP,
    GPTrainConfig,
    PARAM_NAMES,
    compute_stress_normalization,
    normalize_z,
    save_checkpoint,
    save_normalization_stats,
    train_model,
)


# ---------------------------------------------------------------------------
# Study configuration
# ---------------------------------------------------------------------------
DATASET_PATH = ROOT / "data/oracle/oracle_0deg.npz"
PCA_PATH = ROOT / "data/pca/pc_scores.npz"
OUTPUT_DIR = ROOT / "output/studies/pc_dimension"

SEED = 1506
SAMPLE_SIZE = 5000
TRAIN_FRACTION = 0.8
N_INDUCING = 200
PC_COUNTS = tuple(range(1, 9))   # n_z values swept by this study

BATCH_SIZE = 1024
EPOCHS = 500
LEARNING_RATE = 0.01
N_R = 6                 # n_r, total latent processes
S = 64                  # S, Monte Carlo samples for the predictive moments
LOG_EVERY = 100
SAVE_MODELS = True
FIRST_RESTART_EPOCHS = 50


def _set_reproducible(seed: int) -> None:
    """Set the single random seed used throughout the fixed-split study."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def _write_metrics(output_dir: Path, records: list[dict], per_structure_mae: list[np.ndarray]) -> None:
    """Persist progressive tabular results so interrupted runs remain useful."""
    fieldnames = [
        "n_pcs",
        "n_train",
        "n_test",
        "n_inducing",
        "test_mae_mpa",
        "test_rmse_mpa",
        "train_seconds",
        "final_noise",
    ]
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    np.savez_compressed(
        output_dir / "metrics.npz",
        n_pcs=np.asarray([r["n_pcs"] for r in records], dtype=np.int64),
        test_mae_mpa=np.asarray([r["test_mae_mpa"] for r in records], dtype=np.float64),
        test_rmse_mpa=np.asarray([r["test_rmse_mpa"] for r in records], dtype=np.float64),
        train_seconds=np.asarray([r["train_seconds"] for r in records], dtype=np.float64),
        final_noise=np.asarray([r["final_noise"] for r in records], dtype=np.float64),
        per_structure_mae_mpa=np.stack(per_structure_mae, axis=0),
    )


def run_study(
    *,
    dataset_path: Path = DATASET_PATH,
    pca_path: Path = PCA_PATH,
    output_dir: Path = OUTPUT_DIR,
    seed: int = SEED,
    sample_size: int = SAMPLE_SIZE,
    train_fraction: float = TRAIN_FRACTION,
    n_inducing: int = N_INDUCING,
    pc_counts: Sequence[int] = PC_COUNTS,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    n_r: int = N_R,
    mc_samples: int = S,
    log_every: int = LOG_EVERY,
    save_models: bool = SAVE_MODELS,
) -> list[dict]:
    """Run the fixed-split PC dimension study and return summary records."""
    pc_counts = tuple(int(k) for k in pc_counts)
    if not pc_counts or min(pc_counts) < 1:
        raise ValueError("pc_counts must contain positive integers.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie strictly between zero and one.")
    if sample_size < 2:
        raise ValueError("sample_size must be at least two.")

    _set_reproducible(seed)
    device = torch.device("cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_dir = output_dir / "histories"
    history_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    if save_models:
        model_dir.mkdir(parents=True, exist_ok=True)

    pcs_all = load_pca(pca_path, n_components=max(pc_counts))
    n_structures = int(pcs_all.shape[0])
    if pcs_all.shape[1] < max(pc_counts):
        raise ValueError(
            f"Requested {max(pc_counts)} PCs, but {pca_path} contains only {pcs_all.shape[1]}."
        )
    if sample_size > n_structures:
        raise ValueError(f"sample_size={sample_size} exceeds {n_structures} available structures.")

    rng = np.random.default_rng(seed)
    selected_idx = rng.choice(n_structures, size=sample_size, replace=False)
    rng.shuffle(selected_idx)
    n_train = int(round(sample_size * train_fraction))
    if not 0 < n_train < sample_size:
        raise ValueError("The requested split leaves an empty training or test set.")
    train_idx = np.asarray(selected_idx[:n_train], dtype=np.int64)
    test_idx = np.asarray(selected_idx[n_train:], dtype=np.int64)
    if not 0 < n_inducing <= n_train:
        raise ValueError(
            f"n_inducing must lie between 1 and n_train={n_train}; got {n_inducing}."
        )
    inducing_train_positions = np.sort(
        rng.choice(n_train, size=n_inducing, replace=False)
    ).astype(np.int64)
    inducing_idx = train_idx[inducing_train_positions]

    if np.intersect1d(train_idx, test_idx).size:
        raise RuntimeError("Training and test indices overlap.")

    np.savez_compressed(
        output_dir / "split_indices.npz",
        selected_idx=np.asarray(selected_idx, dtype=np.int64),
        train_idx=train_idx,
        test_idx=test_idx,
        inducing_idx=inducing_idx,
        inducing_train_positions=inducing_train_positions,
    )

    data = load_dataset(dataset_path)
    tags = data["deformation_tags"]
    deformation_grid = data["deformation_grid"]
    stresses = data["stresses"]
    if stresses.shape[0] != n_structures:
        raise ValueError(
            f"PC score rows ({n_structures}) != oracle rows ({stresses.shape[0]})."
        )

    fit_idx, fit_tags = select_fit_indices_0deg(tags, deformation_grid)
    design, row_labels = build_design_matrix_nominal(
        deformation_grid[fit_idx], tags[fit_idx],
    )
    selected_stresses = stresses[selected_idx][:, fit_idx]
    targets_selected = stack_stress_targets_nominal(
        selected_stresses, tags[fit_idx],
    )
    y_train = torch.from_numpy(targets_selected[:n_train].astype(np.float32)).to(device)
    y_test = torch.from_numpy(targets_selected[n_train:].astype(np.float32)).to(device)
    design_t = torch.from_numpy(design.astype(np.float32)).to(device).t().contiguous()
    stress_mean, stress_std = compute_stress_normalization(y_train)

    steps_per_epoch = (n_train + batch_size - 1) // batch_size
    train_config = GPTrainConfig(
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        n_r=n_r,
        S=mc_samples,
        log_every=log_every,
        # train_model() advances the scheduler after every minibatch. Convert
        # the intended epoch interval to optimizer steps so larger supervised
        # datasets retain the same restart timing as the manuscript workflow.
        first_restart=FIRST_RESTART_EPOCHS * steps_per_epoch,
    )

    config_payload = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "pca_path": str(Path(pca_path).resolve()),
        "output_dir": str(output_dir.resolve()),
        "seed": int(seed),
        "sample_size": int(sample_size),
        "train_fraction": float(train_fraction),
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "n_inducing": int(n_inducing),
        "inducing_points": "fixed_random_subset_of_training_set",
        "pc_counts": list(pc_counts),
        "batch_size": int(batch_size),
        "steps_per_epoch": int(steps_per_epoch),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "lmc_rank": int(n_r) - len(PARAM_NAMES),
        "n_tasks": len(PARAM_NAMES),
        "n_latent_processes": int(n_r),
        "mc_samples": int(mc_samples),
        "log_every": int(log_every),
        "first_restart_epochs": int(FIRST_RESTART_EPOCHS),
        "first_restart_optimizer_steps": int(train_config.first_restart),
        "kernel": train_config.kernel,
        "ard": bool(train_config.ard),
        "mean_type": train_config.mean_type,
        "input_normalization": "training_set_standardization",
        "stress_normalization": "training_set_row_standardization",
        "fit_paths": list(fit_tags),
        "n_stress_rows": int(targets_selected.shape[1]),
        "active_learning": False,
        # JSON has no comments; this maps the keys above to the symbols
        # used in the manuscript.
        "notation": {
            "pc_counts": "n_z values swept",
            "n_latent_processes": "n_r, latent GPs",
            "mc_samples": "S, Monte Carlo samples",
        },
    }
    (output_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2), encoding="utf-8",
    )

    pcs_selected = np.asarray(pcs_all[selected_idx], dtype=np.float32)
    records: list[dict] = []
    per_structure_mae: list[np.ndarray] = []

    print(
        f"Fixed split: sample={sample_size}, train={train_idx.size}, "
        f"test={test_idx.size}, inducing={n_inducing}"
    )
    print(f"PC counts: {pc_counts}")

    for n_pcs in pc_counts:
        print(f"\n=== Training GP with {n_pcs} PC score(s) ===")
        _set_reproducible(seed)

        z_train_raw = torch.from_numpy(pcs_selected[:n_train, :n_pcs]).to(device)
        z_test_raw = torch.from_numpy(pcs_selected[n_train:, :n_pcs]).to(device)
        z_mean = z_train_raw.mean(0, keepdim=True)
        z_std = z_train_raw.std(0, keepdim=True).clamp_min(1e-8)
        z_train = normalize_z(z_train_raw, z_mean, z_std)
        z_test = normalize_z(z_test_raw, z_mean, z_std)

        # The same fixed subset of training structures is used as inducing
        # points for every retained PC dimension.
        model = CorrelatedLatentSVGP(
            inducing_points=z_train[inducing_train_positions].clone(),
            num_tasks=len(PARAM_NAMES),
            ard=train_config.ard,
            # n_r counts every latent process; the rank argument is the
            # shared count, n_r - n_theta = 6 - 3 = 3.
            rank=train_config.n_r - len(PARAM_NAMES),
            mean_type=train_config.mean_type,
            kernel=train_config.kernel,
        ).to(device)

        start = time.perf_counter()
        model, noise_param, history = train_model(
            model=model,
            design_t=design_t,
            train_x=z_train,
            train_y=y_train,
            stress_mean=stress_mean,
            stress_std=stress_std,
            config=train_config,
            output_dir=output_dir,
            save_checkpoints=False,
        )
        train_seconds = time.perf_counter() - start

        model.eval()
        torch.manual_seed(seed)
        with torch.no_grad():
            output = model(z_test)
            latent_samples = output.rsample(torch.Size([mc_samples]))
            params_samples = F.softplus(latent_samples) + 1e-6
            stress_samples = torch.matmul(params_samples, design_t)
            stress_prediction = stress_samples.mean(dim=0)
            absolute_error = torch.abs(stress_prediction - y_test)
            test_mae = float(absolute_error.mean().item())
            test_rmse = float(torch.sqrt(torch.mean((stress_prediction - y_test) ** 2)).item())
            sample_mae = absolute_error.mean(dim=1).cpu().numpy()

        final_noise = float(F.softplus(noise_param).item())
        record = {
            "n_pcs": int(n_pcs),
            "n_train": int(train_idx.size),
            "n_test": int(test_idx.size),
            "n_inducing": int(n_inducing),
            "test_mae_mpa": test_mae,
            "test_rmse_mpa": test_rmse,
            "train_seconds": float(train_seconds),
            "final_noise": final_noise,
        }
        records.append(record)
        per_structure_mae.append(np.asarray(sample_mae, dtype=np.float64))

        np.savez_compressed(
            history_dir / f"training_history_pc_{n_pcs:02d}.npz",
            loss=np.asarray([h["loss"] for h in history], dtype=np.float64),
            elbo_per_point=np.asarray([h["elbo_per_point"] for h in history], dtype=np.float64),
            noise=np.asarray([h["noise"] for h in history], dtype=np.float64),
        )
        if save_models:
            save_checkpoint(
                model_dir / f"gp_model_pc_{n_pcs:02d}.pt",
                model,
                noise_param,
                epoch=epochs,
            )
            save_normalization_stats(
                model_dir / f"normalization_pc_{n_pcs:02d}.npz",
                z_mean,
                z_std,
                stress_mean,
                stress_std,
                row_labels,
            )

        _write_metrics(output_dir, records, per_structure_mae)
        print(
            f"PCs={n_pcs}: test MAE={test_mae:.6f} MPa, "
            f"test RMSE={test_rmse:.6f} MPa, training={train_seconds:.1f} s"
        )

    print(f"\nStudy complete. Results saved to {output_dir}")
    return records


if __name__ == "__main__":
    run_study()
