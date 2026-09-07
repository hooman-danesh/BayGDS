#!/usr/bin/env python3
"""Active learning loop for surrogate training.

Purpose:
    This module starts from a small labeled set, repeatedly trains the GP
    surrogate, measures uncertainty on the unlabeled pool, and selects the
    next structure to evaluate with the oracle.

Main parameters:
    - DATASET_PATH, PCA_PATH, OUTPUT_DIR: core file locations.
    - SEED: random seed for reproducibility.
    - N_INIT: initial number of labeled structures.
    - T_max: maximum number of active learning acquisitions.
    - TEST_HOLDOUT_SIZE: optional validation hold-out size.
    - BATCH_SIZE, EPOCHS, LEARNING_RATE, N_R, S, and LOG_EVERY:
      surrogate-training controls.

Author:  Hooman Danesh <hooman.danesh@tu-braunschweig.de>, TU Braunschweig
License: MIT (see LICENSE). Released September 2026.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np
import torch
import torch.nn.functional as F

from data_utils import (
    build_design_matrix_nominal,
    load_dataset,
    load_pca,
    select_fit_indices_0deg,
    stack_stress_targets_nominal,
)
from oracle import call_oracle_batch
from gp_core import (
    CorrelatedLatentSVGP,
    PARAM_NAMES,
    GPTrainConfig,
    compute_stress_normalization,
    normalize_z,
    save_checkpoint,
    save_normalization_stats,
    train_model,
)


# === Knobs ===
DATASET_PATH = Path("data/oracle/oracle_0deg.npz")
PCA_PATH = Path("data/pca/pc_scores.npz")
OUTPUT_DIR = Path("output/surrogate/active_learning")

SEED = 0
N_INIT = 10           # |T_0|, size of the initial labeled set
T_max = 210           # T_max, maximum active learning acquisitions
TEST_HOLDOUT_SIZE = 500

BATCH_SIZE = 1024
EPOCHS = 500
LEARNING_RATE = 0.01
N_R = 6               # n_r, total latent processes
S = 64                # S, Monte Carlo samples for the predictive moments
LOG_EVERY = 10

# The model retained for inverse design is selected with the same convergence
# rule used by the reproduction notebook.
STOPPING_L = 5             # L, sliding-window length of the stopping rule
STOPPING_EPSILON = 1.0e-3  # epsilon, stopping threshold on the windowed MAE change


def _log_ticks(low: float, high: float) -> List[float]:
    """Round tick values covering ``[low, high]`` on a logarithmic axis.

    Chosen from a fixed ladder of readable numbers so the labels stay short
    (``0.6``, ``1``, ``2``) instead of falling back to scientific notation.
    """
    ladder = [
        0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8,
        1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0,
        10.0, 15.0, 20.0, 30.0, 50.0, 100.0,
    ]
    ticks = [value for value in ladder if low * 0.98 <= value <= high * 1.02]
    return ticks if len(ticks) >= 3 else [low, 0.5 * (low + high), high]


def _plot_mae_history(
    eval_mae_hist: Sequence[float],
    selected_iteration: int | None,
) -> None:
    """Redraw the running active learning diagnostic plot.

    This is a progress monitor written after every acquisition, not a
    manuscript figure; ``reproduction/plotting.py`` draws the published
    version. It is kept deliberately plain so it stays readable while the run
    is still in progress and the curve is short.

    The abscissa counts observed structures rather than iterations, so it can
    be read against the labeled-set sizes quoted elsewhere, and the checkpoint
    chosen by the stopping rule is marked once it has been selected.
    """
    history = np.asarray(eval_mae_hist, dtype=float)
    if history.size == 0:
        return
    observed = N_INIT + np.arange(1, history.size + 1)

    # Build the figure on an explicit Agg canvas rather than through pyplot,
    # so importing this module never changes a caller's Matplotlib backend.
    fig = Figure(figsize=(6.0, 3.6), constrained_layout=True)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.plot(
        observed, history,
        color="#1f4e79", linewidth=1.4,
        marker="o", markersize=2.6, markerfacecolor="white",
        markeredgewidth=0.7, zorder=3,
    )

    if selected_iteration is not None and 0 < selected_iteration <= history.size:
        x_sel = N_INIT + selected_iteration
        y_sel = history[selected_iteration - 1]
        ax.axvline(x_sel, color="#2e7d32", linewidth=1.0, linestyle="--", zorder=2)
        ax.plot(
            [x_sel], [y_sel], marker="o", markersize=7,
            color="#2e7d32", zorder=4,
        )
        # Keep the label inside the axes: anchor it on whichever side of the
        # marker has room, so a late selection does not run off the figure.
        on_right = x_sel > observed[0] + 0.6 * (observed[-1] - observed[0])
        ax.annotate(
            f"selected: {x_sel} observed\nMAE = {y_sel:.3g} MPa",
            xy=(x_sel, y_sel),
            xytext=(-10 if on_right else 10, 22),
            textcoords="offset points",
            ha="right" if on_right else "left",
            color="#2e7d32", fontsize=9,
        )

    ax.set_yscale("log")
    # A log axis spanning well under a decade otherwise picks up Matplotlib's
    # scientific-notation formatter and prints labels such as "6 x 10^-1".
    # Explicit round ticks over the observed range keep the labels plain.
    ax.set_yticks(_log_ticks(history.min(), history.max()))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())

    ax.set_xlabel("Observed microstructures")
    ax.set_ylabel("Held-out MAE [MPa]")
    ax.set_title("Active learning progress", fontsize=10, loc="left")
    ax.grid(True, which="major", linewidth=0.5, alpha=0.35)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.savefig(OUTPUT_DIR / "mae_history.png", dpi=200)


def _lhs_points(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Latin hypercube points in [0,1]^d."""
    cut = np.linspace(0, 1, n + 1)
    u = rng.random((n, d))
    a, b = cut[:n], cut[1 : n + 1]
    pts = u * (b - a)[:, None] + a[:, None]
    for j in range(d):
        rng.shuffle(pts[:, j])
    return pts


def _lhs_select_indices(
    available_idx: np.ndarray, points: np.ndarray, n_pick: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Select indices via LHS: sample targets in [0,1]^d, pick nearest structure per target.
    Min-max scales points internally (for LHS only; not used for GP training)."""
    pts_min = points.min(axis=0)
    pts_max = points.max(axis=0)
    span = np.where(pts_max > pts_min, pts_max - pts_min, 1.0)
    scaled = (points - pts_min) / span

    lhs_targets = _lhs_points(n_pick, points.shape[1], rng)
    available = set(int(i) for i in available_idx.tolist())
    selected = []
    for tgt in lhs_targets:
        if not available:
            break
        avail_list = np.fromiter(available, dtype=int)
        d2 = np.sum((scaled[avail_list] - tgt) ** 2, axis=1)
        pick = int(avail_list[np.argmin(d2)])
        selected.append(pick)
        available.remove(pick)
    return np.array(selected, dtype=int), np.array(sorted(available), dtype=int)


def _acquisition_alpha(var_diag: torch.Tensor) -> torch.Tensor:
    """Acquisition score: log-determinant of the diagonal predictive variance."""
    return torch.sum(torch.log(var_diag + 1e-8), dim=1)


def _expand_variational_state(state: dict, new_inducing: torch.Tensor) -> dict:
    key_ind = "variational_strategy.base_variational_strategy.inducing_points"
    old_ind = state.get(key_ind)
    if old_ind is None:
        return state
    old_m, new_m = int(old_ind.shape[-2]), int(new_inducing.shape[-2])
    if new_m == old_m:
        state[key_ind] = new_inducing
        return state
    if new_m < old_m:
        state[key_ind] = new_inducing
        for k in list(state.keys()):
            if k.endswith("variational_mean"):
                state[k] = state[k][..., :new_m].contiguous()
            if k.endswith("chol_variational_covar"):
                state[k] = state[k][..., :new_m, :new_m].contiguous()
        return state
    pad = new_m - old_m
    state[key_ind] = new_inducing
    for k in list(state.keys()):
        if k.endswith("variational_mean"):
            v = state[k]
            pad_shape = list(v.shape)
            pad_shape[-1] = pad
            state[k] = torch.cat([v, torch.zeros(pad_shape, device=v.device, dtype=v.dtype)], dim=-1)
        elif k.endswith("chol_variational_covar"):
            v = state[k]
            pad_shape = list(v.shape)
            pad_shape[-2] = pad
            v_pad_rows = torch.zeros(pad_shape, device=v.device, dtype=v.dtype)
            v_exp = torch.cat([v, v_pad_rows], dim=-2)
            pad_shape_cols = list(v_exp.shape)
            pad_shape_cols[-1] = pad
            v_pad_cols = torch.zeros(pad_shape_cols, device=v.device, dtype=v.dtype)
            v_exp = torch.cat([v_exp, v_pad_cols], dim=-1)
            diag = torch.eye(new_m, device=v.device, dtype=v.dtype)
            if v_exp.dim() == 3:
                diag = diag.unsqueeze(0).expand(v_exp.shape[0], -1, -1)
            v_exp = torch.maximum(v_exp, 1e-4 * diag)
            state[k] = v_exp
    return state


def _find_stopping_iteration(
    mae_hist: np.ndarray,
    window: int = STOPPING_L,
    tolerance: float = STOPPING_EPSILON,
) -> int | None:
    """Return the first one-based iteration satisfying the AL rule."""
    values = np.asarray(mae_hist, dtype=float).ravel()
    if window < 2 or values.size < window:
        return None
    for i in range(window, values.size + 1):
        window_values = values[i - window : i]
        previous = window_values[:-1]
        current = window_values[1:]
        denominator = np.where(previous == 0.0, np.nan, previous)
        relative_change = np.abs((current - previous) / denominator)
        mean_change = float(np.nanmean(relative_change))
        if np.isfinite(mean_change) and mean_change < tolerance:
            return i
    return None


def run_active_learning():
    rng = np.random.default_rng(SEED)
    device = torch.device("cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # This runner starts a new acquisition sequence rather than resuming an
    # interrupted one. Remove only the two canonical files that could
    # otherwise be mistaken for outputs of the new run.
    for stale_path in (
        OUTPUT_DIR / "al_model_selected.pt",
        OUTPUT_DIR / "al_state.npz",
    ):
        if stale_path.exists():
            stale_path.unlink()

    config = GPTrainConfig(
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        n_r=N_R,
        S=S,
        log_every=LOG_EVERY,
    )

    data = load_dataset(DATASET_PATH)
    tags = data["deformation_tags"]
    F_grid = data["deformation_grid"]
    stresses = data["stresses"]

    fit_idx, _ = select_fit_indices_0deg(tags, F_grid)
    design, row_labels = build_design_matrix_nominal(F_grid[fit_idx], tags[fit_idx])
    targets = stack_stress_targets_nominal(stresses[:, fit_idx], tags[fit_idx])

    pcs = load_pca(PCA_PATH, n_components=6)
    n_pca_components = pcs.shape[1]
    if pcs.shape[0] != targets.shape[0]:
        raise ValueError(f"PC scores {pcs.shape[0]} != targets {targets.shape[0]}")

    z_raw = torch.from_numpy(pcs.astype(np.float32)).to(device)
    y_full = torch.from_numpy(targets.astype(np.float32)).to(device)

    n_struct = pcs.shape[0]
    all_idx = np.arange(n_struct)
    rng.shuffle(all_idx)
    I_lab, I_unlab = _lhs_select_indices(all_idx, pcs, N_INIT, rng)
    initial_I_lab = I_lab.copy()

    I_test = np.array([], dtype=int)
    if TEST_HOLDOUT_SIZE > 0 and I_unlab.size > 0:
        holdout_size = min(TEST_HOLDOUT_SIZE, I_unlab.size)
        picked, I_unlab = _lhs_select_indices(I_unlab, pcs, holdout_size, rng)
        I_test = picked
        print(f"Hold-out: {I_test.size} samples. Pool: {I_unlab.size}")

    z_mean = z_raw.mean(0, keepdim=True)
    z_std = z_raw.std(0, keepdim=True).clamp_min(1e-8)
    z_norm = normalize_z(z_raw, z_mean, z_std)

    design_t = torch.from_numpy(design.astype(np.float32)).to(device).t()
    I_lab_t = torch.as_tensor(I_lab, device=device)
    stress_mean, stress_std = compute_stress_normalization(
        call_oracle_batch(y_full, I_lab_t),
    )

    # These statistics are independent of the active learning iteration and
    # therefore need to be written only once for the whole run.
    save_normalization_stats(
        OUTPUT_DIR / "normalization_stats.npz",
        z_mean, z_std, stress_mean, stress_std, row_labels,
    )

    z_holdout = z_norm[I_test] if I_test.size else torch.empty((0, z_norm.shape[1]), device=device)
    y_holdout = call_oracle_batch(y_full, I_test) if I_test.size else torch.empty((0, y_full.shape[1]), device=device)

    eval_mae_hist: List[float] = []
    pool_uncert_hist: List[float] = []
    picked_hist: List[int] = []
    selected_train_idx: np.ndarray | None = None
    selected_iteration: int | None = None
    selected_model_path = OUTPUT_DIR / "al_model_selected.pt"
    current_model_path = OUTPUT_DIR / "al_model_current.pt"
    state_path = OUTPUT_DIR / "al_state.npz"
    prev_state = None
    prev_noise = None

    json.dump(
        {
            "dataset_path": str(DATASET_PATH),
            "pca_path": str(PCA_PATH),
            "output_dir": str(OUTPUT_DIR),
            "n_pca_components": n_pca_components,
            "n_init": N_INIT,
            "T_max": T_max,
            "iters": T_max,
            "seed": SEED,
            "test_holdout_size": TEST_HOLDOUT_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            # n_r counts every latent process; the model splits it into
            # shared latents plus one private latent per task.
            "lmc_rank": N_R - len(PARAM_NAMES),
            "n_tasks": len(PARAM_NAMES),
            "n_latent_processes": N_R,
            "mc_samples": S,
            "log_every": LOG_EVERY,
            "stopping_window": STOPPING_L,
            "stopping_tolerance": STOPPING_EPSILON,
            "selected_checkpoint": selected_model_path.name,
            "current_checkpoint": current_model_path.name,
        },
        open(OUTPUT_DIR / "al_config.json", "w"),
        indent=2,
    )

    def _save_active_learning_state(acquired_count: int) -> None:
        """Overwrite the single compact active learning state file."""
        selected = (
            np.asarray(selected_train_idx, dtype=np.int64)
            if selected_train_idx is not None
            else np.empty(0, dtype=np.int64)
        )
        state_tmp = state_path.with_name(state_path.name + ".tmp.npz")
        np.savez_compressed(
            state_tmp,
            eval_mae_hist=np.array(eval_mae_hist),
            pool_uncert_hist=np.array(pool_uncert_hist),
            picked_indices=np.array(picked_hist, dtype=np.int64),
            initial_train_idx=np.asarray(initial_I_lab, dtype=np.int64),
            current_train_idx=np.asarray(I_lab, dtype=np.int64),
            final_train_idx=I_lab,
            test_holdout_idx=I_test,
            acquired_count=np.array([int(acquired_count)], dtype=np.int64),
            selected_train_idx=selected,
            selected_iteration=np.array(
                [-1 if selected_iteration is None else int(selected_iteration)],
                dtype=np.int64,
            ),
            selected_n_train=np.array([selected.size], dtype=np.int64),
        )
        state_tmp.replace(state_path)

    for acquired_count in range(T_max + 1):
        print(f"[acquired {acquired_count}/{T_max}] train={len(I_lab)} pool={len(I_unlab)}")
        I_lab_t = torch.as_tensor(I_lab, device=device)
        I_unlab_t = torch.as_tensor(I_unlab, device=device)
        z_train_raw = z_raw[I_lab_t]
        y_train = call_oracle_batch(y_full, I_lab_t)
        z_pool = z_norm[I_unlab_t]

        inducing_raw = z_train_raw
        inducing_norm = normalize_z(inducing_raw, z_mean, z_std)
        reuse = prev_state is not None
        if reuse:
            model = CorrelatedLatentSVGP(
                inducing_points=inducing_norm,
                num_tasks=len(PARAM_NAMES),
                ard=config.ard,
                # n_r counts every latent process. The model is built from a
                # rank of shared latents plus one private latent per task, so
                # the rank passed here is n_r - n_theta = 6 - 3 = 3.
                rank=config.n_r - len(PARAM_NAMES),
                mean_type=config.mean_type,
                kernel=config.kernel,
            ).to(device)
            state = _expand_variational_state(prev_state, inducing_norm)
            model.load_state_dict(state)
        else:
            model = CorrelatedLatentSVGP(
                inducing_points=inducing_norm,
                num_tasks=len(PARAM_NAMES),
                ard=config.ard,
                # n_r counts every latent process. The model is built from a
                # rank of shared latents plus one private latent per task, so
                # the rank passed here is n_r - n_theta = 6 - 3 = 3.
                rank=config.n_r - len(PARAM_NAMES),
                mean_type=config.mean_type,
                kernel=config.kernel,
            ).to(device)

        model, noise_param, _ = train_model(
            model=model,
            design_t=design_t,
            train_x=inducing_norm,
            train_y=y_train,
            stress_mean=stress_mean,
            stress_std=stress_std,
            config=config,
            output_dir=OUTPUT_DIR,
            init_noise_log_param=prev_noise,
            save_checkpoints=False,
        )
        prev_state = model.state_dict()
        prev_noise = noise_param.detach()

        # Overwrite the running checkpoint after every acquisition. The
        # selected checkpoint is written only once, when the stopping rule
        # fires, so without this an interrupted run would leave no model at
        # all. It pairs with the current_train_idx recorded in al_state.npz
        # further down this iteration.
        save_checkpoint(
            current_model_path, model, noise_param, epoch=acquired_count,
        )

        model.eval()

        eval_z = z_holdout if I_test.size else z_pool
        eval_y = y_holdout if I_test.size else call_oracle_batch(y_full, I_unlab_t)
        mae_list = []
        with torch.no_grad():
            for i in range(0, eval_z.shape[0], BATCH_SIZE):
                zb = eval_z[i : i + BATCH_SIZE]
                yb = eval_y[i : i + BATCH_SIZE]
                out = model(zb)
                samples = out.rsample(torch.Size([S]))
                params_pred = F.softplus(samples) + 1e-6
                pred = torch.matmul(params_pred, design_t).mean(dim=0)
                mae_list.append(torch.mean(torch.abs(pred - yb), dim=1))
        eval_mae = float(torch.cat(mae_list).mean().item()) if mae_list else 0.0
        eval_mae_hist.append(eval_mae)
        print(f"  {'holdout' if I_test.size else 'pool'} MAE: {eval_mae:.4e}")

        # Save only the first model satisfying the established stopping rule.
        # The loop continues so the complete learning curve is still recorded.
        expected_selected_size = None
        if selected_train_idx is None:
            stop_iter = _find_stopping_iteration(
                np.asarray(eval_mae_hist), STOPPING_L, STOPPING_EPSILON,
            )
            if stop_iter is not None:
                # The plotted stopping index is one-based and corresponds to
                # the labeled-set size used by the selected checkpoint.
                expected_selected_size = N_INIT + int(stop_iter)
            if (
                stop_iter is not None
                and len(I_lab) == expected_selected_size
            ):
                selected_iteration = int(stop_iter)
                selected_train_idx = np.asarray(I_lab, dtype=np.int64).copy()
                save_checkpoint(
                    selected_model_path,
                    model,
                    noise_param,
                    epoch=acquired_count,
                )
                print(
                    f"  Selected model at iteration {selected_iteration} "
                    f"({selected_train_idx.size} labeled structures)."
                )

        _plot_mae_history(eval_mae_hist, selected_iteration)

        if I_unlab.size == 0:
            _save_active_learning_state(acquired_count)
            print("Pool empty.")
            break

        score_list = []
        with torch.no_grad():
            for i in range(0, z_pool.shape[0], BATCH_SIZE):
                zb = z_pool[i : i + BATCH_SIZE]
                out = model(zb)
                samples = out.rsample(torch.Size([S]))
                params_uc = F.softplus(samples) + 1e-6
                y_uc = torch.matmul(params_uc, design_t)
                var_diag = y_uc.var(dim=0, unbiased=False)
                score_list.append(_acquisition_alpha(var_diag))
        score_pool = torch.cat(score_list)
        pool_uncert_hist.append(float(score_pool.mean().item()))

        _save_active_learning_state(acquired_count)

        if acquired_count == T_max:
            break

        add_k = min(1, I_unlab.shape[0])
        top_idx = torch.topk(score_pool, k=add_k).indices.cpu().numpy()
        new_idx = I_unlab[top_idx]
        picked_hist.extend(new_idx.tolist())
        I_lab = np.concatenate([I_lab, new_idx])
        mask = np.ones(I_unlab.shape[0], dtype=bool)
        mask[top_idx] = False
        I_unlab = I_unlab[mask]
    # If convergence was not reached, retain the final model as a documented
    # fallback so downstream scripts always have one canonical checkpoint.
    if selected_train_idx is None:
        selected_iteration = -1
        selected_train_idx = np.asarray(I_lab, dtype=np.int64).copy()
        save_checkpoint(
            selected_model_path,
            model,
            noise_param,
            epoch=T_max,
        )
        print(
            "Stopping criterion was not reached; retained the final model "
            f"({selected_train_idx.size} labeled structures)."
        )
    _save_active_learning_state(T_max)
    save_normalization_stats(
        OUTPUT_DIR / "normalization_stats.npz",
        z_mean, z_std, stress_mean, stress_std, row_labels,
    )
    print(f"Selected checkpoint: {selected_model_path}")
    print(f"Active learning state: {state_path}")
    print("Active learning complete.")


if __name__ == "__main__":
    run_active_learning()
