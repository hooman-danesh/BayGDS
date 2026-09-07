"""Plotting utilities for figures and summary tables.

Purpose:
    This module turns stored outputs from the descriptor, training, and
    inverse design steps into reproducible visualizations and summary tables.

Author:  Hooman Danesh <hooman.danesh@tu-braunschweig.de>, TU Braunschweig
License: MIT (see LICENSE). Released September 2026.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, NullFormatter

from data_utils import include_shear_for_tag, preferred_dirs, rotation_angle_from_tag

# ---------------------------------------------------------------------------
# Shared constants (LaTeX column width = 160 mm)
# ---------------------------------------------------------------------------
TEXTWIDTH_MM = 160
TICK_X = 4
TICK_Y = 4

COMP_COLORS = {"P11": "C0", "P22": "C2", "P12": "C1"}
COMP_LABELS = {"P11": r"$P_{11}$", "P22": r"$P_{22}$", "P12": r"$P_{12}$"}
COMP_SUB = {"P11": "11", "P22": "22", "P12": "12"}

LOADING_PATHS = ("tension_x", "off_x", "equibiaxial", "off_y", "tension_y")
PATH_LABELS = {
    "tension_x": "Tension-x",
    "off_x": "Off-x",
    "equibiaxial": "Equibiaxial",
    "off_y": "Off-y",
    "tension_y": "Tension-y",
}

HIST_COMBOS = [("P11",), ("P11", "P22"), ("P11", "P22", "P12")]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def setup_latex_style() -> None:
    """Apply publication-quality matplotlib style."""
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}" "\n" r"\usepackage{amssymb}",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def _save_fig_tight(
    fig: plt.Figure,
    path: Path,
    target_width_mm: float,
    target_height_mm: float | None = None,
    dpi: int = 300,
    pad_inches: float = 0.0,
) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = fig.get_tightbbox(renderer)
    if bbox is None:
        fig.savefig(path, dpi=dpi)
        fig.savefig(path.with_suffix(".png"), dpi=dpi)
        return
    old_size = fig.get_size_inches()
    target_w_in = float(target_width_mm) / 25.4
    if target_height_mm is None:
        scale = target_w_in / float(bbox.width)
        fig.set_size_inches(old_size * scale, forward=True)
    else:
        target_h_in = float(target_height_mm) / 25.4
        scale_w = target_w_in / float(bbox.width)
        scale_h = target_h_in / float(bbox.height)
        fig.set_size_inches(old_size[0] * scale_w, old_size[1] * scale_h, forward=True)
    fig.canvas.draw()
    fig.savefig(path, bbox_inches="tight", pad_inches=pad_inches, dpi=dpi)
    fig.savefig(
        path.with_suffix(".png"), bbox_inches="tight",
        pad_inches=pad_inches, dpi=dpi,
    )
    fig.set_size_inches(old_size, forward=True)


def _combo_label(combo: Sequence[str]) -> str:
    _MAP = {"P11": r"$P^{\star}_{11}$", "P22": r"$P^{\star}_{22}$", "P12": r"$P^{\star}_{12}$"}
    parts = [_MAP.get(c.upper(), c) for c in combo]
    if len(parts) == 1:
        return parts[0]
    return r"$\{$" + r", ".join(parts) + r"$\}$"


def _combo_key(combo: Sequence[str]) -> str:
    return "_".join(c.lower() for c in combo)


def _i1m3_from_F(F_2x2: np.ndarray) -> float:
    a, b = F_2x2[0]
    c, d = F_2x2[1]
    det2 = a * d - b * c
    if abs(det2) < 1e-12:
        return float("nan")
    f33 = 1.0 / det2
    C11 = a * a + c * c
    C22 = b * b + d * d
    C33 = f33 * f33
    return float(C11 + C22 + C33 - 3.0)


def _compute_invariants(F_2x2: np.ndarray, tag: str) -> Tuple[float, float, float]:
    a, b = F_2x2[0]
    c, d = F_2x2[1]
    det2 = a * d - b * c
    if abs(det2) < 1e-12:
        return float("nan"), float("nan"), float("nan")
    f33 = 1.0 / det2
    F = np.array([[a, b, 0.0], [c, d, 0.0], [0.0, 0.0, f33]])
    C = F.T @ F
    I1 = float(np.trace(C))
    n1, n2 = preferred_dirs(tag)
    I4 = float(n1 @ (C @ n1))
    I6 = float(n2 @ (C @ n2))
    return I1, I4, I6


def _parity_points(
    targets: np.ndarray,
    row_labels: Sequence[str],
    target_indices: np.ndarray,
    oracle_indices: np.ndarray,
    comp: str,
    summary: str,
) -> Tuple[np.ndarray, np.ndarray]:
    idxs = [i for i, lbl in enumerate(row_labels) if lbl == comp]
    if not idxs:
        return np.array([]), np.array([])
    targ_all, oracle_all = [], []
    use_mean = summary == "mean"
    for targ_idx, orc_idx in zip(target_indices, oracle_indices):
        targ_slice = targets[targ_idx][idxs]
        oracle_slice = targets[orc_idx][idxs]
        if use_mean:
            targ_all.append(float(np.mean(np.abs(targ_slice))))
            oracle_all.append(float(np.mean(np.abs(oracle_slice))))
        else:
            targ_all.append(targ_slice)
            oracle_all.append(oracle_slice)
    if use_mean:
        return np.array(targ_all), np.array(oracle_all)
    return np.concatenate(targ_all), np.concatenate(oracle_all)


def _r2(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    ss_res = float(np.sum((x - y) ** 2))
    ss_tot = float(np.sum((x - x.mean()) ** 2))
    return float("nan") if ss_tot == 0.0 else 1.0 - ss_res / ss_tot


def _best_indices_within_budget(data: dict, budget: int) -> np.ndarray:
    eval_indices = data.get("random_oracle_eval_indices", np.array([], dtype=object))
    oracle_mismatches = data.get(
        "random_oracle_eval_mismatches",
        data.get("random_oracle_eval_errs", np.array([], dtype=object)),
    )
    best_idx_fallback = np.array(data.get("random_best_indices", []), dtype=np.int64)
    out = []
    for i in range(best_idx_fallback.size):
        idxs = np.array(eval_indices[i]) if eval_indices.size else np.array([], dtype=np.int64)
        errs = np.array(oracle_mismatches[i]) if oracle_mismatches.size else np.array([], dtype=np.float64)
        if idxs.size and errs.size:
            take_n = min(int(budget), idxs.size, errs.size)
            if take_n > 0:
                best_local = int(np.argmin(errs[:take_n]))
                out.append(int(idxs[best_local]))
                continue
        out.append(int(best_idx_fallback[i]))
    return np.array(out, dtype=np.int64)


def _threshold_met(data: dict) -> np.ndarray:
    """Read threshold-met flags, supporting both old and new key names."""
    raw = data.get("random_oracle_eval_threshold_met",
                   data.get("random_oracle_eval_success", []))
    return np.array(raw, dtype=bool)


def _parity_indices(data: dict, budget: int) -> np.ndarray:
    """Feasible-first-then-best oracle index selection for parity plots."""
    eval_indices = data.get("random_oracle_eval_indices", np.array([], dtype=object))
    counts = np.array(data.get("random_oracle_eval_counts", []), dtype=np.int64)
    met = _threshold_met(data)
    best_idx_fallback = _best_indices_within_budget(data, budget)
    out = []
    for i in range(best_idx_fallback.size):
        if met.size and counts.size and met[i]:
            take_n = min(int(counts[i]), int(budget))
            if take_n > 0 and eval_indices.size:
                idxs = np.asarray(eval_indices[i]).ravel()
                if idxs.size >= take_n:
                    out.append(int(idxs[take_n - 1]))
                    continue
        out.append(int(best_idx_fallback[i]))
    return np.array(out, dtype=np.int64)


def _hit_rate(data: dict) -> float:
    """Threshold hit rate R_eta (%)."""
    met = _threshold_met(data)
    if met.size == 0:
        return float("nan")
    return float(np.mean(met) * 100.0)


# ---------------------------------------------------------------------------
# Fig 3: Sample designs (sample_designs.pdf)
# ---------------------------------------------------------------------------

def plot_sample_designs(
    structures: np.ndarray,
    out_path: Path,
    *,
    grid: Tuple[int, int] = (5, 5),
    seed: int = 1235,
) -> None:
    """Plot a grid of randomly selected microstructure unit cells.

    Manuscript: fig:sample_designs at 0.5 * textwidth.
    """
    dark_green = "#0b5d1e"
    cmap = ListedColormap(["white", dark_green])
    rng = np.random.default_rng(seed)
    n_pick = grid[0] * grid[1]
    pick_idx = rng.choice(len(structures), size=n_pick, replace=False)

    fig_w = float(TEXTWIDTH_MM) * 0.5 / 25.4
    fig, axes = plt.subplots(grid[0], grid[1], figsize=(fig_w, fig_w), squeeze=False)
    for ax, idx in zip(axes.flat, pick_idx):
        mask = (np.asarray(structures[int(idx)]) > 0).astype(int)
        ax.imshow(mask, cmap=cmap, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_frame_on(False)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.0)
    plt.savefig(out_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.0, dpi=300)
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Fig 4: Deformation sampling (deformation_sampling.pdf)
# ---------------------------------------------------------------------------

def plot_deformation_sampling(
    F_grid: np.ndarray,
    tags: np.ndarray,
    fit_idx: np.ndarray,
    out_path: Path,
    *,
    rotation: int = 0,
) -> None:
    """Pairwise invariant plots for sampled deformation states."""
    i1m3, i4m1_sq, i6m1_sq, path_tags = [], [], [], []
    for idx in fit_idx:
        tag = str(tags[idx])
        if rotation_angle_from_tag(tag) != rotation:
            continue
        I1, I4, I6 = _compute_invariants(F_grid[idx], tag)
        if not np.isfinite(I1):
            continue
        i1m3.append(I1 - 3.0)
        i4m1_sq.append((I4 - 1.0) ** 2)
        i6m1_sq.append((I6 - 1.0) ** 2)
        path_tags.append(str(tag).split("_rot")[0])
    i1m3 = np.asarray(i1m3)
    i4m1_sq = np.asarray(i4m1_sq)
    i6m1_sq = np.asarray(i6m1_sq)

    fig_w = float(TEXTWIDTH_MM) * 1.0 / 25.4
    fig_h = float(TEXTWIDTH_MM) * 0.35 / 25.4
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h), squeeze=False)
    ax1, ax2, ax3 = axes[0]

    cmap = plt.get_cmap("tab10")
    markers = ["o", "*", "^", "s", "P"]
    color_map = {name: cmap(i % 10) for i, name in enumerate(LOADING_PATHS)}
    marker_map = {name: markers[i % len(markers)] for i, name in enumerate(LOADING_PATHS)}

    unique_paths = [p for p in LOADING_PATHS if p in set(path_tags)]
    for name in unique_paths:
        mask = np.array([t == name for t in path_tags], dtype=bool)
        kw = dict(s=8, alpha=0.75, color=color_map[name], marker=marker_map[name])
        ax1.scatter(i1m3[mask], i4m1_sq[mask], **kw)
        ax2.scatter(i1m3[mask], i6m1_sq[mask], **kw)
        ax3.scatter(i4m1_sq[mask], i6m1_sq[mask], **kw)

    ax1.set_xlabel(r"$(I_1 - 3)$")
    ax1.set_ylabel(r"$(I_4 - 1)^2$")
    ax2.set_xlabel(r"$(I_1 - 3)$")
    ax2.set_ylabel(r"$(I_6 - 1)^2$")
    ax3.set_xlabel(r"$(I_4 - 1)^2$")
    ax3.set_ylabel(r"$(I_6 - 1)^2$")

    handles = [
        plt.Line2D([0], [0], marker=marker_map[n], linestyle="none",
                    markersize=6, color=color_map[n])
        for n in unique_paths
    ]
    labels = [PATH_LABELS.get(n, n) for n in unique_paths]
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=5,
                   frameon=False, bbox_to_anchor=(0.5, -0.04))
    for ax in (ax1, ax2, ax3):
        ax.tick_params(direction="in", which="both")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=TICK_X))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=TICK_X))
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_fig_tight(fig, out_path, TEXTWIDTH_MM, dpi=300)
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Fig 5: Active learning curve (al_curve.pdf)
# ---------------------------------------------------------------------------

def find_stopping_iteration(
    mae_hist: np.ndarray,
    L: int,
    epsilon: float,
) -> int | None:
    r"""Find the first AL iteration where convergence is reached.

    Convergence rule for the active learning stopping criterion:
    $$\Delta_t = \frac{1}{L}\sum_{t=T-L+1}^{T}
        \frac{|\mathrm{MAE}_t - \mathrm{MAE}_{t-1}|}{\mathrm{MAE}_{t-1}}$$

    Returns the iteration index where $\Delta_t \le \epsilon$,
    or None if the criterion is never met.
    """
    if L < 2 or mae_hist.size < L:
        return None
    for i in range(L, mae_hist.size + 1):
        window = mae_hist[i - L : i]
        prev = window[:-1]
        curr = window[1:]
        denom = np.where(prev == 0.0, np.nan, prev)
        rel_change = np.abs((curr - prev) / denom)
        avg_rel_change = float(np.nanmean(rel_change))
        if np.isfinite(avg_rel_change) and avg_rel_change < epsilon:
            return i
    return None


def suggest_model_checkpoint(
    stop_iter: int,
    n_initial: int,
    surrogate_dir: "Path",
    save_every: int = 10,
) -> "Path | None":
    """Return the closest saved checkpoint at or before the stopping iteration.

    New checkpoints are named by total training-data count, including the
    initial labeled set. The old iteration-based name is kept as a fallback for
    existing runs.
    """
    candidate = save_every * (stop_iter // save_every)
    candidate = max(candidate, save_every)
    for ckpt in (
        surrogate_dir / "al_model_selected.pt",
        surrogate_dir / f"al_model_ntrain_{n_initial + candidate}.pt",
        surrogate_dir / f"al_model_iter_{candidate}.pt",
    ):
        if ckpt.exists():
            return ckpt
    return None


def plot_al_curve(
    mae_hist: np.ndarray,
    n_initial: int,
    out_path: Path,
    *,
    L: int = 5,
    epsilon: float = 1e-3,
    max_iters: int | None = 210,
) -> int | None:
    r"""MAE vs. number of observed microstructures with stopping point.

    Parameters L and epsilon define the stopping rule
    Returns the stopping iteration (1-based) or None.
    """
    if max_iters is not None:
        mae_hist = mae_hist[:max_iters]
    x_obs = n_initial + np.arange(1, mae_hist.size + 1)

    stop_iter = find_stopping_iteration(mae_hist, L, epsilon)

    fig_w = float(TEXTWIDTH_MM) * 0.6 / 25.4
    fig_h = float(TEXTWIDTH_MM) * 0.4 / 25.4
    fig = plt.figure(figsize=(fig_w, fig_h))
    plt.plot(x_obs, mae_hist, "-")

    if stop_iter is not None and stop_iter <= mae_hist.size:
        stop_x = x_obs[stop_iter - 1]
        stop_y = mae_hist[stop_iter - 1]
        plt.plot([stop_x], [stop_y], "o", color="green", markersize=5, zorder=5)
        plt.annotate(
            f"{stop_x} observed\nmicrostructures",
            xy=(stop_x, stop_y),
            xytext=(-40, 12),
            textcoords="offset points",
            color="green",
            fontsize=8,
        )

    plt.xlabel(r"$\#$ Observed Microstructures")
    plt.ylabel("MAE [MPa]")
    ax = plt.gca()
    ax.tick_params(direction="in", which="both")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=TICK_X))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_fig_tight(fig, out_path, TEXTWIDTH_MM * 0.6, TEXTWIDTH_MM * 0.4, dpi=1200)
    plt.show()
    plt.close()

    return stop_iter


# ---------------------------------------------------------------------------
# PC dimension sensitivity curve (pc_dimension_curve.pdf)
# ---------------------------------------------------------------------------

def plot_pc_dimension_curve(
    n_pcs: np.ndarray,
    test_mae: np.ndarray,
    out_path: Path,
    *,
    selected_n_pcs: int = 6,
) -> None:
    """Plot held-out MAE against retained PC dimension.

    The sizing and visual style intentionally match :func:`plot_al_curve` so
    the two convergence figures can be placed consistently in the manuscript.
    """
    n_pcs = np.asarray(n_pcs, dtype=int).ravel()
    test_mae = np.asarray(test_mae, dtype=float).ravel()
    if n_pcs.size == 0 or n_pcs.shape != test_mae.shape:
        raise ValueError("n_pcs and test_mae must be non-empty arrays of equal length.")
    if not np.all(np.isfinite(test_mae)):
        raise ValueError("test_mae contains non-finite values.")

    setup_latex_style()
    fig_w = float(TEXTWIDTH_MM) * 0.6 / 25.4
    fig_h = float(TEXTWIDTH_MM) * 0.4 / 25.4
    fig = plt.figure(figsize=(fig_w, fig_h))
    plt.plot(
        n_pcs,
        test_mae,
        "-o",
        linewidth=1.5,
        markersize=4,
        markerfacecolor="C0",
        markeredgecolor="C0",
    )

    selected = np.flatnonzero(n_pcs == int(selected_n_pcs))
    if selected.size:
        idx = int(selected[0])
        plt.plot(
            [n_pcs[idx]], [test_mae[idx]], "o",
            color="green", markersize=5, zorder=5,
        )
        plt.annotate(
            f"{selected_n_pcs} PCs",
            xy=(n_pcs[idx], test_mae[idx]),
            xytext=(-28, 12),
            textcoords="offset points",
            color="green",
            fontsize=8,
        )

    plt.xlabel(r"$n_z$")
    plt.ylabel("MAE [MPa]")
    ax = plt.gca()
    ax.tick_params(direction="in", which="both")
    ax.set_xticks(n_pcs)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_fig_tight(fig, out_path, TEXTWIDTH_MM * 0.6, TEXTWIDTH_MM * 0.4, dpi=1200)
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Fig 6: Parameter distributions (param_distribution.pdf)
# ---------------------------------------------------------------------------

def plot_param_distribution(
    params: np.ndarray,
    out_path: Path,
    *,
    param_names: Sequence[str] = ("theta_I1", "theta_I4", "theta_I6"),
    bins: int = 50,
) -> None:
    """Histograms of constitutive parameter posterior means."""
    theta_labels = [
        r"$\mathbb{E}[\theta_{I_1}]$ [MPa]",
        r"$\mathbb{E}[\theta_{I_4}]$ [MPa]",
        r"$\mathbb{E}[\theta_{I_6}]$ [MPa]",
    ]
    fig_w = float(TEXTWIDTH_MM) * 1.0 / 25.4
    fig_h = float(TEXTWIDTH_MM) * 0.35 / 25.4
    fig = plt.figure(figsize=(fig_w, fig_h))
    for i in range(len(param_names)):
        ax = plt.subplot(1, len(param_names), i + 1)
        ax.hist(params[:, i], bins=bins, alpha=0.8, color=f"C{i}")
        ax.set_xlabel(theta_labels[i], fontsize=10)
        ax.set_ylabel(r"\# Microstructures", fontsize=10)
        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="in")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=TICK_X))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=TICK_X))
        ax.set_box_aspect(1)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_fig_tight(fig, out_path, TEXTWIDTH_MM, dpi=1200)
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Fig 7: Target stress responses (target_stress.pdf)
# ---------------------------------------------------------------------------

def plot_target_stress(
    targets: np.ndarray,
    tags_subset: np.ndarray,
    F_grid_subset: np.ndarray,
    target_indices: np.ndarray,
    include_shear: bool,
    shear_rotations: Sequence[int],
    row_mask: np.ndarray,
    out_path: Path,
) -> None:
    """3-row x 5-col grid of target stress vs (I1-3) per loading path."""
    ncols = len(LOADING_PATHS)
    nrows = 3
    fig_w = 160.0 / 25.4
    fig_h = 160.0 * (3.0 / 6.0) / 25.4
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False, sharey="row")
    path_colors = {name: plt.get_cmap("tab10")(i % 10) for i, name in enumerate(LOADING_PATHS)}

    comp_rows: Dict[str, list] = {c: [] for c in ("P11", "P22", "P12")}
    comp_paths: Dict[str, list] = {c: [] for c in ("P11", "P22", "P12")}
    comp_x: Dict[str, list] = {c: [] for c in ("P11", "P22", "P12")}
    keep = set(int(x) for x in row_mask) if row_mask is not None else None
    full_row = 0
    masked_row = -1
    for j, tag in enumerate(tags_subset):
        base_tag = str(tag).split("_rot")[0]
        xval = _i1m3_from_F(F_grid_subset[j])
        if keep is None or full_row in keep:
            masked_row += 1
            comp_rows["P11"].append(masked_row)
            comp_paths["P11"].append(base_tag)
            comp_x["P11"].append(xval)
        full_row += 1
        if keep is None or full_row in keep:
            masked_row += 1
            comp_rows["P22"].append(masked_row)
            comp_paths["P22"].append(base_tag)
            comp_x["P22"].append(xval)
        full_row += 1
        if include_shear_for_tag(tag, include_shear, shear_rotations):
            if keep is None or full_row in keep:
                masked_row += 1
                comp_rows["P12"].append(masked_row)
                comp_paths["P12"].append(base_tag)
                comp_x["P12"].append(xval)
            full_row += 1

    # Equibiaxial abscissa from P11 row (same deformation samples as P22); P12 uses this
    # for the zero-response panel so (I1-3) range and color match the other rows.
    x_p_equibiaxial: np.ndarray | None = None
    p11_paths_list = comp_paths["P11"]
    p11_xs_arr = np.asarray(comp_x["P11"], dtype=float)
    if p11_paths_list:
        mask_eq_p11 = np.array([p == "equibiaxial" for p in p11_paths_list], dtype=bool)
        if np.any(mask_eq_p11):
            x_raw = p11_xs_arr[mask_eq_p11]
            x_p_rel = x_raw - np.min(x_raw)
            order_eq = np.argsort(x_p_rel)
            x_p_equibiaxial = x_p_rel[order_eq]

    for r, comp in enumerate(("P11", "P22", "P12")):
        rows = comp_rows[comp]
        paths = comp_paths[comp]
        xs = np.asarray(comp_x[comp], dtype=float)
        if not rows:
            for c in range(ncols):
                axes[r][c].set_visible(False)
            continue
        for c, path in enumerate(LOADING_PATHS):
            ax = axes[r][c]
            mask = np.array([p == path for p in paths], dtype=bool)
            # Equibiaxial path has no shear block: P12 is identically zero — draw y=0 line
            # instead of leaving an empty panel.
            if not np.any(mask):
                if (
                    comp == "P12"
                    and path == "equibiaxial"
                    and x_p_equibiaxial is not None
                ):
                    eq_color = path_colors["equibiaxial"]
                    z = np.zeros_like(x_p_equibiaxial, dtype=float)
                    for _ in target_indices:
                        ax.plot(
                            x_p_equibiaxial,
                            z,
                            color=eq_color,
                            alpha=0.05,
                            linewidth=0.5,
                        )
                    if r != nrows - 1:
                        ax.tick_params(labelbottom=False)
                    ax.tick_params(direction="in", which="both")
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=TICK_X))
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=TICK_Y))
                    if r == 0:
                        ax.set_title(PATH_LABELS.get(path, path))
                    if c != 0:
                        ax.tick_params(labelleft=False)
                else:
                    ax.set_visible(False)
                continue
            x_p = xs[mask] - np.min(xs[mask])
            order = np.argsort(x_p)
            idxs = np.array(rows, dtype=int)[mask]
            for t_idx in target_indices:
                y = targets[int(t_idx)][idxs]
                ax.plot(x_p[order], y[order], color=path_colors[path],
                        alpha=0.05, linewidth=0.5)
            if r != nrows - 1:
                ax.tick_params(labelbottom=False)
            ax.tick_params(direction="in", which="both")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=TICK_X))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=TICK_Y))
            if r == 0:
                ax.set_title(PATH_LABELS.get(path, path))
            if c != 0:
                ax.tick_params(labelleft=False)
        comp_star = {"P11": r"$P^{\star}_{11}$", "P22": r"$P^{\star}_{22}$", "P12": r"$P^{\star}_{12}$"}
        axes[r][0].text(-0.25, 0.5, comp_star[comp], transform=axes[r][0].transAxes,
                        ha="right", va="center", fontsize=10, rotation=90)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1, bottom=0.08, right=0.88, left=0.08, top=0.95)
    fig.canvas.draw()
    eq_col = LOADING_PATHS.index("equibiaxial") if "equibiaxial" in LOADING_PATHS else 2
    x0 = axes[-1][eq_col].get_position().x0
    x1 = axes[-1][eq_col].get_position().x1
    fig.text(0.5 * (x0 + x1), -0.02, r"$(I_1 - 3)$", ha="center", va="bottom", fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_fig_tight(fig, out_path, 160.0, dpi=300)
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Fig 8: Hit rate (hit_rate.pdf)
# ---------------------------------------------------------------------------

def plot_hit_rate(
    hit_rate_curves: Dict[str, list[float]],
    out_path: Path,
    *,
    eval_steps: Sequence[int] | None = None,
) -> None:
    """Threshold hit rate R_eta vs oracle budget E_max."""
    if eval_steps is None:
        eval_steps = [1] + list(range(10, 210, 10))
    fig_w = float(TEXTWIDTH_MM) * 0.6 / 25.4
    fig_h = float(TEXTWIDTH_MM) * 0.4 / 25.4
    plt.figure(figsize=(fig_w, fig_h))
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    colors = ["C0", "C1", "C3", "C4", "C5", "C6", "C7"]
    for i, (label, series) in enumerate(hit_rate_curves.items()):
        plt.plot(eval_steps, series, linewidth=1.5, label=label, linestyle="-",
                 marker=markers[i % len(markers)], markersize=3,
                 color=colors[i % len(colors)])
    plt.xlabel(r"$E_\mathrm{max}$")
    plt.ylabel(r"$R_{\eta}^{(\le E_{\mathrm{max}})}$ [\%]")
    plt.ylim(25, 101.0)
    plt.xlim(0.9, 210.0)
    plt.xscale("log")
    plt.legend(fontsize=10, ncol=1, frameon=False, bbox_to_anchor=(0.55, 0.82))
    ax = plt.gca()
    ax.tick_params(direction="in", which="both")
    ax.set_xticks([1, 10, 20, 100, 200])
    ax.set_yticks([25, 50, 75, 90, 100])
    ax.set_xticklabels(["1", "10", "20", "100", "200"])
    ax.set_yticklabels(["25", "50", "75", "90", "100"])
    ax.axhline(90, color="green", linestyle="--", linewidth=1.0)
    ax.axvline(20, color="green", linestyle="--", linewidth=1.0)
    fig = plt.gcf()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_fig_tight(fig, out_path, TEXTWIDTH_MM * 0.6, TEXTWIDTH_MM * 0.4, dpi=300)
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Full library inverse design baselines (inverse_baseline_comparison.pdf)
# ---------------------------------------------------------------------------

def plot_inverse_baseline_comparison(
    eval_steps: np.ndarray,
    proposed_hit_rate: np.ndarray,
    bo_ei_hit_rate: np.ndarray,
    random_hit_rate: np.ndarray,
    out_path: Path,
    *,
    best_nmae_percentiles: np.ndarray,
    percentile_levels: np.ndarray,
    raw_pca_bo_hit_rate: np.ndarray | None = None,
    raw_pca_bo_best_nmae_percentiles: np.ndarray | None = None,
    eval_counts_by_method: np.ndarray | None = None,
    raw_pca_bo_eval_counts: np.ndarray | None = None,
) -> None:
    """Compare search methods by hit rate, accuracy, and oracle efficiency."""
    eval_steps = np.asarray(eval_steps, dtype=int).ravel()
    include_raw_pca = raw_pca_bo_hit_rate is not None
    if include_raw_pca != (raw_pca_bo_best_nmae_percentiles is not None):
        raise ValueError(
            "raw_pca_bo_hit_rate and raw_pca_bo_best_nmae_percentiles "
            "must either both be supplied or both be omitted."
        )
    proposed_label = "Proposed framework"
    engineered_bo_label = (
        "BO--EI (proposed features)" if include_raw_pca else "BO--EI"
    )
    raw_pca_bo_label = "BO--EI (raw-image PCs)"
    random_label = "Random search"
    curves = {
        proposed_label: np.asarray(proposed_hit_rate, dtype=float).ravel(),
        engineered_bo_label: np.asarray(bo_ei_hit_rate, dtype=float).ravel(),
    }
    if include_raw_pca:
        curves[raw_pca_bo_label] = np.asarray(
            raw_pca_bo_hit_rate, dtype=float,
        ).ravel()
    curves[random_label] = np.asarray(random_hit_rate, dtype=float).ravel()
    if eval_steps.size == 0:
        raise ValueError("eval_steps must be non-empty.")
    for label, values in curves.items():
        if values.shape != eval_steps.shape:
            raise ValueError(
                f"{label} hit rate curve must have shape {eval_steps.shape}; "
                f"got {values.shape}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{label} hit rate curve contains non-finite values.")
        if np.any(np.diff(values) < 0.0):
            raise ValueError(f"{label} hit rate curve must be nondecreasing.")

    percentile_levels = np.asarray(percentile_levels, dtype=float).ravel()
    best_nmae_percentiles = np.asarray(
        best_nmae_percentiles, dtype=float,
    )
    base_expected_shape = (3, percentile_levels.size, eval_steps.size)
    if best_nmae_percentiles.shape != base_expected_shape:
        raise ValueError(
            "best_nmae_percentiles must have shape "
            f"{base_expected_shape}; got {best_nmae_percentiles.shape}."
        )
    percentile_by_label = {
        proposed_label: best_nmae_percentiles[0],
        engineered_bo_label: best_nmae_percentiles[1],
        random_label: best_nmae_percentiles[2],
    }
    if include_raw_pca:
        raw_percentiles = np.asarray(
            raw_pca_bo_best_nmae_percentiles, dtype=float,
        )
        raw_expected_shape = (percentile_levels.size, eval_steps.size)
        if raw_percentiles.shape != raw_expected_shape:
            raise ValueError(
                "raw_pca_bo_best_nmae_percentiles must have shape "
                f"{raw_expected_shape}; got {raw_percentiles.shape}."
            )
        percentile_by_label[raw_pca_bo_label] = raw_percentiles
    method_labels = tuple(curves)
    best_nmae_percentiles = np.stack([
        percentile_by_label[label] for label in method_labels
    ])
    percentile_rows = {}
    for level in (25.0, 50.0, 75.0):
        matches = np.flatnonzero(np.isclose(percentile_levels, level))
        if matches.size != 1:
            raise ValueError(
                f"Accuracy bands require exactly one {level:g}th percentile."
            )
        percentile_rows[level] = int(matches[0])
    if not np.all(np.isfinite(best_nmae_percentiles)) or np.any(
        best_nmae_percentiles <= 0.0
    ):
        raise ValueError("Best-so-far nMAE summaries must be finite and positive.")
    q25 = best_nmae_percentiles[:, percentile_rows[25.0], :]
    median = best_nmae_percentiles[:, percentile_rows[50.0], :]
    q75 = best_nmae_percentiles[:, percentile_rows[75.0], :]
    if np.any(q25 > median) or np.any(median > q75):
        raise ValueError("Best-so-far nMAE percentiles are not ordered.")
    if np.any(np.diff(best_nmae_percentiles, axis=2) > 1e-10):
        raise ValueError("Best-so-far nMAE summaries must be nonincreasing.")

    include_paired = eval_counts_by_method is not None
    if include_paired != (raw_pca_bo_eval_counts is not None):
        raise ValueError(
            "eval_counts_by_method and raw_pca_bo_eval_counts must either "
            "both be supplied or both be omitted."
        )
    if include_paired and not include_raw_pca:
        raise ValueError("Paired efficiency requires the raw-PCA BO results.")
    paired_counts = None
    if include_paired:
        base_counts = np.asarray(eval_counts_by_method, dtype=np.int64)
        raw_counts = np.asarray(raw_pca_bo_eval_counts, dtype=np.int64).ravel()
        if base_counts.ndim != 2 or base_counts.shape[0] != 3:
            raise ValueError(
                "eval_counts_by_method must have shape (3, n_targets) in "
                "Proposed, BO--EI, Random order."
            )
        if raw_counts.shape != (base_counts.shape[1],):
            raise ValueError(
                "raw_pca_bo_eval_counts must match the target axis of "
                "eval_counts_by_method."
            )
        paired_counts = np.stack([
            base_counts[0], base_counts[1], raw_counts, base_counts[2],
        ])
        if np.any(paired_counts < 1):
            raise ValueError("Oracle evaluation counts must be positive.")

    setup_latex_style()
    target_height_mm = float(TEXTWIDTH_MM) * (0.79 if include_paired else 0.45)
    fig_w = float(TEXTWIDTH_MM) / 25.4
    fig_h = target_height_mm / 25.4
    if include_paired:
        fig = plt.figure(figsize=(fig_w, fig_h))
        grid = fig.add_gridspec(
            2, 2, height_ratios=(1.48, 1.00), hspace=0.92, wspace=0.42,
        )
        ax_hit = fig.add_subplot(grid[0, 0])
        ax_accuracy = fig.add_subplot(grid[0, 1])
        ax_paired = fig.add_subplot(grid[1, :])
        # Panel (c) has long row labels. Shift the upper pair together so the
        # complete composition, rather than only the axes frames, is centered.
        upper_row_shift = 0.086
        for upper_axis in (ax_hit, ax_accuracy):
            position = upper_axis.get_position()
            upper_axis.set_position([
                position.x0 - upper_row_shift,
                position.y0,
                position.width,
                position.height,
            ])
    else:
        fig, (ax_hit, ax_accuracy) = plt.subplots(
            1, 2, figsize=(fig_w, fig_h),
        )
        ax_paired = None

    styles = {
        "Proposed framework": ("C2", "D"),
        "BO--EI": ("C1", "s"),
        "BO--EI (proposed features)": ("C1", "s"),
        "BO--EI (raw-image PCs)": ("C3", "^"),
        "Random search": ("C0", "o"),
    }
    markevery = sorted(set([0, 4, 9, 19, 29, 39, eval_steps.size - 1]))
    for label, values in curves.items():
        color, marker = styles[label]
        ax_hit.plot(
            eval_steps,
            values,
            color=color,
            linewidth=1.6,
            marker=marker,
            markersize=4,
            markevery=markevery,
            label=label,
        )

    ax_hit.set_xlabel(r"$E_{\mathrm{hit}}$")
    ax_hit.set_ylabel(r"$R_{\eta}^{(\le E_{\mathrm{hit}})}$ [\%]")
    ax_hit.set_xlim(float(eval_steps.min()), float(eval_steps.max()))
    ax_hit.set_ylim(0.0, 100.0)
    ax_hit.set_xticks([1, 10, 20, 30, 40, 50])
    ax_hit.set_yticks([0, 25, 50, 75, 100])
    ax_hit.tick_params(direction="in", which="both")
    ax_hit.text(
        -0.19, 1.05, r"\textbf{(a)}", transform=ax_hit.transAxes,
        ha="left", va="bottom", clip_on=False,
    )

    for method_idx, label in enumerate(method_labels):
        color, marker = styles[label]
        ax_accuracy.fill_between(
            eval_steps,
            q25[method_idx],
            q75[method_idx],
            color=color,
            alpha=0.14,
            linewidth=0.0,
            zorder=1,
        )
        ax_accuracy.plot(
            eval_steps,
            median[method_idx],
            color=color,
            linewidth=1.6,
            marker=marker,
            markersize=4,
            markevery=markevery,
            label=label,
            zorder=2,
        )
    ax_accuracy.axhline(
        5.0, color="0.35", linestyle="--", linewidth=1.0, zorder=0,
    )
    ax_accuracy.annotate(
        r"$\eta=5\%$",
        xy=(float(eval_steps.max()), 5.0),
        xytext=(-3, 4),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color="0.35",
        fontsize=8,
    )
    ax_accuracy.set_xlabel(r"$E_{\mathrm{hit}}$")
    ax_accuracy.set_ylabel(
        r"$\underset{1\leq e\leq E_{\mathrm{hit}}}{\min}\;"
        r"\overline{\mathrm{nMAE}}_{e}$ [\%]"
    )
    ax_accuracy.set_xlim(float(eval_steps.min()), float(eval_steps.max()))
    ax_accuracy.set_xticks([1, 10, 20, 30, 40, 50])
    ax_accuracy.set_yscale("log")
    ax_accuracy.set_ylim(2.0, 60.0)
    ax_accuracy.set_yticks([2, 5, 10, 20, 50])
    ax_accuracy.set_yticklabels(["2", "5", "10", "20", "50"])
    ax_accuracy.yaxis.set_minor_formatter(NullFormatter())
    ax_accuracy.tick_params(direction="in", which="both")
    ax_accuracy.text(
        -0.19, 1.05, r"\textbf{(b)}", transform=ax_accuracy.transAxes,
        ha="left", va="bottom", clip_on=False,
    )
    accuracy_handles, accuracy_labels = ax_accuracy.get_legend_handles_labels()
    iqr_handle = Patch(
        facecolor="0.45", edgecolor="none", alpha=0.20,
    )
    method_legend = fig.legend(
        accuracy_handles,
        accuracy_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.385, 0.465),
        ncol=len(accuracy_labels),
        fontsize=8.0 if include_raw_pca else 9.0,
        columnspacing=1.0,
        handlelength=1.8,
    )
    method_legend.set_zorder(10)
    iqr_legend = fig.legend(
        [iqr_handle],
        [r"25th--75th percentile"],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.385, 0.425),
        ncol=1,
        fontsize=8.0 if include_raw_pca else 9.0,
        handlelength=1.8,
    )
    iqr_legend.set_zorder(10)

    if ax_paired is not None and paired_counts is not None:
        maximum_budget = int(eval_steps.max())
        successful = paired_counts <= maximum_budget
        common_success = np.all(successful, axis=0)
        n_common = int(np.sum(common_success))
        if n_common == 0:
            raise ValueError("No target is successful for every method.")
        conditional_counts = paired_counts[:, common_success].astype(float)
        y_positions = np.arange(len(method_labels), dtype=float)
        method_colors = tuple(styles[label][0] for label in method_labels)
        boxplot = ax_paired.boxplot(
            [conditional_counts[row] for row in range(len(method_labels))],
            positions=y_positions,
            orientation="horizontal",
            widths=0.56,
            whis=1.5,
            showmeans=True,
            showfliers=False,
            patch_artist=True,
            manage_ticks=False,
            medianprops={"linewidth": 1.7},
            whiskerprops={"linewidth": 1.2},
            capprops={"linewidth": 1.2},
            meanprops={
                "marker": "D",
                "markersize": 4.5,
                "markeredgewidth": 0.8,
            },
        )
        for row, color in enumerate(method_colors):
            boxplot["boxes"][row].set_facecolor(color)
            boxplot["boxes"][row].set_edgecolor(color)
            boxplot["boxes"][row].set_alpha(0.20)
            boxplot["medians"][row].set_color(color)
            boxplot["means"][row].set_markerfacecolor(color)
            boxplot["means"][row].set_markeredgecolor(color)
            for item in boxplot["whiskers"][2 * row:2 * row + 2]:
                item.set_color(color)
            for item in boxplot["caps"][2 * row:2 * row + 2]:
                item.set_color(color)

        ax_paired.set_yticks(y_positions)
        ax_paired.set_yticklabels(method_labels)
        ax_paired.invert_yaxis()
        ax_paired.set_xscale("log")
        ax_paired.set_xlim(0.9, 55.0)
        ax_paired.set_xticks([1, 2, 5, 10, 20, 50])
        ax_paired.set_xticklabels(["1", "2", "5", "10", "20", "50"])
        ax_paired.minorticks_off()
        ax_paired.set_xlabel(r"$E_{\eta}$", labelpad=0)
        ax_paired.tick_params(direction="in", which="both")
        ax_paired.tick_params(axis="y", length=0)
        ax_paired.text(
            -0.19, 1.05, r"\textbf{(c)}", transform=ax_paired.transAxes,
            ha="left", va="bottom",
            clip_on=False,
        )

    if not include_paired:
        plt.tight_layout(w_pad=1.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    outer_pad_inches = 0.06 if include_paired else 0.0
    outer_pad_mm = outer_pad_inches * 25.4
    save_width_mm = float(TEXTWIDTH_MM) * (0.935 if include_paired else 1.0)
    save_height_mm = (
        target_height_mm - 2.0 * outer_pad_mm
        if include_paired else target_height_mm
    )
    _save_fig_tight(
        fig, out_path, save_width_mm, save_height_mm, dpi=1200,
        pad_inches=outer_pad_inches,
    )
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close()

# ---------------------------------------------------------------------------
# Uncertainty weight sensitivity (lambda_sensitivity.pdf)
# ---------------------------------------------------------------------------

def plot_lambda_sensitivity(
    eval_steps: np.ndarray,
    gamma_values: np.ndarray,
    hit_rate_percent: np.ndarray,
    out_path: Path,
) -> None:
    r"""Plot threshold-hit rate against the normalized uncertainty weight.

    ``gamma_values`` parameterizes the score as
    :math:`\mu_i + \gamma\lambda_0\sigma_i`, where ``gamma=1`` is the
    automatically balanced score used by the inverse design workflow.
    """
    eval_steps = np.asarray(eval_steps, dtype=int).ravel()
    gamma_values = np.asarray(gamma_values, dtype=float).ravel()
    hit_rate_percent = np.asarray(hit_rate_percent, dtype=float)
    expected_shape = (gamma_values.size, eval_steps.size)
    if eval_steps.size == 0 or gamma_values.size == 0:
        raise ValueError("eval_steps and gamma_values must be non-empty.")
    if hit_rate_percent.shape != expected_shape:
        raise ValueError(
            f"hit_rate_percent must have shape {expected_shape}; "
            f"got {hit_rate_percent.shape}."
        )
    if not np.all(np.isfinite(hit_rate_percent)):
        raise ValueError("hit_rate_percent contains non-finite values.")

    budgets = np.asarray([1, 10, 20, 50], dtype=int)
    budget_columns = []
    for budget in budgets:
        matches = np.flatnonzero(eval_steps == budget)
        if matches.size != 1:
            raise ValueError(f"eval_steps must contain E={budget} exactly once.")
        budget_columns.append(int(matches[0]))
    deterministic = np.flatnonzero(np.isclose(gamma_values, 0.0))
    if deterministic.size != 1:
        raise ValueError("gamma_values must contain lambda/lambda_0=0 exactly once.")
    selected_hit_rate = hit_rate_percent[:, budget_columns]
    hit_rate_improvement = (
        selected_hit_rate - selected_hit_rate[int(deterministic[0])]
    )

    setup_latex_style()
    fig_w = float(TEXTWIDTH_MM) * 0.6 / 25.4
    fig_h = float(TEXTWIDTH_MM) * 0.4 / 25.4
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    markers = ["o", "s", "^", "D"]
    colors = ["C0", "C1", "C3", "C2"]
    gamma_positions = np.arange(gamma_values.size, dtype=float)
    for i, budget in enumerate(budgets):
        ax.plot(
            gamma_positions,
            hit_rate_improvement[:, i],
            color=colors[i % len(colors)],
            linewidth=1.6,
            marker=markers[i % len(markers)],
            markersize=4,
            label=rf"$E_{{\mathrm{{hit}}}}={budget}$",
        )

    automatic = np.flatnonzero(np.isclose(gamma_values, 1.0))
    if automatic.size != 1:
        raise ValueError("gamma_values must contain lambda/lambda_0=1 exactly once.")
    ax.axhline(0.0, color="0.7", linestyle="-", linewidth=0.8, zorder=0)
    ax.axvline(
        float(automatic[0]),
        ymin=0.0,
        ymax=0.75,
        color="0.45",
        linestyle="--",
        linewidth=1.0,
        zorder=0,
    )
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\Delta R_{\eta}^{(\le E_{\mathrm{hit}})}$ [\%]")
    ax.set_xlim(-0.15, float(gamma_positions.max()) + 0.15)
    delta_min = float(np.min(hit_rate_improvement))
    delta_max = float(np.max(hit_rate_improvement))
    delta_span = max(delta_max - delta_min, 1.0)
    padding = max(0.15, 0.08 * delta_span)
    ax.set_ylim(min(-padding, delta_min - padding), delta_max + padding)
    ax.tick_params(direction="in", which="both")
    ax.set_xticks(gamma_positions)
    ax.set_xticklabels([f"{value:g}" for value in gamma_values])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    legend = ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1.0,
        loc="upper left",
        ncol=2,
    )
    legend.set_zorder(10)
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.22, top=0.98)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_fig_tight(
        fig, out_path, TEXTWIDTH_MM * 0.6, TEXTWIDTH_MM * 0.4, dpi=1200,
    )
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Fig 9: Oracle counts + budget-limited nMAE (oracle_counts.pdf)
# ---------------------------------------------------------------------------

def plot_oracle_counts(
    data_map: Dict[Tuple[str, ...], dict],
    targets_map: Dict[Tuple[str, ...], Tuple[np.ndarray, list[str]]],
    out_path: Path,
    *,
    max_evals: int = 50,
    marker: int = 20,
) -> None:
    """2-row panel: oracle eval histograms (top), budget-limited nMAE distribution (bottom)."""
    combos = HIST_COMBOS
    fig_w = float(TEXTWIDTH_MM) * 0.333 / 25.4
    fig_h = float(TEXTWIDTH_MM) * 0.35 / 25.4
    fig, axes = plt.subplots(2, len(combos), figsize=(fig_w * len(combos), fig_h * 2), squeeze=False)

    for ax, combo in zip(axes[0], combos):
        data = data_map[combo]
        counts = np.array(data.get("random_oracle_eval_counts", []), dtype=np.int64)
        met = _threshold_met(data)
        if counts.size == 0:
            ax.set_visible(False)
            continue
        bins = np.arange(1, max_evals + 2, dtype=float)
        counts_met = counts[met]
        counts_budget = counts[~met]
        if counts_budget.size:
            counts_budget = np.full_like(counts_budget, max_evals + 1)
        met_hist = ax.hist(counts_met, bins=bins, alpha=1.0, color="C9", label="Feasible")
        budget_hist = ax.hist(counts_budget, bins=bins, alpha=1.0, color="C4", label="Budget-limited")
        ax.axvline(marker, color="#3a8a5f", linestyle="--", linewidth=1.0)
        budget_pct = float(np.mean(~met) * 100.0)
        hit_marker_pct = float(np.mean((counts <= marker) & met) * 100.0)
        max_count = max(
            float(np.max(met_hist[0])) if len(met_hist[0]) else 0.0,
            float(np.max(budget_hist[0])) if len(budget_hist[0]) else 0.0,
        )
        y_top = max_count + 50.0
        text_y = 0.15 * y_top
        ax.text(max_evals + 1, text_y,
                rf"$1 - R_{{\eta}}^{{(\le {max_evals})}} = {budget_pct:.1f}\%$",
                ha="center", va="bottom", fontsize=10, color="C4", rotation=90)
        ax.text(marker - 5, text_y,
                rf"$R_\eta^{{(\le {marker})}}$={hit_marker_pct:.1f}\%",
                ha="center", va="bottom", fontsize=10, color="#3a8a5f", rotation=90)
        ax.set_xlabel(r"$E_{\eta}$")
        ax.set_ylabel(r"\# Microstructures")
        ax.set_title(f"Target: {_combo_label(combo)}")
        ax.set_xlim(-2, max_evals + 6)
        ax.set_ylim(0, y_top)
        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="in")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=TICK_X + 2))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=TICK_Y, integer=True))

    for ax, combo in zip(axes[1], combos):
        data = data_map[combo]
        targets, row_labels = targets_map[combo]
        met = _threshold_met(data)
        target_indices = np.array(data.get("random_target_indices", []), dtype=np.int64)
        oracle_indices_list = data.get("random_oracle_eval_indices", None)
        if met.size == 0 or target_indices.size == 0 or oracle_indices_list is None:
            ax.set_visible(False)
            continue

        omega_p = {
            "P11": float(np.array(data.get("oracle_stop_weight_p11", [1.0]))[0]),
            "P22": float(np.array(data.get("oracle_stop_weight_p22", [1.0]))[0]),
            "P12": float(np.array(data.get("oracle_stop_weight_p12", [1.0]))[0]),
        }
        comp_indices = {
            comp: [i for i, lbl in enumerate(row_labels) if lbl == comp]
            for comp in ("P11", "P22", "P12")
        }
        active_comps = [c for c, idx in comp_indices.items() if idx and omega_p.get(c, 0) > 0]

        def _comp_nmae(t_vec, o_vec, comp):
            idx = comp_indices.get(comp, [])
            if not idx:
                return float("nan")
            m = float(np.mean(np.abs(t_vec[idx])))
            return float("inf") if m <= 1e-12 else 100.0 * float(np.mean(np.abs(t_vec[idx] - o_vec[idx]))) / m

        def _w_nmae(t_vec, o_vec):
            vals = {c: _comp_nmae(t_vec, o_vec, c) for c in active_comps}
            total = sum(omega_p.get(c, 0) * v for c, v in vals.items() if np.isfinite(v) and omega_p.get(c, 0) > 0)
            denom = sum(omega_p.get(c, 0) for c, v in vals.items() if np.isfinite(v) and omega_p.get(c, 0) > 0)
            return float("nan") if denom <= 0 else total / denom

        best_weighted = np.empty(target_indices.shape[0], dtype=float)
        for i, (t_idx, orc_seq) in enumerate(zip(target_indices, oracle_indices_list)):
            orc_seq = np.asarray(orc_seq).ravel()[:max_evals]
            if orc_seq.size == 0:
                best_weighted[i] = np.nan
                continue
            t_vec = targets[int(t_idx)]
            best_weighted[i] = float(np.nanmin([_w_nmae(t_vec, targets[int(o)]) for o in orc_seq]))

        vals = best_weighted[~met]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            ax.set_visible(False)
            continue
        ax.hist(vals, bins=11, color="C4", alpha=1.0)
        ax.set_xlim(float(np.min(vals)) - 0.5, float(np.max(vals)) + 0.5)
        ax.set_xlabel(r"$\overline{\mathrm{nMAE}}$ [\%]")
        ax.set_ylabel(r"\# Microstructures")
        ax.set_title(f"Target: {_combo_label(combo)}")
        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="in")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=TICK_X + 1))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=TICK_Y, integer=True))
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%d"))

    handles, labels = axes[0][0].get_legend_handles_labels()
    left = axes[0][0].get_position().x0
    bottom = axes[0][0].get_position().y0
    fig.legend(handles, labels, loc="upper left", ncol=1, frameon=False,
               bbox_to_anchor=(left - 0.05, bottom - 0.06), bbox_transform=fig.transFigure, fontsize=10)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_fig_tight(fig, out_path, TEXTWIDTH_MM, dpi=300)
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Fig 10: Parity plots (parity.pdf)
# ---------------------------------------------------------------------------

def plot_parity(
    data_map: Dict[Tuple[str, ...], dict],
    target_cache: Dict[Tuple[str, ...], Tuple[np.ndarray, list[str]]],
    out_path: Path,
    *,
    budget: int = 50,
) -> None:
    """3x3 parity grid: rows = component combos, columns = P11/P22/P12."""
    row_combos = [("P11",), ("P11", "P22"), ("P11", "P22", "P12")]
    col_comps = ["P11", "P22", "P12"]
    fig_w = 160.0 / 25.4
    fig_h = 160.0 / 25.4
    fig, axes = plt.subplots(3, 3, figsize=(fig_w, fig_h), squeeze=False)
    tick_fmt = mpl.ticker.FormatStrFormatter("%.1f")

    row_active_axes: list[list] = []
    for r, combo in enumerate(row_combos):
        data = data_map.get(combo)
        if data is None:
            row_active_axes.append([])
            continue
        targets, row_labels = target_cache[combo]
        target_indices = np.array(data.get("random_target_indices", []), dtype=np.int64)
        if target_indices.size == 0:
            row_active_axes.append([])
            continue
        oracle_indices = _parity_indices(data, budget)
        met = _threshold_met(data)
        active: list = []
        for c, comp in enumerate(col_comps):
            ax = axes[r][c]
            x_all, y_all = _parity_points(targets, row_labels, target_indices, oracle_indices, comp, "mean")
            if x_all.size == 0:
                ax.axis("off")
                continue
            active.append(ax)
            x_feas, y_feas = _parity_points(targets, row_labels, target_indices[met], oracle_indices[met], comp, "mean")
            x_budg, y_budg = _parity_points(targets, row_labels, target_indices[~met], oracle_indices[~met], comp, "mean")
            color = COMP_COLORS.get(comp, "C0")
            ax.scatter(x_feas, y_feas, s=1, alpha=0.75, color=color)
            ax.scatter(x_budg, y_budg, s=1, alpha=0.75, color=color)
            lo = float(min(x_all.min(), y_all.min()))
            hi = float(max(x_all.max(), y_all.max()))
            ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1)
            r2_all = _r2(x_all, y_all)
            ax.text(0.98, 0.1, rf"$R^2_{{{COMP_SUB.get(comp, '')}}}$={r2_all:.3f}",
                    transform=ax.transAxes, va="bottom", ha="right", fontsize=10)
            sub = COMP_SUB.get(comp, "")
            ax.set_xlabel(rf"$\overline{{|P^{{\star}}_{{{sub}}}|}}$ [MPa]", fontsize=10)
            ax.set_ylabel(rf"$\overline{{|P_{{{sub}}}(\mathbf{{M}}^{{\star}})|}}$ [MPa]", fontsize=10)
            ax.set_aspect("equal", adjustable="box")
            ax.tick_params(direction="in", which="both")
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3, min_n_ticks=3))
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3, min_n_ticks=3))
            ax.xaxis.set_major_formatter(tick_fmt)
            ax.yaxis.set_major_formatter(tick_fmt)
        row_active_axes.append(active)

    fig.subplots_adjust(hspace=0.75, wspace=0.05)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    row_titles = [f"Target: {_combo_label(c)}" for c in row_combos]
    for r_idx, (title, axes_list) in enumerate(zip(row_titles, row_active_axes)):
        if not axes_list:
            continue
        bboxes = [ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
                  for ax in axes_list if ax.get_tightbbox(renderer) is not None]
        if not bboxes:
            continue
        row_top = max(b.y1 for b in bboxes)
        row_left = axes_list[0].get_position().x0
        row_right = axes_list[-1].get_position().x1
        sep_y = row_top + 0.02
        fig.add_artist(mpl.lines.Line2D([row_left, row_right], [sep_y, sep_y],
                                         transform=fig.transFigure, color="black", linewidth=1.0))
        fig.text(0.5 * (row_left + row_right), sep_y + 0.01, title,
                 ha="center", va="bottom", fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_fig_tight(fig, out_path, 160.0, dpi=300)
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Table 1: Summary table
# ---------------------------------------------------------------------------

def write_summary_table(rows: list[dict[str, str]], out_dir: Path) -> None:
    """Write inverse design summary as CSV and Markdown."""
    out_dir.mkdir(parents=True, exist_ok=True)
    headers = ["combo", "hit_rate_pct", "hit_rate_le_marker_pct", "r2_p11", "r2_p22", "r2_p12"]
    csv_lines = [",".join(headers)]
    for row in rows:
        csv_lines.append(",".join(str(row.get(h, "")) for h in headers))
    (out_dir / "inverse_design_summary.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    md_lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        md_lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    (out_dir / "inverse_design_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
