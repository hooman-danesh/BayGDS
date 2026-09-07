"""Regenerate all manuscript figures and summary tables.

Run from any directory with::

    python reproduction/reproduce.py

The script follows the same order and uses the same plotting functions as
``reproduce.ipynb``.  Figures are written to ``output/figures`` and a text log
with the selected model, cached-run checks, tables, and saved output paths is
written to ``output/reproduction.log`` as well as echoed to the terminal.

Author:  Hooman Danesh <hooman.danesh@tu-braunschweig.de>, TU Braunschweig
License: MIT (see LICENSE). Released September 2026.
"""
from __future__ import annotations

import importlib
import json
import logging
import sys
from pathlib import Path

import os

# Keep Matplotlib's cache inside the repository when this script is launched
# on a machine where the default user cache is unavailable or read-only.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib

# The notebook uses an inline backend.  A non-interactive backend gives the
# standalone script the same file output without requiring a display.
matplotlib.use("Agg")


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repository root regardless of the launch directory."""
    start = (start or Path(__file__).resolve().parent).resolve()
    for candidate in (start, *start.parents):
        if (
            (candidate / "src").is_dir()
            and (candidate / "experiments").is_dir()
            and (candidate / "reproduction").is_dir()
        ):
            return candidate
    raise RuntimeError(
        "Could not locate repository root. Run this script inside the BayGDS repository."
    )


ROOT = find_repo_root()
sys.path.insert(0, str(ROOT.resolve()))
sys.path.insert(0, str((ROOT / "src").resolve()))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data_utils import load_dataset, load_pca, select_fit_indices_0deg
from gp_core import (
    build_svgp_from_config,
    normalize_z,
)

# Import a fresh copy from this repository, avoiding a stale editable install.
sys.modules.pop("reproduction.plotting", None)
plotting = importlib.import_module("reproduction.plotting")
if Path(plotting.__file__).resolve() != (ROOT / "reproduction/plotting.py").resolve():
    raise RuntimeError(f"Imported plotting module from unexpected path: {plotting.__file__}")

import inverse_design as inv
from experiments.baselines.bo_ei_proposed_features import load_split_results


FIGURES = ROOT / "output/figures"
FIGURES.mkdir(parents=True, exist_ok=True)
LOG_PATH = ROOT / "output/reproduction.log"

#: Seed for the Monte-Carlo posterior sampling behind ``param_distribution``.
#: Every other figure is a deterministic function of the cached results.
MC_SEED = 0

#: Seed for drawing the 1,000 inverse design targets.  It must match
#: ``target_sample_seed`` in ``output/inverse_design/sweep_settings.json``,
#: which is cross-checked at run time.
TARGET_SEED = 42


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("baygds.reproduction")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


LOGGER = configure_logging()


def log_saved(path: Path) -> None:
    """Log the PDF and PNG written by the shared plotting helper."""
    LOGGER.info("Saved plot: %s", path.relative_to(ROOT))
    png_path = path.with_suffix(".png")
    if png_path.exists():
        LOGGER.info("Saved plot: %s", png_path.relative_to(ROOT))


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of these files exist: " + ", ".join(str(p) for p in paths))


def load_run(combo_key, e_max):
    """Load one cached inverse design batch."""
    path = (
        ROOT
        / "output/inverse_design/runs"
        / plotting._combo_key(combo_key)
        / f"eval_{e_max:04d}"
        / "inverse_design.npz"
    )
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def main() -> None:
    setup_latex_style = plotting.setup_latex_style
    setup_latex_style()
    LOGGER.info("Repository root: %s", ROOT)
    LOGGER.info("Figure output directory: %s", FIGURES)

    # ------------------------------------------------------------------
    # 1. Sample designs
    # ------------------------------------------------------------------
    structures_data = np.load(ROOT / "data/structures/structures.npz", allow_pickle=True)
    structures = structures_data["structures"]
    LOGGER.info("Loaded %d structures of size %dx%d", structures.shape[0], structures.shape[1], structures.shape[2])
    plotting.plot_sample_designs(structures, FIGURES / "sample_designs.pdf", seed=1235)
    log_saved(FIGURES / "sample_designs.pdf")

    # ------------------------------------------------------------------
    # 2. PCA dimensionality and deformation sampling
    # ------------------------------------------------------------------
    pca_data = np.load(ROOT / "data/pca/pc_scores.npz", allow_pickle=True)
    pca_scores = pca_data["PCA_components"]
    LOGGER.info("Loaded PC scores: shape=%s", pca_scores.shape)

    pc_study_dir = ROOT / "output/studies/pc_dimension"
    pc_metrics = np.load(pc_study_dir / "metrics.npz")
    plotting.plot_pc_dimension_curve(
        pc_metrics["n_pcs"], pc_metrics["test_mae_mpa"],
        FIGURES / "pc_dimension_curve.pdf", selected_n_pcs=6,
    )
    log_saved(FIGURES / "pc_dimension_curve.pdf")

    oracle_0deg = load_dataset(ROOT / "data/oracle/oracle_0deg.npz")
    tags_0 = oracle_0deg["deformation_tags"]
    F_grid_0 = oracle_0deg["deformation_grid"]
    fit_idx_0, _ = select_fit_indices_0deg(tags_0, F_grid_0)
    plotting.plot_deformation_sampling(
        F_grid_0, tags_0, fit_idx_0,
        FIGURES / "deformation_sampling.pdf", rotation=0,
    )
    log_saved(FIGURES / "deformation_sampling.pdf")

    # ------------------------------------------------------------------
    # 3. Active learning curve and selected surrogate checkpoint
    # ------------------------------------------------------------------
    surrogate_dir = ROOT / "output/surrogate/active_learning"
    al_config_path = surrogate_dir / "al_config.json"
    with open(al_config_path, encoding="utf-8") as handle:
        train_cfg = json.load(handle)
    if "T_max" in train_cfg:
        t_max = int(train_cfg["T_max"])
    elif "iters" in train_cfg:
        t_max = int(train_cfg["iters"])
    else:
        raise KeyError(f"Missing T_max/iters in {al_config_path}")

    state_path = surrogate_dir / "al_state.npz"
    splits_path = surrogate_dir / "al_splits.npz"
    if state_path.exists():
        state = np.load(state_path, allow_pickle=True)
        n_initial = int(np.array(state["initial_train_idx"]).size)
    elif splits_path.exists():
        splits = np.load(splits_path, allow_pickle=True)
        n_initial = int(np.array(splits["initial_train_idx"]).size)
    elif "n_init" in train_cfg:
        n_initial = int(train_cfg["n_init"])
    else:
        raise KeyError(f"Missing n_init in {al_config_path} and {splits_path} not found")

    history_path = first_existing(
        surrogate_dir / "al_state.npz",
        surrogate_dir / f"al_history_ntrain_{n_initial + t_max}.npz",
        surrogate_dir / f"al_history_iter_{t_max:04d}.npz",
        surrogate_dir / "al_history.npz",
    )
    hist = np.load(history_path, allow_pickle=True)
    mae_hist = np.array(hist.get("eval_mae_hist", []), dtype=float)
    stop_iter = plotting.plot_al_curve(
        mae_hist, n_initial, FIGURES / "al_curve.pdf",
        L=5, epsilon=1.0e-3, max_iters=t_max,
    )
    log_saved(FIGURES / "al_curve.pdf")

    model_path = first_existing(
        surrogate_dir / "al_model_selected.pt",
        surrogate_dir / f"al_model_ntrain_{n_initial + t_max}.pt",
        surrogate_dir / f"al_model_iter_{t_max}.pt",
    )
    if stop_iter is not None:
        LOGGER.info("Active learning stopping criterion: iteration %d (%d observed structures)", stop_iter, n_initial + stop_iter)
    LOGGER.info("Selected surrogate checkpoint: %s", model_path.relative_to(ROOT))

    # ------------------------------------------------------------------
    # 4. Surrogate parameter distributions
    # ------------------------------------------------------------------
    device = torch.device("cpu")
    with open(al_config_path, encoding="utf-8") as handle:
        train_cfg = json.load(handle)
    n_pca = train_cfg.get("n_pca_components", 6)
    pcs = load_pca(ROOT / "data/pca/pc_scores.npz", n_pca)
    z_raw = torch.from_numpy(pcs.astype(np.float32)).to(device)
    z_mean = z_raw.mean(0, keepdim=True)
    z_std = z_raw.std(0, keepdim=True).clamp_min(1e-8)
    z_norm = normalize_z(z_raw, z_mean, z_std)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state"]
    inducing_shape = state_dict["variational_strategy.base_variational_strategy.inducing_points"].shape
    coeff = state_dict["variational_strategy.lmc_coefficients"]
    num_latents, num_tasks = coeff.shape
    rank = int(num_latents - num_tasks)
    model = build_svgp_from_config(
        torch.zeros(inducing_shape, device=device), num_tasks, rank, train_cfg,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    LOGGER.info("Loaded surrogate: %d tasks, rank=%d", num_tasks, rank)

    # The posterior is summarised by Monte-Carlo sampling, so the generator is
    # seeded to make this figure reproducible bit for bit.  With 64 samples per
    # structure the estimate is already visually converged; the seed only fixes
    # the residual sampling noise.
    torch.manual_seed(MC_SEED)
    params_list = []
    with torch.no_grad():
        for i in range(0, z_norm.shape[0], 1024):
            out = model(z_norm[i:i + 1024])
            samples = out.rsample(torch.Size([64]))
            params_list.append((F.softplus(samples) + 1e-6).mean(dim=0))
    params_np = torch.cat(params_list, dim=0).cpu().numpy()
    LOGGER.info("Computed parameter distributions: shape=%s", params_np.shape)
    plotting.plot_param_distribution(params_np, FIGURES / "param_distribution.pdf")
    log_saved(FIGURES / "param_distribution.pdf")

    # ------------------------------------------------------------------
    # 5. Inverse design results and summary table
    # ------------------------------------------------------------------
    omega_p = {"P11": 1.0, "P22": 1.0, "P12": 1.0}
    eta = 5.0
    e_max = 50
    marker_evals = 20
    seed = TARGET_SEED
    n_targets = 1000
    component_combos = [
        ("P11",), ("P22",), ("P12",),
        ("P11", "P22"), ("P11", "P12"), ("P22", "P12"),
        ("P11", "P22", "P12"),
    ]
    eval_steps = [1] + list(range(10, 210, 10))
    ctx = inv.setup(
        dataset_path=ROOT / "data/oracle/oracle_45deg.npz",
        pca_path=ROOT / "data/pca/pc_scores.npz",
        checkpoint_path=model_path,
        train_config_path=al_config_path,
    )
    candidates = np.setdiff1d(np.arange(ctx.targets_full.shape[0]), ctx.train_idx)
    random_targets = np.random.default_rng(seed).choice(candidates, size=n_targets, replace=False)
    run_dir = ROOT / "output/inverse_design"
    manifest_path = run_dir / "sweep_settings.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        issues = []
        if int(manifest.get("target_sample_seed", -1)) != seed:
            issues.append(f"target_sample_seed: {manifest.get('target_sample_seed')} vs {seed}")
        if int(manifest.get("n_targets", -1)) != n_targets:
            issues.append(f"n_targets: {manifest.get('n_targets')} vs {n_targets}")
        if float(manifest.get("eta", -1.0)) != eta:
            issues.append(f"eta: {manifest.get('eta')} vs {eta}")
        if issues:
            LOGGER.warning("sweep_settings.json disagrees with script constants: %s", "; ".join(issues))
        else:
            LOGGER.info("sweep_settings.json matches the reproduction constants")
    else:
        LOGGER.warning("sweep_settings.json is missing; cached-run settings were not cross-checked")

    run_cache = {}
    for combo in component_combos:
        combo_key = tuple(combo)
        run_cache[combo_key] = load_run(combo_key, e_max)
        LOGGER.info("%s: hit rate at E_max=%d = %.1f%%", plotting._combo_key(combo_key), e_max, plotting._hit_rate(run_cache[combo_key]))

    combo_full = ("P11", "P22", "P12")
    row_mask = np.array(inv._component_mask(ctx.row_labels_full, combo_full), dtype=np.int64)
    targets_full_masked = ctx.targets_full[:, row_mask]
    target_indices_plot = np.asarray(run_cache[combo_full]["random_target_indices"], dtype=np.int64)
    plotting.plot_target_stress(
        targets=targets_full_masked,
        tags_subset=ctx.tags_subset,
        F_grid_subset=ctx.F_grid_subset,
        target_indices=target_indices_plot,
        include_shear=ctx.include_shear,
        shear_rotations=ctx.shear_rotations,
        row_mask=row_mask,
        out_path=FIGURES / "target_stress.pdf",
    )
    log_saved(FIGURES / "target_stress.pdf")

    table_rows = []
    for combo in component_combos:
        combo_key = tuple(combo)
        data = run_cache[combo_key]
        mask = np.array(inv._component_mask(ctx.row_labels_full, combo_key), dtype=np.int64)
        targets_masked = ctx.targets_full[:, mask]
        row_labels = [ctx.row_labels_full[i] for i in mask]
        target_indices = np.array(data.get("random_target_indices", []), dtype=np.int64)
        oracle_indices = plotting._parity_indices(data, e_max)
        met = plotting._threshold_met(data)
        counts = np.array(data.get("random_oracle_eval_counts", []), dtype=np.int64)
        rate_full = float(np.mean(met) * 100) if met.size else float("nan")
        rate_marker = float(np.mean((counts <= marker_evals) & met) * 100) if counts.size else float("nan")
        r2_map = {}
        for comp in ("P11", "P22", "P12"):
            x, y = plotting._parity_points(targets_masked, row_labels, target_indices, oracle_indices, comp, "mean")
            r2_map[comp] = f"{plotting._r2(x, y):.3f}" if x.size else "--"
        table_rows.append({
            "combo": plotting._combo_key(combo_key),
            "hit_rate_pct": f"{rate_full:.1f}",
            "hit_rate_le_marker_pct": f"{rate_marker:.1f}",
            "r2_p11": r2_map["P11"], "r2_p22": r2_map["P22"], "r2_p12": r2_map["P12"],
        })
    plotting.write_summary_table(table_rows, FIGURES)
    LOGGER.info("Saved table: %s", (FIGURES / "inverse_design_summary.csv").relative_to(ROOT))
    LOGGER.info("Saved table: %s", (FIGURES / "inverse_design_summary.md").relative_to(ROOT))
    LOGGER.info("Inverse design summary table:\n%s", pd.DataFrame(table_rows).to_string(index=False))

    hit_rate_curves = {}
    for combo in component_combos:
        combo_key = tuple(combo)
        label = plotting._combo_label(combo_key)
        hit_rate_curves[label] = [plotting._hit_rate(load_run(combo_key, max_ev)) for max_ev in eval_steps]
    plotting.plot_hit_rate(hit_rate_curves, FIGURES / "hit_rate.pdf", eval_steps=eval_steps)
    log_saved(FIGURES / "hit_rate.pdf")

    hist_data = {}
    hist_targets_map = {}
    for combo in plotting.HIST_COMBOS:
        hist_data[tuple(combo)] = run_cache[tuple(combo)]
        mask = np.array(inv._component_mask(ctx.row_labels_full, combo), dtype=np.int64)
        hist_targets_map[tuple(combo)] = (ctx.targets_full[:, mask], [ctx.row_labels_full[i] for i in mask])
    plotting.plot_oracle_counts(hist_data, hist_targets_map, FIGURES / "oracle_counts.pdf")
    log_saved(FIGURES / "oracle_counts.pdf")

    # ------------------------------------------------------------------
    # 6. Lambda sensitivity and baseline comparison
    # ------------------------------------------------------------------
    lambda_results = np.load(ROOT / "output/studies/lambda_sensitivity/results.npz")
    plotting.plot_lambda_sensitivity(
        lambda_results["eval_steps"], lambda_results["gamma_values"],
        lambda_results["hit_rate_percent"], FIGURES / "lambda_sensitivity.pdf",
    )
    budget_columns = [0, 9, 19, 49]
    selected_rates = lambda_results["hit_rate_percent"][:, budget_columns]
    baseline = selected_rates[np.flatnonzero(np.isclose(lambda_results["gamma_values"], 0.0))[0]]
    LOGGER.info("Lambda sensitivity (change from lambda/lambda_0=0):")
    LOGGER.info("lambda/lambda_0 | E_hit=1 | E_hit=10 | E_hit=20 | E_hit=50 [percent]")
    for gamma, rates in zip(lambda_results["gamma_values"], selected_rates - baseline):
        LOGGER.info("%15g | %6.1f | %7.1f | %7.1f | %7.1f", gamma, *rates)
    log_saved(FIGURES / "lambda_sensitivity.pdf")

    # The proposed, random search, and BO--EI engineered-feature results each
    # live in their own directory; load_split_results restacks them into the
    # per-method arrays this figure expects.
    baseline_root = ROOT / "output/studies/baseline_comparison"
    raw_pca_bo_dir = baseline_root / "bo_ei_raw_image_pcs"
    baseline_results = load_split_results(baseline_root)
    if baseline_results is None:
        raise FileNotFoundError(
            f"Missing per-method baseline results under {baseline_root}"
        )
    raw_pca_bo_results = np.load(raw_pca_bo_dir / "results.npz")
    method_labels = [str(label) for label in baseline_results["method_labels"]]
    hit_rates = baseline_results["hit_rate_percent"]
    ordered_eval_counts = np.stack([
        baseline_results["eval_counts"][method_labels.index("Proposed")],
        baseline_results["eval_counts"][method_labels.index("BO--EI")],
        baseline_results["eval_counts"][method_labels.index("Random")],
    ])
    ordered_percentiles = np.stack([
        baseline_results["best_nmae_percentiles"][method_labels.index("Proposed")],
        baseline_results["best_nmae_percentiles"][method_labels.index("BO--EI")],
        baseline_results["best_nmae_percentiles"][method_labels.index("Random")],
    ])
    plotting.plot_inverse_baseline_comparison(
        baseline_results["eval_steps"], hit_rates[method_labels.index("Proposed")],
        hit_rates[method_labels.index("BO--EI")], hit_rates[method_labels.index("Random")],
        FIGURES / "inverse_baseline_comparison.pdf",
        best_nmae_percentiles=ordered_percentiles,
        percentile_levels=baseline_results["best_nmae_percentile_levels"],
        raw_pca_bo_hit_rate=raw_pca_bo_results["hit_rate_percent"],
        raw_pca_bo_best_nmae_percentiles=raw_pca_bo_results["best_nmae_percentiles"],
        eval_counts_by_method=ordered_eval_counts,
        raw_pca_bo_eval_counts=raw_pca_bo_results["eval_counts"],
    )
    display_labels = ["Proposed framework", "BO--EI (proposed features)", "BO--EI (raw-image PCs)", "Random search"]
    display_hit_rates = [
        hit_rates[method_labels.index("Proposed")], hit_rates[method_labels.index("BO--EI")],
        raw_pca_bo_results["hit_rate_percent"], hit_rates[method_labels.index("Random")],
    ]
    LOGGER.info("Baseline hit rates [E_hit=1, 10, 20, 50] [%]:")
    for label, rates in zip(display_labels, display_hit_rates):
        LOGGER.info("%-31s %s", label, "  ".join(f"{rate:5.1f}" for rate in rates[budget_columns]))
    percentile_levels = baseline_results["best_nmae_percentile_levels"]
    median_row = np.flatnonzero(np.isclose(percentile_levels, 50.0))[0]
    median_best_nmae = baseline_results["best_nmae_percentiles"][:, median_row]
    display_median_best_nmae = [
        median_best_nmae[method_labels.index("Proposed")], median_best_nmae[method_labels.index("BO--EI")],
        raw_pca_bo_results["best_nmae_percentiles"][median_row], median_best_nmae[method_labels.index("Random")],
    ]
    LOGGER.info("Baseline median best-so-far aggregated nMAE [E_hit=1, 10, 20, 50]:")
    for label, errors in zip(display_labels, display_median_best_nmae):
        LOGGER.info("%-31s %s", label, "  ".join(f"{error:5.2f}" for error in errors[budget_columns]))
    log_saved(FIGURES / "inverse_baseline_comparison.pdf")

    parity_combos = [("P11",), ("P11", "P22"), ("P11", "P22", "P12")]
    parity_data = {}
    parity_targets_map = {}
    for combo in parity_combos:
        parity_data[tuple(combo)] = run_cache[tuple(combo)]
        mask = np.array(inv._component_mask(ctx.row_labels_full, combo), dtype=np.int64)
        parity_targets_map[tuple(combo)] = (ctx.targets_full[:, mask], [ctx.row_labels_full[i] for i in mask])
    plotting.plot_parity(parity_data, parity_targets_map, FIGURES / "parity.pdf", budget=e_max)
    log_saved(FIGURES / "parity.pdf")
    LOGGER.info("Reproduction completed successfully.")
    LOGGER.info("Log written to %s", LOG_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
