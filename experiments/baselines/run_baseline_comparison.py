#!/usr/bin/env python3
"""Run the full library search baselines.

Four methods are evaluated on one shared set of targets: the BayGDS evaluation
sequences, random search, BO--EI on the six proposed PC features, and BO--EI on
six principal components of the raw binary structures.

The first three share a target loop and are computed together; the raw image
variant is computed separately. Each writes only its own results. Figures are
drawn from those results by reproduction/reproduce.py, not here.

Author:  Hooman Danesh <hooman.danesh@tu-braunschweig.de>, TU Braunschweig
License: MIT (see LICENSE). Released September 2026.
"""
from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.resolve()))

from experiments.baselines import (  # noqa: E402
    bo_ei_proposed_features,
    bo_ei_raw_image_pcs,
)


OUTPUT_ROOT = ROOT / "output/studies/baseline_comparison"

# ---------------------------------------------------------------------------
# Control parameters shared by every method
# ---------------------------------------------------------------------------
OMEGA_P = {"P11": 1.0, "P22": 1.0, "P12": 1.0}   # omega_p, component weights
ETA = 5.0               # eta, acceptance tolerance on the weighted nMAE [%]
N_TARGETS = 1000        # N_tar, held out targets
ORACLE_BUDGET = 50      # largest E_hit reported by the comparison

# BO--EI controls
BO_INITIAL_POINTS = 200   # shared Latin hypercube initial design
EI_XI = 0.01              # exploration offset of the expected improvement rule

# Raw image descriptors for the BO--EI raw PC baseline
PCA_COMPONENTS = 6        # matched to the proposed feature dimension

_shared = bo_ei_proposed_features._shared


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-targets",
        type=int,
        default=None,
        help="Use the first N fixed targets in an isolated timing pilot.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Independent raw PCA BO targets to process in parallel.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=bo_ei_proposed_features.SAVE_EVERY,
        help="Save resumable progress after this many completed targets.",
    )
    return parser.parse_args()


def _apply_controls() -> None:
    """Push the knobs above into the modules that implement each method.

    ``_shared`` holds the definitions, but both BO modules copy several of them
    into their own namespace at import time, so a value has to be set in every
    place it was copied to. Doing it here keeps this script the single point of
    control.
    """
    for module in (_shared, bo_ei_proposed_features):
        module.OMEGA_P = OMEGA_P
        module.ETA = ETA
        module.N_TARGETS = N_TARGETS
        module.ORACLE_BUDGET = ORACLE_BUDGET
        module.BO_INITIAL_POINTS = BO_INITIAL_POINTS
        module.EI_XI = EI_XI
    bo_ei_raw_image_pcs.PCA_COMPONENTS = PCA_COMPONENTS
    bo_ei_raw_image_pcs.DEFAULT_FEATURE_PATH = (
        bo_ei_raw_image_pcs.DEFAULT_OUTPUT_DIR
        / f"raw_structure_pca_{PCA_COMPONENTS:02d}.npz"
    )


def run_comparison(args: argparse.Namespace) -> None:
    _apply_controls()
    if args.max_targets is None:
        output_root = OUTPUT_ROOT
    else:
        if not 1 <= args.max_targets <= N_TARGETS:
            raise ValueError(f"max_targets must lie in [1, {N_TARGETS}].")
        output_root = OUTPUT_ROOT / "pilots" / f"targets_{args.max_targets:04d}"

    # 1. BayGDS sequences, random search, and BO--EI on the proposed features.
    bo_ei_proposed_features.run_benchmark(Namespace(
        max_targets=args.max_targets,
        output_dir=output_root / bo_ei_proposed_features.METHOD_DIRS["BO--EI"],
        save_every=args.save_every,
    ))

    # 2. BO--EI on the raw image PCs. The descriptor artifact is shared: it is
    #    deterministic and target independent, so a pilot cannot contaminate it.
    bo_ei_raw_image_pcs.run_bo_ei_raw_image_pcs(Namespace(
        max_targets=args.max_targets,
        output_dir=output_root / "bo_ei_raw_image_pcs",
        feature_path=bo_ei_raw_image_pcs.DEFAULT_FEATURE_PATH,
        save_every=args.save_every,
        n_jobs=args.n_jobs,
    ))

    print(f"\nResults for all four methods under: {output_root}")
    print("Draw the comparison figure with reproduction/reproduce.py")


if __name__ == "__main__":
    run_comparison(_parse_args())
