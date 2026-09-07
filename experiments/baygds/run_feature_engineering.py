#!/usr/bin/env python3
"""Compute PC scores from the microstructure dataset.

Purpose:
    This script loads the structure library, computes the selected two-point
    statistics, and stores compact PC scores used by the training
    and inverse design scripts.

Main parameters:
    - INPUT_PATH: location of the raw structure library.
    - OUTPUT_PATH: destination of the compressed PC score file.
    - N_COMPONENTS: number of retained components.
    - RANDOM_STATE: seed for reproducible PC scores.

Outputs:
    Saves an .npz file with key 'PCA_components' for downstream scripts.

Usage:
    python -m experiments.baygds.run_feature_engineering

Author:  Hooman Danesh <hooman.danesh@tu-braunschweig.de>, TU Braunschweig
License: MIT (see LICENSE). Released September 2026.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str((ROOT / "src").resolve()))

from feature_engineering import run_feature_engineering

# ---------------------------------------------------------------------------
# Control parameters and paths
# ---------------------------------------------------------------------------
INPUT_PATH = ROOT / "data/structures/structures.npz"
OUTPUT_PATH = ROOT / "data/pca/pc_scores.npz"
N_COMPONENTS = 8
RANDOM_STATE = 1506


def main() -> None:
    run_feature_engineering(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        n_components=N_COMPONENTS,
        random_state=RANDOM_STATE,
    )


if __name__ == "__main__":
    main()
