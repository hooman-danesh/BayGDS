#!/usr/bin/env python3
"""Train the Gaussian-process surrogate with active learning.

Purpose:
    This script sets the main paths and control parameters, then launches the
    active learning loop that builds the surrogate from a small number of
    oracle-labeled structures.

Main parameters:
    - DATASET_PATH: oracle response database used for labeling.
    - PCA_PATH: PC score file used as GP input.
    - OUTPUT_DIR: folder for the selected checkpoint, compact state, and
      normalization statistics.
    - SEED, N_INIT, T_MAX, TEST_HOLDOUT_SIZE: active learning controls.
    - BATCH_SIZE, EPOCHS, LEARNING_RATE, N_R, S, and LOG_EVERY:
      surrogate-training controls.
    - STOPPING_L and STOPPING_EPSILON: convergence rule that selects the
      retained checkpoint.

Outputs:
    Writes the selected surrogate checkpoint, compact active learning state,
    normalization statistics, and MAE curve for later inverse design runs.

Usage:
    python -m experiments.baygds.run_active_learning

Author:  Hooman Danesh <hooman.danesh@tu-braunschweig.de>, TU Braunschweig
License: MIT (see LICENSE). Released September 2026.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str((ROOT / "src").resolve()))

import active_learning as al

# ---------------------------------------------------------------------------
# Control parameters and paths
# ---------------------------------------------------------------------------
DATASET_PATH = ROOT / "data/oracle/oracle_0deg.npz"
PCA_PATH = ROOT / "data/pca/pc_scores.npz"
OUTPUT_DIR = ROOT / "output/surrogate/active_learning"

SEED = 0
N_INIT = 10              # |T_0|, size of the initial labeled set
T_MAX = 210             # T_max, maximum active learning acquisitions
TEST_HOLDOUT_SIZE = 500

BATCH_SIZE = 1024
EPOCHS = 500
LEARNING_RATE = 0.01
N_R = 6                 # n_r, total latent processes
S = 64                  # S, Monte Carlo samples for the predictive moments
LOG_EVERY = 100

# Stopping rule: the run halts once the MAE improvement averaged over the
# last L acquisitions falls below epsilon.
STOPPING_L = 5             # L, sliding-window length
STOPPING_EPSILON = 1.0e-3  # epsilon, threshold on the windowed MAE change


al.DATASET_PATH = DATASET_PATH
al.PCA_PATH = PCA_PATH
al.OUTPUT_DIR = OUTPUT_DIR
al.SEED = SEED
al.N_INIT = N_INIT
al.T_max = T_MAX
al.TEST_HOLDOUT_SIZE = TEST_HOLDOUT_SIZE
al.BATCH_SIZE = BATCH_SIZE
al.EPOCHS = EPOCHS
al.LEARNING_RATE = LEARNING_RATE
al.N_R = N_R
al.S = S
al.LOG_EVERY = LOG_EVERY
al.STOPPING_L = STOPPING_L
al.STOPPING_EPSILON = STOPPING_EPSILON


def main() -> None:
    al.run_active_learning()


if __name__ == "__main__":
    main()
