"""Interface to the high-fidelity response source.

Purpose:
    This module provides a minimal interface for querying the response of a
    structure. In the current setup it reads from a precomputed stress
    database, but it can be replaced with a live simulation or experimental
    backend.

Main parameters:
    - stress_database: array or tensor containing oracle responses.
    - idx / indices: single or batched design identifiers to be evaluated.

Outputs:
    Returns the requested oracle stress response without altering the stored
    data.

Author:  Hooman Danesh <hooman.danesh@tu-braunschweig.de>, TU Braunschweig
License: MIT (see LICENSE). Released September 2026.
"""
from __future__ import annotations

import numpy as np
import torch


def call_oracle(stress_database: np.ndarray | torch.Tensor, idx: int) -> np.ndarray | torch.Tensor:
    """Query the oracle for a single microstructure."""
    return stress_database[idx]


def call_oracle_batch(
    stress_database: np.ndarray | torch.Tensor,
    indices: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Query the oracle for a batch of microstructures."""
    return stress_database[indices]
