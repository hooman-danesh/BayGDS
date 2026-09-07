"""Shared data loading and preprocessing utilities.

Purpose:
    This module loads descriptor and oracle data, selects deformation paths,
    assembles design matrices, stacks stress targets, and standardizes arrays
    for training and inverse design.

Main parameters:
    - DEFAULT_INCLUDE_PATHS: loading paths included by default.
    - ROTATION_SUFFIX: mapping from rotation angles to dataset tags.
    - rotations, include_paths, include_shear, and shear_rotations arguments in
      the selection/build helpers.
    - n_components in load_pca().

Notation used here:
    Z: low-dimensional descriptors from the retained principal components.
    Q(F): constitutive design matrix associated with a deformation state.
    S_obs: stacked oracle stress observations.

Author:  Hooman Danesh <hooman.danesh@tu-braunschweig.de>, TU Braunschweig
License: MIT (see LICENSE). Released September 2026.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

ROTATION_SUFFIX: Dict[int, str] = {0: "", 45: "_rot45",}

# Loading-path names used in the dataset
DEFAULT_INCLUDE_PATHS = ("tension_x", "tension_y", "equibiaxial", "off_x", "off_y")


def load_dataset(path: Path) -> Dict:
    path = Path(path)
    if path.suffix.lower() in (".npz", ".npy"):
        data = np.load(path, allow_pickle=True)
        return {
            "stresses": data["stresses"],
            "deformation_grid": data["deformation_grid"],
            "deformation_tags": data["deformation_tags"],
        }
    with open(path, "rb") as f:
        return pickle.load(f)


def load_pca(path: Path, n_components: int | None = None) -> np.ndarray:
    """Load low-dimensional PC scores (z) from a .npz or .pkl file."""
    path = Path(path)
    if path.suffix.lower() in (".npz", ".npy"):
        data = np.load(path, allow_pickle=True)
        pcs = np.asarray(data["PCA_components"], dtype=np.float64)
    else:
        with open(path, "rb") as f:
            data = pickle.load(f)
        pcs = np.asarray(data["PCA_components"], dtype=np.float64)
    if n_components is None:
        return pcs
    return pcs[:, :n_components]


def rotation_angle_from_tag(tag: str) -> int:
    if "_rot" in tag:
        try:
            return int(tag.split("_rot")[-1])
        except ValueError:
            return 0
    return 0


def preferred_dirs(tag: str) -> Tuple[np.ndarray, np.ndarray]:
    return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])


def _plane_stress_parts(F_2x2: np.ndarray, tag: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b = F_2x2[0]
    c, d = F_2x2[1]
    det2 = a * d - b * c
    if abs(det2) < 1e-9:
        raise ValueError(f"Singular in-plane deformation for tag {tag}: det2={det2}")
    f33 = 1.0 / det2
    F = np.array([[a, b, 0.0], [c, d, 0.0], [0.0, 0.0, f33]])
    FinvT = np.linalg.inv(F).T
    C = F.T @ F
    n1, n2 = preferred_dirs(tag)
    Fn1 = F @ n1
    Fn2 = F @ n2
    I4 = float(n1 @ (C @ n1))
    I6 = float(n2 @ (C @ n2))
    theta_I1_part = 2.0 * (F - (f33**2) * FinvT)
    theta_I4_part = 4.0 * (I4 - 1.0) * np.outer(Fn1, n1)
    theta_I6_part = 4.0 * (I6 - 1.0) * np.outer(Fn2, n2)
    return theta_I1_part, theta_I4_part, theta_I6_part


def include_shear_for_tag(tag: str, include_shear: bool, shear_rotations: Sequence[int]) -> bool:
    if not include_shear:
        return False
    if tag.split("_rot")[0] == "equibiaxial":
        return False
    ang = rotation_angle_from_tag(tag)
    return ang in shear_rotations


def map_path_indices(tags: np.ndarray) -> Dict[str, np.ndarray]:
    unique = np.unique(tags)
    return {name: np.where(tags == name)[0] for name in unique}


def select_fit_indices(
    tags: np.ndarray,
    F_grid: np.ndarray,
    rotations: Sequence[int],
    *,
    include_paths: Sequence[str] = DEFAULT_INCLUDE_PATHS,
) -> Tuple[np.ndarray, List[str]]:
    """Select fit indices. Always uses 5 paths, single identity, one rotation at a time.
    Expects restructured data (tension_x, tension_y)."""
    path_map = map_path_indices(tags)
    fit_tags: List[str] = []
    rotations_sorted = tuple(sorted(rotations))

    def suffix(rot: int) -> str:
        if rot not in ROTATION_SUFFIX:
            raise ValueError(f"Rotation {rot} not in ROTATION_SUFFIX")
        return ROTATION_SUFFIX[rot]

    for base in include_paths:
        for rot in rotations_sorted:
            cand = f"{base}{suffix(rot)}"
            if cand in path_map:
                fit_tags.append(cand)

    fit_tags_unique: List[str] = []
    for t in fit_tags:
        if t not in fit_tags_unique:
            fit_tags_unique.append(t)

    fit_idxs: List[np.ndarray] = []
    identity_seen = False
    for tag in fit_tags_unique:
        idxs = path_map[tag]
        mask = np.ones(len(idxs), dtype=bool)
        for j, idx in enumerate(idxs):
            if np.allclose(F_grid[idx], np.eye(2)):
                if identity_seen:
                    mask[j] = False
                else:
                    identity_seen = True
        idxs = idxs[mask]
        fit_idxs.append(idxs)

    if not fit_idxs:
        raise ValueError("No fit indices found.")
    return np.concatenate(fit_idxs), fit_tags_unique


def select_fit_indices_0deg(tags: np.ndarray, F_grid: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """0 deg, fixed paths, single identity. Manuscript Tab. 1. Expects restructured data (tension_x, tension_y)."""
    path_map = map_path_indices(tags)
    fit_tags = [base for base in DEFAULT_INCLUDE_PATHS if base in path_map]
    fit_idxs: List[np.ndarray] = []
    identity_seen = False
    for tag in fit_tags:
        idxs = path_map[tag]
        mask = np.ones(len(idxs), dtype=bool)
        for j, idx in enumerate(idxs):
            if np.allclose(F_grid[idx], np.eye(2)):
                if identity_seen:
                    mask[j] = False
                else:
                    identity_seen = True
        idxs = idxs[mask]
        fit_idxs.append(idxs)
    if not fit_idxs:
        raise ValueError("No fit indices found.")
    return np.concatenate(fit_idxs), fit_tags


def build_design_matrix(
    F_subset: np.ndarray,
    tags_subset: Sequence[str],
    include_shear: bool,
    shear_rotations: Sequence[int],
) -> Tuple[np.ndarray, List[str]]:
    """Build energy-basis design matrix Q(F). Rows = stress components (P11, P22, P12)."""
    rows: List[List[float]] = []
    labels: List[str] = []
    for F_2x2, tag in zip(F_subset, tags_subset):
        theta_I1_part, theta_I4_part, theta_I6_part = _plane_stress_parts(F_2x2, tag)
        rows.append([theta_I1_part[0, 0], theta_I4_part[0, 0], theta_I6_part[0, 0]])
        labels.append("P11")
        rows.append([theta_I1_part[1, 1], theta_I4_part[1, 1], theta_I6_part[1, 1]])
        labels.append("P22")
        if include_shear_for_tag(tag, include_shear, shear_rotations):
            rows.append([theta_I1_part[0, 1], theta_I4_part[0, 1], theta_I6_part[0, 1]])
            labels.append("P12")
    return np.asarray(rows), labels


def build_design_matrix_nominal(
    F_subset: np.ndarray,
    tags_subset: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    """P11, P22 only. No shear. Self-contained."""
    rows: List[List[float]] = []
    labels: List[str] = []
    for F_2x2, tag in zip(F_subset, tags_subset):
        theta_I1_part, theta_I4_part, theta_I6_part = _plane_stress_parts(F_2x2, tag)
        rows.append([theta_I1_part[0, 0], theta_I4_part[0, 0], theta_I6_part[0, 0]])
        labels.append("P11")
        rows.append([theta_I1_part[1, 1], theta_I4_part[1, 1], theta_I6_part[1, 1]])
        labels.append("P22")
    return np.asarray(rows), labels


def stack_stress_targets(
    stresses_subset: np.ndarray,
    tags_subset: Sequence[str],
    include_shear: bool,
    shear_rotations: Sequence[int],
) -> np.ndarray:
    """Stack stress observations y into a matrix (structures x stress rows)."""
    rows: List[np.ndarray] = []
    for j, tag in enumerate(tags_subset):
        rows.append(stresses_subset[:, j, 0, 0])
        rows.append(stresses_subset[:, j, 1, 1])
        if include_shear_for_tag(tag, include_shear, shear_rotations):
            rows.append(stresses_subset[:, j, 0, 1])
    return np.stack(rows, axis=1)


def stack_stress_targets_nominal(
    stresses_subset: np.ndarray,
    tags_subset: Sequence[str],
) -> np.ndarray:
    """P11, P22 only. No shear. Self-contained."""
    rows: List[np.ndarray] = []
    for j in range(len(tags_subset)):
        rows.append(stresses_subset[:, j, 0, 0])
        rows.append(stresses_subset[:, j, 1, 1])
    return np.stack(rows, axis=1)


def standardize(train: torch.Tensor, full: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train.mean(0, keepdim=True)
    std = train.std(0, keepdim=True).clamp_min(1e-8)
    return (full - mean) / std, mean, std
