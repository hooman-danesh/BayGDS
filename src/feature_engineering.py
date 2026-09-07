"""Build low-dimensional structure descriptors with NumPy and FFTs.

Purpose:
    This module converts binary microstructure images into compact PC scores
    computed from two-point statistical correlations (solid and interface auto-correlations)
    These PC scores are later used for surrogate training.

Main parameters:
    - n_components and random_state in perform_pca()/run_feature_engineering().
    - input_path and output_path in run_feature_engineering().

Outputs:
    Returns or saves PC score arrays under the key 'PCA_components'.

Reference:
    For details on the descriptor construction, see Danesh et al.,
    Physical Review Materials, 2025:
    https://doi.org/10.1103/8zzt-4b7z

Author:  Hooman Danesh <hooman.danesh@tu-braunschweig.de>, TU Braunschweig
License: MIT (see LICENSE). Released September 2026.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import convolve2d
from sklearn.decomposition import PCA


def create_interface(microstructure: np.ndarray, kernel: np.ndarray | None = None) -> np.ndarray:
    """Identify interface voxels in a binary microstructure using periodic padding."""
    if kernel is None:
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    padded = np.pad(microstructure, pad_width=1, mode="wrap")
    convolved = convolve2d(padded, kernel, mode="same")
    interface = convolved[1:-1, 1:-1]
    return ((microstructure == 0) & (interface > 0) & (interface < 5)).astype(np.uint8)


def _periodic_two_point_correlation(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    if left.shape != right.shape:
        raise ValueError("Expected left and right arrays with matching shapes.")
    if left.ndim != 3:
        raise ValueError("Expected arrays with shape (n_samples, height, width).")

    left_fft = np.fft.fftn(left, axes=(1, 2))
    right_fft = np.fft.fftn(right, axes=(1, 2))
    corr = np.fft.ifftn(np.conj(left_fft) * right_fft, axes=(1, 2)).real
    corr /= left.shape[1] * left.shape[2]
    corr = np.fft.fftshift(corr, axes=(1, 2))

    start_x = 1 if corr.shape[1] % 2 == 0 else 0
    start_y = 1 if corr.shape[2] % 2 == 0 else 0
    corr = corr[:, start_x:, start_y:]
    return corr.astype(np.float32, copy=False)


def compute_two_point_statistics(structures: np.ndarray) -> np.ndarray:
    """Compute solid, interface, and cross-correlation statistics.

    Returns array shaped (n_samples, H, W, 3).
    """
    structures = np.asarray(structures)
    if structures.ndim != 3:
        raise ValueError("Expected structures with shape (n_samples, height, width).")

    structures = structures.astype(np.float32, copy=False)
    interfaces = np.stack([create_interface(s) for s in structures], axis=0).astype(np.float32, copy=False)

    solid_stat = _periodic_two_point_correlation(structures, structures)
    interface_stat = _periodic_two_point_correlation(interfaces, interfaces)
    cross_stat = _periodic_two_point_correlation(structures, interfaces)

    stats = np.stack([solid_stat, interface_stat, cross_stat], axis=-1)

    reference_std = solid_stat.std()
    if reference_std == 0.0:
        return stats
    for idx in range(stats.shape[-1]):
        channel = stats[..., idx]
        std = channel.std()
        if std == 0.0:
            continue
        stats[..., idx] = (channel / std) * reference_std
    return stats


def perform_pca(
    all_stats: np.ndarray,
    n_components: int,   # n_z, retained descriptor dimension
    random_state: int = 42,
) -> np.ndarray:
    """Run PCA on the solid and interface autocorrelations only.

    The input ``all_stats`` stores three channels in the order
    ``[solid, interface, cross]``. PCA is built from the first two channels
    only; the cross-correlation channel is not included in the PCA input.
    """
    flattened_channels = [
        all_stats[:, :, :, i].reshape(all_stats.shape[0], -1)
        for i in range(all_stats.shape[-1])
    ]
    combined = np.hstack([flattened_channels[i] for i in [0, 1]]).astype(np.float64, copy=False)
    pca = PCA(n_components=n_components, random_state=random_state)
    return pca.fit_transform(combined)


def run_feature_engineering(
    input_path: Path = Path("data/structures/structures.npz"),
    output_path: Path = Path("data/pca/pc_scores.npz"),
    n_components: int = 6,   # n_z, retained descriptor dimension
    random_state: int = 42,
) -> np.ndarray:
    """Full pipeline: load structures, compute stats, run PCA, save scores."""
    print(f"Loading microstructures from {input_path} ...")
    data = np.load(input_path, allow_pickle=True)
    structures = np.asarray(data["structures"])
    if structures.ndim != 3:
        raise ValueError("Expected structures with shape (n_samples, height, width).")
    print(f"Loaded {structures.shape[0]} structures of size {structures.shape[1]}x{structures.shape[2]}.")

    print("Computing two-point statistics with NumPy FFTs ...")
    all_stats = compute_two_point_statistics(structures)

    print(f"Running PCA (n_components={n_components}, random_state={random_state}) ...")
    components = perform_pca(all_stats, n_components, random_state)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, PCA_components=components)
    print(f"Saved PC scores to {output_path}")
    return components
