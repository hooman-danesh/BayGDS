"""Representation-independent exact-GP expected-improvement search.

Author:  Hooman Danesh <hooman.danesh@tu-braunschweig.de>, TU Braunschweig
License: MIT (see LICENSE). Released September 2026.
"""
from __future__ import annotations

import time
import warnings
from typing import Any, Dict, Sequence, Tuple

import numpy as np
from scipy.special import ndtr
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Kernel, RBF

from experiments.baselines import _shared


def _fit_gp_with_jitter(
    kernel: Kernel,
    x_observed: np.ndarray,
    y_observed: np.ndarray,
) -> Tuple[GaussianProcessRegressor, float]:
    last_error: Exception | None = None
    for alpha in (_shared.GP_ALPHA, 1e-5, 1e-4, 1e-3):
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            optimizer=None,
            normalize_y=True,
            copy_X_train=False,
        )
        try:
            model.fit(x_observed, y_observed)
            return model, float(alpha)
        except np.linalg.LinAlgError as error:
            last_error = error
    raise RuntimeError(
        "Exact BO GP factorization failed after jitter retries."
    ) from last_error


def _fit_initial_gp(
    x_observed: np.ndarray,
    y_observed: np.ndarray,
) -> Tuple[GaussianProcessRegressor, float]:
    """Learn target-specific BO hyperparameters from the shared LHS data."""
    initial_kernel = ConstantKernel(
        constant_value=1.0,
        constant_value_bounds=_shared.GP_AMPLITUDE_BOUNDS,
    ) * RBF(
        length_scale=np.ones(x_observed.shape[1], dtype=np.float64),
        length_scale_bounds=_shared.GP_LENGTH_SCALE_BOUNDS,
    )
    last_error: Exception | None = None
    for alpha in (_shared.GP_ALPHA, 1e-5, 1e-4, 1e-3):
        model = GaussianProcessRegressor(
            kernel=initial_kernel,
            alpha=alpha,
            normalize_y=True,
            n_restarts_optimizer=0,
            random_state=_shared.SETUP_SEED,
            copy_X_train=False,
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                model.fit(x_observed, y_observed)
            return model, float(alpha)
        except np.linalg.LinAlgError as error:
            last_error = error
    raise RuntimeError(
        "Initial exact BO GP fit failed after jitter retries."
    ) from last_error


def _expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    incumbent: float,
) -> np.ndarray:
    improvement = incumbent - mean - _shared.EI_XI
    z = np.divide(
        improvement,
        std,
        out=np.zeros_like(improvement),
        where=std > 1e-12,
    )
    density = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    expected = improvement * ndtr(z) + std * density
    expected[std <= 1e-12] = 0.0
    return expected


def run_for_target(
    target_vec: np.ndarray,
    lhs_initial_indices: np.ndarray,
    x_all: np.ndarray,
    targets: np.ndarray,
    row_labels: Sequence[str],
    loss_weights: np.ndarray,
) -> Dict[str, Any]:
    """Run standard BO--EI for one target and one feature representation."""
    start_time = time.perf_counter()
    chosen_online: list[int] = []
    nmae_values: list[float] = []
    squared_values: list[float] = []
    evaluated = np.zeros(x_all.shape[0], dtype=bool)
    maximum_alpha = _shared.GP_ALPHA
    threshold_met = False

    initial_squared, initial_nmae = _shared.oracle_metrics(
        target_vec,
        lhs_initial_indices,
        targets,
        row_labels,
        loss_weights,
    )
    observed_indices = np.asarray(lhs_initial_indices, dtype=np.int64).copy()
    observed_log_objective = np.log(np.maximum(initial_squared, 1e-12))
    evaluated[observed_indices] = True
    model, alpha = _fit_initial_gp(
        x_all[observed_indices], observed_log_objective,
    )
    maximum_alpha = max(maximum_alpha, alpha)
    fitted_kernel = model.kernel_

    while (
        not threshold_met
        and len(chosen_online) < _shared.ORACLE_BUDGET
    ):
        predictive_mean, predictive_std = model.predict(x_all, return_std=True)
        acquisition = _expected_improvement(
            predictive_mean,
            predictive_std,
            float(np.min(observed_log_objective)),
        )
        acquisition[evaluated] = -np.inf
        if not np.any(np.isfinite(acquisition)):
            raise RuntimeError("No unevaluated candidate has a finite EI score.")
        if float(np.nanmax(acquisition)) <= 1e-15:
            predictive_mean = np.asarray(predictive_mean).copy()
            predictive_mean[evaluated] = np.inf
            next_idx = int(np.argmin(predictive_mean))
        else:
            next_idx = int(np.nanargmax(acquisition))

        squared, nmae = _shared.oracle_metrics(
            target_vec,
            np.asarray([next_idx]),
            targets,
            row_labels,
            loss_weights,
        )
        chosen_online.append(next_idx)
        evaluated[next_idx] = True
        squared_values.append(float(squared[0]))
        nmae_values.append(float(nmae[0]))
        threshold_met = bool(nmae[0] <= _shared.ETA)
        observed_indices = np.append(observed_indices, next_idx)
        observed_log_objective = np.append(
            observed_log_objective,
            float(np.log(max(squared[0], 1e-12))),
        )
        if (
            not threshold_met
            and len(chosen_online) < _shared.ORACLE_BUDGET
        ):
            model, alpha = _fit_gp_with_jitter(
                fitted_kernel,
                x_all[observed_indices],
                observed_log_objective,
            )
            maximum_alpha = max(maximum_alpha, alpha)

    n_evaluated = len(chosen_online)
    kernel_params = fitted_kernel.get_params()
    lengthscale = np.asarray(
        kernel_params["k2__length_scale"], dtype=np.float64,
    ).reshape(-1)
    return {
        "indices": np.asarray(chosen_online, dtype=np.int64),
        "nmae": np.asarray(nmae_values, dtype=np.float64),
        "squared_mismatch": np.asarray(squared_values, dtype=np.float64),
        "eval_count": (
            n_evaluated
            if threshold_met else _shared.ORACLE_BUDGET + 1
        ),
        "threshold_met": threshold_met,
        "n_evaluated": n_evaluated,
        "max_alpha": maximum_alpha,
        "initial_best_nmae": float(np.min(initial_nmae)),
        "initial_threshold_met_ignored": bool(
            np.any(initial_nmae <= _shared.ETA)
        ),
        "kernel_amplitude": float(kernel_params["k1__constant_value"]),
        "kernel_lengthscale": lengthscale,
        "seconds": time.perf_counter() - start_time,
    }
