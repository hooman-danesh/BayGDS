"""Core Gaussian process surrogate utilities.

Purpose:
    This module defines the multi-output variational GP, its training
    configuration, normalization helpers, checkpoint utilities, and inference
    helpers used during active learning and inverse design.

Main parameters:
    - GPTrainConfig: batch_size, epochs, learning_rate, n_r, S, kernel,
      mean_type, beta_kl, noise_init, and related optimization controls.
    - PARAM_NAMES / N_THETA: constitutive parameter channels predicted by the
      surrogate.
    - build_svgp_from_config(): architecture reconstruction during reload.

Notation used here:
    z_i, Z: low-dimensional descriptor coordinates.
    theta_i: effective constitutive parameters predicted by the GP.
    n_r: number of latent processes (6 = 3 shared + 1 per output).
    sigma^2: observation-noise level in the variational objective.
    S: Monte Carlo sample count used in ELBO evaluation.

Author:  Hooman Danesh <hooman.danesh@tu-braunschweig.de>, TU Braunschweig
License: MIT (see LICENSE). Released September 2026.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import gpytorch
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# The n_theta effective constitutive parameters theta predicted by the surrogate.
PARAM_NAMES = ("theta_I1", "theta_I4", "theta_I6")
N_THETA = len(PARAM_NAMES)   # n_theta


@dataclass
class GPTrainConfig:
    """GP training parameters. Matches NeverTouch_GroundTruth for parity."""
    batch_size: int = 1024
    epochs: int = 500
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    n_r: int = 6  # n_r, total latent processes (shared + one per task)
    S: int = 64  # S, Monte Carlo samples for the predictive moments
    log_every: int = 10
    # GT-aligned
    ard: bool = True
    beta_kl: float = 1.0
    noise_init: float = 0.25
    noise_floor: float = 0.0
    kernel: str = "RBF"
    mean_type: str = "zero"
    cosine_annealing: bool = True
    first_restart: int = 50
    restart_multiplier: int = 2
    eta_min: float = 1e-4


def make_kernel(
    name: str, ard: bool, num_latents: int, in_dim: int, shared_lengthscale: bool
) -> gpytorch.kernels.Kernel:
    """Kernel factory, matches GT."""
    name = name.lower()
    batch = torch.Size([1]) if shared_lengthscale else torch.Size([num_latents])
    if name == "rbf":
        return gpytorch.kernels.RBFKernel(
            batch_shape=batch, ard_num_dims=in_dim if ard else None
        )
    if name == "matern12":
        return gpytorch.kernels.MaternKernel(
            nu=0.5, batch_shape=batch, ard_num_dims=in_dim if ard else None
        )
    if name == "matern32":
        return gpytorch.kernels.MaternKernel(
            nu=1.5, batch_shape=batch, ard_num_dims=in_dim if ard else None
        )
    if name == "matern52":
        return gpytorch.kernels.MaternKernel(
            nu=2.5, batch_shape=batch, ard_num_dims=in_dim if ard else None
        )
    raise ValueError(f"Unknown kernel: {name}")


class CorrelatedLatentSVGP(gpytorch.models.ApproximateGP):
    """Multi-output variational GP with a linear model of coregionalization.

    The n_theta outputs are linear combinations of n_r latent processes.

    n_r = rank + num_tasks: ``rank`` latents shared by all outputs plus one per
    output. That split is only the initialisation; the coefficients train
    freely.

    Inducing points are fixed, never optimised. Placing them at every training
    input, as the active learning loop does, gives the exact variational GP;
    the PC dimension study instead uses a subset, which is the sparse
    approximation.
    """

    def __init__(
        self,
        inducing_points: torch.Tensor,
        num_tasks: int,
        ard: bool,
        rank: int,
        mean_type: str,
        kernel: str,
    ):
        num_latents = rank + num_tasks   # n_r
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )
        base_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=False
        )
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            base_strategy, num_tasks=num_tasks, num_latents=num_latents, latent_dim=-1
        )
        super().__init__(variational_strategy)
        in_dim = inducing_points.size(-1)
        self.covar_module = make_kernel(
            kernel, ard=ard, num_latents=num_latents, in_dim=in_dim, shared_lengthscale=False
        )
        if mean_type.lower() == "zero":
            self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([num_latents]))
        elif mean_type.lower() == "constant":
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        else:
            raise ValueError(f"Unknown mean_type: {mean_type}")
        with torch.no_grad():
            coeff = self.variational_strategy.lmc_coefficients
            coeff[rank:, :] = 0.0
            for t in range(num_tasks):
                coeff[rank + t, t] = 0.1

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def normalize_z(z: torch.Tensor, z_mean: torch.Tensor, z_std: torch.Tensor) -> torch.Tensor:
    """Standardize Z for GP input. Single source of truth for Z normalization."""
    return (z - z_mean) / z_std.clamp_min(1e-8)


def compute_stress_normalization(y_train: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Row-standardize stress targets y (zero mean, unit variance per row)."""
    mean = y_train.mean(0, keepdim=True)
    std = y_train.std(0, keepdim=True).clamp_min(1e-6)
    return mean, std


def save_checkpoint(path: Path, model: gpytorch.models.ApproximateGP, noise_param: torch.nn.Parameter, epoch: int) -> None:
    """Write a checkpoint through a temporary file, so an interrupted save
    cannot leave a truncated file behind."""
    state = {"epoch": epoch, "model_state": model.state_dict(), "noise_log_param": noise_param.detach().cpu()}
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def build_svgp_from_config(
    inducing_points: torch.Tensor,
    num_tasks: int,
    rank: int,
    train_cfg: dict | None = None,
) -> CorrelatedLatentSVGP:
    """Rebuild the model architecture so a checkpoint can be loaded.

    Args:
        inducing_points: shape only; the values come from the checkpoint.
        num_tasks: n_theta, the number of outputs.
        rank: latents shared by all outputs, n_r - n_theta (3 here). Callers
            read it from the checkpoint's (n_r, n_theta) mixing matrix.
        train_cfg: kernel family, mean function, and ARD flag.
    """
    cfg = train_cfg or {}
    return CorrelatedLatentSVGP(
        inducing_points=inducing_points,
        num_tasks=num_tasks,
        ard=cfg.get("ard", True),
        rank=rank,
        mean_type=cfg.get("mean_type", "zero"),
        kernel=cfg.get("kernel", "RBF"),
    )


def save_normalization_stats(
    path: Path,
    z_mean: torch.Tensor,
    z_std: torch.Tensor,
    stress_mean: torch.Tensor,
    stress_std: torch.Tensor,
    row_labels: List[str],
) -> None:
    """Save normalization stats for Z and stress targets."""
    payload = {
        "z_mean": z_mean.detach().cpu().numpy(),
        "z_std": z_std.detach().cpu().numpy(),
        "stress_mean": stress_mean.detach().cpu().numpy(),
        "stress_std": stress_std.detach().cpu().numpy(),
        "row_labels": np.array(row_labels),
        "stress_normalization": np.array(["row_standardize"]),
    }
    np.savez(path, **payload)


def train_model(
    model: gpytorch.models.ApproximateGP,
    design_t: torch.Tensor,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    stress_mean: torch.Tensor,
    stress_std: torch.Tensor,
    config: GPTrainConfig,
    output_dir: Path,
    *,
    init_noise_log_param: torch.Tensor | None = None,
    save_checkpoints: bool = False,
) -> Tuple[gpytorch.models.ApproximateGP, torch.nn.Parameter, List[Dict]]:
    """GT-aligned training: ELBO with beta_kl, noise_floor. Uses train_x (already normalized)."""
    model.train()
    if init_noise_log_param is None:
        noise_param = torch.nn.Parameter(
            torch.log(torch.tensor(config.noise_init, device=train_x.device))
        )
    else:
        noise_param = torch.nn.Parameter(
            init_noise_log_param.detach().to(train_x.device).clone()
        )
    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": [noise_param]}],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = None
    if config.cosine_annealing:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.first_restart,
            T_mult=config.restart_multiplier,
            eta_min=config.eta_min,
        )
    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    num_data = train_x.shape[0]
    history: List[Dict] = []

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        epoch_elbo = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            output = model(xb)
            kl = model.variational_strategy.kl_divergence().sum()
            samples = output.rsample(torch.Size([config.S]))
            params_pos = F.softplus(samples) + 1e-6
            pred = torch.matmul(params_pos, design_t)
            pred_norm = (pred - stress_mean) / stress_std
            yb_norm = (yb - stress_mean) / stress_std
            noise = F.softplus(noise_param) + config.noise_floor
            log_prob = torch.distributions.Normal(pred_norm, noise).log_prob(yb_norm)

            scale = num_data / yb.shape[0]
            expected_log_prob = log_prob.mean(0).sum()
            elbo = scale * expected_log_prob - config.beta_kl * kl
            loss = -elbo

            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            epoch_loss += loss.item()
            epoch_elbo += elbo.item() / num_data

        history.append(
            {"loss": epoch_loss, "elbo_per_point": epoch_elbo, "noise": F.softplus(noise_param).item()}
        )
        if (epoch + 1) % config.log_every == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1}/{config.epochs} loss={epoch_loss:.3f} "
                f"elbo/pt={epoch_elbo:.4f} noise={F.softplus(noise_param).item():.4e}"
            )

    return model, noise_param, history


def predict_params_and_stress(
    model: gpytorch.models.ApproximateGP,
    design_t: torch.Tensor,
    z: torch.Tensor,
    z_mean: torch.Tensor,
    z_std: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Predict theta and stress. Z is normalized internally. Pass raw PC scores."""
    model.eval()
    with torch.no_grad():
        z_norm = normalize_z(z, z_mean, z_std)
        output = model(z_norm)
        params_pos = F.softplus(output.mean) + 1e-6
        preds = torch.matmul(params_pos, design_t)
        return params_pos, preds
