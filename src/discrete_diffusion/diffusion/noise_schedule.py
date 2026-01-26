"""Noise schedule implementations for discrete diffusion."""

from typing import Tuple

import torch


class GeometricNoise:
    """
    Geometric noise schedule for discrete diffusion.

    Interpolates geometrically (in log-space) between sigma_min and sigma_max.
    This provides smoother noise progression across orders of magnitude.

    Args:
        sigma_min: Minimum noise level at t=0
        sigma_max: Maximum noise level at t=1
    """

    def __init__(self, sigma_min: float = 1e-4, sigma_max: float = 20.0):
        self.sigmas = 1.0 * torch.tensor([sigma_min, sigma_max])

    @property
    def sigma_min(self) -> float:
        return self.sigmas[0].item()

    @property
    def sigma_max(self) -> float:
        return self.sigmas[1].item()

    def rate_noise(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute instantaneous noise rate sigma(t) = d/dt sigma_bar(t).

        Args:
            t: Timestep in [0, 1]

        Returns:
            Noise rate at time t
        """
        return (
            self.sigmas[0] ** (1 - t)
            * self.sigmas[1] ** t
            * (self.sigmas[1].log() - self.sigmas[0].log())
        )

    def total_noise(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute accumulated noise sigma_bar(t).

        Args:
            t: Timestep in [0, 1]

        Returns:
            Total accumulated noise at time t
        """
        return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t

    def __call__(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get both total and rate noise at timestep t.

        Args:
            t: Timestep in [0, 1]

        Returns:
            Tuple of (total_noise, rate_noise)
        """
        return self.total_noise(t), self.rate_noise(t)
