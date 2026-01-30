"""Loss functions for discrete diffusion training."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..diffusion.noise_schedule import GeometricNoise
from ..diffusion.perturbation import perturb_batch


def score_entropy_loss(
    score_log: torch.Tensor,
    sigma_bar: torch.Tensor,
    x_t: torch.Tensor,
    x0: torch.Tensor,
    vocab_size: int,
    clamp_exp: float = 30.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute the Score Entropy Loss (Eq. 7) without outer sigma_t multiplier.

    This trains the model to predict log probability ratios for denoising.

    Args:
        score_log: [B, L, V] model outputs = log s_theta(x_t, sigma_bar_t)
        sigma_bar: [B, 1] accumulated noise levels
        x_t: [B, L] noised tokens
        x0: [B, L] clean tokens
        vocab_size: Size of vocabulary
        clamp_exp: Clamp exponent to prevent overflow
        eps: Small constant for numerical stability

    Returns:
        loss: [B, L] per-token loss (before weighting by sigma_t)
    """
    B, L, V = score_log.shape

    # 1) Precompute helpers for forward diffusion probabilities
    # Stably compute exp(sigma_bar) - 1
    esigm1 = torch.where(
        sigma_bar < 0.5,
        torch.expm1(sigma_bar),  # More stable for small values
        torch.exp(sigma_bar) - 1,
    )

    # Ratio of move probability to stay probability from Eq. (3)
    ratio = esigm1 / (esigm1 + vocab_size)
    ratio = torch.clamp(ratio, min=eps)  # Avoid division by zero

    # Clamp log scores and compute scores
    score_log = torch.clamp(score_log, max=clamp_exp)
    s = torch.exp(score_log)

    # Helper to extract values at specific indices
    def take_at(tensor: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Extract values at indices. tensor: [B,L,V], idx: [B,L] -> [B,L]"""
        return torch.gather(tensor, dim=-1, index=idx[..., None]).squeeze(-1)

    # 2) Positive term: E_y[s_theta(x_t, y)] for y != x_t
    s_scaled = s / (vocab_size - 1)  # Average over V-1 tokens
    s_mean_all = s_scaled.sum(dim=-1)  # Sum over all vocab
    s_at_xt = take_at(s_scaled, x_t)  # Value at current token
    pos_term = s_mean_all - s_at_xt  # Exclude current token

    # 3) Negative term: E_y[a_y * log s_theta(x_t, y)] for y != x_t
    # where a_y = p(y|x0) / p(x_t|x0) from forward process

    log_s_mean = score_log.sum(dim=-1) / (vocab_size - 1)
    log_s_at_xt = take_at(score_log, x_t) / (vocab_size - 1)
    base_neg = log_s_mean - log_s_at_xt

    # Split into two cases: no-move (x_t == x0) vs move (x_t != x0)
    no_move = x_t == x0

    # Case 1: No move (x_t == x0)
    # All y != x_t have a_y = ratio
    neg_term_no_move = ratio * base_neg

    # Case 2: Move (x_t != x0)
    # y == x0 has a_y = 1/ratio
    # y != x0 and y != x_t have a_y = 1
    neg_term_move = take_at(score_log, x0) / (ratio * (vocab_size - 1)) + (vocab_size - 2) * base_neg / (vocab_size - 1)

    neg_term = torch.where(no_move, neg_term_no_move, neg_term_move)

    # 4) Constant term: K(a) = sum_y a_y(log a_y - 1) for y != x_t

    # Case 1: No move
    const_no_move = ratio * (torch.log(ratio) - 1.0)

    # Case 2: Move
    const_move = ((-torch.log(ratio) - 1.0) / ratio - (vocab_size - 2)) / (vocab_size - 1)

    const_term = torch.where(no_move, const_no_move, const_move)

    # Final loss: positive - negative + constant
    loss = pos_term - neg_term + const_term  # [B, L]

    return loss


def compute_loss(
    model: nn.Module,
    x0: torch.Tensor,
    noise_schedule: GeometricNoise,
    vocab_size: int,
    t: Optional[torch.Tensor] = None,
    x_t: Optional[torch.Tensor] = None,
    sampling_eps: float = 1e-3,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the full discrete diffusion loss for a batch.

    Args:
        model: DiscreteDiffusionTransformer
        x0: [B, L] clean token sequences
        noise_schedule: GeometricNoise instance
        vocab_size: Size of vocabulary
        t: [B] timesteps in [0,1], sampled uniformly if None
        x_t: [B, L] noised tokens, generated if None
        sampling_eps: Epsilon to avoid t=0 or t=1
        mask: Optional boolean mask [B, L] where True = preserve (don't compute loss)

    Returns:
        loss: Scalar loss value
        metrics: Dict with diagnostic metrics
    """
    B, L = x0.shape
    device = x0.device

    # 1) Sample timesteps if not provided
    # Note: sampling_eps prevents t from getting too close to 0 or 1
    # At very low t (high noise), the loss can become numerically unstable
    if t is None:
        t = (1 - 2 * sampling_eps) * torch.rand(B, device=device) + sampling_eps

    # 2) Get noise levels from schedule
    sigma_bar, sigma = noise_schedule(t)  # Both [B]

    # 3) Generate noised sequences if not provided
    if x_t is None:
        x_t = perturb_batch(x0, sigma_bar, vocab_size, mask=mask)

    # 4) Model forward pass
    output = model(x_t, sigma_bar)
    log_score = output["logits"]  # [B, L, V]

    # 5) Compute score entropy loss (per-token)
    loss_per_token = score_entropy_loss(
        log_score,
        sigma_bar[:, None],  # [B, 1]
        x_t,
        x0,
        vocab_size,
    )  # [B, L]

    # 6) Weight by sigma(t) and average
    # This is importance weighting - higher noise rates get more weight
    # Clamp per-token loss to prevent numerical issues at boundary conditions
    # (can go slightly negative at very low sigma with confident predictions)
    loss_per_token = torch.clamp(loss_per_token, min=0.0)
    weighted_loss = sigma[:, None] * loss_per_token  # [B, L]

    # Apply mask: only compute loss on non-masked positions
    if mask is not None:
        # mask=True means "preserve this position" (don't compute loss)
        # Invert: compute loss where mask=False
        loss_mask = ~mask  # [B, L]
        weighted_loss = weighted_loss * loss_mask.float()

        # Average over non-masked positions only
        num_active = loss_mask.sum()
        if num_active > 0:
            loss = weighted_loss.sum() / num_active
        else:
            loss = torch.tensor(0.0, device=weighted_loss.device)
    else:
        # No masking: average over all positions
        loss = weighted_loss.mean()  # Scalar

    # 7) Collect metrics for logging
    metrics = {
        "loss": loss.item(),
        "avg_sigma_bar": sigma_bar.mean().item(),
        "avg_sigma": sigma.mean().item(),
        "loss_per_token_mean": loss_per_token.mean().item(),
    }

    return loss, metrics
