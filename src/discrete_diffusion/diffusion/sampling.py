"""Sampling (reverse diffusion) for discrete diffusion models."""

from typing import Optional

import torch

from .noise_schedule import GeometricNoise


def transition(x: torch.Tensor, delta_sigma: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """
    Forward transition kernel: exp(sigma_t^{delta_t} Q^{tok})(x_t, y)

    Computes the finite-time forward diffusion probability of moving from
    token x_t to y after a noise increment of delta_sigma.

    Args:
        x: Current tokens [B, L]
        delta_sigma: Change in noise level [B, 1]
        vocab_size: Size of vocabulary

    Returns:
        Transition probabilities [B, L, V]
    """
    # Uniform mixing term from exp(delta_sigma * Q^{tok})
    # Base probability for moving to any other token
    base_prob = (1 - torch.exp(-delta_sigma[..., None])) / vocab_size  # [B, 1, 1]

    # Initialize all positions with base_prob
    trans = torch.ones(*x.shape, vocab_size, device=x.device) * base_prob  # [B, L, V]

    # Zero out the current token position (can't count self-transition in base)
    trans = trans.scatter(-1, x[..., None], torch.zeros_like(trans[..., :1]))

    # Fill diagonal so probabilities sum to 1
    diag_fill = 1 - trans.sum(dim=-1, keepdim=True)
    trans = trans.scatter(-1, x[..., None], diag_fill)

    return trans


def staggered_score(
    score: torch.Tensor, delta_sigma: torch.Tensor, vocab_size: int
) -> torch.Tensor:
    """
    Applies the inverse exponential operator: exp(-sigma_t^{delta_t} Q^{tok}) s_theta(x_t, t)

    This "staggered" score correction accounts for the finite time-step delta_t.

    Args:
        score: Model outputs (exponentiated log scores) [B, L, V]
        delta_sigma: Change in noise level [B, 1]
        vocab_size: Size of vocabulary (unused, kept for API compatibility)

    Returns:
        Adjusted scores [B, L, V]
    """
    V = score.shape[-1]
    exp_factor = torch.exp(-delta_sigma)[..., None]  # [B, 1, 1]
    correction = ((exp_factor - 1) / (V * exp_factor)) * score.sum(dim=-1, keepdim=True)
    return correction + score / exp_factor


def sample_categorical(probs: torch.Tensor) -> torch.Tensor:
    """
    Sample from categorical distribution.

    Args:
        probs: Probability distribution [B, L, V]

    Returns:
        Sampled tokens [B, L]
    """
    B, L, V = probs.shape

    # Reshape to [B*L, V] for easier sampling
    probs_flat = probs.reshape(-1, V)

    # Sample from categorical distribution
    samples = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)

    # Reshape back to [B, L]
    samples = samples.reshape(B, L)

    return samples


def create_timestep_schedule(
    num_steps: int,
    eps: float = 1e-5,
    schedule: str = "linear",
    device: str = "cuda",
) -> torch.Tensor:
    """
    Create a timestep schedule for sampling.

    Args:
        num_steps: Number of denoising steps
        eps: Small epsilon to avoid t=0
        schedule: Type of schedule:
            - 'linear': Linear spacing in t (default, matches reference)
            - 'quadratic': Quadratic spacing, more steps at high noise
            - 'cosine': Cosine spacing, smooth transitions
            - 'log': Logarithmic spacing, uniform in log-sigma space

    Returns:
        Tensor of timesteps from 1 to eps, shape [num_steps + 1]
    """
    if schedule == "linear":
        # Linear spacing in t-space (reference implementation)
        timesteps = torch.linspace(1, eps, num_steps + 1, device=device)

    elif schedule == "quadratic":
        # Quadratic: more steps at high noise levels
        linear = torch.linspace(0, 1, num_steps + 1, device=device)
        timesteps = 1 - (1 - eps) * (linear**2)

    elif schedule == "cosine":
        # Cosine: smooth S-curve transition
        linear = torch.linspace(0, 1, num_steps + 1, device=device)
        timesteps = eps + (1 - eps) * (1 + torch.cos(linear * torch.pi)) / 2

    elif schedule == "log":
        # Log: uniform spacing in log-sigma space
        timesteps = torch.linspace(1, eps, num_steps + 1, device=device)

    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    return timesteps


@torch.no_grad()
def sample_diffusion(
    model: torch.nn.Module,
    noise_schedule: GeometricNoise,
    vocab_size: int,
    batch_size: int = 1,
    seq_len: int = 256,
    num_steps: int = 128,
    eps: float = 1e-5,
    device: str = "cuda",
    schedule: str = "quadratic",
    progress_callback: Optional[callable] = None,
) -> torch.Tensor:
    """
    Generate samples using the reverse diffusion process.

    Args:
        model: Trained DiscreteDiffusionTransformer
        noise_schedule: GeometricNoise instance
        vocab_size: Size of vocabulary
        batch_size: Number of sequences to generate
        seq_len: Length of sequences to generate
        num_steps: Number of denoising steps
        eps: Small epsilon to avoid t=0
        device: Device to run on
        schedule: Timestep schedule type ('linear', 'quadratic', 'cosine')
        progress_callback: Optional callback(step, num_steps, x, sigma) for progress updates

    Returns:
        Generated sequences [batch_size, seq_len]
    """
    model.eval()

    # Initialize with random tokens
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Create timestep schedule
    timesteps = create_timestep_schedule(num_steps, eps, schedule, device)

    # Reverse diffusion loop
    for i in range(num_steps + 1):
        # Current timestep
        t = timesteps[i] * torch.ones(batch_size, device=device)

        # Get current noise level
        curr_sigma_bar = noise_schedule(t)[0]  # [B]

        if i < num_steps:
            # Intermediate steps - use next timestep from schedule
            t_next = timesteps[i + 1] * torch.ones(batch_size, device=device)
            next_sigma_bar = noise_schedule(t_next)[0]  # [B]
            delta_sigma = curr_sigma_bar - next_sigma_bar  # [B]
        else:
            # Final step: denoise completely
            delta_sigma = curr_sigma_bar  # [B]

        # Get model predictions (log scores)
        log_score = model(x, curr_sigma_bar)
        if isinstance(log_score, dict):
            log_score = log_score["logits"]

        # Clamp log scores to prevent overflow in exp
        log_score = torch.clamp(log_score, min=-30, max=30)

        # Convert to scores
        score = torch.exp(log_score)  # [B, L, V]

        # Compute staggered scores
        stag_score = staggered_score(score, delta_sigma[:, None], vocab_size)  # [B, L, V]

        # Compute transition probabilities
        trans_probs = transition(x, delta_sigma[:, None], vocab_size)  # [B, L, V]

        # Combine: element-wise product
        probs = stag_score * trans_probs  # [B, L, V]

        # Ensure probabilities are valid (clamp negatives, handle inf/nan)
        probs = torch.clamp(probs, min=0)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=1e10, neginf=0.0)

        # Normalize
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)

        # Sample next state
        x = sample_categorical(probs)  # [B, L]

        # Progress callback
        if progress_callback is not None:
            progress_callback(i, num_steps, x, curr_sigma_bar)

    return x
