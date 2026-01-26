"""Visualization utilities for discrete diffusion."""

import os
import textwrap
import time

import torch
from torch.utils.data import DataLoader

from ..data.tokenizer import CharacterLevelTokenizer
from ..diffusion.noise_schedule import GeometricNoise
from ..diffusion.perturbation import perturb_batch


def clear_terminal():
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def print_wrapped(text: str, width: int = 80, end: str = "\n"):
    """Print text with word wrapping."""
    wrapped = textwrap.fill(text, width=width)
    print(wrapped, end=end)


def animate_noising(
    loader: DataLoader,
    tokenizer: CharacterLevelTokenizer,
    sigma_min: float = 1e-4,
    sigma_max: float = 20.0,
    num_steps: int = 100,
    delay: float = 0.1,
):
    """
    Animate the forward diffusion (noising) process in the terminal.

    This visualizes how clean text gets progressively corrupted,
    which is useful for understanding the training process.

    Args:
        loader: DataLoader with tokenized sequences
        tokenizer: CharacterLevelTokenizer for decoding
        sigma_min: Minimum noise level
        sigma_max: Maximum noise level
        num_steps: Number of animation steps
        delay: Delay between frames (seconds)
    """
    # Get a batch
    batch = next(iter(loader))["input_ids"]

    # Use the same noise schedule as training
    noise_schedule = GeometricNoise(sigma_min=sigma_min, sigma_max=sigma_max)

    # Sweep t from 0 to 1
    timesteps = torch.linspace(0, 1, num_steps + 2)

    # Show original
    clear_terminal()
    print("=" * 80)
    print("ORIGINAL TEXT (sigma_bar = 0.000)")
    print("=" * 80)
    print_wrapped(tokenizer.decode(batch[0].tolist()), width=80)
    print()
    print("Starting diffusion in 2 seconds...")
    time.sleep(2.0)

    # Animate through noise levels
    for i, t in enumerate(timesteps):
        # Get sigma_bar from the geometric schedule
        sigma_bar = noise_schedule.total_noise(t)

        batch_pert = perturb_batch(batch, sigma_bar, tokenizer.vocab_size)
        clear_terminal()

        # Progress bar
        progress = t.item()
        bar_length = 40
        filled = int(bar_length * progress)
        bar = "\u2588" * filled + "\u2591" * (bar_length - filled)

        print(f"t = {t:.3f}, sigma_bar(t) = {sigma_bar:.4f}")
        print(f"Progress: [{bar}] {progress*100:.1f}%")
        print("=" * 80)

        decoded_text = tokenizer.decode(batch_pert[0].tolist())
        print_wrapped(decoded_text, width=80)
        print("=" * 80)

        # Corruption stats
        corruption_rate = (batch_pert[0] != batch[0]).float().mean()
        print(f"Corruption rate: {corruption_rate*100:.1f}%")

        time.sleep(delay)

    print("\nDiffusion animation complete!")
