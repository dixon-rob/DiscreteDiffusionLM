#!/usr/bin/env python
"""Generation script for discrete diffusion models."""

import argparse
import sys
import textwrap
from pathlib import Path

import torch

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from discrete_diffusion import DiffusionPipeline


def main():
    """
    Generate text samples from a trained model.

    Usage:
        python scripts/generate.py --model ./checkpoints/best_model_hf
        python scripts/generate.py --model ./checkpoints/best_model.pth --num-samples 5
        python scripts/generate.py --model ./checkpoints/best_model_hf --visualize
    """
    parser = argparse.ArgumentParser(description="Generate text with discrete diffusion")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model (HF directory or .pth file)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=256,
        help="Sequence length to generate",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=256,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="quadratic",
        choices=["linear", "quadratic", "cosine"],
        help="Timestep schedule",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show live visualization (only for single sample)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (auto-detected if not specified)",
    )

    args = parser.parse_args()

    # Device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Load pipeline
    print(f"Loading model from {args.model}...")
    pipe = DiffusionPipeline.from_pretrained(args.model, device=device)
    print(f"Model loaded with {pipe.model.get_num_params()/1e6:.2f}M parameters")

    # Generate
    if args.visualize and args.num_samples == 1:
        print("\nGenerating with visualization...")
        text = pipe.generate_with_visualization(
            seq_len=args.seq_len,
            num_steps=args.num_steps,
            schedule=args.schedule,
        )
        print("\n" + "=" * 80)
        print("FINAL OUTPUT:")
        print("=" * 80)
        print(textwrap.fill(text, width=80))
    else:
        if args.visualize and args.num_samples > 1:
            print("Note: Visualization only works with single sample, generating without visualization")

        print(f"\nGenerating {args.num_samples} sample(s)...")
        texts = pipe.generate(
            num_samples=args.num_samples,
            seq_len=args.seq_len,
            num_steps=args.num_steps,
            schedule=args.schedule,
        )

        print("\n" + "=" * 80)
        print("GENERATED SAMPLES")
        print("=" * 80)
        for i, text in enumerate(texts):
            print(f"\nSample {i+1}:")
            print("-" * 80)
            print(textwrap.fill(text, width=80))


if __name__ == "__main__":
    main()
