#!/usr/bin/env python
"""Training script for discrete diffusion model."""

import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from discrete_diffusion import (
    DiscreteDiffusionConfig,
    DiscreteDiffusionTransformer,
    Trainer,
    TrainingConfig,
    setup_dataloaders,
)


@hydra.main(version_base=None, config_path="../configs", config_name="training/default")
def main(cfg: DictConfig) -> None:
    """
    Train a discrete diffusion model.

    Usage:
        python scripts/train.py
        python scripts/train.py learning_rate=1e-4 num_epochs=50
        python scripts/train.py --config-name=training/default model=small
    """
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Load model config if specified, otherwise use defaults
    model_cfg = cfg.get("model", {})

    # Setup data
    train_loader, val_loader, _, tokenizer = setup_dataloaders(
        dataset_name=cfg.get("dataset_name", "roneneldan/TinyStories"),
        train_samples=cfg.get("train_samples", 50000),
        val_samples=cfg.get("val_samples", 2000),
        context_len=cfg.get("context_length", 256),
        batch_size=cfg.get("batch_size", 32),
        num_workers=cfg.get("num_workers", 2),
    )

    # Create model config
    config = DiscreteDiffusionConfig(
        block_size=model_cfg.get("block_size", 256),
        n_layer=model_cfg.get("n_layer", 10),
        n_head=model_cfg.get("n_head", 10),
        n_embd=model_cfg.get("n_embd", 640),
        cond_dim=model_cfg.get("cond_dim", 160),
        dropout=model_cfg.get("dropout", 0.2),
        bias=model_cfg.get("bias", False),
    )

    # Create model
    model = DiscreteDiffusionTransformer(config)
    print(f"Model parameters: {model.get_num_params()/1e6:.2f}M")

    # Create training config
    training_config = TrainingConfig(
        learning_rate=cfg.get("learning_rate", 5e-5),
        weight_decay=cfg.get("weight_decay", 0.01),
        gradient_clip=cfg.get("gradient_clip", 1.0),
        num_epochs=cfg.get("num_epochs", 30),
        sigma_min=cfg.get("sigma_min", 1e-4),
        sigma_max=cfg.get("sigma_max", 20.0),
        sampling_eps=cfg.get("sampling_eps", 1e-3),
        checkpoint_dir=cfg.get("checkpoint_dir", "./checkpoints"),
        save_every=cfg.get("save_every", 5),
        save_hf_format=cfg.get("save_hf_format", True),
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device,
    )

    # Train
    resume_from = cfg.get("resume_from", None)
    trainer.train(resume_from=resume_from)


if __name__ == "__main__":
    main()
