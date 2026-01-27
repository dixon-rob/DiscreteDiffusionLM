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

# Get absolute path to configs directory
SCRIPT_DIR = Path(__file__).parent.resolve()
CONFIG_PATH = str(SCRIPT_DIR.parent / "configs")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="training/default")
def main(cfg: DictConfig) -> None:
    """
    Train a discrete diffusion model.

    Usage:
        python scripts/train.py
        python scripts/train.py learning_rate=1e-4 num_epochs=50
        python scripts/train.py --config-name=training/default model=small
    """
    # Disable struct mode to allow dynamic access
    OmegaConf.set_struct(cfg, False)

    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Load model config (composed from configs/model/base.yaml via defaults)
    model_cfg = cfg.model

    # Setup data
    train_loader, val_loader, _, tokenizer = setup_dataloaders(
        dataset_name=cfg.dataset_name,
        train_samples=cfg.train_samples,
        val_samples=cfg.val_samples,
        context_len=cfg.context_length,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    # Create model config
    config = DiscreteDiffusionConfig(
        block_size=model_cfg.block_size,
        n_layer=model_cfg.n_layer,
        n_head=model_cfg.n_head,
        n_embd=model_cfg.n_embd,
        cond_dim=model_cfg.cond_dim,
        dropout=model_cfg.dropout,
        bias=model_cfg.bias,
        gradient_checkpointing=cfg.gradient_checkpointing,
    )

    # Create model
    model = DiscreteDiffusionTransformer(config)
    print(f"Model parameters: {model.get_num_params()/1e6:.2f}M")

    # Create training config
    training_config = TrainingConfig(
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        gradient_clip=cfg.gradient_clip,
        num_epochs=cfg.num_epochs,
        sigma_min=cfg.sigma_min,
        sigma_max=cfg.sigma_max,
        sampling_eps=cfg.sampling_eps,
        checkpoint_dir=cfg.checkpoint_dir,
        save_every=cfg.save_every,
        save_every_steps=cfg.get("save_every_steps", 0),
        save_hf_format=cfg.save_hf_format,
        mixed_precision=cfg.mixed_precision,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        use_8bit_optimizer=cfg.use_8bit_optimizer,
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
