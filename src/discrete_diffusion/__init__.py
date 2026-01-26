"""
Discrete Diffusion Language Model

A PyTorch implementation of discrete diffusion transformers for text generation,
compatible with HuggingFace Transformers.

Example usage:

    # Simple generation with pipeline
    from discrete_diffusion import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained("./checkpoints/best_model_hf")
    texts = pipe.generate(num_samples=5, num_steps=256)

    # Training
    from discrete_diffusion import (
        DiscreteDiffusionConfig,
        DiscreteDiffusionTransformer,
        CharacterLevelTokenizer,
        Trainer,
        TrainingConfig,
        setup_dataloaders,
    )

    tokenizer = CharacterLevelTokenizer()
    config = DiscreteDiffusionConfig()
    model = DiscreteDiffusionTransformer(config)

    train_loader, val_loader, _, _ = setup_dataloaders()
    trainer = Trainer(model, train_loader, val_loader)
    trainer.train()
"""

__version__ = "0.1.0"

# Model
from .modeling import DiscreteDiffusionConfig, DiscreteDiffusionTransformer

# Tokenizer and data
from .data import CharacterLevelTokenizer, setup_dataloaders

# Diffusion components
from .diffusion import GeometricNoise, perturb_batch, sample_diffusion

# Training
from .training import Trainer, TrainingConfig, compute_loss, score_entropy_loss

# Pipeline (high-level API)
from .pipeline import DiffusionPipeline

__all__ = [
    # Version
    "__version__",
    # Model
    "DiscreteDiffusionConfig",
    "DiscreteDiffusionTransformer",
    # Data
    "CharacterLevelTokenizer",
    "setup_dataloaders",
    # Diffusion
    "GeometricNoise",
    "perturb_batch",
    "sample_diffusion",
    # Training
    "Trainer",
    "TrainingConfig",
    "compute_loss",
    "score_entropy_loss",
    # Pipeline
    "DiffusionPipeline",
]
