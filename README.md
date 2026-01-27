# Discrete Diffusion Language Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch implementation of discrete diffusion for character-level text generation, based on Score Entropy Discrete Diffusion (SEDD).

## Installation

```bash
pip install -e .
```

Or with development dependencies:

```bash
pip install -e ".[dev]"
```

After installation, you can use the command-line tools:
- `ddlm-train` - Train a model
- `ddlm-generate --model <model_path> --visualize` - Generate text from a trained model
- `ddlm-ui --model <model_path>` - Launch the Gradio web interface

## Quick Start

### Generate text with a trained model

```python
from discrete_diffusion import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("./checkpoints/best_model_hf")
texts = pipe.generate(num_samples=5, num_steps=256)

for text in texts:
    print(text)
```

### Generate with live visualization

```python
pipe = DiffusionPipeline.from_pretrained("./checkpoints/best_model_hf")
text = pipe.generate_with_visualization(num_steps=256)
```

### Train a model

```python
from discrete_diffusion import (
    DiscreteDiffusionConfig,
    DiscreteDiffusionTransformer,
    Trainer,
    TrainingConfig,
    setup_dataloaders,
)

# Setup data
train_loader, val_loader, _, tokenizer = setup_dataloaders()

# Create model
config = DiscreteDiffusionConfig(
    block_size=256,
    n_layer=10,
    n_head=10,
    n_embd=640,
    cond_dim=160,
    dropout=0.2,
    bias=False,
)
model = DiscreteDiffusionTransformer(config)

# Train
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=TrainingConfig(num_epochs=30, learning_rate=5e-5),
)
trainer.train()
```

## Scripts

### Training

```bash
# Default training (using configs/training/default.yaml)
ddlm-train

# Resume from checkpoint
ddlm-train +resume_from=./checkpoints/checkpoint_latest.pth
```

### Generation

```bash
# Generate samples
ddlm-generate --model ./checkpoints/best_model_hf --num-samples 5

# With visualization (watch denoising in real-time)
ddlm-generate --model ./checkpoints/best_model_hf --visualize

# Custom sequence length (must match model's block_size)
ddlm-generate --model ./checkpoints/best_model_hf --seq-len 128
```

### Gradio Web UI

Launch an interactive web interface for text generation:

```bash
# Using the installed command (after pip install -e .)
ddlm-ui --model ./checkpoints/best_model_hf

# Or run the script directly
python scripts/gradio_app.py --model ./checkpoints/best_model_hf

# Create a public share link (accessible from anywhere)
ddlm-ui --model ./checkpoints/best_model_hf --share

# Use a custom port
ddlm-ui --model ./checkpoints/best_model_hf --port 8080

# Make accessible on local network
ddlm-ui --model ./checkpoints/best_model_hf --server-name 0.0.0.0
```

The web UI includes:
- Interactive sliders for all generation parameters
- Real-time progress tracking
- Multiple sample generation
- Copy-to-clipboard functionality
- Pre-configured example settings

## Testing

```bash
pytest                                    # Run all tests
pytest tests/test_model.py               # Run specific test file
pytest tests/test_model.py -v            # Verbose output
```

## Project Structure

```
├── configs/
│   ├── model/          # Model architecture configs
│   └── training/       # Training hyperparameter configs
├── src/discrete_diffusion/
│   ├── modeling/       # Model architecture (HuggingFace compatible)
│   ├── diffusion/      # Noise schedules and sampling
│   ├── data/           # Tokenizer and dataset utilities
│   ├── training/       # Loss functions and trainer
│   ├── utils/          # Visualization utilities
│   └── pipeline.py     # High-level generation API
├── scripts/            # Training and generation scripts
└── tests/              # Unit tests
```

## How It Works

Unlike autoregressive models that generate text left-to-right, discrete diffusion:

1. **Forward process**: Gradually corrupts text by randomly replacing characters
2. **Reverse process**: Learns to denoise, predicting which characters to restore
3. **Generation**: Starts from random characters and iteratively denoises to coherent text

The model uses **Score Entropy loss** to learn the denoising process, with a geometric noise schedule.

## API Reference

### Core Classes

- `DiffusionPipeline` - High-level generation interface
- `DiscreteDiffusionTransformer` - Main model (HuggingFace compatible)
- `DiscreteDiffusionConfig` - Model configuration
- `CharacterLevelTokenizer` - Character-level tokenizer (106 tokens)
- `GeometricNoise` - Noise schedule
- `Trainer` - Training loop with checkpointing

### Key Functions

- `sample_diffusion()` - Low-level sampling function
- `perturb_batch()` - Forward diffusion (noising)
- `score_entropy_loss()` - Training loss function
- `setup_dataloaders()` - Create data loaders for TinyStories

## Acknowledgments

This implementation is based on:

- [The Annotated Discrete Diffusion Models](https://github.com/ash80/diffusion-gpt) by ash80 - Tutorial and reference implementation
- [Score Entropy Discrete Diffusion (SEDD)](https://arxiv.org/abs/2310.16834) - The underlying paper

Training data:

- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset (CDLA-Sharing-1.0)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
