"""Training utilities for discrete diffusion models."""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from ..data.tokenizer import CharacterLevelTokenizer
from ..diffusion.noise_schedule import GeometricNoise
from .loss import compute_loss

# Optional: bitsandbytes for 8-bit optimizers
try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False

# Vocab size is fixed and determined by the tokenizer
VOCAB_SIZE = CharacterLevelTokenizer().vocab_size


@dataclass
class TrainingConfig:
    """Configuration for training."""

    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    gradient_clip: float = 1.0
    num_epochs: int = 30
    sigma_min: float = 1e-4
    sigma_max: float = 20.0
    sampling_eps: float = 1e-3
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 5  # Save every N epochs
    save_every_steps: int = 0  # Save every N steps (0 = disabled)
    save_hf_format: bool = True
    mixed_precision: Literal["no", "fp16", "bf16"] = "no"
    gradient_checkpointing: bool = False
    gradient_accumulation_steps: int = 1
    use_8bit_optimizer: bool = False


@dataclass
class TrainerCallback:
    """Base callback for training events."""

    def on_epoch_start(self, epoch: int, trainer: "Trainer") -> None:
        pass

    def on_epoch_end(self, epoch: int, train_loss: float, val_loss: float, trainer: "Trainer") -> None:
        pass

    def on_batch_end(self, batch_idx: int, loss: float, metrics: Dict, trainer: "Trainer") -> None:
        pass

    def on_train_end(self, trainer: "Trainer") -> None:
        pass


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    val_loss: float,
    checkpoint_dir: str,
    filename: str,
    save_hf: bool = False,
    scaler: Optional[GradScaler] = None,
) -> str:
    """
    Save a training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch number
        global_step: Global step count across all epochs
        val_loss: Validation loss at this checkpoint
        checkpoint_dir: Directory to save to
        filename: Checkpoint filename (without extension)
        save_hf: Whether to also save in HuggingFace format
        scaler: Optional GradScaler state to save (for mixed precision)

    Returns:
        Path to saved checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
    }

    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    checkpoint_path = os.path.join(checkpoint_dir, f"{filename}.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path} (epoch {epoch}, step {global_step})")

    if save_hf:
        hf_dir = os.path.join(checkpoint_dir, f"{filename}_hf")
        model.save_pretrained(hf_dir)
        print(f"Saved HuggingFace checkpoint: {hf_dir}")

    return checkpoint_path


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: str,
    device: str,
    scaler: Optional[GradScaler] = None,
) -> Tuple[int, int, float]:
    """
    Load a training checkpoint.

    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        checkpoint_path: Path to checkpoint file
        device: Device to load to
        scaler: Optional GradScaler to load state into (for mixed precision)

    Returns:
        Tuple of (epoch, global_step, val_loss) from the checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    global_step = checkpoint.get("global_step", 0)
    val_loss = checkpoint.get("val_loss", float("inf"))

    print(f"Loaded checkpoint from epoch {epoch}, step {global_step} with val_loss {val_loss:.4f}")
    return epoch, global_step, val_loss


class Trainer:
    """
    Trainer for discrete diffusion models.

    Handles the training loop, validation, checkpointing, and callbacks.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[TrainingConfig] = None,
        device: str = "cuda",
        callbacks: Optional[List[TrainerCallback]] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: DiscreteDiffusionTransformer to train
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            config: Training configuration
            device: Device to train on
            callbacks: List of callbacks for training events
        """
        self.config = config or TrainingConfig()
        self.device = device
        self.callbacks = callbacks or []

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab_size = VOCAB_SIZE

        # Noise schedule
        self.noise_schedule = GeometricNoise(
            sigma_min=self.config.sigma_min,
            sigma_max=self.config.sigma_max,
        )

        # Optimizer
        if self.config.use_8bit_optimizer:
            if not HAS_BITSANDBYTES:
                raise ImportError(
                    "bitsandbytes is required for 8-bit optimizer. "
                    "Install with: pip install bitsandbytes"
                )
            self.optimizer = bnb.optim.AdamW8bit(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay,
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay,
            )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Mixed precision setup
        self.mixed_precision = self.config.mixed_precision
        self.scaler = GradScaler(self.device) if self.mixed_precision == "fp16" else None
        self.autocast_dtype = {
            "no": None,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }.get(self.mixed_precision)

        if self.mixed_precision != "no":
            print(f"Mixed precision training enabled: {self.mixed_precision}")

        # Gradient checkpointing setup
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")

        # Gradient accumulation setup
        if self.config.gradient_accumulation_steps > 1:
            print(f"Gradient accumulation enabled: {self.config.gradient_accumulation_steps} steps")

        # 8-bit optimizer setup
        if self.config.use_8bit_optimizer:
            print("8-bit Adam optimizer enabled")

    @torch.no_grad()
    def validate(self) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            x0 = batch["input_ids"].to(self.device)

            # Use autocast if mixed precision is enabled
            if self.autocast_dtype is not None:
                with autocast(device_type=self.device, dtype=self.autocast_dtype):
                    loss, _ = compute_loss(
                        model=self.model,
                        x0=x0,
                        noise_schedule=self.noise_schedule,
                        vocab_size=self.vocab_size,
                        sampling_eps=self.config.sampling_eps,
                    )
            else:
                loss, _ = compute_loss(
                    model=self.model,
                    x0=x0,
                    noise_schedule=self.noise_schedule,
                    vocab_size=self.vocab_size,
                    sampling_eps=self.config.sampling_eps,
                )

            total_loss += loss.item()
            num_batches += 1

        self.model.train()
        return total_loss / num_batches if num_batches > 0 else float("inf")

    def train_epoch(self) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            x0 = batch["input_ids"].to(self.device)

            # Compute loss with autocast if enabled
            if self.autocast_dtype is not None:
                with autocast(device_type=self.device, dtype=self.autocast_dtype):
                    loss, metrics = compute_loss(
                        model=self.model,
                        x0=x0,
                        noise_schedule=self.noise_schedule,
                        vocab_size=self.vocab_size,
                        sampling_eps=self.config.sampling_eps,
                    )
            else:
                loss, metrics = compute_loss(
                    model=self.model,
                    x0=x0,
                    noise_schedule=self.noise_schedule,
                    vocab_size=self.vocab_size,
                    sampling_eps=self.config.sampling_eps,
                )

            # Gradient accumulation: scale loss and only step optimizer every N batches
            is_accumulating = self.config.gradient_accumulation_steps > 1
            should_step = (batch_idx + 1) % self.config.gradient_accumulation_steps == 0

            # Zero gradients at the start of accumulation
            if batch_idx % self.config.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()

            # Scale loss for gradient accumulation
            if is_accumulating:
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            if self.scaler is not None:
                # FP16 with GradScaler
                self.scaler.scale(loss).backward()

                if should_step:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.global_step += 1
            else:
                # BF16 or no mixed precision
                loss.backward()

                if should_step:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip
                    )
                    self.optimizer.step()
                    self.global_step += 1

            # Track unscaled loss for logging
            epoch_loss += loss.item() * (self.config.gradient_accumulation_steps if is_accumulating else 1)
            num_batches += 1

            # Step-based checkpointing
            if should_step and self.config.save_every_steps > 0:
                if self.global_step % self.config.save_every_steps == 0:
                    # Quick checkpoint without validation (to avoid slowing down training)
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        self.current_epoch,
                        self.global_step,
                        self.best_val_loss,  # Use best known val loss
                        self.config.checkpoint_dir,
                        f"checkpoint_step_{self.global_step}",
                        save_hf=False,  # Don't save HF format for frequent checkpoints
                        scaler=self.scaler,
                    )

            # Callbacks
            for callback in self.callbacks:
                callback.on_batch_end(batch_idx, loss.item(), metrics, self)

            # Print progress
            if batch_idx % 100 == 0:
                print(
                    f"Epoch {self.current_epoch}, Step {self.global_step}, Batch {batch_idx}, Loss: {loss.item():.4f}"
                )

        return epoch_loss / num_batches

    def train(
        self,
        num_epochs: Optional[int] = None,
        resume_from: Optional[str] = None,
    ) -> None:
        """
        Run the full training loop.

        Args:
            num_epochs: Number of epochs (overrides config if provided)
            resume_from: Path to checkpoint to resume from
        """
        num_epochs = num_epochs or self.config.num_epochs
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        # Resume from checkpoint if specified
        if resume_from is not None:
            self.current_epoch, self.global_step, self.best_val_loss = load_checkpoint(
                self.model, self.optimizer, resume_from, self.device, self.scaler
            )
            self.current_epoch += 1  # Start from next epoch

        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        if self.config.save_every_steps > 0:
            print(f"Step-based checkpointing enabled: every {self.config.save_every_steps} steps")
        if resume_from:
            print(
                f"Resuming from epoch {self.current_epoch}, step {self.global_step}, best val_loss: {self.best_val_loss:.4f}"
            )
        print("=" * 60)

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            # Callbacks
            for callback in self.callbacks:
                callback.on_epoch_start(epoch, self)

            # Train
            train_loss = self.train_epoch()
            print(f"Epoch {epoch} complete. Average train loss: {train_loss:.4f}")

            # Validate
            val_loss = self.validate()
            print(f"Validation loss: {val_loss:.4f}")

            # Callbacks
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, train_loss, val_loss, self)

            # Save best checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    self.global_step,
                    val_loss,
                    self.config.checkpoint_dir,
                    "best_model",
                    save_hf=self.config.save_hf_format,
                    scaler=self.scaler,
                )
                print(f"New best model! Val loss: {val_loss:.4f}")

            print("-" * 60)

            # Save periodic checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    self.global_step,
                    val_loss,
                    self.config.checkpoint_dir,
                    f"model_epoch_{epoch+1}",
                    save_hf=self.config.save_hf_format,
                    scaler=self.scaler,
                )

        # Final checkpoint
        final_val_loss = self.validate()
        save_checkpoint(
            self.model,
            self.optimizer,
            num_epochs - 1,
            self.global_step,
            final_val_loss,
            self.config.checkpoint_dir,
            "final_model",
            save_hf=self.config.save_hf_format,
            scaler=self.scaler,
        )

        # Callbacks
        for callback in self.callbacks:
            callback.on_train_end(self)

        print("=" * 60)
        print("Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Final validation loss: {final_val_loss:.4f}")
