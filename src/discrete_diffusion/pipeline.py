"""High-level pipeline for text generation with discrete diffusion."""

import os
import textwrap
from typing import List, Optional, Union

import torch

from .data.tokenizer import CharacterLevelTokenizer
from .diffusion.noise_schedule import GeometricNoise
from .diffusion.sampling import sample_diffusion
from .modeling.configuration import DiscreteDiffusionConfig
from .modeling.model import DiscreteDiffusionTransformer


class DiffusionPipeline:
    """
    High-level pipeline for text generation with discrete diffusion models.

    Provides a simple interface for loading models and generating text,
    similar to HuggingFace's pipeline pattern.

    Example:
        >>> pipe = DiffusionPipeline.from_pretrained("./checkpoints/best_model_hf")
        >>> texts = pipe.generate(num_samples=3, num_steps=256)
        >>> for text in texts:
        ...     print(text)
    """

    def __init__(
        self,
        model: DiscreteDiffusionTransformer,
        tokenizer: CharacterLevelTokenizer,
        noise_schedule: GeometricNoise,
        device: str = "cuda",
    ):
        """
        Initialize the pipeline.

        Args:
            model: Trained DiscreteDiffusionTransformer
            tokenizer: CharacterLevelTokenizer
            noise_schedule: GeometricNoise instance
            device: Device to run inference on
        """
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.noise_schedule = noise_schedule
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: Optional[str] = None,
        sigma_min: float = 1e-4,
        sigma_max: float = 20.0,
    ) -> "DiffusionPipeline":
        """
        Load a pipeline from a pretrained model.

        Args:
            model_path: Path to HuggingFace format checkpoint directory,
                       or path to .pth checkpoint file
            device: Device to run on (auto-detected if None)
            sigma_min: Minimum noise level for schedule
            sigma_max: Maximum noise level for schedule

        Returns:
            Initialized DiffusionPipeline
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = CharacterLevelTokenizer()
        noise_schedule = GeometricNoise(sigma_min=sigma_min, sigma_max=sigma_max)

        # Check if it's a HuggingFace directory or a .pth file
        if os.path.isdir(model_path):
            # HuggingFace format
            model = DiscreteDiffusionTransformer.from_pretrained(model_path)
        elif model_path.endswith(".pth"):
            # PyTorch checkpoint - need config
            # Try to find config in same directory
            config_path = os.path.join(os.path.dirname(model_path), "config.json")
            if os.path.exists(config_path):
                config = DiscreteDiffusionConfig.from_pretrained(
                    os.path.dirname(model_path)
                )
            else:
                # Use default config matching trained model
                config = DiscreteDiffusionConfig(
                    vocab_size=tokenizer.vocab_size,
                    block_size=256,
                    n_layer=10,
                    n_head=10,
                    n_embd=640,
                    cond_dim=160,
                    dropout=0.0,
                    bias=False,
                )
            model = DiscreteDiffusionTransformer(config)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            raise ValueError(
                f"model_path must be a directory (HF format) or .pth file, got: {model_path}"
            )

        return cls(model, tokenizer, noise_schedule, device)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config: DiscreteDiffusionConfig,
        device: Optional[str] = None,
        sigma_min: float = 1e-4,
        sigma_max: float = 20.0,
    ) -> "DiffusionPipeline":
        """
        Load a pipeline from a training checkpoint (.pth file).

        Args:
            checkpoint_path: Path to .pth checkpoint file
            config: Model configuration
            device: Device to run on (auto-detected if None)
            sigma_min: Minimum noise level for schedule
            sigma_max: Maximum noise level for schedule

        Returns:
            Initialized DiffusionPipeline
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = CharacterLevelTokenizer()
        noise_schedule = GeometricNoise(sigma_min=sigma_min, sigma_max=sigma_max)

        model = DiscreteDiffusionTransformer(config)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        return cls(model, tokenizer, noise_schedule, device)

    def generate(
        self,
        num_samples: int = 1,
        seq_len: int = 256,
        num_steps: int = 256,
        schedule: str = "quadratic",
        return_tokens: bool = False,
    ) -> Union[List[str], torch.Tensor]:
        """
        Generate text samples.

        Args:
            num_samples: Number of samples to generate
            seq_len: Length of sequences to generate
            num_steps: Number of denoising steps (more = higher quality)
            schedule: Timestep schedule ('linear', 'quadratic', 'cosine')
            return_tokens: If True, return token tensors instead of decoded text

        Returns:
            List of generated text strings, or tensor of tokens if return_tokens=True
        """
        tokens = sample_diffusion(
            model=self.model,
            noise_schedule=self.noise_schedule,
            vocab_size=self.tokenizer.vocab_size,
            batch_size=num_samples,
            seq_len=seq_len,
            num_steps=num_steps,
            device=self.device,
            schedule=schedule,
        )

        if return_tokens:
            return tokens

        # Decode to text
        texts = []
        for i in range(tokens.shape[0]):
            text = self.tokenizer.decode(tokens[i].tolist())
            texts.append(text)

        return texts

    def generate_streaming(
        self,
        num_steps: int = 256,
        seq_len: int = 256,
        schedule: str = "quadratic",
        initial_sequence: str = None,
    ):
        """
        Generate a single sample with streaming updates.

        This is a generator function that yields intermediate denoising results
        at each step. Designed for use with Gradio or other streaming UIs.

        Args:
            num_steps: Number of denoising steps
            seq_len: Length of sequence to generate
            schedule: Timestep schedule ('linear', 'quadratic', 'cosine')
            initial_sequence: Initial text sequence (underscores will be filled in).
                            If None, defaults to all underscores.

        Yields:
            Tuple of (step_index, intermediate_text, noise_level)

        Returns:
            Final generated text string
        """
        from .diffusion.sampling import (
            create_timestep_schedule,
            sample_categorical,
            staggered_score,
            transition,
        )

        # Initialize tokens from initial sequence
        if initial_sequence is None:
            initial_sequence = "_" * seq_len

        # Encode the initial sequence
        initial_tokens = self.tokenizer.encode(initial_sequence)

        # Ensure sequence is correct length
        if len(initial_tokens) != seq_len:
            # Truncate or pad to seq_len
            if len(initial_tokens) > seq_len:
                initial_tokens = initial_tokens[:seq_len]
            else:
                initial_tokens = initial_tokens + [self.tokenizer.encode("_")[0]] * (seq_len - len(initial_tokens))

        # Convert to tensor
        x = torch.tensor([initial_tokens], device=self.device, dtype=torch.long)

        # Find underscore token ID and replace with random tokens
        underscore_token_id = self.tokenizer.encode("_")[0]
        underscore_mask = (x == underscore_token_id)

        # Generate random tokens for underscore positions
        random_tokens = torch.randint(
            0, self.tokenizer.vocab_size, x.shape, device=self.device, dtype=torch.long
        )

        # Replace underscores with random tokens
        x = torch.where(underscore_mask, random_tokens, x)

        # Create timestep schedule
        timesteps = create_timestep_schedule(num_steps, 1e-5, schedule, self.device)

        # Denoising loop
        for i in range(num_steps + 1):
            t = timesteps[i] * torch.ones(1, device=self.device)
            curr_sigma_bar = self.noise_schedule(t)[0]

            if i < num_steps:
                t_next = timesteps[i + 1] * torch.ones(1, device=self.device)
                next_sigma_bar = self.noise_schedule(t_next)[0]
                delta_sigma = curr_sigma_bar - next_sigma_bar
            else:
                delta_sigma = curr_sigma_bar

            # Model forward
            log_score = self.model(x, curr_sigma_bar)
            if isinstance(log_score, dict):
                log_score = log_score["logits"]

            log_score = torch.clamp(log_score, min=-30, max=30)
            score = torch.exp(log_score)

            # Compute probabilities
            stag_score = staggered_score(
                score, delta_sigma[:, None], self.tokenizer.vocab_size
            )
            probs = stag_score * transition(
                x, delta_sigma[:, None], self.tokenizer.vocab_size
            )

            # Ensure valid probabilities
            probs = torch.clamp(probs, min=0)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=1e10, neginf=0.0)
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)

            # Sample next tokens
            x = sample_categorical(probs)

            # Decode and yield intermediate result
            intermediate_text = self.tokenizer.decode(x[0].tolist())
            noise_level = curr_sigma_bar.item()

            yield (i, intermediate_text, noise_level)

        # Return final text
        return intermediate_text

    def generate_with_visualization(
        self,
        seq_len: int = 256,
        num_steps: int = 256,
        schedule: str = "quadratic",
        delay: float = 0.02,
    ) -> str:
        """
        Generate a single sample with live terminal visualization.

        Args:
            seq_len: Length of sequence to generate
            num_steps: Number of denoising steps
            schedule: Timestep schedule
            delay: Delay between visualization updates (seconds)

        Returns:
            Generated text string
        """
        import time

        def clear_terminal():
            os.system("cls" if os.name == "nt" else "clear")

        def print_wrapped(text, width=80):
            print(textwrap.fill(text, width=width))

        # Initialize
        x = torch.randint(
            0, self.tokenizer.vocab_size, (1, seq_len), device=self.device
        )

        print("Initial random sequence:")
        print_wrapped(self.tokenizer.decode(x[0].tolist()))
        print("\nStarting denoising...\n")
        time.sleep(2)

        # Create timestep schedule
        from .diffusion.sampling import (
            create_timestep_schedule,
            sample_categorical,
            staggered_score,
            transition,
        )

        timesteps = create_timestep_schedule(num_steps, 1e-5, schedule, self.device)

        # Denoising loop
        for i in range(num_steps + 1):
            t = timesteps[i] * torch.ones(1, device=self.device)
            curr_sigma_bar = self.noise_schedule(t)[0]

            if i < num_steps:
                t_next = timesteps[i + 1] * torch.ones(1, device=self.device)
                next_sigma_bar = self.noise_schedule(t_next)[0]
                delta_sigma = curr_sigma_bar - next_sigma_bar
            else:
                delta_sigma = curr_sigma_bar

            # Model forward
            log_score = self.model(x, curr_sigma_bar)
            if isinstance(log_score, dict):
                log_score = log_score["logits"]

            log_score = torch.clamp(log_score, min=-30, max=30)
            score = torch.exp(log_score)

            # Compute probabilities
            stag_score = staggered_score(
                score, delta_sigma[:, None], self.tokenizer.vocab_size
            )
            probs = stag_score * transition(
                x, delta_sigma[:, None], self.tokenizer.vocab_size
            )

            # Ensure valid probabilities
            probs = torch.clamp(probs, min=0)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=1e10, neginf=0.0)
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)

            # Sample
            x = sample_categorical(probs)

            # Display
            clear_terminal()
            print(f"Denoising step {i}/{num_steps}")
            print(f"Noise level: {curr_sigma_bar.item():.4f}")
            print("=" * 80)
            print_wrapped(self.tokenizer.decode(x[0].tolist()))
            print("=" * 80)

            time.sleep(delay)

        print("\nGeneration complete!")
        return self.tokenizer.decode(x[0].tolist())

    def __call__(
        self,
        num_samples: int = 1,
        seq_len: int = 256,
        num_steps: int = 256,
        **kwargs,
    ) -> List[str]:
        """
        Generate text (callable interface).

        Args:
            num_samples: Number of samples to generate
            seq_len: Length of sequences
            num_steps: Number of denoising steps
            **kwargs: Additional arguments passed to generate()

        Returns:
            List of generated text strings
        """
        return self.generate(
            num_samples=num_samples,
            seq_len=seq_len,
            num_steps=num_steps,
            **kwargs,
        )
