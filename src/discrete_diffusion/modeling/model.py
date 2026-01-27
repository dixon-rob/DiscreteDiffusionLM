"""Main Discrete Diffusion Transformer model."""

import math
from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from .configuration import DiscreteDiffusionConfig
from .transformer import DDiTBlock, DDitFinalLayer, TimestepEmbedder


class DiscreteDiffusionTransformer(PreTrainedModel):
    """
    Discrete Diffusion Transformer for character-level text generation.

    Compatible with HuggingFace Transformers library.
    """

    config_class = DiscreteDiffusionConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: DiscreteDiffusionConfig):
        super().__init__(config)
        self.config = config
        self.gradient_checkpointing = False

        # Timestep/noise embedder
        self.sigma_map = TimestepEmbedder(config.cond_dim)

        # Main transformer components
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),  # Token embeddings
                wpe=nn.Embedding(config.block_size, config.n_embd),  # Position embeddings
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([DDiTBlock(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        # Final output layer
        self.lm_head = DDitFinalLayer(config)

        # Initialize weights
        self.apply(self._init_weights)

        # Special scaled init for residual projections (GPT-2 style)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.

        Args:
            non_embedding: If True, subtract position embedding params

        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights following GPT-2 initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _set_gradient_checkpointing(self, enable: bool = True, gradient_checkpointing_func=None) -> None:
        """Enable/disable gradient checkpointing for the model."""
        self.gradient_checkpointing = enable
        if gradient_checkpointing_func is not None:
            self._gradient_checkpointing_func = gradient_checkpointing_func

    def forward(
        self,
        input_ids: torch.Tensor,
        sigma: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            input_ids: Token indices [B, T]
            sigma: Noise levels [B] or scalar
            return_dict: Whether to return dict or tuple

        Returns:
            Logits [B, T, vocab_size] representing log probability ratios
        """
        # Ensure sigma is 1D
        if sigma.dim() == 0:
            sigma = sigma.unsqueeze(0).expand(input_ids.shape[0])
        sigma = sigma.reshape(-1)

        device = input_ids.device
        B, T = input_ids.size()

        # Embed timestep/noise
        c = F.silu(self.sigma_map(sigma))  # [B, cond_dim]

        # Check sequence length
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        )

        # Position indices
        pos = torch.arange(0, T, dtype=torch.long, device=device)  # [T]

        # Token and position embeddings
        tok_emb = self.transformer.wte(input_ids)  # [B, T, n_embd]
        pos_emb = self.transformer.wpe(pos)  # [T, n_embd]
        x = self.transformer.drop(tok_emb + pos_emb)

        # Apply transformer blocks
        if self.gradient_checkpointing and self.training:
            # Use gradient checkpointing to save memory
            for block in self.transformer.h:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, c, use_reentrant=False
                )
        else:
            for block in self.transformer.h:
                x = block(x, c)

        # Final layer norm
        x = self.transformer.ln_f(x)

        # Project to vocabulary
        logits = self.lm_head(x, c)  # [B, T, vocab_size]

        # Zero out logits for current tokens (can't transition to self in 1-Hamming)
        # This implements the "different token" constraint
        logits = torch.scatter(
            logits, dim=-1, index=input_ids[..., None], src=torch.zeros_like(logits[..., :1])
        )

        if return_dict:
            return {"logits": logits}
        return logits
