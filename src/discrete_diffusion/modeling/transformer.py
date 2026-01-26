"""Transformer building blocks for Discrete Diffusion Transformer."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration import DiscreteDiffusionConfig


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply affine transformation: x * (1 + scale) + shift"""
    return x * (1 + scale) + shift


def bias_add_scale(
    x: torch.Tensor,
    bias: Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: Optional[torch.Tensor],
) -> torch.Tensor:
    """Scale x (optionally with bias), then add residual."""
    if bias is not None:
        out = scale * (x + bias)
    else:
        out = scale * x

    if residual is not None:
        out = residual + out
    return out


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, config: DiscreteDiffusionConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class SelfAttention(nn.Module):
    """
    Bidirectional self-attention (no causal masking).
    Key difference from autoregressive models.
    """

    def __init__(self, config: DiscreteDiffusionConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Key, query, value projections for all heads in batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Flash attention support (PyTorch >= 2.0)
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, sequence length, embedding dim

        # Calculate Q, K, V for all heads
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape for multi-head attention: (B, T, n_head, head_size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Self-attention
        if self.flash:
            # Use Flash Attention (efficient)
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=False,  # BIDIRECTIONAL - key difference!
            )
        else:
            # Manual attention computation
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Re-assemble all head outputs
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps/noise levels into vector representations.
    Uses sinusoidal embeddings similar to Transformer positional encoding.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: 1-D Tensor of N indices (can be fractional)
            dim: Dimension of output
            max_period: Controls minimum frequency

        Returns:
            (N, dim) Tensor of positional embeddings
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Tensor of shape [B] containing timestep/noise values

        Returns:
            Tensor of shape [B, hidden_size]
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DDiTBlock(nn.Module):
    """
    Discrete Diffusion Transformer Block.
    Combines self-attention and MLP with adaptive layer normalization (adaLN)
    conditioned on timestep/noise level.
    """

    def __init__(self, config: DiscreteDiffusionConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

        # Adaptive LayerNorm modulation
        # Produces 6 values: shift/scale/gate for both attention and MLP
        self.adaLN_modulation = nn.Linear(config.cond_dim, 6 * config.n_embd)
        # Initialize to zero (identity transformation initially)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, T, n_embd]
            c: Conditioning tensor [B, cond_dim] (timestep embedding)

        Returns:
            Output tensor [B, T, n_embd]
        """
        # Extract modulation parameters for attention and MLP
        # Shape: [B, 1, 6*n_embd] -> 6 tensors of [B, 1, n_embd]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)[
            :, None
        ].chunk(6, dim=2)

        # Save residual
        x_skip = x

        # Attention block with adaptive LayerNorm
        x = bias_add_scale(
            self.attn(modulate(self.ln_1(x), shift_msa, scale_msa)), None, gate_msa, x_skip
        )

        # MLP block with adaptive LayerNorm
        x = bias_add_scale(self.mlp(modulate(self.ln_2(x), shift_mlp, scale_mlp)), None, gate_mlp, x)

        return x


class DDitFinalLayer(nn.Module):
    """
    Final layer that maps hidden states to vocabulary logits.
    Also uses adaptive LayerNorm conditioned on timestep.
    """

    def __init__(self, config: DiscreteDiffusionConfig):
        super().__init__()
        self.norm_final = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.linear = nn.Linear(config.n_embd, config.vocab_size)

        # Initialize to zero for stability
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        # Adaptive LayerNorm modulation (shift and scale only)
        self.adaLN_modulation = nn.Linear(config.cond_dim, 2 * config.n_embd)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Hidden states [B, T, n_embd]
            c: Conditioning tensor [B, cond_dim]

        Returns:
            Logits [B, T, vocab_size]
        """
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
