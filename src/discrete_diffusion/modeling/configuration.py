"""Configuration class for Discrete Diffusion Transformer."""

from transformers import PretrainedConfig

from ..data.tokenizer import CharacterLevelTokenizer

# Vocab size is fixed and determined by the tokenizer
VOCAB_SIZE = CharacterLevelTokenizer().vocab_size


class DiscreteDiffusionConfig(PretrainedConfig):
    """
    Configuration class for Discrete Diffusion Transformer model.

    Args:
        block_size: Maximum sequence length
        n_layer: Number of transformer blocks
        n_head: Number of attention heads
        n_embd: Embedding dimension
        cond_dim: Dimension for conditioning (timestep embedding)
        dropout: Dropout probability
        bias: Whether to use bias in linear layers
    """

    model_type = "discrete_diffusion"

    def __init__(
        self,
        block_size: int = 256,
        n_layer: int = 6,
        n_head: int = 6,
        n_embd: int = 384,
        cond_dim: int = 384,
        dropout: float = 0.1,
        bias: bool = True,
        **kwargs,
    ):
        # Remove vocab_size from kwargs if present (for backwards compatibility with saved configs)
        kwargs.pop("vocab_size", None)
        super().__init__(**kwargs)
        self.vocab_size = VOCAB_SIZE
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.cond_dim = cond_dim
        self.dropout = dropout
        self.bias = bias
