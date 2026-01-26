"""Tests for the discrete diffusion model."""

import pytest
import torch

from discrete_diffusion import (
    CharacterLevelTokenizer,
    DiscreteDiffusionConfig,
    DiscreteDiffusionTransformer,
)

# Vocab size is fixed by the tokenizer
VOCAB_SIZE = CharacterLevelTokenizer().vocab_size


@pytest.fixture
def small_config():
    """Small model config for testing."""
    return DiscreteDiffusionConfig(
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=64,
        cond_dim=32,
        dropout=0.0,
        bias=True,
    )


@pytest.fixture
def model(small_config):
    """Create a small model for testing."""
    return DiscreteDiffusionTransformer(small_config)


class TestDiscreteDiffusionConfig:
    def test_default_config(self):
        config = DiscreteDiffusionConfig()
        assert config.vocab_size == VOCAB_SIZE
        assert config.block_size == 256
        assert config.n_layer == 6

    def test_custom_config(self, small_config):
        assert small_config.vocab_size == VOCAB_SIZE  # Always uses tokenizer vocab_size
        assert small_config.block_size == 32
        assert small_config.n_layer == 2


class TestDiscreteDiffusionTransformer:
    def test_model_creation(self, model, small_config):
        assert model is not None
        assert model.config == small_config

    def test_forward_pass(self, model, small_config):
        batch_size = 2
        seq_len = 16

        input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
        sigma = torch.rand(batch_size)

        output = model(input_ids, sigma)

        assert "logits" in output
        assert output["logits"].shape == (batch_size, seq_len, VOCAB_SIZE)

    def test_forward_scalar_sigma(self, model, small_config):
        batch_size = 2
        seq_len = 16

        input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
        sigma = torch.tensor(0.5)

        output = model(input_ids, sigma)

        assert output["logits"].shape == (batch_size, seq_len, VOCAB_SIZE)

    def test_self_transition_zeroed(self, model, small_config):
        """Verify that logits for current tokens are zeroed (no self-transition)."""
        batch_size = 2
        seq_len = 16

        input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
        sigma = torch.rand(batch_size)

        output = model(input_ids, sigma)
        logits = output["logits"]

        # Check that logits at current token positions are zero
        for b in range(batch_size):
            for t in range(seq_len):
                token_id = input_ids[b, t].item()
                assert logits[b, t, token_id].item() == 0.0

    def test_get_num_params(self, model):
        num_params = model.get_num_params()
        assert num_params > 0

        # With embeddings should be larger
        num_params_with_emb = model.get_num_params(non_embedding=False)
        assert num_params_with_emb >= num_params
