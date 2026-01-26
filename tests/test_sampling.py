"""Tests for sampling functions."""

import pytest
import torch

from discrete_diffusion.diffusion.sampling import (
    create_timestep_schedule,
    sample_categorical,
    staggered_score,
    transition,
)


class TestCreateTimestepSchedule:
    def test_linear_schedule(self):
        schedule = create_timestep_schedule(10, eps=1e-5, schedule="linear", device="cpu")

        assert schedule.shape == (11,)  # num_steps + 1
        assert schedule[0].item() == pytest.approx(1.0)
        assert schedule[-1].item() == pytest.approx(1e-5)

    def test_quadratic_schedule(self):
        schedule = create_timestep_schedule(10, eps=1e-5, schedule="quadratic", device="cpu")

        assert schedule.shape == (11,)
        assert schedule[0].item() == pytest.approx(1.0)
        # Quadratic should still end near eps
        assert schedule[-1].item() < 0.1

    def test_cosine_schedule(self):
        schedule = create_timestep_schedule(10, eps=1e-5, schedule="cosine", device="cpu")

        assert schedule.shape == (11,)
        assert schedule[0].item() == pytest.approx(1.0)

    def test_invalid_schedule(self):
        with pytest.raises(ValueError):
            create_timestep_schedule(10, schedule="invalid")


class TestTransition:
    def test_output_shape(self):
        x = torch.randint(0, 50, (2, 16))
        delta_sigma = torch.rand(2, 1)

        trans = transition(x, delta_sigma, vocab_size=50)

        assert trans.shape == (2, 16, 50)

    def test_probabilities_sum_to_one(self):
        x = torch.randint(0, 50, (2, 16))
        delta_sigma = torch.rand(2, 1)

        trans = transition(x, delta_sigma, vocab_size=50)

        # Each position should sum to 1
        sums = trans.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_probabilities_non_negative(self):
        x = torch.randint(0, 50, (2, 16))
        delta_sigma = torch.rand(2, 1)

        trans = transition(x, delta_sigma, vocab_size=50)

        assert (trans >= 0).all()


class TestStaggeredScore:
    def test_output_shape(self):
        score = torch.rand(2, 16, 50)
        delta_sigma = torch.rand(2, 1)

        stag = staggered_score(score, delta_sigma, vocab_size=50)

        assert stag.shape == score.shape


class TestSampleCategorical:
    def test_output_shape(self):
        probs = torch.ones(2, 16, 50) / 50  # Uniform

        samples = sample_categorical(probs)

        assert samples.shape == (2, 16)

    def test_samples_in_range(self):
        probs = torch.ones(2, 16, 50) / 50

        samples = sample_categorical(probs)

        assert samples.min() >= 0
        assert samples.max() < 50

    def test_respects_distribution(self):
        # Make token 0 have much higher probability
        probs = torch.ones(1000, 1, 50) * 0.001
        probs[:, :, 0] = 0.999
        probs = probs / probs.sum(dim=-1, keepdim=True)

        samples = sample_categorical(probs)

        # Most samples should be token 0 (with higher sample count for stability)
        assert (samples == 0).float().mean() > 0.9
