"""Tests for noise schedule and perturbation."""

import pytest
import torch

from discrete_diffusion import GeometricNoise, perturb_batch


class TestGeometricNoise:
    def test_creation(self):
        noise = GeometricNoise(sigma_min=1e-4, sigma_max=20.0)
        assert noise.sigma_min == pytest.approx(1e-4)
        assert noise.sigma_max == pytest.approx(20.0)

    def test_total_noise_bounds(self):
        noise = GeometricNoise(sigma_min=1e-4, sigma_max=20.0)

        # At t=0, should be close to sigma_min
        t0 = torch.tensor(0.0)
        assert noise.total_noise(t0).item() == pytest.approx(1e-4, rel=1e-3)

        # At t=1, should be close to sigma_max
        t1 = torch.tensor(1.0)
        assert noise.total_noise(t1).item() == pytest.approx(20.0, rel=1e-3)

    def test_monotonic_increase(self):
        noise = GeometricNoise()

        timesteps = torch.linspace(0, 1, 100)
        noise_values = [noise.total_noise(t).item() for t in timesteps]

        # Should be monotonically increasing
        for i in range(1, len(noise_values)):
            assert noise_values[i] >= noise_values[i - 1]

    def test_call_returns_tuple(self):
        noise = GeometricNoise()
        t = torch.tensor(0.5)

        result = noise(t)
        assert isinstance(result, tuple)
        assert len(result) == 2

        total, rate = result
        assert total.shape == t.shape
        assert rate.shape == t.shape


class TestPerturbBatch:
    def test_output_shape(self):
        batch = torch.randint(0, 50, (4, 32))
        sigma_bar = torch.tensor(1.0)

        perturbed = perturb_batch(batch, sigma_bar, vocab_size=50)

        assert perturbed.shape == batch.shape

    def test_no_perturbation_at_zero_noise(self):
        batch = torch.randint(0, 50, (4, 32))
        sigma_bar = torch.tensor(0.0)

        perturbed = perturb_batch(batch, sigma_bar, vocab_size=50)

        # With zero noise, should be unchanged
        assert torch.equal(perturbed, batch)

    def test_high_noise_changes_tokens(self):
        batch = torch.randint(0, 50, (4, 32))
        sigma_bar = torch.tensor(100.0)  # Very high noise

        perturbed = perturb_batch(batch, sigma_bar, vocab_size=50)

        # With very high noise, most tokens should change
        changed = (perturbed != batch).float().mean()
        assert changed > 0.5  # At least half should change

    def test_per_sample_noise(self):
        batch = torch.randint(0, 50, (4, 32))
        sigma_bar = torch.tensor([0.0, 1.0, 10.0, 100.0])

        perturbed = perturb_batch(batch, sigma_bar, vocab_size=50)

        # First sample (zero noise) should be unchanged
        assert torch.equal(perturbed[0], batch[0])

        # Last sample (high noise) should have many changes
        changed = (perturbed[3] != batch[3]).float().mean()
        assert changed > 0.3

    def test_tokens_stay_in_vocab(self):
        vocab_size = 50
        batch = torch.randint(0, vocab_size, (4, 32))
        sigma_bar = torch.tensor(5.0)

        perturbed = perturb_batch(batch, sigma_bar, vocab_size=vocab_size)

        assert perturbed.min() >= 0
        assert perturbed.max() < vocab_size
