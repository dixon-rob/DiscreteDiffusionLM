"""Masking utilities for masked infilling."""

import random

import torch


def generate_continuous_mask(
    seq_len: int,
    min_span_len: int = 3,
    max_span_len: int = 50,
    num_spans: int = None,
    mask_fraction: float = 0.5,
) -> torch.Tensor:
    """
    Generate a binary mask with continuous spans.

    Args:
        seq_len: Total sequence length
        min_span_len: Minimum span length (inclusive)
        max_span_len: Maximum span length (inclusive)
        num_spans: Number of masked spans (if None, determined by mask_fraction)
        mask_fraction: Target fraction of sequence to mask (used if num_spans is None)

    Returns:
        Boolean tensor of shape (seq_len,) where True = masked (to be infilled)
    """
    mask = torch.zeros(seq_len, dtype=torch.bool)

    if num_spans is None:
        # Calculate number of spans needed to achieve target mask fraction
        target_masked = int(seq_len * mask_fraction)
        avg_span_len = (min_span_len + max_span_len) / 2
        num_spans = max(1, int(target_masked / avg_span_len))

    masked_positions = set()
    attempts = 0
    max_attempts = 100

    for _ in range(num_spans):
        while attempts < max_attempts:
            # Random span length
            span_len = random.randint(min_span_len, max_span_len)

            # Random start position (ensure span fits)
            max_start = seq_len - span_len
            if max_start < 0:
                break

            start = random.randint(0, max_start)
            end = start + span_len

            # Check for overlap with existing masked spans
            if not any(pos in masked_positions for pos in range(start, end)):
                # Add this span
                for pos in range(start, end):
                    masked_positions.add(pos)
                    mask[pos] = True
                break

            attempts += 1

    return mask


def generate_batch_masks(
    batch_size: int,
    seq_len: int,
    min_span_len: int = 3,
    max_span_len: int = 50,
    mask_fraction: float = 0.5,
) -> torch.Tensor:
    """
    Generate masks for a batch.

    Returns:
        Boolean tensor of shape (batch_size, seq_len)
    """
    masks = []
    for _ in range(batch_size):
        mask = generate_continuous_mask(
            seq_len=seq_len,
            min_span_len=min_span_len,
            max_span_len=max_span_len,
            mask_fraction=mask_fraction,
        )
        masks.append(mask)

    return torch.stack(masks)
