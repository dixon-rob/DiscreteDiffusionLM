"""Forward diffusion perturbation for discrete tokens."""

import torch


def perturb_batch(
    batch: torch.Tensor,
    sigma_bar: torch.Tensor,
    vocab_size: int,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Diffuse each token independently (forward diffusion process).

    With probability e^{-sigma_bar} + (1 - e^{-sigma_bar})/N, a token stays the same.
    Otherwise, it jumps uniformly to one of the other N-1 tokens.

    Args:
        batch: LongTensor of shape [B, L], each entry in [0, vocab_size-1]
        sigma_bar: Tensor of shape [] (scalar) or [B] (per-sample noise levels)
                   The accumulated noise sigma_bar(t)
        vocab_size: int, size of vocabulary (N)
        mask: Optional boolean mask [B, L] where True = preserve (don't perturb)

    Returns:
        batch_pert: perturbed batch, LongTensor of shape [B, L]
    """
    B, L = batch.shape
    device = batch.device

    # Ensure sigma_bar is broadcastable to [B, L]
    # If scalar, expand to [1, 1]; if [B], expand to [B, 1]
    if sigma_bar.dim() == 0:  # scalar
        sigma_bar = sigma_bar.unsqueeze(0).unsqueeze(1)  # [1, 1]
    elif sigma_bar.dim() == 1:  # [B]
        sigma_bar = sigma_bar.unsqueeze(1)  # [B, 1]

    # 1) Compute move probability: (1 - e^{-sigma_bar}) * (1 - 1/N)
    stay_base = torch.exp(-sigma_bar)  # [B, 1] or [1, 1]
    move_prob = (1 - stay_base) * (1 - 1.0 / vocab_size)  # [B, 1] or [1, 1]

    # 2) Bernoulli: should this token move?
    # move_prob broadcasts to [B, L]
    move_mask = torch.rand(B, L, device=device) < move_prob

    # 3) For tokens that move, sample a *different* id uniformly from the other N-1 ids.
    # Sample r in [0, N-2], then map to [0..N-1]\{orig} by skipping the original.
    r = torch.randint(low=0, high=vocab_size - 1, size=(B, L), device=device)

    # Shift up by 1 wherever r >= original id
    # This covers {0, ..., k-1, k+1, ..., N-1}
    new_ids = r + (r >= batch).long()  # Convert boolean to long for addition

    # 4) Apply moves; else keep original
    batch_pert = torch.where(move_mask, new_ids, batch)

    # 5) Apply mask: preserve tokens where mask=True
    if mask is not None:
        batch_pert = torch.where(mask, batch, batch_pert)

    return batch_pert
