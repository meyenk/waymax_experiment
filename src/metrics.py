"""Simple trajectory metrics. pred/target are (batch, future_len, 2) absolute
positions in the ego frame at t -- same space, so a direct distance is valid."""

import torch


def compute_ade_fde(pred, target):
    """Average and final displacement error, in the same units as pred/target
    (meters, since positions are ego-frame). Returns (ade, fde) as floats."""
    dist = torch.norm(pred - target, dim=-1)  # (batch, future_len)
    ade = dist.mean().item()
    fde = dist[:, -1].mean().item()
    return ade, fde
