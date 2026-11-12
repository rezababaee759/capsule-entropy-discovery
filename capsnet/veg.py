"""Entropy-gate (VEG) measurements on trained models: per-capsule coupling entropy
and the distribution of the final-iteration gate by capsule role."""

from __future__ import annotations

import numpy as np
import torch

from capsnet.taxonomy import ORDERED_ROLE_NAMES

#: Test-loader batch size of the gate measurements.
GATE_BATCH = 256


def coupling_entropy(c, eps: float = 1e-8):
    """H_i = -sum_j c_ij log(c_ij + eps) over the class axis: [B, O, I] -> [B, I]."""
    return -torch.sum(c * torch.log(c + eps), dim=1)


@torch.no_grad()
def gate_distribution(model, loader, device, cap_role, max_images=2000):
    """Summary of the final-iteration entropy_gate values grouped by role.

    cap_role: capsule index -> role name (capsule_roles). Reads loader batches until
    at least `max_images` images are seen. Roles without capsules are omitted.
    """
    model.eval()
    per_role_vals = {name: [] for name in ORDERED_ROLE_NAMES}
    seen = 0
    for images, _ in loader:
        if seen >= max_images:
            break
        _, _, _, hist = model(images.to(device))
        gate = hist[-1]["entropy_gate"]  # [B, 1, I]
        gate = gate.squeeze(1).cpu().numpy()  # [B, I]
        for cap_idx, role in cap_role.items():
            per_role_vals[role].append(gate[:, cap_idx])
        seen += images.size(0)
    return {
        role: gate_summary(np.concatenate(chunks))
        for role, chunks in per_role_vals.items()
        if chunks
    }


def gate_summary(arr: np.ndarray) -> dict:
    return {
        "n_obs": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
