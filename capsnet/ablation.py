"""Exhaustive single-capsule ablation: the drop in one class capsule's length when
each primary capsule is zeroed in turn."""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm


@torch.no_grad()
def single_capsule_ablation_drops(
    model, primary_out, cls: int, baseline: float, batch_size: int = 128
) -> np.ndarray:
    """float64 [n_primary_capsules]: `baseline` minus the length of class capsule
    `cls` with that primary capsule zeroed.

    primary_out: [1, n_primary_capsules, dim], the primary capsules of one image;
    only the routing stage (model.digits) is re-run, `batch_size` ablations per pass.
    `baseline` is a Python float, so each drop is computed in float32 and stored in
    float64.
    """
    num_capsules = primary_out.shape[1]
    drops = np.zeros(num_capsules)
    for start in range(0, num_capsules, batch_size):
        end = min(start + batch_size, num_capsules)
        span = end - start
        batch = primary_out.repeat(span, 1, 1)
        for j in range(span):
            batch[j, start + j, :] = 0.0
        ablated, _, _, _ = model.digits(batch)
        drops[start:end] = baseline - torch.norm(ablated, dim=-1)[:, cls].cpu().numpy()
    return drops


@torch.no_grad()
def single_capsule_ablation_effects(
    model, primary_out, cls: int, baseline: float, desc: str | None = None
) -> torch.Tensor:
    """float32 [n_primary_capsules] on primary_out's device: `baseline` minus the
    length of class capsule `cls` with that primary capsule zeroed.

    One routing pass per capsule; each difference is taken between Python floats and
    then stored in float32. `desc`, if given, labels a progress bar.
    """
    num_capsules = primary_out.size(1)
    effects = torch.zeros(num_capsules, device=primary_out.device)
    indices = range(num_capsules)
    if desc is not None:
        indices = tqdm(indices, desc=desc, leave=False)
    for cap_idx in indices:
        ablated = primary_out.clone()
        ablated[0, cap_idx, :] = 0.0
        digit_caps, _, _, _ = model.digits(ablated)
        effects[cap_idx] = baseline - torch.norm(digit_caps[0], dim=-1)[cls].item()
    return effects
