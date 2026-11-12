"""Shared pieces of the symmetric Hub/Specialist evaluation: per-dataset record
settings, checkpoint and taxonomy paths per model slot, image pools, batched
coupling extraction, per-capsule correctness, and the model design frame."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from capsnet import PROJECT_ROOT
from capsnet.data import split_indices

SAMPLES_PER_CLASS = 200
N_RANDOM = 50

# Slot 42 is the primary checkpoint, which records no RNG seed; the other slots are
# the seeds of the replications. `batch_size` is the forward batch of
# batched_couplings and changes the couplings at the floating-point level.
RECORDS = {
    "mnist": {
        "seeds": [42, 123, 789],
        "output_dir": "results/data/mnist/pooled_42_123_789",
        "batch_size": 200,
    },
    "cifar10": {
        "seeds": [42, 123, 456],
        "output_dir": "results/data/cifar10",
        "batch_size": 100,
    },
}


def model_path(dataset: str, slot: int) -> Path:
    if slot == 42:
        return PROJECT_ROOT / f"models/capsule_{dataset}_best_dynamic.pt"
    return PROJECT_ROOT / f"models/capsule_{dataset}_dynamic_seed{slot}.pt"


def taxonomy_paths(dataset: str, slot: int) -> tuple[Path, Path]:
    """(hub_taxonomy.json, importance_matrix.npy) of the model in `slot`."""
    if slot == 42:
        base = PROJECT_ROOT / f"results/data/{dataset}"
    else:
        base = PROJECT_ROOT / f"results/data/{dataset}/seed{slot}"
    return base / "hub_taxonomy.json", base / "importance_matrix.npy"


def class_pools(ds, dataset: str, n_per_class: int) -> dict:
    """Class -> np.ndarray of `n_per_class` positions from the evaluation split; the
    same images for every model."""
    return split_indices(ds, dataset, "evaluation", n_per_class)


@torch.no_grad()
def batched_couplings(model, ds, indices, device, batch_size: int) -> np.ndarray:
    """Final coupling coefficients [n_images, n_classes, n_primary_capsules] of the
    images at `indices`, forwarded `batch_size` at a time."""
    outs = []
    imgs = torch.stack([ds[int(i)][0] for i in indices]).to(device)
    for start in range(0, imgs.size(0), batch_size):
        chunk = imgs[start : start + batch_size]
        _, c_ij, _, _ = model(chunk)
        outs.append(c_ij.cpu().numpy())
    return np.concatenate(outs, axis=0)


def per_capsule_correctness(coupling_stack, importance_matrix, caps):
    """Per-capsule within and cross routing correctness, and each capsule's home class.

    coupling_stack: [shown class (10), image, class capsule (10), primary capsule].
    A capsule routes an image correctly when its largest coupling is to the shown
    class; home is the argmax of its importance column. Returns (within, cross, home).
    """
    cap_arr = np.asarray(caps)
    home = np.array([int(np.argmax(importance_matrix[:, c])) for c in cap_arr])
    sub = coupling_stack[:, :, :, cap_arr]  # [10, n_img, 10, C]
    top = sub.argmax(axis=2)  # [10, n_img, C]
    test_digit = np.arange(10)[:, None, None]
    correct = top == test_digit  # [10, n_img, C]

    C = len(cap_arr)
    within = np.empty(C)
    cross = np.empty(C)
    for j in range(C):
        h = home[j]
        within[j] = correct[h, :, j].mean()
        mask = np.ones(10, dtype=bool)
        mask[h] = False
        cross[j] = correct[mask, :, j].mean()
    return within, cross, home


def design_frame(df: pd.DataFrame) -> pd.DataFrame:
    """A copy of the raw records with the model factors as categoricals; 'random' and
    'cross' are the reference levels."""
    model_df = df.copy()
    model_df["role"] = pd.Categorical(
        model_df["role"], categories=["random", "specialist", "hub"]
    )
    model_df["condition"] = pd.Categorical(
        model_df["condition"], categories=["cross", "within"]
    )
    model_df["test_digit"] = model_df["test_digit"].astype("category")
    model_df["seed"] = model_df["seed"].astype("category")
    return model_df


def cluster_codes(model_df: pd.DataFrame):
    """(capsule, image) cluster codes per observation: a capsule is (seed,
    capsule_idx), an image is (seed, test_digit, image_idx)."""
    capsule_uid = (
        model_df["seed"].astype(str) + "_" + model_df["capsule_idx"].astype(str)
    )
    image_uid = (
        model_df["seed"].astype(str)
        + "_"
        + model_df["test_digit"].astype(str)
        + "_"
        + model_df["image_idx"].astype(str)
    )
    return pd.Categorical(capsule_uid).codes, pd.Categorical(image_uid).codes
