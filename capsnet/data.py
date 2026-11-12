"""Test-set image splits, the input transform each dataset's checkpoints expect,
seeding, the test sets and their loaders, and per-class index lists."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from capsnet import PROJECT_ROOT

DATA_ROOT = str(PROJECT_ROOT / "data")

# ---------------------------------------------------------------------------
# Image splits
#
# Every analysis draws its images from a fixed, non-overlapping partition of the
# test set, so the images used to discover a capsule's role are never the images
# used to evaluate it. Positions are within each class's own index list.
# ---------------------------------------------------------------------------

# purpose -> (start, stop) per class, half-open.
SPLITS: dict[str, dict[str, tuple[int, int]]] = {
    "mnist": {
        "surrogate_train": (0, 10),  # train/train_surrogate.py
        "surrogate_heldout": (10, 60),  # experiments/held_out_validation.py
        "taxonomy_calibration": (60, 110),  # experiments/hub_taxonomy.py
        "evaluation": (110, 310),  # experiments/symmetric_raw_records.py
        # experiments/top_capsule_ablation_effects.py. Separate from
        # taxonomy_calibration because that analysis selects each digit's top capsule
        # on the taxonomy_calibration matrix; measuring on the same images would make
        # its rank test's exchangeability null false by construction.
        "top_capsule_evaluation": (310, 360),
    },
    # CIFAR-10 has no surrogate held-out split, so calibration begins at 10, not 60.
    "cifar10": {
        "surrogate_train": (0, 10),  # train/train_surrogate.py
        "taxonomy_calibration": (10, 60),  # experiments/hub_taxonomy.py
        "evaluation": (60, 260),  # experiments/symmetric_raw_records.py
    },
}


class SplitViolation(AssertionError):
    """Raised when a requested slice would leave its declared purpose's range."""


class SplitOverlap(AssertionError):
    """Raised when two declared purposes for one dataset share a position."""


def assert_disjoint(dataset: str) -> None:
    """Every declared purpose for `dataset` must occupy a disjoint position range."""
    table = SPLITS[dataset]
    ordered = sorted(table.items(), key=lambda kv: kv[1])
    for (p_a, (a0, a1)), (p_b, (b0, b1)) in zip(ordered, ordered[1:]):
        if b0 < a1:
            raise SplitOverlap(
                f"{dataset}: purposes {p_a} [{a0}:{a1}) and {p_b} [{b0}:{b1}) overlap. "
                f"Disjointness is what lets an analysis measure on images that did not "
                f"take part in choosing what it measures."
            )


for _ds in SPLITS:
    assert_disjoint(_ds)


def _dataset_key(dataset: str) -> str:
    return dataset.lower().replace("-", "").replace("_", "")


def available(dataset: str) -> dict[str, tuple[int, int]]:
    """Return the split table for `dataset`, with a useful error if unknown."""
    key = _dataset_key(dataset)
    if key not in SPLITS:
        raise KeyError(
            f"No split contract for dataset {dataset!r}. "
            f"Known: {sorted(SPLITS)}. Add it here rather than hardcoding offsets."
        )
    return SPLITS[key]


def slice_for(dataset: str, purpose: str, n_per_class: int) -> tuple[int, int]:
    """Return the (start, stop) slice for `purpose`, asserting it fits.

    Raises SplitViolation if `n_per_class` would run past the end of the purpose's
    range and into a neighbouring split.
    """
    table = available(dataset)
    if purpose not in table:
        raise KeyError(
            f"{dataset}: unknown purpose {purpose!r}. Known: {sorted(table)}."
        )
    start, stop = table[purpose]
    capacity = stop - start
    if n_per_class > capacity:
        raise SplitViolation(
            f"{dataset}/{purpose}: requested {n_per_class} images per class but the "
            f"split only holds {capacity} (positions [{start}:{stop})). Taking more "
            f"would run into the next split and break the disjointness between splits. "
            f"Either lower the sample count or widen the contract in capsnet/data.py."
        )
    return start, start + n_per_class


def describe_split(dataset: str) -> str:
    """One-line rendering of the split table, for logging into artifacts."""
    table = available(dataset)
    parts = [
        f"{p} [{a}:{b})" for p, (a, b) in sorted(table.items(), key=lambda kv: kv[1])
    ]
    return f"{dataset}: " + " | ".join(parts)


# ---------------------------------------------------------------------------
# Input transforms
# ---------------------------------------------------------------------------

# Channel statistics the CIFAR-10 checkpoints were trained with.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

#: dataset -> short description of the transform, for logging into artifacts.
TRANSFORM_CONTRACT = {
    "mnist": "ToTensor() only; images in [0,1]. NO Normalize.",
    "fashionmnist": "ToTensor() only; images in [0,1]. NO Normalize.",
    "cifar10": f"ToTensor() + Normalize({CIFAR10_MEAN}, {CIFAR10_STD}).",
}


def _transform_key(dataset: str) -> str:
    k = _dataset_key(dataset)
    if k not in TRANSFORM_CONTRACT:
        raise KeyError(
            f"No input-transform contract for {dataset!r}. "
            f"Known: {sorted(TRANSFORM_CONTRACT)}."
        )
    return k


def cifar10_normalize() -> transforms.Normalize:
    return transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)


def test_transform(dataset: str) -> transforms.Compose:
    """The transform this dataset's checkpoints were trained under.

    MNIST and FashionMNIST models take plain ToTensor() inputs in [0, 1]: the
    reconstruction decoder ends in a sigmoid, so it can only target [0, 1].
    """
    if _transform_key(dataset) == "cifar10":
        return transforms.Compose([transforms.ToTensor(), cifar10_normalize()])
    return transforms.Compose([transforms.ToTensor()])


def describe_transform(dataset: str) -> str:
    """One-line rendering of the transform, for recording in output artifacts."""
    return f"{dataset}: {TRANSFORM_CONTRACT[_transform_key(dataset)]}"


# ---------------------------------------------------------------------------
# Seeding and loaders
# ---------------------------------------------------------------------------


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed random, NumPy, torch and CUDA.

    With `deterministic`, also pin cuDNN to deterministic algorithms and disable its
    autotuner; other CUDA kernels are not constrained.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


_TEST_SETS = {
    "mnist": datasets.MNIST,
    "fashionmnist": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
}


def test_set(dataset: str):
    """The full test set of `dataset`, in order, under its checkpoints' transform."""
    return _TEST_SETS[_transform_key(dataset)](
        DATA_ROOT, train=False, download=True, transform=test_transform(dataset)
    )


def test_loader(dataset: str, batch_size: int) -> DataLoader:
    """test_set(dataset) in order, in batches of `batch_size`."""
    return DataLoader(
        test_set(dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )


def class_indices(ds) -> dict[int, list[int]]:
    """Class -> dataset positions of its images, ascending; read from `ds.targets`,
    so no image is loaded."""
    targets = ds.targets
    if torch.is_tensor(targets):
        targets = targets.tolist()
    out = {c: [] for c in range(len(ds.classes))}
    for i, label in enumerate(targets):
        out[int(label)].append(i)
    return out


def split_indices(ds, dataset: str, purpose: str, n_per_class: int) -> dict:
    """Class -> np.ndarray of the dataset positions of its first `n_per_class` images
    inside the `purpose` split (see slice_for)."""
    start, stop = slice_for(dataset, purpose, n_per_class)
    per_class = class_indices(ds)
    for c, avail in per_class.items():
        if len(avail) < stop:
            raise SplitViolation(
                f"{dataset}/{purpose}: class {c} has only {len(avail)} test images; "
                f"the slice needs positions [{start}:{stop})."
            )
    return {c: np.array(avail[start:stop]) for c, avail in per_class.items()}
