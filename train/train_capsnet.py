"""Train a capsule network on MNIST, FashionMNIST or CIFAR-10.

MNIST keeps the epoch with the best test accuracy, FashionMNIST and CIFAR-10 the
final epoch. A checkpoint is written only with --save_path.

Usage:
    python train/train_capsnet.py --dataset mnist --routing_strategy dynamic \\
        --seed 42 --save_path models/tier3/capsule_mnist_best_dynamic.pt
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from capsnet.data import (
    DATA_ROOT,
    cifar10_normalize,
    set_seed,
    test_set,
    test_transform,
)
from capsnet.io import get_device, guard_overwrite
from capsnet.model import build_capsule_network
from capsnet.training import evaluate, train_epoch

BATCH_SIZE = 128
DEVICE = get_device()


def mnist(model):
    optimizer = Adam(model.parameters(), lr=3e-3)
    return dict(
        optimizer=optimizer,
        scheduler=None,
        recon_weight=0.0005,
        train_transform=test_transform("mnist"),
    )


def fashionmnist(model):
    optimizer = Adam(model.parameters(), lr=0.001)
    return dict(
        optimizer=optimizer,
        scheduler=None,
        recon_weight=0.0005,
        train_transform=test_transform("fashionmnist"),
    )


def cifar10(model):
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=200, verbose=True
    )
    # The CapsNet weight 0.0005 applies to a summed squared error; reconstruction_loss
    # is a per-pixel mean, so the weight is scaled by the pixel count.
    recon_weight = 0.0005 * 3072  # 3 x 32 x 32 = 3072
    train_transform = transforms.Compose(
        [
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            cifar10_normalize(),
        ]
    )
    return dict(
        optimizer=optimizer,
        scheduler=scheduler,
        recon_weight=recon_weight,
        train_transform=train_transform,
    )


# setup: optimizer, LR scheduler, reconstruction weight and training transform.
# keep: which epoch's weights --save_path receives.
DATASETS = {
    "mnist": dict(train_set=datasets.MNIST, setup=mnist, epochs=100, keep="best"),
    "fashionmnist": dict(
        train_set=datasets.FashionMNIST, setup=fashionmnist, epochs=20, keep="final"
    ),
    "cifar10": dict(
        train_set=datasets.CIFAR10, setup=cifar10, epochs=300, keep="final"
    ),
}


def loaders(dataset: str, train_transform):
    train_ds = DATASETS[dataset]["train_set"](
        root=DATA_ROOT, train=True, download=True, transform=train_transform
    )
    common = dict(batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True, **common),
        DataLoader(test_set(dataset), shuffle=False, **common),
    )


def train(dataset: str, strategy: str, epochs: int, save_path: str | None) -> float:
    """Train one network and return its best test accuracy in [0, 1]."""
    model = build_capsule_network(dataset, strategy).to(DEVICE)
    s = DATASETS[dataset]["setup"](model)
    optimizer, scheduler = s["optimizer"], s["scheduler"]
    recon_weight = s["recon_weight"]
    train_loader, test_loader = loaders(dataset, s["train_transform"])

    staging = best_path = None
    if save_path and DATASETS[dataset]["keep"] == "best":
        staging = tempfile.TemporaryDirectory(
            dir=Path(save_path).resolve().parent, prefix=".train_capsnet_"
        )
        # torch.save records the file's basename inside the archive, so the staged
        # file keeps a fixed name and the checkpoint bytes do not depend on --save_path.
        best_path = Path(staging.name) / f"capsule_{dataset}_best_{strategy}.pt"

    best_acc = 0.0
    try:
        print(f"Training {epochs} epochs with {strategy} routing...")
        for epoch in range(1, epochs + 1):
            m = train_epoch(model, train_loader, optimizer, DEVICE, recon_weight)
            test = evaluate(model, test_loader, DEVICE, recon_weight)
            test_acc = test["accuracy"] / 100.0
            if test_acc > best_acc:
                best_acc = test_acc
                if best_path is not None:
                    torch.save(model.state_dict(), best_path)
            if scheduler is not None:
                scheduler.step(best_acc)
            print(
                f"Epoch {epoch}/{epochs}: "
                f"Train Loss={m['loss']:.4f}, Train Acc={m['accuracy'] / 100.0:.4f}, "
                f"Test Acc={test_acc:.4f} (Best: {best_acc:.4f})"
            )

        if best_path is not None:
            os.replace(best_path, save_path)
        elif save_path:
            torch.save(model.state_dict(), save_path)
        if save_path:
            print(f"Model saved to: {save_path}")
    finally:
        if staging is not None:
            staging.cleanup()
    return best_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CapsuleNetwork")
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument(
        "--routing_strategy", required=True, choices=["dynamic", "entropy"]
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="default: 100 / 20 / 300 by dataset"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed for NumPy, PyTorch and CUDA; omit for a nondeterministic run",
    )
    parser.add_argument(
        "--save_path",
        default=None,
        help="checkpoint path; nothing is written without it",
    )
    parser.add_argument(
        "--force", action="store_true", help="allow overwriting --save_path"
    )
    args = parser.parse_args()
    epochs = DATASETS[args.dataset]["epochs"] if args.epochs is None else args.epochs

    if args.seed is not None:
        set_seed(args.seed)
        print(f"Random seed: {args.seed} (NumPy, PyTorch, CUDA)")
    if args.save_path:
        guard_overwrite(args.save_path, args.force)

    print(f"{args.dataset}, {args.routing_strategy} routing, device {DEVICE}")
    best_acc = train(args.dataset, args.routing_strategy, epochs, args.save_path)
    print(f"Best test accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
