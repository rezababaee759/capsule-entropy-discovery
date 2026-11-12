#!/usr/bin/env python3
"""VEG component ablation: trains the dynamic-routing baseline, VEG with one shared
beta (entropy_uniform) and VEG with per-capsule beta for every (dataset, seed), and
writes each arm's best test accuracy.

Usage:
    python train/component_ablation.py --seeds 42 123 456 789 1000 --epochs 300 --force
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
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

# This script's own settings, not train_capsnet.py's: recon weight 5e-4 on every
# dataset, no ColorJitter on CIFAR-10, ReduceLROnPlateau patience 150.
DATASET_CONFIGS = {
    "mnist": {
        "recon_weight": 0.0005,
        "train_set": lambda: datasets.MNIST(
            DATA_ROOT, train=True, download=True, transform=test_transform("mnist")
        ),
    },
    "fashionmnist": {
        "recon_weight": 0.0005,
        "train_set": lambda: datasets.FashionMNIST(
            DATA_ROOT,
            train=True,
            download=True,
            transform=test_transform("fashionmnist"),
        ),
    },
    "cifar10": {
        "recon_weight": 0.0005,
        "train_set": lambda: datasets.CIFAR10(
            DATA_ROOT,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    cifar10_normalize(),
                ]
            ),
        ),
    },
}


def get_loaders(dataset_name, batch_size=128):
    train_loader = DataLoader(
        DATASET_CONFIGS[dataset_name]["train_set"](),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set(dataset_name),
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, test_loader


def run_single(dataset_name, strategy, seed, epochs, device, save_path=None):
    """Train one arm; save its best-accuracy weights to save_path if given."""
    set_seed(seed)
    cfg = DATASET_CONFIGS[dataset_name]

    model = build_capsule_network(dataset_name, strategy).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = None
    if dataset_name == "cifar10":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=150, verbose=False
        )

    train_loader, test_loader = get_loaders(dataset_name)
    best_acc = 0.0
    best_state = None

    print(
        f"  [{dataset_name.upper()} | {strategy} | seed={seed}]  "
        f"Training {epochs} epochs..."
    )

    for epoch in range(1, epochs + 1):
        train_stats = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            cfg["recon_weight"],
            clip_grad_norm=5.0,
        )
        test = evaluate(model, test_loader, device)
        test_acc = test["correct"] / test["total"]

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if scheduler:
            scheduler.step(best_acc)

        if epoch % 50 == 0 or epoch == epochs:
            print(
                f"    Epoch {epoch:3d}/{epochs}: "
                f"Loss={train_stats['loss']:.4f}  Train={train_stats['accuracy']:.2f}%  "
                f"Test={test_acc:.4f}  Best={best_acc:.4f}"
            )

    if save_path and best_state:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(best_state, save_path)
        print(f"    Saved: {save_path}")

    beta_stats = None
    routing = model.digits.routing_module
    if hasattr(routing, "beta"):
        b = routing.beta.detach().cpu().numpy()
        beta_stats = {
            "shape": list(b.shape),
            "mean": float(b.mean()),
            "std": float(b.std()),
            "min": float(b.min()),
            "max": float(b.max()),
        }

    return {"best_acc": float(best_acc), "seed": seed, "beta_stats": beta_stats}


def run_dataset(dataset_name, strategies, seeds, epochs, device, output_dir):
    """Every arm x seed of one dataset: raw runs and per-arm mean/std."""
    print(f"\nDataset: {dataset_name.upper()}")

    results = {s: [] for s in strategies}

    for strategy in strategies:
        for seed in seeds:
            save_path = None
            if output_dir:
                fname = f"capsule_{dataset_name}_{strategy}_seed{seed}.pt"
                save_path = str(Path(output_dir) / fname)
            r = run_single(dataset_name, strategy, seed, epochs, device, save_path)
            results[strategy].append(r)
            print(f"    {strategy} seed={seed}: {r['best_acc']*100:.2f}%")

    summary = {}
    for strategy, runs in results.items():
        accs = [r["best_acc"] * 100 for r in runs]
        std_ddof1 = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
        summary[strategy] = {
            "mean": float(np.mean(accs)),
            "std": std_ddof1,
            "all": accs,
        }

    print(f"\n  Results for {dataset_name.upper()}:")
    print(f"  {'Strategy':<25} {'Mean Acc':>10} {'Std':>8}")
    for strat, s in summary.items():
        print(f"  {strat:<25} {s['mean']:>9.2f}% {s['std']:>7.2f}%")

    return {"raw": results, "summary": summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["mnist", "fashionmnist", "cifar10"],
        choices=["mnist", "fashionmnist", "cifar10"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--output", default="results/veg_ablation/component_ablation.json"
    )
    parser.add_argument(
        "--model_dir",
        default="models",
        help="directory for the best-accuracy checkpoints",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow overwriting --output and existing checkpoints in --model_dir",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Datasets: {args.datasets}")
    print(f"Seeds: {args.seeds}")
    print(f"Epochs: {args.epochs}")

    strategies = ["dynamic", "entropy_uniform", "entropy"]
    strategy_labels = {
        "dynamic": "Baseline (Dynamic Routing)",
        "entropy_uniform": "VEG — Uniform β (shared scalar)",
        "entropy": "VEG — Per-Capsule β (ours)",
    }

    guard_overwrite(args.output, args.force)
    if args.model_dir:
        for dataset in args.datasets:
            for strategy in strategies:
                for seed in args.seeds:
                    guard_overwrite(
                        Path(args.model_dir)
                        / f"capsule_{dataset}_{strategy}_seed{seed}.pt",
                        args.force,
                    )

    all_results = {}
    for dataset in args.datasets:
        all_results[dataset] = run_dataset(
            dataset,
            strategies,
            args.seeds,
            args.epochs,
            device,
            args.model_dir,
        )

    print("\nComponent ablation")
    header = f"{'Variant':<35}"
    for ds in args.datasets:
        header += f" {ds.upper():>15}"
    print(header)
    print("-" * len(header))

    for strategy in strategies:
        row = f"{strategy_labels[strategy]:<35}"
        for ds in args.datasets:
            s = all_results[ds]["summary"][strategy]
            row += f" {s['mean']:>7.2f}±{s['std']:.2f}%"
        print(row)

    output = {
        "timestamp": datetime.now().isoformat(),
        "seeds": args.seeds,
        "epochs": args.epochs,
        "strategies": strategies,
        "strategy_labels": strategy_labels,
        "beta_stats_note": (
            "results.<dataset>.raw.<arm>[].beta_stats is measured after the last epoch, "
            "not at the saved best-accuracy checkpoint; per-checkpoint beta is in "
            "results/veg_ablation/uniform_beta_checkpoints.json and results/beta_values/"
        ),
        "results": all_results,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to: {args.output}")


if __name__ == "__main__":
    main()
