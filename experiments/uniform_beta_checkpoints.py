#!/usr/bin/env python3
"""Shared scalar beta of every Uniform-beta VEG checkpoint (best-accuracy, as saved in
models/), with mean, SD and range overall and per dataset.

    python experiments/uniform_beta_checkpoints.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from capsnet import PROJECT_ROOT
from capsnet.io import resolve_output
from capsnet.provenance import attach

BETA_KEY = "digits.routing_module.beta"


def read_beta(path: Path) -> float:
    """Read the shared scalar beta out of one Uniform-beta checkpoint."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if BETA_KEY not in state:
        raise SystemExit(
            f"{path}: no '{BETA_KEY}' in the checkpoint. Either this is not a VEG "
            f"model or the routing module was renamed; refusing to guess."
        )
    tensor = state[BETA_KEY]
    if tuple(tensor.shape) != (1,):
        raise SystemExit(
            f"{path}: expected the Uniform-beta arm's single shared scalar, got "
            f"shape {tuple(tensor.shape)}."
        )
    return float(tensor.item())


def reduce_beta(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--datasets", nargs="+", default=["mnist", "fashionmnist", "cifar10"]
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1000])
    ap.add_argument("--model_dir", default="models")
    ap.add_argument(
        "--output", help="default: results/veg_ablation/uniform_beta_checkpoints.json"
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="write the canonical output from other --datasets, --seeds or --model_dir",
    )
    args = ap.parse_args()
    output = resolve_output(
        ap,
        args,
        "results/veg_ablation/uniform_beta_checkpoints.json",
        ["datasets", "seeds", "model_dir"],
    )

    per_model = []
    checkpoint_paths = []
    for dataset in args.datasets:
        for seed in args.seeds:
            path = (
                PROJECT_ROOT
                / args.model_dir
                / f"capsule_{dataset}_entropy_uniform_seed{seed}.pt"
            )
            if not path.exists():
                raise SystemExit(
                    f"missing checkpoint {path}; the summary must cover every trained "
                    f"model (run train/component_ablation.py first)."
                )
            per_model.append(
                {"dataset": dataset, "seed": seed, "beta": read_beta(path)}
            )
            checkpoint_paths.append(str(path))

    values = np.array([m["beta"] for m in per_model], dtype=float)
    overall = reduce_beta(values)
    overall["argmin"] = per_model[int(values.argmin())] | {"which": "min"}
    overall["argmax"] = per_model[int(values.argmax())] | {"which": "max"}

    out = {
        "description": "Shared scalar beta learned by the Uniform-beta VEG arm, read "
        "from the best-accuracy checkpoint of every trained model. The "
        "per-run beta_stats in component_ablation.json samples beta "
        "after the final epoch instead and is not interchangeable.",
        "n_models": len(per_model),
        "datasets": args.datasets,
        "seeds": args.seeds,
        "parameter": BETA_KEY,
        "checkpoint": "best accuracy",
        "across_all_models": overall,
        "per_dataset": {
            ds: reduce_beta(
                np.array(
                    [m["beta"] for m in per_model if m["dataset"] == ds], dtype=float
                )
            )
            for ds in args.datasets
        },
        "per_model": per_model,
    }

    attach(
        out,
        script="experiments/uniform_beta_checkpoints.py",
        inputs=checkpoint_paths,
        seeds={
            "rng": None,
            "note": "Pure read-and-reduce over checkpoints: no training, no "
            "randomness, no thresholds.",
        },
        sample_counts={
            "n_models": len(per_model),
            "datasets": sorted({m["dataset"] for m in per_model}),
            "seeds": sorted({m["seed"] for m in per_model}),
        },
        method=(
            "Reads the shared scalar digits.routing_module.beta from each "
            "best-accuracy Uniform-beta checkpoint and reduces to mean, SD, min, "
            "max, argmin and argmax overall and per dataset. Deliberately not "
            "read from component_ablation.json beta_stats, which samples beta "
            "after the final epoch."
        ),
        packages=["numpy"],
    )

    path = PROJECT_ROOT / output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")

    print(f"{'dataset':<14} {'seed':>6} {'beta':>9}")
    print("-" * 31)
    for m in per_model:
        print(f"{m['dataset']:<14} {m['seed']:>6} {m['beta']:>9.4f}")
    print("-" * 31)
    print(
        f"{len(per_model)} models: mean {overall['mean']:.4f}, "
        f"sd {overall['std']:.4f}, "
        f"range {overall['min']:.4f}-{overall['max']:.4f} "
        f"({overall['argmin']['dataset']} seed {overall['argmin']['seed']} to "
        f"{overall['argmax']['dataset']} seed {overall['argmax']['seed']})."
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
