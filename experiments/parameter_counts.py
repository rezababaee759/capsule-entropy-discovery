#!/usr/bin/env python3
"""Parameters of every checkpoint in models/, counted from the saved state dict, with
the VEG beta parameters (digits.routing_module.beta) and their share of the rest:
beta_share_pct = 100 * beta / (total - beta). Surrogate training sets are skipped.

    python experiments/parameter_counts.py           # write the artifact
    python experiments/parameter_counts.py --check   # exit 1 if it is stale
"""

from __future__ import annotations

import argparse
import json

import torch

from capsnet import PROJECT_ROOT
from capsnet.io import load_state_dict
from capsnet.provenance import attach, check_artifact

MODELS = PROJECT_ROOT / "models"
OUTPUT = "results/data/parameter_counts.json"
BETA_KEY = "digits.routing_module.beta"
TENSOR_STORE_SUFFIX = "_training_data.pt"


def build() -> dict:
    paths = sorted(MODELS.glob("*.pt"))
    if not paths:
        raise SystemExit(
            "no checkpoints in models/: parameter counts are read from the checkpoints "
            "(see README, 'Checkpoints')"
        )
    checkpoints, skipped = {}, []
    for path in paths:
        if path.name.endswith(TENSOR_STORE_SUFFIX):
            skipped.append(path.name)
            continue
        sd = load_state_dict(path)
        total = sum(t.numel() for t in sd.values() if torch.is_tensor(t))
        beta = int(sd[BETA_KEY].numel()) if BETA_KEY in sd else 0
        checkpoints[path.stem] = {
            "total_parameters": int(total),
            "beta_parameters": beta,
            "beta_share_pct": 100.0 * beta / (total - beta),
        }
    return {
        "description": (
            "Parameters per checkpoint, counted from the saved state dict, and the VEG "
            "beta parameters' share of the rest of the network."
        ),
        "beta_key": BETA_KEY,
        "checkpoints": checkpoints,
        "skipped_tensor_stores": skipped,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if the committed artifact differs from a fresh count",
    )
    ap.add_argument(
        "--output",
        default=OUTPUT,
        help="artifact path, relative to the repository root",
    )
    args = ap.parse_args()

    out = build()
    if args.check:
        check_artifact(args.output, out, "experiments/parameter_counts.py")
        return

    attach(
        out,
        script="experiments/parameter_counts.py",
        inputs=[p.relative_to(PROJECT_ROOT) for p in sorted(MODELS.glob("*.pt"))],
        seeds={
            "rng": None,
            "note": "Counts read from saved state dicts; no randomness.",
        },
        sample_counts={"n_checkpoints": len(out["checkpoints"])},
        method=(
            "Sum of numel over every tensor in each checkpoint's state dict; beta = numel "
            f"of {BETA_KEY}; share = beta / (total - beta)."
        ),
        packages=["torch"],
    )
    (PROJECT_ROOT / args.output).write_text(json.dumps(out, indent=2) + "\n")
    for name, c in out["checkpoints"].items():
        print(
            f"{name:45s} {c['total_parameters']:>10,d}  beta {c['beta_parameters']:>5d}  "
            f"{c['beta_share_pct']:.4f}%"
        )


if __name__ == "__main__":
    main()
