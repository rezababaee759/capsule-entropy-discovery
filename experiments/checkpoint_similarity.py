#!/usr/bin/env python3
"""Weight-space similarity of each primary ("best") checkpoint to every seeded
checkpoint of the same dataset and arm: overall relative L2, per-tensor relative L2 and
cosine, and whether the two are bit-identical.

    python experiments/checkpoint_similarity.py
"""

from __future__ import annotations

import argparse
import json

import torch

from capsnet import PROJECT_ROOT
from capsnet.io import load_state_dict
from capsnet.provenance import attach

DATASETS = ("mnist", "fashionmnist", "cifar10")
ARMS = ("dynamic", "entropy")
SEEDS = (42, 123, 456, 789, 1000)


def compare(a: dict, b: dict) -> dict:
    per_tensor = {}
    num = den = 0.0
    for k in a:
        x, y = a[k].double().flatten(), b[k].double().flatten()
        d = float((x - y).norm())
        n = float(x.norm())
        num += d * d
        den += n * n
        cos = (
            float(torch.nn.functional.cosine_similarity(x, y, dim=0))
            if x.numel() > 1
            else None
        )
        per_tensor[k] = {"relative_l2": d / n if n else None, "cosine": cos}
    cosines = [v["cosine"] for v in per_tensor.values() if v["cosine"] is not None]
    return {
        "relative_l2": (num**0.5) / (den**0.5),
        "min_cosine": min(cosines),
        "exact_equal": all(torch.equal(a[k], b[k]) for k in a),
        "per_tensor": per_tensor,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", default="results/data/checkpoint_similarity.json")
    args = ap.parse_args()

    out = {"pairs": {}}
    inputs = []
    for ds in DATASETS:
        for arm in ARMS:
            best_path = PROJECT_ROOT / f"models/capsule_{ds}_best_{arm}.pt"
            best = load_state_dict(best_path)
            inputs.append(str(best_path.relative_to(PROJECT_ROOT)))
            key = f"{ds}_best_{arm}"
            out["pairs"][key] = {"stored_keys": sorted(best.keys()), "vs": {}}
            for seed in SEEDS:
                p = PROJECT_ROOT / f"models/capsule_{ds}_{arm}_seed{seed}.pt"
                if not p.exists():
                    continue
                inputs.append(str(p.relative_to(PROJECT_ROOT)))
                out["pairs"][key]["vs"][f"seed{seed}"] = compare(
                    best, load_state_dict(p)
                )
            rel = {s: v["relative_l2"] for s, v in out["pairs"][key]["vs"].items()}
            print(f"{key}: " + ", ".join(f"{s}={r:.4f}" for s, r in rel.items()))

    attach(
        out,
        script="experiments/checkpoint_similarity.py",
        inputs=inputs,
        seeds={"rng": None, "note": "Deterministic; no RNG."},
        sample_counts={"n_pairs": sum(len(v["vs"]) for v in out["pairs"].values())},
        method=(
            "Relative L2 distance between the full parameter vectors of each "
            "primary checkpoint and each seeded checkpoint of the same dataset and "
            "arm, plus per-tensor cosine similarity."
        ),
        packages=["torch"],
    )
    path = PROJECT_ROOT / args.output
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
