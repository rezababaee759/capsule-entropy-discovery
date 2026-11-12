#!/usr/bin/env python3
"""Roll the four per-model VEG beta analyses of one dataset into the `per_model` summary
that experiments/role_intervention.py reads.

    python experiments/veg_beta_aggregate.py --dataset mnist
"""

from __future__ import annotations

import argparse
import json
import sys

from capsnet import PROJECT_ROOT
from capsnet.io import veg_beta_analysis
from capsnet.provenance import attach
from capsnet.taxonomy import ORDERED_ROLE_NAMES

# per_model key -> VEG model tag
MODEL_KEYS = {
    "best_entropy_pt": "best_entropy",
    "seed42": "seed42",
    "seed123": "seed123",
    "seed456": "seed456",
}

DATASETS = {
    "mnist": {
        "name": "MNIST",
        "target": "results/beta_values/veg_beta_analysis_mnist_native.json",
    },
    "cifar10": {
        "name": "CIFAR-10",
        "target": "results/beta_values/veg_beta_analysis_cifar_native.json",
    },
}


def summarise(per_model_json: dict) -> dict:
    """Reduce one experiments/veg_beta_analysis.py output to its aggregate entry."""
    role_stats = per_model_json["role_statistics"]

    def is_role(v):
        return isinstance(v, dict) and "count" in v and "mean" in v

    roles = [r for r in ORDERED_ROLE_NAMES if is_role(role_stats.get(r))]
    roles += [r for r in role_stats if is_role(role_stats[r]) and r not in roles]
    pairwise = role_stats.get("pairwise_tests", {})

    return {
        "model_path": per_model_json["model_path"],
        "levene": dict(role_stats["levene"]),
        "role_counts": {r: role_stats[r]["count"] for r in roles},
        "role_means": {r: role_stats[r]["mean"] for r in roles},
        "anova": dict(role_stats["anova"]),
        "welch_anova": dict(role_stats["welch_anova"]),
        "effect_size": dict(role_stats["effect_size"]),
        "primary_hub_vs_specialist": pairwise.get("hub_vs_specialist", {}),
        "pairwise_tests_all": {
            k: v for k, v in pairwise.items() if isinstance(v, dict)
        },
        "pairwise_multiplicity": pairwise.get("_multiplicity", {}),
        "beta_vs_importance_pearson": per_model_json["correlations"][
            "beta_vs_importance"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--output", help="default: the dataset's native roll-up path")
    args = parser.parse_args()
    cfg = DATASETS[args.dataset]
    output = args.output or cfg["target"]

    inputs = {
        key: veg_beta_analysis(args.dataset, tag) for key, tag in MODEL_KEYS.items()
    }
    missing = [p for p in inputs.values() if not (PROJECT_ROOT / p).exists()]
    if missing:
        print("ERROR: missing per-model inputs:", *missing, sep="\n  ", file=sys.stderr)
        return 1

    per_model = {
        key: summarise(json.loads((PROJECT_ROOT / path).read_text()))
        for key, path in inputs.items()
    }
    description = (
        f"VEG beta-by-role statistics for each trained per-capsule-beta {cfg['name']} "
        "VEG model, with roles derived natively on that model by exhaustive "
        "single-capsule ablation (ground truth) of its own routing behaviour, on the "
        "taxonomy-calibration slice. Capsule indices carry no guaranteed "
        "correspondence across separately trained models, so roles are not "
        "transferred by index from the dynamic-routing baseline."
    )
    document = {"description": description, "per_model": per_model}
    attach(
        document,
        script="experiments/veg_beta_aggregate.py",
        inputs=sorted(inputs.values()),
        seeds={"rng": None, "note": "Aggregation only; no randomness."},
        sample_counts={"n_models": len(per_model), "models": sorted(per_model)},
        method=(
            "Rolls the per-model VEG beta analyses into the per_model summary and "
            "writes description; the file holds no other key."
        ),
        packages=["numpy"],
    )
    path = PROJECT_ROOT / output
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(document, handle, indent=2)
    print(f"Wrote {output} ({len(per_model)} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
