#!/usr/bin/env python3
"""Per-digit sparsity of surrogate-predicted capsule importance on the MNIST
taxonomy_calibration split: the Gini coefficient, the number of capsules carrying 50%
and 80% of the mean importance, and the top 10 capsules. Also writes a
<stem>.provenance.json sidecar.

    python experiments/per_digit_sparsity_surrogate.py [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from capsnet.data import class_indices, slice_for, test_set
from capsnet.io import get_device, load_capsule_model, load_predictor, resolve_output
from capsnet.provenance import PROVENANCE_KEY, provenance_block
from capsnet.stats import compute_gini, top_share_count
from capsnet.surrogate import predict_capsule_importance

DATASET = "mnist"


def class_sparsity(capsule_model, predictor_model, images, device) -> dict:
    """Mean predicted importance over `images` [each 1xCxHxW], its top 10 capsules and
    its concentration."""
    importances = torch.stack(
        [
            predict_capsule_importance(
                capsule_model, predictor_model, image.to(device)
            )[0]
            for image in images
        ]
    )
    mean_importance = importances.mean(dim=0)
    top_indices = mean_importance.argsort(descending=True)

    sorted_importance = torch.sort(mean_importance, descending=True)[0].numpy()
    return {
        "sparsity_50": int(top_share_count(sorted_importance, 0.5)),
        "sparsity_80": int(top_share_count(sorted_importance, 0.8)),
        "gini": compute_gini(sorted_importance),
        "top_10_capsules": top_indices[:10].tolist(),
        "top_10_importances": mean_importance[top_indices[:10]].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--capsule_model", default="models/capsule_mnist_best_dynamic.pt"
    )
    parser.add_argument("--predictor", default="models/ablation_predictor.pt")
    parser.add_argument(
        "--samples", type=int, default=50, help="calibration images per class"
    )
    parser.add_argument(
        "--output", help="default: results/data/mnist/per_digit_sparsity_surrogate.json"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="write the canonical output from --capsule_model or --predictor",
    )
    args = parser.parse_args()
    capsule_path, predictor_path = args.capsule_model, args.predictor
    output = resolve_output(
        parser,
        args,
        "results/data/mnist/per_digit_sparsity_surrogate.json",
        ["capsule_model", "predictor"],
    )
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    ds = test_set(DATASET)
    capsule_model = load_capsule_model(capsule_path, DATASET, device=device)
    predictor_model = load_predictor(predictor_path, device)

    start, stop = slice_for(DATASET, "taxonomy_calibration", args.samples)
    results = []
    for c, indices in class_indices(ds).items():
        images = [ds[i][0].unsqueeze(0) for i in indices[start:stop]]
        sparsity = class_sparsity(capsule_model, predictor_model, images, device)
        results.append({"digit": c, **sparsity})
        print(f"Digit {c}: Gini coefficient {sparsity['gini']:.3f}")

    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {output}")

    # The artifact is a bare JSON list, so its provenance goes to a sidecar.
    sidecar = os.path.splitext(output)[0] + ".provenance.json"
    with open(sidecar, "w") as f:
        json.dump(
            {
                PROVENANCE_KEY: provenance_block(
                    script="experiments/per_digit_sparsity_surrogate.py",
                    inputs=[capsule_path, predictor_path],
                    seeds={
                        "rng": None,
                        "note": "No RNG. The calibration image slice is a "
                        "deterministic range from capsnet.data, not a random draw.",
                    },
                    sample_counts={
                        "samples_per_digit": args.samples,
                        "calibration_slice": [start, stop],
                        "n_digits": len(results),
                    },
                    method=(
                        "Per-digit Gini coefficient and cumulative-mass thresholds "
                        "(sparsity_50, sparsity_80) over surrogate-predicted capsule "
                        "importance. Descriptive: no test, no interval."
                    ),
                    packages=["numpy", "scipy"],
                )
            },
            f,
            indent=2,
        )
    print(f"Wrote {sidecar}")


if __name__ == "__main__":
    main()
