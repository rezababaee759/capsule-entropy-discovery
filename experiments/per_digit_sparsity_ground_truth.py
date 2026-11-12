#!/usr/bin/env python3
"""Per-digit sparsity of the MNIST ground-truth importance matrix (negatives set to
zero): Gini and the capsule count carrying 80% of the mass, compared with the
surrogate's.

    python experiments/per_digit_sparsity_ground_truth.py [--check] [--output PATH]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from capsnet import PROJECT_ROOT
from capsnet.provenance import attach, check_artifact
from capsnet.stats import compute_gini, top_share_count

MATRIX = "results/data/mnist/importance_matrix.npy"
SURROGATE = "results/data/mnist/per_digit_sparsity_surrogate.json"
OUTPUT = "results/data/mnist/per_digit_sparsity_ground_truth.json"
SHARE = 0.8


def concentration(values: np.ndarray) -> dict:
    """Gini and the smallest number of capsules carrying SHARE of the clipped total."""
    clipped = np.maximum(values, 0.0)
    ranked = np.sort(clipped)[::-1]
    k = int(top_share_count(ranked, SHARE))
    return {
        "gini": float(compute_gini(clipped)),
        "k_80": k,
        "percent_80": 100.0 * k / values.size,
    }


def build() -> dict:
    matrix = np.load(PROJECT_ROOT / MATRIX)
    n_digits, n_capsules = matrix.shape
    surrogate = {
        int(r["digit"]): r for r in json.loads((PROJECT_ROOT / SURROGATE).read_text())
    }

    per_digit = []
    for d in range(n_digits):
        row = matrix[d]
        entry = {
            "digit": d,
            **concentration(row),
            "n_negative": int((row < 0).sum()),
            "top_capsule": int(row.argmax()),
        }
        entry["surrogate_top_capsule"] = int(surrogate[d]["top_10_capsules"][0])
        entry["surrogate_gini"] = float(surrogate[d]["gini"])
        per_digit.append(entry)

    return {
        "description": (
            "Per-digit sparsity of ground-truth single-capsule ablation importance, MNIST "
            "primary dynamic-routing model, taxonomy-calibration slice. Negative mean "
            "importances are set to zero before the Gini coefficient and the 80% share are "
            "computed, as in experiments/ground_truth_sparsity.py."
        ),
        "importance_source": MATRIX,
        "importance_definition": (
            "Row d is the mean over the calibration images of true digit d of the drop in "
            "the predicted class's capsule length when one primary capsule is zeroed "
            "(experiments/hub_taxonomy.py)."
        ),
        "negative_values": "set to zero",
        "share": SHARE,
        "n_capsules": n_capsules,
        "n_digits": n_digits,
        "per_digit": per_digit,
        "mean_over_digits": {
            "gini": float(np.mean([r["gini"] for r in per_digit])),
            "percent_80": float(np.mean([r["percent_80"] for r in per_digit])),
            "percent_80_range": [
                min(r["percent_80"] for r in per_digit),
                max(r["percent_80"] for r in per_digit),
            ],
            "k_80_range": [
                min(r["k_80"] for r in per_digit),
                max(r["k_80"] for r in per_digit),
            ],
            "gini_range": [
                min(r["gini"] for r in per_digit),
                max(r["gini"] for r in per_digit),
            ],
        },
        "digit_averaged_importance": concentration(matrix.mean(axis=0)),
        "negative_fraction_of_matrix": float((matrix < 0).mean()),
        "surrogate_comparison": {
            "source": SURROGATE,
            "top_capsule_agreement": sum(
                r["top_capsule"] == r["surrogate_top_capsule"] for r in per_digit
            ),
            "mean_gini_surrogate": float(
                np.mean([r["surrogate_gini"] for r in per_digit])
            ),
            "mean_gini_ground_truth": float(np.mean([r["gini"] for r in per_digit])),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if the committed artifact differs from a fresh reduction",
    )
    ap.add_argument(
        "--output",
        default=OUTPUT,
        help="artifact path, relative to the repository root",
    )
    args = ap.parse_args()

    out = build()
    if args.check:
        check_artifact(args.output, out, "experiments/per_digit_sparsity_ground_truth.py")
        return

    attach(
        out,
        script="experiments/per_digit_sparsity_ground_truth.py",
        inputs=[MATRIX, SURROGATE],
        seeds={"rng": None, "note": "Pure read-and-reduce over a saved matrix."},
        sample_counts={"n_digits": out["n_digits"], "n_capsules": out["n_capsules"]},
        method=(
            "Per digit: clip negative importance to zero, Gini coefficient, and the "
            "smallest k whose top-k cumulative sum reaches 80% of the clipped total."
        ),
        packages=["numpy"],
    )
    (PROJECT_ROOT / args.output).write_text(json.dumps(out, indent=2) + "\n")
    m = out["mean_over_digits"]
    print(f"{'digit':>5} {'gini':>6} {'k80':>5} {'%80':>6}")
    for r in out["per_digit"]:
        print(
            f"{r['digit']:>5} {r['gini']:>6.3f} {r['k_80']:>5} {r['percent_80']:>6.1f}"
        )
    print(
        f"mean gini {m['gini']:.3f}, mean %80 {m['percent_80']:.1f}; "
        f"digit-averaged gini {out['digit_averaged_importance']['gini']:.3f}"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
