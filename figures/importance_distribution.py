"""Histograms of ablation importance by taxonomy role (Primary Hubs, Secondary Hubs,
Generalists, Specialists), over each role's capsules and all digits.

Usage: python figures/importance_distribution.py --taxonomy_json results/data/mnist/hub_taxonomy.json
"""

import argparse
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")

from capsnet import paper_style as ps
from capsnet.io import resolve_output
from capsnet.taxonomy import load_importance_matrix


def collect_importance_scores_by_type(importance_matrix, classification):
    """For each role, every importance value of its capsules across all digits.

    importance_matrix is [n_capsules, 10]; classification maps role -> capsule indices.
    """
    scores_by_type = {}

    type_config = {
        "primary_hubs": ("Primary Hubs", "#d62728"),
        "secondary_hubs": ("Secondary Hubs", "#ff7f0e"),
        "generalists": ("Generalists", "#ffbb78"),
        "specialists": ("Specialists", "#1f77b4"),
    }

    for key, (label, color) in type_config.items():
        capsule_indices = classification.get(key, [])

        if len(capsule_indices) == 0:
            print(f"  {label}: no capsules in the classification")
            scores_by_type[key] = {
                "scores": np.array([]),
                "label": label,
                "color": color,
                "count": 0,
            }
            continue

        # importance_matrix: [n_capsules, 10]; each capsule contributes one value per digit
        scores_list = []
        valid_indices = 0
        invalid_indices = []

        for cap_idx in capsule_indices:
            if cap_idx < importance_matrix.shape[0]:
                cap_scores = importance_matrix[cap_idx, :]
                scores_list.append(cap_scores)
                valid_indices += 1
            else:
                invalid_indices.append(cap_idx)

        if invalid_indices:
            raise SystemExit(
                f"{label}: {len(invalid_indices)} capsule indices are outside the "
                f"importance matrix (>= {importance_matrix.shape[0]}): "
                f"{invalid_indices[:10]}"
            )

        all_scores = np.concatenate(scores_list)
        # Negative importance is kept: ablating the capsule raised confidence
        # (an antagonistic capsule), which is part of the distribution shown.
        n_neg = int((all_scores < 0).sum())
        if n_neg:
            print(
                f"    {n_neg}/{len(all_scores)} values are negative "
                f"(antagonistic); retained."
            )

        print(
            f"  {label}: {len(all_scores)} importance values from "
            f"{valid_indices}/{len(capsule_indices)} capsules "
            f"(range=[{all_scores.min():.4f}, {all_scores.max():.4f}])"
        )

        scores_by_type[key] = {
            "scores": all_scores,
            "label": label,
            "color": color,
            "count": len(capsule_indices),
        }

    if all(entry["count"] == 0 for entry in scores_by_type.values()):
        raise SystemExit(
            f"every role in {list(type_config)} is empty; there is nothing to plot."
        )

    return scores_by_type


def create_importance_distribution_figure(scores_by_type, output_path, bins=50):
    """Four role histograms; scores_by_type maps role -> dict with 'scores', 'label',
    'color'."""
    ps.apply()
    fig, axes = ps.subplots(2, 2, height_in=4.2)
    axes = axes.flatten()

    type_order = ["primary_hubs", "secondary_hubs", "generalists", "specialists"]

    for idx, key in enumerate(type_order):
        ax = axes[idx]
        data = scores_by_type[key]
        scores = data["scores"]
        label = data["label"]
        color = data["color"]

        if len(scores) == 0:
            ax.text(
                0.5,
                0.5,
                f"No data for {label}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(label)
            ax.set_xlabel("Importance score")
            ax.set_ylabel("Frequency")
            continue

        # Log-scale y-axis handles the heavy-tailed distributions.
        ax.hist(
            scores, bins=bins, color=color, alpha=0.8, edgecolor="black", linewidth=0.3
        )
        ax.set_yscale("log")

        ax.set_xlabel("Importance score")
        ax.set_ylabel("Frequency (log scale)")
        ax.set_title(label)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_axisbelow(True)

        mean_score = scores.mean()
        median_score = np.median(scores)
        max_score = scores.max()
        # Significant figures, not fixed decimals: role means and medians can sit far
        # below 1e-3, which three fixed decimals print as 0.000.
        stats_text = (
            f"Mean: {mean_score:.2g}\nMedian: {median_score:.2g}\nMax: {max_score:.2g}"
        )
        ax.text(
            0.97,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=ps.SMALL_PT,
            va="top",
            ha="right",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.6, linewidth=0.4
            ),
        )

        # The axis spans the full data range. Its left edge is the most negative
        # (antagonistic) value, not zero: a zero origin would hide the negative
        # tail and fold the negatives of the bin straddling zero into a bar drawn
        # as positive mass. The dashed line marks zero.
        span = scores.max() - min(scores.min(), 0.0)
        ax.set_xlim(
            left=min(scores.min(), 0.0) - 0.02 * span, right=scores.max() + 0.05 * span
        )
        ax.axvline(0.0, color="black", linewidth=0.6, linestyle="--")

    ps.save(fig, output_path)
    print(f"  Saved figure to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot importance distribution histograms by capsule type"
    )
    parser.add_argument(
        "--taxonomy_json",
        type=str,
        default="results/data/mnist/hub_taxonomy.json",
        help="hub_taxonomy.json; importance_matrix.npy is read from beside it",
    )
    parser.add_argument(
        "--output", help="default: docs/figures/importance_distribution.png"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="write the canonical figure from --taxonomy_json",
    )
    args = parser.parse_args()
    output = resolve_output(
        parser, args, "docs/figures/importance_distribution.png", ["taxonomy_json"]
    )

    print(f"Loading taxonomy data from: {args.taxonomy_json}")
    with open(args.taxonomy_json, "r") as f:
        classification = json.load(f)
    importance_matrix = load_importance_matrix(args.taxonomy_json).T  # [capsules, 10]

    scores_by_type = collect_importance_scores_by_type(
        importance_matrix, classification
    )
    create_importance_distribution_figure(scores_by_type, output)


if __name__ == "__main__":
    main()
