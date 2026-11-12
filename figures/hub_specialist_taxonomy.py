"""Hub-Specialist taxonomy figure: the number of capsules in each role, with its share.

Usage: python figures/hub_specialist_taxonomy.py
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from capsnet import PROJECT_ROOT
from capsnet import paper_style as ps

ROLE_COLOURS = {
    "Primary\nHubs": "#E8735A",
    "Secondary\nHubs": "#F5C26B",
    "Specialists": "#5B9BD5",
    "Other": "#B0B0B0",
}


def build_taxonomy_figure(tax, out_path):
    counts = [
        len(tax["primary_hubs"]),
        len(tax["secondary_hubs"]),
        len(tax["specialists"]),
        len(tax["generalists"]) + len(tax["non_critical"]),
    ]
    n_total = sum(counts)
    labels = list(ROLE_COLOURS)
    print(
        "MNIST taxonomy: "
        + ", ".join(f"{c} {l.replace(chr(10), ' ')}" for c, l in zip(counts, labels))
        + f" ({n_total} total)"
    )

    ps.apply()
    fig = ps.figure(height_in=2.6)
    ax = fig.subplots()
    bars = ax.bar(
        labels,
        counts,
        color=list(ROLE_COLOURS.values()),
        edgecolor="white",
        linewidth=0.5,
        width=0.6,
    )
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 12,
            f"{count}\n({100 * count / n_total:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=ps.SMALL_PT,
            fontweight="bold",
        )
    ax.set_ylabel("Number of capsules")
    ax.set_ylim(0, max(counts) * 1.3)
    ax.spines[["top", "right"]].set_visible(False)

    ps.save(fig, out_path)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Build hub_specialist_taxonomy.png")
    ap.add_argument(
        "--output", help="default: docs/figures/hub_specialist_taxonomy.png"
    )
    args = ap.parse_args()
    output = args.output or PROJECT_ROOT / "docs/figures/hub_specialist_taxonomy.png"

    with open(PROJECT_ROOT / "results/data/mnist/hub_taxonomy.json") as f:
        tax = json.load(f)

    build_taxonomy_figure(tax, Path(output))


if __name__ == "__main__":
    main()
