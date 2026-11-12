"""Voting-analysis figure: routing correctness, coupling and entropy by role, and
sample sizes, from the pooled three-seed symmetric evaluation.

Usage: python figures/voting_analysis.py
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

from capsnet import PROJECT_ROOT
from capsnet import paper_style as ps
from capsnet.io import resolve_output


def main() -> None:
    ap = argparse.ArgumentParser(description="Build voting_analysis_comprehensive.png")
    ap.add_argument(
        "--output", help="default: docs/figures/voting_analysis_comprehensive.png"
    )
    ap.add_argument(
        "--records",
        default="results/data/mnist/pooled_42_123_789/symmetric_raw_records.parquet",
    )
    ap.add_argument(
        "--sym_json",
        default="results/data/mnist/pooled_42_123_789/symmetric_evaluation_glmm.json",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="write the canonical figure from --records or --sym_json",
    )
    args = ap.parse_args()
    out_path = Path(
        resolve_output(
            ap,
            args,
            str(PROJECT_ROOT / "docs/figures/voting_analysis_comprehensive.png"),
            ["records", "sym_json"],
        )
    )

    df = pd.read_parquet(PROJECT_ROOT / args.records)
    with open(PROJECT_ROOT / args.sym_json) as f:
        glmm = json.load(f)

    pooled = glmm["companion_analysis_capsule_level"]["pooled"]

    colors = {"hub": "steelblue", "specialist": "darkred", "random": "gray"}
    role_order = ["hub", "specialist", "random"]
    # 'random' is the artifact's key for the matched non-critical control.
    role_labels = ["Hub", "Specialist", "Non-critical"]
    cond_order = ["within", "cross"]

    ps.apply()
    fig, axes = ps.subplots(2, 2, height_in=3.9)

    def style_violins(parts):
        for pc, r in zip(parts["bodies"], role_order):
            pc.set_facecolor(colors[r])
            pc.set_alpha(0.7)
            pc.set_edgecolor("black")
            pc.set_linewidth(0.6)
        for pn in ("cbars", "cmins", "cmaxes", "cmeans", "cmedians"):
            if pn in parts:
                parts[pn].set_color("black")
                parts[pn].set_linewidth(0.8)

    # Panel A: routing correctness, within vs cross, grouped by role
    ax = axes[0, 0]
    x = np.arange(len(role_order))
    width = 0.36
    for j, cond in enumerate(cond_order):
        means_pct = [
            100
            * (
                pooled[r]["within_mean"]
                if cond == "within"
                else pooled[r]["cross_mean"]
            )
            for r in role_order
        ]
        offset = (j - 0.5) * width
        ax.bar(
            x + offset,
            means_pct,
            width,
            label=f"{cond.capitalize()}-domain",
            color=[colors[r] for r in role_order],
            alpha=0.9 if cond == "within" else 0.45,
            edgecolor="black",
            linewidth=0.5,
        )
        for xi, m in zip(x + offset, means_pct):
            ax.text(
                xi,
                m + 1.5,
                f"{m:.1f}",
                ha="center",
                va="bottom",
                fontsize=ps.SMALL_PT,
                fontweight="bold",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(role_labels)
    ax.set_ylabel("Routing correctness (%)")
    ax.set_ylim([0, 128])
    ax.set_yticks(range(0, 101, 20))
    ax.set_title("(A) Routing correctness")
    ax.legend(
        loc="upper center",
        frameon=False,
        ncol=2,
        handlelength=1.2,
        columnspacing=1.0,
        borderaxespad=0.2,
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Panel B: coupling-to-true-digit distribution (pooled within+cross)
    ax = axes[0, 1]
    data_b = [
        df.loc[df["role"] == r, "coupling_to_true_digit"].values for r in role_order
    ]
    style_violins(
        ax.violinplot(
            data_b,
            positions=range(len(role_order)),
            widths=0.6,
            showmeans=True,
            showmedians=True,
            showextrema=True,
        )
    )
    ax.set_xticks(range(len(role_order)))
    ax.set_xticklabels(role_labels)
    for i in range(len(role_order)):
        mv = data_b[i].mean()
        ax.text(
            i,
            0.22,
            f"μ={mv:.3f}",
            ha="center",
            va="bottom",
            fontsize=ps.SMALL_PT,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none"),
        )
    ax.set_ylabel("Coupling to true digit")
    ax.set_title("(B) Coupling coefficient")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Panel C: routing entropy distribution (pooled within+cross)
    ax = axes[1, 0]
    data_c = [df.loc[df["role"] == r, "entropy"].values for r in role_order]
    style_violins(
        ax.violinplot(
            data_c,
            positions=range(len(role_order)),
            widths=0.6,
            showmeans=True,
            showmedians=True,
            showextrema=True,
        )
    )
    ax.set_xticks(range(len(role_order)))
    ax.set_xticklabels(role_labels)
    for i in range(len(role_order)):
        mv = data_c[i].mean()
        ax.text(
            i,
            1.55,
            f"μ={mv:.2f}",
            ha="center",
            va="center",
            fontsize=ps.SMALL_PT,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none"),
        )
    ax.set_ylabel("Routing entropy (nats)")
    ax.set_title("(C) Routing entropy")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Panel D: capsule-seed replicate counts (the unit of analysis), with raw rows as a note
    ax = axes[1, 1]
    n_capsule_seed = [pooled[r]["n_capsules_pooled_across_seeds"] for r in role_order]
    n_raw_rows = [len(df[df["role"] == r]) for r in role_order]
    bars = ax.bar(
        role_labels,
        n_capsule_seed,
        color=[colors[r] for r in role_order],
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
        width=0.6,
    )
    for bar, n_cs, n_raw in zip(bars, n_capsule_seed, n_raw_rows):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            n_cs + max(n_capsule_seed) * 0.03,
            f"n = {n_cs:,}\n{n_raw / 1000:,.0f}k obs.",
            ha="center",
            va="bottom",
            fontsize=ps.SMALL_PT,
            fontweight="bold",
        )
    ax.set_ylabel("Capsule×model replicates")
    ax.set_ylim([0, max(n_capsule_seed) * 1.4])
    ax.set_title("(D) Sample size")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    ps.save(fig, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
