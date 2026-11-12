#!/usr/bin/env python3
"""Learned VEG beta_i by native capsule role, and all beta_i pooled, drawn at print
size from a checkpoint and its own taxonomy.

    python figures/veg_beta_by_role.py --dataset mnist
    python figures/veg_beta_by_role.py --dataset cifar10
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")

from capsnet import paper_style as ps  # noqa: E402
from capsnet.io import (  # noqa: E402
    VEG_PRIMARY_TAG,
    load_capsule_model,
    resolve_output,
    veg_checkpoint,
    veg_taxonomy,
)
from capsnet.taxonomy import (  # noqa: E402
    ORDERED_ROLE_NAMES,
    assert_role_coverage,
    capsule_roles,
)

OUTPUT = {
    "mnist": "results/beta_values/native/veg_beta_by_role_mnist.png",
    "cifar10": "results/beta_values/native_cifar10/veg_beta_by_role_cifar10.png",
}


def plot_beta_by_role(beta_values, capsule_to_role: dict, out_path: str) -> None:
    num_capsules = len(beta_values)
    assert_role_coverage(capsule_to_role, num_capsules)
    role_betas = {}
    for i in range(num_capsules):
        role_betas.setdefault(capsule_to_role[i], []).append(beta_values[i])
    ordered = [r for r in ORDERED_ROLE_NAMES if r in role_betas]

    ps.apply()
    fig, axes = ps.subplots(1, 2, height_in=2.5, width_ratios=[1.6, 1])
    ax = axes[0]
    ax.boxplot(
        [role_betas[r] for r in ordered],
        labels=[
            r.replace(" ", "\n").replace("Non-critical", "Non-\ncritical")
            for r in ordered
        ],
        widths=0.55,
        flierprops={"markersize": 2, "markeredgewidth": 0.4},
        medianprops={"color": "#d62728", "linewidth": 1.0},
    )
    ax.set_ylabel(r"Learned $\beta_i$")
    ax.set_title(r"$\beta_i$ by functional role")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    ax.hist(
        beta_values,
        bins=50,
        alpha=0.8,
        color="#5B9BD5",
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_xlabel(r"Learned $\beta_i$")
    ax.set_ylabel("Number of capsules")
    ax.set_title(f"All {num_capsules:,} capsules")
    ax.grid(True, alpha=0.3)

    ps.save(fig, out_path)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, choices=sorted(OUTPUT))
    ap.add_argument("--model", help="default models/capsule_{dataset}_best_entropy.pt")
    ap.add_argument(
        "--taxonomy",
        help="default results/data/{dataset}/veg_best_entropy/hub_taxonomy.json",
    )
    ap.add_argument("--output", help="PNG path; default per dataset")
    ap.add_argument(
        "--force",
        action="store_true",
        help="write the canonical figure from --model or --taxonomy",
    )
    args = ap.parse_args()
    ds = args.dataset
    output = resolve_output(ap, args, OUTPUT[ds], ["model", "taxonomy"])
    model_path = args.model or veg_checkpoint(ds, VEG_PRIMARY_TAG)
    tax_path = args.taxonomy or veg_taxonomy(ds, VEG_PRIMARY_TAG)

    model = load_capsule_model(model_path, ds, "entropy", "cpu")
    beta_values = model.digits.routing_module.beta.detach().cpu().numpy()
    with open(tax_path) as f:
        capsule_to_role = capsule_roles(json.load(f))
    plot_beta_by_role(beta_values, capsule_to_role, output)


if __name__ == "__main__":
    main()
