"""Single-sample Specialist tracking: every taxonomy Specialist responding to one test
image, grouped by home digit and ranked by importance for it.

Writes <output_dir>/single_sample_d<digit>_s<sample_idx>_<n>caps.png.

Usage: python figures/specialist_interactions.py --digit 4 --sample_idx 4 --top_k_per_digit 2 --force
"""

import argparse
import os
from typing import Dict, List, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch

from capsnet import paper_style as ps
from capsnet.data import class_indices
from capsnet.explain import (
    add_figure_args,
    capsule_routing,
    load_figure_inputs,
    spatial_attribution,
)
from capsnet.io import guard_overwrite
from capsnet.model import CapsuleNetwork


def create_single_sample_tracking_figure(
    model: CapsuleNetwork,
    image: torch.Tensor,
    true_label: int,
    specialists_by_digit: Dict[int, List[Tuple[int, float]]],
    device: torch.device,
) -> plt.Figure:
    """One column per home digit, in two blocks (digits 0-4 above, 5-9 below). Within a
    column, each specialist (capsule, importance for its home digit) shows its
    attribution heatmap over the sample and its coupling distribution."""
    ps.apply()

    max_per_digit = max(max(len(s) for s in specialists_by_digit.values()), 1)

    fig = ps.figure(height_in=0.95 + 1.8 * max_per_digit, layout=None)
    outer_gs = gridspec.GridSpec(
        2,
        5,
        figure=fig,
        wspace=0.25,
        hspace=0.16,
        left=0.02,
        right=0.98,
        top=0.935,
        bottom=0.115,
    )

    for digit in range(10):
        specs = specialists_by_digit[digit]
        n_specs = len(specs)
        cell = outer_gs[digit // 5, digit % 5]
        # Label row, then per specialist: heatmap, coupling bar, spacer (for the next title)
        height_ratios = [0.45] + [1.0, 0.3, 0.3] * max_per_digit
        col_gs = gridspec.GridSpecFromSubplotSpec(
            1 + 3 * max_per_digit,
            1,
            subplot_spec=cell,
            height_ratios=height_ratios,
            hspace=0.1,
        )

        ax_label = fig.add_subplot(col_gs[0, 0])
        is_true = digit == true_label
        ax_label.text(
            0.5,
            0.95,
            f"D{digit} (true digit)" if is_true else f"D{digit}",
            ha="center",
            va="top",
            fontsize=ps.BASE_PT,
            fontweight="bold",
            color="green" if is_true else "black",
        )
        ax_label.axis("off")

        for i, (cap_idx, imp) in enumerate(specs):
            routing_info = capsule_routing(model, image.unsqueeze(0), cap_idx, device)
            coupling = routing_info["coupling_coeffs"]
            top_digit = routing_info["top_digit"]
            # Border: does this capsule route the sample to its true digit?
            border_color = "green" if top_digit == true_label else "red"

            ax_heat = fig.add_subplot(col_gs[1 + 3 * i, 0])
            heatmap = spatial_attribution(model, image, cap_idx, device)
            ax_heat.imshow(image.squeeze().numpy(), cmap="gray", alpha=0.4)
            ax_heat.imshow(heatmap, cmap="hot", alpha=0.7, vmin=0, vmax=1)
            ax_heat.set_title(
                f"#{cap_idx} · {imp:.3f}",
                fontsize=ps.SMALL_PT,
                fontweight="normal",
                pad=1.5,
            )
            for spine in ax_heat.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(1.0)
            ax_heat.set_xticks([])
            ax_heat.set_yticks([])

            ax_bar = fig.add_subplot(col_gs[2 + 3 * i, 0])
            colors = ["lightgray"] * 10
            colors[top_digit] = "green"
            if true_label != top_digit:
                colors[true_label] = "red"
            ax_bar.bar(
                np.arange(10), coupling, color=colors, edgecolor="black", linewidth=0.2
            )
            ax_bar.set_xlim(-0.6, 9.6)
            ax_bar.set_ylim(0, 1)
            ax_bar.set_xticks([])
            ax_bar.set_yticks([])
            for spine in ax_bar.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(0.6)

        if n_specs == 0:
            ax_empty = fig.add_subplot(col_gs[1, 0])
            ax_empty.text(
                0.5,
                0.5,
                "no specialist",
                ha="center",
                va="center",
                fontsize=ps.SMALL_PT,
                color="gray",
            )
            ax_empty.axis("off")

    fig.suptitle(
        f"All specialists responding to one digit-{true_label} sample", y=0.995
    )
    fig.text(
        0.5,
        0.004,
        "Columns: specialists by home digit; titles give capsule ID and ablation importance "
        "for that digit.\n"
        "Border: green = routes this sample to its true digit, red = routes it elsewhere. "
        "Bars: coupling to digits 0-9\n(green = highest, red = true digit if different).",
        ha="center",
        va="bottom",
        fontsize=ps.SMALL_PT,
        linespacing=1.25,
    )
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Single-sample Specialist tracking: every taxonomy Specialist responding to one image"
    )
    add_figure_args(parser)
    parser.add_argument(
        "--digit", type=int, required=True, help="Digit of the test image"
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Index of the image among the test images of that digit",
    )
    parser.add_argument(
        "--top_k_per_digit",
        type=int,
        default=None,
        help="Limit to the top K Specialists per home digit (by importance)",
    )
    args = parser.parse_args()
    output_dir, device, model, classification, importance_matrix, test_dataset = (
        load_figure_inputs(parser, args)
    )

    taxonomy_specialists = classification.get("specialists", [])
    print(f"\n{len(taxonomy_specialists)} taxonomy Specialists")

    # Specialists by home digit (highest importance), most important first
    specialists_by_digit = {d: [] for d in range(10)}
    for cap_idx in taxonomy_specialists:
        home = int(np.argmax(importance_matrix[cap_idx]))
        specialists_by_digit[home].append((cap_idx, importance_matrix[cap_idx, home]))
    for digit, specs in specialists_by_digit.items():
        specs.sort(key=lambda x: x[1], reverse=True)
        if args.top_k_per_digit:
            del specs[args.top_k_per_digit :]
        print(f"  Digit {digit}: {[c for c, _ in specs]}")
    n_tracked = sum(len(specs) for specs in specialists_by_digit.values())
    print(f"\nTotal specialists: {n_tracked}")

    save_path = os.path.join(
        output_dir,
        f"single_sample_d{args.digit}_s{args.sample_idx}_{n_tracked}caps.png",
    )
    guard_overwrite(save_path, args.force)

    digit_indices = class_indices(test_dataset)[args.digit]
    if args.sample_idx >= len(digit_indices):
        raise SystemExit(
            f"Error: Only {len(digit_indices)} samples of digit {args.digit}"
        )

    dataset_idx = digit_indices[args.sample_idx]
    image, true_label = test_dataset[dataset_idx]

    print(f"Sample: digit {args.digit}, index {args.sample_idx}")

    fig = create_single_sample_tracking_figure(
        model=model,
        image=image,
        true_label=true_label,
        specialists_by_digit=specialists_by_digit,
        device=device,
    )
    ps.save(fig, save_path)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
