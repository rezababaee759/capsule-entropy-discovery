"""Capsule tracking figure: the same capsules across several samples of chosen digits.

The tracked capsules are the top taxonomy Hubs and, per requested digit, the most
digit-specific capsules (optionally random ones too).

Usage: python figures/capsule_tracking.py --confusable_digits 0 8 --force
"""

import argparse
import os
from typing import List

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch

from capsnet import paper_style as ps
from capsnet.data import class_indices
from capsnet.explain import (
    add_figure_args,
    capsule_routing,
    importance_band,
    load_figure_inputs,
    most_specific,
    spatial_attribution,
)
from capsnet.io import guard_overwrite
from capsnet.model import CapsuleNetwork
from capsnet.taxonomy import threshold_counts

BAND_COLORS = {"HIGH": "darkred", "MED": "darkorange", "LOW": "dimgray"}


def create_capsule_tracking_figure(
    model: CapsuleNetwork,
    dataset,
    tracked_capsules: List[int],
    importance_matrix: np.ndarray,
    target_digits: List[int],
    n_samples_per_digit: int,
    device: torch.device,
    capsule_types: dict = None,
) -> plt.Figure:
    """One row per sample. Column 0 is the input; each further column is one tracked
    capsule, labelled once at the top, with its attribution heatmap above its coupling
    distribution."""
    ps.apply()

    n_capsules = len(tracked_capsules)
    n_total_samples = len(target_digits) * n_samples_per_digit

    fig = ps.figure(
        height_in=min(0.84 * ps.TEXT_HEIGHT_IN, 0.55 + 1.0 * n_total_samples),
        layout=None,
    )
    height_ratios = []
    for _ in range(n_total_samples):
        height_ratios.extend([1, 0.55, 0.32])  # heatmap, coupling, gap
    gs = gridspec.GridSpec(
        n_total_samples * 3,
        1 + n_capsules,
        figure=fig,
        hspace=0.08,
        wspace=0.35,
        height_ratios=height_ratios,
        width_ratios=[0.8] + [1] * n_capsules,
        left=0.07,
        right=0.965,
        top=0.905,
        bottom=0.095,
    )

    sample_row_idx = 0
    for digit in target_digits:
        indices = class_indices(dataset)[digit]
        selected_indices = np.random.choice(
            indices, min(n_samples_per_digit, len(indices)), replace=False
        )

        for sample_idx, dataset_idx in enumerate(selected_indices):
            img, label = dataset[dataset_idx]
            heatmap_row = sample_row_idx * 3
            routing_row = sample_row_idx * 3 + 1
            first_row = sample_row_idx == 0
            last_row = sample_row_idx == n_total_samples - 1

            # Column 0: the input image
            ax_img = fig.add_subplot(gs[heatmap_row : routing_row + 1, 0])
            ax_img.imshow(img.squeeze().numpy(), cmap="gray")
            ax_img.set_ylabel(
                f"Digit {digit}\nsample {sample_idx + 1}", fontweight="bold"
            )
            for spine in ax_img.spines.values():
                spine.set_edgecolor("blue")
                spine.set_linewidth(1.0)
            ax_img.set_xticks([])
            ax_img.set_yticks([])

            for cap_col_idx, cap_idx in enumerate(tracked_capsules):
                routing_info = capsule_routing(model, img.unsqueeze(0), cap_idx, device)
                coupling = routing_info["coupling_coeffs"]
                entropy = routing_info["entropy"]
                top_digit = routing_info["top_digit"]
                importance = importance_matrix[cap_idx, digit]

                role_label = importance_band(importance_matrix[:, digit], importance)

                # Heatmap: where the capsule looks
                ax_heat = fig.add_subplot(gs[heatmap_row, 1 + cap_col_idx])
                heatmap = spatial_attribution(model, img.unsqueeze(0), cap_idx, device)
                ax_heat.imshow(img.squeeze().numpy(), cmap="gray", alpha=0.5)
                ax_heat.imshow(heatmap, cmap="hot", alpha=0.5)
                # Significant figures: importances can be too small for fixed decimals.
                cell_title = f"I={importance:.2g} · {role_label}"
                if first_row:
                    header = (capsule_types or {}).get(cap_idx, f"Cap #{cap_idx}")
                    header = (
                        header.replace(" (", "\n").replace(")", "")
                        if "-specific" in header
                        else header
                    )
                    ax_heat.text(
                        0.5,
                        1.38,
                        f"#{cap_idx} · {header}",
                        transform=ax_heat.transAxes,
                        ha="center",
                        va="bottom",
                        fontsize=ps.BASE_PT,
                        fontweight="bold",
                    )
                ax_heat.set_title(
                    cell_title,
                    fontsize=ps.SMALL_PT,
                    color=BAND_COLORS[role_label],
                    pad=2,
                )
                for spine in ax_heat.spines.values():
                    spine.set_edgecolor("purple")
                    spine.set_linewidth(0.8)
                ax_heat.set_xticks([])
                ax_heat.set_yticks([])

                # Coupling distribution: where the capsule routes
                ax_coupling = fig.add_subplot(gs[routing_row, 1 + cap_col_idx])
                colors = [
                    "red" if d == digit else "green" if d == top_digit else "lightgray"
                    for d in range(10)
                ]
                ax_coupling.bar(
                    range(10),
                    coupling,
                    color=colors,
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=0.3,
                )
                ax_coupling.text(
                    0.98,
                    0.95,
                    f"H={entropy:.2f}",
                    transform=ax_coupling.transAxes,
                    fontsize=ps.SMALL_PT,
                    va="top",
                    ha="right",
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        linewidth=0,
                        facecolor=(
                            "yellow"
                            if entropy < 0.5
                            else "orange" if entropy < 1.5 else "salmon"
                        ),
                        alpha=0.8,
                    ),
                )
                if top_digit != digit:
                    ax_coupling.annotate(
                        "",
                        xy=(top_digit, coupling[top_digit]),
                        xytext=(top_digit, coupling[top_digit] + 0.4),
                        arrowprops=dict(arrowstyle="->", color="green", lw=0.8),
                    )
                ax_coupling.set_xticks(range(10))
                ax_coupling.set_xticklabels(
                    [str(d) for d in range(10)] if last_row else []
                )
                ax_coupling.set_ylim([0, 1])
                ax_coupling.set_yticks([0, 1])
                if cap_col_idx > 0:
                    ax_coupling.set_yticklabels([])
                ax_coupling.tick_params(length=1.5, pad=1)
                if last_row:
                    ax_coupling.set_xlabel(
                        "Digit capsule", fontsize=ps.SMALL_PT, labelpad=1
                    )
                for spine in ax_coupling.spines.values():
                    spine.set_edgecolor("darkgreen")
                    spine.set_linewidth(0.8)

            sample_row_idx += 1

    fig.suptitle(
        f"Tracking the same capsules across digits "
        f'{" and ".join(map(str, target_digits))}',
        y=0.995,
    )
    fig.text(
        0.5,
        0.004,
        "I = ground-truth ablation importance for the image's digit "
        "(confidence drop when the capsule is masked).\n"
        "Band: HIGH ≥ 95th, MED ≥ 75th percentile within that digit. "
        "H = routing entropy in nats (max ln 10 ≈ 2.30).\n"
        "Bars: coupling to each digit capsule; red = true digit, green = top-routed digit "
        "if different;\narrow = misrouting.",
        ha="center",
        va="bottom",
        fontsize=ps.SMALL_PT,
        linespacing=1.25,
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description="Capsule routing tracking figure")
    add_figure_args(parser)
    parser.add_argument(
        "--confusable_digits",
        type=int,
        nargs="+",
        default=[0, 6, 8, 9],
        help="Digits to track",
    )
    parser.add_argument(
        "--n_samples_per_digit",
        type=int,
        default=3,
        help="Number of examples per digit",
    )
    parser.add_argument(
        "--top_hubs",
        type=int,
        default=1,
        help="Number of top hubs to include (ranked by digit coverage, then mean importance)",
    )
    parser.add_argument(
        "--top_specialists",
        type=int,
        default=1,
        help="Number of most digit-specific capsules per digit, of any taxonomy role",
    )
    parser.add_argument(
        "--include_random",
        type=int,
        default=0,
        help="Number of random capsules to add",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    np.random.seed(args.seed)

    output_dir, device, model, classification, importance_matrix, test_dataset = (
        load_figure_inputs(parser, args)
    )
    save_path = os.path.join(
        output_dir, f"capsule_tracking_{'_'.join(map(str, args.confusable_digits))}.png"
    )
    guard_overwrite(save_path, args.force)

    print("\nCapsule taxonomy")

    primary_hubs = classification.get("primary_hubs", [])
    secondary_hubs = classification.get("secondary_hubs", [])
    specialists = classification.get("specialists", [])

    print(f"Primary Hubs: {len(primary_hubs)} capsules")
    print(f"Secondary Hubs: {len(secondary_hubs)} capsules")
    print(f"Specialists: {len(specialists)} capsules")

    mean_importance = np.mean(importance_matrix, axis=1)

    tracked_capsules_set = set()
    capsule_types = {}

    # Hub candidates are the taxonomy's Hubs, ranked by the number of digits for which
    # the capsule is high (taxonomy rule), then by mean importance.
    high_count = threshold_counts(importance_matrix.T)[0]
    hub_candidates = [
        (i, mean_importance[i], int(high_count[i]))
        for i in list(primary_hubs) + list(secondary_hubs)
    ]
    hub_candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)

    hubs_added = 0
    for hub_idx, mean_imp, num_digits in hub_candidates:
        if hubs_added >= args.top_hubs:
            break
        if hub_idx not in tracked_capsules_set:
            tracked_capsules_set.add(hub_idx)
            capsule_types[hub_idx] = f"Hub ({num_digits}D)"
            print(
                f"  Hub #{hub_idx}: affects {num_digits} digits, mean ablation={mean_imp:.3f}"
            )
            hubs_added += 1

    for digit in args.confusable_digits:
        for idx, specialization in most_specific(
            importance_matrix, digit, args.top_specialists, tracked_capsules_set
        ):
            tracked_capsules_set.add(idx)
            capsule_types[idx] = f"D{digit}-specific (×{specialization:.1f})"
            print(
                f"  Most digit-specific #{idx} for digit {digit}: specialization={specialization:.1f}x"
            )

    if args.include_random > 0:
        n_capsules = importance_matrix.shape[0]
        available_capsules = set(range(n_capsules)) - tracked_capsules_set
        random_capsules = np.random.choice(
            list(available_capsules),
            min(args.include_random, len(available_capsules)),
            replace=False,
        )
        for rand_cap in random_capsules:
            tracked_capsules_set.add(rand_cap)
            capsule_types[rand_cap] = f"Random #{rand_cap}"
            print(
                f"  Random #{rand_cap}: mean ablation={mean_importance[rand_cap]:.3f}"
            )

    tracked_capsules = sorted(list(tracked_capsules_set))
    n_tracked = len(tracked_capsules)

    print(f"\nTracked capsules ({n_tracked} total): {tracked_capsules}")
    for cap_idx in tracked_capsules:
        print(f"  #{cap_idx}: {capsule_types.get(cap_idx, 'Unknown')}")

    fig = create_capsule_tracking_figure(
        model,
        test_dataset,
        tracked_capsules,
        importance_matrix,
        args.confusable_digits,
        args.n_samples_per_digit,
        device,
        capsule_types=capsule_types,
    )
    ps.save(fig, save_path)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
