"""Single-sample explanation figure: why the network predicted one test digit.

Usage: python figures/sample_explanation.py --digit 3 --sample_idx 4
"""

import argparse
import os
from typing import List, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

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

BAND_COLORS = {"HIGH": "darkgreen", "MED": "darkorange", "LOW": "dimgray"}


def get_network_prediction(
    model: CapsuleNetwork, image: torch.Tensor, device: torch.device
) -> Tuple[int, float, np.ndarray]:
    """Get network's prediction, confidence, and all digit scores."""
    image = image.to(device)

    if image.dim() == 3:
        image = image.unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        v = output[0]  # Digit capsule outputs: [batch, num_classes, dim_caps]
        norms = torch.norm(v, dim=2)[0]  # [num_classes]
        probs = F.softmax(norms, dim=0).cpu().numpy()
        predicted = int(torch.argmax(norms).item())
        confidence = float(probs[predicted])

    return predicted, confidence, probs


def get_top_confusable_digits(
    class_probs: np.ndarray, predicted: int, max_confusable: int
) -> List[int]:
    """Alternative digits ranked by score (the softmax over digit-capsule lengths), at
    most max_confusable of them with score > 0.01."""
    scores = class_probs.copy()
    scores[predicted] = 0

    sorted_indices = np.argsort(scores)[::-1]
    confusable = [
        int(idx) for idx in sorted_indices[:max_confusable] if scores[idx] > 0.01
    ]

    return confusable


def create_sample_explanation_figure(
    model: CapsuleNetwork,
    image: torch.Tensor,
    true_label: int,
    predicted: int,
    importance_matrix: np.ndarray,
    supporting_capsules: List[Tuple[int, str]],  # (idx, label)
    hub_capsules: List[Tuple[int, str]],
    alternative_capsules: List[Tuple[int, str]],
    device: torch.device,
) -> plt.Figure:
    """The input on the left; on the right two rows of capsule panels (attribution
    heatmap above, coupling distribution below): the capsules most specific to the
    predicted digit, then the Hubs and the capsules most specific to each alternative
    digit. The border colour says which group a capsule belongs to."""
    ps.apply()

    groups = {"supporting": "green", "hub": "blue", "alternative": "orange"}
    row1 = [(i, l, groups["supporting"]) for i, l in supporting_capsules]
    row2 = [(i, l, groups["hub"]) for i, l in hub_capsules] + [
        (i, l, groups["alternative"]) for i, l in alternative_capsules
    ]
    ncols = max(len(row1), len(row2), 1)

    fig = ps.figure(height_in=3.6, layout=None)
    outer_gs = gridspec.GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[0.8, ncols],
        wspace=0.08,
        left=0.012,
        right=0.975,
        top=0.855,
        bottom=0.175,
    )

    # LEFT: the input image
    ax_input = fig.add_subplot(outer_gs[0, 0])
    ax_input.imshow(image.squeeze().numpy(), cmap="gray")
    border_color = "green" if predicted == true_label else "red"
    for spine in ax_input.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(1.5)
    ax_input.set_xticks([])
    ax_input.set_yticks([])
    ax_input.set_xlabel(f"Input\n(true: {true_label})", fontweight="bold")

    # RIGHT: the evidence rows
    evidence_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer_gs[0, 1], hspace=0.42
    )

    def draw_capsule_row(parent_gs, capsules: List[Tuple[int, str, str]]):
        row_gs = gridspec.GridSpecFromSubplotSpec(
            2,
            ncols,
            subplot_spec=parent_gs,
            hspace=0.12,
            wspace=0.5,
            height_ratios=[1, 0.5],
        )
        for col_idx, (cap_idx, cap_label, border) in enumerate(capsules):
            routing_info = capsule_routing(model, image.unsqueeze(0), cap_idx, device)
            coupling = routing_info["coupling_coeffs"]
            top_digit = routing_info["top_digit"]

            imp_level = importance_band(
                importance_matrix[:, predicted], importance_matrix[cap_idx, predicted]
            )

            # Top: spatial attribution heatmap over the input
            ax_heat = fig.add_subplot(row_gs[0, col_idx])
            heatmap = spatial_attribution(model, image, cap_idx, device)
            ax_heat.imshow(image.squeeze().numpy(), cmap="gray", alpha=0.4)
            ax_heat.imshow(heatmap, cmap="hot", alpha=0.7, vmin=0, vmax=1)
            short_label = cap_label.replace(" (", " ").replace(")", "")
            ax_heat.set_title(
                f"#{cap_idx} · {short_label}\nImp: {imp_level}",
                fontsize=ps.SMALL_PT,
                fontweight="normal",
                color=BAND_COLORS[imp_level],
                pad=2,
            )
            for spine in ax_heat.spines.values():
                spine.set_edgecolor(border)
                spine.set_linewidth(1.2)
            ax_heat.set_xticks([])
            ax_heat.set_yticks([])

            # Bottom: coupling distribution over the ten digit capsules
            ax_bar = fig.add_subplot(row_gs[1, col_idx])
            colors = ["lightgray"] * 10
            colors[top_digit] = "green"
            if true_label != top_digit:
                colors[true_label] = "red"
            ax_bar.bar(
                np.arange(10), coupling, color=colors, edgecolor="black", linewidth=0.3
            )
            ax_bar.set_xlim(-0.6, 9.6)
            ax_bar.set_ylim(0, 1)
            # Every digit is ticked; only the even ones are labelled, so the labels
            # stay legible at 7 pt in the narrower columns.
            ax_bar.set_xticks(range(10))
            ax_bar.set_xticklabels(
                [str(d) if d % 2 == 0 else "" for d in range(10)], fontsize=ps.SMALL_PT
            )
            ax_bar.set_yticks([0, 1])
            ax_bar.tick_params(axis="both", length=1.5, pad=1)
            for spine in ax_bar.spines.values():
                spine.set_edgecolor(border)
                spine.set_linewidth(0.8)

    draw_capsule_row(evidence_gs[0], row1)
    draw_capsule_row(evidence_gs[1], row2)

    outcome = (
        "correct" if predicted == true_label else f"wrong, true digit {true_label}"
    )
    fig.suptitle(
        f"Prediction: digit {predicted} ({outcome})",
        fontsize=ps.BASE_PT,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.008,
        f"Border: green = most digit-{predicted}-specific (any role); blue = Hub; "
        "orange = most specific for an alternative digit.\n"
        "Bars: coupling to each digit capsule; green = highest, red = true digit "
        "if different.\n"
        "Imp: ablation-importance band for the predicted digit.",
        ha="center",
        va="bottom",
        fontsize=ps.SMALL_PT,
        linespacing=1.25,
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description="Single-sample capsule explanation")
    add_figure_args(parser)
    parser.add_argument(
        "--digit", type=int, required=True, help="Target digit to explain (0-9)"
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Index of the sample within that digit class",
    )
    parser.add_argument(
        "--specialists",
        type=int,
        default=4,
        help="Number of capsules most specific to the predicted digit",
    )
    parser.add_argument(
        "--hubs", type=int, default=2, help="Number of top hubs to include"
    )
    parser.add_argument(
        "--max_confusable",
        type=int,
        default=3,
        help="Maximum number of confusable digits to show alternatives for",
    )
    parser.add_argument(
        "--specialists_per_confusable",
        type=int,
        default=1,
        help="Number of most-specific capsules per confusable digit",
    )
    args = parser.parse_args()
    output_dir, device, model, _, importance_matrix, test_dataset = load_figure_inputs(
        parser, args
    )
    mean_importance = np.mean(importance_matrix, axis=1)

    print(f"Finding sample {args.sample_idx} of digit {args.digit}")
    digit_indices = class_indices(test_dataset)[args.digit]

    if args.sample_idx >= len(digit_indices):
        raise SystemExit(
            f"Error: Only {len(digit_indices)} samples of digit {args.digit} available"
        )

    dataset_idx = digit_indices[args.sample_idx]
    image, true_label = test_dataset[dataset_idx]

    predicted, confidence, probs = get_network_prediction(model, image, device)
    print(f"  True label: {true_label}")
    print(f"  Predicted: {predicted} (confidence: {confidence*100:.1f}%)")

    # Supporting: capsules most specific to the predicted digit (any role)
    print(f"\nTop {args.specialists} capsules most specific to digit {predicted}:")
    supporting_capsules = []
    for idx, spec_ratio in most_specific(
        importance_matrix, predicted, args.specialists
    ):
        supporting_capsules.append((idx, f"D{predicted} (×{spec_ratio:.1f})"))
        print(
            f"  #{idx}: ablation={importance_matrix[idx, predicted]:.3f}, spec={spec_ratio:.1f}x"
        )

    # Hubs: capsules critical for the most digits
    print(f"\nTop {args.hubs} hubs:")
    high_count = threshold_counts(importance_matrix.T)[0]
    hub_candidates = [
        (i, mean_importance[i], int(high_count[i])) for i in range(len(mean_importance))
    ]
    hub_candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)

    hub_capsules = []
    used_caps = set(c[0] for c in supporting_capsules)
    for hub_idx, mean_imp, num_digits in hub_candidates:
        if len(hub_capsules) >= args.hubs:
            break
        if hub_idx not in used_caps:
            label = f"Hub ({num_digits}D)"
            hub_capsules.append((hub_idx, label))
            used_caps.add(hub_idx)
            print(f"  #{hub_idx}: {num_digits} digits, mean={mean_imp:.3f}")

    # Alternatives: confusable digits and their most specific capsules
    print(f"\nUp to {args.max_confusable} confusable digits:")
    confusable_digits = get_top_confusable_digits(probs, predicted, args.max_confusable)
    print(f"  Confusable digits: {confusable_digits}")

    alternative_capsules = []
    for conf_digit in confusable_digits:
        for idx, spec_ratio in most_specific(
            importance_matrix, conf_digit, args.specialists_per_confusable, used_caps
        ):
            alternative_capsules.append((idx, f"D{conf_digit} (×{spec_ratio:.1f})"))
            used_caps.add(idx)
            print(f"  #{idx}: most specific for D{conf_digit}")

    fig = create_sample_explanation_figure(
        model=model,
        image=image,
        true_label=true_label,
        predicted=predicted,
        importance_matrix=importance_matrix,
        supporting_capsules=supporting_capsules,
        hub_capsules=hub_capsules,
        alternative_capsules=alternative_capsules,
        device=device,
    )

    save_path = os.path.join(
        output_dir, f"explanation_d{args.digit}_s{args.sample_idx}_pred{predicted}.png"
    )
    guard_overwrite(save_path, args.force)
    ps.save(fig, save_path)
    print(f"\nSaved: {save_path}")
    print(f"Supporting capsules: {len(supporting_capsules)}")
    print(f"Hub capsules: {len(hub_capsules)}")
    print(
        f"Alternative capsules: {len(alternative_capsules)} (from digits {confusable_digits})"
    )


if __name__ == "__main__":
    main()
