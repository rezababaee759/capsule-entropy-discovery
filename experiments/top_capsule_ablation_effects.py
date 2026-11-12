#!/usr/bin/env python3
"""Per digit: the ablation effect of the top capsule (argmax of the importance
matrix) against 30 non-critical control capsules on the top_capsule_evaluation
slice: a one-sided rank test (BH and Holm across digits) and a bootstrap CI on the
excess drop.

    python experiments/top_capsule_ablation_effects.py [--output PATH]
"""

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from statsmodels.stats.multitest import multipletests

from capsnet.data import describe_split, slice_for, split_indices, test_set
from capsnet.io import get_device, load_capsule_model
from capsnet.provenance import attach

IMPORTANCE_MATRIX = "results/data/mnist/importance_matrix.npy"


def load_top_capsules(path=IMPORTANCE_MATRIX):
    """argmax importance per digit, from the [10, N_capsules] matrix."""
    m = np.load(path)
    if m.ndim != 2 or m.shape[0] != 10:
        raise ValueError(
            f"{path}: expected a [10, n_capsules] importance matrix, got {m.shape}"
        )
    return {d: int(m[d].argmax()) for d in range(10)}


N_IMAGES = 50
# Disjoint from taxonomy_calibration, where the top capsule and controls are chosen.
EVAL_SPLIT_PURPOSE = "top_capsule_evaluation"
N_RANDOM_CAPSULES = 30
N_BOOTSTRAP = 10000
SEED = 42


@torch.no_grad()
def ablation_drops_for_capsule(model, images, digit, capsule_idx, device):
    """Return array of confidence drops (len = len(images)) for one capsule."""
    drops = []
    for img in images:
        img = img.unsqueeze(0).to(device)
        conv1_out = F.relu(model.conv1(img))
        primary_out = model.primary(conv1_out)
        digit_caps, _, _, _ = model.digits(primary_out)
        orig_conf = torch.norm(digit_caps[0], dim=-1)[digit].item()

        primary_ablated = primary_out.clone()
        primary_ablated[0, capsule_idx, :] = 0.0
        digit_caps_abl, _, _, _ = model.digits(primary_ablated)
        abl_conf = torch.norm(digit_caps_abl[0], dim=-1)[digit].item()
        drops.append(orig_conf - abl_conf)
    return np.array(drops)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--output", default="results/data/mnist/top_capsule_ablation_effects.json"
    )
    args = ap.parse_args()
    device = get_device()
    print(f"Device: {device}")
    rng = np.random.default_rng(SEED)

    model = load_capsule_model(
        "models/capsule_mnist_best_dynamic.pt", "mnist", device=device
    )
    with open("results/data/mnist/hub_taxonomy.json") as f:
        taxonomy = json.load(f)
    non_critical = taxonomy["non_critical"]
    print(f"Non-critical population: {len(non_critical)} capsules")

    top_capsules = load_top_capsules()
    print(f"Top capsules (from {IMPORTANCE_MATRIX}): {top_capsules}")

    ds = test_set("mnist")
    eval_indices = split_indices(ds, "mnist", EVAL_SPLIT_PURPOSE, N_IMAGES)

    random_caps = sorted(
        rng.choice(non_critical, size=N_RANDOM_CAPSULES, replace=False).tolist()
    )
    print(f"Random baseline capsules ({N_RANDOM_CAPSULES}): {random_caps}")

    results = {}
    for digit in range(10):
        top_cap = top_capsules[digit]
        images = [ds[i][0] for i in eval_indices[digit].tolist()]

        top_drops = ablation_drops_for_capsule(model, images, digit, top_cap, device)

        random_capsule_means = []
        for rc in random_caps:
            rc_drops = ablation_drops_for_capsule(model, images, digit, rc, device)
            random_capsule_means.append(rc_drops.mean())
        random_capsule_means = np.array(random_capsule_means)

        # Impact-ratio CI, resampling the control capsules.
        boot_ratios = []
        top_mean = top_drops.mean()
        for _ in range(N_BOOTSTRAP):
            resample = rng.choice(
                random_capsule_means, size=len(random_capsule_means), replace=True
            )
            denom = resample.mean()
            if denom > 1e-12:
                boot_ratios.append(top_mean / denom)
        boot_ratios = np.array(boot_ratios)
        ci_low, ci_high = np.percentile(boot_ratios, [2.5, 97.5])

        random_sd = random_capsule_means.std(ddof=1)
        effect_size_z = (
            float((top_mean - random_capsule_means.mean()) / random_sd)
            if random_sd > 1e-12
            else float("nan")
        )

        n_exceeded = int((top_mean > random_capsule_means).sum())
        n_random = int(len(random_capsule_means))
        # Exact one-sided randomisation p: (#{random >= top} + 1) / (n_random + 1).
        rank_p = float(
            (int((random_capsule_means >= top_mean).sum()) + 1) / (n_random + 1)
        )
        # Bootstrap CI on the excess (top minus control mean), resampling both the
        # top capsule's images and the control capsules.
        boot_diffs = np.array(
            [
                rng.choice(top_drops, size=len(top_drops), replace=True).mean()
                - rng.choice(random_capsule_means, size=n_random, replace=True).mean()
                for _ in range(N_BOOTSTRAP)
            ]
        )
        diff_lo, diff_hi = np.percentile(boot_diffs, [2.5, 97.5])

        # The same, holding the top capsule's mean fixed.
        boot_diffs_baseline_only = np.array(
            [
                top_mean
                - rng.choice(random_capsule_means, size=n_random, replace=True).mean()
                for _ in range(N_BOOTSTRAP)
            ]
        )
        base_lo, base_hi = np.percentile(boot_diffs_baseline_only, [2.5, 97.5])

        results[digit] = {
            "top_capsule": top_cap,
            "top_conf_drop_pct_mean": float(top_mean * 100),
            "top_conf_drop_pct_sd": float(top_drops.std(ddof=1) * 100),
            "n_images": N_IMAGES,
            "random_baseline_conf_drop_pct_mean": float(
                random_capsule_means.mean() * 100
            ),
            "random_baseline_conf_drop_pct_sd": float(
                random_capsule_means.std(ddof=1) * 100
            ),
            "n_random_capsules": N_RANDOM_CAPSULES,
            "exceeds_n_random_capsules": n_exceeded,
            "n_random_capsules_compared": n_random,
            "rank_p_one_sided": rank_p,
            "excess_conf_drop_pct_mean": float(
                (top_mean - random_capsule_means.mean()) * 100
            ),
            "excess_conf_drop_pct_ci95": [float(diff_lo * 100), float(diff_hi * 100)],
            "excess_conf_drop_pct_ci95_sources": "top-capsule images + control capsules",
            "excess_conf_drop_pct_ci95_baseline_only": [
                float(base_lo * 100),
                float(base_hi * 100),
            ],
            "excess_conf_drop_pct_ci95_baseline_only_note": (
                "Resamples only the control capsules, holding the top capsule mean "
                "fixed, so it is an interval on the baseline rather than on the "
                "excess. Use excess_conf_drop_pct_ci95."
            ),
            "effect_size_z": effect_size_z,
            "effect_size_z_note": (
                "Denominator is the SD of N_IMAGES-averaged means, so this scales as "
                "sqrt(N_IMAGES). Use rank_p_one_sided and excess_conf_drop_pct_ci95."
            ),
            "impact_ratio_mean": float(top_mean / random_capsule_means.mean()),
            "impact_ratio_ci95": [float(ci_low), float(ci_high)],
        }
        print(
            f"Digit {digit}: top_drop={top_mean*100:.3f}% +/- {top_drops.std(ddof=1)*100:.3f}%  "
            f"random_drop={random_capsule_means.mean()*100:.4f}% +/- {random_capsule_means.std(ddof=1)*100:.4f}%  "
            f"exceeds={n_exceeded}/{n_random}  rank_p={rank_p:.4f}"
        )

    # Multiplicity over the per-digit rank tests; the smallest attainable p is
    # 1/(N_RANDOM_CAPSULES + 1).
    digits_sorted = sorted(results)
    rank_ps = [results[d]["rank_p_one_sided"] for d in digits_sorted]
    p_floor = 1.0 / (N_RANDOM_CAPSULES + 1)
    bh_rej, bh_adj, _, _ = multipletests(rank_ps, alpha=0.05, method="fdr_bh")
    holm_rej, holm_adj, _, _ = multipletests(rank_ps, alpha=0.05, method="holm")
    for i, d in enumerate(digits_sorted):
        results[d]["rank_p_bh_adjusted"] = float(bh_adj[i])
        results[d]["rank_p_significant_bh"] = bool(bh_rej[i])
        results[d]["rank_p_holm_adjusted"] = float(holm_adj[i])
        results[d]["rank_p_significant_holm"] = bool(holm_rej[i])
        results[d]["rank_p_attainable_floor"] = float(p_floor)

    n_controls_for_fwer = int(np.ceil(len(digits_sorted) / 0.05)) - 1

    output = {
        "description": "Per-digit confidence-drop uncertainty and a matched random-capsule "
        "baseline: %d non-critical capsules, each ablated across the same images as the "
        "top capsule; mean and SD are taken across capsules (capsule-level replicates), "
        "not over pooled images." % N_RANDOM_CAPSULES,
        "selection_vs_evaluation": {
            "selection_source": IMPORTANCE_MATRIX,
            "selection_split_purpose": "taxonomy_calibration",
            "evaluation_split_purpose": EVAL_SPLIT_PURPOSE,
            "evaluation_slice": list(slice_for("mnist", EVAL_SPLIT_PURPOSE, N_IMAGES)),
            "split_contract": describe_split("mnist"),
            "note": (
                "The top capsule is the argmax of the importance matrix, and the "
                "controls are drawn from a category thresholded on that same "
                "matrix; both are properties of the taxonomy_calibration slice. "
                "The ablation effect, the rank test and the bootstrap interval are "
                "measured on a disjoint slice, so the rank test compares the "
                "selected capsule against its controls on images that took no part "
                "in the selection. capsnet/data.py asserts the two ranges are "
                "disjoint at import."
            ),
        },
        "seed": SEED,
        "n_images_per_digit": N_IMAGES,
        "n_random_capsules": N_RANDOM_CAPSULES,
        "random_baseline_capsule_ids": random_caps,
        "multiplicity": {
            "family_size": len(digits_sorted),
            "family": "one one-sided rank test per digit",
            "primary": "Benjamini-Hochberg, q=0.05",
            "secondary": "Holm-Bonferroni, alpha=0.05",
            "implementation": "statsmodels.stats.multitest.multipletests",
            "n_significant_bh": int(bh_rej.sum()),
            "n_significant_holm": int(holm_rej.sum()),
            "attainable_one_sided_p_floor": float(p_floor),
            "floor_note": (
                "The smallest one-sided rank p attainable with %d controls is "
                "1/%d = %.5f. A digit at that floor carries no resolution beyond "
                '"the top capsule exceeded all %d controls". '
                "Surviving Bonferroni over %d tests would need at least %d "
                "controls per digit."
                % (
                    N_RANDOM_CAPSULES,
                    N_RANDOM_CAPSULES + 1,
                    p_floor,
                    N_RANDOM_CAPSULES,
                    len(digits_sorted),
                    n_controls_for_fwer,
                )
            ),
        },
        "per_digit": results,
    }
    attach(
        output,
        script="experiments/top_capsule_ablation_effects.py",
        inputs=[
            IMPORTANCE_MATRIX,
            "results/data/mnist/hub_taxonomy.json",
            "models/capsule_mnist_best_dynamic.pt",
        ],
        seeds={
            "SEED": SEED,
            "note": "One default_rng(SEED) stream is shared by the control-capsule "
            "draw and all three bootstrap loops, in that order.",
        },
        sample_counts={
            "n_images_per_digit": N_IMAGES,
            "n_random_capsules": N_RANDOM_CAPSULES,
            "n_bootstrap": N_BOOTSTRAP,
            "n_digits": len(results),
            "non_critical_pool_size": len(non_critical),
        },
        method=(
            "Per digit: the top capsule is the argmax of the importance matrix "
            "(built on the taxonomy_calibration slice); its ablation effect, an "
            "exact one-sided randomisation rank test against the non-critical "
            "controls, and a percentile bootstrap CI on the excess resampling BOTH "
            "images and control capsules, are all measured on the disjoint "
            "top_capsule_evaluation slice. The per-digit rank tests are corrected with "
            "Benjamini-Hochberg and Holm-Bonferroni."
        ),
        packages=["numpy", "statsmodels"],
    )

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
