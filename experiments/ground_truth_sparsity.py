"""Ground-truth ablation sparsity: per routing variant, the Gini coefficient of the mean
true-class confidence drop per primary capsule and the share of capsules carrying 50%
and 80% of it, on the taxonomy_calibration split.

    python experiments/ground_truth_sparsity.py --dataset mnist [--output PATH]
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from capsnet.ablation import single_capsule_ablation_drops
from capsnet.data import (
    describe_split,
    describe_transform,
    slice_for,
    split_indices,
    test_set,
)
from capsnet.io import get_device, load_capsule_model, routing_from_path
from capsnet.stats import compute_gini, top_share_count

NUM_SAMPLES_PER_CLASS = 50
SPLIT_PURPOSE = "taxonomy_calibration"
DEVICE = get_device()
ARMS = [
    ("Dynamic Routing", "models/capsule_{}_best_dynamic.pt"),
    ("Entropy Routing (VEG)", "models/capsule_{}_best_entropy.pt"),
]


def calibration_loader(dataset):
    ds = test_set(dataset)
    per_class = split_indices(ds, dataset, SPLIT_PURPOSE, NUM_SAMPLES_PER_CLASS)
    indices = [i for c in range(10) for i in per_class[c].tolist()]
    return DataLoader(Subset(ds, indices), batch_size=1, shuffle=False)


def sparsity(model, dataset, name):
    """Mean true-class confidence drop per capsule over the calibration images, and its
    concentration."""
    num_capsules = model.digits.num_input_caps
    importance_scores = np.zeros(num_capsules)
    total_samples = 0
    for img, label in tqdm(calibration_loader(dataset), desc=f"Ablating {name}"):
        img = img.to(DEVICE)
        label_idx = int(label.item())
        with torch.no_grad():
            output, _, _, _ = model(img)
            baseline_conf = torch.norm(output, dim=-1)[0, label_idx].item()
            # Skips low true-class confidence; correctness is not checked.
            if baseline_conf < 0.1:
                continue
            total_samples += 1
            primary_out = model.primary(F.relu(model.conv1(img)))
            importance_scores += single_capsule_ablation_drops(
                model, primary_out, label_idx, baseline_conf
            )
    if total_samples == 0:
        raise SystemExit(f"{name}: no valid samples processed.")

    # Clip negative means to 0: Gini and top-k shares need non-negative values.
    avg_importance = np.maximum(importance_scores / total_samples, 0)
    gini = compute_gini(avg_importance)
    sorted_imp = np.sort(avg_importance)[::-1]
    k_50 = top_share_count(sorted_imp, 0.5)
    k_80 = top_share_count(sorted_imp, 0.8)
    return {
        "gini": float(gini),
        "percent_50": float((k_50 / num_capsules) * 100),
        "percent_80": float((k_80 / num_capsules) * 100),
        "k_50": int(k_50),
        "k_80": int(k_80),
        "total_samples": int(total_samples),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", required=True, choices=["mnist", "cifar10"])
    ap.add_argument(
        "--output", help="default: results/data/{dataset}/ground_truth_ablation.json"
    )
    args = ap.parse_args()
    dataset = args.dataset
    output = args.output or f"results/data/{dataset}/ground_truth_ablation.json"
    print(f"Ground-truth ablation sparsity, {dataset}, on {DEVICE}")

    results = {}
    for name, pattern in ARMS:
        path = pattern.format(dataset)
        if not os.path.exists(path):
            raise SystemExit(f"{name}: model file {path} not found.")
        model = load_capsule_model(path, dataset, routing_from_path(path), DEVICE)
        res = sparsity(model, dataset, name)
        results[name] = res
        print(
            f"{name}: Gini = {res['gini']:.4f}, "
            f"Top {res['k_80']}/{model.digits.num_input_caps} capsules carry 80% "
            f"({res['percent_80']:.2f}%)"
        )

    start, stop = slice_for(dataset, SPLIT_PURPOSE, NUM_SAMPLES_PER_CLASS)
    envelope = {
        "dataset": dataset,
        "method": "ground_truth_ablation",
        "num_capsules": model.digits.num_input_caps,
        "samples_per_class": NUM_SAMPLES_PER_CLASS,
        "split_purpose": SPLIT_PURPOSE,
        "split_positions_per_class": [start, stop],
        "split_contract": describe_split(dataset),
        "input_transform": describe_transform(dataset),
        "results": results,
    }
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(envelope, f, indent=2)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
