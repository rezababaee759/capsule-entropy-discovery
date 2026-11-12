#!/usr/bin/env python3
"""Validate the ablation surrogate on MNIST test images it was not trained on (the
surrogate_heldout split): Pearson, Spearman and Precision@50 of its predicted
importance against the ground-truth single-capsule ablation effects.

Usage:
    python experiments/held_out_validation.py \
        --capsule_model models/capsule_mnist_best_dynamic.pt \
        --predictor_model models/ablation_predictor.pt \
        --output results/data/mnist/held_out_validation.json \
        --images_per_digit 50
"""

import os
import json
import argparse

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from capsnet.data import slice_for, split_indices, test_set
from capsnet.io import (
    get_device,
    load_capsule_model,
    predictor_from_checkpoint,
    resolve_output,
)
from capsnet.surrogate import predict_and_ablate


def precision_at_k(pred, target, k=50):
    pred_top = set(np.argsort(pred)[-k:])
    targ_top = set(np.argsort(target)[-k:])
    return len(pred_top & targ_top) / k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--capsule_model", default="models/capsule_mnist_best_dynamic.pt"
    )
    parser.add_argument("--predictor_model", default="models/ablation_predictor.pt")
    parser.add_argument(
        "--output", help="default: results/data/mnist/held_out_validation.json"
    )
    parser.add_argument("--images_per_digit", type=int, default=50)
    parser.add_argument(
        "--force",
        action="store_true",
        help="write the canonical output from --capsule_model or --predictor_model",
    )
    args = parser.parse_args()
    output_path = resolve_output(
        parser,
        args,
        "results/data/mnist/held_out_validation.json",
        ["capsule_model", "predictor_model"],
    )

    device = get_device()
    print(f"Device: {device}")

    capsule_model = load_capsule_model(args.capsule_model, "mnist", device=device)
    ckpt = torch.load(args.predictor_model, map_location=device, weights_only=False)
    predictor = predictor_from_checkpoint(ckpt, device)
    # The checkpoint's in-sample metrics, written by train/train_surrogate.py.
    training_perf = ckpt.get("final_metrics")

    dataset = test_set("mnist")
    n = args.images_per_digit
    heldout = split_indices(dataset, "mnist", "surrogate_heldout", n)
    heldout_start, _ = slice_for("mnist", "surrogate_heldout", n)

    all_predictions = []
    all_targets = []
    per_digit_results = {}

    for digit in range(10):
        selected_indices = heldout[digit].tolist()
        print(f"  Digit {digit}: {len(selected_indices)} held-out images")
        images = [dataset[idx][0] for idx in selected_indices]

        digit_preds, digit_targets = predict_and_ablate(
            capsule_model,
            predictor,
            tqdm(images, desc=f"    Digit {digit}", leave=False),
            device,
        )
        for preds, targets in zip(digit_preds, digit_targets):
            all_predictions.extend(preds)
            all_targets.extend(targets)

        digit_preds_flat = np.concatenate(digit_preds)
        digit_targets_flat = np.concatenate(digit_targets)

        pearson_r, _ = pearsonr(digit_preds_flat, digit_targets_flat)
        spearman_r, _ = spearmanr(digit_preds_flat, digit_targets_flat)
        prec50 = precision_at_k(digit_preds_flat, digit_targets_flat, k=50)

        per_digit_results[digit] = {
            "n_images": len(selected_indices),
            "pearson_r": float(pearson_r),
            "spearman_r": float(spearman_r),
            "precision_at_50": float(prec50),
        }

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    overall_pearson, _ = pearsonr(all_predictions, all_targets)
    overall_spearman, _ = spearmanr(all_predictions, all_targets)
    overall_prec50 = precision_at_k(all_predictions, all_targets, k=50)
    print(
        f"Held-out: Pearson {overall_pearson:.4f}, Spearman {overall_spearman:.4f}, "
        f"P@50 {overall_prec50:.4f}"
    )

    output = {
        "images_per_digit": args.images_per_digit,
        "skip_per_digit": heldout_start,
        "total_images": sum(r["n_images"] for r in per_digit_results.values()),
        "overall": {
            "pearson_r": float(overall_pearson),
            "spearman_r": float(overall_spearman),
            "precision_at_50": float(overall_prec50),
        },
        "training_set_reference": training_perf,
        "per_digit": per_digit_results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
