#!/usr/bin/env python3
"""Per-class importance of every primary capsule by exhaustive single-capsule
ablation on the taxonomy_calibration split, the five-role taxonomy derived from it,
and hub clusters. Writes importance_matrix.npy [n_classes, n_capsules] and
hub_taxonomy.json into --output_dir.

    python experiments/hub_taxonomy.py --dataset mnist \
        [--capsule_model PT --routing_strategy entropy --output_dir DIR]
"""

import argparse
import json
import os

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from tqdm import tqdm

from capsnet.ablation import single_capsule_ablation_drops
from capsnet.data import class_indices, slice_for, test_set
from capsnet.io import get_device, load_capsule_model, resolve_output
from capsnet.taxonomy import classify

CIFAR10_CLASS_NAMES = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
]

TAXONOMY = {
    "mnist": {"clustering": "kmeans"},
    "cifar10": {"clustering": "ward", "class_names": CIFAR10_CLASS_NAMES},
}
N_HUB_CLUSTERS = 5


@torch.no_grad()
def ablation_importance(model, image):
    """Drop in the predicted class's capsule length as each primary capsule is zeroed."""
    primary_out = model.primary(torch.relu(model.conv1(image)))
    digit_caps, _, _, _ = model.digits(primary_out)
    lengths = torch.norm(digit_caps, dim=-1)
    predicted = int(lengths[0].argmax())
    return single_capsule_ablation_drops(
        model, primary_out, predicted, float(lengths[0, predicted])
    )


def importance_matrix(model, dataset, device, samples_per_class):
    """[n_classes, n_capsules]: mean ablation importance over each class's
    calibration images."""
    ds = test_set(dataset)
    per_class = class_indices(ds)
    start, stop = slice_for(dataset, "taxonomy_calibration", samples_per_class)
    matrix = np.zeros((len(per_class), model.digits.num_input_caps))
    for c, indices in per_class.items():
        print(f"\nClass {c}:")
        scores = [
            ablation_importance(model, ds[i][0].unsqueeze(0).to(device))
            for i in tqdm(indices[start:stop], desc="  Processing")
        ]
        matrix[c] = np.array(scores).mean(axis=0)
        print(f"  Mean importance: {matrix[c].mean():.6f}")
    return matrix


def kmeans_clusters(patterns, hub_indices, n_clusters):
    """Clusters "0".."k-1": {capsules, pattern, dominant_digits}."""
    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(
        patterns
    )
    info = {}
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        avg_pattern = patterns[mask].mean(axis=0)
        info[str(cluster_id)] = {
            "capsules": [hub_indices[i] for i, m in enumerate(mask) if m],
            "pattern": avg_pattern.tolist(),
            "dominant_digits": np.argsort(avg_pattern)[::-1][:3].tolist(),
        }
    return info


def ward_clusters(patterns, hub_indices, n_clusters, class_names):
    """Clusters "1".."k" of Ward linkage: {capsules, top_classes, pattern}."""
    tree = linkage(patterns, method="ward")
    labels = fcluster(tree, t=n_clusters, criterion="maxclust")
    info = {}
    for cluster_id in range(1, n_clusters + 1):
        mask = labels == cluster_id
        avg_pattern = patterns[mask].mean(axis=0)
        info[str(cluster_id)] = {
            "capsules": [hub_indices[i] for i, m in enumerate(mask) if m],
            "top_classes": [class_names[i] for i in avg_pattern.argsort()[::-1][:3]],
            "pattern": avg_pattern.tolist(),
        }
    return info


def cluster_hubs(matrix, hub_indices, config):
    """Cluster hubs by their importance pattern over classes, normalised to sum 1."""
    n_clusters = N_HUB_CLUSTERS
    if len(hub_indices) < n_clusters:
        n_clusters = max(2, len(hub_indices) // 2)
    patterns = matrix[:, hub_indices].T
    patterns = patterns / (patterns.sum(axis=1, keepdims=True) + 1e-8)
    if config["clustering"] == "kmeans":
        return kmeans_clusters(patterns, hub_indices, n_clusters)
    return ward_clusters(patterns, hub_indices, n_clusters, config["class_names"])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", required=True, choices=sorted(TAXONOMY))
    parser.add_argument(
        "--capsule_model", help="default: models/capsule_{dataset}_best_dynamic.pt"
    )
    parser.add_argument(
        "--routing_strategy",
        default="dynamic",
        choices=["dynamic", "entropy", "entropy_uniform"],
        help="routing of --capsule_model; VEG checkpoints need entropy",
    )
    parser.add_argument(
        "--samples", type=int, default=50, help="calibration images per class"
    )
    parser.add_argument("--output_dir", help="default: results/data/{dataset}")
    parser.add_argument(
        "--force",
        action="store_true",
        help="write the canonical output from --capsule_model",
    )
    args = parser.parse_args()
    config = TAXONOMY[args.dataset]
    capsule_model = (
        args.capsule_model or f"models/capsule_{args.dataset}_best_dynamic.pt"
    )
    output_dir = resolve_output(
        parser, args, f"results/data/{args.dataset}", ["capsule_model"], "output_dir"
    )

    os.makedirs(output_dir, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")
    model = load_capsule_model(
        capsule_model, args.dataset, args.routing_strategy, device
    )

    matrix = importance_matrix(model, args.dataset, device, args.samples)
    np.save(os.path.join(output_dir, "importance_matrix.npy"), matrix)

    taxonomy = classify(matrix)
    for role, members in taxonomy.items():
        print(f"  {role}: {len(members)}")
    hubs = taxonomy["primary_hubs"] + taxonomy["secondary_hubs"]
    if hubs:
        taxonomy["hub_clusters"] = cluster_hubs(matrix, hubs, config)

    path = os.path.join(output_dir, "hub_taxonomy.json")
    with open(path, "w") as f:
        json.dump(taxonomy, f, indent=2)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
