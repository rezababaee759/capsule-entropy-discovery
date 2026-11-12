#!/usr/bin/env python3
"""Inference-only role experiments on the primary VEG checkpoint of one dataset:
  1. the realised gate G_i = exp(-beta_i * H_i) by native role;
  2. the cross-model spread of role-mean beta_i (from the native beta roll-up);
  3. test accuracy with beta_i overridden for role groups (suppression, role swap,
     zero gate everywhere, zero gate on hubs only) against the unmodified control.

    python experiments/role_intervention.py --dataset mnist
"""

import argparse
import json
from pathlib import Path

import numpy as np

from capsnet.data import test_loader
from capsnet.io import (
    VEG_PRIMARY_TAG,
    get_device,
    load_capsule_model,
    veg_checkpoint,
    veg_taxonomy,
)
from capsnet.model import build_capsule_network
from capsnet.provenance import attach
from capsnet.taxonomy import ORDERED_ROLE_NAMES, capsule_roles
from capsnet.training import evaluate
from capsnet.veg import GATE_BATCH, gate_distribution

NATIVE_JSON = {
    "mnist": "results/beta_values/veg_beta_analysis_mnist_native.json",
    "cifar10": "results/beta_values/veg_beta_analysis_cifar_native.json",
}


def role_mean_beta(beta_arr, cap_role, role_name):
    idxs = [c for c, r in cap_role.items() if r == role_name]
    if not idxs:
        return 0.0
    return float(np.mean(beta_arr[idxs]))


def cross_model_stability(per_model: dict) -> dict:
    """Spread of each role's mean beta_i across the models of the native roll-up."""
    stability = {}
    for role in ORDERED_ROLE_NAMES:
        vals = [
            m["role_means"][role]
            for m in per_model.values()
            if role in m.get("role_means", {})
        ]
        if len(vals) >= 2:
            arr = np.array(vals)
            cv = float(arr.std(ddof=1) / abs(arr.mean())) if arr.mean() != 0 else None
            stability[role] = {
                "n_models": len(vals),
                "values": vals,
                "mean": float(arr.mean()),
                "std_ddof1": float(arr.std(ddof=1)),
                "cv": cv,
            }
    return stability


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", choices=sorted(NATIVE_JSON), required=True)
    ap.add_argument("--max_gate_images", type=int, default=2000)
    ap.add_argument(
        "--output", help="default results/veg_ablation/role_intervention_{dataset}.json"
    )
    args = ap.parse_args()
    out_json = (
        args.output or f"results/veg_ablation/role_intervention_{args.dataset}.json"
    )

    device = get_device()
    model_path = veg_checkpoint(args.dataset, VEG_PRIMARY_TAG)
    tax_path = veg_taxonomy(args.dataset, VEG_PRIMARY_TAG)
    model = load_capsule_model(model_path, args.dataset, "entropy", device)
    loader = test_loader(args.dataset, GATE_BATCH)
    print(f"Device: {device}  Dataset: {args.dataset}")

    with open(tax_path) as f:
        cap_role = capsule_roles(json.load(f))
    capsules_per_role = {
        name: sum(1 for r in cap_role.values() if r == name)
        for name in ORDERED_ROLE_NAMES
    }

    routing_module = model.digits.routing_module
    beta_orig = routing_module.beta.detach().cpu().clone()
    beta_orig_np = beta_orig.numpy()

    gate_stats = gate_distribution(
        model, loader, device, cap_role, max_images=args.max_gate_images
    )

    with open(NATIVE_JSON[args.dataset]) as f:
        stability = cross_model_stability(json.load(f)["per_model"])

    n_primary = capsules_per_role["Primary Hub"]
    n_secondary = capsules_per_role["Secondary Hub"]
    n_hub = n_primary + n_secondary
    hub_mean = (
        role_mean_beta(beta_orig_np, cap_role, "Primary Hub") * n_primary
        + role_mean_beta(beta_orig_np, cap_role, "Secondary Hub") * n_secondary
    )
    hub_mean = hub_mean / n_hub if n_hub else 0.0
    spec_mean = role_mean_beta(beta_orig_np, cap_role, "Specialist")
    noncritical_mean = role_mean_beta(beta_orig_np, cap_role, "Non-critical")

    hub_idx = [c for c, r in cap_role.items() if r in ("Primary Hub", "Secondary Hub")]
    spec_idx = [c for c, r in cap_role.items() if r == "Specialist"]
    noncritical_idx = [c for c, r in cap_role.items() if r == "Non-critical"]

    def make_beta(overrides):
        b = beta_orig.clone()
        for idxs, val in overrides:
            if idxs:
                b[idxs] = val
        return b

    conditions = {
        "control_unmodified": [],
        "suppress_hubs_to_noncritical_mean": [(hub_idx, noncritical_mean)],
        "suppress_specialists_to_noncritical_mean": [(spec_idx, noncritical_mean)],
        "unsuppress_noncritical_to_hub_mean": [(noncritical_idx, hub_mean)],
        "hub_specialist_swap": [(hub_idx, spec_mean), (spec_idx, hub_mean)],
        "zero_gate_all": [(list(cap_role.keys()), 0.0)],
        "zero_gate_hub_only": [(hub_idx, 0.0)],
    }

    results = {}
    for name, overrides in conditions.items():
        routing_module.beta.data = make_beta(overrides).to(device)
        test = evaluate(model, loader, device, num_classes=10)
        acc = test["correct"] / test["total"]
        per_class = (
            test["per_class_correct"] / np.maximum(test["per_class_total"], 1)
        ).tolist()
        results[name] = {"accuracy": acc, "per_class_accuracy": per_class}
        print(f"  {name:38s} acc={acc*100:.3f}%")
        routing_module.beta.data = beta_orig.to(device)

    output = {
        "dataset": args.dataset,
        "model_path": model_path,
        "taxonomy_path": tax_path,
        "gate_distribution_by_role": gate_stats,
        "beta_init_value": float(
            build_capsule_network(args.dataset, "entropy").digits.routing_module.beta[0]
        ),
        "beta_cross_seed_stability": stability,
        "role_mean_beta_learned": {
            "Hub(P+S)": hub_mean,
            "Specialist": spec_mean,
            "Non-critical": noncritical_mean,
        },
        "intervention_accuracy": results,
    }
    attach(
        output,
        script="experiments/role_intervention.py",
        inputs=[model_path, tax_path],
        seeds={
            "rng": None,
            "note": "No RNG. Gate collection and every intervention are "
            "deterministic passes over an unshuffled test loader; the "
            "beta overrides are exact.",
        },
        sample_counts={
            "max_gate_images_requested": args.max_gate_images,
            "gate_observations_per_role": {
                r: v["n_obs"] for r, v in gate_stats.items()
            },
            "capsules_per_role": capsules_per_role,
            "n_intervention_conditions": len(conditions),
        },
        method=(
            "Inference-only on one frozen best-accuracy checkpoint per dataset. "
            "(1) Realised gate G_i = exp(-beta_i*H_i) from the final routing "
            "iteration, summarised per native role. (2) Cross-model spread of "
            "role-mean beta. (3) The unmodified control and six beta overrides, "
            "each with full test-set and per-class accuracy; no delta is stored. "
            "No hypothesis test and no interval are computed."
        ),
        packages=["numpy"],
    )

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
