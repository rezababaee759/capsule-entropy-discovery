#!/usr/bin/env python3
"""Re-classify an importance matrix over a grid of taxonomy thresholds (hub
percentile, Primary/Secondary Hub class counts, moderate percentile); report role
counts and, unless --no_beta, whether the per-capsule beta of the same VEG model
(--model) keeps Primary Hub < Secondary Hub < Specialist < Generalist < Non-critical.

    python experiments/threshold_sensitivity.py [--importance NPY --model PT |
        --no_beta --importance NPY] [--output PATH]
"""

import os
import json
import argparse
from itertools import product

import math
import numpy as np

from capsnet.io import load_capsule_model, resolve_output
from capsnet.provenance import attach
from capsnet.taxonomy import (
    HUB_PCT,
    MODERATE_PCT,
    PRIMARY_MIN,
    ROLE_ORDER,
    SECONDARY_MIN,
    classify,
)


def check_beta_ordering(classification, beta_values):
    """(mean beta ascends along ROLE_ORDER, mean beta per role); the first is None
    when a role is empty."""
    means = {}
    for role, indices in classification.items():
        if indices:
            means[role] = float(np.mean(beta_values[indices]))
        else:
            means[role] = float("nan")

    seq = [means.get(r) for r in ROLE_ORDER]
    if any(v is None or math.isnan(v) for v in seq):
        return None, means

    ordering_preserved = all(a < b for a, b in zip(seq, seq[1:]))
    return ordering_preserved, means


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--importance",
        default="results/data/mnist/veg_best_entropy/importance_matrix.npy",
    )
    parser.add_argument(
        "--model",
        default="models/capsule_mnist_best_entropy.pt",
        help="MNIST checkpoint supplying the per-capsule beta; the same model as "
        "--importance",
    )
    parser.add_argument(
        "--routing_strategy",
        default="entropy",
        choices=["dynamic", "entropy", "entropy_uniform"],
        help="routing strategy of --model",
    )
    parser.add_argument(
        "--no_beta",
        action="store_true",
        help="role-count sweep only (a model without beta); every ordering is null",
    )
    parser.add_argument(
        "--output", help="default: results/data/mnist/threshold_sensitivity.json"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="write the canonical output from --importance or --model",
    )
    args = parser.parse_args()
    output = resolve_output(
        parser,
        args,
        "results/data/mnist/threshold_sensitivity.json",
        ["importance", "model", "routing_strategy", "no_beta"],
    )

    importance_matrix = np.load(args.importance)  # [n_classes, n_capsules]
    n_caps = importance_matrix.shape[1]
    print(f"Importance matrix: {importance_matrix.shape}")

    beta_values = np.zeros(n_caps)
    # A beta/importance length mismatch means two different models.
    beta_values_available = False
    if args.no_beta:
        print(
            "--no_beta: skipping the beta ordering check "
            "(role-count sweep only; every row will report null)."
        )
    else:
        if not os.path.exists(args.model):
            raise SystemExit(
                f"--model {args.model} does not exist. Supply the checkpoint whose "
                f"per-capsule beta pairs with --importance, or pass --no_beta."
            )
        veg_model = load_capsule_model(
            args.model, "mnist", args.routing_strategy, "cpu"
        )
        beta_param = getattr(veg_model.digits.routing_module, "beta", None)
        if beta_param is None:
            raise SystemExit(
                f"--model {args.model} has no routing_module.beta. A dynamic-routing "
                f"checkpoint carries no per-capsule beta; pass --no_beta to run the "
                f"role-count sweep without the ordering check."
            )
        beta_values = beta_param.detach().cpu().numpy().reshape(-1)
        if beta_values.shape[0] != n_caps:
            raise SystemExit(
                f"beta/importance mismatch: --model {args.model} has {beta_values.shape[0]} "
                f"beta values but --importance {args.importance} has {n_caps} capsules. "
                f"These are different models; pairing them by index is invalid."
            )
        beta_values_available = True
        print(
            f"Loaded per-capsule beta values from {args.model}: "
            f"shape={beta_values.shape}, min={beta_values.min():.3f}, "
            f"max={beta_values.max():.3f}"
        )

    percentiles = [80, 85, 90, 95]
    primary_hub_thresholds = [6, 7, 8]
    secondary_hub_thresholds = [3, 4, 5]
    moderate_percentiles = [40, 50, 60]

    results = []
    print(
        f"\n{'Pct':>5} {'PH>=':>5} {'SH>=':>5} | {'N_PH':>6} {'N_SH':>6} {'N_Sp':>6} {'N_Gen':>6} {'N_NC':>6} | {'order':>8} {'Current':>8}"
    )
    print(f"{'-'*80}")

    for pct, ph_min, sh_min, moderate_pct in product(
        percentiles,
        primary_hub_thresholds,
        secondary_hub_thresholds,
        moderate_percentiles,
    ):
        if sh_min >= ph_min:
            continue
        cls = classify(
            importance_matrix,
            hub_pct=pct,
            moderate_pct=moderate_pct,
            primary_min=ph_min,
            secondary_min=sh_min,
        )

        if beta_values_available:
            ordering_ok, beta_means = check_beta_ordering(cls, beta_values)
        else:
            ordering_ok = None
            beta_means = {}

        is_current = (
            pct == HUB_PCT
            and ph_min == PRIMARY_MIN
            and sh_min == SECONDARY_MIN
            and moderate_pct == MODERATE_PCT
        )

        row = {
            "percentile": pct,
            "moderate_pct": moderate_pct,
            "primary_hub_min_digits": ph_min,
            "secondary_hub_min_digits": sh_min,
            "n_primary_hubs": len(cls["primary_hubs"]),
            "n_secondary_hubs": len(cls["secondary_hubs"]),
            "n_specialists": len(cls["specialists"]),
            "n_generalists": len(cls["generalists"]),
            "n_non_critical": len(cls["non_critical"]),
            "beta_ordering_preserved": ordering_ok,
            "beta_means": beta_means,
            "is_current_config": is_current,
        }
        results.append(row)

        print(
            f"{pct:>5} {ph_min:>5} {sh_min:>5} | "
            f"{len(cls['primary_hubs']):>6} {len(cls['secondary_hubs']):>6} "
            f"{len(cls['specialists']):>6} {len(cls['generalists']):>6} "
            f"{len(cls['non_critical']):>6} | "
            f"{'n/a' if ordering_ok is None else ('YES' if ordering_ok else 'NO'):>8}"
            f" {'<-- current' if is_current else '':>8}"
        )

    # None (not evaluable) counts in neither the numerator nor n_evaluable.
    ordering_preserved_count = sum(
        1 for r in results if r["beta_ordering_preserved"] is True
    )
    n_evaluable = sum(1 for r in results if r["beta_ordering_preserved"] is not None)
    if n_evaluable:
        print(
            f"\nBeta ordering preserved in {ordering_preserved_count}/{n_evaluable} evaluable "
            f"configurations ({100*ordering_preserved_count/n_evaluable:.0f}%) "
            f"of {len(results)} swept"
        )
    else:
        print(f"\nBeta ordering not evaluable for any of {len(results)} configurations")

    ph_counts = [r["n_primary_hubs"] for r in results]
    sh_counts = [r["n_secondary_hubs"] for r in results]
    sp_counts = [r["n_specialists"] for r in results]
    canonical_row = next(r for r in results if r["is_current_config"])
    print(
        f"\nPrimary Hub count range:    {min(ph_counts)}-{max(ph_counts)} "
        f"(canonical: {canonical_row['n_primary_hubs']})"
    )
    print(
        f"Secondary Hub count range:  {min(sh_counts)}-{max(sh_counts)} "
        f"(canonical: {canonical_row['n_secondary_hubs']})"
    )
    print(
        f"Specialist count range:     {min(sp_counts)}-{max(sp_counts)} "
        f"(canonical: {canonical_row['n_specialists']})"
    )

    output_data = {
        "n_capsules": n_caps,
        "beta_values_from_model": beta_values_available,
        "inputs": {
            "importance": args.importance,
            "model": None if args.no_beta else args.model,
            "routing_strategy": None if args.no_beta else args.routing_strategy,
        },
        "canonical_config": {
            "percentile": HUB_PCT,
            "primary_hub_min_digits": PRIMARY_MIN,
            "secondary_hub_min_digits": SECONDARY_MIN,
            "moderate_pct": MODERATE_PCT,
        },
        "results": results,
        "summary": {
            "total_configs": len(results),
            "n_evaluable": n_evaluable,
            "beta_ordering_preserved": ordering_preserved_count,
            "primary_hub_count_range": [min(ph_counts), max(ph_counts)],
            "secondary_hub_count_range": [min(sh_counts), max(sh_counts)],
            "specialist_count_range": [min(sp_counts), max(sp_counts)],
        },
    }

    attach(
        output_data,
        script="experiments/threshold_sensitivity.py",
        inputs=[args.importance] + ([] if args.no_beta else [args.model]),
        seeds={
            "rng": None,
            "note": "No RNG. The sweep re-classifies a precomputed importance "
            "matrix and reads a fixed beta vector.",
        },
        sample_counts={
            "n_capsules": int(n_caps),
            "total_configs": len(results),
            "n_evaluable": n_evaluable,
            "grid": {
                "percentiles": percentiles,
                "primary_hub_min_digits": primary_hub_thresholds,
                "secondary_hub_min_digits": secondary_hub_thresholds,
                "moderate_percentiles": moderate_percentiles,
            },
        },
        method=(
            "Exhaustive re-classification over the four-dimensional threshold "
            "grid, then an ordering predicate on per-capsule beta means "
            "(Primary Hub < Secondary Hub < Specialist < Generalist < "
            "Non-critical). Configurations whose ordering cannot be evaluated "
            "record null and are counted in neither the numerator nor "
            "n_evaluable."
        ),
        packages=["numpy"],
    )

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()
