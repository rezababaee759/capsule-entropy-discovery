#!/usr/bin/env python3
"""Jaccard overlap of the Primary-Hub and all-Hub index sets between a VEG-native
taxonomy and the dynamic-routing taxonomy of the same dataset, with a hypergeometric
chance model.

    python experiments/index_transfer_overlap.py [--output PATH]
"""

import argparse
import json

import numpy as np
from scipy import stats

from capsnet import PROJECT_ROOT
from capsnet.io import veg_taxonomy
from capsnet.provenance import attach
from capsnet.taxonomy import ROLES

# Each pairing is two models that do not share an initialisation. The primary MNIST VEG
# and dynamic-routing models do (results/data/checkpoint_similarity.json), so MNIST
# pairs the seed-42 VEG model with the primary dynamic-routing model.
PAIRINGS = {
    "mnist_seed42": {
        "veg": veg_taxonomy("mnist", "seed42"),
        "baseline": "results/data/mnist/hub_taxonomy.json",
    },
    "cifar10_best_entropy": {
        "veg": veg_taxonomy("cifar10", "best_entropy"),
        "baseline": "results/data/cifar10/hub_taxonomy.json",
    },
}


def jaccard(a, b):
    """Jaccard index of two index sets. Empty-vs-empty is defined as 0.0 here:
    two models that both nominate no hubs share no evidence of transfer."""
    a, b = set(a), set(b)
    union = a | b
    return (len(a & b) / len(union)) if union else 0.0


def universe_size(taxonomy):
    """Number of capsules the taxonomy classifies (the union of its five role lists)."""
    indices = set()
    for key in ROLES:
        indices |= set(taxonomy.get(key, []))
    return len(indices)


def chance_overlap(n_universe, n_a, n_b, observed):
    """Intersection of two independently chosen index sets under
    hypergeometric(n_universe, n_a, n_b): expected overlap and an exact two-sided p that
    sums the outcomes no more likely than the observed one."""
    if n_universe <= 0 or n_a <= 0 or n_b <= 0:
        return None
    hg = stats.hypergeom(n_universe, n_a, n_b)
    expected = float(hg.mean())
    p_observed = hg.pmf(observed)
    support = np.arange(0, min(n_a, n_b) + 1)
    pmf = hg.pmf(support)
    p_two_sided = float(pmf[pmf <= p_observed + 1e-15].sum())

    if observed < expected and hg.cdf(observed) < 0.05:
        verdict = "below chance"
    elif observed > expected and hg.sf(observed - 1) < 0.05:
        verdict = "above chance"
    else:
        verdict = "indistinguishable from chance"

    denom = n_a + n_b - expected
    return {
        "n_universe": int(n_universe),
        "expected_intersection": expected,
        "expected_jaccard": float(expected / denom) if denom else None,
        "p_two_sided": min(p_two_sided, 1.0),
        "p_at_most_observed": float(hg.cdf(observed)),
        "p_at_least_observed": float(hg.sf(observed - 1)),
        "verdict": verdict,
        "null": (
            "hypergeometric(n_universe, n_veg, n_baseline) -- the two "
            "taxonomies select capsules independently"
        ),
    }


def overlap(veg_path, baseline_path):
    veg = json.loads((PROJECT_ROOT / veg_path).read_text())
    base = json.loads((PROJECT_ROOT / baseline_path).read_text())

    v_primary, b_primary = veg["primary_hubs"], base["primary_hubs"]
    v_all = list(veg["primary_hubs"]) + list(veg["secondary_hubs"])
    b_all = list(base["primary_hubs"]) + list(base["secondary_hubs"])

    n_universe = universe_size(veg)
    n_universe_base = universe_size(base)
    if n_universe != n_universe_base:
        raise SystemExit(
            f"{veg_path} classifies {n_universe} capsules but {baseline_path} "
            f"classifies {n_universe_base}; the two taxonomies must cover the same "
            f"index space for an overlap to mean anything."
        )

    ph_inter = len(set(v_primary) & set(b_primary))
    ah_inter = len(set(v_all) & set(b_all))

    return {
        "veg_taxonomy": veg_path,
        "baseline_taxonomy": baseline_path,
        "n_universe": int(n_universe),
        "primary_hub_jaccard": jaccard(v_primary, b_primary),
        "primary_hub_intersection": ph_inter,
        "primary_hub_veg_n": len(v_primary),
        "primary_hub_baseline_n": len(b_primary),
        "primary_hub_chance": chance_overlap(
            n_universe, len(v_primary), len(b_primary), ph_inter
        ),
        "all_hub_jaccard": jaccard(v_all, b_all),
        "all_hub_intersection": ah_inter,
        "all_hub_veg_n": len(v_all),
        "all_hub_baseline_n": len(b_all),
        "all_hub_chance": chance_overlap(n_universe, len(v_all), len(b_all), ah_inter),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--output", default="results/data/index_transfer_overlap.json")
    args = ap.parse_args()

    results = {}
    print(
        f"{'pairing':<24} {'PH veg':>7} {'PH base':>8} {'PH cap':>7} {'PH Jacc':>9} {'all Jacc':>9}"
    )
    print("-" * 68)
    for name, paths in PAIRINGS.items():
        r = overlap(paths["veg"], paths["baseline"])
        results[name] = r
        print(
            f"{name:<24} {r['primary_hub_veg_n']:>7} {r['primary_hub_baseline_n']:>8} "
            f"{r['primary_hub_intersection']:>7} {r['primary_hub_jaccard']:>9.4f} "
            f"{r['all_hub_jaccard']:>9.4f}"
        )

    out = PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    attach(
        results,
        script="experiments/index_transfer_overlap.py",
        inputs=sorted({p for pair in PAIRINGS.values() for p in pair.values()}),
        seeds={"rng": None, "note": "Fully deterministic; set comparison only."},
        sample_counts={
            k: {
                "primary_hub_veg_n": v.get("primary_hub_veg_n"),
                "primary_hub_baseline_n": v.get("primary_hub_baseline_n"),
            }
            for k, v in results.items()
            if isinstance(v, dict)
        },
        method=(
            "Jaccard overlap of Primary-Hub and combined-Hub index sets between a "
            "VEG-native taxonomy and the dynamic-routing baseline, with a "
            "hypergeometric chance model giving the expected overlap and an exact "
            "two-sided p by summing outcomes no more likely than observed."
        ),
        packages=["numpy", "scipy"],
    )

    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
