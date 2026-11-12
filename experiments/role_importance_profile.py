#!/usr/bin/env python3
"""Role-level profile of ground-truth importance: each role's share of the positive
importance mass, threshold counts of the taxonomy rule (checked against
hub_taxonomy.json), the beta_i displacement of each suppression intervention, and the
agreement of the primary VEG model with the seeded ones.

    python experiments/role_importance_profile.py [--check] [--output PATH]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import spearmanr

from capsnet import PROJECT_ROOT
from capsnet.provenance import attach, check_artifact
from capsnet.taxonomy import ROLE_NAMES, ROLES, classify, threshold_counts

OUTPUT = "results/data/role_importance_profile.json"
MODELS = {
    "mnist_dynamic_primary": "results/data/mnist",
    "mnist_veg_primary": "results/data/mnist/veg_best_entropy",
    "cifar10_dynamic_primary": "results/data/cifar10",
    "cifar10_veg_primary": "results/data/cifar10/veg_best_entropy",
}
VEG_SEEDS = ("seed42", "seed123", "seed456")
INTERVENTIONS = {
    "mnist": "results/veg_ablation/role_intervention_mnist.json",
    "cifar10": "results/veg_ablation/role_intervention_cifar10.json",
}
# Role display name -> role key, in the rule's order.
ROLE_KEYS = {ROLE_NAMES[key]: key for key in ROLES}


def profile(directory: str) -> dict:
    matrix = np.load(PROJECT_ROOT / directory / "importance_matrix.npy")
    taxonomy = json.loads((PROJECT_ROOT / directory / "hub_taxonomy.json").read_text())
    high_count, moderate_count, high, moderate = threshold_counts(matrix)
    if any(sorted(v) != sorted(taxonomy[key]) for key, v in classify(matrix).items()):
        raise SystemExit(
            f"{directory}: re-applying the taxonomy rule does not reproduce "
            f"hub_taxonomy.json; refusing to report counts from a different rule"
        )

    role_of = {i: name for name, key in ROLE_KEYS.items() for i in taxonomy[key]}
    positive = np.maximum(matrix, 0.0).sum(axis=0)
    total = positive.sum()
    groups = {
        "Hub (Primary + Secondary)": taxonomy["primary_hubs"]
        + taxonomy["secondary_hubs"],
        "Specialist": taxonomy["specialists"],
        "Generalist": taxonomy["generalists"],
        "Non-critical": taxonomy["non_critical"],
    }

    critical_1_2 = [i for i in range(matrix.shape[1]) if 1 <= high_count[i] <= 2]
    specialists = np.array(taxonomy["specialists"])
    top = matrix.argmax(axis=1)
    return {
        "importance_matrix": f"{directory}/importance_matrix.npy",
        "taxonomy": f"{directory}/hub_taxonomy.json",
        "n_capsules": int(matrix.shape[1]),
        "share_of_positive_importance": {
            name: {"n_capsules": len(idx), "share": float(positive[idx].sum() / total)}
            for name, idx in groups.items()
        },
        "taxonomy_rule": {
            "per_digit_median": [float(x) for x in moderate],
            "per_digit_p90": [float(x) for x in high],
            "negative_fraction_of_matrix": float((matrix < 0).mean()),
            "critical_for_1_or_2_digits": {
                "n": len(critical_1_2),
                "by_role": {
                    name: sum(role_of[i] == name for i in critical_1_2)
                    for name in ROLE_KEYS
                },
            },
            "specialists_at_or_above_median_for_7_or_more_digits": int(
                (moderate_count[specialists] >= 7).sum()
            ),
            "n_specialists": int(specialists.size),
            "top_capsule_per_digit": [
                {"digit": d, "capsule": int(c), "role": role_of[int(c)]}
                for d, c in enumerate(top)
            ],
            "n_digits_whose_top_capsule_is_a_specialist": int(
                sum(role_of[int(c)] == "Specialist" for c in top)
            ),
        },
    }


def displacement(path: str) -> dict:
    d = json.loads((PROJECT_ROOT / path).read_text())
    beta = d["role_mean_beta_learned"]
    acc = d["intervention_accuracy"]
    target = beta["Non-critical"]
    control = acc["control_unmodified"]["accuracy"]
    hub_shift = target - beta["Hub(P+S)"]
    spec_shift = target - beta["Specialist"]
    return {
        "source": path,
        "target_beta_non_critical_mean": target,
        "hub_mean_beta": beta["Hub(P+S)"],
        "specialist_mean_beta": beta["Specialist"],
        "hub_delta_beta": hub_shift,
        "specialist_delta_beta": spec_shift,
        "ratio_hub_to_specialist": hub_shift / spec_shift,
        "hub_accuracy_change_pp": 100
        * (acc["suppress_hubs_to_noncritical_mean"]["accuracy"] - control),
        "specialist_accuracy_change_pp": 100
        * (acc["suppress_specialists_to_noncritical_mean"]["accuracy"] - control),
    }


def native_agreement(dataset: str) -> dict:
    """Primary VEG model against each seeded VEG model, on their native taxonomies."""
    base = f"results/data/{dataset}/veg_best_entropy"
    m0 = np.load(PROJECT_ROOT / base / "importance_matrix.npy")
    t0 = json.loads((PROJECT_ROOT / base / "hub_taxonomy.json").read_text())
    out = {}
    for seed in VEG_SEEDS:
        other = f"results/data/{dataset}/veg_{seed}"
        m1 = np.load(PROJECT_ROOT / other / "importance_matrix.npy")
        t1 = json.loads((PROJECT_ROOT / other / "hub_taxonomy.json").read_text())
        hubs0 = set(t0["primary_hubs"] + t0["secondary_hubs"])
        hubs1 = set(t1["primary_hubs"] + t1["secondary_hubs"])
        out[seed] = {
            "importance_spearman": float(spearmanr(m0.ravel(), m1.ravel()).statistic),
            "primary_hubs_primary_model": len(t0["primary_hubs"]),
            "primary_hubs_seed_model": len(t1["primary_hubs"]),
            "primary_hubs_shared": len(
                set(t0["primary_hubs"]) & set(t1["primary_hubs"])
            ),
            "hubs_primary_model": len(hubs0),
            "hubs_seed_model": len(hubs1),
            "hubs_shared": len(hubs0 & hubs1),
        }
    return out


def build() -> dict:
    return {
        "description": (
            "Role-level profile of ground-truth single-capsule ablation importance for the "
            "primary dynamic-routing and primary per-capsule-beta models: each role's share "
            "of the positive importance mass (negative entries set to zero, summed over "
            "digits and capsules), how the taxonomy rule sorts capsules critical for one or "
            "two digits, and how far each suppression intervention moves beta_i."
        ),
        "share_definition": (
            "share = sum over the role's capsules and all digits of max(I, 0), divided by "
            "the same sum over all capsules"
        ),
        "models": {name: profile(directory) for name, directory in MODELS.items()},
        "intervention_displacement": {
            ds: displacement(p) for ds, p in INTERVENTIONS.items()
        },
        "veg_native_agreement_with_primary": {
            ds: native_agreement(ds) for ds in INTERVENTIONS
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if the committed artifact differs from a fresh reduction",
    )
    ap.add_argument(
        "--output",
        default=OUTPUT,
        help="artifact path, relative to the repository root",
    )
    args = ap.parse_args()

    out = build()
    if args.check:
        check_artifact(args.output, out, "experiments/role_importance_profile.py")
        return

    dirs = list(MODELS.values()) + [
        f"results/data/{ds}/veg_{seed}" for ds in INTERVENTIONS for seed in VEG_SEEDS
    ]
    inputs = [
        f"{d}/{f}" for d in dirs for f in ("importance_matrix.npy", "hub_taxonomy.json")
    ]
    attach(
        out,
        script="experiments/role_importance_profile.py",
        inputs=inputs + list(INTERVENTIONS.values()),
        seeds={"rng": None, "note": "Pure read-and-reduce over saved artifacts."},
        sample_counts={name: m["n_capsules"] for name, m in out["models"].items()},
        method=(
            "Role shares of clipped importance mass; taxonomy-rule threshold counts "
            "re-derived from the saved matrix and checked against hub_taxonomy.json; "
            "beta_i displacement of each suppression intervention; Spearman "
            "correlation and hub overlap between the primary and seeded VEG models' "
            "native importance matrices and taxonomies."
        ),
        packages=["numpy", "scipy"],
    )
    (PROJECT_ROOT / args.output).write_text(json.dumps(out, indent=2) + "\n")
    for name, m in out["models"].items():
        shares = ", ".join(
            f"{k} {100 * v['share']:.1f}%"
            for k, v in m["share_of_positive_importance"].items()
        )
        print(f"{name}: {shares}")
    for ds, d in out["intervention_displacement"].items():
        print(
            f"{ds}: delta beta hub {d['hub_delta_beta']:.3f}, specialist "
            f"{d['specialist_delta_beta']:.3f}, ratio {d['ratio_hub_to_specialist']:.2f}"
        )
    for ds, pairs in out["veg_native_agreement_with_primary"].items():
        for seed, a in pairs.items():
            print(
                f"{ds} primary vs {seed}: spearman {a['importance_spearman']:.3f}, primary hubs "
                f"shared {a['primary_hubs_shared']}/{a['primary_hubs_primary_model']}"
            )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
