#!/usr/bin/env python3
"""Learned VEG beta_i of one trained VEG model, by that model's own capsule roles:
per-role statistics, omnibus and pairwise tests, and correlations with routing entropy
and importance. The checkpoint, its native taxonomy and its importance wrapper
(experiments/importance_wrapper.py) are all named by --dataset and --tag.

    python experiments/veg_beta_analysis.py --dataset mnist --tag best_entropy
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import f_oneway, levene, pearsonr, spearmanr, ttest_ind
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway, confint_effectsize_oneway
from torch.utils.data import DataLoader, Subset

from capsnet.data import test_set
from capsnet.io import (
    get_device,
    load_capsule_model,
    veg_beta_analysis,
    veg_checkpoint,
    veg_taxonomy,
)
from capsnet.taxonomy import ORDERED_ROLE_NAMES, assert_role_coverage, capsule_roles
from capsnet.veg import coupling_entropy


@torch.no_grad()
def mean_routing_entropy(model, loader, device) -> np.ndarray:
    """Mean final-iteration coupling entropy per primary capsule over the loader."""
    model.eval()
    entropy_sum = np.zeros(model.digits.routing_module.num_input_caps)
    count = 0
    for images, _ in loader:
        images = images.to(device)
        _, _, _, history = model.digits(model.primary(F.relu(model.conv1(images))))
        H = coupling_entropy(history[-1]["coupling_coeffs"])  # [B, num_capsules]
        entropy_sum += H.sum(dim=0).cpu().numpy()
        count += images.size(0)
    return entropy_sum / count


def load_importance(path, num_capsules: int) -> np.ndarray:
    """Per-capsule importance: the wrapper's [classes, capsules] matrix averaged over
    classes."""
    if not os.path.exists(path):
        raise SystemExit(f"--importance {path} does not exist.")
    with open(path) as f:
        matrix = np.array(json.load(f)["importance_matrix"])
    if matrix.ndim != 2:
        raise SystemExit(f"--importance {path}: importance_matrix is not 2-D.")
    importance = matrix.mean(axis=0)
    if len(importance) != num_capsules:
        raise SystemExit(
            f"--importance gives {len(importance)} capsules but the model has "
            f"{num_capsules}; it does not belong to this checkpoint."
        )
    return importance


def welch(a, b) -> tuple[float, float]:
    t_stat, p_val = ttest_ind(a, b, equal_var=False)
    return float(t_stat), float(p_val)


def omnibus(groups) -> dict:
    """ANOVA, Brown-Forsythe Levene, Welch ANOVA and eta^2 / omega^2 with CIs."""
    out = {}
    f_stat, p_value = f_oneway(*groups)
    out["anova"] = {"f_stat": float(f_stat), "p_value": float(p_value)}
    w_stat, w_p = levene(*groups, center="median")
    out["levene"] = {
        "statistic": float(w_stat),
        "p_value": float(w_p),
        "center": "median",
        "note": (
            "Brown-Forsythe variant (median-centred). Welch is used for the "
            "pairwise comparisons regardless of this result."
        ),
    }
    welch_res = anova_oneway(groups, use_var="unequal", welch_correction=True)
    out["welch_anova"] = {
        "f_stat": float(welch_res.statistic),
        "df_num": float(welch_res.df[0]),
        "df_denom": float(welch_res.df[1]),
        "p_value": float(welch_res.pvalue),
        "method": (
            "Welch's ANOVA (statsmodels anova_oneway, "
            "use_var='unequal', welch_correction=True)"
        ),
        "why": (
            "Sensitivity check for the equal-variance F above, which "
            "assumes homoscedasticity; the Levene test "
            + ("rejects" if w_p < 0.05 else "does not reject")
            + " it for this model at alpha=0.05. Does not replace it; "
            "both are reported."
        ),
    }
    # One-way design: eta^2 and omega^2 follow from F and the degrees of freedom.
    group_sizes = np.array([len(g) for g in groups], dtype=float)
    n_total = float(group_sizes.sum())
    k_groups = int(len(groups))
    df_b = float(k_groups - 1)
    df_w = float(n_total - k_groups)
    eta2_identity = (f_stat * df_b) / (f_stat * df_b + df_w)
    omega2_identity = ((f_stat - 1.0) * df_b) / (f_stat * df_b + df_w + 1.0)
    ci = confint_effectsize_oneway(
        float(f_stat), df=(df_b, df_w), alpha=0.05, nobs=int(n_total)
    )
    out["effect_size"] = {
        "eta_squared": float(eta2_identity),
        "eta_squared_ci95": [float(ci.eta2[0]), float(ci.eta2[1])],
        "omega_squared": float(omega2_identity),
        "omega_squared_ci95": [float(ci.ci_omega2[0]), float(ci.ci_omega2[1])],
        "n_total": n_total,
        "k_groups": k_groups,
        "df_between": df_b,
        "df_within": df_w,
        "method": (
            "eta^2 = F*df_b / (F*df_b + df_w) on the equal-variance "
            "decomposition; CI from statsmodels "
            "confint_effectsize_oneway (noncentral F)."
        ),
    }
    return out


def pairwise(role_betas: dict) -> dict:
    """Welch t-tests of hubs against specialists and non-critical capsules, with BH and
    Holm adjustment over the family."""
    tests = {}
    if "Primary Hub" in role_betas and "Specialist" in role_betas:
        hub = np.array(role_betas["Primary Hub"])
        spec = np.array(role_betas["Specialist"])
        t_stat, p_val = welch(hub, spec)
        tests["hub_vs_specialist"] = {
            "t_statistic": t_stat,
            "p_value": p_val,
            "hub_mean": float(hub.mean()),
            "specialist_mean": float(spec.mean()),
            "test": "Welch two-sample t-test (unequal variance)",
        }

    all_hub_betas = []
    for role in ("Primary Hub", "Secondary Hub"):
        all_hub_betas.extend(role_betas.get(role, []))

    if all_hub_betas and "Specialist" in role_betas:
        hubs = np.array(all_hub_betas)
        spec = np.array(role_betas["Specialist"])
        t_stat, p_val = welch(hubs, spec)
        tests["all_hubs_vs_specialists"] = {
            "t_statistic": t_stat,
            "p_value": p_val,
            "hubs_mean": float(hubs.mean()),
            "specialists_mean": float(spec.mean()),
            "test": "Welch two-sample t-test (unequal variance)",
        }

    if all_hub_betas and "Non-critical" in role_betas:
        hubs = np.array(all_hub_betas)
        other = np.array(role_betas["Non-critical"])
        t_stat, p_val = welch(hubs, other)
        tests["hubs_vs_non_critical"] = {
            "t_statistic": t_stat,
            "p_value": p_val,
            "hubs_mean": float(hubs.mean()),
            "non_critical_mean": float(other.mean()),
            "test": "Welch two-sample t-test (unequal variance)",
        }

    if tests:
        keys = sorted(tests)
        p = np.array([tests[k]["p_value"] for k in keys], dtype=float)
        rej_bh, p_bh, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")
        rej_holm, p_holm, _, _ = multipletests(p, alpha=0.05, method="holm")
        for i, k in enumerate(keys):
            tests[k]["p_value_bh"] = float(p_bh[i])
            tests[k]["significant_bh_q05"] = bool(rej_bh[i])
            tests[k]["p_value_holm"] = float(p_holm[i])
            tests[k]["significant_holm_a05"] = bool(rej_holm[i])
        tests["_multiplicity"] = {
            "method": "Benjamini-Hochberg (primary) and Holm-Bonferroni (reported alongside)",
            "implementation": "statsmodels.stats.multitest.multipletests",
            "q": 0.05,
            "alpha": 0.05,
            "family_size": len(p),
            "family": keys,
            "n_significant_bh": int(rej_bh.sum()),
            "n_significant_holm": int(rej_holm.sum()),
            "note": (
                "Post-hoc family following the omnibus ANOVA on the same capsules. "
                "Note these are capsules from a SINGLE trained model; the "
                "cross-model replication is in the per_model block of the native "
                "roll-up, not here."
            ),
        }
    return tests


def beta_by_role(beta_values, capsule_to_role: dict, num_capsules: int) -> dict:
    assert_role_coverage(capsule_to_role, num_capsules)
    role_betas = {}  # roles in order of their first capsule index
    for cap_id in range(num_capsules):
        role_betas.setdefault(capsule_to_role[cap_id], []).append(beta_values[cap_id])

    role_stats = {}
    for role, betas in role_betas.items():
        b = np.array(betas)
        role_stats[role] = {
            "count": len(betas),
            "mean": float(b.mean()),
            "std": float(b.std()),
            "median": float(np.median(b)),
            "min": float(b.min()),
            "max": float(b.max()),
            "q25": float(np.percentile(b, 25)),
            "q75": float(np.percentile(b, 75)),
        }
        print(f"  {role:15s} n={len(betas):5d} mean={b.mean():.6f} std={b.std():.6f}")

    role_stats["role_group_set"] = {
        "n_groups": len(role_betas),
        "groups": list(role_betas),
        "contract_roles": list(ORDERED_ROLE_NAMES),
        "note": (
            "The omnibus is fitted over exactly these groups. "
            "assert_role_coverage() guarantees they are a subset of "
            "contract_roles and that every capsule index is in one of them."
        ),
    }

    groups = [np.array(b) for b in role_betas.values()]
    if len(groups) >= 2:
        role_stats.update(omnibus(groups))

    role_stats["pairwise_tests"] = pairwise(role_betas)
    return role_stats


def correlations(beta_values, entropy_values, importance_values) -> dict:
    out = {}
    pearson_r, pearson_p = pearsonr(beta_values, entropy_values)
    spearman_r, spearman_p = spearmanr(beta_values, entropy_values)
    out["beta_vs_entropy"] = {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
    }
    pearson_r, pearson_p = pearsonr(beta_values, importance_values)
    spearman_r, spearman_p = spearmanr(beta_values, importance_values)
    out["beta_vs_importance"] = {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
    }
    return out


def summary(x) -> dict:
    return {
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "median": float(np.median(x)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, choices=["mnist", "cifar10"])
    parser.add_argument(
        "--tag", required=True, help="VEG model: best_entropy or seed<N>"
    )
    parser.add_argument("--output", help="default: the model's per-model beta analysis")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="random test images for the entropy pass (0 = all)",
    )
    parser.add_argument("--seed", type=int, default=42, help="seeds the image subset")
    args = parser.parse_args()
    model_path = veg_checkpoint(args.dataset, args.tag)
    taxonomy_path = veg_taxonomy(args.dataset, args.tag)
    importance_path = str(
        Path(taxonomy_path).with_name("per_digit_sparsity_wrapper.json")
    )
    output = args.output or veg_beta_analysis(args.dataset, args.tag)

    np.random.seed(args.seed)
    device = get_device()
    model = load_capsule_model(model_path, args.dataset, "entropy", device)
    beta_values = model.digits.routing_module.beta.detach().cpu().numpy()
    num_capsules = len(beta_values)

    with open(taxonomy_path) as f:
        capsule_to_role = capsule_roles(json.load(f))

    test_dataset = test_set(args.dataset)
    if args.num_samples > 0:
        indices = np.random.choice(
            len(test_dataset),
            size=min(args.num_samples, len(test_dataset)),
            replace=False,
        )
        test_dataset = Subset(test_dataset, indices)
    loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    entropy_values = mean_routing_entropy(model, loader, device)

    importance_values = load_importance(importance_path, num_capsules)

    print(f"{args.dataset}: {num_capsules} capsules, beta by role:")
    role_stats = beta_by_role(beta_values, capsule_to_role, num_capsules)

    results = {
        "dataset": args.dataset,
        "model_path": model_path,
        "num_capsules": int(num_capsules),
        "beta_statistics": summary(beta_values),
        "entropy_statistics": summary(entropy_values),
        "role_statistics": role_stats,
        "correlations": correlations(beta_values, entropy_values, importance_values),
        "capsule_roles": {str(k): v for k, v in capsule_to_role.items()},
    }

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
