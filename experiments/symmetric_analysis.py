#!/usr/bin/env python3
"""Clustering-aware analysis of the symmetric raw records: a two-way cluster-robust
linear-probability model and logit model of correct ~ role x condition, per-seed logit
refits, and a capsule-level Wilcoxon / rank-biserial / BCa companion analysis.

Usage:
    python experiments/symmetric_analysis.py --dataset mnist
    python experiments/symmetric_analysis.py --dataset cifar10
"""

import argparse
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as sp_stats
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups

from capsnet import PROJECT_ROOT
from capsnet.provenance import attach
from capsnet.stats import (
    N_BOOT,
    bca_bootstrap_ci,
    rank_biserial_matched,
    two_way_logit_cov,
)
from capsnet.io import resolve_output
from capsnet.symmetric import RECORDS, SAMPLES_PER_CLASS, cluster_codes, design_frame

ROLES = ["hub", "specialist", "random"]

DATASETS = {
    "mnist": {"name": "MNIST", "samples_key": "samples_per_digit"},
    "cifar10": {"name": "CIFAR-10", "samples_key": "samples_per_class"},
}


def load_records(records):
    path = PROJECT_ROOT / records
    if not path.exists():
        raise SystemExit(
            f"{path} not found; experiments/symmetric_raw_records.py writes it."
        )
    return pd.read_parquet(path)


def capsule_level_table(df):
    """One row per (seed, role, capsule_idx, condition): mean and count of correct."""
    return (
        df.groupby(["seed", "role", "capsule_idx", "condition"])["correct"]
        .agg(["mean", "count"])
        .reset_index()
    )


def companion_analysis(cap_df):
    """Per-seed and pooled paired tests and BCa CIs on the within-minus-cross gap.

    Capsules missing one condition are dropped, and role x seed cells with fewer than
    three paired capsules are skipped; both are recorded under 'filtering'.
    """
    out = {"per_seed": {}, "pooled": {}}
    seeds = sorted(cap_df["seed"].unique().tolist())
    out["filtering"] = {
        "per_cell": [],
        "cells_skipped_min_n": [],
        "min_paired_capsules": 3,
    }
    out["bca_diagnostics"] = {"per_seed": {}, "pooled": {}}

    for role in ROLES:
        pooled_within = []
        pooled_cross = []
        out["per_seed"].setdefault(role, {})
        out["bca_diagnostics"]["per_seed"].setdefault(role, {})
        for seed in seeds:
            sub = cap_df[(cap_df.role == role) & (cap_df.seed == seed)]
            wide_all = sub.pivot(
                index="capsule_idx", columns="condition", values="mean"
            )
            wide = wide_all.dropna()
            n_before = int(len(wide_all))
            n_after = int(len(wide))
            out["filtering"]["per_cell"].append(
                {
                    "role": role,
                    "seed": int(seed),
                    "n_capsules_before_dropna": n_before,
                    "n_capsules_after_dropna": n_after,
                    "n_dropped_incomplete": n_before - n_after,
                }
            )
            w = wide["within"].values
            c = wide["cross"].values
            if len(w) < 3:
                out["filtering"]["cells_skipped_min_n"].append(
                    {"role": role, "seed": int(seed), "n_paired_capsules": int(len(w))}
                )
                continue
            stat, p = sp_stats.wilcoxon(w, c)
            gap, lo, hi, bca_diag = bca_bootstrap_ci(w, c, seed=seed)
            out["bca_diagnostics"]["per_seed"][role][int(seed)] = bca_diag
            eff = rank_biserial_matched(w - c)
            out["per_seed"][role][int(seed)] = {
                "n_capsules": int(len(w)),
                "within_mean": float(w.mean()),
                "within_sd": float(w.std(ddof=1)),
                "cross_mean": float(c.mean()),
                "cross_sd": float(c.std(ddof=1)),
                "domain_gap": gap,
                "domain_gap_95ci": [lo, hi],
                "wilcoxon_stat": float(stat),
                "wilcoxon_p": float(p),
                "rank_biserial_r": eff,
            }
            pooled_within.append(w)
            pooled_cross.append(c)

        w_all = np.concatenate(pooled_within)
        c_all = np.concatenate(pooled_cross)
        stat, p = sp_stats.wilcoxon(w_all, c_all)
        gap, lo, hi, bca_diag = bca_bootstrap_ci(w_all, c_all, seed=0)
        out["bca_diagnostics"]["pooled"][role] = bca_diag
        eff = rank_biserial_matched(w_all - c_all)
        out["pooled"][role] = {
            "n_capsules_pooled_across_seeds": int(len(w_all)),
            "within_mean": float(w_all.mean()),
            "within_sd": float(w_all.std(ddof=1)),
            "cross_mean": float(c_all.mean()),
            "cross_sd": float(c_all.std(ddof=1)),
            "domain_gap": gap,
            "domain_gap_95ci": [lo, hi],
            "wilcoxon_stat": float(stat),
            "wilcoxon_p": float(p),
            "rank_biserial_r": eff,
            "note": f"Pools capsules across all {len(seeds)} trained models as additional "
            "replicates of the same role; see per_seed for the model-level breakdown.",
        }
    return out


def fit_clustered_lpm(df):
    """OLS linear-probability model with two-way (capsule, image) cluster-robust SEs.
    Written under the artifact key `primary_model`."""
    model_df = design_frame(df)
    model_df["_capsule_uid"], model_df["_image_uid"] = cluster_codes(model_df)

    formula = "correct ~ C(role) * C(condition) + C(test_digit) + C(seed)"
    ols_res = smf.ols(formula, data=model_df).fit()

    cov_both, cov_capsule, cov_image = cov_cluster_2groups(
        ols_res, model_df["_capsule_uid"].values, model_df["_image_uid"].values
    )
    se_both = np.sqrt(np.diag(cov_both))

    params = ols_res.params
    tvals = params.values / se_both
    pvals = 2 * sp_stats.norm.sf(np.abs(tvals))
    ci_lo = params.values - 1.96 * se_both
    ci_hi = params.values + 1.96 * se_both

    coef_table = []
    for name, b, se, t, p, lo, hi in zip(
        params.index, params.values, se_both, tvals, pvals, ci_lo, ci_hi
    ):
        coef_table.append(
            {
                "term": name,
                "coef": float(b),
                "cluster_robust_se": float(se),
                "z": float(t),
                "p_value": float(p),
                "ci_95": [float(lo), float(hi)],
            }
        )

    # Per-cell mean prediction. xbar is the covariate mean within the cell, so the
    # prediction equals the cell's raw rate (the fixed effects do not adjust it); the
    # SE is the delta-method SE of that linear combination under the clustered cov.
    cells = []
    for role in ROLES:
        for condition in ["within", "cross"]:
            mask = (model_df.role == role) & (model_df.condition == condition)
            sub_exog = ols_res.model.exog[mask.values]
            pred = sub_exog @ ols_res.params.values
            xbar = sub_exog.mean(axis=0)
            var_pred = xbar @ cov_both @ xbar.T
            se_pred = float(np.sqrt(var_pred))
            mean_pred = float(pred.mean())
            cells.append(
                {
                    "role": role,
                    "condition": condition,
                    "cell_mean_unadjusted": mean_pred,
                    "predicted_prob": mean_pred,
                    "se": se_pred,
                    "ci_95": [mean_pred - 1.96 * se_pred, mean_pred + 1.96 * se_pred],
                    "raw_mean": float(model_df.loc[mask, "correct"].mean()),
                    "n": int(mask.sum()),
                }
            )

    # Interaction-term SEs under each variance assumption; no refit needed.
    se_variants = {
        "iid_nonrobust": np.sqrt(np.diag(ols_res.cov_params().values)),
        "hc1": np.sqrt(
            np.diag(ols_res.get_robustcov_results(cov_type="HC1").cov_params())
        ),
        "cluster_capsule": np.sqrt(np.diag(cov_capsule)),
        "cluster_image": np.sqrt(np.diag(cov_image)),
        "cluster_two_way_capsule_image": se_both,
    }
    interaction_idx = [i for i, name in enumerate(params.index) if ":" in name]
    se_sensitivity = []
    for i in interaction_idx:
        entry = {"term": str(params.index[i]), "coef": float(params.values[i])}
        for label, ses in se_variants.items():
            se_i = float(ses[i])
            z_i = float(params.values[i] / se_i)
            entry[label] = {
                "se": se_i,
                "ratio_vs_iid": float(se_i / se_variants["iid_nonrobust"][i]),
                "z": z_i,
                "p_value": float(2 * sp_stats.norm.sf(abs(z_i))),
            }
        se_sensitivity.append(entry)

    return {
        "formula": formula,
        "n_obs": int(len(model_df)),
        "n_capsule_clusters": int(model_df["_capsule_uid"].nunique()),
        "n_image_clusters": int(model_df["_image_uid"].nunique()),
        "coefficients": coef_table,
        "predicted_cells": cells,
        "r_squared": float(ols_res.rsquared),
        "se_sensitivity_by_clustering": {
            "reported": "cluster_two_way_capsule_image",
            "why": (
                "The primary SEs are two-way cluster-robust by capsule and "
                "image; these are the same coefficients under weaker and "
                "alternative variance assumptions."
            ),
            "terms": se_sensitivity,
        },
    }


def fit_clustered_logit(df, include_seed_fe=True):
    """Logit with the LPM's design and two-way cluster-robust (CGM) SEs, plus the
    model-based role contrasts. Written under `ceiling_robust_model`.

    include_seed_fe=False fits a single seed's records, where the seed effect has one
    level.
    """
    model_df = design_frame(df)
    g_cap, g_img = cluster_codes(model_df)

    formula = (
        "correct ~ C(role) * C(condition) + C(test_digit) + C(seed)"
        if include_seed_fe
        else "correct ~ C(role) * C(condition) + C(test_digit)"
    )
    base = smf.logit(formula, data=model_df).fit(disp=0)
    cov, n_neg = two_way_logit_cov(base, g_cap, g_img)
    se = np.sqrt(np.diag(cov))

    params = base.params
    zvals = params.values / se
    pvals = 2 * sp_stats.norm.sf(np.abs(zvals))

    coef_table = [
        {
            "term": name,
            "coef": float(b),
            "odds_ratio": float(np.exp(b)),
            "cluster_robust_se": float(s),
            "z": float(z),
            "p_value": float(p),
            "ci_95": [float(b - 1.96 * s), float(b + 1.96 * s)],
        }
        for name, b, s, z, p in zip(params.index, params.values, se, zvals, pvals)
    ]

    def logit(p):
        return float(np.log(p / (1 - p)))

    # Raw cell rates, descriptive only: unadjusted for test_digit and seed.
    gaps = []
    for role in ROLES:
        w = model_df.loc[
            (model_df.role == role) & (model_df.condition == "within"), "correct"
        ].mean()
        c = model_df.loc[
            (model_df.role == role) & (model_df.condition == "cross"), "correct"
        ].mean()
        gaps.append(
            {
                "role": role,
                "within_rate": float(w),
                "cross_rate": float(c),
                "additive_gap_pp": float((w - c) * 100),
                "raw_log_odds_gap_descriptive_only": logit(w) - logit(c),
            }
        )

    # With 'random' as reference, the interaction coefficients are each role's excess
    # domain gap in log-odds; hub vs specialist is their difference.
    names = list(params.index)
    i_hub = names.index("C(role)[T.hub]:C(condition)[T.within]")
    i_spec = names.index("C(role)[T.specialist]:C(condition)[T.within]")

    def _contrast(weights, label):
        """Linear combination c'beta with its two-way cluster-robust SE."""
        c = np.zeros(len(names))
        for idx, w_ in weights:
            c[idx] = w_
        est = float(c @ params.values)
        var = float(c @ cov @ c)
        se_c = float(np.sqrt(var)) if var > 0 else float("nan")
        z_c = est / se_c if se_c == se_c and se_c > 0 else float("nan")
        p_c = float(2 * sp_stats.norm.sf(abs(z_c))) if z_c == z_c else float("nan")
        return {
            "contrast": label,
            "estimate_log_odds": est,
            "cluster_robust_se": se_c,
            "z": float(z_c),
            "p_value": p_c,
            "ci_95": [est - 1.96 * se_c, est + 1.96 * se_c],
        }

    contrasts = [
        _contrast([(i_hub, 1.0)], "hub vs random (excess domain gap, log-odds)"),
        _contrast(
            [(i_spec, 1.0)], "specialist vs random (excess domain gap, log-odds)"
        ),
        _contrast(
            [(i_hub, 1.0), (i_spec, -1.0)],
            "hub vs specialist (excess domain gap, log-odds)",
        ),
    ]

    hub_v_rand, spec_v_rand, hub_v_spec = contrasts
    ordering_supported = (
        hub_v_rand["estimate_log_odds"] > 0
        and hub_v_rand["p_value"] < 0.05
        and spec_v_rand["estimate_log_odds"] > 0
        and spec_v_rand["p_value"] < 0.05
        and hub_v_spec["estimate_log_odds"] > 0
        and hub_v_spec["p_value"] < 0.05
    )
    ordering = [
        g["role"]
        for g in sorted(gaps, key=lambda g: -g["raw_log_odds_gap_descriptive_only"])
    ]
    return {
        "role_contrasts_model_based": contrasts,
        "ordering_hub_gt_specialist_gt_random_supported_by_model": bool(
            ordering_supported
        ),
        "ordering_note": (
            "The ordering is evaluated from the three model-based contrasts above, "
            "each with a two-way cluster-robust SE. `log_odds_ordering` below is the "
            "descriptive ordering of the raw cell rates."
        ),
        "formula": formula,
        "link": "logit",
        "cluster_robust": "two-way (Cameron-Gelbach-Miller): capsule + image - intersection",
        "n_obs": int(len(model_df)),
        "coefficients": coef_table,
        "role_gaps": gaps,
        "log_odds_ordering": ordering,
        "ordering_matches_mechanism": ordering == ["hub", "specialist", "random"],
        "pseudo_r_squared": float(base.prsquared),
        "seed_fixed_effect_included": bool(include_seed_fe),
        "psd_projection_applied": bool(n_neg),
        "psd_negative_eigenvalues": n_neg,
    }


def per_seed_logit_fits(df, pooled):
    """The logit refitted on each seed alone, and per contrast how the seeds compare
    with the pooled estimate. Returns (per_seed_fits, between_model_heterogeneity)."""
    per_seed = {}
    for seed in sorted(df.seed.unique().tolist()):
        fit = fit_clustered_logit(df[df.seed == seed], include_seed_fe=False)
        per_seed[int(seed)] = {
            "n_obs": fit["n_obs"],
            "formula": fit["formula"],
            "role_contrasts_model_based": fit["role_contrasts_model_based"],
            "ordering_hub_gt_specialist_gt_random_supported_by_model": fit[
                "ordering_hub_gt_specialist_gt_random_supported_by_model"
            ],
            "psd_projection_applied": fit["psd_projection_applied"],
            "psd_negative_eigenvalues": fit["psd_negative_eigenvalues"],
        }
        for c in fit["role_contrasts_model_based"]:
            print(
                f"    seed {seed}  {c['contrast']:<52} b={c['estimate_log_odds']:+.3f} "
                f"[{c['ci_95'][0]:+.3f},{c['ci_95'][1]:+.3f}] p={c['p_value']:.3g}"
            )

    heterogeneity = []
    for i, pooled_c in enumerate(pooled["role_contrasts_model_based"]):
        label = pooled_c["contrast"]
        pooled_est = pooled_c["estimate_log_odds"]
        ests = [
            per_seed[s]["role_contrasts_model_based"][i]["estimate_log_odds"]
            for s in sorted(per_seed)
        ]
        heterogeneity.append(
            {
                "contrast": label,
                "pooled_estimate_log_odds": pooled_est,
                "per_seed_estimates_log_odds": {
                    str(s): per_seed[s]["role_contrasts_model_based"][i][
                        "estimate_log_odds"
                    ]
                    for s in sorted(per_seed)
                },
                "per_seed_significant_at_05": {
                    str(s): bool(
                        per_seed[s]["role_contrasts_model_based"][i]["p_value"] < 0.05
                    )
                    for s in sorted(per_seed)
                },
                "n_models_same_sign_as_pooled": int(
                    sum(1 for e in ests if (e > 0) == (pooled_est > 0))
                ),
                "n_models": len(ests),
                "unweighted_mean_of_per_seed_estimates": float(np.mean(ests)),
                "pooled_within_per_seed_range": bool(
                    min(ests) <= pooled_est <= max(ests)
                ),
            }
        )
        print(
            f"  {label}: pooled={pooled_est:+.3f}, per-seed={[round(e, 3) for e in ests]}, "
            f"same sign as pooled in {heterogeneity[-1]['n_models_same_sign_as_pooled']}"
            f"/{len(ests)} models"
        )
    return per_seed, heterogeneity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--records", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument(
        "--force", action="store_true", help="write the canonical output from --records"
    )
    args = ap.parse_args()
    cfg = DATASETS[args.dataset]
    out_dir = RECORDS[args.dataset]["output_dir"]
    records = args.records or f"{out_dir}/symmetric_raw_records.parquet"
    output = resolve_output(
        ap, args, f"{out_dir}/symmetric_evaluation_glmm.json", ["records"]
    )

    print("Loading raw records...")
    df = load_records(records)
    print(
        f"  {len(df):,} rows, seeds={sorted(df.seed.unique().tolist())}, "
        f"roles={sorted(df.role.unique().tolist())}"
    )

    print("Running companion (capsule-level, paired) analysis...")
    companion = companion_analysis(capsule_level_table(df))
    for role in ROLES:
        p = companion["pooled"][role]
        print(
            f"  [pooled] {role}: within={p['within_mean']:.3f} cross={p['cross_mean']:.3f} "
            f"gap={p['domain_gap']:.3f} 95%CI=[{p['domain_gap_95ci'][0]:.3f},{p['domain_gap_95ci'][1]:.3f}] "
            f"Wilcoxon p={p['wilcoxon_p']:.2e} r={p['rank_biserial_r']:.3f}"
        )

    print("\nFitting the clustered linear-probability model...")
    lpm = fit_clustered_lpm(df)
    print(
        f"  n_obs={lpm['n_obs']:,}  capsule_clusters={lpm['n_capsule_clusters']}  "
        f"image_clusters={lpm['n_image_clusters']}  R^2={lpm['r_squared']:.3f}"
    )
    for c in lpm["predicted_cells"]:
        print(
            f"  {c['role']:<10} {c['condition']:<7} p={c['predicted_prob']:.3f} "
            f"95%CI=[{c['ci_95'][0]:.3f},{c['ci_95'][1]:.3f}] (raw mean={c['raw_mean']:.3f}, n={c['n']:,})"
        )

    print("\nFitting the clustered logit model...")
    logit_m = fit_clustered_logit(df)
    for g in logit_m["role_gaps"]:
        print(
            f"  {g['role']:<11} within={g['within_rate']*100:6.2f}%  cross={g['cross_rate']*100:6.2f}%  "
            f"additive={g['additive_gap_pp']:6.2f}pp  "
            f"raw log-odds={g['raw_log_odds_gap_descriptive_only']:6.3f} (descriptive)"
        )
    for c in logit_m["role_contrasts_model_based"]:
        print(
            f"    {c['contrast']:<52} b={c['estimate_log_odds']:+.3f} "
            f"[{c['ci_95'][0]:+.3f},{c['ci_95'][1]:+.3f}] p={c['p_value']:.3g}"
        )
    print(
        f"  ordering hub>specialist>random supported by model: "
        f"{logit_m['ordering_hub_gt_specialist_gt_random_supported_by_model']}"
    )

    print("\nFitting the logit model separately per seed...")
    logit_m["per_seed_fits"], logit_m["between_model_heterogeneity"] = (
        per_seed_logit_fits(df, logit_m)
    )

    random_cells = {
        c["condition"]: c for c in lpm["predicted_cells"] if c["role"] == "random"
    }

    output_doc = {
        "description": "Clustering-aware analysis of symmetric Hub/Specialist routing "
        f"correctness on {cfg['name']}.",
        "seeds": sorted(df.seed.unique().tolist()),
        cfg["samples_key"]: SAMPLES_PER_CLASS,
        "primary_model": {
            "type": "linear probability model (OLS on binary correct outcome), "
            "two-way cluster-robust SEs (capsule, image); test_digit and seed as fixed effects",
            "note_on_seed_clustering": "seed (trained model) entered as a fixed effect rather than a "
            "third cluster dimension because there are too few seed levels for "
            "cluster-robust asymptotics. "
            "Per-seed breakdowns (below / companion_analysis.per_seed) show "
            "the trained-model dimension directly instead.",
            **lpm,
        },
        "ceiling_robust_model": {
            "type": "logistic regression on the binary correct outcome, same design "
            "as the linear probability model under primary_model, two-way "
            "cluster-robust SEs (Cameron-Gelbach-Miller)",
            "why": "On the additive scale a role x condition interaction near a ceiling is "
            "bounded by the baseline rate; the logit link models log-odds, where it is not.",
            **logit_m,
        },
        "companion_analysis_method": {
            "bootstrap": "BCa (bias-corrected and accelerated), resampling capsules",
            "n_resamples": N_BOOT,
            "ci_level": 0.95,
            "rng_seed_pooled": 0,
            "rng_seed_per_seed": "the seed id itself",
            "paired_test": "Wilcoxon signed-rank, matched by capsule",
            "effect_size": "matched-pairs rank-biserial correlation",
        },
        "companion_analysis_capsule_level": companion,
        "matched_empirical_random_baseline": {
            "description": "Matched empirical control: the random role uses the identical "
            "protocol (same image pools, same routing-correctness definition) as "
            "Hub/Specialist, sampled from the taxonomy's non-critical capsules.",
            "within_domain": random_cells.get("within"),
            "cross_domain": random_cells.get("cross"),
        },
    }

    attach(
        output_doc,
        script="experiments/symmetric_analysis.py",
        inputs=[records],
        seeds={
            "bca_pooled": 0,
            "bca_per_seed": "the seed id itself",
            "note": "BCa resampling seeds only; the model fits are deterministic.",
        },
        sample_counts={
            "n_obs": lpm["n_obs"],
            "n_capsule_clusters": lpm.get("n_capsule_clusters"),
            "n_image_clusters": lpm.get("n_image_clusters"),
            "capsules_per_role_pooled": {
                r: companion["pooled"][r]["n_capsules_pooled_across_seeds"]
                for r in companion["pooled"]
            },
            "complete_case_filtering": companion.get("filtering"),
        },
        method=(
            "Additive model (artifact key primary_model): OLS linear-probability "
            "model correct ~ C(role)*C(condition) + C(test_digit) + C(seed) with "
            "two-way cluster-robust SEs (Cameron-Gelbach-Miller: capsule + image - "
            "intersection). Logit model (artifact key ceiling_robust_model): "
            "clustered logit, same design, plus per-seed refits for between-model "
            "heterogeneity. Model-free companion: paired "
            "Wilcoxon signed-rank matched by capsule, matched-pairs rank-biserial "
            "effect size, and BCa bootstrap 95% CIs resampling CAPSULES over "
            f"{N_BOOT} resamples."
        ),
        packages=["numpy", "scipy", "statsmodels", "pandas"],
    )

    # symmetric_digit_confound.py adds its block to this file afterwards.
    out_path = PROJECT_ROOT / output
    with open(out_path, "w") as f:
        json.dump(output_doc, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
