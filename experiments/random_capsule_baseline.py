#!/usr/bin/env python3
"""Role-label permutation null for the Hub and Specialist within-vs-cross domain
gaps: each role is compared with same-size groups of other capsules of the same
models (four pool variants), plus a descriptive all-roles BCa companion table.

Usage:
    python experiments/random_capsule_baseline.py --dataset mnist
    python experiments/random_capsule_baseline.py --dataset cifar10
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

from capsnet import PROJECT_ROOT
from capsnet.data import slice_for, test_set
from capsnet.io import get_device, load_capsule_model, resolve_output
from capsnet.provenance import attach
from capsnet.stats import N_BOOT, bca_bootstrap_ci
from capsnet.symmetric import (
    RECORDS,
    SAMPLES_PER_CLASS,
    batched_couplings,
    class_pools,
    model_path,
    per_capsule_correctness,
    taxonomy_paths,
)
from capsnet.taxonomy import ROLE_ORDER

# dataset -> its id in the null RNG
DATASET_ID = {"mnist": 0, "cifar10": 1}

ROLES = list(ROLE_ORDER)

# role compared -> (rng id, taxonomy roles it contains)
GROUPS = {
    "hubs": (0, ["primary_hubs", "secondary_hubs"]),
    "specialists": (1, ["specialists"]),
}

# variant -> (rng id, pool definition)
VARIANTS = {
    "outside_role": (0, "every capsule of the model not in the compared role"),
    "generalist_noncritical": (1, "Generalists and Non-critical capsules only"),
    "home_matched": (
        2,
        "outside_role pool, drawn per home class to equal the role's home-class "
        "counts in that model",
    ),
    "excl_nonpositive_importance": (
        3,
        "outside_role with capsules whose importance column is all <= 0 removed "
        "from the pool and from the observed group",
    ),
}
PRIMARY_VARIANT = "outside_role"

COMPANION = {
    "primary_hubs": ["primary_hubs"],
    "secondary_hubs": ["secondary_hubs"],
    "specialists": ["specialists"],
    "generalists": ["generalists"],
    "non_critical": ["non_critical"],
    "hubs": ["primary_hubs", "secondary_hubs"],
    "whole_model": ROLES,
}

STATS = ["within_pct", "cross_pct", "gap_pp", "logit_gap"]


def evaluate_model(dataset_name, seed, dataset, pools, device) -> dict:
    """Per-capsule within / cross correctness, home and role for every primary capsule."""
    model = load_capsule_model(
        str(model_path(dataset_name, seed)), dataset_name, device=device
    )
    tax_p, imp_p = taxonomy_paths(dataset_name, seed)
    taxonomy = json.loads(tax_p.read_text())
    importance = np.load(imp_p)

    batch_size = RECORDS[dataset_name]["batch_size"]
    stack = np.stack(
        [
            batched_couplings(model, dataset, pools[d], device, batch_size)
            for d in range(10)
        ],
        axis=0,
    )  # [shown class, image, class capsule, primary capsule]
    n_caps = stack.shape[3]
    if stack.shape[:3] != (10, SAMPLES_PER_CLASS, 10) or importance.shape != (
        10,
        n_caps,
    ):
        raise ValueError(
            f"unexpected shapes: stack {stack.shape}, importance {importance.shape}"
        )

    within, cross, home = per_capsule_correctness(stack, importance, np.arange(n_caps))
    del stack

    role = np.full(n_caps, "", dtype=object)
    for r in ROLES:
        for c in taxonomy[r]:
            if role[int(c)] != "":
                raise ValueError(f"seed {seed}: capsule {c} is in two roles")
            role[int(c)] = r
    if (role == "").any():
        raise ValueError(f"seed {seed}: the five roles do not cover every capsule")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return dict(
        within=within,
        cross=cross,
        home=home.astype(int),
        role=role.astype(str),
        nonpos=(importance <= 0).all(axis=0),
    )


def logit(p: float) -> float:
    return float(np.log(p / (1 - p))) if 0 < p < 1 else float("nan")


def group_stats(w, c, wt=None):
    """(within, cross, gap, logit gap) of a group, pooled with weights wt (default equal)."""
    if wt is None:
        wt = np.ones_like(w)
    W = float(np.sum(wt * w) / np.sum(wt))
    C = float(np.sum(wt * c) / np.sum(wt))
    return W, C, W - C, logit(W) - logit(C)


def as_named(st) -> dict:
    W, C, G, L = st
    return {
        "within_pct": 100 * W,
        "cross_pct": 100 * C,
        "gap_pp": 100 * G,
        "logit_gap": L,
    }


def summarise(obs: float, null, n_draws: int) -> dict:
    null = np.asarray(null, dtype=float)
    k_low = int((null <= obs).sum())
    k_high = int((null >= obs).sum())
    p_low = (1 + k_low) / (n_draws + 1)
    p_high = (1 + k_high) / (n_draws + 1)
    lo, hi = np.percentile(null, [2.5, 97.5])
    return {
        "observed": float(obs),
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "null_p2_5": float(lo),
        "null_p97_5": float(hi),
        "observed_minus_null_mean": float(obs - null.mean()),
        "observed_percentile_in_null": float(
            100 * (null < obs).mean() + 50 * (null == obs).mean()
        ),
        "n_draws_le_observed": k_low,
        "n_draws_ge_observed": k_high,
        "p_one_sided_lower": p_low,
        "p_one_sided_upper": p_high,
        "p_two_sided": min(1.0, 2 * min(p_low, p_high)),
    }


def summarise_all(obs_named: dict, null_arr: np.ndarray, n_draws: int) -> dict:
    scale = {"within_pct": 100, "cross_pct": 100, "gap_pp": 100, "logit_gap": 1}
    col = {"within_pct": 0, "cross_pct": 1, "gap_pp": 2, "logit_gap": 3}
    return {
        s: summarise(obs_named[s], scale[s] * null_arr[:, col[s]], n_draws)
        for s in STATS
    }


def null_cell(per_seed, seeds, ds_id, group, variant, base_seed, n_draws) -> dict:
    group_id, roles_in = GROUPS[group]
    variant_id, pool_definition = VARIANTS[variant]
    rng = np.random.default_rng([base_seed, ds_id, group_id, variant_id])
    excl_nonpos = variant == "excl_nonpositive_importance"
    matched = variant == "home_matched"

    cells = {}
    feas = {}
    for s in seeds:
        m = per_seed[s]
        keep = ~m["nonpos"] if excl_nonpos else np.ones(len(m["role"]), dtype=bool)
        in_role = np.isin(m["role"], roles_in)
        g_idx = np.flatnonzero(keep & in_role)
        if variant == "generalist_noncritical":
            p_idx = np.flatnonzero(
                keep & np.isin(m["role"], ["generalists", "non_critical"])
            )
        else:
            p_idx = np.flatnonzero(keep & ~in_role)
        need = np.bincount(m["home"][g_idx], minlength=10)
        have = np.bincount(m["home"][p_idx], minlength=10)
        cells[s] = dict(
            g=g_idx,
            p=p_idx,
            ph={h: p_idx[m["home"][p_idx] == h] for h in range(10)},
            need=need,
        )
        feas[s] = {
            "need_by_home": need.tolist(),
            "available_by_home": have.tolist(),
            "infeasible_classes": [
                {"class": h, "need": int(need[h]), "available": int(have[h])}
                for h in range(10)
                if need[h] > have[h]
            ],
            "min_available_over_needed": float(
                min(have[h] / need[h] for h in range(10) if need[h] > 0)
            ),
        }
        if matched and any(need[h] > 0 and have[h] == 0 for h in range(10)):
            raise ValueError(
                f"seed {s}: a home class the role needs has no pool capsule"
            )

    g_w = np.concatenate([per_seed[s]["within"][cells[s]["g"]] for s in seeds])
    g_c = np.concatenate([per_seed[s]["cross"][cells[s]["g"]] for s in seeds])
    obs = as_named(group_stats(g_w, g_c))
    obs_seed = {
        s: as_named(
            group_stats(
                per_seed[s]["within"][cells[s]["g"]],
                per_seed[s]["cross"][cells[s]["g"]],
            )
        )
        for s in seeds
    }

    stats = np.empty((n_draws, 4))
    seed_stats = {s: np.empty((n_draws, 4)) for s in seeds}
    comp = {r: np.empty(n_draws) for r in ROLES}
    for b in range(n_draws):
        ws, cs, wts, rs = [], [], [], []
        for s in seeds:
            cell, m = cells[s], per_seed[s]
            if not matched:
                sel = cell["p"][
                    rng.choice(len(cell["p"]), size=len(cell["g"]), replace=False)
                ]
                wt = np.ones(len(sel))
            else:
                parts, wparts = [], []
                for h in range(10):
                    n_h = int(cell["need"][h])
                    if n_h == 0:
                        continue
                    ph = cell["ph"][h]
                    k = min(n_h, len(ph))  # k < n_h only in an infeasible class
                    parts.append(ph[rng.choice(len(ph), size=k, replace=False)])
                    wparts.append(np.full(k, n_h / k))
                sel = np.concatenate(parts)
                wt = np.concatenate(wparts)
            seed_stats[s][b] = group_stats(m["within"][sel], m["cross"][sel], wt)
            ws.append(m["within"][sel])
            cs.append(m["cross"][sel])
            wts.append(wt)
            rs.append(m["role"][sel])
        stats[b] = group_stats(
            np.concatenate(ws), np.concatenate(cs), np.concatenate(wts)
        )
        rr = np.concatenate(rs)
        for r in ROLES:
            comp[r][b] = (rr == r).mean()

    res = {
        "role": group,
        "variant": variant,
        "pool_definition": pool_definition,
        "rng": f"numpy default_rng([{base_seed}, {ds_id}, {group_id}, {variant_id}])",
        "n_draws": n_draws,
        "n_units_observed": int(len(g_w)),
        "n_by_seed": {str(s): int(len(cells[s]["g"])) for s in seeds},
        "pool_size_by_seed": {str(s): int(len(cells[s]["p"])) for s in seeds},
        "pooled": summarise_all(obs, stats, n_draws),
        "per_seed": {
            str(s): summarise_all(obs_seed[s], seed_stats[s], n_draws) for s in seeds
        },
        "drawn_group_mean_role_share": {r: float(comp[r].mean()) for r in ROLES},
    }
    if matched:
        infeasible_by_seed = {
            str(s): bool(feas[s]["infeasible_classes"]) for s in seeds
        }
        any_infeasible = any(infeasible_by_seed.values())
        res["home_matching_feasibility"] = {str(s): feas[s] for s in seeds}
        res["feasible_by_seed"] = {s: not v for s, v in infeasible_by_seed.items()}
        res["feasible"] = not any_infeasible
        res["estimate"] = "post_stratified" if any_infeasible else "exact_home_matched"
        res["estimate_by_seed"] = {
            s: ("post_stratified" if v else "exact_home_matched")
            for s, v in infeasible_by_seed.items()
        }
        if any_infeasible:
            res["label"] = (
                "INFEASIBLE as an exact home-count-matched draw. The pooled numbers, and "
                "the per-seed numbers of every seed marked post_stratified, are a "
                "POST-STRATIFIED estimate: in each short home class every available pool "
                "capsule is taken in every draw and weighted need/available, so the group "
                "carries the role's home-class composition; that class contributes no "
                "draw-to-draw variability, so the null band is slightly too narrow."
            )
    return res


def bca_triplet(w, c, seed) -> dict:
    """Pooled within / cross / gap with capsule-level BCa CIs (same rng seed for all three)."""
    gap, glo, ghi, gd = bca_bootstrap_ci(w, c, seed=seed)
    z = np.zeros_like(w)
    wm, wlo, whi, wd = bca_bootstrap_ci(w, z, seed=seed)
    cm, clo, chi, cd = bca_bootstrap_ci(c, z, seed=seed)
    diag = {"gap": gd, "within": wd, "cross": cd}
    flagged = any(
        d["z0_degenerate"] or d["quantile_clipped_low"] or d["quantile_clipped_high"]
        for d in diag.values()
    )
    d = w - c
    return {
        "n": int(len(w)),
        "within_pct": 100 * wm,
        "within_ci95": [100 * wlo, 100 * whi],
        "cross_pct": 100 * cm,
        "cross_ci95": [100 * clo, 100 * chi],
        "gap_pp": 100 * gap,
        "gap_ci95": [100 * glo, 100 * ghi],
        "logit_gap": logit(float(w.mean())) - logit(float(c.mean())),
        "n_gap_le_0": int((d <= 0).sum()),
        "n_gap_lt_0": int((d < 0).sum()),
        "n_gap_eq_0": int((d == 0).sum()),
        "min_gap_pp": float(100 * d.min()),
        "bca_any_degenerate_or_clipped": bool(flagged),
        "bca_diagnostics": diag,
    }


def companion(per_seed, seeds) -> dict:
    out = {}
    for name, roles in COMPANION.items():
        sel = {s: np.flatnonzero(np.isin(per_seed[s]["role"], roles)) for s in seeds}
        ent = {
            "roles": roles,
            "per_seed": {
                str(s): bca_triplet(
                    per_seed[s]["within"][sel[s]], per_seed[s]["cross"][sel[s]], seed=s
                )
                for s in seeds
            },
        }
        w = np.concatenate([per_seed[s]["within"][sel[s]] for s in seeds])
        c = np.concatenate([per_seed[s]["cross"][sel[s]] for s in seeds])
        ent["pooled"] = bca_triplet(w, c, seed=0)
        out[name] = ent
    return out


def json_safe(x):
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", required=True, choices=sorted(DATASET_ID))
    ap.add_argument(
        "--n_draws", type=int, default=2000, help="random groups per null cell"
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=20261111,
        help="base of the null RNG; each cell uses default_rng([seed, dataset, role, variant])",
    )
    ap.add_argument(
        "--output", default=None, help="default: the per-dataset artifact path"
    )
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="trained-model slots (default: the dataset's canonical three); 42 is the "
        "primary checkpoint",
    )
    ap.add_argument(
        "--force", action="store_true", help="write the canonical output from --seeds"
    )
    args = ap.parse_args()

    seeds = sorted(
        args.seeds if args.seeds is not None else RECORDS[args.dataset]["seeds"]
    )
    output = resolve_output(
        ap,
        args,
        f"{RECORDS[args.dataset]['output_dir']}/random_capsule_baseline.json",
        ["seeds"],
    )

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = get_device()
    print(f"dataset {args.dataset}  seeds {seeds}  device {device}")

    dataset = test_set(args.dataset)
    pools = class_pools(dataset, args.dataset, SAMPLES_PER_CLASS)
    eval_slice = list(slice_for(args.dataset, "evaluation", SAMPLES_PER_CLASS))

    per_seed = {}
    for s in seeds:
        t0 = time.time()
        per_seed[s] = evaluate_model(args.dataset, s, dataset, pools, device)
        print(
            f"  model {s}: {len(per_seed[s]['role'])} capsules in {time.time() - t0:.1f}s"
        )

    role_sizes = {
        str(s): {r: int((per_seed[s]["role"] == r).sum()) for r in ROLES} for s in seeds
    }
    nonpos = {
        "per_seed": {
            str(s): {
                "n": int(per_seed[s]["nonpos"].sum()),
                "by_role": {
                    r: int((per_seed[s]["nonpos"] & (per_seed[s]["role"] == r)).sum())
                    for r in ROLES
                },
            }
            for s in seeds
        },
        "total": int(sum(per_seed[s]["nonpos"].sum() for s in seeds)),
    }

    observed = {}
    for group, (_, roles_in) in GROUPS.items():
        sel = {s: np.flatnonzero(np.isin(per_seed[s]["role"], roles_in)) for s in seeds}
        w = np.concatenate([per_seed[s]["within"][sel[s]] for s in seeds])
        c = np.concatenate([per_seed[s]["cross"][sel[s]] for s in seeds])
        observed[group] = {
            "pooled": {"n": int(len(w)), **as_named(group_stats(w, c))},
            "per_seed": {
                str(s): {
                    "n": int(len(sel[s])),
                    **as_named(
                        group_stats(
                            per_seed[s]["within"][sel[s]], per_seed[s]["cross"][sel[s]]
                        )
                    ),
                }
                for s in seeds
            },
        }

    null = {}
    for group in GROUPS:
        null[group] = {}
        for variant in VARIANTS:
            t0 = time.time()
            r = null_cell(
                per_seed,
                seeds,
                DATASET_ID[args.dataset],
                group,
                variant,
                args.seed,
                args.n_draws,
            )
            null[group][variant] = r
            g = r["pooled"]["gap_pp"]
            lg = r["pooled"]["logit_gap"]
            print(
                f"  {group:<12} {variant:<28} gap {g['observed']:.2f} vs "
                f"{g['null_mean']:.2f} [{g['null_p2_5']:.2f}, {g['null_p97_5']:.2f}] "
                f"p2 {g['p_two_sided']:.4f} | logit {lg['observed']:.3f} vs "
                f"{lg['null_mean']:.3f} p2 {lg['p_two_sided']:.4f}  ({time.time() - t0:.1f}s)"
            )

    t0 = time.time()
    comp = companion(per_seed, seeds)
    print(f"  companion table in {time.time() - t0:.1f}s")

    out = {
        "description": (
            "Random-capsule (role-label permutation) baseline for the Hub and Specialist "
            "within-vs-cross domain gaps: each role is compared with same-size groups of "
            "other capsules of the same models, evaluated under the canonical symmetric "
            "protocol, plus a descriptive all-roles companion table."
        ),
        "dataset": args.dataset,
        "seeds": seeds,
        "samples_per_class": SAMPLES_PER_CLASS,
        "eval_slice": eval_slice,
        "n_primary_capsules_per_model": {
            str(s): len(per_seed[s]["role"]) for s in seeds
        },
        "role_sizes": role_sizes,
        "definitions": {
            "unit": "capsule x trained model",
            "home": "argmax over classes of the capsule's calibration importance column",
            "within": "per-capsule mean routing correctness over the images of its home class",
            "cross": "per-capsule mean routing correctness over the images of the other nine classes",
            "within_pct / cross_pct": "100 x mean over units, equal weight per unit across models",
            "gap_pp": "within_pct - cross_pct",
            "logit_gap": "logit(pooled mean within) - logit(pooled mean cross), group level",
            "p_one_sided_lower": "(1 + #{null <= observed}) / (n_draws + 1)",
            "p_one_sided_upper": "(1 + #{null >= observed}) / (n_draws + 1)",
            "p_two_sided": "min(1, 2 * min(p_one_sided_lower, p_one_sided_upper))",
            "drawn_group_mean_role_share": "mean over draws of the share of drawn units in each role",
            "hubs": "primary_hubs + secondary_hubs",
        },
        "primary_variant": PRIMARY_VARIANT,
        "importance_all_nonpositive": nonpos,
        "observed": observed,
        "random_capsule_null": null,
        "companion": comp,
    }
    attach(
        out,
        script="experiments/random_capsule_baseline.py",
        inputs=[str(model_path(args.dataset, s)) for s in seeds]
        + [str(p) for s in seeds for p in taxonomy_paths(args.dataset, s)],
        seeds={
            "--seed": args.seed,
            "null_rng": f"numpy default_rng([{args.seed}, dataset_id, role_id, variant_id]) per cell",
            "dataset_id": DATASET_ID,
            "role_id": {k: v[0] for k, v in GROUPS.items()},
            "variant_id": {k: v[0] for k, v in VARIANTS.items()},
            "bca_rng": "numpy default_rng(0) pooled, default_rng(model slot id) per model, "
            "inside capsnet.stats.bca_bootstrap_ci",
            "trained_model_seeds": seeds,
            "note": "The image pools are a deterministic slice; only the null draws and "
            "the BCa resampling consume RNGs.",
        },
        sample_counts={
            "n_draws": args.n_draws,
            "n_boot": N_BOOT,
            "samples_per_class": SAMPLES_PER_CLASS,
            "n_primary_capsules_per_model": out["n_primary_capsules_per_model"],
            "role_sizes": role_sizes,
        },
        method=(
            "Role-label permutation null: per model, same-size groups drawn without "
            "replacement from the capsules outside the role (plus three pool variants), "
            "pooled across models with equal weight per unit; null mean, SD, 2.5/97.5 "
            "percentiles and two-sided resampling p for within, cross, additive gap and "
            "group-level logit gap. Companion: capsule-level BCa 95% CIs for every role."
        ),
        packages=["numpy", "scipy", "torchvision"],
    )

    out_path = PROJECT_ROOT / output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(json_safe(out), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
