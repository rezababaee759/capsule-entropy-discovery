#!/usr/bin/env python3
"""Paired significance and equivalence tests between the arms of the VEG component
ablation, per dataset: paired t (primary), Wilcoxon signed-rank (companion), Cohen's d_z
and TOST against +-SESOI_PP; BH and Holm over one family of all contrasts.

    python experiments/accuracy_significance.py
"""

import argparse
import json
from itertools import combinations

import numpy as np
from scipy import stats as sp_stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.weightstats import ttost_paired

from capsnet import PROJECT_ROOT
from capsnet.io import resolve_output
from capsnet.provenance import attach

SESOI_PP = 0.5  # TOST equivalence margin, accuracy percentage points
DIFF_DECIMALS = 9  # resolution at which paired accuracy differences are ranked

ARM_LABEL = {
    "dynamic": "Baseline (Dynamic)",
    "entropy_uniform": "Uniform-beta",
    "entropy": "Per-Capsule-beta (ours)",
}


def adjust(pvals, method, alpha=0.05):
    """Return (adjusted p-values, reject flags) for the whole family."""
    reject, adjusted, _, _ = multipletests(
        np.asarray(pvals, dtype=float), alpha=alpha, method=method
    )
    return adjusted, reject


def nonzero_differences(diff):
    """The paired differences at DIFF_DECIMALS decimals, zeros dropped (Wilcoxon's rule)."""
    # Rounded far below one test image's share of accuracy, so that differences
    # equal as counts of images are equal as floats and tie.
    d = np.round(np.asarray(diff, dtype=float), DIFF_DECIMALS)
    return d[d != 0]


def wilcoxon_two_sided_floor(n_nonzero):
    """Smallest two-sided p the exact signed-rank test can return with n non-zero
    differences: 2 / 2**n."""
    return 2.0 / (2.0**n_nonzero)


def wilcoxon_exact_two_sided(diff):
    """Exact two-sided Wilcoxon signed-rank p of the paired differences `diff`.

    Zero differences are dropped, equal |diff| share their mean rank, and the null
    distribution is every sign assignment to the remaining ranks; 1.0 when every
    difference is zero.
    """
    d = nonzero_differences(diff)
    if d.size == 0:
        return 1.0
    # Doubled, so mean ranks of ties are integers and rank sums index a table.
    ranks = np.rint(2 * sp_stats.rankdata(np.abs(d))).astype(int)
    total = int(ranks.sum())
    n_assignments = [1] + [0] * total  # by rank sum of the positive differences
    for r in ranks:
        for s in range(total, r - 1, -1):
            n_assignments[s] += n_assignments[s - r]
    w_plus = int(ranks[d > 0].sum())
    tail = min(sum(n_assignments[: w_plus + 1]), sum(n_assignments[w_plus:]))
    return min(1.0, 2 * tail / 2**d.size)


def per_seed(data, dataset, arm):
    rows = data["results"][dataset]["raw"][arm]
    return {r["seed"]: r["best_acc"] for r in rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--ablation", default="results/veg_ablation/component_ablation.json"
    )
    ap.add_argument(
        "--output", help="default: results/veg_ablation/accuracy_significance.json"
    )
    ap.add_argument("--fdr_q", type=float, default=0.05)
    ap.add_argument(
        "--force",
        action="store_true",
        help="write the canonical output from another --ablation or --fdr_q",
    )
    args = ap.parse_args()
    output = resolve_output(
        ap,
        args,
        "results/veg_ablation/accuracy_significance.json",
        ["ablation", "fdr_q"],
    )

    data = json.loads((PROJECT_ROOT / args.ablation).read_text())
    seeds = sorted(data["seeds"])
    datasets = list(data["results"].keys())
    arms = data["strategies"]

    contrasts = []
    for ds in datasets:
        for a, b in combinations(arms, 2):
            sa, sb = per_seed(data, ds, a), per_seed(data, ds, b)
            common = [s for s in seeds if s in sa and s in sb]
            xa = np.array([sa[s] for s in common], dtype=float)
            xb = np.array([sb[s] for s in common], dtype=float)
            diff = xb - xa  # positive => arm b is better

            t_stat, p_t = sp_stats.ttest_rel(xb, xa)
            p_w = wilcoxon_exact_two_sided(diff)
            sd = diff.std(ddof=1)
            dz = float(diff.mean() / sd) if sd > 0 else float("inf")

            bound = SESOI_PP / 100.0
            p_tost, _lower, _upper = ttost_paired(xb, xa, -bound, bound)

            contrasts.append(
                {
                    "dataset": ds,
                    "arm_a": a,
                    "arm_a_label": ARM_LABEL.get(a, a),
                    "arm_b": b,
                    "arm_b_label": ARM_LABEL.get(b, b),
                    "n_pairs": len(common),
                    "seeds": common,
                    "mean_a_pct": float(xa.mean() * 100),
                    "mean_b_pct": float(xb.mean() * 100),
                    "mean_diff_pp": float(diff.mean() * 100),
                    "sd_diff_pp": float(sd * 100),
                    "t_stat": float(t_stat),
                    "p_paired_t": float(p_t),
                    "p_wilcoxon": float(p_w),
                    "p_wilcoxon_two_sided_floor": float(
                        wilcoxon_two_sided_floor(nonzero_differences(diff).size)
                    ),
                    "cohens_dz": dz,
                    "p_tost": float(p_tost),
                    "equivalent_within_sesoi": bool(p_tost < 0.05),
                }
            )

    # One family over all datasets, corrected with BH (primary) and Holm.
    p_raw = [c["p_paired_t"] for c in contrasts]
    adj_bh, rej_bh = adjust(p_raw, "fdr_bh", alpha=args.fdr_q)
    adj_holm, rej_holm = adjust(p_raw, "holm", alpha=args.fdr_q)
    for c, a_bh, r_bh, a_h, r_h in zip(contrasts, adj_bh, rej_bh, adj_holm, rej_holm):
        c["p_bh_adjusted"] = float(a_bh)
        c["significant_bh"] = bool(r_bh)
        c["p_holm_adjusted"] = float(a_h)
        c["significant_holm"] = bool(r_h)

    n_pairs_seen = sorted({c["n_pairs"] for c in contrasts})
    out = {
        "description": "Paired significance tests for the VEG component-ablation accuracy "
        "table. Seeds are paired across arms; all contrasts are corrected "
        "together with Benjamini-Hochberg (primary) and Holm-Bonferroni "
        "(reported alongside). Equivalence is tested by TOST against a "
        f"+-{SESOI_PP:g} pp equivalence margin.",
        "n_seeds": len(seeds),
        "seeds": seeds,
        "family_size": len(contrasts),
        "fdr_q": args.fdr_q,
        "multiplicity": {
            "family_size": len(contrasts),
            "family": f"{len(arms)} arms choose 2, times {len(datasets)} datasets",
            "primary": "Benjamini-Hochberg, q=%g" % args.fdr_q,
            "secondary": "Holm-Bonferroni, alpha=%g" % args.fdr_q,
            "implementation": "statsmodels.stats.multitest.multipletests",
            "n_significant_bh": int(sum(c["significant_bh"] for c in contrasts)),
            "n_significant_holm": int(sum(c["significant_holm"] for c in contrasts)),
        },
        "equivalence": {
            "method": "TOST (two one-sided paired t-tests), statsmodels ttost_paired",
            "sesoi_pp": SESOI_PP,
            "alpha": 0.05,
            "n_equivalent": int(sum(c["equivalent_within_sesoi"] for c in contrasts)),
            "why": (
                "A non-significant paired t does not establish that two arms "
                "perform the same. TOST does, against a stated bound: a +-%g pp "
                "equivalence margin." % SESOI_PP
            ),
        },
        "wilcoxon_note": (
            "p_wilcoxon is the exact two-sided Wilcoxon signed-rank p: zero "
            "differences are dropped (Wilcoxon's rule), equal absolute "
            "differences share their mean rank, and the null distribution "
            "enumerates every sign assignment to the remaining ranks. The test "
            "has a floor of 2/2**m over the m non-zero differences of a contrast, "
            "which is %s when none of its pairs differs by zero (%s); each dropped "
            "zero difference doubles the smallest p that contrast can attain. Where "
            "that floor exceeds alpha the test "
            "cannot reach significance at any effect size, so it is reported as a "
            "distribution-free companion only and never as the deciding "
            "test. The floor is recorded per contrast in "
            "p_wilcoxon_two_sided_floor."
            % (
                ", ".join("%g" % wilcoxon_two_sided_floor(n) for n in n_pairs_seen),
                ", ".join("n=%d" % n for n in n_pairs_seen),
            )
        ),
        "contrasts": contrasts,
    }
    attach(
        out,
        script="experiments/accuracy_significance.py",
        inputs=[args.ablation],
        seeds={"rng": None, "note": "No RNG. Deterministic given the input artifact."},
        sample_counts={
            "n_seeds": len(seeds),
            "seeds": seeds,
            "n_contrasts": len(contrasts),
            "n_pairs_per_contrast": sorted({c["n_pairs"] for c in contrasts}),
        },
        method=(
            "Paired t-test per contrast (primary); exact Wilcoxon signed-rank "
            "companion with its two-sided floor; TOST equivalence against a "
            "+-%g pp equivalence margin. Benjamini-Hochberg (primary) and Holm-Bonferroni "
            "reported together over one family of all contrasts." % SESOI_PP
        ),
        packages=["numpy", "scipy", "statsmodels"],
    )

    path = PROJECT_ROOT / output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")

    hdr = (
        f"{'dataset':<14} {'contrast':<44} {'diff pp':>8} {'p (t)':>9} "
        f"{'p (BH)':>9} {'BH':>3} {'p (Holm)':>9} {'Holm':>5} "
        f"{'p (TOST)':>9} {'equiv':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for c in contrasts:
        name = f"{c['arm_b_label']} - {c['arm_a_label']}"
        print(
            f"{c['dataset']:<14} {name:<44} {c['mean_diff_pp']:>+8.3f} "
            f"{c['p_paired_t']:>9.4f} "
            f"{c['p_bh_adjusted']:>9.4f} {'yes' if c['significant_bh'] else 'no':>3} "
            f"{c['p_holm_adjusted']:>9.4f} {'yes' if c['significant_holm'] else 'no':>5} "
            f"{c['p_tost']:>9.4f} "
            f"{'yes' if c['equivalent_within_sesoi'] else 'no':>6}"
        )
    print(
        f"\nFamily size {len(contrasts)}. "
        f"BH q={args.fdr_q} rejects {out['multiplicity']['n_significant_bh']}; "
        f"Holm alpha={args.fdr_q} rejects {out['multiplicity']['n_significant_holm']}. "
        f"TOST at +-{SESOI_PP} pp finds {out['equivalence']['n_equivalent']} "
        f"equivalent.\nWrote {path}"
    )


if __name__ == "__main__":
    main()
