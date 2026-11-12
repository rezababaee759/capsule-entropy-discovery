#!/usr/bin/env python3
"""Digit-composition confound check: fits the role x condition interaction with and
without a test_digit fixed effect (OLS and logit, two-way clustered) and jointly
tests the hub and specialist interaction terms. Adds the block
`sensitivity_digit_composition_confound` to the analysis JSON in place.

Usage:
    python experiments/symmetric_digit_confound.py --dataset mnist
    python experiments/symmetric_digit_confound.py --dataset cifar10
"""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups

from capsnet import PROJECT_ROOT
from capsnet.io import resolve_output
from capsnet.provenance import provenance_block
from capsnet.stats import joint_wald, two_way_logit_cov
from capsnet.symmetric import RECORDS, cluster_codes, design_frame

TERMS = [
    "C(role)[T.specialist]:C(condition)[T.within]",
    "C(role)[T.hub]:C(condition)[T.within]",
]
F_NO_DIGIT = "correct ~ C(role) * C(condition) + C(seed)"
F_DIGIT = "correct ~ C(role) * C(condition) + C(test_digit) + C(seed)"


def interaction_test(params, cov, names):
    """Per-term and joint Wald tests of TERMS."""
    b, se, z, p, chi2, chi2_p = joint_wald(params, cov, names, TERMS)
    return {
        "interaction_terms": {
            t: {
                "coef": float(b[i]),
                "se": float(se[i]),
                "z": float(z[i]),
                "p": float(p[i]),
            }
            for i, t in enumerate(TERMS)
        },
        "joint_wald_chi2": chi2,
        "joint_wald_df": len(b),
        "joint_wald_p": chi2_p,
    }


def fit_ols(model_df, formula):
    res = smf.ols(formula, data=model_df).fit()
    cov_both, _, _ = cov_cluster_2groups(
        res, model_df["_capsule_uid"].values, model_df["_image_uid"].values
    )
    return interaction_test(res.params.values, cov_both, list(res.params.index))


def fit_logit(model_df, formula):
    base = smf.logit(formula, data=model_df).fit(disp=0)
    cov, n_neg = two_way_logit_cov(
        base, model_df["_capsule_uid"].values, model_df["_image_uid"].values
    )
    return {
        "link": "logit",
        **interaction_test(base.params.values, cov, list(base.params.index)),
        "psd_projection_applied": bool(n_neg),
        "psd_negative_eigenvalues": n_neg,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(RECORDS))
    ap.add_argument("--records", default=None)
    ap.add_argument(
        "--json", default=None, help="analysis artifact to augment in place"
    )
    ap.add_argument(
        "--force", action="store_true", help="augment the canonical JSON from --records"
    )
    args = ap.parse_args()
    out_dir = RECORDS[args.dataset]["output_dir"]
    records = args.records or f"{out_dir}/symmetric_raw_records.parquet"
    output = resolve_output(
        ap,
        args,
        f"{out_dir}/symmetric_evaluation_glmm.json",
        ["records"],
        output="json",
    )

    model_df = design_frame(pd.read_parquet(PROJECT_ROOT / records))
    model_df["_capsule_uid"], model_df["_image_uid"] = cluster_codes(model_df)

    without_digit = fit_ols(model_df, F_NO_DIGIT)
    with_digit = fit_ols(model_df, F_DIGIT)
    without_digit_logit = fit_logit(model_df, F_NO_DIGIT)
    with_digit_logit = fit_logit(model_df, F_DIGIT)
    result = {
        "description": "Role x condition interaction with and without a test_digit fixed "
        "effect (OLS and logit, two-way clustered).",
        "scale_note": "OLS and logit fits, each with and without the test_digit control. "
        "Read each interaction's sign from its coefficients.",
        "without_test_digit_control": without_digit,
        "with_test_digit_control": with_digit,
        "without_test_digit_control_logit": without_digit_logit,
        "with_test_digit_control_logit": with_digit_logit,
    }

    # Run order: symmetric_raw_records.py, symmetric_analysis.py (writes the JSON),
    # then this script.
    out_path = PROJECT_ROOT / output
    if not out_path.exists():
        raise SystemExit(
            f"{out_path} does not exist. Run experiments/symmetric_analysis.py first; "
            f"this script only adds a block to that script's output."
        )
    with open(out_path) as f:
        full = json.load(f)
    for required in ("primary_model", "companion_analysis_capsule_level"):
        if required not in full:
            raise SystemExit(
                f"{out_path} is missing '{required}', so it was not written by "
                f"symmetric_analysis.py. Re-run that script before this one."
            )
    full["sensitivity_digit_composition_confound"] = result
    script = "experiments/symmetric_digit_confound.py"
    full["_provenance_augmentations"] = [
        b
        for b in full.get("_provenance_augmentations", [])
        if b.get("script") != script
    ]
    full["_provenance_augmentations"].append(
        provenance_block(
            script=script,
            inputs=[records],
            seeds={"rng": None, "note": "Fully deterministic; no RNG."},
            sample_counts={"n_obs": int(len(model_df))},
            method=(
                "Two-way cluster-robust OLS and clustered logit, each fitted with and "
                "without a C(test_digit) fixed effect, then a joint Wald test on the two "
                "role x condition interaction terms in each fit. Appends sensitivity_digit_composition_confound to the analysis "
                "artifact; it does not recompute any other field."
            ),
            packages=["numpy", "scipy", "statsmodels", "pandas"],
        )
    )

    with open(out_path, "w") as f:
        json.dump(full, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
