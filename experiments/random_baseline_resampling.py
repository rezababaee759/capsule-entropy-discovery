#!/usr/bin/env python3
"""Resampling distribution of the MNIST non-critical control's domain gap: every
non-critical capsule is evaluated under the symmetric protocol, then subsets of
N_RANDOM capsules are drawn per model without replacement and pooled across models.

Usage:
    python experiments/random_baseline_resampling.py [--n_resamples 2000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch

from capsnet import PROJECT_ROOT
from capsnet.data import slice_for, test_set
from capsnet.io import get_device, load_capsule_model, resolve_output
from capsnet.provenance import attach
from capsnet.symmetric import (
    N_RANDOM,
    RECORDS,
    SAMPLES_PER_CLASS,
    batched_couplings,
    class_pools,
    model_path,
    per_capsule_correctness,
    taxonomy_paths,
)

DATASET = "mnist"
EVAL_START, _ = slice_for(DATASET, "evaluation", SAMPLES_PER_CLASS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for resampling")
    ap.add_argument(
        "--output",
        help="default: random_baseline_resampling.json in the MNIST records directory",
    )
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=RECORDS[DATASET]["seeds"],
        help="trained-model slots; 42 is the primary checkpoint",
    )
    ap.add_argument(
        "--force", action="store_true", help="write the canonical output from --seeds"
    )
    args = ap.parse_args()
    seeds = list(args.seeds)
    output = resolve_output(
        ap,
        args,
        f"{RECORDS[DATASET]['output_dir']}/random_baseline_resampling.json",
        ["seeds"],
    )

    device = get_device()
    dataset = test_set(DATASET)
    pools = class_pools(dataset, DATASET, SAMPLES_PER_CLASS)
    batch_size = RECORDS[DATASET]["batch_size"]

    per_seed: dict[int, dict] = {}
    for seed in seeds:
        t0 = time.time()
        print(f"\nseed {seed}")
        model = load_capsule_model(model_path(DATASET, seed), DATASET, device=device)
        tax_p, imp_p = taxonomy_paths(DATASET, seed)
        taxonomy = json.loads(tax_p.read_text())
        importance = np.load(imp_p)
        pool = sorted(int(c) for c in taxonomy["non_critical"])
        hubs = sorted(
            int(c) for c in taxonomy["primary_hubs"] + taxonomy["secondary_hubs"]
        )
        specialists = sorted(int(c) for c in taxonomy["specialists"])
        print(f"  non-critical pool: {len(pool)} capsules")

        stack = np.stack(
            [
                batched_couplings(model, dataset, pools[d], device, batch_size)
                for d in range(10)
            ],
            axis=0,
        )

        w_pool, c_pool, _ = per_capsule_correctness(stack, importance, pool)
        w_hub, c_hub, _ = per_capsule_correctness(stack, importance, hubs)
        w_spec, c_spec, _ = per_capsule_correctness(stack, importance, specialists)
        per_seed[seed] = dict(
            pool=pool,
            w_pool=w_pool,
            c_pool=c_pool,
            w_hub=w_hub,
            c_hub=c_hub,
            w_spec=w_spec,
            c_spec=c_spec,
        )
        print(f"  done in {time.time() - t0:.1f}s")

    def pooled_gap(key_w, key_c):
        """(gap pp, within %, cross %, n) over all capsules of the role, pooled across seeds."""
        w = np.concatenate([per_seed[s][key_w] for s in seeds])
        c = np.concatenate([per_seed[s][key_c] for s in seeds])
        return 100 * (w.mean() - c.mean()), 100 * w.mean(), 100 * c.mean(), len(w)

    hub_gap, hub_w, hub_c, hub_n = pooled_gap("w_hub", "c_hub")
    spec_gap, spec_w, spec_c, spec_n = pooled_gap("w_spec", "c_spec")
    full_gap, full_w, full_c, full_n = pooled_gap("w_pool", "c_pool")

    rng = np.random.default_rng(args.seed)
    gaps = np.empty(args.n_resamples)
    for b in range(args.n_resamples):
        ws, cs = [], []
        for s in seeds:
            n = len(per_seed[s]["pool"])
            take = rng.choice(n, size=min(N_RANDOM, n), replace=False)
            ws.append(per_seed[s]["w_pool"][take])
            cs.append(per_seed[s]["c_pool"][take])
        w = np.concatenate(ws)
        c = np.concatenate(cs)
        gaps[b] = 100 * (w.mean() - c.mean())

    lo, hi = np.percentile(gaps, [2.5, 97.5])
    frac_hub = float((gaps >= hub_gap).mean())
    frac_spec = float((gaps >= spec_gap).mean())
    print(
        f"\nResampled gap ({args.n_resamples} draws of {N_RANDOM}/seed): "
        f"mean {gaps.mean():.2f} pp, sd {gaps.std(ddof=1):.2f} pp, "
        f"95% [{lo:.2f}, {hi:.2f}] pp"
    )
    print(f"  full pool  {full_gap:6.2f} pp (n={full_n})")
    print(f"  hub        {hub_gap:6.2f} pp (n={hub_n}), P(draw >= hub) {frac_hub:.3f}")
    print(
        f"  specialist {spec_gap:6.2f} pp (n={spec_n}), P(draw >= specialist) {frac_spec:.3f}"
    )

    out = {
        "description": (
            "Resampling distribution of the Random-baseline domain gap. The "
            f"canonical analysis draws {N_RANDOM} non-critical capsules per seed once; "
            "this evaluates the entire non-critical pool with the identical protocol "
            f"and resamples {N_RANDOM}-capsule subsets, so the interval reflects which "
            "capsules the protocol could have selected."
        ),
        "n_resamples": args.n_resamples,
        "n_random_per_seed": N_RANDOM,
        "seeds": seeds,
        "samples_per_digit": SAMPLES_PER_CLASS,
        "eval_start": EVAL_START,
        "pool_sizes": {str(s): len(per_seed[s]["pool"]) for s in seeds},
        "random_baseline_gap_resampled": {
            "mean_pp": float(gaps.mean()),
            "sd_pp": float(gaps.std(ddof=1)),
            "ci95_pp": [float(lo), float(hi)],
            "min_pp": float(gaps.min()),
            "max_pp": float(gaps.max()),
        },
        "full_pool_no_sampling": {
            "gap_pp": float(full_gap),
            "within_pct": float(full_w),
            "cross_pct": float(full_c),
            "n_capsule_seats": int(full_n),
        },
        "hub": {
            "gap_pp": float(hub_gap),
            "within_pct": float(hub_w),
            "cross_pct": float(hub_c),
            "n_capsule_seats": int(hub_n),
        },
        "specialist": {
            "gap_pp": float(spec_gap),
            "within_pct": float(spec_w),
            "cross_pct": float(spec_c),
            "n_capsule_seats": int(spec_n),
        },
        "one_sided_resampling_p": {"hub": frac_hub, "specialist": frac_spec},
        "environment": {"torch": torch.__version__, "python": sys.version.split()[0]},
    }
    attach(
        out,
        script="experiments/random_baseline_resampling.py",
        inputs=[str(model_path(DATASET, s)) for s in seeds]
        + [str(p) for s in seeds for p in taxonomy_paths(DATASET, s)],
        seeds={
            "--seed": args.seed,
            "resampling_rng": f"numpy default_rng({args.seed})",
            "trained_model_seeds": seeds,
            "note": "The image pools are a deterministic slice; only the "
            "capsule subsampling consumes the RNG.",
        },
        sample_counts={
            "n_resamples": args.n_resamples,
            "n_random_per_seed": N_RANDOM,
            "samples_per_digit": SAMPLES_PER_CLASS,
            "non_critical_pool_sizes": out["pool_sizes"],
            "hub_capsule_seats": out["hub"]["n_capsule_seats"],
            "specialist_capsule_seats": out["specialist"]["n_capsule_seats"],
        },
        method=(
            "Subsampling (not bootstrap) distribution of the non-critical "
            "domain gap: every non-critical capsule is evaluated under the "
            f"canonical protocol, then {N_RANDOM}-capsule subsets are drawn WITHOUT "
            "replacement per seed and pooled across seeds, giving a 95% "
            "percentile interval over which capsules the protocol could have "
            "selected. One-sided tail fractions are reported for the Hub and "
            "Specialist gaps."
        ),
        packages=["numpy", "scipy"],
    )

    out_path = PROJECT_ROOT / output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
