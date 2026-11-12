#!/usr/bin/env python3
"""Per-observation routing-correctness records of the symmetric Hub/Specialist
evaluation: one row per (model, role, capsule, shown class, image).

Usage:
    python experiments/symmetric_raw_records.py --dataset mnist
    python experiments/symmetric_raw_records.py --dataset cifar10
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

from capsnet import PROJECT_ROOT
from capsnet.data import test_set
from capsnet.io import get_device, load_capsule_model, resolve_output
from capsnet.provenance import PROVENANCE_KEY, provenance_block
from capsnet.symmetric import (
    N_RANDOM,
    RECORDS,
    SAMPLES_PER_CLASS,
    batched_couplings,
    class_pools,
    model_path,
    taxonomy_paths,
)


def process_seed(dataset, seed, ds, pools, device):
    """Records of one trained model; `seed` is its slot."""
    print(f"\nSeed {seed}")
    t0 = time.time()

    model = load_capsule_model(str(model_path(dataset, seed)), dataset, device=device)

    tax_path, imp_path = taxonomy_paths(dataset, seed)
    with open(tax_path) as f:
        taxonomy = json.load(f)
    importance_matrix = np.load(imp_path)  # [10, n_primary_caps]

    all_hubs = sorted(taxonomy["primary_hubs"] + taxonomy["secondary_hubs"])
    specialists = sorted(taxonomy["specialists"])
    non_critical = taxonomy["non_critical"]

    rng = np.random.default_rng(seed + 1000)
    n_random = min(N_RANDOM, len(non_critical))
    random_caps = sorted(
        rng.choice(non_critical, size=n_random, replace=False).tolist()
    )

    roles = {"hub": all_hubs, "specialist": specialists, "random": random_caps}
    print(
        f"  Roles: hub n={len(all_hubs)}, specialist n={len(specialists)}, random n={len(random_caps)}"
    )

    batch_size = RECORDS[dataset]["batch_size"]
    # [shown class (10), image, class capsule (10), primary capsule]
    coupling_stack = np.stack(
        [batched_couplings(model, ds, pools[c], device, batch_size) for c in range(10)],
        axis=0,
    )
    n_img = coupling_stack.shape[1]
    print(
        f"  coupling_stack shape: {coupling_stack.shape}  ({time.time() - t0:.1f}s elapsed)"
    )

    records = []
    for role, cap_list in roles.items():
        if not cap_list:
            continue
        cap_arr = np.array(cap_list)
        home_class = np.array(
            [int(np.argmax(importance_matrix[:, c])) for c in cap_list]
        )

        sub = coupling_stack[:, :, :, cap_arr]  # [10, n_img, 10, C]
        top = sub.argmax(axis=2)  # [10, n_img, C]

        C = len(cap_list)
        test_class_grid = np.broadcast_to(np.arange(10)[:, None, None], (10, n_img, C))
        image_idx_grid = np.broadcast_to(
            np.arange(n_img)[None, :, None], (10, n_img, C)
        )
        capsule_grid = np.broadcast_to(cap_arr[None, None, :], (10, n_img, C))
        home_grid = np.broadcast_to(home_class[None, None, :], (10, n_img, C))

        # Correct means routing to the class actually shown, also on cross rows.
        correct = (top == test_class_grid).astype(np.int8)
        condition = np.where(test_class_grid == home_grid, "within", "cross")

        eps = 1e-10
        coupling_to_true = np.take_along_axis(
            sub, test_class_grid[:, :, None, :], axis=2
        )[:, :, 0, :]
        entropy = -np.sum(sub * np.log(np.clip(sub, eps, 1.0)), axis=2)

        df_role = pd.DataFrame(
            {
                "seed": np.int16(seed),
                "role": role,
                "capsule_idx": capsule_grid.ravel().astype(np.int16),
                "home_digit": home_grid.ravel().astype(np.int8),
                "test_digit": test_class_grid.ravel().astype(np.int8),
                "image_idx": image_idx_grid.ravel().astype(np.int16),
                "condition": condition.ravel(),
                "correct": correct.ravel(),
                "coupling_to_true_digit": coupling_to_true.ravel().astype(np.float32),
                "entropy": entropy.ravel().astype(np.float32),
            }
        )
        records.append(df_role)
        print(
            f"  {role}: {len(df_role):,} records "
            f"(within mean={df_role.loc[df_role.condition == 'within', 'correct'].mean():.3f}, "
            f"cross mean={df_role.loc[df_role.condition == 'cross', 'correct'].mean():.3f})"
        )

    seed_df = pd.concat(records, ignore_index=True)
    print(
        f"  Seed {seed} done in {time.time() - t0:.1f}s, {len(seed_df):,} total records."
    )
    return seed_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(RECORDS))
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="trained-model slots to evaluate; 42 is the primary checkpoint",
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--force", action="store_true", help="write the canonical output from --seeds"
    )
    args = parser.parse_args()
    dataset = args.dataset
    seeds = list(args.seeds or RECORDS[dataset]["seeds"])
    output_dir = resolve_output(
        parser, args, RECORDS[dataset]["output_dir"], ["seeds"], "output_dir"
    )

    device = get_device()
    print(f"Device: {device}")

    out_dir = PROJECT_ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = test_set(dataset)
    pools = class_pools(ds, dataset, SAMPLES_PER_CLASS)
    full_df = pd.concat(
        [process_seed(dataset, seed, ds, pools, device) for seed in seeds],
        ignore_index=True,
    )
    print(f"\nTotal records across {len(seeds)} seeds: {len(full_df):,}")

    parquet_path = out_dir / "symmetric_raw_records.parquet"
    full_df.to_parquet(parquet_path, index=False)

    # Parquet has no place for the provenance key, so it goes into a sidecar.
    side_path = out_dir / "symmetric_raw_records.provenance.json"
    caps = full_df.groupby(["seed", "role"])["capsule_idx"].nunique().to_dict()
    side_path.write_text(
        json.dumps(
            {
                PROVENANCE_KEY: provenance_block(
                    script="experiments/symmetric_raw_records.py",
                    inputs=[str(model_path(dataset, s)) for s in seeds]
                    + [str(p) for s in seeds for p in taxonomy_paths(dataset, s)],
                    seeds={
                        "trained_model_seeds": seeds,
                        "random_capsule_draw": "numpy default_rng(seed + 1000) per seed",
                        "note": "The image pools are a deterministic disjoint slice from "
                        "capsnet.data, not a random draw.",
                    },
                    sample_counts={
                        "n_records": int(len(full_df)),
                        "samples_per_class": SAMPLES_PER_CLASS,
                        "n_random_per_seed": N_RANDOM,
                        "capsules_per_seed_role": {
                            f"{k[0]}|{k[1]}": int(v) for k, v in caps.items()
                        },
                    },
                    method=(
                        "Per-observation routing-correctness records. Correctness is "
                        "argmax over the ten digit capsules of the coupling coefficient "
                        "equalling the TRUE label of the shown image; condition is within "
                        "when the shown class equals the capsule home class (argmax of its "
                        "importance column), cross otherwise. No inference here: the "
                        "clustering-aware models are fitted downstream."
                    ),
                    packages=["numpy", "pandas", "pyarrow"],
                )
            },
            indent=2,
        )
    )
    print(f"Provenance sidecar: {side_path}")
    print(f"Saved raw records to {parquet_path}")


if __name__ == "__main__":
    main()
