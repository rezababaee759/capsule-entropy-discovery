#!/usr/bin/env python3
"""Keep the paper's copies of result figures in docs/figures/ identical to their originals.

Some paper figures are written by figure scripts into results/; this pairs each copy in
docs/figures/ with its original and refreshes or checks it, byte for byte.

    python paper/sync_figures.py               # refresh every copy from its original
    python paper/sync_figures.py --check       # fail if any copy is missing or differs
    python paper/sync_figures.py --originals   # print the originals, one per line
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

from capsnet import PROJECT_ROOT

FIGURES = PROJECT_ROOT / "docs" / "figures"

# Original (written by the producer named beside it) -> file name in docs/figures/.
PAIRS: dict[str, str] = {
    # figures/veg_beta_by_role.py
    "results/beta_values/native/veg_beta_by_role_mnist.png": "veg_beta_by_role_mnist.png",
    "results/beta_values/native_cifar10/veg_beta_by_role_cifar10.png": "veg_beta_by_role_cifar10.png",
    # figures/sample_explanation.py
    "results/figures/explanations/explanation_d3_s4_pred3.png": "explanation_d3_s4_pred3.png",
    "results/figures/explanations/explanation_d5_s4_pred5.png": "explanation_d5_s4_pred5.png",
    # figures/capsule_tracking.py
    "results/figures/explanations/capsule_tracking_0_8.png": "capsule_tracking_0_8.png",
    "results/figures/explanations/capsule_tracking_2_7.png": "capsule_tracking_2_7.png",
    # figures/specialist_interactions.py
    "results/figures/explanations/single_sample_d3_s4_20caps.png": "single_sample_d3_s4_20caps.png",
    "results/figures/explanations/single_sample_d4_s4_20caps.png": "single_sample_d4_s4_20caps.png",
    "results/figures/explanations/single_sample_d5_s4_20caps.png": "single_sample_d5_s4_20caps.png",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--check",
        action="store_true",
        help="fail if any copy is missing or differs from its original",
    )
    ap.add_argument(
        "--originals",
        action="store_true",
        help="print the original paths, relative to the repository root",
    )
    args = ap.parse_args()

    if args.originals:
        print("\n".join(PAIRS))
        return 0

    missing = [src for src in PAIRS if not (PROJECT_ROOT / src).is_file()]
    if missing:
        print(
            "original figure(s) not on disk -- regenerate them first:", file=sys.stderr
        )
        for src in missing:
            print(f"  {src}", file=sys.stderr)
        return 1

    stale = []
    for src, name in PAIRS.items():
        copy = FIGURES / name
        if not copy.is_file() or digest(copy) != digest(PROJECT_ROOT / src):
            stale.append((src, copy))

    if args.check:
        if stale:
            print(
                f"{len(stale)} paper figure(s) differ from their originals:",
                file=sys.stderr,
            )
            for src, copy in stale:
                state = "missing" if not copy.is_file() else "differs"
                print(
                    f"  {copy.relative_to(PROJECT_ROOT)} ({state}) <- {src}",
                    file=sys.stderr,
                )
            print("run: python paper/sync_figures.py", file=sys.stderr)
            return 1
        print(f"all {len(PAIRS)} paper figures match their originals")
        return 0

    for src, copy in stale:
        shutil.copyfile(PROJECT_ROOT / src, copy)
        print(f"updated {copy.relative_to(PROJECT_ROOT)} <- {src}")
    print(
        f"{len(PAIRS) - len(stale)} of {len(PAIRS)} already current"
        if not stale
        else f"{len(stale)} of {len(PAIRS)} copies refreshed"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
