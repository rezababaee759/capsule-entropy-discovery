#!/usr/bin/env python3
"""Write <dir>/per_digit_sparsity_wrapper.json, {"importance_matrix": [[...], ...]},
from <dir>/importance_matrix.npy [10, n_capsules]: the --importance input of the VEG
beta analysis.

    python experiments/importance_wrapper.py DIR [DIR ...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def convert(directory: str) -> bool:
    """False if the directory has no importance_matrix.npy."""
    npy_path = os.path.join(directory, "importance_matrix.npy")
    out_path = os.path.join(directory, "per_digit_sparsity_wrapper.json")
    if not os.path.exists(npy_path):
        print(f"  SKIP {directory}: no importance_matrix.npy", file=sys.stderr)
        return False

    matrix = np.load(npy_path)
    if matrix.ndim != 2 or matrix.shape[0] != 10:
        raise ValueError(
            f"{npy_path}: expected (10, num_capsules) with classes on axis 0, "
            f"got {matrix.shape}"
        )
    with open(out_path, "w") as handle:
        json.dump({"importance_matrix": matrix.tolist()}, handle)
    print(f"  wrote {out_path}  shape={matrix.shape}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "directories", nargs="+", help="each holds an importance_matrix.npy"
    )
    args = parser.parse_args()
    written = sum(convert(directory) for directory in args.directories)
    print(f"Wrote {written} wrapper file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
