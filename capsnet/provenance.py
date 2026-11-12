"""Provenance blocks for result artifacts: which script, code state, invocation,
input hashes, seeds, sample counts, software and method produced an artifact."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from capsnet import PROJECT_ROOT as ROOT

#: the key every producer writes its block under
PROVENANCE_KEY = "_provenance"

#: schema version, bumped only if the field set changes meaning
SCHEMA = 2

#: What a producer executes: the library, the producers and trainers, the figure
#: and table generators, the pipeline script and its `make` wrapper, and the
#: dependency pins. The `dirty` flag of a block is decided over these paths alone.
CODE_PATHS = (
    "capsnet",
    "experiments",
    "train",
    "figures",
    "paper",
    "reproduce.sh",
    "Makefile",
    "pyproject.toml",
    "requirements.txt",
)


def _git(*args: str, root=ROOT) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=str(root), capture_output=True, text=True, timeout=30
        )
        return out.stdout if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def git_state(code_paths=CODE_PATHS, root=ROOT) -> dict:
    """Commit id, plus whether the code differed from that commit at run time.

    The commit identifies the code that produced an artifact; the inputs are
    identified separately, by the hashes recorded in the block's `inputs`. So
    `dirty` is true only when a file under `code_paths` is modified, staged, deleted
    or untracked relative to the commit, and `dirty_files` lists those files.
    Anything else in the working tree does not set it: result files (a regeneration
    rewrites them as it runs), figures, generated tables, documents, local settings.
    `code_paths` is recorded so a block states the scope its flag was decided over.
    """
    commit = _git("rev-parse", "HEAD", root=root)
    status = _git(
        "status", "--porcelain", "--untracked-files=all", "--", *code_paths, root=root
    )
    dirty = None if status is None else bool(status.strip())
    files = []
    if dirty:
        files = sorted(ln[3:].strip() for ln in status.splitlines() if ln[3:].strip())[
            :40
        ]
    return {
        "commit": commit.strip() if commit else None,
        "dirty": dirty,
        "dirty_files": files or None,
        "dirty_note": (
            "Files under code_paths differed from the commit (modified, staged, "
            "deleted or untracked): the commit id alone does not identify the "
            "code that produced this artifact."
            if dirty
            else None
        ),
        "code_paths": list(code_paths),
    }


def file_fingerprint(path) -> dict:
    """SHA-256 and size of one input, or a recorded absence."""
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    rel = os.path.relpath(p, ROOT)
    if not p.exists():
        return {"path": rel, "sha256": None, "bytes": None, "status": "ABSENT"}
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {
        "path": rel,
        "sha256": h.hexdigest(),
        "bytes": p.stat().st_size,
        "status": "present",
    }


def software_versions(packages=()) -> dict:
    """Python and the named packages, as actually imported in this process."""
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name in packages:
        mod_name = {"scikit-learn": "sklearn", "pyyaml": "yaml", "pillow": "PIL"}.get(
            name, name
        )
        try:
            out[name] = getattr(
                importlib.import_module(mod_name), "__version__", "unknown"
            )
        except Exception:
            out[name] = "not importable"
    try:  # device matters for any artifact produced on a GPU
        import torch

        out["torch"] = torch.__version__
        out["cuda"] = torch.version.cuda
        out["gpu"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except Exception:
        pass
    return out


def provenance_block(
    *,
    script: str,
    inputs=(),
    seeds=None,
    sample_counts=None,
    method: str,
    packages=(),
) -> dict:
    """Build the provenance block for one artifact.

    `inputs` are paths relative to the repository root (or absolute); each is recorded
    with its SHA-256 and size. `seeds` records every seed the run consumed and whether
    it affects the output. `method` states what was computed and is required.
    """
    if not method:
        raise ValueError("method is required; provenance without it is not provenance")
    block = {
        "schema": SCHEMA,
        "script": script,
        "git": git_state(),
        "argv": list(sys.argv),
        "inputs": [file_fingerprint(p) for p in inputs],
        "random_seed": seeds,
        "sample_counts": sample_counts,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "software": software_versions(packages),
        "statistical_method": method,
    }
    return block


def attach(document: dict, **kwargs) -> dict:
    """Attach a provenance block to `document` in place and return it."""
    document[PROVENANCE_KEY] = provenance_block(**kwargs)
    return document


def check_artifact(output: str, fresh: dict, producer: str) -> None:
    """Exit 1 unless the committed artifact at `output` equals `fresh` once its
    provenance block is set aside; `producer` is named in the message."""
    path = ROOT / output
    current = json.loads(path.read_text()) if path.exists() else {}
    current.pop(PROVENANCE_KEY, None)
    if current != json.loads(json.dumps(fresh)):
        print(f"{output} is out of date; run {producer}")
        sys.exit(1)
    print(f"{output} is up to date")
