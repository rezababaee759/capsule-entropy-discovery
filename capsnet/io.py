"""Checkpoint loading, device selection, the overwrite and canonical-output guards, and
the paths of the VEG checkpoints, their native taxonomies and beta analyses."""

from __future__ import annotations

import re
from pathlib import Path

import torch

from capsnet.model import CapsuleNetwork, build_capsule_network
from capsnet.surrogate import AblationPredictor


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def guard_overwrite(path, force: bool) -> None:
    """Refuse to replace an existing output unless `force` is set."""
    path = Path(path)
    if path.exists() and not force:
        raise SystemExit(
            f"{path} already exists. Refusing to overwrite it (pass --force to "
            f"override). Other scripts in this repo may treat it as a fixed input; "
            f"overwriting it without meaning to cannot be undone."
        )


def resolve_output(parser, args, default: str, inputs, output: str = "output") -> str:
    """args.<output> if given, else `default`, the canonical path.

    Writing the canonical path from inputs that differ from their defaults needs the
    output path given explicitly or --force (args.force).
    """
    given = getattr(args, output)
    if given is not None:
        return given
    changed = [
        f"--{name}"
        for name in inputs
        if getattr(args, name) != parser.get_default(name)
    ]
    if changed and not args.force:
        parser.error(
            f"{', '.join(changed)} differs from the default, so this run would overwrite "
            f"the canonical {default}; pass --{output} or --force"
        )
    return default


def state_dict_of(checkpoint):
    """The state dict inside a checkpoint: capsule-network checkpoints are bare state
    dicts, surrogate checkpoints hold theirs under ``model_state_dict``."""
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def load_state_dict(path, map_location="cpu"):
    return state_dict_of(
        torch.load(path, map_location=map_location, weights_only=False)
    )


def routing_from_path(path) -> str:
    """'entropy' for a checkpoint whose file name says so, else 'dynamic'."""
    return "entropy" if "entropy" in str(path).lower() else "dynamic"


def load_capsule_model(
    path,
    dataset: str = "mnist",
    routing_strategy: str = "dynamic",
    device=None,
    **overrides,
) -> CapsuleNetwork:
    """A capsule network with the dataset's architecture, loaded from `path`, in eval mode.

    `overrides` replace architecture arguments (e.g. num_primary_capsules).
    """
    if device is None:
        device = get_device()
    model = build_capsule_network(dataset, routing_strategy, **overrides).to(device)
    model.load_state_dict(load_state_dict(path, map_location=device))
    model.eval()
    return model


def load_predictor(path, device="cpu") -> AblationPredictor:
    return predictor_from_checkpoint(
        torch.load(path, map_location=device, weights_only=False), device
    )


def predictor_from_checkpoint(checkpoint, device="cpu") -> AblationPredictor:
    """An AblationPredictor from a full checkpoint dict (with its configuration) or a
    bare state dict, in eval mode."""
    config = checkpoint if "model_state_dict" in checkpoint else {}
    model = AblationPredictor(
        hidden_dim=config.get("hidden_dim", 128),
        use_digit_features=config.get("use_digit_features", True),
    ).to(device)
    model.load_state_dict(state_dict_of(checkpoint))
    model.eval()
    return model


VEG_PRIMARY_TAG = "best_entropy"


def _veg_tag(tag: str) -> str:
    if tag != VEG_PRIMARY_TAG and not re.fullmatch(r"seed\d+", tag):
        raise ValueError(f"unknown model tag {tag!r}; use best_entropy or seed<N>")
    return tag


def veg_checkpoint(dataset: str, tag: str) -> str:
    """Repo-relative checkpoint of a VEG model tag (best_entropy or seed<N>)."""
    if _veg_tag(tag) == VEG_PRIMARY_TAG:
        return f"models/capsule_{dataset}_best_entropy.pt"
    return f"models/capsule_{dataset}_entropy_{tag}.pt"


def veg_taxonomy(dataset: str, tag: str) -> str:
    """Repo-relative hub_taxonomy.json derived natively on that VEG model."""
    return f"results/data/{dataset}/veg_{_veg_tag(tag)}/hub_taxonomy.json"


def veg_beta_analysis(dataset: str, tag: str) -> str:
    """Repo-relative per-model output of experiments/veg_beta_analysis.py."""
    if dataset == "mnist":
        return (
            f"results/beta_values/native/veg_beta_analysis_mnist_{_veg_tag(tag)}.json"
        )
    return f"results/beta_values/native_cifar10/veg_beta_analysis_cifar_{_veg_tag(tag)}.json"
