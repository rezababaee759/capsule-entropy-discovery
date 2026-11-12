"""Per-capsule explanation helpers for the MNIST figures: routing entropy, the
routing of one primary capsule, gradient x input attribution, importance bands and
the most digit-specific capsules, and the command-line prologue the figures share."""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from capsnet.data import test_set
from capsnet.io import get_device, load_capsule_model, resolve_output
from capsnet.taxonomy import load_importance_matrix

#: canonical output directory of the explanation figures
FIGURE_DIR = "results/figures/explanations"


def routing_entropy(coupling_coeffs: np.ndarray) -> float:
    """Shannon entropy -sum_j c_j ln(c_j) of one capsule's coupling distribution, in
    nats, with c clipped to [1e-10, 1]."""
    c = np.clip(coupling_coeffs, 1e-10, 1.0)
    return -np.sum(c * np.log(c))


def capsule_routing(model, image: torch.Tensor, capsule_idx: int, device) -> dict:
    """Final coupling of primary capsule `capsule_idx` for one image.

    Returns coupling_coeffs ([n_classes] np.ndarray), entropy (routing_entropy) and
    top_digit (int argmax of the couplings).
    """
    image = image.to(device)
    if image.dim() == 3:
        image = image.unsqueeze(0)
    with torch.no_grad():
        _, c_ij, _, _ = model(image)
        coupling = c_ij[0, :, capsule_idx].cpu().numpy()  # c_ij: [B, classes, caps]
    return {
        "coupling_coeffs": coupling,
        "entropy": routing_entropy(coupling),
        "top_digit": int(np.argmax(coupling)),
    }


def spatial_attribution(
    model, image: torch.Tensor, capsule_idx: int, device
) -> np.ndarray:
    """|d||u_i||/dx * x| for primary capsule i over the input pixels, min-max scaled to
    [0, 1]; [H, W] for a one-channel image."""
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.clone().to(device).requires_grad_(True)

    conv1_out = F.relu(model.conv1(image))
    primary_out = model.primary(conv1_out)
    capsule_norm = torch.norm(primary_out[0, capsule_idx])

    model.zero_grad()
    capsule_norm.backward()

    grad = image.grad.data.squeeze().cpu().numpy()
    input_np = image.detach().squeeze().cpu().numpy()
    saliency = np.abs(grad * input_np)
    if saliency.max() > 0:
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
    return saliency


def importance_band(column: np.ndarray, value: float) -> str:
    """HIGH at or above the 95th percentile of `column`, MED at or above the 75th, else LOW."""
    if value >= np.percentile(column, 95):
        return "HIGH"
    if value >= np.percentile(column, 75):
        return "MED"
    return "LOW"


def most_specific(importance: np.ndarray, digit: int, k: int, exclude=()) -> list:
    """Up to `k` (capsule, importance / mean importance) pairs for `digit`, ranked by
    importance for `digit` minus mean importance over digits, among capsules at or above
    the digit's 90th percentile and not in `exclude`. importance: [capsules, digits]."""
    scores = importance[:, digit]
    mean = np.mean(importance, axis=1)
    floor = np.percentile(scores, 90)
    out = []
    for idx in np.argsort(scores - mean)[::-1]:
        if len(out) >= k:
            break
        if idx not in exclude and scores[idx] >= floor:
            out.append((idx, scores[idx] / (mean[idx] + 1e-6)))
    return out


def add_figure_args(parser) -> None:
    """Options every explanation figure takes: checkpoint, taxonomy, output directory,
    device and --force."""
    parser.add_argument("--model_path", default="models/capsule_mnist_best_dynamic.pt")
    parser.add_argument("--json_path", default="results/data/mnist/hub_taxonomy.json")
    parser.add_argument("--output_dir", help=f"default: {FIGURE_DIR}")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting an existing figure, or writing the canonical directory "
        "from another --model_path or --json_path",
    )


def load_figure_inputs(parser, args):
    """Resolve and create the output directory, choose the device and load the model,
    the taxonomy JSON, its importance matrix as [capsules, digits] and the MNIST test
    set. Returns (output_dir, device, model, taxonomy, importance_matrix, test_dataset)."""
    output_dir = resolve_output(
        parser, args, FIGURE_DIR, ["model_path", "json_path"], "output_dir"
    )
    device = get_device() if args.device == "auto" else torch.device(args.device)
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from {args.model_path}")
    model = load_capsule_model(args.model_path, "mnist", device=device)

    print(f"Loading importance data from {args.json_path}")
    with open(args.json_path, "r") as f:
        taxonomy = json.load(f)
    importance_matrix = load_importance_matrix(args.json_path).T

    return output_dir, device, model, taxonomy, importance_matrix, test_set("mnist")
