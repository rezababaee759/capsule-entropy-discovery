"""The ablation surrogate: the AblationPredictor network and its inference contract
(per-capsule features in, softplus(raw) + 1e-8 importance out)."""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from capsnet.ablation import single_capsule_ablation_effects
from capsnet.model import CapsuleNetwork

# Positional order of AblationPredictor.forward's six per-capsule inputs.
FEATURE_KEYS = ("activations", "couplings", "entropies", "grad_a", "grad_c", "grad_H")
NUM_CLASSES = 10
OUTPUT_EPS = 1e-8
ROUTING_EPS = 1e-8


class AblationPredictor(nn.Module):
    """Per-capsule MLP from the six routing features (plus the class one-hot) to the
    raw predicted ablation effect.

    Input:  six [N, num_capsules] tensors, optional digit_onehot [N, 10]
    Output: [N, num_capsules] raw scores; importance_from_raw maps them to importance.
    """

    def __init__(self, hidden_dim: int = 128, use_digit_features: bool = True) -> None:
        super().__init__()
        self.use_digit_features = use_digit_features
        input_dim = len(FEATURE_KEYS) + (NUM_CLASSES if use_digit_features else 0)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # softplus(x) ~ exp(x) for x < 0, so a bias of -9.2 starts the output at the
        # small magnitude typical of a single-capsule ablation effect.
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.01)
        self.net[-1].bias.data.fill_(-9.2)

    def forward(
        self,
        activation: Tensor,
        coupling: Tensor,
        entropy: Tensor,
        grad_a: Tensor,
        grad_c: Tensor,
        grad_H: Tensor,
        digit_onehot: Optional[Tensor] = None,
    ) -> Tensor:
        x = torch.stack([activation, coupling, entropy, grad_a, grad_c, grad_H], dim=-1)
        if self.use_digit_features and digit_onehot is not None:
            num_caps = x.size(1)
            digit_expanded = digit_onehot.unsqueeze(1).expand(-1, num_caps, -1)
            x = torch.cat([x, digit_expanded], dim=-1)
        return self.net(x).squeeze(-1)


def importance_from_raw(raw: Tensor) -> Tensor:
    """softplus(raw) + 1e-8: the surrogate's importance on the confidence-drop scale.

    Differentiable, with no cap, clip or rescale; the surrogate is trained with it.
    """
    return F.softplus(raw) + OUTPUT_EPS


def signed_log1p(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


@dataclass(frozen=True)
class CapsuleFeatures:
    features: Dict[
        str, Tensor
    ]  # FEATURE_KEYS -> [num_capsules], on the capsule model's device
    predicted_class: int  # j* = argmax_j ||v_j||
    confidence: float  # ||v_{j*}||
    primary_out: Tensor  # [1, num_capsules, capsule_dim], detached
    coupling: Tensor  # final-iteration c[0], [num_classes, num_capsules]


def extract_capsule_features(
    capsule_model: CapsuleNetwork, image: Tensor
) -> CapsuleFeatures:
    """Predictor inputs for one image [1, C, H, W], conditioned on its predicted class
    j* = argmax_j ||v_j||:

      activations  ||u_i||
      couplings    final-iteration c_{j*, i}
      entropies    -sum_j c_{j,i} log2(c_{j,i} + eps) / log2(10)
      grad_a       ||d(||v_{j*}||^2) / d u_i||
      grad_c       |d(||v_{j*}||^2) / d c_{j*, i}|
      grad_H       |<d(||v_{j*}||^2) / d c_{:, i}, unit(-(log2 c_{:, i} + 1))>|

    Coupling gradients are taken with respect to the final-iteration coupling tensor
    the routing module returns; the three gradient features are sign(g) * log1p(|g|).
    """
    if image.dim() != 4 or image.size(0) != 1:
        raise ValueError(f"expected one image [1, C, H, W], got {tuple(image.shape)}")
    eps = ROUTING_EPS
    with torch.no_grad():
        primary_out = capsule_model.primary(F.relu(capsule_model.conv1(image)))
        digit_caps, c_ij, _, _ = capsule_model.digits(primary_out)
        activations = torch.norm(primary_out[0], dim=-1)
        lengths = torch.norm(digit_caps[0], dim=-1)
        predicted_class = lengths.argmax().item()
        couplings = c_ij[0, predicted_class, :]
        coupling_dist = c_ij[0, :, :]
        entropies = -(coupling_dist * torch.log2(coupling_dist + eps)).sum(
            dim=0
        ) / np.log2(10)
        confidence = lengths[predicted_class].item()

    grad_a, grad_c, grad_H = gradient_features(
        capsule_model, primary_out, predicted_class
    )

    features = {
        "activations": activations,
        "couplings": couplings,
        "entropies": entropies,
        "grad_a": signed_log1p(grad_a),
        "grad_c": signed_log1p(grad_c),
        "grad_H": signed_log1p(grad_H),
    }
    n = primary_out.size(1)
    for k in FEATURE_KEYS:
        if features[k].shape != (n,):
            raise RuntimeError(f"{k}: shape {tuple(features[k].shape)} != ({n},)")
    return CapsuleFeatures(
        features, predicted_class, confidence, primary_out.detach(), coupling_dist
    )


def gradient_features(
    capsule_model: CapsuleNetwork, primary_out: Tensor, predicted_class: int
) -> Tuple[Tensor, Tensor, Tensor]:
    """grad_a, grad_c and grad_H of one image (see extract_capsule_features), before
    the signed log1p; primary_out is its [1, num_capsules, capsule_dim] primary capsules.
    """
    eps = ROUTING_EPS
    with torch.enable_grad():
        primary_out_g = primary_out.detach().requires_grad_(True)
        digit_caps_g, c_ij_g, _, _ = capsule_model.digits(primary_out_g)
        pred_conf = (digit_caps_g[0, predicted_class] ** 2).sum()
        grad_primary = torch.autograd.grad(pred_conf, primary_out_g, retain_graph=True)[
            0
        ]
        grad_c_full = torch.autograd.grad(pred_conf, c_ij_g)[0]
    grad_a = torch.norm(grad_primary[0], dim=-1)
    grad_c = torch.abs(grad_c_full[0, predicted_class, :])
    c_all = c_ij_g.detach()[0].clamp(min=eps)
    h_grad = -(torch.log2(c_all) + 1.0)
    h_unit = h_grad / (torch.norm(h_grad, dim=0).unsqueeze(0) + eps)
    grad_H = torch.abs((grad_c_full[0] * h_unit).sum(dim=0))
    return grad_a, grad_c, grad_H


def _check_class_index(class_index) -> int:
    if isinstance(class_index, bool) or not isinstance(class_index, numbers.Integral):
        raise TypeError(f"class_index must be an int, got {type(class_index).__name__}")
    if not 0 <= class_index < NUM_CLASSES:
        raise ValueError(f"class_index {class_index} outside [0, {NUM_CLASSES})")
    return int(class_index)


def predict_importance(
    predictor: AblationPredictor, features: Mapping[str, Tensor], class_index: int
) -> Tensor:
    """softplus(g_theta(features, onehot(class_index))) + 1e-8 for one image, [num_capsules].

    The result is on the predictor's device. Non-finite raw output raises.
    """
    missing = set(FEATURE_KEYS) - set(features)
    if missing:
        raise KeyError(f"missing features: {sorted(missing)}")
    if not getattr(predictor, "use_digit_features", False):
        raise ValueError("predictor was built without the class one-hot input")
    class_index = _check_class_index(class_index)
    inputs = [features[k] for k in FEATURE_KEYS]
    shape = inputs[0].shape
    if len(shape) != 1:
        raise ValueError(f"features must be 1-D per capsule, got {tuple(shape)}")
    for k, x in zip(FEATURE_KEYS, inputs):
        if x.shape != shape:
            raise ValueError(f"{k}: shape {tuple(x.shape)} != {tuple(shape)}")
        if x.dtype != torch.float32:
            raise TypeError(f"{k}: dtype {x.dtype}, expected torch.float32")
    device = next(predictor.parameters()).device
    onehot = (
        F.one_hot(torch.tensor([class_index]), num_classes=NUM_CLASSES)
        .float()
        .to(device)
    )
    with torch.no_grad():
        raw = predictor(
            *(x.unsqueeze(0).to(device) for x in inputs), digit_onehot=onehot
        )
    if not torch.isfinite(raw).all():
        raise ValueError("predictor returned non-finite output")
    return importance_from_raw(raw[0])


def predict_capsule_importance(
    capsule_model: CapsuleNetwork,
    predictor_model: AblationPredictor,
    image: Tensor,
) -> Tuple[Tensor, Dict[str, Tensor], Dict[str, Any]]:
    """Predicted importance for one image, with its features and routing metadata, all on CPU.

    The class one-hot is the predicted class j*.
    """
    capsule_model.eval()
    predictor_model.eval()

    cf = extract_capsule_features(capsule_model, image)
    importance = predict_importance(predictor_model, cf.features, cf.predicted_class)

    features = {k: cf.features[k].cpu() for k in FEATURE_KEYS}
    metadata = {
        "predicted_digit": cf.predicted_class,
        "confidence": cf.confidence,
        "routing_distribution": cf.coupling.cpu(),
    }
    return importance.cpu(), features, metadata


def predict_and_ablate(
    capsule_model: CapsuleNetwork, predictor: AblationPredictor, images, device
) -> Tuple[list, list]:
    """Per image [C, H, W]: the predicted importance of every capsule, and each capsule's
    ablation effect on ||v_{j*}|| clipped at 0, as float32 arrays. Features, the class
    one-hot and the ablation target all use the predicted class j*."""
    capsule_model.eval()
    predictor.eval()

    all_predictions = []
    all_targets = []

    for img in images:
        cf = extract_capsule_features(capsule_model, img.unsqueeze(0).to(device))
        ablation_effects = single_capsule_ablation_effects(
            capsule_model, cf.primary_out, cf.predicted_class, cf.confidence
        )

        preds = predict_importance(predictor, cf.features, cf.predicted_class)
        all_predictions.append(preds.cpu().numpy())
        all_targets.append(np.clip(ablation_effects.cpu().numpy(), 0, None))

    return all_predictions, all_targets
