"""Capsule network (Sabour et al., 2017) with dynamic routing and Vectorized Entropy
Gating (per-capsule or uniform beta). ``forward(x)`` returns ``(v, c, b, history)``."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def squash(x: Tensor, dim: int = -1, eps: float = 1e-8) -> Tensor:
    """v = (||s||^2 / (1 + ||s||^2)) * s / ||s||: length in [0, 1), direction kept."""
    squared_norm = (x * x).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1.0 + squared_norm)
    return scale * x / torch.sqrt(squared_norm + eps)


class PrimaryCapsules(nn.Module):
    """9x9 stride-2 convolution reshaped into capsules.

    [B, in_channels, H, W] -> [B, H' * W' * num_capsules, capsule_dim], squashed.
    Capsule index order is (row, column, type).
    """

    def __init__(
        self,
        in_channels: int,
        num_capsules: int = 32,
        capsule_dim: int = 8,
    ) -> None:
        super().__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.conv = nn.Conv2d(
            in_channels,
            num_capsules * capsule_dim,
            kernel_size=9,
            stride=2,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        batch_size, _, height, width = x.size()

        # Reshape: [B, C*D, H, W] → [B, C, D, H, W]
        x = x.view(batch_size, self.num_capsules, self.capsule_dim, height, width)

        # Reorder dimensions and flatten spatial grid: [B, H*W*C, D]
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        x = x.view(batch_size, height * width * self.num_capsules, self.capsule_dim)

        return squash(x)


class RoutingStrategy(nn.Module):
    """Base class: holds the transformation weights W [1, O, I, D_out, D_in].

    ``forward(u)`` takes u [B, I, D_in] and returns (v [B, O, D_out], final coupling
    c [B, O, I], final agreement b [B, O, I], per-iteration history).
    """

    def __init__(
        self,
        num_input_caps: int,
        input_dim: int,
        num_output_caps: int,
        output_dim: int,
        routing_iters: int = 3,
    ) -> None:
        super().__init__()
        self.num_input_caps = num_input_caps
        self.input_dim = input_dim
        self.num_output_caps = num_output_caps
        self.output_dim = output_dim
        self.routing_iters = routing_iters

        weight = (
            torch.randn(1, num_output_caps, num_input_caps, output_dim, input_dim)
            * 0.01
        )
        self.W = nn.Parameter(weight)

    def forward(self, u: Tensor) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Any]]]:
        raise NotImplementedError("Subclasses must implement forward()")


class DynamicRouting(RoutingStrategy):
    """Routing by agreement: c = softmax(b) over outputs, v = squash(sum_i c u_hat),
    b += u_hat . v between iterations."""

    def forward(self, u: Tensor) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Any]]]:
        batch_size = u.size(0)

        # Transform input capsules: [B, 1, I, D, 1] × [1, O, I, D_out, D_in] → [B, O, I, D_out]
        u = u.unsqueeze(1).unsqueeze(-1)
        W = self.W.expand(batch_size, -1, -1, -1, -1)
        u_hat = torch.matmul(W, u).squeeze(-1)

        b = u.new_zeros(batch_size, self.num_output_caps, self.num_input_caps)

        routing_history: List[Dict[str, Any]] = []

        for iteration in range(self.routing_iters):
            # Softmax over output capsules
            c = F.softmax(b, dim=1)

            s = (c.unsqueeze(-1) * u_hat).sum(dim=2)
            v = squash(s)

            routing_history.append(
                {
                    "iteration": iteration,
                    "coupling_coeffs": c.clone(),
                    "agreement_scores": b.clone(),
                    "digit_capsules": v.clone(),
                }
            )

            # Update agreement scores (except on last iteration)
            if iteration < self.routing_iters - 1:
                b = b + (u_hat * v.unsqueeze(2)).sum(dim=-1)

        return v, c, b, routing_history


class EntropyRouting(RoutingStrategy):
    """Entropy-gated routing (VEG): capsule i's contribution to every parent is scaled
    by G_i = exp(-beta_i * H_i), with H_i the entropy of its coupling distribution.

    beta is unconstrained: beta_i > 0 suppresses high-entropy capsules (G_i = 1 at
    H_i = 0), beta_i < 0 amplifies them. With ``uniform_beta`` one scalar beta is
    shared by all capsules. History entries also carry 'entropy_gate' [B, 1, I].
    """

    def __init__(
        self,
        num_input_caps: int,
        input_dim: int,
        num_output_caps: int,
        output_dim: int,
        routing_iters: int = 3,
        beta: float = 0.0,
        uniform_beta: bool = False,
    ) -> None:
        super().__init__(
            num_input_caps, input_dim, num_output_caps, output_dim, routing_iters
        )
        self.uniform_beta = uniform_beta
        if uniform_beta:
            self.beta = nn.Parameter(torch.tensor([float(beta)]))
        else:
            self.beta = nn.Parameter(torch.ones(num_input_caps) * float(beta))

    def forward(self, u: Tensor) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Any]]]:
        batch_size = u.size(0)

        u = u.unsqueeze(1).unsqueeze(-1)
        W = self.W.expand(batch_size, -1, -1, -1, -1)
        u_hat = torch.matmul(W, u).squeeze(-1)

        b = u.new_zeros(batch_size, self.num_output_caps, self.num_input_caps)

        routing_history: List[Dict[str, Any]] = []
        eps = 1e-8

        for iteration in range(self.routing_iters):
            c = F.softmax(b, dim=1)

            # Entropy per primary capsule, H = -sum_j c_j log c_j over outputs; c: [B, O, I]
            entropy = -torch.sum(c * torch.log(c + eps), dim=1)  # [B, I]

            # G = exp(-beta * H); self.beta: [I], entropy: [B, I] -> gate: [B, 1, I]
            gate = torch.exp(-self.beta * entropy).unsqueeze(1)

            s = (c.unsqueeze(-1) * u_hat * gate.unsqueeze(-1)).sum(dim=2)
            v = squash(s)

            routing_history.append(
                {
                    "iteration": iteration,
                    "coupling_coeffs": c.clone(),
                    "agreement_scores": b.clone(),
                    "digit_capsules": v.clone(),
                    "entropy_gate": gate.detach().clone(),
                }
            )

            if iteration < self.routing_iters - 1:
                b = b + (u_hat * v.unsqueeze(2)).sum(dim=-1)

        return v, c, b, routing_history


class DigitCapsules(nn.Module):
    """Class capsules computed from the primary capsules by the chosen routing rule:
    'dynamic', 'entropy' (per-capsule beta, VEG) or 'entropy_uniform' (one shared beta).
    """

    def __init__(
        self,
        num_input_caps: int,
        input_dim: int,
        num_output_caps: int,
        output_dim: int,
        routing_iters: int = 3,
        routing_strategy: str = "dynamic",
        **routing_kwargs: Any,
    ) -> None:
        super().__init__()
        self.num_input_caps = num_input_caps
        self.input_dim = input_dim
        self.num_output_caps = num_output_caps
        self.output_dim = output_dim
        self.routing_iters = routing_iters
        self.routing_strategy = routing_strategy
        self.routing_module = self._create_routing_module(
            routing_strategy, **routing_kwargs
        )

    def _create_routing_module(self, strategy: str, **kwargs: Any) -> RoutingStrategy:
        base_args = (
            self.num_input_caps,
            self.input_dim,
            self.num_output_caps,
            self.output_dim,
            self.routing_iters,
        )

        if strategy == "dynamic":
            return DynamicRouting(*base_args)
        if strategy == "entropy":
            return EntropyRouting(
                *base_args, beta=kwargs.get("beta", 0.0), uniform_beta=False
            )
        if strategy == "entropy_uniform":
            return EntropyRouting(
                *base_args, beta=kwargs.get("beta", 0.0), uniform_beta=True
            )

        raise ValueError(f"Unknown routing strategy: {strategy}")

    def forward(self, u: Tensor) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Any]]]:
        return self.routing_module(u)


class CapsuleNetwork(nn.Module):
    """Conv1 (256 filters, 9x9) -> primary capsules -> 16-D class capsules -> decoder.

    The length of class capsule j is the score of class j. The decoder (160 -> 512 ->
    1024 -> C*H*W, sigmoid) reconstructs the input from the masked class capsules.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        routing_iters: int = 3,
        routing_strategy: str = "dynamic",
        img_width: int = 28,
        img_height: int = 28,
        num_primary_capsules: int = 32,
        primary_capsule_dim: int = 8,
        **routing_kwargs: Any,
    ) -> None:
        super().__init__()
        self.img_width = img_width
        self.img_height = img_height
        self.in_channels = in_channels

        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=9, stride=1)

        self.primary = PrimaryCapsules(
            256, num_capsules=num_primary_capsules, capsule_dim=primary_capsule_dim
        )

        # Conv1: (W - 9) / 1 + 1 = W - 8
        # Primary: (W - 8 - 9) / 2 + 1 = (W - 17) / 2 + 1
        width_after_conv1 = (img_width - 9) // 1 + 1
        height_after_conv1 = (img_height - 9) // 1 + 1
        width_after_primary = (width_after_conv1 - 9) // 2 + 1
        height_after_primary = (height_after_conv1 - 9) // 2 + 1

        num_primary_caps_total = (
            num_primary_capsules * width_after_primary * height_after_primary
        )

        self.digits = DigitCapsules(
            num_input_caps=num_primary_caps_total,
            input_dim=primary_capsule_dim,
            num_output_caps=num_classes,
            output_dim=16,
            routing_iters=routing_iters,
            routing_strategy=routing_strategy,
            **routing_kwargs,
        )

        self.num_classes = num_classes

        self.decoder = nn.Sequential(
            nn.Linear(16 * num_classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, in_channels * img_width * img_height),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Any]]]:
        """x [B, C, H, W] -> (v [B, num_classes, 16], c [B, num_classes, I],
        b [B, num_classes, I], routing history)."""
        x = F.relu(self.conv1(x))
        u = self.primary(x)
        v, c_ij, b_ij, routing_history = self.digits(u)

        return v, c_ij, b_ij, routing_history

    def reconstruct(self, v: Tensor, labels: Optional[Tensor] = None) -> Tensor:
        """Decode the capsule of `labels` (default: the longest one) to [B, C*H*W]."""
        lengths = torch.norm(v, dim=-1)

        if labels is None:
            labels = lengths.argmax(dim=1)

        mask = F.one_hot(labels, num_classes=self.num_classes).float()

        # Keep only the target capsule's vector
        masked = (v * mask.unsqueeze(-1)).reshape(v.size(0), -1)

        reconstruction = self.decoder(masked)

        return reconstruction


def capsule_loss(y_true: Tensor, v: Tensor) -> Tensor:
    """Margin loss, summed over classes and averaged over the batch:
    L_k = T_k max(0, 0.9 - ||v_k||)^2 + 0.5 (1 - T_k) max(0, ||v_k|| - 0.1)^2.

    y_true: one-hot [B, num_classes]; v: [B, num_classes, dim].
    """
    m_plus = 0.9  # Upper margin for present classes
    m_minus = 0.1  # Lower margin for absent classes
    lambda_weight = 0.5  # Down-weighting factor for absent classes

    lengths = torch.norm(v, dim=-1)

    present_loss = y_true * F.relu(m_plus - lengths) ** 2
    absent_loss = lambda_weight * (1.0 - y_true) * F.relu(lengths - m_minus) ** 2

    # Sum over classes, mean over batch
    return (present_loss + absent_loss).sum(dim=1).mean()


def reconstruction_loss(images: Tensor, reconstructions: Tensor) -> Tensor:
    """Mean squared error between images [B, C, H, W] and reconstructions [B, C*H*W]."""
    images_flat = images.view(images.size(0), -1)
    return F.mse_loss(reconstructions, images_flat, reduction="mean")


#: Architecture of each dataset's checkpoints. Capsule count = types x grid^2:
#: 32 x 6 x 6 = 1152 for 28x28 inputs, 64 x 8 x 8 = 4096 for 32x32.
ARCHITECTURES: Dict[str, Dict[str, int]] = {
    "mnist": dict(in_channels=1, img_width=28, img_height=28, num_primary_capsules=32),
    "fashionmnist": dict(
        in_channels=1, img_width=28, img_height=28, num_primary_capsules=32
    ),
    "cifar10": dict(
        in_channels=3, img_width=32, img_height=32, num_primary_capsules=64
    ),
}


def build_capsule_network(
    dataset: str = "mnist", routing_strategy: str = "dynamic", **overrides: Any
) -> CapsuleNetwork:
    """A freshly initialised 10-class, 3-iteration network with the dataset's architecture."""
    key = dataset.lower().replace("-", "").replace("_", "")
    return CapsuleNetwork(
        num_classes=10,
        routing_iters=3,
        routing_strategy=routing_strategy,
        **{**ARCHITECTURES[key], **overrides},
    )
