"""One-hot labels and the capsule-network train and evaluate loops."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from capsnet.model import CapsuleNetwork, capsule_loss, reconstruction_loss


def one_hot(labels: Tensor, num_classes: int) -> Tensor:
    """Integer labels [B] -> float one-hot [B, num_classes]."""
    return F.one_hot(labels, num_classes=num_classes).float()


def train_epoch(
    model: CapsuleNetwork,
    train_loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    recon_weight: float = 0.0005,
    clip_grad_norm: float | None = None,
) -> Dict[str, float]:
    """One epoch of margin loss + recon_weight * reconstruction loss, with the gradient
    norm clipped to `clip_grad_norm` when it is given.

    Returns the per-batch mean losses and the accuracy in percent.
    """
    model.train()
    total_loss = 0.0
    total_capsule_loss = 0.0
    total_recon_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        v, c_ij, b_ij, routing_history = model(images)
        reconstructions = model.reconstruct(v, labels)

        y_one_hot = one_hot(labels, model.num_classes)
        cap_loss = capsule_loss(y_one_hot, v)
        rec_loss = reconstruction_loss(images, reconstructions)
        loss = cap_loss + recon_weight * rec_loss

        optimizer.zero_grad()
        loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        total_capsule_loss += cap_loss.item()
        total_recon_loss += rec_loss.item()

        lengths = torch.norm(v, dim=-1)
        preds = lengths.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "acc": f"{100.0 * correct / total:.2f}%",
            }
        )

    n_batches = len(train_loader)
    return {
        "loss": total_loss / n_batches,
        "capsule_loss": total_capsule_loss / n_batches,
        "recon_loss": total_recon_loss / n_batches,
        "accuracy": 100.0 * correct / total,
    }


@torch.no_grad()
def evaluate(
    model: CapsuleNetwork,
    test_loader: DataLoader,
    device: torch.device,
    recon_weight: float = 0.0005,
    num_classes: int | None = None,
) -> Dict[str, Any]:
    """Per-batch mean losses, accuracy in percent and the correct/total counts on
    `test_loader`, without gradients. With `num_classes`, also per-class correct and
    total counts (`per_class_correct`, `per_class_total`)."""
    model.eval()
    total_loss = 0.0
    total_capsule_loss = 0.0
    total_recon_loss = 0.0
    correct = 0
    total = 0
    if num_classes:
        per_class_correct = np.zeros(num_classes)
        per_class_total = np.zeros(num_classes)

    for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
        images, labels = images.to(device), labels.to(device)

        v, c_ij, b_ij, routing_history = model(images)
        reconstructions = model.reconstruct(v, labels)

        y_one_hot = one_hot(labels, model.num_classes)
        cap_loss = capsule_loss(y_one_hot, v)
        rec_loss = reconstruction_loss(images, reconstructions)
        loss = cap_loss + recon_weight * rec_loss

        total_loss += loss.item()
        total_capsule_loss += cap_loss.item()
        total_recon_loss += rec_loss.item()

        lengths = torch.norm(v, dim=-1)
        preds = lengths.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        if num_classes:
            for cls in range(num_classes):
                mask = labels == cls
                if mask.any():
                    per_class_correct[cls] += (preds[mask] == cls).sum().item()
                    per_class_total[cls] += mask.sum().item()

    n_batches = len(test_loader)
    out = {
        "loss": total_loss / n_batches,
        "capsule_loss": total_capsule_loss / n_batches,
        "recon_loss": total_recon_loss / n_batches,
        "accuracy": 100.0 * correct / total,
        "correct": correct,
        "total": total,
    }
    if num_classes:
        out["per_class_correct"] = per_class_correct
        out["per_class_total"] = per_class_total
    return out
