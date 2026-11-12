"""Train the ablation surrogate: an MLP from per-capsule routing features to each
primary capsule's single-capsule ablation effect, on the surrogate_train split.

Writes the checkpoint (with its final_metrics) to --output and the training set
beside it as <output stem>_training_data.pt.

Usage:
    python train/train_surrogate.py --force
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.model_selection import KFold
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from capsnet.ablation import single_capsule_ablation_effects
from capsnet.data import split_indices, test_set
from capsnet.io import get_device, guard_overwrite, load_capsule_model, resolve_output
from capsnet.model import ARCHITECTURES, CapsuleNetwork
from capsnet.surrogate import (
    AblationPredictor,
    extract_capsule_features,
    importance_from_raw,
)

DigitStats = Dict[int, Dict[str, float]]
DatasetDict = Dict[str, Tensor | DigitStats]
MetricsDict = Dict[str, float]

TENSOR_KEYS = (
    "activations",
    "couplings",
    "entropies",
    "ablation_effects",
    "ablation_effects_std",
    "grad_a",
    "grad_c",
    "grad_H",
    "measured_mask",
)
GRAD_KEYS = ("grad_a", "grad_c", "grad_H")


def collect_ablation_data(
    model: CapsuleNetwork,
    loaders: Dict[int, DataLoader],
    device: torch.device,
) -> DatasetDict:
    """Ablate every primary capsule on every image, one at a time.

    `loaders` yields each digit's images in batches of one. One row per image: the
    image's own activation, coupling and entropy features; the ablation effects and
    gradient features are the means over that digit's images. All tensors are
    [n_images, n_capsules], digit_labels is [n_images].
    """
    model.eval()
    num_capsules = model.digits.num_input_caps
    images_per_digit = len(loaders[0].dataset)
    print(
        f"Collecting ablation data: {images_per_digit} images/digit x 10 digits "
        f"x {num_capsules} capsules"
    )

    data = {key: [] for key in TENSOR_KEYS}
    digit_labels = []
    for digit in range(10):
        effects, features = [], []
        for n, (image, _) in enumerate(loaders[digit]):
            cf = extract_capsule_features(model, image.to(device))
            ablation_effects = single_capsule_ablation_effects(
                model,
                cf.primary_out,
                cf.predicted_class,
                cf.confidence,
                desc=f"  Digit {digit} image {n + 1}/{images_per_digit}",
            )
            effects.append(ablation_effects.detach().cpu())
            features.append({k: v.detach().cpu() for k, v in cf.features.items()})

        effects_stack = torch.stack(effects)
        mean_effects, std_effects = effects_stack.mean(dim=0), effects_stack.std(dim=0)
        mean_grads = {
            k: torch.stack([f[k] for f in features]).mean(dim=0) for k in GRAD_KEYS
        }
        for f in features:
            for k in ("activations", "couplings", "entropies"):
                data[k].append(f[k])
            data["ablation_effects"].append(mean_effects)
            data["ablation_effects_std"].append(std_effects)
            for k in GRAD_KEYS:
                data[k].append(mean_grads[k])
            data["measured_mask"].append(torch.ones(num_capsules, dtype=torch.bool))
            digit_labels.append(digit)
        print(f"  Digit {digit}: {len(features)} images")

    data = {key: torch.stack(rows) for key, rows in data.items()}
    data["digit_labels"] = torch.tensor(digit_labels)
    print(
        f"Collected {data['activations'].size(0)} rows; "
        f"features {tuple(data['activations'].shape)}"
    )
    return data


def compute_digit_statistics(
    targets: Tensor, mask: Tensor, digit_labels: Tensor
) -> DigitStats:
    """Per-digit mean and std of the measured targets; stored with the training
    data, read by no loss or metric."""
    digit_stats = {}
    for digit in range(10):
        digit_mask_samples = digit_labels == digit
        digit_mean, digit_std = 0.0, 1.0
        if digit_mask_samples.any():
            digit_targets = targets[digit_mask_samples]
            digit_mask_caps = mask[digit_mask_samples]
            if digit_mask_caps.any():
                valid_targets = digit_targets[digit_mask_caps]
                digit_mean = valid_targets.mean().item()
                digit_std = valid_targets.std(unbiased=False).clamp(min=1e-6).item()
        digit_stats[digit] = {"mean": digit_mean, "std": digit_std}
    return digit_stats


def compute_topk_loss(
    predictions_raw: Tensor,
    targets_raw: Tensor,
    k: int = 50,
    margin: float | None = None,
) -> Tensor:
    """Soft-margin pairwise loss pushing the k largest targets' predictions above
    all others. Both inputs are on the original (not log1p) scale; the margin
    defaults to 10% of the mean |target|. Capped at 10."""
    predictions_flat = predictions_raw.flatten()
    targets_flat = targets_raw.flatten().detach()

    with torch.no_grad():
        _, topk_indices = torch.topk(targets_flat, k=k)
    topk_mask = torch.zeros_like(targets_flat, dtype=torch.bool)
    topk_mask[topk_indices] = True

    pred_topk = predictions_flat[topk_mask]
    pred_rest = predictions_flat[~topk_mask]

    if margin is None:
        with torch.no_grad():
            target_scale = targets_flat.abs().mean()
        margin = max(1e-6, target_scale * 0.1)

    score_diff = pred_topk.unsqueeze(1) - pred_rest.unsqueeze(0)  # [k, n-k]
    margin_tensor = score_diff.new_tensor(margin)
    loss = F.softplus(margin_tensor - score_diff).mean()
    return torch.clamp(loss, max=10.0)


def compute_ranking_metrics(
    predictions: Tensor,
    targets: Tensor,
    k_values: Tuple[int, ...] = (10, 50, 100, 200),
) -> MetricsDict:
    """precision@k and ndcg@k (binary relevance: membership in the target top-k)."""
    predictions_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()

    pred_topk_indices = {k: np.argsort(predictions_np)[-k:][::-1] for k in k_values}
    targ_topk_indices = {k: np.argsort(targets_np)[-k:][::-1] for k in k_values}

    metrics = {}
    for k in k_values:
        pred_set = set(pred_topk_indices[k])
        targ_set = set(targ_topk_indices[k])
        metrics[f"precision@{k}"] = len(pred_set.intersection(targ_set)) / k

        relevance = np.array(
            [1 if idx in targ_set else 0 for idx in pred_topk_indices[k]]
        )
        dcg = np.sum(relevance / np.log2(np.arange(2, k + 2)))
        ideal_relevance = np.ones(min(k, len(targ_set)))
        idcg = np.sum(ideal_relevance / np.log2(np.arange(2, len(ideal_relevance) + 2)))
        metrics[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0.0
    return metrics


def compute_loss(
    model: AblationPredictor,
    data: DatasetDict,
    device: torch.device,
    log_scale_factor: float = 10000.0,
    topk_loss_weight: float = 0.0,
    topk_k: int = 50,
    verbose: bool = False,
) -> Tuple[Tensor, MetricsDict]:
    """MSE between log1p(log_scale_factor * x) of prediction and target, plus the
    weighted top-k loss; returns (loss, metrics on the original scale)."""
    activations = data["activations"].to(device)
    couplings = data["couplings"].to(device)
    entropies = data["entropies"].to(device)
    grad_a = data["grad_a"].to(device)
    grad_c = data["grad_c"].to(device)
    grad_H = data["grad_H"].to(device)
    targets = data["ablation_effects"].to(device)
    mask = data["measured_mask"].to(device)
    digit_labels = data["digit_labels"].to(device)

    digit_onehot = F.one_hot(digit_labels, num_classes=10).float().to(device)

    predictions = model(
        activations,
        couplings,
        entropies,
        grad_a,
        grad_c,
        grad_H,
        digit_onehot=digit_onehot,
    )

    pred_masked = predictions[mask]
    targ_masked = targets[mask]

    targ_clamped = torch.clamp(targ_masked, min=0.0)

    targ_transformed = torch.log1p(targ_clamped * log_scale_factor)
    pred_positive = importance_from_raw(pred_masked)
    pred_transformed = torch.log1p(pred_positive * log_scale_factor)

    if torch.isnan(targ_transformed).any() or torch.isinf(targ_transformed).any():
        targ_transformed = torch.nan_to_num(
            targ_transformed, nan=0.0, posinf=10.0, neginf=0.0
        )
    if torch.isnan(pred_transformed).any() or torch.isinf(pred_transformed).any():
        pred_transformed = torch.nan_to_num(
            pred_transformed, nan=0.0, posinf=10.0, neginf=0.0
        )

    individual_loss = F.mse_loss(pred_transformed, targ_transformed)

    topk_loss = torch.tensor(0.0, device=device)
    if topk_loss_weight > 0:
        pred_positive_for_loss = importance_from_raw(predictions)
        targ_for_topk = targets.detach()
        topk_loss = compute_topk_loss(pred_positive_for_loss, targ_for_topk, k=topk_k)

    total_loss = individual_loss + topk_loss_weight * topk_loss

    with torch.no_grad():
        pred_raw_positive = importance_from_raw(predictions)

        pred_flat = pred_raw_positive[mask].cpu().numpy()
        targ_flat = targets[mask].cpu().numpy()

        if len(pred_flat) > 1 and targ_flat.std() > 0:
            pearson = pearsonr(pred_flat, targ_flat)[0]
        else:
            pearson = 0.0

        if len(pred_flat) > 1 and len(np.unique(targ_flat)) > 1:
            spearman = spearmanr(pred_flat, targ_flat)[0]
        else:
            spearman = 0.0

        mse = F.mse_loss(pred_raw_positive[mask], targets[mask]).item()

        ranking_metrics = compute_ranking_metrics(
            pred_raw_positive[mask].flatten(),
            targets[mask].flatten(),
            k_values=[10, 50, 100, 200],
        )

        metrics = {
            "pearson": pearson,
            "spearman": spearman,
            "mse": mse,
            "individual_loss": individual_loss.item(),
            "topk_loss": topk_loss.item(),
            **ranking_metrics,
        }

        if verbose and len(pred_flat) > 100:
            pred_cv = pred_flat.std() / (pred_flat.mean() + 1e-8)
            targ_cv = targ_flat.std() / (targ_flat.mean() + 1e-8)

            pred_ranks = rankdata(pred_flat, method="average")
            targ_ranks = rankdata(targ_flat, method="average")
            rank_diff = np.abs(pred_ranks - targ_ranks)

            value_diff = np.abs(pred_flat - targ_flat)
            small_diff_mask = value_diff < (targ_flat.std() * 0.1)
            small_diff_ratio = small_diff_mask.mean()

            metrics["diagnostics"] = {
                "pred_cv": pred_cv,
                "targ_cv": targ_cv,
                "median_rank_diff": np.median(rank_diff),
                "mean_rank_diff": np.mean(rank_diff),
                "small_diff_ratio": small_diff_ratio,
            }

    return total_loss, metrics


def cross_validate(
    data: DatasetDict,
    device: torch.device,
    hidden_dim: int = 128,
    num_epochs: int = 500,
    cv_folds: int = 5,
    log_scale_factor: float = 10000.0,
    topk_loss_weight: float = 0.0,
    topk_k: int = 50,
) -> List[MetricsDict]:
    """K-fold CV over rows split at random. A digit's rows share one averaged
    target, so the folds are not target-disjoint: a fit diagnostic only."""
    print(f"\nCross-validation: {cv_folds}-fold")

    indices = np.arange(data["activations"].size(0))
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\nFold {fold + 1}/{cv_folds}")

        train_data = {k: v[train_idx] for k, v in data.items()}
        val_data = {k: v[val_idx] for k, v in data.items()}
        model = AblationPredictor(hidden_dim=hidden_dim, use_digit_features=True).to(
            device
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=50, factor=0.5, min_lr=1e-6, verbose=False
        )

        for epoch in range(num_epochs):
            model.train()
            loss, train_metrics = compute_loss(
                model,
                train_data,
                device,
                log_scale_factor=log_scale_factor,
                topk_loss_weight=topk_loss_weight,
                topk_k=topk_k,
                verbose=False,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())

            if epoch % 100 == 0:
                print(
                    f"    Epoch {epoch:3d}: Loss={loss.item():.4f}, "
                    f"Pearson={train_metrics['pearson']:.4f}, "
                    f"Spearman={train_metrics['spearman']:.4f}"
                )

        model.eval()
        with torch.no_grad():
            _, val_metrics = compute_loss(
                model,
                val_data,
                device,
                log_scale_factor=log_scale_factor,
                topk_loss_weight=0.0,
                topk_k=topk_k,
                verbose=False,
            )
        print(
            f"  Validation: Pearson={val_metrics['pearson']:.4f}, "
            f"Spearman={val_metrics['spearman']:.4f}, "
            f"MSE={val_metrics['mse']:.2e}"
        )
        cv_results.append(val_metrics)

    for name in ("pearson", "spearman", "mse"):
        scores = [r[name] for r in cv_results]
        print(f"CV {name}: {np.mean(scores):.4g} +/- {np.std(scores):.4g}")
    return cv_results


def train_full(
    data: DatasetDict,
    device: torch.device,
    hidden_dim: int = 128,
    num_epochs: int = 500,
    log_scale_factor: float = 10000.0,
    topk_loss_weight: float = 0.8,
    topk_k: int = 50,
    output_path: str = "ablation_predictor.pt",
) -> Tuple[AblationPredictor, MetricsDict]:
    """Train on every row, then save the checkpoint and the training data."""
    print("\nTraining on the full dataset")

    data["digit_stats"] = compute_digit_statistics(
        data["ablation_effects"], data["measured_mask"], data["digit_labels"]
    )

    model = AblationPredictor(hidden_dim=hidden_dim, use_digit_features=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=50, factor=0.5, min_lr=1e-6, verbose=True
    )

    for epoch in range(num_epochs):
        model.train()
        loss, metrics = compute_loss(
            model,
            data,
            device,
            log_scale_factor=log_scale_factor,
            topk_loss_weight=topk_loss_weight,
            topk_k=topk_k,
            verbose=(epoch % 50 == 0),
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())

        if epoch % 50 == 0:
            print(
                f"Epoch {epoch:3d}: Loss={loss.item():.4f}, "
                f"Individual={metrics['individual_loss']:.4f}, "
                f"TopK={metrics['topk_loss']:.4f}, "
                f"Pearson={metrics['pearson']:.4f}, "
                f"Spearman={metrics['spearman']:.4f}, "
                f"P@50={metrics['precision@50']:.4f}"
            )

    model.eval()
    with torch.no_grad():
        _, final_metrics = compute_loss(
            model,
            data,
            device,
            log_scale_factor=log_scale_factor,
            topk_loss_weight=0.0,
            topk_k=topk_k,
            verbose=True,
        )
    print(
        "\nFinal (in-sample): "
        + ", ".join(
            f"{k}={final_metrics[k]:.4f}"
            for k in ("pearson", "spearman", "precision@10", "precision@50", "ndcg@50")
        )
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_dim": hidden_dim,
            "use_digit_features": True,
            "use_log_scale": True,
            "log_scale_factor": log_scale_factor,
            "final_metrics": final_metrics,
        },
        output_path,
    )
    print(f"Model saved to: {output_path}")

    data_path = training_data_path(output_path)
    torch.save(data, data_path)
    print(f"Training data saved to: {data_path}")

    return model, final_metrics


def training_data_path(output_path: str) -> str:
    """<output stem>_training_data.pt beside the checkpoint."""
    p = Path(output_path)
    return str(p.with_name(p.stem + "_training_data.pt"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ablation surrogate")
    parser.add_argument(
        "--model",
        help="trained CapsNet checkpoint (default: the dataset's primary one)",
    )
    parser.add_argument("--images_per_digit", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument(
        "--output",
        help="default: models/ablation_predictor.pt (MNIST), "
        "models/ablation_predictor_cifar10.pt (CIFAR-10)",
    )
    parser.add_argument("--log_scale_factor", type=float, default=10000.0)
    parser.add_argument("--topk_loss_weight", type=float, default=0.8)
    parser.add_argument("--topk_k", type=int, default=50)
    parser.add_argument(
        "--cv_folds", type=int, default=5, help="0 skips cross-validation"
    )
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seeds the predictor's weight initialisation; data collection is "
        "deterministic",
    )
    parser.add_argument(
        "--force", action="store_true", help="allow overwriting --output"
    )
    args = parser.parse_args()
    model_path = args.model or f"models/capsule_{args.dataset}_best_dynamic.pt"
    suffix = "" if args.dataset == "mnist" else f"_{args.dataset}"
    output = resolve_output(
        parser, args, f"models/ablation_predictor{suffix}.pt", ["model"]
    )

    guard_overwrite(output, args.force)
    guard_overwrite(training_data_path(output), args.force)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device()
    print(f"Device: {device}, seed: {args.seed}")
    print(f"CapsNet: {model_path} ({args.dataset} {ARCHITECTURES[args.dataset]})")
    try:
        capsule_model = load_capsule_model(model_path, args.dataset, device=device)
    except RuntimeError as e:
        raise SystemExit(f"Error loading model state_dict from {model_path}: {e}")

    test_dataset = test_set(args.dataset)
    split = split_indices(
        test_dataset, args.dataset, "surrogate_train", args.images_per_digit
    )
    # Starting each loader's iteration draws from the torch RNG, ahead of the
    # predictor's weight initialisation.
    loaders = {
        digit: DataLoader(Subset(test_dataset, split[digit].tolist()), batch_size=1)
        for digit in range(10)
    }

    data = collect_ablation_data(capsule_model, loaders, device)

    if args.cv_folds > 0:
        cross_validate(
            data,
            device,
            hidden_dim=args.hidden_dim,
            num_epochs=args.epochs,
            cv_folds=args.cv_folds,
            log_scale_factor=args.log_scale_factor,
            topk_loss_weight=args.topk_loss_weight,
            topk_k=args.topk_k,
        )

    train_full(
        data,
        device,
        hidden_dim=args.hidden_dim,
        num_epochs=args.epochs,
        log_scale_factor=args.log_scale_factor,
        topk_loss_weight=args.topk_loss_weight,
        topk_k=args.topk_k,
        output_path=output,
    )


if __name__ == "__main__":
    main()
