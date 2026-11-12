#!/usr/bin/env python3
"""Time one instance-level explanation stage by stage: the forward pass, the
gradient features (grad_a, grad_c, grad_H), the surrogate MLP alone, and the full
pipeline end to end.

Usage:
    python experiments/explanation_cost_benchmark.py \
        --output results/explanation_cost_benchmark.json
"""

import argparse
import json
import platform
import time

import numpy as np
import torch
import torch.nn.functional as F

from capsnet.data import test_set
from capsnet.io import get_device, load_capsule_model, load_predictor
from capsnet.surrogate import (
    FEATURE_KEYS,
    extract_capsule_features,
    gradient_features,
    predict_importance,
)

MODELS = {
    "mnist": ("models/capsule_mnist_best_dynamic.pt", "models/ablation_predictor.pt"),
    "cifar10": (
        "models/capsule_cifar10_best_dynamic.pt",
        "models/ablation_predictor_cifar10.pt",
    ),
}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark_stage(fn, n_warmup, n_trials, device):
    for _ in range(n_warmup):
        fn()
    sync(device)
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)  # ms
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times, ddof=1)),
        "median_ms": float(np.median(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "n_trials": n_trials,
    }


def run_benchmark(dataset_name, device, n_warmup=20, n_trials=200):
    print(f"\n{dataset_name.upper()}")
    capsule_model_path, predictor_path = MODELS[dataset_name]
    model = load_capsule_model(capsule_model_path, dataset_name, device=device)
    predictor = load_predictor(predictor_path, device)

    image = test_set(dataset_name)[0][0].unsqueeze(0).to(device)
    num_caps = model.digits.num_input_caps

    # Stage A: forward pass only (no_grad), base features (activation/coupling/entropy)
    def stage_forward_only():
        with torch.no_grad():
            primary_out = model.primary(F.relu(model.conv1(image)))
            digit_caps, c_ij, _, _ = model.digits(primary_out)
            _ = torch.norm(primary_out[0], dim=-1)
            lengths = torch.norm(digit_caps[0], dim=-1)
            _ = lengths.argmax().item()

    cf = extract_capsule_features(model, image)

    # Stage B: gradient feature extraction (grad_a, grad_c, grad_H)
    def stage_gradients():
        gradient_features(model, cf.primary_out, cf.predicted_class)

    # Stage C: MLP surrogate forward pass only
    inputs = [cf.features[k].unsqueeze(0) for k in FEATURE_KEYS]
    digit_onehot = (
        F.one_hot(torch.tensor([cf.predicted_class]), num_classes=10).float().to(device)
    )

    def stage_mlp_only():
        with torch.no_grad():
            predictor(*inputs, digit_onehot=digit_onehot)

    # Stage D: full end-to-end pipeline (features, gradients, surrogate), single call
    def stage_full_pipeline():
        full = extract_capsule_features(model, image)
        predict_importance(predictor, full.features, full.predicted_class)

    results = {
        "dataset": dataset_name,
        "num_capsules": num_caps,
        "device": str(device),
        "forward_pass_only_ms": benchmark_stage(
            stage_forward_only, n_warmup, n_trials, device
        ),
        "gradient_features_only_ms": benchmark_stage(
            stage_gradients, n_warmup, n_trials, device
        ),
        "mlp_surrogate_only_ms": benchmark_stage(
            stage_mlp_only, n_warmup, n_trials, device
        ),
        "full_pipeline_ms": benchmark_stage(
            stage_full_pipeline, n_warmup, n_trials, device
        ),
    }
    for k, v in results.items():
        if isinstance(v, dict):
            print(
                f"  {k:28s} mean={v['mean_ms']:.4f}ms  median={v['median_ms']:.4f}ms  std={v['std_ms']:.4f}ms"
            )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/explanation_cost_benchmark.json")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    out = {ds: run_benchmark(ds, device) for ds in ("mnist", "cifar10")}
    out["environment"] = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "python_version": platform.python_version(),
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
