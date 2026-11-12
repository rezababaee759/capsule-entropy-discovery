"""SHAP DeepExplainer attribution of the trained ablation surrogate over all 16 of its
inputs.

The explained function is the surrogate's predicted importance
softplus(g_theta(...)) + 1e-8 on states sampled from its training data, each with its
own inputs (see TARGET_DEFINITION).

Usage:
    python experiments/surrogate_attribution.py --num_samples 1000 --seed 42 --force
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Tuple

import numpy as np
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F

from capsnet.io import (
    get_device,
    guard_overwrite,
    predictor_from_checkpoint,
    resolve_output,
)
from capsnet.provenance import attach
from capsnet.surrogate import (
    FEATURE_KEYS,
    NUM_CLASSES,
    OUTPUT_EPS,
    AblationPredictor,
    importance_from_raw,
)

# Columns of the per-state input matrix: these six slots, passed to
# AblationPredictor.forward by keyword, then the 10 digit one-hot columns.
INPUT_SLOTS = ("activation", "coupling", "entropy", "grad_a", "grad_c", "grad_H")
DIGIT_COLS = slice(len(INPUT_SLOTS), len(INPUT_SLOTS) + NUM_CLASSES)
NUM_INPUTS = len(INPUT_SLOTS) + NUM_CLASSES

# SHAP input groups; the digit group is the 10 one-hot columns.
SHAP_GROUPS = {
    "A": [0],
    "C": [1],
    "entropy": [2],
    "grad_a": [3],
    "grad_c": [4],
    "H": [5],
    "digit": list(range(DIGIT_COLS.start, DIGIT_COLS.stop)),
}

TARGET_DEFINITION = (
    "y_i = softplus(g_theta(A_i, C_i, E_i, grad_a_i, grad_c_i, H_i, onehot(digit_i))) + 1e-8, "
    "the surrogate predicted importance on each sampled capsule state. Every input is "
    "the state's own value at [row, capsule] of the training-data checkpoint "
    "(row = idx // n_capsules, capsule = idx % n_capsules), digit_i = digit_labels[row], and H is "
    "the grad_H input (delta_H). No cap is applied."
)

FEATURE_LEGEND = {
    "A": "activation input (capsule activation norm)",
    "C": "coupling input (routing coefficient to the image's class)",
    "entropy": "entropy input (normalized coupling entropy)",
    "grad_a": "grad_a input (stored log1p gradient w.r.t. activation)",
    "grad_c": "grad_c input (stored log1p gradient w.r.t. coupling)",
    "H": "grad_H input (delta_H, stored log1p entropy-gradient feature)",
    "digit": "digit one-hot input, onehot(digit_labels[row]); 10 columns",
}


def predictor_raw(model: AblationPredictor, inputs: torch.Tensor) -> torch.Tensor:
    """Raw (pre-softplus) predictor output g_theta for each row of `inputs`.

    inputs: [N, 16] in INPUT_SLOTS order followed by the digit one-hot. Each row is
    one capsule state, evaluated as a single-capsule batch entry ([N, 1] per slot),
    so no replication over capsules is involved. Returns [N].
    """
    slots = {name: inputs[:, k : k + 1] for k, name in enumerate(INPUT_SLOTS)}
    return model(**slots, digit_onehot=inputs[:, DIGIT_COLS]).squeeze(-1)


class ImportanceModel(nn.Module):
    """f(inputs) = softplus(g_theta(inputs)) + 1e-8, shape [N, 1].

    nn.Softplus is a module, not the functional form, so that SHAP DeepExplainer
    applies its nonlinear rule to it.
    """

    def __init__(self, base_model: AblationPredictor) -> None:
        super().__init__()
        self.base_model = base_model
        self.softplus = nn.Softplus()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return (
            self.softplus(predictor_raw(self.base_model, inputs)) + OUTPUT_EPS
        ).unsqueeze(1)


def build_state_inputs(
    data: Dict[str, Any], indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[N, 16] predictor inputs of the sampled states, plus their rows and capsules."""
    n_caps = data["activations"].shape[1]
    rows = indices // n_caps
    caps = indices % n_caps
    cols = [data[key].numpy()[rows, caps] for key in FEATURE_KEYS]
    digits = data["digit_labels"].numpy()[rows]
    onehot = np.eye(NUM_CLASSES, dtype=np.float32)[digits]
    inputs = np.column_stack(cols + [onehot]).astype(np.float32)
    return inputs, rows, caps


def check_state_inputs(
    data: Dict[str, Any],
    indices: np.ndarray,
    inputs: np.ndarray,
    rows: np.ndarray,
    caps: np.ndarray,
) -> Dict[str, Any]:
    """Assert that every predictor input is the sampled state's own stored value."""
    for k, key in enumerate(FEATURE_KEYS):
        flat = data[key].numpy().flatten()[indices]
        assert np.array_equal(
            inputs[:, k], flat
        ), f"{key}: fed value differs from the stored value"
        stored = data[key].numpy()[rows, caps]
        if np.any(stored != 0):
            assert np.any(
                inputs[:, k] != 0
            ), f"{key}: fed all-zero although stored values are non-zero"
    digits = data["digit_labels"].numpy()[rows]
    onehot = inputs[:, DIGIT_COLS]
    assert np.array_equal(
        onehot, np.eye(NUM_CLASSES, dtype=np.float32)[digits]
    ), "digit one-hot is not onehot(digit_labels[row])"
    assert np.all(onehot.sum(axis=1) == 1.0)
    if len(np.unique(digits)) > 1:
        assert (
            len(np.unique(onehot.argmax(axis=1))) > 1
        ), "digit one-hot is constant across states"
    return {
        "inputs_equal_stored_values": True,
        "digit_onehot_equals_onehot_of_digit_labels_row": True,
        "distinct_digits_sampled": int(len(np.unique(digits))),
    }


def check_predictor_calls(
    model: AblationPredictor,
    data: Dict[str, Any],
    inputs_t: torch.Tensor,
    rows: np.ndarray,
    caps: np.ndarray,
    device: torch.device,
) -> Dict[str, Any]:
    """Assert that (1) the per-state raw output equals entry [row, capsule] of the
    predictor applied to the full [rows, capsules] arrays, as in training; (2)
    perturbing input column k acts like perturbing forward()'s keyword
    INPUT_SLOTS[k] (or the one-hot); (3) the output is softplus(raw) + 1e-8 > 0."""
    out = {}
    with torch.no_grad():
        full = {
            name: data[key].to(device) for name, key in zip(INPUT_SLOTS, FEATURE_KEYS)
        }
        onehot_full = F.one_hot(
            data["digit_labels"].to(device), num_classes=NUM_CLASSES
        ).float()
        raw_full = model(**full, digit_onehot=onehot_full).cpu().numpy()
        raw_state = predictor_raw(model, inputs_t).cpu().numpy()
        gap = float(np.abs(raw_state - raw_full[rows, caps]).max())
        assert (
            gap <= 1e-4
        ), f"per-state raw output differs from the training-layout call by {gap}"
        out["max_abs_raw_diff_vs_training_layout_call"] = gap

        probe = inputs_t[:8].clone()
        base = predictor_raw(model, probe)
        for k, name in enumerate(INPUT_SLOTS):
            bumped = probe.clone()
            bumped[:, k] += 0.5
            via_matrix = predictor_raw(model, bumped)
            slots = {s: probe[:, j : j + 1] for j, s in enumerate(INPUT_SLOTS)}
            slots[name] = slots[name] + 0.5
            via_keyword = model(**slots, digit_onehot=probe[:, DIGIT_COLS]).squeeze(-1)
            assert torch.allclose(
                via_matrix, via_keyword, atol=1e-6
            ), f"slot {name} misrouted"
            assert not torch.allclose(
                via_matrix, base
            ), f"predictor does not respond to slot {name}"
        swapped = probe.clone()
        swapped[:, DIGIT_COLS] = torch.roll(probe[:, DIGIT_COLS], 1, dims=1)
        via_keyword = model(
            **{s: probe[:, j : j + 1] for j, s in enumerate(INPUT_SLOTS)},
            digit_onehot=swapped[:, DIGIT_COLS],
        ).squeeze(-1)
        assert torch.allclose(
            predictor_raw(model, swapped), via_keyword, atol=1e-6
        ), "digit one-hot misrouted"
        assert not torch.allclose(
            via_keyword, base
        ), "predictor does not respond to the digit one-hot"
        out["slot_order_matches_forward_keywords"] = True

        imp = ImportanceModel(model).to(device)
        y_probe = imp(probe).squeeze(1).double().cpu().numpy()
        raw_probe = base.double().cpu().numpy()
        expected = np.log1p(np.exp(raw_probe)) + OUTPUT_EPS
        assert np.all(y_probe > 0)
        assert np.allclose(
            y_probe, expected, rtol=1e-5, atol=1e-10
        ), "output transform is not softplus + 1e-8"
        assert torch.equal(imp(probe).squeeze(1), importance_from_raw(base))
        out["output_transform_is_softplus_plus_1e-8"] = True
    return out


def extract_shap_values(
    model: AblationPredictor,
    inputs: np.ndarray,
    device: torch.device,
    num_background: int = 100,
    num_test: int = 500,
) -> Dict[str, Any]:
    """SHAP DeepExplainer attributions of f = softplus(g_theta) + 1e-8 over all 16
    predictor inputs [N, 16], against a background of sampled states."""
    print("\nSHAP values")

    model.eval()
    wrapped_model = ImportanceModel(model).to(device)

    indices = np.random.choice(
        len(inputs), size=min(num_background, len(inputs)), replace=False
    )
    background = torch.tensor(inputs[indices], dtype=torch.float32, device=device)

    test_indices = np.random.choice(
        len(inputs), size=min(num_test, len(inputs)), replace=False
    )
    test_samples = torch.tensor(
        inputs[test_indices], dtype=torch.float32, device=device
    )

    print("Computing SHAP values...")
    print(f"  Background samples: {background.size(0)}")
    print(f"  Test samples: {test_samples.size(0)}")

    explainer = shap.DeepExplainer(wrapped_model, background)
    shap_values = explainer.shap_values(test_samples)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values).reshape(test_samples.size(0), NUM_INPUTS)

    # Additivity: the attributions of each test state sum to f(x) - E_background[f].
    with torch.no_grad():
        f_test = wrapped_model(test_samples).squeeze(1).double().cpu().numpy()
        f_bg_mean = float(wrapped_model(background).double().mean().item())
    additivity_gap = float(np.abs(shap_values.sum(axis=1) - (f_test - f_bg_mean)).max())
    print(f"  Max additivity gap: {additivity_gap:.3e}")

    # Group attribution: the SHAP values of a group's columns summed per test state;
    # group importance is the mean absolute value of that sum.
    group_importance = {
        name: float(np.abs(shap_values[:, cols].sum(axis=1)).mean())
        for name, cols in SHAP_GROUPS.items()
    }
    total = sum(group_importance.values())
    relative = {name: v / total for name, v in group_importance.items()}

    print("\nInput-group importance (mean |SHAP|) and share:")
    for name in SHAP_GROUPS:
        print(f"  {name}: {group_importance[name]:.6g} ({relative[name]*100:.1f}%)")

    column_names = list(INPUT_SLOTS) + [f"digit_{d}" for d in range(NUM_CLASSES)]
    shap_dict = {
        "method": "shap",
        "explainer": "DeepExplainer",
        "explained_function": "softplus(g_theta(all 16 inputs)) + 1e-8",
        "num_background": int(background.size(0)),
        "num_test": int(test_samples.size(0)),
        "input_columns": column_names,
        "groups": {
            name: [column_names[c] for c in cols] for name, cols in SHAP_GROUPS.items()
        },
        "group_rule": "per test state, the SHAP values of a group's columns are summed; "
        "feature_importance is the mean absolute value of that sum over test states",
        "expected_value_background_mean": f_bg_mean,
        "max_additivity_gap": additivity_gap,
        "feature_importance": group_importance,
        "relative_importance": relative,
        "per_column_mean_abs_shap": {
            name: float(v)
            for name, v in zip(column_names, np.abs(shap_values).mean(axis=0))
        },
        "shap_values_sample": shap_values[
            :10
        ].tolist(),  # first 10 test states, 16 columns
    }

    return shap_dict


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SHAP attribution of the ablation surrogate"
    )
    parser.add_argument("--predictor", default="models/ablation_predictor.pt")
    parser.add_argument(
        "--training_data", default="models/ablation_predictor_training_data.pt"
    )
    parser.add_argument(
        "--output",
        help="directory for interpretation_results.json "
        "(default: docs/figures/interpretation_output)",
    )
    parser.add_argument(
        "--num_samples", type=int, default=1000, help="capsule states to sample"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force", action="store_true", help="allow overwriting existing outputs"
    )

    args = parser.parse_args()
    inputs = ["predictor", "training_data", "num_samples", "seed"]
    output_dir = resolve_output(
        parser, args, "docs/figures/interpretation_output", inputs
    )
    guard_overwrite(os.path.join(output_dir, "interpretation_results.json"), args.force)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print(f"Random seed: {args.seed}")

    device = get_device()
    print(f"Using device: {device}")

    print(f"\nLoading predictor model from: {args.predictor}")
    checkpoint = torch.load(args.predictor, map_location=device, weights_only=False)
    predictor_model = predictor_from_checkpoint(checkpoint, device)
    assert (
        predictor_model.use_digit_features
    ), "the target needs the digit one-hot input"

    print(f"Loading training data from: {args.training_data}")
    data = torch.load(args.training_data, map_location="cpu", weights_only=False)

    # States are flat indices into the row-major [rows, capsules] arrays.
    N_total = data["activations"].numel()
    indices = np.random.choice(
        N_total, size=min(args.num_samples, N_total), replace=False
    )

    inputs, rows, caps = build_state_inputs(data, indices)
    digits = data["digit_labels"].numpy()[rows]
    checks = check_state_inputs(data, indices, inputs, rows, caps)
    inputs_t = torch.tensor(inputs, dtype=torch.float32, device=device)
    checks.update(
        check_predictor_calls(predictor_model, data, inputs_t, rows, caps, device)
    )

    print(f"Sampled states: {len(inputs)}")

    results = {
        "target_definition": TARGET_DEFINITION,
        "feature_legend": FEATURE_LEGEND,
        "sampled_states": {
            "n": int(len(inputs)),
            "distinct_rows": int(len(np.unique(rows))),
            "digit_counts": {
                str(d): int((digits == d).sum()) for d in range(NUM_CLASSES)
            },
        },
        "input_checks": checks,
    }
    results["shap"] = extract_shap_values(
        predictor_model, inputs, device, num_background=100
    )

    results_path = os.path.join(output_dir, "interpretation_results.json")

    attach(
        results,
        script="experiments/surrogate_attribution.py",
        inputs=[args.predictor, args.training_data],
        seeds={
            "--seed": args.seed,
            "note": "Seeds numpy and torch globally. The SHAP "
            "background and test index draws use the global "
            "numpy stream, so they depend on call order after "
            "seeding.",
        },
        sample_counts={"num_samples_requested": args.num_samples},
        method=(
            "SHAP DeepExplainer attribution of the ablation surrogate's predicted "
            "importance softplus(g_theta)+1e-8 over all predictor inputs, on states "
            "sampled from its training data, each with its own inputs."
        ),
        packages=["numpy", "shap"],
    )

    os.makedirs(output_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
