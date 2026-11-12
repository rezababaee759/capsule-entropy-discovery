"""Semantic tests of capsnet/surrogate.py.

Every check compares against an independent reference computed from the model, never
against source text:

  features   extract_capsule_features vs first principles: ||u_i||, argmax ||v_j||,
             c_{j*,i}, normalised entropy, grad_a from autograd on a fresh leaf,
             grad_c and grad_H from the closed form S = ||squash(sum_i c_i u_hat_i)_{j*}||^2
             with the final-iteration coupling as the leaf (not through model.digits).
  output     predict_importance(..., k) bitwise equal to softplus(raw) + 1e-8 where raw
             comes from calling the predictor by keyword with onehot(k), for all ten k;
             also close to the float64 log1p(exp(raw)) + 1e-8.
  class      the ten class indices do not all give the same output.
  wrappers   predict_capsule_importance and predict_and_ablate give outputs bitwise
             equal to the reference conditioned on the independently computed predicted
             class, with bitwise-equal features; predict_and_ablate's ablation targets
             equal an independent ablation, clipped at 0.
  arguments  invalid calls raise (TypeError / ValueError / KeyError), not assert.

Probe images are real calibration images selected by rule: misclassified images first,
then the highest predicted outputs from a scan of the slice. The suite checks its own
preconditions (a misclassified probe, maximum output above 5, two or more classes,
non-zero reference gradients), so it cannot pass vacuously.

The mutation self-check injects known defects into the API by monkeypatching its module
attributes, and requires the same suite to fail on each.

Needs the MNIST and CIFAR-10 checkpoints, surrogates and test sets; without them the
module is skipped, and the script exits 77 (skipped, nothing verified).

    python tests/test_surrogate_api.py      # nonzero exit on any failure or skip
    python -m pytest tests/test_surrogate_api.py
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # no bytecode for the modules imported below

import contextlib
import functools
import time
from typing import Callable, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

from capsnet import PROJECT_ROOT  # noqa: E402
from capsnet import data as capsdata  # noqa: E402
from capsnet import surrogate as API  # noqa: E402
from capsnet.data import slice_for  # noqa: E402
from capsnet.io import (
    get_device,
    load_capsule_model,
    predictor_from_checkpoint,
)  # noqa: E402
from capsnet.model import squash  # noqa: E402

DEVICE = get_device()
N_MISCLASSIFIED = 3
N_HIGH_OUTPUT = 3
SCAN_STRIDE = 10
MIN_PROBE_MAX = 5.0
HELDOUT_ABLATION_CHECKS = 16

CONFIGS = {
    "mnist": dict(
        capsule_model="models/capsule_mnist_best_dynamic.pt",
        predictor="models/ablation_predictor.pt",
        dataset=lambda: datasets.MNIST(
            root=capsdata.DATA_ROOT,
            train=False,
            download=False,
            transform=capsdata.test_transform("mnist"),
        ),
    ),
    "cifar10": dict(
        capsule_model="models/capsule_cifar10_best_dynamic.pt",
        predictor="models/ablation_predictor_cifar10.pt",
        dataset=lambda: datasets.CIFAR10(
            root=capsdata.DATA_ROOT,
            train=False,
            download=False,
            transform=capsdata.test_transform("cifar10"),
        ),
    ),
}


def missing_inputs() -> List[str]:
    """Checkpoints and test sets the suite needs that are not on disk."""
    missing = []
    for name, cfg in CONFIGS.items():
        for key in ("capsule_model", "predictor"):
            if not (PROJECT_ROOT / cfg[key]).is_file():
                missing.append(cfg[key])
        try:
            cfg["dataset"]()
        except RuntimeError:
            missing.append(f"data/ ({name} test set)")
    return missing


MISSING = missing_inputs()
if MISSING and "pytest" in sys.modules:
    import pytest

    pytest.skip(
        "SKIPPED, nothing verified -- missing: " + ", ".join(MISSING),
        allow_module_level=True,
    )


@functools.lru_cache(maxsize=None)
def load(name: str):
    cfg = CONFIGS[name]
    cm = load_capsule_model(PROJECT_ROOT / cfg["capsule_model"], name, device=DEVICE)
    ck = torch.load(
        PROJECT_ROOT / cfg["predictor"], map_location=DEVICE, weights_only=False
    )
    return cm, predictor_from_checkpoint(ck, DEVICE), ck


@functools.lru_cache(maxsize=None)
def probes(name: str):
    """[(dataset index, image [1,C,H,W], true label)] chosen by rule from the calibration slice."""
    cm, pr, _ = load(name)
    ds = CONFIGS[name]["dataset"]()
    targets = [int(t) for t in ds.targets]
    start, stop = slice_for(name, "taxonomy_calibration", 50)
    by_class = {d: [i for i, t in enumerate(targets) if t == d] for d in range(10)}
    calib = [i for d in range(10) for i in by_class[d][start:stop]]

    misclassified = []
    with torch.no_grad():
        for i in calib:
            v = cm(ds[i][0].unsqueeze(0).to(DEVICE))[0]
            if int(torch.norm(v[0], dim=-1).argmax()) != targets[i]:
                misclassified.append(i)
            if len(misclassified) == N_MISCLASSIFIED:
                break

    scores = []
    for i in calib[::SCAN_STRIDE]:
        image = ds[i][0].unsqueeze(0).to(DEVICE)
        cf = API.extract_capsule_features(cm, image)
        scores.append(
            (
                float(
                    API.predict_importance(pr, cf.features, cf.predicted_class).max()
                ),
                i,
            )
        )
    high = [i for _, i in sorted(scores, reverse=True) if i not in misclassified][
        :N_HIGH_OUTPUT
    ]
    return [
        (i, ds[i][0].unsqueeze(0).to(DEVICE), targets[i]) for i in misclassified + high
    ]


def signed_log1p(x):
    return torch.sign(x) * torch.log1p(x.abs())


def reference_features(cm, image):
    eps = 1e-8
    with torch.no_grad():
        u = cm.primary(F.relu(cm.conv1(image)))
        v, c, _, _ = cm.digits(u)
    j = int(torch.norm(v[0], dim=-1).argmax())
    c0 = c[0]
    activations = u[0].norm(dim=-1)
    couplings = c0[j]
    entropies = -(c0 * torch.log2(c0 + eps)).sum(0) / np.log2(10)
    with torch.enable_grad():
        leaf_u = u.detach().clone().requires_grad_(True)
        v_u, _, _, _ = cm.digits(leaf_u)
        grad_a = torch.autograd.grad(v_u[0, j].pow(2).sum(), leaf_u)[0][0].norm(dim=-1)
        W = cm.digits.routing_module.W
        u_hat = torch.matmul(W, u.detach().unsqueeze(1).unsqueeze(-1)).squeeze(
            -1
        )  # [1, O, I, D]
        leaf_c = c.detach().clone().requires_grad_(True)
        s = (leaf_c.unsqueeze(-1) * u_hat).sum(dim=2)
        dS_dc = torch.autograd.grad(squash(s)[0, j].pow(2).sum(), leaf_c)[0][
            0
        ]  # [O, I]
    grad_c = dS_dc[j].abs()
    h = -(torch.log2(c0.clamp(min=eps)) + 1.0)
    grad_H = (dS_dc * h / (h.norm(dim=0, keepdim=True) + eps)).sum(0).abs()
    raw_grads = {"grad_a": grad_a, "grad_c": grad_c, "grad_H": grad_H}
    feats = {
        "activations": activations,
        "couplings": couplings,
        "entropies": entropies,
        **{k: signed_log1p(g) for k, g in raw_grads.items()},
    }
    return feats, j, raw_grads


def reference_raw(pr, feats, k):
    """Predictor called by keyword with onehot(k); independent of FEATURE_KEYS' order."""
    onehot = torch.eye(10, device=DEVICE)[k : k + 1]
    with torch.no_grad():
        return pr(
            activation=feats["activations"][None].to(DEVICE),
            coupling=feats["couplings"][None].to(DEVICE),
            entropy=feats["entropies"][None].to(DEVICE),
            grad_a=feats["grad_a"][None].to(DEVICE),
            grad_c=feats["grad_c"][None].to(DEVICE),
            grad_H=feats["grad_H"][None].to(DEVICE),
            digit_onehot=onehot,
        )[0]


def reference_importance(pr, feats, k):
    return F.softplus(reference_raw(pr, feats, k)) + 1e-8


def reference_ablation(cm, image, j, capsules):
    """Confidence drop on ||v_j|| when each listed capsule is zeroed, clipped at 0."""
    with torch.no_grad():
        u = cm.primary(F.relu(cm.conv1(image)))
        full = torch.norm(cm.digits(u)[0][0], dim=-1)[j].item()
        out = []
        for i in capsules:
            ua = u.clone()
            ua[0, i, :] = 0.0
            out.append(full - torch.norm(cm.digits(ua)[0][0], dim=-1)[j].item())
    return np.clip(np.array(out, dtype=np.float32), 0, None)


_heldout_cache: Dict[tuple, tuple] = {}


def _heldout_run(cm, pr, image):
    key = (
        id(cm),
        id(pr),
        id(API.extract_capsule_features),
        id(API.predict_importance),
        id(API.predict_and_ablate),
        id(image),
    )
    if key not in _heldout_cache:
        preds, targets = API.predict_and_ablate(cm, pr, [image[0].cpu()], DEVICE)
        _heldout_cache[key] = (preds[0], targets[0])
    return _heldout_cache[key]


def semantic_suite(name: str) -> List[str]:
    """Return a list of failures (empty when every check passes)."""
    fails: List[str] = []
    cm, pr, _ = load(name)
    probe_set = probes(name)
    probe_max = 0.0
    misclassified_seen = False
    heldout_checked = False

    if len({y for _, _, y in probe_set}) < 2:
        fails.append(f"{name}: probe set covers fewer than 2 classes")

    for idx, image, y in probe_set:
        tag = f"{name}[{idx}]"
        rf, j, raw_grads = reference_features(cm, image)
        if j != y:
            misclassified_seen = True
        for k, g in raw_grads.items():
            if float(g.abs().sum()) == 0.0:
                fails.append(f"{tag}: reference {k} is all-zero")

        # features
        cf = API.extract_capsule_features(cm, image)
        if cf.predicted_class != j:
            fails.append(f"{tag}: predicted_class {cf.predicted_class} != argmax {j}")
        for k in API.FEATURE_KEYS:
            a, b = cf.features.get(k), rf[k]
            if (
                a is None
                or a.shape != b.shape
                or not torch.allclose(a, b, rtol=1e-4, atol=1e-6)
            ):
                fails.append(f"{tag}: feature {k} differs from first principles")

        # output vs direct keyword invocation, all ten classes
        outputs = []
        for k in range(10):
            got = API.predict_importance(pr, cf.features, k)
            raw = reference_raw(pr, cf.features, k)
            exact = F.softplus(raw) + 1e-8
            f64 = torch.log1p(torch.exp(raw.double())) + 1e-8
            if got.shape != exact.shape or not torch.equal(got, exact):
                fails.append(
                    f"{tag}: predict_importance(class {k}) != softplus(raw)+1e-8 bitwise"
                )
                break
            if not torch.allclose(got.double(), f64, rtol=1e-6, atol=1e-12):
                fails.append(
                    f"{tag}: predict_importance(class {k}) != float64 softplus"
                )
                break
            outputs.append(got)
            probe_max = max(probe_max, float(exact.max()))
        if len(outputs) == 10 and all(torch.equal(outputs[0], o) for o in outputs[1:]):
            fails.append(f"{tag}: output does not depend on class_index")

        # predict_capsule_importance (one-hot = predicted class)
        expected_j = reference_importance(pr, cf.features, j).cpu()
        imp, feats, meta = API.predict_capsule_importance(cm, pr, image)
        if not torch.equal(imp, expected_j):
            fails.append(f"{tag}: predict_capsule_importance != reference at j*")
        if meta["predicted_digit"] != j:
            fails.append(f"{tag}: predict_capsule_importance predicted_digit != argmax")
        if not torch.equal(meta["routing_distribution"], cf.coupling.cpu()):
            fails.append(
                f"{tag}: predict_capsule_importance routing_distribution != "
                "final-iteration coupling"
            )
        if not all(
            torch.equal(feats[k], cf.features[k].cpu()) for k in API.FEATURE_KEYS
        ):
            fails.append(f"{tag}: predict_capsule_importance features differ")

        # predict_and_ablate (MNIST only), on the first misclassified probe
        if name == "mnist" and j != y and not heldout_checked:
            heldout_checked = True
            preds, targets = _heldout_run(cm, pr, image)
            if not np.array_equal(preds, expected_j.numpy()):
                fails.append(
                    f"{tag}: predict_and_ablate prediction != reference at j* "
                    "(one-hot class)"
                )
            half = HELDOUT_ABLATION_CHECKS // 2
            top = np.argsort(-expected_j.numpy())[:half].tolist()
            spread = list(range(0, targets.size, targets.size // half))[:half]
            caps = sorted(set(top + spread))
            ref_t = reference_ablation(cm, image, j, caps)
            if not np.array_equal(targets[caps], ref_t):
                fails.append(
                    f"{tag}: predict_and_ablate targets != independent ablation on j*"
                )

    if not misclassified_seen:
        fails.append(f"{name}: probe set has no misclassified image")
    if probe_max <= MIN_PROBE_MAX:
        fails.append(f"{name}: probe max output {probe_max:.3f} <= {MIN_PROBE_MAX}")
    if name == "mnist" and not heldout_checked:
        fails.append("mnist: predict_and_ablate was not exercised")
    return fails


def argument_suite() -> List[str]:
    """Invalid calls must raise a non-assert exception."""
    fails: List[str] = []
    cm, pr, _ = load("mnist")
    _, image, _ = probes("mnist")[0]
    cf = API.extract_capsule_features(cm, image)
    f = cf.features

    class NoOneHot(nn.Module):
        use_digit_features = False

        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1, device=DEVICE))

    class NonFinite(NoOneHot):
        use_digit_features = True

        def forward(self, *xs, digit_onehot=None):
            return torch.full_like(xs[0], float("nan"))

    ragged = dict(f, grad_H=f["grad_H"][:-1])
    two_d = {k: v[None] for k, v in f.items()}
    f64 = dict(f, couplings=f["couplings"].double())
    cases: Dict[str, Callable[[], object]] = {
        "missing class_index": lambda: API.predict_importance(pr, f),
        "bool class_index": lambda: API.predict_importance(pr, f, True),
        "float class_index": lambda: API.predict_importance(pr, f, 3.0),
        "class_index 10": lambda: API.predict_importance(pr, f, 10),
        "class_index -1": lambda: API.predict_importance(pr, f, -1),
        "missing key": lambda: API.predict_importance(
            pr, {k: v for k, v in f.items() if k != "grad_c"}, 0
        ),
        "ragged shapes": lambda: API.predict_importance(pr, ragged, 0),
        "2-D features": lambda: API.predict_importance(pr, two_d, 0),
        "float64 feature": lambda: API.predict_importance(pr, f64, 0),
        "use_digit_features=False": lambda: API.predict_importance(NoOneHot(), f, 0),
        "non-finite output": lambda: API.predict_importance(NonFinite(), f, 0),
        "batch of two images": lambda: API.extract_capsule_features(
            cm, image.repeat(2, 1, 1, 1)
        ),
    }
    for label, call in cases.items():
        try:
            call()
        except AssertionError:
            fails.append(f'argument check "{label}" uses assert')
        except (TypeError, ValueError, KeyError):
            continue
        else:
            fails.append(f'argument check "{label}" did not raise')
    if (
        not API.predict_importance(pr, f, np.int64(cf.predicted_class)).shape
        == f["activations"].shape
    ):
        fails.append("numpy integer class_index rejected")

    x = torch.linspace(-30.0, 30.0, 61, requires_grad=True)
    y = API.importance_from_raw(x)
    if not torch.equal(y, F.softplus(x) + 1e-8):
        fails.append("importance_from_raw != softplus + 1e-8")
    if not bool((y > 0).all()):
        fails.append("importance_from_raw not strictly positive")
    y.sum().backward()
    if not torch.allclose(x.grad, torch.sigmoid(x.detach())):
        fails.append("importance_from_raw gradient != sigmoid")
    return fails


@contextlib.contextmanager
def patched(attr: str, value, modules=(API,)):
    saved = [(m, getattr(m, attr)) for m in modules if hasattr(m, attr)]
    try:
        for m, _ in saved:
            setattr(m, attr, value)
        _heldout_cache.clear()
        yield
    finally:
        for m, orig in saved:
            setattr(m, attr, orig)
        _heldout_cache.clear()


def _predict_post(post):
    real = API.predict_importance
    return lambda p, f, k: post(real(p, f, k))


def _extract_edit(edit):
    real = API.extract_capsule_features

    def ex(cm, image):
        cf = real(cm, image)
        feats = dict(cf.features)
        edit(feats, cf)
        return API.CapsuleFeatures(
            feats, cf.predicted_class, cf.confidence, cf.primary_out, cf.coupling
        )

    return ex


def _raw_output(p, f, k):
    onehot = F.one_hot(torch.tensor([k]), 10).float().to(DEVICE)
    with torch.no_grad():
        return p(
            *(f[x].unsqueeze(0).to(DEVICE) for x in API.FEATURE_KEYS),
            digit_onehot=onehot,
        )[0]


def _swap(a, b):
    def edit(feats, cf):
        feats[a], feats[b] = feats[b], feats[a]

    return edit


def _zero(key):
    def edit(feats, cf):
        feats[key] = torch.zeros_like(feats[key])

    return edit


def _coupling_class_0(feats, cf):
    feats["couplings"] = cf.coupling[0]


def _predict_swap_ac():
    real = API.predict_importance

    def pi(p, f, k):
        g = dict(f)
        g["activations"], g["couplings"] = f["couplings"], f["activations"]
        return real(p, g, k)

    return pi


def _inside(wrapper_name: str, class_of):
    """`wrapper_name` with predict_importance conditioned on class_of(j*) instead of j*."""
    real = API.predict_importance
    wrapper = getattr(API, wrapper_name)

    def run(*args, **kwargs):
        with patched("predict_importance", lambda p, f, k: real(p, f, class_of(k))):
            return wrapper(*args, **kwargs)

    return run


def mutants():
    real_predict = API.predict_importance
    return {
        "fixed class 0": lambda: patched(
            "predict_importance", lambda p, f, k: real_predict(p, f, 0)
        ),
        "one-hot index reversed (9-k)": lambda: patched(
            "predict_importance", lambda p, f, k: real_predict(p, f, 9 - k)
        ),
        "predict_and_ablate one-hot on another class": lambda: patched(
            "predict_and_ablate", _inside("predict_and_ablate", lambda k: (k + 1) % 10)
        ),
        "predict_capsule_importance one-hot class 0": lambda: patched(
            "predict_capsule_importance",
            _inside("predict_capsule_importance", lambda k: 0),
        ),
        "zero grad_a": lambda: patched(
            "extract_capsule_features", _extract_edit(_zero("grad_a"))
        ),
        "zero grad_c": lambda: patched(
            "extract_capsule_features", _extract_edit(_zero("grad_c"))
        ),
        "swapped feature slots grad_a<->grad_c": lambda: patched(
            "extract_capsule_features", _extract_edit(_swap("grad_a", "grad_c"))
        ),
        "swapped predictor slots A<->C": lambda: patched(
            "predict_importance", _predict_swap_ac()
        ),
        "coupling row of class 0": lambda: patched(
            "extract_capsule_features", _extract_edit(_coupling_class_0)
        ),
        "raw output": lambda: patched("predict_importance", _raw_output),
        "clamp(max=2.2025)": lambda: patched(
            "predict_importance",
            _predict_post(lambda t: t.clamp(max=float(np.expm1(10.0) / 1e4))),
        ),
        "clamp(max=5)": lambda: patched(
            "predict_importance", _predict_post(lambda t: t.clamp(max=5.0))
        ),
        "multiplicative rescale x0.5": lambda: patched(
            "predict_importance", _predict_post(lambda t: t * 0.5)
        ),
        "log1p/expm1 round trip": lambda: patched(
            "predict_importance",
            _predict_post(lambda t: torch.expm1(torch.log1p(t * 1e4)) / 1e4),
        ),
    }


def mutation_check() -> Dict[str, List[str]]:
    """For each mutant, the failures the suite reports on MNIST + CIFAR-10 (must be non-empty)."""
    out = {}
    for label, make in mutants().items():
        with make():
            found = []
            for name in CONFIGS:
                try:
                    found += semantic_suite(name)
                except (
                    Exception
                ) as e:  # a crash under a mutant also counts as detection
                    found.append(f"{name}: {type(e).__name__}: {e}")
        out[label] = found
    return out


def require_none(problems, what: str) -> None:
    """Fail with every problem listed; an explicit raise, so python -O cannot skip it."""
    if problems:
        raise AssertionError(f"{what}:\n  " + "\n  ".join(map(str, problems)))


def test_mnist_semantics() -> None:
    require_none(semantic_suite("mnist"), "mnist semantics")


def test_cifar10_semantics() -> None:
    require_none(semantic_suite("cifar10"), "cifar10 semantics")


def test_argument_validation() -> None:
    require_none(argument_suite(), "argument validation")


def test_mutants_are_caught() -> None:
    missed = [m for m, f in mutation_check().items() if not f]
    require_none(missed, "mutants not detected")


if __name__ == "__main__":
    if MISSING:
        print("SKIPPED -- nothing verified. Missing: " + ", ".join(MISSING))
        sys.exit(77)
    t0 = time.time()
    failed = False
    for label, fn in [
        ("mnist", lambda: semantic_suite("mnist")),
        ("cifar10", lambda: semantic_suite("cifar10")),
        ("arguments", argument_suite),
    ]:
        fails = fn()
        failed |= bool(fails)
        print(
            ("FAIL " if fails else "PASS ")
            + label
            + "".join(f"\n    {f}" for f in fails)
        )
    for name in CONFIGS:
        print(f"probes {name}: " + ", ".join(f"{i}(y={y})" for i, _, y in probes(name)))
    for label, found in mutation_check().items():
        failed |= not found
        print(
            ("caught " if found else "MISSED ")
            + f"mutant {label}"
            + (f": {found[0]}" if found else "")
        )
    print(f'{"FAILED" if failed else "OK"} in {time.time() - t0:.1f}s on {DEVICE}')
    sys.exit(1 if failed else 0)
