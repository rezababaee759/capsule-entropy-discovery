#!/usr/bin/env python3
"""Image-to-image variation of the realised VEG gate G_in = exp(-beta_i * H_in) on frozen
checkpoints: exact ddof=0 splits of the variance of log G, G and H on the capsule x image
grid (between capsules; image main effect; capsule x image interaction), overall and per
native role, at the last two routing iterations. For the primary model the per-role gate
statistics of role_intervention_{dataset}.json are recomputed and compared.

    python experiments/gate_variation.py --dataset mnist
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np
import torch

from capsnet import PROJECT_ROOT
from capsnet.data import test_loader
from capsnet.io import (
    VEG_PRIMARY_TAG,
    get_device,
    load_capsule_model,
    veg_checkpoint,
    veg_taxonomy,
)
from capsnet.provenance import attach, file_fingerprint
from capsnet.taxonomy import ORDERED_ROLE_NAMES, assert_role_coverage, capsule_roles
from capsnet.veg import GATE_BATCH, coupling_entropy, gate_distribution, gate_summary

ROLES = list(ORDERED_ROLE_NAMES)
EPS = 1e-8  # the entropy epsilon of capsnet/model.py
LN10 = math.log(10.0)
DEFAULT_MODELS = [VEG_PRIMARY_TAG, "seed42", "seed123", "seed456"]


@torch.no_grad()
def collect(model, loader, device, max_images):
    """Per-image H and stored G at every routing iteration; max_images=None reads all."""
    model.eval()
    beta = model.digits.routing_module.beta.detach()
    H: dict[int, list] = {}
    G: dict[int, list] = {}
    recompute_maxdiff = 0.0
    seen = 0
    for images, _ in loader:
        if max_images is not None and seen >= max_images:
            break
        images = images.to(device)
        _, _, _, hist = model(images)
        for k, step in enumerate(hist):
            Hk = coupling_entropy(step["coupling_coeffs"], EPS)  # [B, I]
            Gk = step["entropy_gate"].squeeze(1)  # [B, I]
            recompute_maxdiff = max(
                recompute_maxdiff, (Gk - torch.exp(-beta * Hk)).abs().max().item()
            )
            H.setdefault(k, []).append(Hk.cpu().numpy())
            G.setdefault(k, []).append(Gk.cpu().numpy())
        seen += images.size(0)
    return {
        "H": {k: np.concatenate(v) for k, v in H.items()},
        "G": {k: np.concatenate(v) for k, v in G.items()},
        "beta": beta.cpu().numpy().astype(np.float64),
        "gate_recompute_max_abs_diff": recompute_maxdiff,
    }


def decompose(X) -> dict:
    """Exact population (ddof=0) variance split on a balanced [image, capsule] grid."""
    X = np.asarray(X, dtype=np.float64)
    m = X.mean()
    a = X.mean(axis=0)  # per-capsule means
    b = X.mean(axis=1)  # per-image means
    total = float(((X - m) ** 2).mean())
    between = float(((a - m) ** 2).mean())
    within = float(X.var(axis=0).mean())
    image = float(((b - m) ** 2).mean())
    interaction = float(((X - a[None, :] - b[:, None] + m) ** 2).mean())
    tol = 1e-9 * max(total, 1e-300)
    assert abs(total - between - within) <= tol, "between + within != total"
    assert abs(within - image - interaction) <= tol, "image + interaction != within"

    def share(v):
        return v / total if total > 0 else None

    return {
        "grand_mean": float(m),
        "total_var": total,
        "between_capsule_var": between,
        "within_capsule_var": within,
        "image_main_var": image,
        "interaction_var": interaction,
        "within_share": share(within),
        "three_way_shares": {
            "capsule": share(between),
            "image": share(image),
            "interaction": share(interaction),
        },
    }


def quantiles(x) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {
        "median": float(np.median(x)),
        "q25": float(np.percentile(x, 25)),
        "q75": float(np.percentile(x, 75)),
        "max": float(x.max()),
    }


def role_indices(cap_role: dict, n_caps: int) -> dict:
    idx = {r: [] for r in ROLES}
    for c in range(n_caps):
        idx[cap_role[c]].append(c)
    return {r: np.array(v, dtype=int) for r, v in idx.items()}


def analyse_iteration(Hk, Gk_stored, beta, ridx) -> dict:
    """All summaries for one routing iteration on one image set."""
    H64 = Hk.astype(np.float64)
    LG = -beta[None, :] * H64
    G = np.exp(LG)
    sd_LG = LG.std(axis=0)
    sd_G = G.std(axis=0)
    mean_G = G.mean(axis=0)
    cv_G = sd_G / mean_G
    var_LG = LG.var(axis=0)
    summed_within = var_LG.sum()

    by_role = {}
    for r in ROLES:
        ix = ridx[r]
        if ix.size == 0:
            continue
        by_role[r] = {
            "n_capsules": int(ix.size),
            "beta_mean": float(beta[ix].mean()),
            "beta_min": float(beta[ix].min()),
            "beta_max": float(beta[ix].max()),
            "n_negative_beta": int((beta[ix] < 0).sum()),
            "share_of_summed_within_capsule_var_logG": float(
                var_LG[ix].sum() / summed_within
            ),
            "within_share": {
                "logG": decompose(LG[:, ix])["within_share"],
                "G": decompose(G[:, ix])["within_share"],
                "H": decompose(H64[:, ix])["within_share"],
            },
            "per_capsule_sd_logG": quantiles(sd_LG[ix]),
            "per_capsule_sd_G": quantiles(sd_G[ix]),
            "per_capsule_cv_G": quantiles(cv_G[ix]),
        }

    primary_hubs = [
        {
            "capsule": int(c),
            "beta": float(beta[c]),
            "static_G_exp_neg_beta_ln10": float(math.exp(-beta[c] * LN10)),
            "mean_G": float(mean_G[c]),
            "sd_G": float(sd_G[c]),
            "min_G": float(G[:, c].min()),
            "max_G": float(G[:, c].max()),
            "cv_G": float(cv_G[c]),
            "sd_logG": float(sd_LG[c]),
            "mean_H": float(H64[:, c].mean()),
            "min_H": float(H64[:, c].min()),
        }
        for c in ridx["Primary Hub"]
    ]

    return {
        "n_images": int(H64.shape[0]),
        "all_capsules": {
            "logG": decompose(LG),
            "G": decompose(G),
            "H": decompose(H64),
            "per_capsule_sd_logG": quantiles(sd_LG),
            "per_capsule_sd_G": quantiles(sd_G),
            "per_capsule_cv_G": quantiles(cv_G),
        },
        "by_role": by_role,
        "primary_hubs": primary_hubs,
        "checks": {
            "logG_vs_log_of_stored_gate_max_abs_diff": float(
                np.abs(LG - np.log(Gk_stored.astype(np.float64))).max()
            ),
        },
    }


def analyse_image_set(data, n_images, beta, ridx, iterations) -> dict:
    return {
        str(k): analyse_iteration(
            data["H"][k][:n_images], data["G"][k][:n_images], beta, ridx
        )
        for k in iterations
    }


def role_gate_stats_from_arrays(G_final, cap_role) -> dict:
    """Per-role gate statistics in the concatenation order of gate_distribution()."""
    per = {r: [] for r in ROLES}
    for start in range(0, G_final.shape[0], GATE_BATCH):
        g = G_final[start : start + GATE_BATCH]
        for c, r in cap_role.items():
            per[r].append(g[:, c])
    return {
        r: gate_summary(np.concatenate(chunks)) for r, chunks in per.items() if chunks
    }


def compare_role_stats(recomputed, committed) -> dict:
    max_diff = 0.0
    n_obs_equal = set(recomputed) == set(committed)
    for r in committed:
        if r not in recomputed:
            continue
        n_obs_equal &= recomputed[r]["n_obs"] == committed[r]["n_obs"]
        for k in ("mean", "median", "std", "min", "max"):
            max_diff = max(max_diff, abs(recomputed[r][k] - committed[r][k]))
    return {"max_abs_diff": max_diff, "n_obs_all_equal": bool(n_obs_equal)}


def reproduction_block(
    dataset,
    model,
    loader,
    device,
    cap_role,
    data,
    n_sub,
    max_images,
    ckpt,
    tax_path,
    final,
) -> dict:
    committed_path = f"results/veg_ablation/role_intervention_{dataset}.json"
    p = PROJECT_ROOT / committed_path
    if not p.exists():
        return {"committed_artifact": committed_path, "status": "ABSENT"}
    committed = json.loads(p.read_text())
    committed_stats = committed["gate_distribution_by_role"]
    cprov = committed.get("_provenance", {})
    committed_max = (cprov.get("sample_counts") or {}).get("max_gate_images_requested")
    via_fn = gate_distribution(model, loader, device, cap_role, max_images=max_images)
    via_arrays = role_gate_stats_from_arrays(data["G"][final][:n_sub], cap_role)
    committed_inputs = {i["path"]: i["sha256"] for i in cprov.get("inputs", [])}
    return {
        "committed_artifact": committed_path,
        "committed_field": "gate_distribution_by_role",
        "committed_max_gate_images_requested": committed_max,
        "max_images_requested": max_images,
        "same_image_set": committed_max == max_images,
        "checkpoint_sha256_matches_committed_input": committed_inputs.get(ckpt)
        == file_fingerprint(ckpt)["sha256"],
        "taxonomy_sha256_matches_committed_input": committed_inputs.get(tax_path)
        == file_fingerprint(tax_path)["sha256"],
        "recomputed_via_gate_distribution": via_fn,
        "recomputed_from_arrays": via_arrays,
        "gate_distribution_vs_committed": compare_role_stats(via_fn, committed_stats),
        "arrays_vs_committed": compare_role_stats(via_arrays, committed_stats),
    }


def span(values) -> dict:
    values = list(values)
    return {"min": min(values), "max": max(values), "n_models": len(values)}


def across_models(models: dict, iterations) -> dict:
    """Minimum and maximum over the analysed models of the headline shares."""
    out = {}
    for k in map(str, iterations):
        alls = [m["subset"]["iterations"][k]["all_capsules"] for m in models.values()]
        it = {
            f"within_share_{q}": span(a[q]["within_share"] for a in alls)
            for q in ("logG", "G", "H")
        }
        it["total_var_H"] = span(a["H"]["total_var"] for a in alls)
        it["image_share_logG"] = span(
            a["logG"]["three_way_shares"]["image"] for a in alls
        )
        roles = {}
        for r in ROLES:
            present = [
                m["subset"]["iterations"][k]["by_role"][r]
                for m in models.values()
                if r in m["subset"]["iterations"][k]["by_role"]
            ]
            if present:
                roles[r] = {
                    "share_of_summed_within_capsule_var_logG": span(
                        p["share_of_summed_within_capsule_var_logG"] for p in present
                    ),
                    "within_share_logG": span(
                        p["within_share"]["logG"] for p in present
                    ),
                }
        it["by_role"] = roles
        out[k] = it
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", choices=["mnist", "cifar10"], required=True)
    ap.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="model tags: best_entropy or seed<N>",
    )
    ap.add_argument(
        "--max_images",
        type=int,
        default=2000,
        help="read whole batches of 256 until at least this many images",
    )
    ap.add_argument(
        "--full_test_models",
        nargs="*",
        default=[VEG_PRIMARY_TAG],
        help="tags also analysed on the full test set",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="default results/veg_ablation/gate_variation_{dataset}.json",
    )
    args = ap.parse_args()
    ds = args.dataset
    out_rel = args.output or f"results/veg_ablation/gate_variation_{ds}.json"
    unknown = set(args.full_test_models) - set(args.models)
    if unknown:
        ap.error(f"--full_test_models not in --models: {sorted(unknown)}")

    device = get_device()
    loader = test_loader(ds, GATE_BATCH)
    n_test = len(loader.dataset)
    n_sub = min(math.ceil(args.max_images / GATE_BATCH) * GATE_BATCH, n_test)
    print(f"Device: {device}  Dataset: {ds}  images: {n_sub} (full test set {n_test})")

    inputs = []
    models = {}
    for tag in args.models:
        ckpt, tax_path = veg_checkpoint(ds, tag), veg_taxonomy(ds, tag)
        inputs += [ckpt, tax_path]
        print(f"  {tag}: {ckpt}", flush=True)
        model = load_capsule_model(PROJECT_ROOT / ckpt, ds, "entropy", device)
        cap_role = capsule_roles(json.loads((PROJECT_ROOT / tax_path).read_text()))
        n_caps = model.digits.routing_module.num_input_caps
        assert_role_coverage(cap_role, n_caps)
        ridx = role_indices(cap_role, n_caps)
        full = tag in args.full_test_models

        data = collect(model, loader, device, None if full else args.max_images)
        beta = data["beta"]
        final = max(data["H"])
        iterations = (final - 1, final)
        n_collected = data["H"][final].shape[0]
        assert n_collected == (n_test if full else n_sub)

        G0 = data["G"][0].astype(np.float64)
        H0 = data["H"][0].astype(np.float64)
        static = np.exp(-beta * LN10)
        entry = {
            "checkpoint": ckpt,
            "taxonomy": tax_path,
            "n_capsules": int(n_caps),
            "role_counts": {r: int(ridx[r].size) for r in ROLES},
            "beta": {
                "mean": float(beta.mean()),
                "sd": float(beta.std()),
                "min": float(beta.min()),
                "max": float(beta.max()),
                "n_negative": int((beta < 0).sum()),
            },
            "checks": {
                "n_images": int(n_collected),
                "gate_recompute_max_abs_diff": data["gate_recompute_max_abs_diff"],
                "iteration0_max_abs_H0_minus_ln10": float(np.abs(H0 - LN10).max()),
                "iteration0_max_abs_G0_minus_exp_neg_beta_ln10": float(
                    np.abs(G0 - static[None, :]).max()
                ),
                "iteration0_max_within_capsule_sd_G0": float(G0.std(axis=0).max()),
            },
            "subset": {
                "n_images": int(n_sub),
                "iterations": analyse_image_set(data, n_sub, beta, ridx, iterations),
            },
        }
        if full:
            entry["full_test"] = {
                "n_images": int(n_collected),
                "iterations": analyse_image_set(
                    data, n_collected, beta, ridx, iterations
                ),
            }
        if tag == VEG_PRIMARY_TAG:
            entry["reproduction"] = reproduction_block(
                ds,
                model,
                loader,
                device,
                cap_role,
                data,
                n_sub,
                args.max_images,
                ckpt,
                tax_path,
                final,
            )
            rep = entry["reproduction"]
            if rep.get("status") != "ABSENT":
                inputs.append(rep["committed_artifact"])
            if "gate_distribution_vs_committed" in rep:
                print(
                    "    reproduction max |diff|: gate_distribution "
                    f"{rep['gate_distribution_vs_committed']['max_abs_diff']}, arrays "
                    f"{rep['arrays_vs_committed']['max_abs_diff']}",
                    flush=True,
                )
        models[tag] = entry
        del data, model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = {
        "dataset": ds,
        "image_set": {
            "subset": f"first {n_sub} test images in loader order "
            f"(whole batches of {GATE_BATCH} until max_images={args.max_images} is reached)",
            "full_test": f"all {n_test} test images, for {sorted(args.full_test_models)}",
            "loader": "capsnet.data test loader, shuffle=False"
            + ("; normalized=True" if ds == "cifar10" else ""),
        },
        "definitions": {
            "iterations": "routing iteration index; the largest key is the final "
            "iteration, whose gate weights the output",
            "logG": "-beta_i * H_in in float64, H recomputed from coupling_coeffs "
            f"with eps={EPS}",
            "variance": "population (ddof=0) on the balanced capsule x image grid; "
            "total = between_capsule + within_capsule; within_capsule = image_main + "
            "interaction",
            "within_share": "within_capsule_var / total_var",
            "share_of_summed_within_capsule_var_logG": "sum over the role's capsules "
            "of Var_images(log G) / the same sum over all capsules",
            "per_capsule": "quantiles over the role's capsules of each capsule's "
            "across-image SD (ddof=0) or CV = SD / mean",
            "roles": "native to each model's own hub taxonomy; not matched across models",
        },
        "settings": {
            "device": str(device),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "models": models,
        "across_models": across_models(models, iterations),
    }

    attach(
        out,
        script="experiments/gate_variation.py",
        inputs=inputs,
        seeds={
            "rng": None,
            "trained_model_tags": list(args.models),
            "note": "No RNG: inference only on frozen checkpoints in eval mode over an "
            "unshuffled test loader, with library defaults left untouched as in "
            "experiments/role_intervention.py.",
        },
        sample_counts={
            "max_images_requested": args.max_images,
            "n_images_subset": int(n_sub),
            "n_images_full_test": int(n_test),
            "full_test_models": sorted(args.full_test_models),
            "batch_size": GATE_BATCH,
            "n_models": len(models),
            "n_capsules": {t: m["n_capsules"] for t, m in models.items()},
            "capsules_per_role": {t: m["role_counts"] for t, m in models.items()},
        },
        method=(
            "Inference-only on frozen VEG checkpoints. Per-image coupling entropy H "
            "and log gate -beta*H at the last two routing iterations; exact ddof=0 "
            "split of the variance of log G, G and H on the balanced capsule x image "
            "grid into between-capsule, image main effect and capsule x image "
            "interaction, overall and within each native role, plus quantiles of "
            "per-capsule SD and CV. Descriptive only: no hypothesis test and no "
            "interval."
        ),
        packages=["numpy", "torchvision"],
    )

    out_path = PROJECT_ROOT / out_rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
