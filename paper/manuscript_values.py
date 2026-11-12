#!/usr/bin/env python3
"""Write the generated table bodies, docs/results_tables.tex, from the result artifacts.

Every value is read from a JSON artifact under results/; the generator only selects,
orders and formats. Thresholds, such as the threshold sweep's percentile cut and the
significance level, are read from the artifacts that used them.

    python paper/manuscript_values.py            # write docs/results_tables.tex
    python paper/manuscript_values.py --check    # fail if regeneration would differ
"""

from __future__ import annotations

import argparse
import json
import math

from capsnet import PROJECT_ROOT
from capsnet import taxonomy as tax
from capsnet.io import veg_beta_analysis

TABLES = PROJECT_ROOT / "docs" / "results_tables.tex"

BANNER = r"""%% =====================================================================
%%  GENERATED FILE -- DO NOT EDIT BY HAND
%% =====================================================================
%%  Produced by paper/manuscript_values.py
%%  Every value below is read from a result artifact under results/; README.md
%%  maps each table to its artifact. No value in this file was typed by a person,
%%  taken from a code comment, or copied from the manuscript.
%%
%%  Editing this file by hand defeats its purpose: the next regeneration will
%%  silently discard your edit. To change a number here, change the analysis that
%%  produces it and re-run its producer.
%%
%%  Regenerate:  python paper/manuscript_values.py
%%  Check only:  python paper/manuscript_values.py --check
%% =====================================================================
"""

ROLES = list(tax.ORDERED_ROLE_NAMES)
# (directory under results/data, beta-analysis tag, veg_ablation tag, label)
DATASETS = [
    ("mnist/pooled_42_123_789", "mnist", "mnist", "MNIST"),
    ("cifar10", "cifar", "cifar10", "CIFAR-10"),
]
# The roll-up calls the primary VEG model best_entropy_pt.
NATIVE_FILE = {"best_entropy_pt": "best_entropy"}


def artifact(path: str):
    """The JSON artifact at `path` (repository-relative); stops the run if it is missing."""
    p = PROJECT_ROOT / path
    if not p.is_file():
        raise SystemExit(f"table input {path} is missing")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        # An unreadable input would otherwise drop its table silently.
        raise SystemExit(f"table input {path} could not be read: {exc}") from exc


def fmt_p(p: float) -> str:
    """p-value as printed: 3 decimals at or above 0.001, otherwise mantissa x 10^exp."""
    if p < 1e-308:  # zero or subnormal: below double precision
        return "$<10^{-308}$"
    if p >= 0.001:
        return f"${p:.3f}$"
    exp = math.floor(math.log10(p))
    mant = p / 10**exp
    if round(mant, 1) >= 10:
        mant, exp = mant / 10, exp + 1
    return f"${mant:.1f}\\times10^{{{exp}}}$"


def sig2(x: float) -> str:
    """`x` to two significant figures, in fixed-point notation."""
    return f"{x:.{max(0, 1 - math.floor(math.log10(abs(x))))}f}"


def fmt_p_sig(p: float) -> str:
    """p-value to two significant figures; fmt_p's form below 0.001."""
    return f"${sig2(p)}$" if p >= 0.001 else fmt_p(p)


def fmt_p_bound(ps) -> str:
    """The largest of `ps` as a power-of-ten upper bound, $<10^{-k}$."""
    worst = max(ps)
    if worst < 1e-308:
        return "$<10^{-308}$"
    return f"$<10^{{-{math.floor(-math.log10(worst))}}}$"


def num(x: float, d: int, signed: bool = False) -> str:
    """`x` at `d` decimals in math mode, with an explicit sign if `signed`."""
    return f"${x:+.{d}f}$" if signed else f"${x:.{d}f}$"


def ci(lo: float, hi: float, d: int, scale: float = 1) -> str:
    """A bracketed interval, [lo, hi]."""
    return f"[{num(lo * scale, d)}, {num(hi * scale, d)}]"


def span(values, d: int) -> str:
    """Minimum to maximum of `values`: an en dash between non-negative ends, else "to"."""
    lo, hi = min(values), max(values)
    if lo >= 0:
        return f"{lo:.{d}f}--{hi:.{d}f}"
    return f"{num(lo, d)} to {num(hi, d, signed=hi > 0)}"


def per_digit_sparsity() -> str:
    """Per-digit sparsity, top-capsule ablation and surrogate Precision@50 (MNIST)."""
    sp = artifact("results/data/mnist/per_digit_sparsity_ground_truth.json")
    top_effects = artifact("results/data/mnist/top_capsule_ablation_effects.json")
    hv = artifact("results/data/mnist/held_out_validation.json")
    per = {int(r["digit"]): r for r in sp["per_digit"]}
    rows = []
    for d in sorted(per):
        g, u = per[d], top_effects["per_digit"][str(d)]
        lo, hi = u["excess_conf_drop_pct_ci95"]
        rows.append(
            f"        {d} & {g['gini']:.3f} & \\#{u['top_capsule']} & "
            f"{g['percent_80']:.1f} & "
            f"${u['top_conf_drop_pct_mean']:.2f} \\pm {u['top_conf_drop_pct_sd']:.2f}$ & "
            f"${u['excess_conf_drop_pct_mean']:.2f}$ (${lo:.2f}$--${hi:.2f}$) & "
            f"{hv['per_digit'][str(d)]['precision_at_50']:.2f} \\\\"
        )
    mean_g = sp["mean_over_digits"]["gini"]
    mean_80 = sp["mean_over_digits"]["percent_80"]
    mean_drop = sum(
        v["top_conf_drop_pct_mean"] for v in top_effects["per_digit"].values()
    ) / len(top_effects["per_digit"])
    rows.append(
        f"        \\textbf{{Mean}} & \\textbf{{{mean_g:.3f}}} & --- & "
        f"\\textbf{{{mean_80:.1f}}} & \\textbf{{{mean_drop:.2f}}} & --- & --- \\\\"
    )
    return "\n".join(rows)


def routing_correctness() -> str:
    """Pooled routing correctness by role, beside size-matched random capsule groups."""
    roles = [
        ("hub", "hubs", "Hubs"),
        ("specialist", "specialists", "Specialists"),
        ("random", None, "Non-critical (control)"),
    ]
    blocks = []
    for directory, _, _, label in DATASETS:
        g = artifact(f"results/data/{directory}/symmetric_evaluation_glmm.json")
        r = artifact(f"results/data/{directory}/random_capsule_baseline.json")
        rows = [f"        \\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{label}}}}} \\\\"]
        pooled = g["companion_analysis_capsule_level"]["pooled"]
        for role, group, role_label in roles:
            x = pooled[role]
            lo, hi = x["domain_gap_95ci"]
            cells = [
                str(x["n_capsules_pooled_across_seeds"]),
                f"{100 * x['within_mean']:.1f}",
                f"{100 * x['cross_mean']:.1f}",
                f"{100 * x['domain_gap']:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]",
                f"{fmt_p(x['wilcoxon_p'])} ({x['rank_biserial_r']:.2f})",
            ]
            if group:
                null = r["random_capsule_null"][group]["outside_role"]["pooled"]
                gap = null["gap_pp"]
                cells += [
                    f"{null['cross_pct']['null_mean']:.1f}",
                    f"{gap['null_mean']:.1f} [{gap['null_p2_5']:.1f}, {gap['null_p97_5']:.1f}]",
                ]
            else:
                cells += ["---", "---"]
            rows.append(f"        {role_label} & " + " & ".join(cells) + " \\\\")
        if label == "MNIST":  # the only dataset with a baseline resampling
            rs = artifact(f"results/data/{directory}/random_baseline_resampling.json")
            res, full = rs["random_baseline_gap_resampled"], rs["full_pool_no_sampling"]
            lo, hi = res["ci95_pp"]
            note = "        \\multicolumn{8}{@{}l}{\\scriptsize\\itshape %s} \\\\"
            rows.append(
                note % f"Non-critical, full pool resampled: {res['mean_pp']:.1f} pp, "
                f"95\\% resampling interval [{lo:.1f}, {hi:.1f}];"
            )
            rows.append(
                note % f"full pool, no sampling: {full['gap_pp']:.1f} pp "
                f"({full['n_capsule_seats']} capsule$\\times$model replicates)"
            )
        blocks.append("\n".join(rows))
    return "\n        \\midrule\n".join(blocks)


def routing_contrasts() -> str:
    """Role contrasts of the domain gap on the additive and log-odds scales."""
    sym, rnd = [], []
    for directory, _, _, _ in DATASETS:
        g = artifact(f"results/data/{directory}/symmetric_evaluation_glmm.json")
        r = artifact(f"results/data/{directory}/random_capsule_baseline.json")
        sym.append(g)
        rnd.append(r)

    def coefficient(g, term):
        c = next(c for c in g["primary_model"]["coefficients"] if c["term"] == term)
        return f"{num(c['coef'], 3, True)} {ci(*c['ci_95'], 3)}", fmt_p(c["p_value"])

    def log_odds(g, contrast):
        crm = g["ceiling_robust_model"]["role_contrasts_model_based"]
        c = next(c for c in crm if c["contrast"] == contrast)
        return (
            f"{num(c['estimate_log_odds'], 3, True)} {ci(*c['ci_95'], 3)}",
            fmt_p(c["p_value"]),
        )

    def per_model(g, contrast):
        het = g["ceiling_robust_model"]["between_model_heterogeneity"]
        c = next(c for c in het if c["contrast"] == contrast)
        n, sig = c["n_models"], sum(c["per_seed_significant_at_05"].values())
        return f"{c['n_models_same_sign_as_pooled']}/{n}, {sig}/{n}", "---"

    def wald(g, fit):
        s = g["sensitivity_digit_composition_confound"][fit]
        return f"{s['joint_wald_chi2']:.1f}", fmt_p(s["joint_wald_p"])

    def random_groups(r, group):
        x = r["random_capsule_null"][group]["outside_role"]["pooled"]["logit_gap"]
        return (
            f"{x['observed']:.2f} vs {x['null_mean']:.2f} "
            f"[{x['null_p2_5']:.2f}, {x['null_p97_5']:.2f}]",
            "---",
        )

    def contrast(role_a, role_b):
        return f"{role_a} vs {role_b} (excess domain gap, log-odds)"

    n_models = {
        c["n_models"] for g in sym for c in g["ceiling_robust_model"]["between_model_heterogeneity"]
    }
    if len(n_models) != 1:
        raise SystemExit("the per-model refits differ in their number of models")
    within = "C(condition)[T.within]"
    rows = [
        ("Additive: Hub $\\times$ within (vs control)",
         lambda g, r: coefficient(g, f"C(role)[T.hub]:{within}")),
        ("Additive: Specialist $\\times$ within (vs control)",
         lambda g, r: coefficient(g, f"C(role)[T.specialist]:{within}")),
        ("Log-odds: Hub $-$ control",
         lambda g, r: log_odds(g, contrast("hub", "random"))),
        ("Log-odds: Specialist $-$ control",
         lambda g, r: log_odds(g, contrast("specialist", "random"))),
        ("Log-odds: Hub $-$ Specialist",
         lambda g, r: log_odds(g, contrast("hub", "specialist"))),
        (f"Per model, Hub $-$ control: same sign / $p<0.05$ (of {n_models.pop()})",
         lambda g, r: per_model(g, contrast("hub", "random"))),
        ("Per model, Specialist $-$ control: same sign / $p<0.05$",
         lambda g, r: per_model(g, contrast("specialist", "random"))),
        ("Per model, Hub $-$ Specialist: same sign / $p<0.05$",
         lambda g, r: per_model(g, contrast("hub", "specialist"))),
        ("Joint Wald $\\chi^2$ (log-odds), unadjusted",
         lambda g, r: wald(g, "without_test_digit_control_logit")),
        ("Joint Wald $\\chi^2$ (log-odds), class-adjusted",
         lambda g, r: wald(g, "with_test_digit_control_logit")),
        ("Log-odds gap vs random groups: Hubs",
         lambda g, r: random_groups(r, "hubs")),
        ("Log-odds gap vs random groups: Specialists",
         lambda g, r: random_groups(r, "specialists")),
    ]
    out = []
    for label, cell in rows:
        cells = [c for g, r in zip(sym, rnd) for c in cell(g, r)]
        out.append(f"        {label} & " + " & ".join(cells) + " \\\\")
    return "\n".join(out)


def component_ablation(out: dict) -> None:
    """VEG component-ablation accuracies, with the seeds and epoch budget they share."""
    ca = artifact("results/veg_ablation/component_ablation.json")
    label = {
        "dynamic": "Baseline (Dynamic)",
        "entropy_uniform": "Uniform-$\\beta$",
        "entropy": "Per-Capsule-$\\beta$ (Ours)",
    }
    order = ["mnist", "fashionmnist", "cifar10"]
    rows = []
    for arm in ["dynamic", "entropy_uniform", "entropy"]:
        cells = []
        for ds in order:
            s = ca["results"].get(ds, {}).get("summary", {}).get(arm)
            cells.append(f"{s['mean']:.2f} $\\pm$ {s['std']:.2f}" if s else "---")
        rows.append(f"        & {label[arm]} & " + " & ".join(cells) + " \\\\")
    out["ComponentAblationBody"] = "\n".join(rows)
    out["ComponentAblationSeeds"] = ", ".join(str(s) for s in ca["seeds"])
    out["ComponentAblationEpochs"] = str(ca["epochs"])


def cifar_sparsity() -> str:
    """CIFAR-10 ground-truth sparsity and accuracy of the two routing strategies."""
    gt = artifact("results/data/cifar10/ground_truth_ablation.json")
    ca = artifact("results/veg_ablation/component_ablation.json")
    strategies = [
        ("Dynamic Routing", "cifar10", "dynamic", "Dynamic"),
        ("Entropy Routing (VEG)", "cifar10/veg_best_entropy", "entropy",
         "VEG Per-Capsule-$\\beta$ (Ours)"),
    ]
    rows = []
    for key, directory, arm, label in strategies:
        t = artifact(f"results/data/{directory}/hub_taxonomy.json")
        n_caps = sum(len(t[k]) for k in tax.ROLE_ORDER)
        g, acc = gt["results"][key], ca["results"]["cifar10"]["summary"][arm]
        rows.append(
            f"        {label} & {g['gini']:.2f} & {g['percent_80']:.1f} & "
            f"$\\sim${round(g['percent_80'] / 100 * n_caps)} & "
            f"{acc['mean']:.2f} $\\pm$ {acc['std']:.2f} \\\\"
        )
    return "\n".join(rows)


def gate_by_role() -> str:
    """Per-role capsule count, mean learned beta and realised gate, primary VEG models."""
    cols = {}
    for _, tag, ds, _ in DATASETS:
        b = artifact(f"results/beta_values/veg_beta_analysis_{tag}_native.json")
        ri = artifact(f"results/veg_ablation/role_intervention_{ds}.json")
        gv = artifact(f"results/veg_ablation/gate_variation_{ds}.json")
        m = b["per_model"]["best_entropy_pt"]
        n_images = gv["models"]["best_entropy"]["subset"]["n_images"]
        for r in ROLES:
            gate = ri["gate_distribution_by_role"][r]
            if gate["n_obs"] != m["role_counts"][r] * n_images:
                raise SystemExit(
                    f"{ds} {r}: {gate['n_obs']} gate observations, expected "
                    f"{m['role_counts'][r]} capsules x {n_images} images"
                )
            cols.setdefault(r, []).append(
                f"{m['role_counts'][r]} & {m['role_means'][r]:.3f} & "
                f"${gate['mean']:.3f} \\pm {gate['std']:.3f}$"
            )
    return "\n".join(f"        {r} & " + " & ".join(cols[r]) + " \\\\" for r in ROLES)


def gate_replication() -> str:
    """Replication of the beta-by-role statistics over the VEG models of each dataset."""
    cols, alphas = [], set()
    for _, tag, ds, _ in DATASETS:
        b = artifact(f"results/beta_values/veg_beta_analysis_{tag}_native.json")
        gv = artifact(f"results/veg_ablation/gate_variation_{ds}.json")
        ri = artifact(f"results/veg_ablation/role_intervention_{ds}.json")
        models = b["per_model"]
        native = {
            m: artifact(veg_beta_analysis(ds, NATIVE_FILE.get(m, m))) for m in models
        }
        n = len(models)
        ordered = sum(
            all(a < c for a, c in zip(means, means[1:]))
            for means in ([mm["role_means"][r] for r in ROLES] for mm in models.values())
        )
        corr = [mm["beta_vs_importance_pearson"] for mm in models.values()]
        alphas |= {mm["pairwise_multiplicity"]["alpha"] for mm in models.values()}
        alpha = max(alphas)
        last = str(max(int(k) for k in gv["across_models"]))
        primary = gv["models"]["best_entropy"]["subset"]["iterations"][last]["all_capsules"]
        across = gv["across_models"][last]
        hub_share = across["by_role"][tax.ROLE_NAMES["primary_hubs"]][
            "share_of_summed_within_capsule_var_logG"
        ]
        share_d = len(sig2(100 * hub_share["min"]).partition(".")[2])
        h = native["best_entropy_pt"]["entropy_statistics"]
        cv = [ri["beta_cross_seed_stability"][r]["cv"] for r in ROLES]
        cols.append(
            [
                f"{ordered} of {n}",
                span([mm["anova"]["f_stat"] for mm in models.values()], 1),
                fmt_p_bound([mm["anova"]["p_value"] for mm in models.values()]),
                span([mm["effect_size"]["eta_squared"] for mm in models.values()], 3),
                fmt_p_sig(
                    max(mm["primary_hub_vs_specialist"]["p_value"] for mm in models.values())
                ),
                span([c["pearson_r"] for c in corr], 3),
                f"{sum(c['pearson_p'] < alpha for c in corr)} of {n}",
                span([c["spearman_r"] for c in corr], 3),
                span(
                    [f["correlations"]["beta_vs_entropy"]["spearman_r"] for f in native.values()],
                    3,
                ),
                f"{h['mean']:.4f} ({h['std']:.4f})",
                f"{sig2(100 * primary['logG']['within_share'])}\\% "
                f"[{sig2(100 * across['within_share_logG']['max'])}\\%]",
                f"{100 * primary['H']['within_share']:.1f}\\%",
                f"{100 * hub_share['min']:.{share_d}f}--{100 * hub_share['max']:.{share_d}f}\\%",
                span(cv, 2),
            ]
        )
    if len(alphas) != 1:
        raise SystemExit("the beta analyses were run at different significance levels")
    labels = [
        "Models with full ordering PH $<$ SH $<$ Spec $<$ Gen $<$ NC",
        "ANOVA $F$ (range)",
        "ANOVA $p$ (largest)",
        "$\\eta^2$ (range)",
        "Primary Hub vs Specialist, Welch $p$ (largest)",
        "Pearson $r$ ($\\beta_i$, importance) (range)",
        f"Pearson $r$ significant at $\\alpha = {alphas.pop()}$ (models)",
        "Spearman $\\rho$ ($\\beta_i$, importance) (range)",
        "Spearman $\\rho$ ($\\beta_i$, mean $H_i$) (range)",
        "Mean $H_i$ across capsules, primary (SD)",
        "Within-capsule share of Var$(\\log G_i)$, primary [max over models]",
        "Within-capsule share of Var$(H_i)$, primary",
        "Primary Hubs' share of within-capsule Var$(\\log G_i)$ (range)",
        "CV of role-mean $\\beta_i$ across models (range over roles)",
    ]
    return "\n".join(
        f"        {lab} & " + " & ".join(c[i] for c in cols) + " \\\\"
        for i, lab in enumerate(labels)
    )


def interventions() -> str:
    """Test accuracy of the primary VEG models under inference-time beta overwrites."""
    gm = artifact("results/veg_ablation/role_intervention_mnist.json")
    gc = artifact("results/veg_ablation/role_intervention_cifar10.json")
    cond = [
        ("control_unmodified", "Control (unmodified $\\beta_i$)"),
        (
            "suppress_hubs_to_noncritical_mean",
            "Suppress Hubs $\\to$ Non-critical-level",
        ),
        (
            "suppress_specialists_to_noncritical_mean",
            "Suppress Specialists $\\to$ Non-critical-level",
        ),
        (
            "unsuppress_noncritical_to_hub_mean",
            "Un-suppress Non-critical $\\to$ Hub-level",
        ),
        ("hub_specialist_swap", "Hub$\\leftrightarrow$Specialist $\\beta_i$ swap"),
        ("zero_gate_all", "$\\beta_i = 0$ everywhere (unit gate, $G_i = 1$)"),
        ("zero_gate_hub_only", "$\\beta_i = 0$ for Hubs only (unit gate)"),
    ]
    ma, ca_ = gm["intervention_accuracy"], gc["intervention_accuracy"]
    base_m = ma["control_unmodified"]["accuracy"]
    base_c = ca_["control_unmodified"]["accuracy"]
    rows = []
    for key, lab in cond:
        am, ac = ma[key]["accuracy"], ca_[key]["accuracy"]
        dm = "---" if key == "control_unmodified" else f"${(am-base_m)*100:+.2f}$\\,pp"
        dc = "---" if key == "control_unmodified" else f"${(ac-base_c)*100:+.2f}$\\,pp"
        rows.append(
            f"        {lab} & {am*100:.2f}\\% & {dm} & {ac*100:.2f}\\% & {dc} \\\\"
        )
    return "\n".join(rows)


def intervention_groups() -> str:
    """Size, learned beta, displacement and importance share of each targeted group."""
    prof = artifact("results/data/role_importance_profile.json")
    groups = [
        ("Hub (Primary + Secondary)", "Hub(P+S)", "hub_delta_beta", "Hubs (Primary + Secondary)"),
        ("Specialist", "Specialist", "specialist_delta_beta", "Specialists"),
        ("Non-critical", "Non-critical", None, "Non-critical"),
    ]
    cols = {}
    for _, _, ds, _ in DATASETS:
        ri = artifact(f"results/veg_ablation/role_intervention_{ds}.json")
        share = prof["models"][f"{ds}_veg_primary"]["share_of_positive_importance"]
        disp = prof["intervention_displacement"][ds]
        learned = ri["role_mean_beta_learned"]
        if disp["target_beta_non_critical_mean"] != learned["Non-critical"]:
            raise SystemExit(
                f"{ds}: displacement target {disp['target_beta_non_critical_mean']} is not "
                f"the learned Non-critical mean {learned['Non-critical']}"
            )
        for group, beta_key, delta_key, _ in groups:
            delta = f"{disp[delta_key]:.3f}" if delta_key else "target"
            cols.setdefault(group, []).append(
                f"{share[group]['n_capsules']} & {learned[beta_key]:.3f} & {delta} & "
                f"{100 * share[group]['share']:.1f}"
            )
    return "\n".join(
        f"        {label} & " + " & ".join(cols[group]) + " \\\\"
        for group, _, _, label in groups
    )


def threshold_sweep() -> str:
    """Threshold sensitivity of the taxonomy on the VEG and dynamic-routing models."""
    veg = artifact("results/data/mnist/threshold_sensitivity.json")
    dyn = artifact("results/data/mnist/threshold_sensitivity_dynamic.json")
    beta = artifact("results/beta_values/veg_beta_analysis_mnist_native.json")
    taxonomy = artifact("results/data/mnist/hub_taxonomy.json")
    cut = veg["canonical_config"]["percentile"]
    if dyn["canonical_config"]["percentile"] != cut:
        raise SystemExit("the two threshold sweeps were run at different settings")

    def preserved(ts):
        s = ts["summary"]
        if not s["n_evaluable"]:
            return "---"
        return f"{s['beta_ordering_preserved']} ({100 * s['beta_ordering_preserved'] / s['n_evaluable']:.0f}\\%)"

    def preserved_high(ts):
        if not ts["summary"]["n_evaluable"]:
            return "---"
        high = [r for r in ts["results"] if r["percentile"] >= cut]
        ok = sum(r["beta_ordering_preserved"] is True for r in high)
        return f"{ok} / {len(high)} ({100 * ok / len(high):.0f}\\%)"

    def count_range(ts, key):
        lo, hi = ts["summary"][key]
        return f"{lo}--{hi} capsules"

    counts = beta["per_model"]["best_entropy_pt"]["role_counts"]
    rows = [
        ("Configurations tested", str(veg["summary"]["total_configs"]),
         str(dyn["summary"]["total_configs"])),
        ("Configurations preserving $\\beta$ ordering", preserved(veg), preserved(dyn)),
        (f"Preserved at the {cut}th percentile and above", preserved_high(veg),
         preserved_high(dyn)),
        ("Primary Hub count range", count_range(veg, "primary_hub_count_range"),
         count_range(dyn, "primary_hub_count_range")),
        ("Specialist count range", count_range(veg, "specialist_count_range"),
         count_range(dyn, "specialist_count_range")),
        ("Primary Hubs / Specialists at the setting used",
         f"{counts[tax.ROLE_NAMES['primary_hubs']]} / {counts[tax.ROLE_NAMES['specialists']]}",
         f"{len(taxonomy['primary_hubs'])} / {len(taxonomy['specialists'])}"),
    ]
    return "\n".join(f"        {a} & {b} & {c} \\\\" for a, b, c in rows)


def table_bodies() -> dict[str, str]:
    """LaTeX body of each results table, by name."""
    out = {}
    component_ablation(out)
    for name, build in [
        ("PerDigitSparsitySurrogateBody", per_digit_sparsity),
        ("RoutingCorrectnessBody", routing_correctness),
        ("RoutingContrastsBody", routing_contrasts),
        ("CifarSparsityBody", cifar_sparsity),
        ("GateByRoleBetaBody", gate_by_role),
        ("GateReplicationBody", gate_replication),
        ("InterventionBody", interventions),
        ("InterventionGroupsBody", intervention_groups),
        ("ThresholdSweepBothBody", threshold_sweep),
    ]:
        out[name] = build()
    return out


def render(tables: dict[str, str]) -> str:
    """Text of docs/results_tables.tex: one \\Res<name> macro per table body."""
    lines = [BANNER, ""]
    for name in sorted(tables):
        lines.append(f"%% ---- {name}")
        lines.append(f"\\newcommand{{\\Res{name}}}{{%")
        # A one-line value is used inline, so its line ends in % to keep the line
        # break out of the text; a table body ends in \\ and needs no guard.
        body = tables[name]
        lines.append(body if "\n" in body else body + "%")
        lines.append("}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if regeneration would change the output",
    )
    args = ap.parse_args()

    tables = table_bodies()
    text = render(tables)
    print(f"generated table bodies: {len(tables)}  ({', '.join(sorted(tables))})")

    if args.check:
        # A missing file counts as changed: regeneration would create it.
        if not TABLES.exists() or TABLES.read_bytes() != text.encode("utf-8"):
            print(f"\nCHECK FAILED -- regeneration would change: {TABLES.name}")
            return 1
        print(f"\ncheck: {TABLES.name} is up to date with the artifacts")
        return 0

    TABLES.parent.mkdir(parents=True, exist_ok=True)
    TABLES.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
