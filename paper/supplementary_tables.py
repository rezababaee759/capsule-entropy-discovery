#!/usr/bin/env python3
"""Write the supplementary tables, docs/supplementary_tables.tex, from result artifacts.

Every value is read from a committed artifact and only formatted; nothing is recomputed.

    python paper/supplementary_tables.py            # write docs/supplementary_tables.tex
    python paper/supplementary_tables.py --check    # exit 1 if the file is out of date
"""

from __future__ import annotations

import argparse
import json
import sys

from capsnet import PROJECT_ROOT
from capsnet import taxonomy as tax
from capsnet.io import veg_beta_analysis
from manuscript_values import fmt_p, fmt_p_sig

ACC = PROJECT_ROOT / "results/veg_ablation/accuracy_significance.json"
ABLATION = PROJECT_ROOT / "results/veg_ablation/component_ablation.json"
UNIFORM_BETA = PROJECT_ROOT / "results/veg_ablation/uniform_beta_checkpoints.json"
BETA = {
    "MNIST": PROJECT_ROOT / "results/beta_values/veg_beta_analysis_mnist_native.json",
    "CIFAR-10": PROJECT_ROOT
    / "results/beta_values/veg_beta_analysis_cifar_native.json",
}
SYM = {
    "MNIST": PROJECT_ROOT
    / "results/data/mnist/pooled_42_123_789/symmetric_evaluation_glmm.json",
    "CIFAR-10": PROJECT_ROOT / "results/data/cifar10/symmetric_evaluation_glmm.json",
}
OUT = PROJECT_ROOT / "docs/supplementary_tables.tex"

DATASETS = [
    ("mnist", "MNIST"),
    ("fashionmnist", "FashionMNIST"),
    ("cifar10", "CIFAR-10"),
]
ARM = {
    "dynamic": "Baseline",
    "entropy_uniform": "Uniform-$\\beta$",
    "entropy": "Per-Capsule-$\\beta$",
}
ARM_ROW = {
    "dynamic": "Baseline (Dynamic)",
    "entropy_uniform": "Uniform-$\\beta$",
    "entropy": "Per-Capsule-$\\beta$",
}
# Row order and orientation (minuend, subtrahend) used in the table.
PAIRS = [
    ("entropy_uniform", "dynamic"),
    ("entropy", "dynamic"),
    ("entropy_uniform", "entropy"),
]
# (per_model key of the roll-up, VEG model tag, label)
MODELS = [
    ("best_entropy_pt", "best_entropy", "Primary"),
    ("seed42", "seed42", "Seed 42"),
    ("seed123", "seed123", "Seed 123"),
    ("seed456", "seed456", "Seed 456"),
]
DATASET_KEY = {"MNIST": "mnist", "CIFAR-10": "cifar10"}
NUMBER_WORDS = "zero one two three four five six seven eight nine ten".split()
ROLES = [tax.ROLE_NAMES[key] for key in tax.ROLE_ORDER]
SYM_ROLES = [("hub", "Hubs"), ("specialist", "Specialists"), ("random", "Non-critical")]
CONTRASTS = [
    ("hub vs random", "Hub $-$ control"),
    ("specialist vs random", "Spec.\\ $-$ control"),
    ("hub vs specialist", "Hub $-$ Spec."),
]
# (dataset, directory under results/data, model label); the first model of each dataset
# is the primary checkpoint, the slot the symmetric evaluation calls seed 42.
TAXONOMIES = {
    "Dynamic routing": [
        ("MNIST", "mnist", "Primary"),
        ("MNIST", "mnist/seed123", "Seed 123"),
        ("MNIST", "mnist/seed789", "Seed 789"),
        ("CIFAR-10", "cifar10", "Primary"),
        ("CIFAR-10", "cifar10/seed123", "Seed 123"),
        ("CIFAR-10", "cifar10/seed456", "Seed 456"),
    ],
    "Per-capsule-$\\beta$ VEG": [
        (ds, f"{d}/veg_{tag}", label)
        for ds, d in (("MNIST", "mnist"), ("CIFAR-10", "cifar10"))
        for tag, label in (
            ("best_entropy", "Primary"),
            ("seed42", "Seed 42"),
            ("seed123", "Seed 123"),
            ("seed456", "Seed 456"),
        )
    ],
}


def yes(flag) -> str:
    return "yes" if flag else "no"


def interval(x, lohi) -> str:
    return f"${x:.3f}$ [${lohi[0]:.3f}$, ${lohi[1]:.3f}$]"


def table_s3(acc: dict, ablation: dict, uniform: dict) -> str:
    """Per-seed accuracy and learned shared beta, and the paired accuracy contrasts."""
    seeds = ablation["seeds"]
    beta = {(m["dataset"], m["seed"]): m["beta"] for m in uniform["per_model"]}
    arows = []
    for ds, ds_label in DATASETS:
        summary = ablation["results"][ds]["summary"]
        for i, arm in enumerate(ARM):
            by_seed = dict(zip(summary[arm]["seeds"], summary[arm]["all"]))
            first = f"\\multirow{{{len(ARM) + 1}}}{{*}}{{{ds_label}}}" if i == 0 else ""
            arows.append(
                f"{first} & {ARM_ROW[arm]} & "
                + " & ".join(f"{by_seed[s]:.2f}" for s in seeds)
                + " \\\\"
            )
        arows.append(
            f" & {ARM_ROW['entropy_uniform']}: learned $\\beta$ & "
            + " & ".join(f"{beta[(ds, s)]:.3f}" for s in seeds)
            + " \\\\"
        )
        arows.append("\\midrule")
    arows.pop()  # no rule after the last block

    by_pair = {(c["dataset"], c["arm_a"], c["arm_b"]): c for c in acc["contrasts"]}
    rows = []
    for ds, ds_label in DATASETS:
        rows.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{ds_label}}}}} \\\\")
        for minuend, subtrahend in PAIRS:
            if (ds, subtrahend, minuend) in by_pair:  # stored as minuend - subtrahend
                c, sign = by_pair[(ds, subtrahend, minuend)], 1
            elif (ds, minuend, subtrahend) in by_pair:  # stored the other way round
                c, sign = by_pair[(ds, minuend, subtrahend)], -1
            else:
                raise SystemExit(
                    f"accuracy_significance.json has no {ds} {minuend}/{subtrahend} contrast"
                )
            rows.append(
                f"{ARM[minuend]} $-$ {ARM[subtrahend]} & "
                f"${sign * c['mean_diff_pp']:+.3f}$ & ${sign * c['cohens_dz']:+.2f}$ & "
                f"{fmt_p(c['p_paired_t'])} & ${c['p_wilcoxon']:.4f}$ & "
                f"{yes(c['significant_bh'])} & {yes(c['significant_holm'])} & "
                f"{fmt_p(c['p_tost'])} \\\\"
            )
    n, q = acc["family_size"], acc["fdr_q"]
    alpha = acc["equivalence"]["alpha"]
    margin = acc["equivalence"]["sesoi_pp"]
    return "\n".join(
        [
            "\\begin{table}[!htbp]",
            "    \\centering",
            "    \\caption{Accuracy of the VEG component ablation by training seed (main-text "
            "Table~\\MainTabAccuracy). \\textbf{(a)} Best-epoch test accuracy (\\%) of each arm for each "
            f"of the {NUMBER_WORDS[len(seeds)]} training seeds and, for Uniform-$\\beta$, the learned "
            "shared $\\beta$ of each run's best-accuracy checkpoint. \\textbf{(b)} Paired contrasts: "
            "$\\Delta$ is the mean paired difference in percentage points, $d_z$ Cohen's paired effect "
            "size, $p$ the paired $t$-test and Wilcoxon $p$ the exact two-sided signed-rank test "
            "(zero differences dropped), a "
            f"distribution-free companion that cannot reach $\\alpha = {alpha}$ at $n = {acc['n_seeds']}$ "
            "(main-text Appendix~\\MainAppStatistics). The "
            f"{NUMBER_WORDS[n]} $t$-test contrasts form one family, corrected with Benjamini--Hochberg "
            f"at $q = {q}$ and, alongside, Holm--Bonferroni at $\\alpha = {alpha}$. TOST $p$ is the "
            "larger of the two one-sided tests of equivalence within the "
            f"$\\pm {margin}$\\,pp margin; $p < {alpha}$ establishes equivalence.}}",
            "    \\label{tabS:accuracy_contrasts}",
            "    \\scriptsize",
            "    \\setlength{\\tabcolsep}{4pt}",
            "    \\textbf{(a)} Accuracy and learned shared $\\beta$ by training seed\\\\[2pt]",
            "    \\begin{tabular}{@{}llrrrrr@{}}",
            "        \\toprule",
            f"        & & \\multicolumn{{{len(seeds)}}}{{c}}{{\\textbf{{Training seed}}}} \\\\",
            f"        \\cmidrule(l){{3-{2 + len(seeds)}}}",
            "        \\textbf{Dataset} & \\textbf{Arm} & "
            + " & ".join(f"\\textbf{{{s}}}" for s in seeds)
            + " \\\\",
            "        \\midrule",
            *("        " + r for r in arows),
            "        \\bottomrule",
            "    \\end{tabular}",
            "",
            "    \\medskip",
            "    \\textbf{(b)} Paired contrasts\\\\[2pt]",
            "    \\begin{tabular}{@{}lrrrrccr@{}}",
            "        \\toprule",
            "        \\textbf{Contrast} & \\textbf{$\\Delta$ (pp)} & \\textbf{$d_z$} & "
            "\\textbf{$p$} & \\textbf{Wilcoxon $p$} & \\textbf{BH} & \\textbf{Holm} & "
            "\\textbf{TOST $p$} \\\\",
            "        \\midrule",
            *("        " + r for r in rows),
            "        \\bottomrule",
            "    \\end{tabular}",
            "\\end{table}",
        ]
    )


def table_s4(beta: dict) -> str:
    """Mean learned beta per role in each VEG model, with its ANOVA and effect sizes."""
    rows, brows, crows = [], [], {key: [] for key, _, _ in MODELS}
    for ds_label, d in beta.items():
        brows.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{ds_label}}}}} \\\\")
        for i, (key, tag, label) in enumerate(MODELS):
            m = d["per_model"][key]
            ph = m["primary_hub_vs_specialist"]
            if not ph["p_value"] == ph["p_value_bh"] == ph["p_value_holm"]:
                sys.exit(
                    f"{ds_label} {label}: the Primary Hub vs Specialist p is not the largest of "
                    "its post-hoc family, so the caption's statement that adjustment leaves it "
                    "unchanged does not hold."
                )
            means = " & ".join(f"${m['role_means'][r]:.3f}$" for r in ROLES)
            first = f"\\multirow{{{len(MODELS)}}}{{*}}{{{ds_label}}}" if i == 0 else ""
            rows.append(
                f"{first} & {label} & {means} & ${m['anova']['f_stat']:.1f}$ & "
                f"{fmt_p(m['anova']['p_value'])} & "
                f"{fmt_p(m['primary_hub_vs_specialist']['p_value'])} \\\\"
            )
            es, w, corr = m["effect_size"], m["welch_anova"], m["beta_vs_importance_pearson"]
            native = json.loads(
                (PROJECT_ROOT / veg_beta_analysis(DATASET_KEY[ds_label], tag)).read_text()
            )
            brows.append(
                f"{label} & "
                f"{interval(es['eta_squared'], es['eta_squared_ci95'])} & "
                f"{interval(es['omega_squared'], es['omega_squared_ci95'])} & "
                f"${w['f_stat']:.1f}$ (${w['df_num']:.0f}$, ${w['df_denom']:.1f}$) & "
                f"{fmt_p_sig(w['p_value'])} & {fmt_p_sig(m['levene']['p_value'])} \\\\"
            )
            crows[key].append(
                f"${corr['pearson_r']:.3f}$ & ${corr['spearman_r']:.3f}$ & "
                f"${native['correlations']['beta_vs_entropy']['spearman_r']:.3f}$"
            )
        rows.append("\\midrule")
    rows.pop()  # no rule after the last block
    corr_rows = [f"{label} & " + " & ".join(crows[key]) + " \\\\" for key, _, label in MODELS]
    corr_head = " & ".join(
        f"\\multicolumn{{3}}{{c}}{{\\textbf{{{ds_label}}}}}" for ds_label in beta
    )
    return "\n".join(
        [
            "\\begin{table}[!htbp]",
            "    \\centering",
            "    \\caption{$\\beta_i$ by functional role in each trained VEG model (main-text "
            "Section~\\MainSecVegMechanism{} and Table~\\MainTabGates), with roles derived natively on "
            "each model by exhaustive single-capsule ablation. \\textbf{(a)} Mean learned $\\beta_i$ per "
            f"role, one-way ANOVA across the {NUMBER_WORDS[len(ROLES)]} roles, and the Primary Hub versus "
            "Specialist Welch "
            "$t$-test, whose $p$ is the largest of the model's three post-hoc contrasts, so "
            "Benjamini--Hochberg and Holm adjustment leave it unchanged. \\textbf{(b)} Effect sizes and sensitivity statistics: $\\eta^2$ and $\\omega^2$, "
            "each with its 95\\% CI; Welch's ANOVA; the Brown--Forsythe test of equal variances; "
            "Pearson $r$ and Spearman $\\rho$ between $\\beta_i$ and ablation importance; and Spearman "
            "$\\rho$ between $\\beta_i$ and mean routing entropy. ``Primary'' is the primary "
            "per-capsule-$\\beta$ checkpoint; $p < 10^{-308}$ marks a value below double precision.}",
            "    \\label{tabS:beta_by_role}",
            "    \\scriptsize",
            "    \\setlength{\\tabcolsep}{3.5pt}",
            "    \\textbf{(a)} Mean learned $\\beta_i$ and ANOVA\\\\[2pt]",
            "    \\begin{tabular}{@{}llccccccc c@{}}",
            "        \\toprule",
            "        & & \\multicolumn{5}{c}{\\textbf{Mean $\\beta_i$}} & \\multicolumn{2}{c}{\\textbf{ANOVA}} & "
            "\\textbf{PH vs.\\ Spec.} \\\\",
            "        \\cmidrule(lr){3-7}\\cmidrule(lr){8-9}",
            "        \\textbf{Dataset} & \\textbf{Model} & \\textbf{PH} & \\textbf{SH} & \\textbf{Spec.} & "
            "\\textbf{Gen.} & \\textbf{NC} & \\textbf{$F$} & \\textbf{$p$} & \\textbf{Welch $p$} \\\\",
            "        \\midrule",
            *("        " + r for r in rows),
            "        \\bottomrule",
            "    \\end{tabular}",
            "",
            "    \\smallskip",
            "    {\\scriptsize PH, Primary Hub; SH, Secondary Hub; Spec., Specialist; Gen., Generalist; "
            "NC, Non-critical.}",
            "",
            "    \\medskip",
            "    \\textbf{(b)} Effect sizes and sensitivity statistics\\\\[2pt]",
            "    \\begin{tabular}{@{}lccrrr@{}}",
            "        \\toprule",
            "        & & & \\multicolumn{2}{c}{\\textbf{Welch ANOVA}} & \\\\",
            "        \\cmidrule(lr){4-5}",
            "        \\textbf{Model} & \\textbf{$\\eta^2$ [95\\% CI]} & \\textbf{$\\omega^2$ [95\\% CI]} & "
            "\\textbf{$F$ ($\\mathrm{df}_1$, $\\mathrm{df}_2$)} & \\textbf{$p$} & \\textbf{BF $p$} \\\\",
            "        \\midrule",
            *("        " + r for r in brows),
            "        \\bottomrule",
            "    \\end{tabular}",
            "",
            "    \\smallskip",
            "    \\begin{tabular}{@{}lrrrrrr@{}}",
            "        \\toprule",
            f"        & {corr_head} \\\\",
            "        \\cmidrule(lr){2-4}\\cmidrule(l){5-7}",
            "        \\textbf{Model} & "
            + " & ".join(
                ["\\textbf{$r$ ($\\beta_i$, imp.)}", "\\textbf{$\\rho$ ($\\beta_i$, imp.)}",
                 "\\textbf{$\\rho$ ($\\beta_i$, $\\bar{H}_i$)}"] * len(beta)
            )
            + " \\\\",
            "        \\midrule",
            *("        " + r for r in corr_rows),
            "        \\bottomrule",
            "    \\end{tabular}",
            "\\end{table}",
        ]
    )


def seed_label(sym: dict, seed: str) -> str:
    """The symmetric evaluation puts the primary checkpoint in its first seed slot, so
    that slot is labelled Primary."""
    return "Primary" if seed == str(sym["seeds"][0]) else f"Seed {seed}"


def table_s5(sym: dict) -> str:
    rows = []
    for ds_label, d in sym.items():
        per_seed = d["companion_analysis_capsule_level"]["per_seed"]
        seeds = [str(s) for s in d["seeds"]]
        for role, role_label in SYM_ROLES:
            for i, seed in enumerate(seeds):
                r = per_seed[role][seed]
                lo, hi = r["domain_gap_95ci"]
                first = f"\\multirow{{{len(seeds)}}}{{*}}{{{role_label}}}" if i == 0 else ""
                rows.append(
                    f"{ds_label if role == 'hub' and i == 0 else ''} & {first} & "
                    f"{seed_label(d, seed)} & {r['n_capsules']} & "
                    f"{100 * r['within_mean']:.1f} & {100 * r['cross_mean']:.1f} & "
                    f"{100 * r['domain_gap']:.1f} [{100 * lo:.1f}, {100 * hi:.1f}] & "
                    f"{fmt_p(r['wilcoxon_p'])} & ${r['rank_biserial_r']:.2f}$ \\\\"
                )
        rows.append("\\midrule")
    rows.pop()
    crows = []
    for ds_label, d in sym.items():
        fits = d["ceiling_robust_model"]["per_seed_fits"]
        seeds = [str(s) for s in d["seeds"]]
        for i, (key, label) in enumerate(CONTRASTS):
            cells = []
            for seed in seeds:
                c = next(
                    x
                    for x in fits[seed]["role_contrasts_model_based"]
                    if x["contrast"].startswith(key)
                )
                cells.append(f"${c['estimate_log_odds']:+.3f}$ & {fmt_p(c['p_value'])}")
            first = f"\\multirow{{{len(CONTRASTS)}}}{{*}}{{{ds_label}}}" if i == 0 else ""
            crows.append(f"{first} & {label} & " + " & ".join(cells) + " \\\\")
        crows.append("\\midrule")
    crows.pop()
    controls = {
        r["n_capsules"]
        for d in sym.values()
        for r in d["companion_analysis_capsule_level"]["per_seed"]["random"].values()
    }
    if len(controls) != 1:
        raise SystemExit("the matched controls differ in size between models")
    n_control = controls.pop()
    later_seeds = "seeds " + ", ".join(
        " and ".join(str(x) for x in d["seeds"][1:]) + f" ({ds_label})"
        for ds_label, d in sym.items()
    )
    return "\n".join(
        [
            "\\begin{table}[!htbp]",
            "    \\centering",
            "    \\caption{Routing correctness per trained dynamic-routing model; main-text "
            "Table~\\MainTabRouting{} and Section~\\MainSecCifar{} report the pooled values. "
            "\\textbf{(a)} Mean routing correctness (\\%) within and across domains over the capsules "
            "of each role; the gap is within minus cross (pp) with a capsule-level BCa 95\\% interval, "
            "and the Wilcoxon signed-rank test with its matched-pairs rank-biserial $r$ tests each "
            f"role's own gap against zero. Non-critical is the {n_control}-capsule matched control of "
            "each model. \\textbf{(b)} Difference between two roles' log-odds domain gaps, each model "
            "refitted with the pooled estimator (test-class fixed effects, two-way cluster-robust "
            "standard errors); $p$ is the two-sided Wald test, uncorrected. Second and third models: "
            f"{later_seeds}.}}",
            "    \\label{tabS:routing_per_model}",
            "    \\scriptsize",
            "    \\setlength{\\tabcolsep}{3.5pt}",
            "    \\textbf{(a)} Routing correctness and domain gap\\\\[2pt]",
            "    \\begin{tabular}{@{}lllrrrlrr@{}}",
            "        \\toprule",
            "        \\textbf{Dataset} & \\textbf{Role} & \\textbf{Model} & \\textbf{$N$} & \\textbf{Within} & "
            "\\textbf{Cross} & \\textbf{Gap [95\\% CI]} & \\textbf{Wilcoxon $p$} & \\textbf{$r$} \\\\",
            "        \\midrule",
            *("        " + r for r in rows),
            "        \\bottomrule",
            "    \\end{tabular}",
            "",
            "    \\medskip",
            "    \\textbf{(b)} Per-model log-odds contrasts\\\\[2pt]",
            "    \\begin{tabular}{@{}llrrrrrr@{}}",
            "        \\toprule",
            "        & & \\multicolumn{2}{c}{\\textbf{Primary}} & \\multicolumn{2}{c}{\\textbf{Second model}} & "
            "\\multicolumn{2}{c}{\\textbf{Third model}} \\\\",
            "        \\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
            "        \\textbf{Dataset} & \\textbf{Contrast} & \\textbf{Est.} & \\textbf{$p$} & \\textbf{Est.} & "
            "\\textbf{$p$} & \\textbf{Est.} & \\textbf{$p$} \\\\",
            "        \\midrule",
            *("        " + r for r in crows),
            "        \\bottomrule",
            "    \\end{tabular}",
            "\\end{table}",
        ]
    )


def table_s6() -> str:
    def per_dataset(family):
        """The number of models per dataset in a taxonomy family, as a word."""
        counts = {sum(m[0] == ds for m in TAXONOMIES[family]) for ds, _, _ in TAXONOMIES[family]}
        if len(counts) != 1:
            raise SystemExit(f"{family}: datasets differ in their number of models")
        return NUMBER_WORDS[counts.pop()]

    rows = []
    for family, models in TAXONOMIES.items():
        rows.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\textit{{{family}}}}} \\\\")
        for ds_label, directory, label in models:
            t = json.loads(
                (
                    PROJECT_ROOT / "results/data" / directory / "hub_taxonomy.json"
                ).read_text()
            )
            n = [len(t[k]) for k in tax.ROLE_ORDER]
            rows.append(
                f"{ds_label} & {label} & "
                + " & ".join(str(x) for x in n)
                + f" & {n[0] + n[1]} & {sum(n)} \\\\"
            )
        rows.append("\\midrule")
    rows.pop()
    n_dynamic, n_veg = (per_dataset(family) for family in TAXONOMIES)
    return "\n".join(
        [
            "\\begin{table}[!htbp]",
            "    \\centering",
            "    \\caption{Role counts of every taxonomy used in the main text, each derived natively on its "
            "own model by exhaustive single-capsule ablation on the taxonomy-calibration slice. Hubs are "
            "Primary plus Secondary Hubs, the grouping used in main-text Tables~\\MainTabRouting{} "
            f"and~\\MainTabIntervention. Dynamic routing: the {n_dynamic} pooled "
            "models per dataset of Table~\\MainTabRouting{} and Section~\\MainSecCifar. VEG: the "
            f"{n_veg} per-capsule-$\\beta$ models per dataset of "
            "Section~\\MainSecVegMechanism.}",
            "    \\label{tabS:role_counts}",
            "    \\footnotesize",
            "    \\begin{tabular}{@{}llrrrrrrr@{}}",
            "        \\toprule",
            "        \\textbf{Dataset} & \\textbf{Model} & \\textbf{PH} & \\textbf{SH} & \\textbf{Spec.} & "
            "\\textbf{Gen.} & \\textbf{NC} & \\textbf{Hubs} & \\textbf{Total} \\\\",
            "        \\midrule",
            *("        " + r for r in rows),
            "        \\bottomrule",
            "    \\end{tabular}",
            "",
            "    \\smallskip",
            "    {\\scriptsize PH, Primary Hub; SH, Secondary Hub; Spec., Specialist; Gen., Generalist; "
            "NC, Non-critical.}",
            "\\end{table}",
        ]
    )


def render() -> str:
    acc = json.loads(ACC.read_text())
    ablation = json.loads(ABLATION.read_text())
    uniform = json.loads(UNIFORM_BETA.read_text())
    beta = {k: json.loads(p.read_text()) for k, p in BETA.items()}
    sym = {k: json.loads(p.read_text()) for k, p in SYM.items()}
    header = (
        "%% Generated by paper/supplementary_tables.py from\n"
        "%%   results/veg_ablation/{accuracy_significance,component_ablation,uniform_beta_checkpoints}.json,\n"
        "%%   results/beta_values/veg_beta_analysis_{mnist,cifar}_native.json,\n"
        "%%   results/beta_values/{native,native_cifar10}/*.json,\n"
        "%%   results/data/{mnist/pooled_42_123_789,cifar10}/symmetric_evaluation_glmm.json and\n"
        "%%   results/data/{mnist,cifar10}/**/hub_taxonomy.json.\n"
        "%% Values are read and formatted only; regenerate rather than editing by hand.\n"
        "%% The captions cite the main paper's numbering through \\Main... macros (for\n"
        "%% example \\MainTabAccuracy), which the including document must define.\n"
    )
    return (
        header
        + "\n"
        + table_s3(acc, ablation, uniform)
        + "\n\n"
        + table_s4(beta)
        + "\n\n"
        + table_s5(sym)
        + "\n\n"
        + table_s6()
        + "\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--check", action="store_true", help="exit 1 if the committed fragment is stale"
    )
    args = ap.parse_args()
    text = render()
    if args.check:
        current = OUT.read_text() if OUT.exists() else ""
        if current != text:
            print(
                f"{OUT.relative_to(PROJECT_ROOT)} is out of date; run paper/supplementary_tables.py"
            )
            sys.exit(1)
        print(f"{OUT.relative_to(PROJECT_ROOT)} is up to date")
        return
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
