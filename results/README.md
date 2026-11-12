# Results Directory

**Paper:** The Entropy Bridge: From Mechanistic Interpretability to Explainability in Capsule Networks
**Journal:** Information Sciences (under review)

## What this file is (and is not)

This is a **map from reported quantity → the artifact that holds it → the script that
produced it**. It deliberately contains **no result values**.

Every number belongs in exactly one place: the artifact its producer wrote, which the
manuscript then reports. Restating values in documentation creates a second copy that
drifts on the next re-run and gives a reader two answers with no way to tell which is
current, so this file states none.

**To quote a number:** open the artifact named below — or re-run its producer — and read
it there. Never copy a value out of prose, this file's included.

---

## Directory Structure

```
results/
├── data/
│   ├── mnist/
│   │   ├── hub_taxonomy.json           # CANONICAL: capsule role classification
│   │   ├── importance_matrix.npy       # CANONICAL: (10, 1152) per-digit importance
│   │   ├── held_out_validation.json    # CANONICAL: surrogate generalisation
│   │   ├── threshold_sensitivity.json  # CANONICAL: threshold sweep
│   │   ├── threshold_sensitivity_dynamic.json  # CANONICAL: threshold sweep, dynamic-routing model
│   │   ├── top_capsule_ablation_effects.json  # CANONICAL: top-capsule effects + non-critical baseline
│   │   ├── per_digit_sparsity_ground_truth.json  # CANONICAL: per-digit Gini and 80%-at
│   │   ├── per_digit_sparsity_surrogate.json     # surrogate per-digit sparsity
│   │   ├── pooled_42_123_789/          # CANONICAL: MNIST symmetric evaluation -- raw records,
│   │   │                               # routing correctness by role, baseline resampling --
│   │   │                               # pooled over the primary model and seeds 123, 789
│   │   ├── ground_truth_ablation.json  # CANONICAL: ground-truth Gini
│   │   ├── seed123/, seed789/          # taxonomy replicates
│   │   └── veg_seed{42,123,456}/, veg_best_entropy/  # VEG-native taxonomies
│   ├── cifar10/                        # same layout; the CIFAR-10
│   │                                   # symmetric evaluation (42, 123, 456) sits at this level
│   ├── role_importance_profile.json    # CANONICAL: role importance shares, rule counts
│   ├── index_transfer_overlap.json     # CANONICAL: index overlap of native vs baseline roles
│   ├── checkpoint_similarity.json      # CANONICAL: weight similarity between checkpoints
│   └── parameter_counts.json           # CANONICAL: parameter counts read from the checkpoints
├── beta_values/
│   ├── veg_beta_analysis_{mnist,cifar}_native.json  # CANONICAL: native-role beta_i roll-ups
│   ├── native/, native_cifar10/        # per-model stats + figures
├── veg_ablation/
│   ├── component_ablation.json         # CANONICAL: 3-variant x 3-dataset x 5-seed ablation
│   ├── component_ablation_extra_seeds.json  # the seed-789 and seed-1000 runs
│   ├── accuracy_significance.json      # CANONICAL: paired contrasts, BH/Holm, TOST
│   ├── uniform_beta_checkpoints.json   # CANONICAL: Uniform-beta scalar per checkpoint
│   ├── gate_variation_{mnist,cifar10}.json  # CANONICAL: realised gates and their variation
│   └── role_intervention_{mnist,cifar10}.json  # CANONICAL: causal beta_i interventions
├── explanation_cost_benchmark.json     # CANONICAL: per-stage explanation latency
└── figures/
    └── explanations/                   # the seven figures the paper and supplement cite
```

Figures are written by their generators under `results/` or straight into `docs/figures/`.
`paper/sync_figures.py` copies the former into `docs/figures/`, where the paper
includes them from, and its `--check` confirms every copy matches its original.

---

## Canonical Sources

| Reported quantity | Artifact | Key path inside it | Producer |
|---|---|---|---|
| Hub/Specialist role counts | `data/{mnist,cifar10}/hub_taxonomy.json` | `primary_hubs`, `secondary_hubs`, `generalists`, `specialists`, `non_critical` (lists; take `len`) | `experiments/hub_taxonomy.py --dataset {mnist,cifar10}` |
| Per-digit capsule importance | `data/*/importance_matrix.npy` | `(10, n_caps)` float64 array | same |
| Routing correctness by role | `data/mnist/pooled_42_123_789/symmetric_evaluation_glmm.json`, `data/cifar10/symmetric_evaluation_glmm.json` | `companion_analysis_capsule_level.pooled.{hub,specialist,random}.{within_mean,cross_mean,domain_gap,domain_gap_95ci,wilcoxon_p,rank_biserial_r,n_capsules_pooled_across_seeds}` | `experiments/symmetric_raw_records.py` → `experiments/symmetric_analysis.py` |
| Cluster-robust model coefficients | same | `primary_model.{coefficients,predicted_cells,n_obs,r_squared}` | `experiments/symmetric_analysis.py` |
| Digit/class-composition confound | same | `sensitivity_digit_composition_confound.{with,without}_test_digit_control.joint_wald_p` | `experiments/symmetric_digit_confound.py` |
| Random-baseline gap resampling | `data/mnist/pooled_42_123_789/random_baseline_resampling.json` | `random_baseline_gap_resampled`, `hub`, `specialist`, `one_sided_resampling_p` | `experiments/random_baseline_resampling.py` |
| Random-capsule (role-label permutation) baseline | `data/mnist/pooled_42_123_789/random_capsule_baseline.json`, `data/cifar10/random_capsule_baseline.json` | `observed.{hubs,specialists}.pooled.*`, `random_capsule_null.{hubs,specialists}.<variant>.pooled.{gap_pp,logit_gap,within_pct,cross_pct}.{null_mean,null_p2_5,null_p97_5,p_two_sided}` (primary variant `outside_role`), `companion.<role>.pooled.*` | `experiments/random_capsule_baseline.py` |
| Held-out surrogate validation | `data/mnist/held_out_validation.json` | `overall.{pearson_r,spearman_r,precision_at_50}`, `per_digit.<d>` | `experiments/held_out_validation.py` |
| Threshold sensitivity | `data/mnist/threshold_sensitivity.json` | `summary.*`, `results[]`, `beta_values_from_model` | `experiments/threshold_sensitivity.py` |
| VEG component ablation | `veg_ablation/component_ablation.json` | `results.<dataset>.summary.<strategy>.{mean,std,all}` (percent, ddof=1) | `train/component_ablation.py` |
| Top-capsule conf. drop + non-critical baseline | `data/mnist/top_capsule_ablation_effects.json` | `per_digit.<d>.{top_capsule,top_conf_drop_pct_mean,top_conf_drop_pct_sd,random_baseline_conf_drop_pct_mean,excess_conf_drop_pct_mean,excess_conf_drop_pct_ci95,rank_p_one_sided,effect_size_z}` | `experiments/top_capsule_ablation_effects.py` |
| Ground-truth sparsity (Gini) | `data/*/ground_truth_ablation.json` | `results.<model>.{gini,percent_50,percent_80,k_50,k_80,total_samples}` | `experiments/ground_truth_sparsity.py` |
| Role shares of importance, taxonomy-rule counts, intervention β displacement, VEG native agreement | `data/role_importance_profile.json` | `models.<model>.{share_of_positive_importance,taxonomy_rule}`, `intervention_displacement.<ds>`, `veg_native_agreement_with_primary.<ds>.<seed>` | `experiments/role_importance_profile.py` |
| Index-transfer overlap | `data/index_transfer_overlap.json` | `<dataset>_<tag>.{primary_hub_jaccard,primary_hub_intersection,all_hub_jaccard,all_hub_intersection,n_universe}` | `experiments/index_transfer_overlap.py` |
| Accuracy contrasts (BH/Holm) | `veg_ablation/accuracy_significance.json` | per-contrast paired tests over the 5-seed ablation | `experiments/accuracy_significance.py` |
| Uniform-beta shared scalar | `veg_ablation/uniform_beta_checkpoints.json` | mean and spread of the single shared beta over the Uniform-β checkpoints | `experiments/uniform_beta_checkpoints.py` |
| Surrogate feature attribution (SHAP) | `docs/figures/interpretation_output/interpretation_results.json` | `target_definition`, `sampled_states`, `shap.{num_background,num_test,relative_importance}` | `experiments/surrogate_attribution.py` |
| VEG beta_i by role | `beta_values/veg_beta_analysis_{mnist,cifar}_native.json` | `per_model.<model>.{role_counts,role_means,anova,primary_hub_vs_specialist,beta_vs_importance_pearson}` | `experiments/veg_beta_analysis.py` → `experiments/veg_beta_aggregate.py` |
| Realised gate G_i by role | `veg_ablation/role_intervention_{mnist,cifar10}.json` | `gate_distribution_by_role.<role>` | `experiments/role_intervention.py` |
| Causal intervention accuracies | same | `intervention_accuracy.<condition>.accuracy` (fraction) | same |
| Within- vs between-capsule gate variation | `veg_ablation/gate_variation_{mnist,cifar10}.json` | `models.<tag>.subset.iterations.<it>.all_capsules.{logG,G,H}.within_share`, `...by_role.<role>.share_of_summed_within_capsule_var_logG`, `models.best_entropy.full_test`, `models.best_entropy.reproduction`, `across_models.<it>` | `experiments/gate_variation.py` |
| Explanation latency | `explanation_cost_benchmark.json` | `<dataset>.{forward_pass_only_ms,gradient_features_only_ms,mlp_surrogate_only_ms,full_pipeline_ms}` | `experiments/explanation_cost_benchmark.py` |

---

## Paper element → artifact

Numbering is the rendered manuscript's: appendix tables and figures carry their appendix
letter (Table C.8, Figure E.6). `\label` is the manuscript's stable identifier for each item.

| # | `\label` | What it reports | Artifact it is read from |
|---|---|---|---|
| Table 1 | `tab:training_config` | training configuration per dataset | the training recipe, except the component-ablation epoch budget and seeds, read from `veg_ablation/component_ablation.json` (`epochs`, `seeds`) — generated, `\ResComponentAblationEpochs`, `\ResComponentAblationSeeds` |
| Table 2 | `tab:sparsity_analysis` | per-digit sparsity, MNIST | Gini and 80%-at columns from `data/mnist/per_digit_sparsity_ground_truth.json` (`per_digit[].{digit,gini,percent_80}`, `mean_over_digits`), produced by `experiments/per_digit_sparsity_ground_truth.py` from the ground-truth `data/mnist/importance_matrix.npy`; Top Cap, Conf. Drop and excess columns from `data/mnist/top_capsule_ablation_effects.json`; P@50 column from `data/mnist/held_out_validation.json` (`per_digit.<d>.precision_at_50`) — body generated, `\ResPerDigitSparsitySurrogateBody` |
| Table 3 | `tab:voting_statistics` | symmetric routing correctness by role | `data/{mnist/pooled_42_123_789,cifar10}/symmetric_evaluation_glmm.json`, `data/{mnist/pooled_42_123_789,cifar10}/random_capsule_baseline.json`, and for the MNIST full-pool note `data/mnist/pooled_42_123_789/random_baseline_resampling.json` — bodies generated, `\ResRoutingCorrectnessBody`, `\ResRoutingContrastsBody` |
| Table 4 | `tab:veg_results` | VEG component ablation | upper block from `veg_ablation/component_ablation.json` (`results.<dataset>.summary.<strategy>.{mean,std}`, `seeds`) — body generated, `\ResComponentAblationBody`, `\ResComponentAblationSeeds`; its paired contrasts are in Table S3; the lower block quotes published values |
| Table 5 | `tab:cifar_sparsity_comparison` | CIFAR-10 sparsity by routing strategy | `data/cifar10/ground_truth_ablation.json` (`results.<model>.{gini,percent_80}`), `data/cifar10/hub_taxonomy.json` and `data/cifar10/veg_best_entropy/hub_taxonomy.json` (capsule count), `veg_ablation/component_ablation.json` (accuracy) — body generated, `\ResCifarSparsityBody` |
| Table 6 | `tab:gate_distribution` | learned β and realised gate by role, with replication | `veg_ablation/role_intervention_{mnist,cifar10}.json`, `veg_ablation/gate_variation_{mnist,cifar10}.json`, `beta_values/veg_beta_analysis_{mnist,cifar}_native.json` and the per-model files under `beta_values/native{,_cifar10}/` — bodies generated, `\ResGateByRoleBetaBody`, `\ResGateReplicationBody` |
| Table 7 | `tab:intervention` | accuracy under inference-time interventions, and the groups targeted | `veg_ablation/role_intervention_{mnist,cifar10}.json`, `data/role_importance_profile.json` (`intervention_displacement.<ds>`, `models.<ds>_veg_primary.share_of_positive_importance`) — bodies generated, `\ResInterventionBody`, `\ResInterventionGroupsBody` |
| Table C.8 | `tab:threshold_sensitivity` | threshold sweep summary, VEG and dynamic-routing models | `data/mnist/threshold_sensitivity.json`, `data/mnist/threshold_sensitivity_dynamic.json`, `beta_values/veg_beta_analysis_mnist_native.json`, `data/mnist/hub_taxonomy.json` — body generated, `\ResThresholdSweepBothBody` |
| Tables S3–S6 | `tabS:accuracy_contrasts`, `tabS:beta_by_role`, `tabS:routing_per_model`, `tabS:role_counts` | accuracy contrasts with TOST; β by role; per-model routing correctness and log-odds contrasts; role counts | `veg_ablation/accuracy_significance.json`; `beta_values/veg_beta_analysis_{mnist,cifar}_native.json`; `data/{mnist/pooled_42_123_789,cifar10}/symmetric_evaluation_glmm.json` (`companion_analysis_capsule_level.per_seed`, `ceiling_robust_model.per_seed_fits`); `data/{mnist,cifar10}/**/hub_taxonomy.json` — generated by `paper/supplementary_tables.py` |

| # | `\label` | Image file | Written by |
|---|---|---|---|
| Figure 1 | `fig:entropy_scheme` | `docs/figures/figure_truncated.pdf` | hand-drawn schematic, no producer script |
| Figure 2 | `fig:importance_dist` | `docs/figures/importance_distribution.png` | `figures/importance_distribution.py` |
| Figure 3 | `fig:hub_taxonomy` | `docs/figures/hub_specialist_taxonomy.png` | `figures/hub_specialist_taxonomy.py` |
| Figure 4 | `fig:voting_analysis` | `docs/figures/voting_analysis_comprehensive.png` | `figures/voting_analysis.py` |
| Figure 5 | `fig:veg_beta_distributions` (panels `fig:veg_beta_mnist`, `fig:veg_beta_cifar10`) | `beta_values/native{,_cifar10}/veg_beta_by_role_{mnist,cifar10}.png` | `figures/veg_beta_by_role.py` |
| Figure E.6 | `fig:ten_figures` | `figures/explanations/explanation_d{3,5}_s4_pred{3,5}.png` | `figures/sample_explanation.py` |

Figure 2 is written by `figures/importance_distribution.py` to
`docs/figures/importance_distribution.png` from `data/mnist/hub_taxonomy.json` and
`data/mnist/importance_matrix.npy`. Every figure in the table above except the Figure 1
schematic has a producer script.

---

## Methodological notes

These explain how the committed artifacts were produced; the values themselves are in the
artifacts.

**Importance comes from ground-truth ablation.** The taxonomy and the sparsity figures, apart
from the surrogate's own concentration that the paper compares, are built by exhaustive
single-capsule ablation. The `AblationPredictor` surrogate is not used to assign roles or to measure
sparsity; it provides per-image explanation, and the paper compares its
per-digit concentration with ground truth.

The surrogate's training data, `models/ablation_predictor_training_data.pt`, holds 100 rows
but only 10 distinct target vectors: the ablation targets are averaged per digit and then
replicated across that digit's ten images. `grad_a`, `grad_c` and `grad_H` are signed-log
transformed per image and then averaged per digit, so they are digit-constant too; only `activations`, `couplings` and `entropies` vary per
image. Its in-sample metrics (stored in `models/ablation_predictor.pt` as `final_metrics.*`)
are therefore not evidence of generalisation; `data/mnist/held_out_validation.json`, measured
against per-image ablation on held-out images, is. The share of negative (antagonistic)
ablation targets that the paper reports is read from the `ablation_effects` tensor of
`models/ablation_predictor{,_cifar10}_training_data.pt`; no file under `results/` holds it.

`./reproduce.sh --artifacts` also rewrites `data/checkpoint_similarity.json` and
`data/parameter_counts.json` from the checkpoints. No tier trains the CIFAR-10
surrogate `models/ablation_predictor_cifar10.pt`; `--train` retrains only the MNIST one.

**Disjoint image slices.** Role discovery, surrogate training and validation, and the
evaluations draw from fixed, deterministic per-class slices of the test set defined in
`capsnet/data.py` and applied identically across trained models.

**Native taxonomies.** Capsule indices carry no guaranteed correspondence across
separately trained models, so each VEG model's taxonomy is derived on that model
(`--routing_strategy entropy`) rather than transferred by index from the dynamic-routing
model. `experiments/index_transfer_overlap.py` writes the index overlap between the two
(`data/index_transfer_overlap.json`) and runs in `./reproduce.sh --recompute`.

**Standard deviations** across seeds are sample standard deviations (n − 1 denominator),
over five seeds per arm. The per-role `role_statistics.<role>.std` in the per-model files
under `beta_values/native{,_cifar10}/` is a population standard deviation (n denominator).
The per-capsule β values are not under `results/`: they are the `digits.routing_module.beta`
parameter of each VEG checkpoint; with the `capsule_roles` of the same per-model file they
give β by role.

**Uniform-β.** The shared scalar of the Uniform-β arm is read from each best-accuracy
checkpoint by `experiments/uniform_beta_checkpoints.py` (`veg_ablation/uniform_beta_checkpoints.json`).
The per-run `beta_stats` in `veg_ablation/component_ablation.json` sample β after the final
epoch and are not interchangeable with it.

**Top-capsule effect sizes.** Each digit's top capsule is compared with 30 non-critical capsules
ablated on the same images (the population of the routing-correctness evaluation's non-critical
control). `data/mnist/top_capsule_ablation_effects.json` records the excess drop over that
baseline with a bootstrap interval and an exact rank test.

**β–importance correlation on CIFAR-10.** Both Pearson and Spearman correlations are
recorded (`per_model.<model>.beta_vs_importance_pearson` and the per-model files'
`correlations` block).

**β_i is unconstrained** (β_i ∈ ℝ): negative values amplify a capsule's contribution rather
than suppress it.

**CIFAR-10 input normalisation.** CIFAR-10 artifacts are measured through
`capsnet.data.test_transform('cifar10')`, which holds the channel statistics the
checkpoints were trained under; each artifact records the transform it used in its
`input_transform` field.

---

## Is this artifact current? Most files answer for themselves

The statistical artifacts carry a `_provenance` block, written by `capsnet/provenance.py`, so
you do not have to guess whether the copy in front of you is the one the paper used, or what
produced it:

- `data/mnist/pooled_42_123_789/{symmetric_evaluation_glmm,random_baseline_resampling}.json`,
  `data/cifar10/symmetric_evaluation_glmm.json`
- `data/mnist/{top_capsule_ablation_effects,threshold_sensitivity,threshold_sensitivity_dynamic,per_digit_sparsity_ground_truth}.json`
- `data/{index_transfer_overlap,checkpoint_similarity,role_importance_profile,parameter_counts}.json`
- `beta_values/veg_beta_analysis_{mnist,cifar}_native.json`
- `data/{mnist/pooled_42_123_789,cifar10}/random_capsule_baseline.json`
- `veg_ablation/{accuracy_significance,role_intervention_mnist,role_intervention_cifar10,uniform_beta_checkpoints,gate_variation_mnist,gate_variation_cifar10}.json`
- `docs/figures/interpretation_output/interpretation_results.json`

Three files that cannot hold a key at their top level (a parquet table and a bare JSON list)
carry a sibling `.provenance.json` instead: `data/{mnist/pooled_42_123_789,cifar10}/symmetric_raw_records.provenance.json`
and `data/mnist/per_digit_sparsity_surrogate.provenance.json`.

The remaining artifacts carry **no** provenance block: the taxonomies and importance
matrices (`**/hub_taxonomy.json`, `**/importance_matrix.npy`, `**/per_digit_sparsity_wrapper.json`),
the per-model VEG beta analyses under `beta_values/native{,_cifar10}/`,
`data/{mnist,cifar10}/ground_truth_ablation.json`, `data/mnist/held_out_validation.json`,
`veg_ablation/component_ablation{,_extra_seeds}.json` and `explanation_cost_benchmark.json`.
For these, the producer and its invocation are given by the tables above and by the stage
that writes them in `reproduce.sh`. `veg_ablation/component_ablation.json` was merged by hand
from two runs of `train/component_ablation.py` at 300 epochs, one with seeds 42,
123 and 456 and one with seeds 789 and 1000 (`veg_ablation/component_ablation_extra_seeds.json`);
no script performs the merge, and `--train` regenerates the file in a single five-seed run.

Read a provenance block with:

```bash
python -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1]))['_provenance'], indent=2))" \
  results/data/mnist/pooled_42_123_789/symmetric_evaluation_glmm.json
```

| Field | What it tells you |
|---|---|
| `script`, `argv` | the producer and the exact invocation — copy `argv` to re-run it |
| `inputs[]` | every input file with its SHA-256 and size at the time of the run. **This is the staleness check**: if an input's hash no longer matches the file on disk, this artifact predates a change to its input |
| `sample_counts` | the n the statistics were computed on, recorded by the producer rather than inferred later |
| `random_seed` | every seed consumed, and whether it actually affects the output |
| `statistical_method` | one line stating what was computed |
| `git`, `timestamp_utc`, `software` | the code state (not in the public repository; see below), the clock and the package/GPU versions of that run |
| `schema` | version of this field set |

The public repository omits the `git` field: it names commits of the authors' development
history, which the public repository's single commit does not contain. Identify a run by
its `argv` and `inputs[]` hashes instead: `./reproduce.sh --recompute` re-derives the Tier 1
artifacts from this tree and compares them with the committed ones.

A producer you run yourself in a git checkout writes the field: `commit` is the checked-out
commit, and `dirty` says whether the code differed from it. `dirty` is decided over the code
alone, the paths listed in `code_paths` (`CODE_PATHS` in `capsnet/provenance.py`): it is true
when a file under them is modified, staged, deleted or untracked, and `dirty_files` then names
those files. Result files, which a regeneration rewrites as it runs, and anything else outside
those paths do not set it; the inputs of a run are identified by their `inputs[]` hashes, not
by the commit. A block with `dirty: false` therefore means: the code at `commit`, run on the
inputs with the recorded hashes.

An artifact augmented in place by a second script — the symmetric-evaluation JSONs, which
the digit-confound stage adds a block to — carries a `_provenance_augmentations` list with
one further block per augmenting run.

A provenance block records how a run happened -- clock, software, inputs -- rather
than what it computed, so `./reproduce.sh --recompute` excludes the provenance blocks and
compares every result field exactly.

---

## Reproducibility

- **Dynamic-routing models:** `models/capsule_{mnist,fashionmnist,cifar10}_best_dynamic.pt`
  plus `*_dynamic_seed{123,789}.pt` (MNIST) and `*_dynamic_seed{123,456}.pt` (CIFAR-10) for the
  3-model symmetric evaluation. The primary MNIST checkpoint shares its initialisation with
  `capsule_mnist_dynamic_seed456.pt` (`results/data/checkpoint_similarity.json`), so seed 456
  is not used as an independent MNIST replicate.
- **VEG models:** `models/capsule_*_best_entropy.pt` and `*_entropy_seed{42,123,456}.pt`.
- **Ablation seeds:** 42, 123, 456, 789, 1000 (component ablation, 300 epochs each).
- **Analysis seeds:** 42 for held-out validation and threshold sensitivity; the symmetric
  routing evaluation pools slots 42, 123, 789 on MNIST and 42, 123, 456 on CIFAR-10.
- **Per-model VEG beta analyses** (`beta_values/native{,_cifar10}/`) compute their entropy
  statistics on 1,000 test images drawn at random; the artifacts record neither the seed nor
  the sample size (the producer's defaults, `--seed 42 --num_samples 1000`, apply).

The role counts of every trained model are in its `hub_taxonomy.json`; specific capsule
indices vary with initialisation. The taxonomy, importance matrices and raw symmetric
records reproduce bit-identically on re-run — so any diff against a committed artifact is a
real change, not noise. Use that: probe a producer into a scratch directory and diff before
overwriting anything canonical.
