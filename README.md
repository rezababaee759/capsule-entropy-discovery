# The Entropy Bridge: From Mechanistic Interpretability to Explainability in Capsule Networks

Code, result artifacts and reproduction pipeline for the paper
*The Entropy Bridge: From Mechanistic Interpretability to Explainability in Capsule
Networks* (Babaei, Cheng and Zhao; under review at *Information Sciences*).

## Overview

The study analyses Capsule Networks through causal ablation of individual primary
capsules:

- **Causal ablation framework.** Each primary capsule is removed in turn and the drop in
  the predicted class's confidence is measured. A small surrogate network learns to
  predict that importance from local routing statistics, for instance-level explanations
  without exhaustive ablation at test time.
- **Hub–Specialist taxonomy.** Importance is sparse and organises into *Hubs*, critical
  for many classes, and *Specialists*, critical for one or two. Roles are derived on each
  trained model from its own ablation importance.
- **Vectorized Entropy Gating (VEG).** A routing rule that multiplies each capsule's
  contribution by a learned gate $G_i = \exp(-\beta_i H_i)$, where $H_i$ is the capsule's
  routing entropy and $\beta_i$ a per-capsule parameter. The learned $\beta_i$ stratify by
  functional role.

Datasets: MNIST, FashionMNIST and CIFAR-10. **No result values appear in this file.** Every
reported number is read from an artifact under `results/`; the table below maps each paper
table and figure to its script and artifact.

## Installation

Python 3.10 (tested with 3.10.10) and the exact pins in `requirements.txt`, taken from the
environment every reported number was produced in:

```bash
git clone https://github.com/rezababaee759/capsule-entropy-discovery.git
cd capsule-entropy-discovery
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
pip install --no-deps -e .
```

The last line installs the shared library `capsnet/` that every script imports.

The pins include the CUDA 11.8 builds of torch and torchvision, about 2.3 GB to download.
A CPU-only machine is enough for `make recompute` and `make figures`; artifact regeneration
and training need a GPU. For a CPU build, replace the two `+cu118` pins with `+cpu` and
install from `https://download.pytorch.org/whl/cpu` instead of the cu118 index. Figures regenerated on CPU can differ from the
committed ones by a few hundred pixels. `make env` prints the running environment next to
the pinned one.

## Data

MNIST, FashionMNIST and CIFAR-10 are downloaded by torchvision into `./data` the first time a
script needs them. No dataset is needed for `make recompute`.

## Model checkpoints

The trained checkpoints, with the surrogates' training-data tensors (about 2.4 GB), are
distributed separately from the code:

> **Download:** [capsule-entropy-discovery-checkpoints.tar](https://drive.google.com/file/d/1SiuQ31Z5zCEPZQGXIxpVNV5wnxEEKaex/view?usp=sharing)

Unpack the archive into `models/` and verify it against the manifest committed in this
repository:

```bash
cd models && sha256sum -c CHECKSUMS.sha256
```

A mismatch means the file is not the one the results came from. The checkpoints are needed
to regenerate artifacts or figures, and by the one `make recompute` stage that checks the
parameter counts (`results/data/parameter_counts.json`) against them.

The two primary checkpoints per dataset (`capsule_<dataset>_best_{dynamic,entropy}.pt`) were
saved as model weights only and record no training seed. The seeded checkpoints
(`capsule_<dataset>_<arm>_seed<N>.pt`) come from `train/component_ablation.py`,
which seeds the random state before building each model, so runs with the same seed share
their initialisation across the three arms. The primary MNIST checkpoints share their
initialisation with the seed-456 runs (`results/data/checkpoint_similarity.json`). The
analysis code refers to each primary checkpoint by the slot label `42` (for example
`--seeds 42 123 789`); that label is a model identifier, not a recorded seed.

## Quick start

```bash
make recompute
```

This re-derives the paper's statistics from the committed artifacts, compares every result
field with the committed value, and reports any difference. It then checks the generated
tables and the paper's figure copies against the artifacts, and the parameter counts against
the checkpoints. It needs no GPU and no dataset; the checkpoints must be in `models/` (see
above). It takes about ten minutes. The other tiers:

| Command | What it does | Needs |
|---|---|---|
| `make recompute` | re-derive the statistics and compare with the committed artifacts | checkpoints (parameter counts only); no GPU |
| `make artifacts` | regenerate every result artifact from the checkpoints | checkpoints, GPU; hours |
| `make figures` | regenerate the figures and copy them into `docs/figures` | checkpoints (GPU optional); minutes |
| `make train` | retrain every model from scratch | GPU; days |
| `make all` | train, regenerate the artifacts and figures, then `recompute` against that run | GPU; days |
| `make checksums` | rewrite `models/CHECKSUMS.sha256` from the checkpoints on disk | checkpoints |

Deterministic cuDNN/cuBLAS modes are not forced, so other GPUs or library versions can
differ in the last floating-point digits.

The generated tables are rebuilt from the artifacts with `python paper/manuscript_values.py`
(`docs/results_tables.tex`) and `python paper/supplementary_tables.py`
(`docs/supplementary_tables.tex`); `make recompute` checks that both match the artifacts.

Concentration (sparsity) of capsule importance is measured two ways: by exhaustive
single-capsule ablation (ground truth) and from the surrogate's predicted importance.
Run from the repository root:

```bash
python experiments/ground_truth_sparsity.py --dataset {mnist,cifar10} [--output PATH]
python experiments/per_digit_sparsity_surrogate.py --samples 50 --output PATH
```

The tests check the surrogate API in `capsnet/surrogate.py` against first principles on
real images:

```bash
python -m pytest tests    # or: make test
```

They need the checkpoints in `models/` and the MNIST and CIFAR-10 test sets in `./data`,
and are skipped, with the missing file named, when either is absent. `make test` also
passes `-B`, so no Python bytecode cache is written into the tree.

`reproduce.sh` holds the stages and their exact arguments; the Makefile only names the tiers.
Several producers refuse to overwrite an existing output unless `--force` is passed, and a
producer given a non-default input refuses to write its canonical output path unless the
output path is given or `--force` is passed. To compare a re-run with a committed file,
write it elsewhere with `--output` (or `--output_dir`).

## Paper tables and figures → script → artifact

| Paper item | Producing script | Artifact(s) |
|---|---|---|
| Table 1 (training configuration) | `train/train_capsnet.py`, `train/component_ablation.py` | configuration, except the component-ablation epochs and seeds, read from `results/veg_ablation/component_ablation.json` |
| §4.3.2 checkpoint provenance, Table S2 | `experiments/checkpoint_similarity.py` | `results/data/checkpoint_similarity.json`, `models/CHECKSUMS.sha256` |
| Table 2 (per-digit sparsity) | `experiments/per_digit_sparsity_ground_truth.py`, `experiments/top_capsule_ablation_effects.py`, `experiments/held_out_validation.py` | `results/data/mnist/per_digit_sparsity_ground_truth.json`, `results/data/mnist/top_capsule_ablation_effects.json`, `results/data/mnist/held_out_validation.json` |
| §5.5 held-out surrogate validation | `experiments/held_out_validation.py` | `results/data/mnist/held_out_validation.json` |
| §5.5 surrogate versus ground-truth concentration | `experiments/per_digit_sparsity_surrogate.py`, `experiments/per_digit_sparsity_ground_truth.py` | `results/data/mnist/per_digit_sparsity_surrogate.json`, `results/data/mnist/per_digit_sparsity_ground_truth.json` |
| §6.1 aggregate sparsity | `experiments/ground_truth_sparsity.py` | `results/data/{mnist,cifar10}/ground_truth_ablation.json` |
| §6.2 taxonomy, Fig. 3 | `experiments/hub_taxonomy.py --dataset mnist`, `figures/hub_specialist_taxonomy.py` | `results/data/mnist/hub_taxonomy.json`, `importance_matrix.npy` |
| §6.2 taxonomy-rule counts, §7.4 VEG model agreement, §7.5 importance shares and β displacement | `experiments/role_importance_profile.py` | `results/data/role_importance_profile.json` |
| §6.2 threshold sweep of the dynamic-routing taxonomy | `experiments/threshold_sensitivity.py --no_beta` | `results/data/mnist/threshold_sensitivity_dynamic.json` |
| Fig. 2 | `figures/importance_distribution.py` | `docs/figures/importance_distribution.png` |
| Table 3, Fig. 4, §6.3 | `experiments/hub_taxonomy.py` (seeds 123, 789), `experiments/symmetric_{raw_records,analysis,digit_confound}.py --dataset mnist`, `experiments/random_baseline_resampling.py`, `experiments/random_capsule_baseline.py`, `figures/voting_analysis.py` | `results/data/mnist/pooled_42_123_789/`, `results/data/cifar10/random_capsule_baseline.json` |
| §6.3.1 CIFAR-10 symmetric evaluation | `experiments/symmetric_{raw_records,analysis,digit_confound}.py --dataset cifar10` | `results/data/cifar10/symmetric_evaluation_glmm.json` |
| Table 4, §4.4 accuracy tests, Table S3 | `train/component_ablation.py`, `experiments/accuracy_significance.py`, `experiments/uniform_beta_checkpoints.py` | `results/veg_ablation/component_ablation*.json`, `accuracy_significance.json`, `uniform_beta_checkpoints.json` |
| Table 5 | `experiments/ground_truth_sparsity.py --dataset cifar10` | `results/data/cifar10/ground_truth_ablation.json`, with the accuracies of `results/veg_ablation/component_ablation.json` |
| §7.4, Fig. 5, Table S4 | `experiments/hub_taxonomy.py --routing_strategy entropy`, `experiments/importance_wrapper.py`, `experiments/veg_beta_analysis.py`, `experiments/veg_beta_aggregate.py`, `figures/veg_beta_by_role.py`, `experiments/index_transfer_overlap.py`, `experiments/gate_variation.py` | `results/beta_values/veg_beta_analysis_{mnist,cifar}_native.json`, `results/beta_values/native{,_cifar10}/`, `results/data/index_transfer_overlap.json`, `results/veg_ablation/gate_variation_{mnist,cifar10}.json` |
| Tables 6, 7 | `experiments/role_intervention.py`, with the §7.4 producers and `experiments/role_importance_profile.py` | `results/veg_ablation/role_intervention_{mnist,cifar10}.json`, `results/veg_ablation/gate_variation_{mnist,cifar10}.json`, `results/beta_values/veg_beta_analysis_{mnist,cifar}_native.json`, `results/data/role_importance_profile.json` |
| §8.2 explanation cost | `experiments/explanation_cost_benchmark.py` | `results/explanation_cost_benchmark.json` |
| Appendix C, Table C.8 | `experiments/threshold_sensitivity.py` | `results/data/mnist/threshold_sensitivity{,_dynamic}.json` |
| Appendix D (surrogate feature attribution) | `experiments/surrogate_attribution.py` | `docs/figures/interpretation_output/interpretation_results.json` |
| Fig. E.6, Figs. S1–S2 | `figures/sample_explanation.py`, `figures/capsule_tracking.py`, `figures/specialist_interactions.py` | `results/figures/explanations/` |
| Tables S3–S6 | `paper/supplementary_tables.py` | `docs/supplementary_tables.tex` |
| Generated table bodies (Tables 1–7, C.8) | `paper/manuscript_values.py` | `docs/results_tables.tex` |
| Parameter counts | `experiments/parameter_counts.py` | `results/data/parameter_counts.json` |
| Surrogate training (Tier 3) | `train/train_surrogate.py` | `models/ablation_predictor.pt`, `models/ablation_predictor_training_data.pt` |

`results/README.md` lists, for each reported quantity, the artifact field it is read from,
which artifacts carry a provenance block, and how to read one.

## Repository layout

```
capsnet/         shared library: model and routing rules (model), surrogate (surrogate),
                 image partition, transforms, seeding and loaders (data), checkpoint
                 loading (io), taxonomy rule (taxonomy), statistics (stats), single-capsule
                 ablation (ablation), symmetric evaluation (symmetric), VEG gates (veg),
                 per-sample attribution (explain), provenance, training loops (training),
                 figure style (paper_style)
train/           training: capsule networks, the VEG component ablation, the surrogate;
                 checksum_models.sh rewrites models/CHECKSUMS.sha256
experiments/     one script per analysis, most named after the artifact they write
figures/         figure scripts
paper/           generators: the LaTeX table bodies (docs/*_tables.tex) and the figure
                 sync into docs/figures
tests/           surrogate API tests (python -m pytest tests)
results/         committed result artifacts (see results/README.md)
docs/figures/    figures as included by the paper
docs/*_tables.tex  generated LaTeX table bodies (results_tables, supplementary_tables)
models/          checkpoints (not in git; see Model checkpoints) and CHECKSUMS.sha256
reproduce.sh     every stage of every tier, in dependency order
pyproject.toml   makes capsnet/ installable (pip install -e .) and configures pytest
Makefile         names the tiers of reproduce.sh
```

Conventions that hold across the code:

- **Capsule counts are derived from the input size.** 28×28 inputs with 32 capsule types
  give a 6×6 grid and 1,152 primary capsules (MNIST, FashionMNIST); 32×32 inputs with 64
  types give 8×8 and 4,096 (CIFAR-10).
- **Capsule index order is (row, column, type)**: on MNIST `type = i % 32`,
  `row = (i // 32) // 6`, `col = (i // 32) % 6`.
- **`CapsuleNetwork.forward(x)` returns `(v, c, b, routing_history)`**, with `v` of shape
  `[batch, classes, 16]`; class scores are `v.norm(dim=-1)`.
- **Input transforms are part of the model.** MNIST and FashionMNIST models expect plain
  `ToTensor()` inputs in [0, 1]; CIFAR-10 models expect the training-set channel
  normalisation. `capsnet/data.py` defines both.
- **Image partition.** `capsnet/data.py` defines the per-class test-set slices (surrogate
  training, held-out validation, taxonomy calibration, evaluation and top-capsule effects)
  and refuses any request that would cross into a neighbouring slice.
- **Checkpoint formats.** Capsule-network checkpoints are bare `state_dict`s; the
  surrogate checkpoints (`ablation_predictor*.pt`) are dictionaries holding the state dict
  and its configuration.

## Citation

```bibtex
@article{babaei2026entropy,
  title   = {The Entropy Bridge: From Mechanistic Interpretability to
             Explainability in Capsule Networks},
  author  = {Babaei, Reza and Cheng, Samuel and Zhao, Shangqing},
  journal = {Information Sciences},
  year    = {2026},
  note    = {Under review}
}
```

`CITATION.cff` carries the same metadata in Citation File Format.

## License

MIT; see [LICENSE](LICENSE).
