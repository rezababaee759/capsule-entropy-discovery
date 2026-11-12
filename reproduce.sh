#!/usr/bin/env bash
# =============================================================================
# reproduce.sh -- the documented path from a fresh clone to the paper's numbers.
#
# There are three tiers, plus figure regeneration and a run-everything option. The
# tiers need very different things from you:
#
#   --recompute  Re-derive the paper's central statistics from the committed
#                artifacts and check that you get the committed values back. No
#                GPU and no dataset download; only the final parameter-count check
#                reads the checkpoints. Minutes. START HERE.
#
#   --artifacts  Regenerate every result artifact from the trained checkpoints.
#                Needs the checkpoints (see README: they are ~2.4 GB and are
#                distributed separately, not in git) and a GPU for the ablation
#                stages. Hours.
#
#   --train      Retrain every model from scratch. Needs a GPU and days. Only
#                necessary if you want to verify the training pipeline itself
#                rather than the analysis.
#
#   --figures    Regenerate the figures. Needs the checkpoints (several figures
#                run the model).
#
#   --all        --train, --artifacts and --figures, then --recompute against the
#                artifacts this run wrote.
#
# Stages run in dependency order: each stage runs after the stages whose outputs it
# reads. One dependency is easy to miss and is flagged inline: the digit-confound
# script AUGMENTS the symmetric-analysis JSON in place, so it must run after it.
# =============================================================================
set -uo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
PY="${PYTHON:-python}"
# MNIST pooled routing-correctness set. The primary MNIST checkpoint shares its
# initialisation with the seed-456 component-ablation run (results/data/checkpoint_similarity.json),
# so the pooled analysis uses slots 42 (primary), 123 and 789; CIFAR-10 keeps 42, 123, 456.
MNIST_POOLED_SEEDS="42 123 789"
MNIST_POOLED=results/data/mnist/pooled_42_123_789
# Manifest lines of the checkpoints that --train rewrites in models/.
RETRAINED='  (capsule_[a-z0-9]+_(dynamic|entropy|entropy_uniform)_seed[0-9]+|ablation_predictor(_training_data)?)\.pt$'

BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GRN=$'\033[32m'; RST=$'\033[0m'
FAILED=0                        # stages that exited non-zero
GATE=0                          # output checks that failed: missing, stale or drifted
STAGE_N=0
TRAINED=0                       # set by train(): the checkpoints are this run's, not the archive's
ARTIFACTS_RAN=0                 # set by artifacts(): Tier 1 then compares against this run's outputs

# Per-run, not a fixed /tmp name: a predictable path in a shared /tmp can be
# pre-created by another user as a symlink, and two concurrent runs would
# otherwise overwrite each other's stage output mid-tier.
STAGE_LOG="$(mktemp -t reproduce_stage.XXXXXXXX)"
# Holds the per-tier freshness markers (see require_fresh).
MARK_DIR="$(mktemp -d -t reproduce_mark.XXXXXXXX)"

stage() {                       # stage "<description>" <command...>
  local desc="$1"; shift
  STAGE_N=$((STAGE_N + 1))
  printf '%s[%02d]%s %s\n' "$BOLD" "$STAGE_N" "$RST" "$desc"
  printf '%s     $ %s%s\n' "$DIM" "$*" "$RST"
  if "$@" >"$STAGE_LOG" 2>&1; then
    printf '     %sok%s\n' "$GRN" "$RST"
  else
    printf '     %sFAILED%s (last 15 lines)\n' "$RED" "$RST"
    tail -15 "$STAGE_LOG" | sed 's/^/       /'
    FAILED=$((FAILED + 1))
  fi
}

# Abort a tier rather than let a later stage consume an earlier stage's stale
# output. stage() only counts failures, so without this a failed producer leaves
# the run going and its consumers roll up whatever is already on disk, producing a
# plausible-looking artifact from a pipeline that did not run.
require_clean() {               # require_clean "<what is about to consume them>"
  if [ $((FAILED + GATE)) -ne 0 ]; then
    printf '\n%sABORT%s  %d stage(s) and %d output check(s) failed; refusing to run %s on stale inputs.\n' \
           "$RED" "$RST" "$FAILED" "$GATE" "$1"
    printf '        Fix the failures above and re-run. A partially-run tier\n'
    printf '        yields a plausible artifact from a pipeline that did not run.\n\n'
    exit 1
  fi
}

# The recompute tier's one-line verdict. The green line speaks for every statistic,
# so it needs every stage and output check of the run to have passed, not only the
# files compared.
recompute_verdict() {           # recompute_verdict <files that drifted>
  if [ "$1" -ne 0 ]; then
    return
  elif [ $((FAILED + GATE)) -ne 0 ]; then
    printf '\n  %sNo drift among the files compared, but %d stage(s) and %d output check(s) failed above; nothing is re-derived until they pass.%s\n' \
           "$RED" "$FAILED" "$GATE" "$RST"
  elif [ "$ARTIFACTS_RAN" -eq 1 ]; then
    # Tier 2 rewrote these artifacts earlier in this run, so the snapshot holds its
    # outputs, not the committed files.
    printf '\n  %sEvery statistic re-derived consistently with this run'"'"'s Tier 2 outputs.%s\n' "$GRN" "$RST"
  else
    printf '\n  %sEvery statistic re-derived to the committed value.%s\n' "$GRN" "$RST"
  fi
}

# The run's summary line and exit status: 1 on any failure, since a count as the
# exit status wraps at 256 and would read as success.
finish() {
  printf '\n'
  if [ $((FAILED + GATE)) -eq 0 ]; then
    printf '%sAll %d stage(s) completed; every output check passed.%s\n' "$GRN" "$STAGE_N" "$RST"
    exit 0
  fi
  printf '%s%d of %d stage(s) failed; %d output check(s) failed.%s\n' \
         "$RED" "$FAILED" "$STAGE_N" "$GATE" "$RST"
  exit 1
}

# A producer can exit 0 without writing its output: a swallowed write error, a
# data-dependent filename, an early `return`. The committed file then stays at the
# canonical path and is compared or copied as if it were new. Every
# producer here is deterministic and several read their own previous output, so the
# guard is an mtime marker touched before the producing stages rather than a
# pre-stage delete: each listed path must exist and be newer than the marker.
stale_reason() {                # stale_reason <marker> <path>; prints and returns 1 if stale
  if [ ! -e "$2" ]; then
    printf '  %sMISSING%s         %s\n' "$RED" "$RST" "$2"; return 1
  elif [ ! "$2" -nt "$1" ]; then
    printf '  %sNOT REGENERATED%s %s (no stage of this run wrote it)\n' "$RED" "$RST" "$2"
    return 1
  fi
}

require_fresh() {               # require_fresh <marker> <path...>
  local marker="$1" bad=0 n=0 f; shift
  for f in "$@"; do
    n=$((n + 1))
    stale_reason "$marker" "$f" || bad=$((bad + 1))
  done
  if [ "$n" -eq 0 ]; then
    printf '  %sFAILED%s freshness check was given no paths\n' "$RED" "$RST"
    GATE=$((GATE + 1)); return 1
  fi
  if [ "$bad" -ne 0 ]; then
    printf '  %d of %d output(s) were not written by this run.\n' "$bad" "$n"
    GATE=$((GATE + 1)); return 1
  fi
  printf '  %sfresh%s  all %d output(s) written by this run\n' "$GRN" "$RST" "$n"
}

# Tier 1 regenerates canonical artifacts in place and restores them afterwards.
# The restore has to survive require_clean's abort as well as the normal path,
# or a failed run leaves regenerated artifacts sitting at canonical paths.
RECOMPUTE_SNAP=""

# A single EXIT handler. `trap` keeps one command per signal, so installing a
# second one anywhere would silently discard the first -- and the one it would
# discard is the restore that keeps a failed Tier 1 from leaving regenerated
# files at canonical paths.
cleanup() {
  restore_recompute
  rm -f "$STAGE_LOG"
  rm -rf "$MARK_DIR"
}

# An artifact listed in .absent did not exist when the tier started, so whatever is
# at its path now was written by this run and is removed.
restore_recompute() {
  [ -n "$RECOMPUTE_SNAP" ] || return 0
  for f in "${RECOMPUTE_ARTIFACTS[@]}"; do
    if [ -f "$RECOMPUTE_SNAP/$f" ]; then
      cp "$RECOMPUTE_SNAP/$f" "$f"
    elif grep -qxF -- "$f" "$RECOMPUTE_SNAP/.absent" 2>/dev/null; then
      rm -f "$f"
    fi
  done
  rm -rf "$RECOMPUTE_SNAP"
  RECOMPUTE_SNAP=""
}

trap cleanup EXIT

need_checkpoints() {
  if [ ! -f models/capsule_mnist_best_dynamic.pt ]; then
    cat >&2 <<'EOF'
ERROR: models/ has no checkpoints.

  This tier needs the trained models, which are ~2.4 GB and are therefore not in
  the git repository. See the "Model checkpoints" section of README.md for the
  download link, unpack them into models/, and re-run.

  If you only want to check the paper's numbers, use:  ./reproduce.sh --recompute
EOF
    exit 2
  fi
  # Checkpoints that --train rewrote in this run cannot match the distributed hashes;
  # every other manifest line is still checked.
  local skip='^$'
  if [ "$TRAINED" -eq 1 ]; then
    printf '  (checkpoints retrained by this run are not checked against models/CHECKSUMS.sha256)\n'
    skip="$RETRAINED"
  fi
  if ! (cd models && grep -vE "$skip" CHECKSUMS.sha256 | sha256sum --quiet -c -); then
    printf 'ERROR: models/ does not match models/CHECKSUMS.sha256 (mismatches listed above).\n' >&2
    printf '  These are not the checkpoints the paper used; re-download the archive.\n' >&2
    exit 2
  fi
}

# -----------------------------------------------------------------------------
# Tier 1. Everything here reads only files that are in the repository: the raw
# per-observation records (.parquet), the taxonomies, the per-model VEG beta
# analyses, and the component-ablation results. The re-derivations load no checkpoint;
# only the final parameter-count check reads models/*.pt.
#
# The stages overwrite their artifacts in place, so we diff the regenerated files
# against a snapshot of the committed ones and then restore the snapshot. An
# artifact the stages did not rewrite, or that had no committed file to compare
# against, fails the tier. A clean run means: the statistics in the paper are
# exactly what this code produces from the data in this repository.
RECOMPUTE_ARTIFACTS=(
  $MNIST_POOLED/symmetric_evaluation_glmm.json
  results/data/cifar10/symmetric_evaluation_glmm.json
  results/veg_ablation/accuracy_significance.json
  results/data/index_transfer_overlap.json
  results/beta_values/veg_beta_analysis_mnist_native.json
  results/beta_values/veg_beta_analysis_cifar_native.json
  results/data/mnist/per_digit_sparsity_ground_truth.json
  results/data/role_importance_profile.json
)

recompute() {
  printf '\n%s== Tier 1: re-derive the statistics from the committed data ==%s\n\n' "$BOLD" "$RST"

  RECOMPUTE_SNAP="$(mktemp -d)"
  CMP_OUT="$RECOMPUTE_SNAP/.compare_output"
  : >"$RECOMPUTE_SNAP/.absent"
  for f in "${RECOMPUTE_ARTIFACTS[@]}"; do
    if [ -f "$f" ]; then
      install -D "$f" "$RECOMPUTE_SNAP/$f" || {
        printf '%sABORT%s  could not snapshot %s\n' "$RED" "$RST" "$f"; exit 2; }
    else
      printf '%s\n' "$f" >>"$RECOMPUTE_SNAP/.absent"
    fi
  done
  # Every artifact must be rewritten by the stages below; see require_fresh.
  touch "$RECOMPUTE_SNAP/.started"

  # Order matters: analysis writes the JSON, confound augments it.
  stage "MNIST symmetric analysis (LPM + logit + contrasts + BCa)" \
        $PY experiments/symmetric_analysis.py --dataset mnist \
           --records "$MNIST_POOLED/symmetric_raw_records.parquet" \
           --output "$MNIST_POOLED/symmetric_evaluation_glmm.json"
  require_clean "the MNIST digit-composition confound, which augments that JSON in place"
  stage "MNIST digit-composition confound" \
        $PY experiments/symmetric_digit_confound.py --dataset mnist \
           --records "$MNIST_POOLED/symmetric_raw_records.parquet" \
           --json "$MNIST_POOLED/symmetric_evaluation_glmm.json"
  stage "CIFAR-10 symmetric analysis" \
        $PY experiments/symmetric_analysis.py --dataset cifar10
  require_clean "the CIFAR-10 digit-composition confound, which augments that JSON in place"
  stage "CIFAR-10 digit-composition confound" \
        $PY experiments/symmetric_digit_confound.py --dataset cifar10
  require_clean "the remaining Tier 1 statistics"
  stage "Accuracy significance tests (paired, BH-corrected)" \
        $PY experiments/accuracy_significance.py
  stage "Index-transfer overlap" \
        $PY experiments/index_transfer_overlap.py
  stage "Aggregate VEG beta (MNIST)" \
        $PY experiments/veg_beta_aggregate.py --dataset mnist
  stage "Aggregate VEG beta (CIFAR-10)" \
        $PY experiments/veg_beta_aggregate.py --dataset cifar10
  stage "Per-digit ground-truth sparsity" \
        $PY experiments/per_digit_sparsity_ground_truth.py \
           --output results/data/mnist/per_digit_sparsity_ground_truth.json
  stage "Role importance profile" \
        $PY experiments/role_importance_profile.py \
           --output results/data/role_importance_profile.json

  printf '\n%s-- comparing regenerated artifacts against the committed ones --%s\n' "$BOLD" "$RST"
  local drift=0
  for f in "${RECOMPUTE_ARTIFACTS[@]}"; do
    if ! stale_reason "$RECOMPUTE_SNAP/.started" "$f"; then
      drift=$((drift + 1)); GATE=$((GATE + 1)); continue
    fi
    if [ ! -f "$RECOMPUTE_SNAP/$f" ]; then
      printf '  %sNO BASELINE%s     %s (no committed file to compare against)\n' "$RED" "$RST" "$f"
      drift=$((drift + 1)); GATE=$((GATE + 1)); continue
    fi
    if $PY - "$f" "$RECOMPUTE_SNAP/$f" >"$CMP_OUT" 2>&1 <<'PYCMP'
import json, sys, math, re
# An empty container is a leaf of its own, so adding or removing `{}` or `[]` is a
# difference. A key present on one side only is MISSING on the other, never None.
EMPTY_DICT, EMPTY_LIST, MISSING = ("empty", "dict"), ("empty", "list"), ("missing",)
# A path is a tuple of ("k", key) / ("i", index) steps, so a key that contains a dot
# or a bracket can never collide with a nested path; it is only joined for display.
def flat(o, p=()):
    if isinstance(o, dict):
        if not o: yield p, EMPTY_DICT
        for k, v in o.items(): yield from flat(v, p + (("k", k),))
    elif isinstance(o, list):
        if not o: yield p, EMPTY_LIST
        for i, v in enumerate(o): yield from flat(v, p + (("i", i),))
    else: yield p, o
def label(p):
    return "".join(f".{s}" if t == "k" else f"[{s}]" for t, s in p)
# Exact: same type (bool, int and float are distinct) and same value; NaN equals NaN.
def same(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, float) and math.isnan(a):
        return math.isnan(b)
    return a == b
def show(v):
    return "<missing>" if v is MISSING else "{}" if v is EMPTY_DICT else "[]" if v is EMPTY_LIST else repr(v)
# A provenance block records HOW a run happened -- commit and code state,
# wall clock, package versions, argv and input hashes -- not WHAT it computed, so
# both provenance containers (`_provenance`, and the `_provenance_augmentations`
# list a script appends to when it augments another script's artifact) are
# excluded from the comparison; their differing fields are counted, never hidden.
# Every result field is compared exactly.
def run_identity(p):
    return p[:1] == (("k", "_provenance"),) or (
        len(p) >= 2 and p[0] == ("k", "_provenance_augmentations") and p[1][0] == "i")
new, old = (json.load(open(a)) for a in sys.argv[1:3])
dn, do = dict(flat(new)), dict(flat(old))
bad, env = [], 0
for k in sorted(set(dn) | set(do), key=lambda p: [(t, str(s)) for t, s in p]):
    a, b = dn.get(k, MISSING), do.get(k, MISSING)
    if same(a, b):
        continue
    if run_identity(k):
        env += 1
    else:
        bad.append(f"{label(k)}: {show(b)} -> {show(a)}")
if bad:
    print("\n".join(f"      {x}" for x in bad[:8]))
    if len(bad) > 8:
        print(f"      ... and {len(bad) - 8} further field(s)")
    sys.exit(1)
print(f"  ({env} provenance field(s) record this run, not its result)" if env else "")
PYCMP
    then
      printf '  %sidentical%s  %s%s\n' "$GRN" "$RST" "$f" "$(cat "$CMP_OUT")"
    else
      printf '  %sDRIFT%s      %s\n' "$RED" "$RST" "$f"
      cat "$CMP_OUT"
      drift=$((drift + 1)); GATE=$((GATE + 1))
    fi
  done

  # Leave the tree as we found it. restore_recompute is idempotent, so the EXIT
  # handler calling it again is a no-op.
  restore_recompute

  recompute_verdict "$drift"

  # The committed tables, the two reductions above and the paper's figure copies
  # must match the artifacts they are generated from.
  printf '\n%s-- checking the generated files against the artifacts --%s\n' "$BOLD" "$RST"
  stage "Generated LaTeX tables match the artifacts" \
        $PY paper/manuscript_values.py --check
  stage "Supplementary tables match the artifacts" \
        $PY paper/supplementary_tables.py --check
  stage "Per-digit ground-truth sparsity matches its reduction" \
        $PY experiments/per_digit_sparsity_ground_truth.py --check \
           --output results/data/mnist/per_digit_sparsity_ground_truth.json
  stage "Role importance profile matches its reduction" \
        $PY experiments/role_importance_profile.py --check \
           --output results/data/role_importance_profile.json
  stage "Paper figure copies match their originals" \
        $PY paper/sync_figures.py --check
  # The one check that reads the checkpoints: parameter counts come from the saved
  # state dicts, so without models/*.pt this stage fails rather than passing.
  stage "Parameter counts match the checkpoints" \
        $PY experiments/parameter_counts.py --check
}

# -----------------------------------------------------------------------------
artifacts() {
  need_checkpoints
  require_clean "Tier 2"        # --all runs train() first; do not rebuild on a failed train
  printf '\n%s== Tier 2: regenerate artifacts from checkpoints ==%s\n\n' "$BOLD" "$RST"
  ARTIFACTS_RAN=1
  touch "$MARK_DIR/tier2"       # every path in TIER2_OUTPUTS must be newer than this

  # -- importance and taxonomy: everything downstream keys off these -----------
  stage "MNIST taxonomy (exhaustive ablation -> importance matrix + roles)" \
        $PY experiments/hub_taxonomy.py --dataset mnist
  stage "CIFAR-10 taxonomy" \
        $PY experiments/hub_taxonomy.py --dataset cifar10

  # -- replicate and VEG-native taxonomies ------------------------------------
  # The dynamic replicates (MNIST seed123, seed789; CIFAR-10 seed123, seed456) feed
  # the symmetric routing evaluation and the non-critical baseline resampling; the
  # VEG-native taxonomies feed the VEG beta analyses, the role interventions, the
  # index-transfer overlap and the VEG-native threshold sweeps. The VEG ones need
  # --routing_strategy entropy, or the roles are derived under the wrong routing
  # rule for the checkpoint being analysed.
  # Checkpoint naming: the replicate seeds are `entropy_seed42`, not `seed42`.
  veg_model_infix() {                   # role tag -> checkpoint infix
    case "$1" in
      best_entropy) printf 'best_entropy'      ;;
      *)            printf 'entropy_%s' "$1"   ;;
    esac
  }
  for seed in 123 789; do
    stage "MNIST taxonomy replicate (dynamic, seed $seed)" \
          $PY experiments/hub_taxonomy.py --dataset mnist \
             --capsule_model "models/capsule_mnist_dynamic_seed$seed.pt" \
             --output_dir "results/data/mnist/seed$seed"
  done
  for seed in 123 456; do
    stage "CIFAR-10 taxonomy replicate (dynamic, seed $seed)" \
          $PY experiments/hub_taxonomy.py --dataset cifar10 \
             --capsule_model "models/capsule_cifar10_dynamic_seed$seed.pt" \
             --output_dir "results/data/cifar10/seed$seed"
  done
  for tag in best_entropy seed42 seed123 seed456; do
    infix="$(veg_model_infix "$tag")"
    stage "MNIST VEG-native taxonomy ($tag)" \
          $PY experiments/hub_taxonomy.py --dataset mnist \
             --capsule_model "models/capsule_mnist_$infix.pt" \
             --routing_strategy entropy \
             --output_dir "results/data/mnist/veg_$tag"
    stage "CIFAR-10 VEG-native taxonomy ($tag)" \
          $PY experiments/hub_taxonomy.py --dataset cifar10 \
             --capsule_model "models/capsule_cifar10_$infix.pt" \
             --routing_strategy entropy \
             --output_dir "results/data/cifar10/veg_$tag"
  done

  # Every stage below reads a taxonomy or an importance matrix written above.
  require_clean "the analyses that read the taxonomies and importance matrices"

  # -- sparsity ---------------------------------------------------------------
  stage "MNIST ground-truth sparsity (Gini, 80%-at)" \
        $PY experiments/ground_truth_sparsity.py --dataset mnist
  stage "CIFAR-10 ground-truth sparsity" \
        $PY experiments/ground_truth_sparsity.py --dataset cifar10
  # The result is a bare JSON list, so the producer writes its provenance to a
  # <stem>.provenance.json sidecar beside it.
  stage "MNIST per-digit sparsity (surrogate route)" \
        $PY experiments/per_digit_sparsity_surrogate.py \
           --capsule_model models/capsule_mnist_best_dynamic.pt \
           --predictor models/ablation_predictor.pt --samples 50 \
           --output results/data/mnist/per_digit_sparsity_surrogate.json

  # -- top-capsule ablation effects and surrogate validation --------------------
  stage "Top-capsule confidence drop + non-critical baseline" \
        $PY experiments/top_capsule_ablation_effects.py
  stage "Surrogate held-out validation" \
        $PY experiments/held_out_validation.py

  # -- symmetric routing evaluation -------------------------------------------
  # ORDER IS LOAD-BEARING: records -> analysis -> digit confound. The confound
  # script adds a block to the analysis script's JSON; running the analysis after
  # it drops the block. The script asserts this itself, but keep the order here too.
  # shellcheck disable=SC2086  # $MNIST_POOLED_SEEDS is a list of arguments on purpose
  stage "MNIST symmetric evaluation: raw per-observation records" \
        $PY experiments/symmetric_raw_records.py --dataset mnist --seeds $MNIST_POOLED_SEEDS \
           --output_dir "$MNIST_POOLED"
  require_clean "the MNIST symmetric analysis, which reads the records above"
  stage "MNIST symmetric analysis (LPM + logit + BCa + Wilcoxon)" \
        $PY experiments/symmetric_analysis.py --dataset mnist \
           --records "$MNIST_POOLED/symmetric_raw_records.parquet" \
           --output "$MNIST_POOLED/symmetric_evaluation_glmm.json"
  require_clean "the MNIST digit-composition confound, which augments the JSON above"
  stage "MNIST digit-composition confound (augments the JSON above)" \
        $PY experiments/symmetric_digit_confound.py --dataset mnist \
           --records "$MNIST_POOLED/symmetric_raw_records.parquet" \
           --json "$MNIST_POOLED/symmetric_evaluation_glmm.json"
  # The analysis stage drops this block, so its presence shows the confound stage wrote it.
  stage "MNIST confound block present" \
        $PY -c "import json,sys; d=json.load(open('$MNIST_POOLED/symmetric_evaluation_glmm.json')); sys.exit(0 if 'sensitivity_digit_composition_confound' in d else 1)"
  stage "CIFAR-10 symmetric evaluation: raw records" \
        $PY experiments/symmetric_raw_records.py --dataset cifar10
  require_clean "the CIFAR-10 symmetric analysis, which reads the records above"
  stage "CIFAR-10 symmetric analysis" \
        $PY experiments/symmetric_analysis.py --dataset cifar10
  require_clean "the CIFAR-10 digit-composition confound, which augments the JSON above"
  stage "CIFAR-10 digit-composition confound" \
        $PY experiments/symmetric_digit_confound.py --dataset cifar10
  # The analysis stage drops this block, so its presence shows the confound stage wrote it.
  stage "CIFAR-10 confound block present" \
        $PY -c "import json,sys; d=json.load(open('results/data/cifar10/symmetric_evaluation_glmm.json')); sys.exit(0 if 'sensitivity_digit_composition_confound' in d else 1)"
  stage "Non-critical baseline resampling" \
        $PY experiments/random_baseline_resampling.py --seeds $MNIST_POOLED_SEEDS \
           --output "$MNIST_POOLED/random_baseline_resampling.json"
  stage "Random-capsule baseline (MNIST)" \
        $PY experiments/random_capsule_baseline.py --dataset mnist
  stage "Random-capsule baseline (CIFAR-10)" \
        $PY experiments/random_capsule_baseline.py --dataset cifar10

  # -- VEG beta analyses ------------------------------------------------------
  # The `--importance` adapter is rebuilt from each directory's own
  # importance_matrix.npy first, so every beta run reads the matrix belonging to
  # the model it analyses. experiments/veg_beta_analysis.py requires it.
  require_clean "the VEG importance wrappers, which read each taxonomy's own importance matrix"
  stage "VEG importance wrappers (the --importance adapter)" \
        $PY experiments/importance_wrapper.py \
           results/data/mnist/veg_best_entropy results/data/mnist/veg_seed42 \
           results/data/mnist/veg_seed123 results/data/mnist/veg_seed456 \
           results/data/cifar10/veg_best_entropy results/data/cifar10/veg_seed42 \
           results/data/cifar10/veg_seed123 results/data/cifar10/veg_seed456

  for ds in mnist cifar10; do
    for tag in best_entropy seed42 seed123 seed456; do
      stage "VEG beta analysis: $ds $tag" \
            $PY experiments/veg_beta_analysis.py --dataset "$ds" --tag "$tag"
    done
  done
  require_clean "the VEG beta aggregators"
  stage "Aggregate VEG beta (MNIST)"   $PY experiments/veg_beta_aggregate.py --dataset mnist
  stage "Aggregate VEG beta (CIFAR-10)" $PY experiments/veg_beta_aggregate.py --dataset cifar10
  stage "Summarise Uniform-beta shared scalar (15 checkpoints)" \
        $PY experiments/uniform_beta_checkpoints.py

  # -- causal interventions and the remaining analyses ------------------------
  stage "Role interventions (MNIST)"   $PY experiments/role_intervention.py --dataset mnist
  stage "Role interventions (CIFAR-10)" $PY experiments/role_intervention.py --dataset cifar10
  stage "Gate variation within and between capsules (MNIST)" \
        $PY experiments/gate_variation.py --dataset mnist
  stage "Gate variation within and between capsules (CIFAR-10)" \
        $PY experiments/gate_variation.py --dataset cifar10
  stage "Index-transfer overlap"       $PY experiments/index_transfer_overlap.py
  require_clean "the two reductions, which read the matrices, the surrogate sparsity and the interventions"
  stage "Per-digit ground-truth sparsity" \
        $PY experiments/per_digit_sparsity_ground_truth.py \
           --output results/data/mnist/per_digit_sparsity_ground_truth.json
  stage "Role importance profile" \
        $PY experiments/role_importance_profile.py \
           --output results/data/role_importance_profile.json
  stage "Threshold sensitivity sweep"  $PY experiments/threshold_sensitivity.py
  # The dynamic-routing taxonomy's own sweep.
  stage "Threshold sensitivity sweep (MNIST, dynamic routing)" \
        $PY experiments/threshold_sensitivity.py --no_beta \
           --importance results/data/mnist/importance_matrix.npy \
           --output results/data/mnist/threshold_sensitivity_dynamic.json
  stage "Accuracy significance tests (paired, BH-corrected)" \
        $PY experiments/accuracy_significance.py
  stage "Checkpoint similarity" \
        $PY experiments/checkpoint_similarity.py --output results/data/checkpoint_similarity.json
  stage "Parameter counts" \
        $PY experiments/parameter_counts.py --output results/data/parameter_counts.json
  stage "Surrogate feature attribution (SHAP)" \
        $PY experiments/surrogate_attribution.py \
           --predictor models/ablation_predictor.pt \
           --training_data models/ablation_predictor_training_data.pt \
           --num_samples 1000 --seed 42 \
           --output docs/figures/interpretation_output --force
  stage "End-to-end explanation cost benchmark" \
        $PY experiments/explanation_cost_benchmark.py --output results/explanation_cost_benchmark.json

  # Every canonical artifact this tier writes. A stage that exits 0 without
  # writing one leaves the committed file in place, and later stages would then
  # treat it as this run's output.
  local TIER2_OUTPUTS=(
    results/data/mnist/hub_taxonomy.json          results/data/mnist/importance_matrix.npy
    results/data/cifar10/hub_taxonomy.json        results/data/cifar10/importance_matrix.npy
    results/data/mnist/ground_truth_ablation.json results/data/cifar10/ground_truth_ablation.json
    results/data/mnist/per_digit_sparsity_surrogate.json
    results/data/mnist/per_digit_sparsity_surrogate.provenance.json
    results/data/mnist/top_capsule_ablation_effects.json
    results/data/mnist/held_out_validation.json
    $MNIST_POOLED/symmetric_raw_records.parquet
    $MNIST_POOLED/symmetric_raw_records.provenance.json
    $MNIST_POOLED/symmetric_evaluation_glmm.json
    results/data/cifar10/symmetric_raw_records.parquet
    results/data/cifar10/symmetric_raw_records.provenance.json
    results/data/cifar10/symmetric_evaluation_glmm.json
    $MNIST_POOLED/random_baseline_resampling.json
    $MNIST_POOLED/random_capsule_baseline.json
    results/data/cifar10/random_capsule_baseline.json
    results/beta_values/veg_beta_analysis_mnist_native.json
    results/beta_values/veg_beta_analysis_cifar_native.json
    results/veg_ablation/uniform_beta_checkpoints.json
    results/veg_ablation/role_intervention_mnist.json
    results/veg_ablation/role_intervention_cifar10.json
    results/veg_ablation/gate_variation_mnist.json
    results/veg_ablation/gate_variation_cifar10.json
    results/data/index_transfer_overlap.json
    results/data/mnist/per_digit_sparsity_ground_truth.json
    results/data/role_importance_profile.json
    results/data/mnist/threshold_sensitivity.json
    results/data/mnist/threshold_sensitivity_dynamic.json
    results/veg_ablation/accuracy_significance.json
    results/data/checkpoint_similarity.json
    results/data/parameter_counts.json
    docs/figures/interpretation_output/interpretation_results.json
    results/explanation_cost_benchmark.json
  )
  local ds seed tag
  for seed in 123 789; do
    TIER2_OUTPUTS+=("results/data/mnist/seed$seed/hub_taxonomy.json"
                    "results/data/mnist/seed$seed/importance_matrix.npy")
  done
  for seed in 123 456; do
    TIER2_OUTPUTS+=("results/data/cifar10/seed$seed/hub_taxonomy.json"
                    "results/data/cifar10/seed$seed/importance_matrix.npy")
  done
  for ds in mnist cifar10; do
    for tag in best_entropy seed42 seed123 seed456; do
      TIER2_OUTPUTS+=("results/data/$ds/veg_$tag/hub_taxonomy.json"
                      "results/data/$ds/veg_$tag/importance_matrix.npy"
                      "results/data/$ds/veg_$tag/per_digit_sparsity_wrapper.json")
    done
  done
  for tag in best_entropy seed42 seed123 seed456; do
    TIER2_OUTPUTS+=("results/beta_values/native/veg_beta_analysis_mnist_$tag.json"
                    "results/beta_values/native_cifar10/veg_beta_analysis_cifar_$tag.json")
  done
  require_clean "the Tier 2 freshness check"
  printf '\n%s-- checking that every Tier 2 output was written by this run --%s\n' "$BOLD" "$RST"
  require_fresh "$MARK_DIR/tier2" "${TIER2_OUTPUTS[@]}"

}

# -----------------------------------------------------------------------------
train() {
  printf '\n%s== Tier 3: retrain from scratch (days on one GPU) ==%s\n\n' "$BOLD" "$RST"
  # The VEG component ablation below (default --model_dir models) replaces the
  # distributed seeded checkpoints models/capsule_{ds}_{dynamic,entropy_uniform,entropy}_seed{N}.pt
  # in place, at its 300-epoch configuration, and train_surrogate.py replaces
  # models/ablation_predictor{,_training_data}.pt; Tier 2 then reads these.
  # --force: the training scripts refuse to overwrite an existing checkpoint by
  # default (guard_overwrite), since other analyses treat those paths as fixed
  # inputs. --train is an explicit, expensive action, so it passes --force rather
  # than refusing here.
  TRAINED=1
  touch "$MARK_DIR/tier3"       # every path in TIER3_OUTPUTS must be newer than this
  # Retrained primary checkpoints go to models/tier3/; the distributed
  # models/capsule_*_best_*.pt stay as they are. Names state what is saved:
  # for MNIST the epoch with the best test accuracy ("best"); for FashionMNIST and
  # CIFAR-10 the final epoch ("final"). A checkpoint is written only with --save_path.
  mkdir -p models/tier3
  for ds in mnist fashionmnist cifar10; do
    case "$ds" in mnist) sel=best ;; *) sel=final ;; esac
    for strategy in dynamic entropy; do
      stage "Train ${ds} (${strategy})" \
            $PY train/train_capsnet.py --dataset "$ds" --routing_strategy "$strategy" --seed 42 \
               --save_path "models/tier3/capsule_${ds}_${sel}_${strategy}.pt" --force
    done
  done
  # The seeds are the ones component_ablation.json records; without --seeds the
  # script runs its default subset and would overwrite that file with fewer seeds.
  stage "VEG component ablation (3 arms x 3 datasets x 5 seeds, 300 epochs)" \
        $PY train/component_ablation.py --seeds 42 123 456 789 1000 \
           --epochs 300 --force
  stage "Train the ablation surrogate (MNIST)"         $PY train/train_surrogate.py --force
  # A stage that exits 0 without writing would leave the previous checkpoints and the
  # hand-merged component_ablation.json in place for Tier 2 to consume.
  TIER3_OUTPUTS=(results/veg_ablation/component_ablation.json
                 models/ablation_predictor.pt models/ablation_predictor_training_data.pt)
  for ckpt in mnist_best fashionmnist_final cifar10_final; do
    for strategy in dynamic entropy; do
      TIER3_OUTPUTS+=("models/tier3/capsule_${ckpt}_${strategy}.pt")
    done
  done
  require_fresh "$MARK_DIR/tier3" "${TIER3_OUTPUTS[@]}"
}

# -----------------------------------------------------------------------------
figures() {
  require_clean "the figure regeneration"
  printf '\n%s== Figures ==%s\n\n' "$BOLD" "$RST"
  need_checkpoints
  touch "$MARK_DIR/figures"     # every figure checked before the sync must be newer
  stage "Hub-Specialist taxonomy figure" \
        $PY figures/hub_specialist_taxonomy.py
  stage "Voting figure" \
        $PY figures/voting_analysis.py \
           --records "$MNIST_POOLED/symmetric_raw_records.parquet" \
           --sym_json "$MNIST_POOLED/symmetric_evaluation_glmm.json"
  stage "Importance distribution"  $PY figures/importance_distribution.py
  # Drawn from the primary VEG checkpoint of each dataset.
  stage "VEG beta by role (MNIST)" \
        $PY figures/veg_beta_by_role.py --dataset mnist \
           --model models/capsule_mnist_best_entropy.pt \
           --taxonomy results/data/mnist/veg_best_entropy/hub_taxonomy.json \
           --output results/beta_values/native/veg_beta_by_role_mnist.png
  stage "VEG beta by role (CIFAR-10)" \
        $PY figures/veg_beta_by_role.py --dataset cifar10 \
           --model models/capsule_cifar10_best_entropy.pt \
           --taxonomy results/data/cifar10/veg_best_entropy/hub_taxonomy.json \
           --output results/beta_values/native_cifar10/veg_beta_by_role_cifar10.png
  # These two producers put a result in the filename (the predicted class, the
  # number of tracked capsules). The paper's names are removed first, so a run
  # that writes under another name leaves them missing instead of stale.
  rm -f results/figures/explanations/explanation_d3_s4_pred3.png \
        results/figures/explanations/explanation_d5_s4_pred5.png \
        results/figures/explanations/single_sample_d3_s4_20caps.png \
        results/figures/explanations/single_sample_d4_s4_20caps.png \
        results/figures/explanations/single_sample_d5_s4_20caps.png
  for digit in 3 5; do
    stage "Sample explanation, digit $digit" \
          $PY figures/sample_explanation.py --digit "$digit" --sample_idx 4 \
             --output_dir results/figures/explanations --force
  done
  for pair in "0 8" "2 7"; do
    # shellcheck disable=SC2086  # $pair is two arguments on purpose
    stage "Capsule tracking, digits $pair" \
          $PY figures/capsule_tracking.py --confusable_digits $pair \
             --output_dir results/figures/explanations --force
  done
  for digit in 3 4 5; do
    stage "Specialist interactions, digit $digit" \
          $PY figures/specialist_interactions.py --digit "$digit" \
             --sample_idx 4 --top_k_per_digit 2 --output_dir results/figures/explanations --force
  done
  # The sync and its --check compare a copy with its original, so neither can tell
  # an original this run did not write. Check the originals, and the three figures
  # written straight into docs/figures, against the marker first.
  printf '\n%s-- checking that every paper figure was written by this run --%s\n' "$BOLD" "$RST"
  local originals
  if ! originals="$($PY paper/sync_figures.py --originals)" || [ -z "$originals" ]; then
    printf '  %sFAILED%s could not list the figure originals\n' "$RED" "$RST"
    GATE=$((GATE + 1))
  fi
  local fig_outputs=()
  mapfile -t fig_outputs <<<"$originals"
  fig_outputs+=(docs/figures/hub_specialist_taxonomy.png
                docs/figures/voting_analysis_comprehensive.png
                docs/figures/importance_distribution.png)
  require_fresh "$MARK_DIR/figures" "${fig_outputs[@]}"
  require_clean "the figure sync"
  stage "Copy result figures into docs/figures" $PY paper/sync_figures.py
  # No other tier writes the generated LaTeX tables; regenerate them from the
  # artifacts on disk.
  stage "Generated LaTeX tables" $PY paper/manuscript_values.py
  stage "Supplementary tables" $PY paper/supplementary_tables.py
}

usage() {
  # The header comment, between its two rule lines.
  awk 'NR == 1 { next } /^# =+$/ { if (++rules == 2) exit; next } { sub(/^# ?/, ""); print }' "$0"
  exit 1
}

[ $# -eq 0 ] && usage
for arg in "$@"; do
  case "$arg" in
    --recompute) recompute ;;
    --artifacts) artifacts ;;
    --train)     train ;;
    --figures)   figures ;;
    --all)       train; artifacts; figures; recompute ;;
    -h|--help)   usage ;;
    *) echo "unknown option: $arg" >&2; usage ;;
  esac
done

finish
