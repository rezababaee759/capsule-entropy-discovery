#!/usr/bin/env bash
# Regenerate models/CHECKSUMS.sha256 from the checkpoints currently in models/.
# Run it after adding or replacing a checkpoint.
#
# Usage: train/checksum_models.sh
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models"

shopt -s nullglob
CKPTS=(*.pt)
if [ "${#CKPTS[@]}" -eq 0 ]; then
    echo "no .pt files in models/ -- refusing to write an empty manifest" >&2
    exit 1
fi

OUT="CHECKSUMS.sha256"
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

{
  echo "# SHA-256 for every checkpoint distributed with 'The Entropy Bridge'."
  echo "#"
  echo "# The checkpoints are ~2.4 GB and are distributed as a separate archive rather"
  echo "# than in git (see the \"Model checkpoints\" section of ../README.md). This file"
  echo "# IS in git, so an unpacked archive can be checked against the repository:"
  echo "#"
  echo "#     cd models && sha256sum -c CHECKSUMS.sha256"
  echo "#"
  echo "# Regenerate with:  train/checksum_models.sh"
  echo "#"
  echo "# A mismatch means the file is not the one the paper's numbers came from. Do not"
  echo "# proceed with a reproduction run; re-download instead."
  echo "#"
} > "$TMP"

printf '%s\n' "${CKPTS[@]}" | sort | xargs sha256sum >> "$TMP"

mv "$TMP" "$OUT"
trap - EXIT
echo "wrote models/$OUT -- ${#CKPTS[@]} checkpoint(s)"
