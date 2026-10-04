#!/bin/bash
# Copy the small result files of all runs (metrics, configs, summaries; no checkpoints or logs)
# from an outputs/ directory into charna_project/results/, keeping the directory layout.
#   bash scripts_charna/collect_results.sh [outputs_dir] [destination]

set -euo pipefail
SRC=${1:-outputs}
DEST=${2:-charna_project/results}
mkdir -p "${DEST}"
DEST=$(realpath "${DEST}")

cd "${SRC}"
find . -type f \( -name metrics.csv -o -name model_config.json -o -name summary.txt \
    -o -name queue.txt -o -name "nt_results*.md" \) |
  while read -r f; do
    mkdir -p "${DEST}/$(dirname "$f")"
    cp "$f" "${DEST}/$f"
  done
find "${DEST}" -type f | sort
