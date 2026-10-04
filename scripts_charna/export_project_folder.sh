#!/bin/bash
# Rebuild charna_project/: copies of every file this project added or changed relative to the
# upstream DNA-xLSTM code (origin/main), in their repo sub-paths, plus a patch of the changes
# to existing upstream files. The copies are for browsing/downloading in one place; the working
# code is the original files in the repo. Run from the repo root after committing changes:
#   bash scripts_charna/export_project_folder.sh

set -euo pipefail
OUT=charna_project
BASE=$(git merge-base HEAD "${UPSTREAM:-origin/main}")

find "${OUT}" -mindepth 1 ! -name README.md -delete 2>/dev/null || true
mkdir -p "${OUT}"

git diff --name-only --diff-filter=AM "${BASE}" HEAD -- . ":(exclude)${OUT}" ":(exclude)README.md" |
  while read -r f; do
    mkdir -p "${OUT}/$(dirname "$f")"
    cp "$f" "${OUT}/$f"
  done

# Exact edits to files that already existed upstream (README.md pointer included).
git diff "${BASE}" HEAD --diff-filter=M -- . ":(exclude)${OUT}" > "${OUT}/changes_to_existing_files.patch"

echo "upstream base: ${BASE}"
find "${OUT}" -type f | sort
