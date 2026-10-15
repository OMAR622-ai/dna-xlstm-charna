#!/bin/bash
# Back up the fine-tuned NT-benchmark models (best-validation checkpoint of every run).
# 1. copies them from the server's local disk (/tmp, not permanent) to charna_project/nt_finetuned_models/
# 2. packs every complete set (one model, one seed, all 18 tasks) into one archive
# 3. uploads new archives to the GitHub release "nt-finetuned-models" (too large for the git history)
#   bash scripts_charna/upload_nt_models.sh
set -e
cd "$(dirname "$0")/.."
REPO=OMAR622-ai/dna-xlstm-charna
TAG=nt-finetuned-models
LOCAL=charna_project/nt_finetuned_models
PACK=${PACK_DIR:-charna_project/.nt_pack}
mkdir -p ${LOCAL} ${PACK}
rsync -a student07:/tmp/omohamed_nt_ckpts/ ${LOCAL}/

if ! gh release view ${TAG} -R ${REPO} > /dev/null 2>&1; then
  gh release create ${TAG} -R ${REPO} --target charna-gbst --title "Fine-tuned models: NT benchmark" \
    --notes "Best-validation checkpoints of the Nucleotide Transformer benchmark fine-tuning runs (scripts_charna/server/run_nt_benchmark.sh). One archive per model and seed, each with all 18 tasks, laid out as <task>/<model>_ph/LR-<lr>_BATCH_SIZE-<batch>/seed-<k>/checkpoints/val/<metric>.ckpt. full_charna_mlm = GBST-BERT, full_llama_mlm = Transformer++."
fi
uploaded=$(gh release view ${TAG} -R ${REPO} --json assets -q '.assets[].name')

for model in charna llama; do
  for seed in $(seq 1 10); do
    name=nt_finetuned_${model}_seed${seed}.tar
    n=$(ls ${LOCAL}/*/full_${model}_mlm_ph/*/seed-${seed}/checkpoints/val/*.ckpt 2>/dev/null | wc -l)
    [ "${n}" -lt 18 ] && continue                       # set not complete yet
    echo "${uploaded}" | grep -qx "${name}" && continue # already on GitHub
    (cd ${LOCAL} && tar -cf ${PACK}/${name} */full_${model}_mlm_ph/*/seed-${seed})
    gh release upload ${TAG} -R ${REPO} ${PACK}/${name}
    rm ${PACK}/${name}
    echo "uploaded ${name}"
  done
done
gh release view ${TAG} -R ${REPO} --json assets -q '.assets[] | "\(.name)  \(.size/1e6|floor) MB"'
