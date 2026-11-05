#!/bin/bash
# Genomic Benchmarks (8 tasks) for the two final pretrained models (step 10,000): GBST-BERT
# (full_charna_mlm) and Transformer++ (full_llama_mlm). Protocol of Bio-xLSTM (App. C.5): PH, mean
# pooling + linear head, full fine-tuning, 10 epochs, 5 seeds, best validation epoch tested, top-1
# accuracy. The paper gives no per-task settings for this benchmark, so every task uses one setting
# from its search grid (LR 1e-3, batch 128). One worker per GPU pulls jobs from a shared queue.
#   bash run_gb_benchmark.sh "1 2 3 4 5"      # seeds to run (finished runs are skipped)
source ~/envs/charna/bin/activate
ROOT=~/charna/dna-xlstm
OUT=${OUT_DIR:-${ROOT}/outputs/downstream/gb_queue}  # queue + log
mkdir -p ${OUT}
CKPT_STORE=${CKPT_STORE:-/tmp/omohamed_gb_ckpts}  # fine-tuned checkpoints (local disk, not permanent)
mkdir -p ${CKPT_STORE} && chmod 700 ${CKPT_STORE}
export GB_DATA=${GB_DATA:-/tmp/omohamed_genomic_benchmarks}  # one text file per sequence: off the home disk
SEEDS=${1:-"1 2 3 4 5"}
GPUS=${GPUS:-"1 2 4"}
GB_OUT_ROOT=${GB_OUT_ROOT:-./outputs/downstream/gb_cv5_ep10}  # results folder (relative to the repository)
LR=${LR:-1e-3}
BS=${BS:-128}
# small tasks first, so the first results arrive early
TASKS="dummy_mouse_enhancers_ensembl human_nontata_promoters human_enhancers_cohn demo_coding_vs_intergenomic_seqs
  demo_human_or_worm human_enhancers_ensembl human_ocr_ensembl human_ensembl_regulatory"

log() { echo "$(date "+%F %T") $*" >> ${OUT}/queue_log.txt; }

# download every task once before the workers start (parallel downloads of one task would collide)
for task in ${TASKS}; do
  python -c "
from genomic_benchmarks.data_check import is_downloaded
from genomic_benchmarks.loc2seq import download_dataset
if not is_downloaded('${task}', cache_path='${GB_DATA}'):
    download_dataset('${task}', version=0, dest_path='${GB_DATA}')
" >> ${OUT}/download.log 2>&1 || { log "download of ${task} failed"; exit 1; }
done
log "data ready in ${GB_DATA} ($(du -sh ${GB_DATA} | cut -f1))"

QUEUE=${OUT}/jobs.txt
: > ${QUEUE}
for seed in ${SEEDS}; do
  for task in ${TASKS}; do
    for model in charna llama; do
      echo "${model} ${task} ${LR} ${BS} ${seed}" >> ${QUEUE}
    done
  done
done
log "queue with $(wc -l < ${QUEUE}) jobs (seeds: ${SEEDS}) on GPUs ${GPUS}"

worker() {
  local gpu=$1
  while true; do
    job=$(flock ${QUEUE}.lock bash -c "head -1 ${QUEUE}; sed -i 1d ${QUEUE}")
    [ -z "${job}" ] && break
    read -r model task lr bs seed <<< "${job}"
    start=$(date +%s)
    run=${ROOT}/${GB_OUT_ROOT#./}/${task}/full_${model}_mlm_ph/LR-${lr}_BATCH_SIZE-${bs}/seed-${seed}
    # a job counts as done only when its metrics contain test scores
    for attempt in 1 2 3; do
      ( cd ${ROOT}/scripts_downstream && CUDA_VISIBLE_DEVICES=${gpu} MODEL=${model} \
          PRETRAIN_DIR=../outputs/pretrain/hg38/full_${model}_mlm \
          PRETRAINED_PATH=${ROOT}/outputs/pretrain/hg38/full_${model}_mlm/checkpoints/last.ckpt \
          TASKS=${task} LRS=${lr} BATCH_SIZES=${bs} SEEDS=${seed} NUM_WORKERS=4 OUT_ROOT=${GB_OUT_ROOT} \
          bash run_genomics_charna.sh > /dev/null 2>&1 )
      grep -qs "test/" ${run}/metrics.csv && break
      [ ${attempt} -lt 3 ] && sleep 300
    done
    if grep -qs "test/" ${run}/metrics.csv; then status=ok; else status=FAILED; fi
    # keep the home disk free: drop the last-epoch checkpoint, move the tested one to the local disk
    if [ ${status} = ok ] && [ ! -L ${run}/checkpoints ]; then
      rm -f ${run}/checkpoints/last.ckpt
      dest=${CKPT_STORE}/${task}/full_${model}_mlm_ph/LR-${lr}_BATCH_SIZE-${bs}/seed-${seed}
      mkdir -p ${dest} && mv ${run}/checkpoints ${dest}/ && ln -s ${dest}/checkpoints ${run}/checkpoints
    fi
    log "gpu${gpu} ${model} ${task} lr=${lr} bs=${bs} seed=${seed} status=${status} attempts=${attempt} seconds=$(( $(date +%s) - start ))"
  done
  log "gpu${gpu} worker done"
}
for g in ${GPUS}; do worker ${g} & done
wait
log GB_DONE
