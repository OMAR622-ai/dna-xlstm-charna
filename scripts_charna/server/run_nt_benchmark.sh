#!/bin/bash
# Nucleotide Transformer benchmark (18 tasks) for the two final pretrained models (step 10,000):
# GBST-BERT (full_charna_mlm) and Transformer++ (full_llama_mlm). Protocol of Bio-xLSTM (App. C.5):
# PH (reverse-complement averaging at test time), mean pooling + linear head, full fine-tuning,
# 20 epochs, best validation epoch tested. Per task, the learning rate and batch size that
# Bio-xLSTM chose for DNA-xLSTM-PH (Table A5). One worker per GPU pulls jobs from a shared queue.
#   bash run_nt_benchmark.sh "1 2 3"          # seeds to run (finished runs are skipped)
source ~/envs/charna/bin/activate
ROOT=~/charna/dna-xlstm
OUT=${OUT_DIR:-${ROOT}/outputs/downstream/nt_queue}  # queue + log (OUT_DIR for a separate queue)
mkdir -p ${OUT}
CKPT_STORE=${CKPT_STORE:-/tmp/omohamed_nt_ckpts}  # fine-tuned checkpoints (local disk, not permanent)
mkdir -p ${CKPT_STORE} && chmod 700 ${CKPT_STORE}
SEEDS=${1:-"1 2 3"}
GPUS=${GPUS:-"1 2 3 4"}

# task  lr  batch   (Bio-xLSTM Table A5, DNA-xLSTM-Ph column)
TABLE_A5="H3 8e-4 128
H3K14ac 6e-4 128
H3K36me3 6e-4 64
H3K4me1 8e-4 128
H3K4me2 6e-4 64
H3K4me3 8e-4 128
H3K79me3 1e-3 128
H3K9ac 4e-4 64
H4 8e-4 64
H4ac 4e-4 64
enhancers 2e-3 512
enhancers_types 2e-3 512
promoter_all 4e-4 64
promoter_no_tata 1e-3 128
promoter_tata 3e-3 128
splice_sites_all 8e-4 64
splice_sites_acceptors 2e-3 128
splice_sites_donors 3e-3 128"
# NT_TABLE=<file> replaces these settings (same format), e.g. for a learning-rate check
[ -n "${NT_TABLE}" ] && TABLE_A5=$(cat ${NT_TABLE})

QUEUE=${OUT}/jobs.txt
: > ${QUEUE}
for seed in ${SEEDS}; do
  for model in charna llama; do
    # an optional 4th column (charna or llama) limits a line to one model
    echo "${TABLE_A5}" | while read -r task lr bs only; do
      [ -n "${only}" ] && [ "${only}" != "${model}" ] && continue
      echo "${model} ${task} ${lr} ${bs} ${seed}" >> ${QUEUE}
    done
  done
done
log() { echo "$(date "+%F %T") $*" >> ${OUT}/queue_log.txt; }
log "queue with $(wc -l < ${QUEUE}) jobs (seeds: ${SEEDS}) on GPUs ${GPUS}"

worker() {
  local gpu=$1
  while true; do
    job=$(flock ${QUEUE}.lock bash -c "head -1 ${QUEUE}; sed -i 1d ${QUEUE}")
    [ -z "${job}" ] && break
    read -r model task lr bs seed <<< "${job}"
    start=$(date +%s)
    run=${ROOT}/outputs/downstream/nt_cv10_ep20/${task}/full_${model}_mlm_ph/LR-${lr}_BATCH_SIZE-${bs}/seed-${seed}
    # run_nucleotide_charna.sh exits 0 even when training fails (e.g. a Hugging Face server error
    # while loading the data), so a job counts as done only when its metrics contain test scores
    for attempt in 1 2 3; do
      ( cd ${ROOT}/scripts_downstream && CUDA_VISIBLE_DEVICES=${gpu} MODEL=${model} \
          PRETRAIN_DIR=../outputs/pretrain/hg38/full_${model}_mlm \
          PRETRAINED_PATH=${ROOT}/outputs/pretrain/hg38/full_${model}_mlm/checkpoints/last.ckpt \
          TASKS=${task} LRS=${lr} BATCH_SIZES=${bs} SEEDS=${seed} NUM_WORKERS=4 \
          bash run_nucleotide_charna.sh > /dev/null 2>&1 )
      grep -qs "test/" ${run}/metrics.csv && break
      [ ${attempt} -lt 3 ] && sleep 600
    done
    if grep -qs "test/" ${run}/metrics.csv; then status=ok; else status=FAILED; fi
    # the home disk is only 20 GB: once a run is tested, delete its last-epoch checkpoint and move
    # the best-validation checkpoint (the one that was tested) to the server's local disk, which is
    # large but not permanent; a link to it stays in the run folder
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
log NT_DONE
