#!/bin/bash
# Nucleotide Transformer benchmark (18 tasks) for the two final pretrained models (step 10,000):
# GBST-BERT (full_charna_mlm) and Transformer++ (full_llama_mlm). Protocol of Bio-xLSTM (App. C.5):
# PH (reverse-complement averaging at test time), mean pooling + linear head, full fine-tuning,
# 20 epochs, best validation epoch tested. Per task, the learning rate and batch size that
# Bio-xLSTM chose for DNA-xLSTM-PH (Table A5). One worker per GPU pulls jobs from a shared queue.
#   bash run_nt_benchmark.sh "1 2 3"          # seeds to run (finished runs are skipped)
source ~/envs/charna/bin/activate
ROOT=~/charna/dna-xlstm
OUT=${ROOT}/outputs/downstream/nt_queue
mkdir -p ${OUT}
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

QUEUE=${OUT}/jobs.txt
: > ${QUEUE}
for seed in ${SEEDS}; do
  for model in charna llama; do
    echo "${TABLE_A5}" | while read -r task lr bs; do echo "${model} ${task} ${lr} ${bs} ${seed}" >> ${QUEUE}; done
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
    ( cd ${ROOT}/scripts_downstream && CUDA_VISIBLE_DEVICES=${gpu} MODEL=${model} \
        PRETRAIN_DIR=../outputs/pretrain/hg38/full_${model}_mlm \
        PRETRAINED_PATH=${ROOT}/outputs/pretrain/hg38/full_${model}_mlm/checkpoints/last.ckpt \
        TASKS=${task} LRS=${lr} BATCH_SIZES=${bs} SEEDS=${seed} NUM_WORKERS=4 \
        bash run_nucleotide_charna.sh > /dev/null 2>&1 )
    status=$?
    # the home disk is only 20 GB: once a run is tested, keep its best-validation checkpoint
    # (the one that was tested) and delete the last-epoch copy (about 34 MB per run)
    run=${ROOT}/outputs/downstream/nt_cv10_ep20/${task}/full_${model}_mlm_ph/LR-${lr}_BATCH_SIZE-${bs}/seed-${seed}
    grep -qs "test/" ${run}/metrics.csv && ls ${run}/checkpoints/val/*.ckpt > /dev/null 2>&1 && rm -f ${run}/checkpoints/last.ckpt
    log "gpu${gpu} ${model} ${task} lr=${lr} bs=${bs} seed=${seed} exit=${status} seconds=$(( $(date +%s) - start ))"
  done
  log "gpu${gpu} worker done"
}
for g in ${GPUS}; do worker ${g} & done
wait
log NT_DONE
