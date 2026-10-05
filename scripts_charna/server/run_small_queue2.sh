#!/bin/bash
# Small experiment, part 2 (replaces the rest of run_small_queue.sh; the NT benchmark is skipped
# for now): wait for the running GBST-BERT pretraining, pretrain Transformer++-2M the same way,
# then measure masked-base accuracy and loss of both final models on held-out sequences.
source ~/envs/charna/bin/activate
ROOT=~/charna/dna-xlstm
OUT=${ROOT}/outputs/small
log() { echo "$(date "+%F %T") $*" >> ${OUT}/queue.txt; }

WAIT_PID=$1
while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 60; done
log "end pretrain charna (exit status not available after queue change)"

cd ${ROOT}/scripts_pretrain
log "start pretrain llama"
MODEL=llama RUN_NAME=small_llama_500 MAX_STEPS=500 VAL_EVERY=100 LIMIT_VAL=0.02 CKPT_EVERY=100 \
  bash run_pretrain_charna.sh > ${OUT}/pretrain_llama.log 2>&1
STATUS=$?; log "end pretrain llama exit=${STATUS}"

cd ${ROOT}
for SPLIT in valid test; do
  log "start evaluation ${SPLIT}"
  CUDA_VISIBLE_DEVICES=0 python scripts_charna/evaluate_mlm.py \
    outputs/pretrain/hg38/small_charna_500:GBST-BERT outputs/pretrain/hg38/small_llama_500:Transformer++ \
    --split ${SPLIT} --n_seqs 2048 --csv ${OUT}/eval_${SPLIT}.csv --markdown ${OUT}/eval_${SPLIT}.md \
    > ${OUT}/eval_${SPLIT}.log 2>&1
  STATUS=$?; log "end evaluation ${SPLIT} exit=${STATUS}"
done
log SMALL_DONE
