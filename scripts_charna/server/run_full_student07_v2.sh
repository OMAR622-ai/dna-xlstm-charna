#!/bin/bash
# Runs on student07 (version 2, after the data-loader fix of 5 Oct 2026): continue the full
# GBST-BERT run (R16) from its step-3000 checkpoint, then the Transformer++ full run (R17) from
# scratch, then the final accuracy test (R18).
# GPUs 1-4 (GPU 0 is used by another user); 4 GPUs x 32 sequences x 8 accumulation = 1,024
# sequences per step, as on student01. Same shared home and files as student01.
source ~/envs/charna/bin/activate
ROOT=~/charna/dna-xlstm
OUT=${ROOT}/outputs/full
RESUME=${OUT}/resume_from_step3000.ckpt
export CUDA_VISIBLE_DEVICES=1,2,3,4
log() { echo "$(date "+%F %T") [student07] $*" >> ${OUT}/queue.txt; }

test -f "${RESUME}" || { log "missing ${RESUME}"; exit 1; }

cd ${ROOT}/scripts_pretrain
log "resume full pretrain charna (LR 2e-3) from step 3000, fixed data loader, GPUs ${CUDA_VISIBLE_DEVICES}"
MODEL=charna LR=2e-3 RUN_NAME=full_charna_mlm VAL_EVERY=2000 NUM_DEVICES=4 MICRO_BATCH=32 NUM_WORKERS=4 \
  bash run_pretrain_charna.sh train.ckpt=${RESUME} > ${OUT}/full_charna_mlm_student07_v2.log 2>&1
STATUS=$?; log "end full pretrain charna exit=${STATUS}"
[ ${STATUS} -eq 0 ] || { log "stopping the queue: charna run failed"; exit 1; }

log "start full pretrain llama (LR 1e-2, fixed data loader)"
MODEL=llama RUN_NAME=full_llama_mlm VAL_EVERY=2000 NUM_DEVICES=4 MICRO_BATCH=32 NUM_WORKERS=4 \
  bash run_pretrain_charna.sh > ${OUT}/full_llama_mlm.log 2>&1
STATUS=$?; log "end full pretrain llama exit=${STATUS}"
[ ${STATUS} -eq 0 ] || { log "stopping the queue: llama run failed"; exit 1; }

cd ${ROOT}
for SPLIT in valid test; do
  CUDA_VISIBLE_DEVICES=1 python scripts_charna/evaluate_mlm.py \
    outputs/pretrain/hg38/full_charna_mlm:GBST-BERT outputs/pretrain/hg38/full_llama_mlm:Transformer++ \
    --split ${SPLIT} --n_seqs 2048 --csv ${OUT}/eval_${SPLIT}.csv --markdown ${OUT}/eval_${SPLIT}.md \
    > ${OUT}/eval_${SPLIT}.log 2>&1
  STATUS=$?; log "end evaluation ${SPLIT} exit=${STATUS}"
done
log FULL_DONE
