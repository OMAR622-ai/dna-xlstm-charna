#!/bin/bash
# Runs on student07: continue the full GBST-BERT run (R16) from its last checkpoint once student01
# has stopped it, then the Transformer++ full run (R17) and the final accuracy test (R18).
# GPUs 1-4 (GPU 0 is used by another user); 4 GPUs x 32 sequences x 8 accumulation = 1,024
# sequences per step, exactly as on student01. Same shared home and files as student01.
source ~/envs/charna/bin/activate
ROOT=~/charna/dna-xlstm
OUT=${ROOT}/outputs/full
FLAG=${OUT}/moved_to_student07.flag
CKPT=${ROOT}/outputs/pretrain/hg38/full_charna_mlm/checkpoints/last.ckpt
export CUDA_VISIBLE_DEVICES=1,2,3,4
log() { echo "$(date "+%F %T") [student07] $*" >> ${OUT}/queue.txt; }

until [ -f "${FLAG}" ]; do sleep 60; done
cp "${CKPT}" "${OUT}/resume_from_step$(cat ${FLAG}).ckpt"  # keep the exact file the resume starts from

cd ${ROOT}/scripts_pretrain
log "resume full pretrain charna (LR 2e-3) from step $(cat ${FLAG}) on GPUs ${CUDA_VISIBLE_DEVICES}"
MODEL=charna LR=2e-3 RUN_NAME=full_charna_mlm VAL_EVERY=2000 NUM_DEVICES=4 MICRO_BATCH=32 NUM_WORKERS=4 \
  bash run_pretrain_charna.sh train.ckpt=${OUT}/resume_from_step$(cat ${FLAG}).ckpt \
  > ${OUT}/full_charna_mlm_student07.log 2>&1
STATUS=$?; log "end full pretrain charna exit=${STATUS}"

log "start full pretrain llama (LR 1e-2)"
MODEL=llama RUN_NAME=full_llama_mlm VAL_EVERY=2000 NUM_DEVICES=4 MICRO_BATCH=32 NUM_WORKERS=4 \
  bash run_pretrain_charna.sh > ${OUT}/full_llama_mlm.log 2>&1
STATUS=$?; log "end full pretrain llama exit=${STATUS}"

cd ${ROOT}
for SPLIT in valid test; do
  CUDA_VISIBLE_DEVICES=1 python scripts_charna/evaluate_mlm.py \
    outputs/pretrain/hg38/full_charna_mlm:GBST-BERT outputs/pretrain/hg38/full_llama_mlm:Transformer++ \
    --split ${SPLIT} --n_seqs 2048 --csv ${OUT}/eval_${SPLIT}.csv --markdown ${OUT}/eval_${SPLIT}.md \
    > ${OUT}/eval_${SPLIT}.log 2>&1
  STATUS=$?; log "end evaluation ${SPLIT} exit=${STATUS}"
done
log FULL_DONE
