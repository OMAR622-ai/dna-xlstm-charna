#!/bin/bash
# Learning-rate check for GBST-BERT: same 500-step setup as the small experiment
# (0.5B tokens, validation every 100 steps on 2% of the validation set), LR in {1e-3, 2e-3, 5e-3}.
# Compared with small_charna_500 (LR 5e-4) by validation loss only. NOT STARTED automatically.
source ~/envs/charna/bin/activate
OUT=~/charna/dna-xlstm/outputs/lr_check
mkdir -p ${OUT}
cd ~/charna/dna-xlstm/scripts_pretrain
for LR in 1e-3 2e-3 5e-3; do
  echo "$(date "+%F %T") start LR ${LR}" >> ${OUT}/queue.txt
  MODEL=charna LR=${LR} RUN_NAME=small_charna_500_lr${LR} MAX_STEPS=500 VAL_EVERY=100 LIMIT_VAL=0.02 CKPT_EVERY=100 \
    bash run_pretrain_charna.sh > ${OUT}/charna_lr${LR}.log 2>&1
  STATUS=$?
  echo "$(date "+%F %T") end LR ${LR} exit=${STATUS}" >> ${OUT}/queue.txt
done
echo LR_CHECK_DONE >> ${OUT}/queue.txt
