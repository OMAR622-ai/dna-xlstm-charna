#!/bin/bash
# Missing step-10,000 validation point of the full GBST-BERT run (R16): load the final model and
# run the same validation as during training (validation + test sets, 12.5% of the batches, 4 GPUs,
# 32 sequences per GPU). Runs in a scratch run directory; the final model itself is not changed.
source ~/envs/charna/bin/activate
ROOT=~/charna/dna-xlstm
CKPT=${ROOT}/outputs/pretrain/hg38/full_charna_mlm/checkpoints/last.ckpt
OUT=${ROOT}/outputs/full/final_val_charna
mkdir -p ${OUT}
python -c "import torch; print('checkpoint step', torch.load('${CKPT}', map_location='cpu')['global_step'])" > ${OUT}/check.txt
export CUDA_VISIBLE_DEVICES=1,2,3,4
cd ${ROOT}/scripts_pretrain
MODEL=charna LR=2e-3 RUN_NAME=final_val_charna_scratch MAX_STEPS=1 VAL_EVERY=1000 NUM_DEVICES=4 MICRO_BATCH=32 \
  NUM_WORKERS=4 CKPT_EVERY=100000 \
  bash run_pretrain_charna.sh train.pretrained_model_path=${CKPT} train.validate_at_start=true \
  > ${OUT}/run.log 2>&1
echo "exit=$?" >> ${OUT}/check.txt
cp ${ROOT}/outputs/pretrain/hg38/final_val_charna_scratch/metrics.csv ${OUT}/metrics.csv 2>/dev/null
echo DONE >> ${OUT}/check.txt
