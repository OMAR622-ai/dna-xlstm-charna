#!/bin/bash

# Pretrain the ChaRNABERT-style GBST + BERT model (2M) on hg38, short context (1k), MLM.
# Protocol from Bio-xLSTM App. C.5 / Tables A3-A4 (same as the repo's Transformer++-2M):
#   2^20 tokens per step (1024 x 1024), 10,000 steps, AdamW (0.9, 0.95), wd 0.1,
#   cosine decay with 1,000 warmup steps (10% of max_steps), grad clip 1.0, 15% MLM (80/10/10).
# See all model parameters in configs/model/charna.yaml.
#
# Every setting below can be overridden from the environment, e.g. for a short smoke run:
#   MAX_STEPS=48 LIMIT_VAL=8 VAL_EVERY=12 sh run_pretrain_charna.sh

cd ../ || exit

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NUM_DEVICES=${NUM_DEVICES:-4}

# Wandb Logging
WANDB_MODE=${WANDB_MODE:-"disabled"} # online, disabled
WANDB_PROJECT=${WANDB_PROJECT:-"xlstm_dna"}

SEQLEN=${SEQLEN:-1024}
MAX_STEPS=${MAX_STEPS:-10000}
D_MODEL=${D_MODEL:-256}
N_LAYER=${N_LAYER:-4}
N_HEADS=${N_HEADS:-8}
HIDDEN_DIM=${HIDDEN_DIM:-320}      # 1.25:1 up-projection (Bio-xLSTM Table A3)
LR=${LR:-"1e-2"}                   # chosen value for Transformer++-2M (Table A3)
TOKENIZER=${TOKENIZER:-"gbst"}     # gbst | embedding
DOWNSAMPLE=${DOWNSAMPLE:-1}
MLM_STRATEGY=${MLM_STRATEGY:-"mlm"}  # mlm | ul2 (train split only; validation always uses 15% MLM)
RC_AUG=${RC_AUG:-"true"}           # reverse complement augmentation (PH setting)
PRECISION=${PRECISION:-32}         # GTX 1080 Ti has no bf16; the paper used bf16 on A100
MICRO_BATCH=${MICRO_BATCH:-32}     # per-GPU batch; gradient accumulation fills the global batch
LIMIT_VAL=${LIMIT_VAL:-0.125}      # repo default (fraction of the validation set)
VAL_EVERY=${VAL_EVERY:-$(( MAX_STEPS / 10 ))}
CKPT_EVERY=${CKPT_EVERY:-500}
NUM_WORKERS=${NUM_WORKERS:-2}
BATCH_SIZE=$(( 1048576 / SEQLEN ))  # constant number of tokens across context sizes
# Lightning 1.8 counts val_check_interval in micro-batches, not optimizer steps; the repo's own
# scripts use no gradient accumulation, so scale by it to validate every VAL_EVERY optimizer steps.
ACCUM=$(( BATCH_SIZE / (NUM_DEVICES * MICRO_BATCH) ))
VAL_CHECK_BATCHES=$(( VAL_EVERY * ACCUM ))

SEQLEN_DIS="$(echo "scale=0; ${SEQLEN} / 1000" | bc)k"
TIMESTAMP=$(date +"%y%m%d%H%M%S")
WANDB_NAME=${RUN_NAME:-"charna_${TOKENIZER}-ds${DOWNSAMPLE}_${MLM_STRATEGY}_rcaug-${RC_AUG}_seqlen-${SEQLEN_DIS}_dmodel-${D_MODEL}_nlayer-${N_LAYER}_lr-${LR}_runid-${TIMESTAMP}"}
HYDRA_RUN_DIR="./outputs/pretrain/hg38/${WANDB_NAME}"
mkdir -p "${HYDRA_RUN_DIR}"

python scripts_charna/train_charna.py \
  experiment=hg38/hg38 \
  callbacks.model_checkpoint_every_n_steps.every_n_train_steps=${CKPT_EVERY} \
  dataset.max_length=${SEQLEN} \
  dataset.batch_size=${MICRO_BATCH} \
  dataset.mlm=true \
  dataset.mlm_probability=0.15 \
  +dataset.mlm_strategy=${MLM_STRATEGY} \
  dataset.rc_aug="${RC_AUG}" \
  model=charna \
  model.config.d_model=${D_MODEL} \
  model.config.n_layer=${N_LAYER} \
  model.config.n_heads=${N_HEADS} \
  model.config.hidden_dim=${HIDDEN_DIM} \
  model.config.max_length=${SEQLEN} \
  model.config.tokenizer=${TOKENIZER} \
  model.config.downsample_rate=${DOWNSAMPLE} \
  optimizer.lr="${LR}" \
  optimizer.weight_decay=0.1 \
  train.global_batch_size=${BATCH_SIZE} \
  trainer.max_steps=${MAX_STEPS} \
  trainer.devices=${NUM_DEVICES} \
  trainer.precision=${PRECISION} \
  trainer.gradient_clip_val=1.0 \
  trainer.limit_val_batches=${LIMIT_VAL} \
  +trainer.val_check_interval=${VAL_CHECK_BATCHES} \
  wandb.project="${WANDB_PROJECT}" \
  wandb.mode="${WANDB_MODE}" \
  wandb.group=pretrain_hg38 \
  wandb.name="${WANDB_NAME}" \
  hydra.run.dir="${HYDRA_RUN_DIR}" \
  loader.num_workers=${NUM_WORKERS} \
  loader.pin_memory=false \
  "$@"
