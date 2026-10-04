#!/bin/bash

# Nucleotide Transformer benchmark for the GBST + BERT model (MODEL=charna) and the same-shape
# Transformer++-2M baseline (MODEL=llama). Same protocol as run_nucleotide.sh (DNA-xLSTM PH models):
#   post-hoc conjoining (PH): RC augmentation on the train split, forward/RC averaged at test time,
#   mean pooling + linear head, full fine-tuning, 20 epochs, 10 train/val splits (seeds),
#   LR x batch-size grid; the best grid point per task is picked by mean validation metric
#   (scripts_charna/summarize_nt.py).
# Differences from run_nucleotide.sh: launched through scripts_charna/train_charna.py (runs
# without mamba_ssm / flash_attn), fp32 (GTX 1080 Ti has no fast fp16/bf16), validation/test
# metrics written to metrics.csv in each run dir, and finished runs are skipped on restart.
#
# Usage (from scripts_downstream/):
#   PRETRAIN_DIR=../outputs/pretrain/hg38/full_charna_mlm MODEL=charna bash run_nucleotide_charna.sh
#   TASKS="H3 promoter_tata" LRS="1e-3" BATCH_SIZES="128" SEEDS="1 2" ... for a subset.
# Extra arguments are passed to train.py as Hydra overrides (e.g. trainer.limit_train_batches=10).
# OUT_ROOT overrides the output directory.

# Resolve the pretraining run directory relative to where the script is called from.
PRETRAIN_DIR=$(realpath -e "${PRETRAIN_DIR:?set PRETRAIN_DIR to the pretraining run directory}") || exit 1

cd ../ || exit

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

MODEL=${MODEL:-charna}  # charna | llama
case "${MODEL}" in
  charna) MODEL_NAME="dna_embedding_charna" ;;
  llama) MODEL_NAME="dna_embedding_llama" ;;
  *) echo "unknown MODEL=${MODEL}"; exit 1 ;;
esac
CONFIG_PATH="${PRETRAIN_DIR}/model_config.json"
PRETRAINED_PATH=${PRETRAINED_PATH:-"${PRETRAIN_DIR}/checkpoints/test/loss.ckpt"}  # as in run_nucleotide.sh
DISPLAY_NAME=${DISPLAY_NAME:-"$(basename "${PRETRAIN_DIR}")_ph"}

# PH setting (run_nucleotide.sh "PH models")
CONJOIN_TRAIN_DECODER="false"
CONJOIN_TEST="true"
RC_AUG="false"  # the dataset applies RC augmentation to the train split itself when conjoin_test=true

ALL_TASKS="H3 H3K14ac H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K79me3 H3K9ac H4 H4ac enhancers enhancers_types
  promoter_all promoter_no_tata promoter_tata splice_sites_all splice_sites_acceptors splice_sites_donors"
TASKS=${TASKS:-${ALL_TASKS}}
LRS=${LRS:-"1e-3 2e-3 4e-4 6e-4 8e-4"}  # run_nucleotide.sh grid
BATCH_SIZES=${BATCH_SIZES:-"64 128"}     # run_nucleotide.sh grid
SEEDS=${SEEDS:-$(seq 1 10)}
EPOCHS=${EPOCHS:-20}
PRECISION=${PRECISION:-32}
NUM_WORKERS=${NUM_WORKERS:-4}
OUT_ROOT=${OUT_ROOT:-"./outputs/downstream/nt_cv10_ep${EPOCHS}"}

for TASK in ${TASKS}; do
  for LR in ${LRS}; do
    for BATCH_SIZE in ${BATCH_SIZES}; do
      CONFIG_NAME="LR-${LR}_BATCH_SIZE-${BATCH_SIZE}"
      for seed in ${SEEDS}; do
        HYDRA_RUN_DIR="${OUT_ROOT}/${TASK}/${DISPLAY_NAME}/${CONFIG_NAME}/seed-${seed}"
        if grep -q "test/" "${HYDRA_RUN_DIR}/metrics.csv" 2>/dev/null; then
          echo "done already: ${HYDRA_RUN_DIR}"; continue
        fi
        rm -f "${HYDRA_RUN_DIR}/metrics.csv"  # drop rows of an interrupted attempt
        mkdir -p "${HYDRA_RUN_DIR}"
        echo "*****************************************************"
        echo "NT: ${DISPLAY_NAME}, TASK: ${TASK}, LR: ${LR}, BATCH_SIZE: ${BATCH_SIZE}, SEED: ${seed}"
        python scripts_charna/train_charna.py \
          experiment=hg38/nucleotide_transformer \
          callbacks.model_checkpoint_every_n_steps.every_n_train_steps=5000 \
          +callbacks.metrics_csv.filename=metrics.csv \
          dataset.dataset_name="${TASK}" \
          dataset.train_val_split_seed=${seed} \
          dataset.batch_size=$((BATCH_SIZE / 2)) \
          dataset.rc_aug="${RC_AUG}" \
          +dataset.conjoin_test="${CONJOIN_TEST}" \
          model="${MODEL}" \
          model._name_="${MODEL_NAME}" \
          +model.config_path="${CONFIG_PATH}" \
          +model.conjoin_test="${CONJOIN_TEST}" \
          +decoder.conjoin_train="${CONJOIN_TRAIN_DECODER}" \
          +decoder.conjoin_test="${CONJOIN_TEST}" \
          optimizer.lr="${LR}" \
          train.pretrained_model_path="${PRETRAINED_PATH}" \
          train.pretrained_model_strict_load=false \
          train.global_batch_size=${BATCH_SIZE} \
          trainer.max_epochs=${EPOCHS} \
          trainer.precision=${PRECISION} \
          wandb.mode=disabled \
          hydra.run.dir="${HYDRA_RUN_DIR}" \
          loader.num_workers=${NUM_WORKERS} \
          "$@" \
          > "${HYDRA_RUN_DIR}/stdout.log" 2>&1
        echo "exit=$? (log: ${HYDRA_RUN_DIR}/stdout.log)"
      done
    done
  done
done
