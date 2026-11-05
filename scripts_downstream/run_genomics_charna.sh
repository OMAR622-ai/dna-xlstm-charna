#!/bin/bash

# Genomic Benchmarks (Gresova et al. 2023, 8 tasks) for the GBST + BERT model (MODEL=charna) and the
# same-shape Transformer++-2M baseline (MODEL=llama). Protocol of Bio-xLSTM (App. C.5) / run_genomics.sh:
#   post-hoc conjoining (PH): RC augmentation on the train split, forward/RC averaged at test time,
#   mean pooling + linear head, full fine-tuning, 10 epochs, 5 train/val splits (seeds),
#   best validation epoch tested, top-1 accuracy.
# Same launcher as run_nucleotide_charna.sh (scripts_charna/train_charna.py, fp32, metrics.csv per run,
# finished runs skipped on restart).
#
# Usage (from scripts_downstream/):
#   PRETRAIN_DIR=../outputs/pretrain/hg38/full_charna_mlm MODEL=charna bash run_genomics_charna.sh
#   TASKS="human_nontata_promoters" LRS="1e-3" BATCH_SIZES="128" SEEDS="1" ... for a subset.
# GB_DATA sets where the benchmark is downloaded (one small text file per sequence, so keep it off
# the 20 GB home disk). Extra arguments are passed to train.py as Hydra overrides.

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
PRETRAINED_PATH=${PRETRAINED_PATH:-"${PRETRAIN_DIR}/checkpoints/test/loss.ckpt"}
DISPLAY_NAME=${DISPLAY_NAME:-"$(basename "${PRETRAIN_DIR}")_ph"}

# PH setting (run_genomics.sh "PH models")
CONJOIN_TRAIN_DECODER="false"
CONJOIN_TEST="true"
RC_AUG="false"  # the dataset applies RC augmentation to the train split itself when conjoin_test=true

ALL_TASKS="dummy_mouse_enhancers_ensembl demo_coding_vs_intergenomic_seqs demo_human_or_worm
  human_enhancers_cohn human_enhancers_ensembl human_ensembl_regulatory human_ocr_ensembl human_nontata_promoters"
TASKS=${TASKS:-${ALL_TASKS}}
LRS=${LRS:-"1e-3 2e-3 4e-4 6e-4 8e-4"}  # run_genomics.sh grid
BATCH_SIZES=${BATCH_SIZES:-"64 128"}     # run_genomics.sh grid
SEEDS=${SEEDS:-$(seq 1 5)}
EPOCHS=${EPOCHS:-10}
PRECISION=${PRECISION:-32}
NUM_WORKERS=${NUM_WORKERS:-4}
GB_DATA=${GB_DATA:-/tmp/omohamed_genomic_benchmarks}
OUT_ROOT=${OUT_ROOT:-"./outputs/downstream/gb_cv5_ep${EPOCHS}"}

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
        echo "GB: ${DISPLAY_NAME}, TASK: ${TASK}, LR: ${LR}, BATCH_SIZE: ${BATCH_SIZE}, SEED: ${seed}"
        python scripts_charna/train_charna.py \
          experiment=hg38/genomic_benchmark \
          callbacks.model_checkpoint_every_n_steps.every_n_train_steps=5000 \
          +callbacks.metrics_csv.filename=metrics.csv \
          dataset.dataset_name="${TASK}" \
          dataset.dest_path="${GB_DATA}" \
          dataset.train_val_split_seed=${seed} \
          dataset.batch_size=$((BATCH_SIZE / 2)) \
          dataset.rc_aug="${RC_AUG}" \
          +dataset.conjoin_train=false \
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
