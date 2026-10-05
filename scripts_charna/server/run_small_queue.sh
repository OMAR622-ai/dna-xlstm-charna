#!/bin/bash
# Small experiment: pretrain both models on ~0.5B tokens (500 steps of 2^20 tokens, 5% of the
# full run; LR schedule scaled to 500 steps: 50 warmup steps), then fine-tune on 4 NT tasks with
# the DNA-xLSTM-PH settings of Bio-xLSTM Table A5, 3 seeds, 20 epochs, PH (forward + RC averaged).
source ~/envs/charna/bin/activate
ROOT=~/charna/dna-xlstm
OUT=${ROOT}/outputs/small
mkdir -p ${OUT}
log() { echo "$(date "+%F %T") $*" >> ${OUT}/queue.txt; }

cd ${ROOT}/scripts_pretrain
for MODEL in charna llama; do
  log "start pretrain ${MODEL}"
  MODEL=${MODEL} RUN_NAME=small_${MODEL}_500 MAX_STEPS=500 VAL_EVERY=100 LIMIT_VAL=0.02 CKPT_EVERY=100 \
    bash run_pretrain_charna.sh > ${OUT}/pretrain_${MODEL}.log 2>&1
  STATUS=$?; log "end pretrain ${MODEL} exit=${STATUS}"
done

# task, LR, batch size (Table A5, DNA-xLSTM-Ph column), GPU
JOBS="H3:8e-4:128:0 enhancers:2e-3:512:1 promoter_tata:3e-3:128:2 splice_sites_all:8e-4:64:3"
cd ${ROOT}/scripts_downstream
for MODEL in charna llama; do
  log "start NT ${MODEL}"
  for JOB in ${JOBS}; do
    IFS=: read -r TASK LR BS GPU <<< "${JOB}"
    CUDA_VISIBLE_DEVICES=${GPU} MODEL=${MODEL} PRETRAIN_DIR=../outputs/pretrain/hg38/small_${MODEL}_500 \
      OUT_ROOT=./outputs/downstream/nt_small TASKS=${TASK} LRS=${LR} BATCH_SIZES=${BS} SEEDS="1 2 3" NUM_WORKERS=2 \
      bash run_nucleotide_charna.sh > ${OUT}/nt_${MODEL}_${TASK}.log 2>&1 &
  done
  wait
  log "end NT ${MODEL}"
done
cd ${ROOT} && python scripts_charna/summarize_nt.py outputs/downstream/nt_small --seeds 3 --markdown ${OUT}/nt_results.md >> ${OUT}/queue.txt 2>&1
log SMALL_DONE
