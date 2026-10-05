#!/bin/bash
# Runs on student01: wait until the full GBST-BERT run has saved its step-3000 checkpoint, then
# stop that run (and its queue script) and leave a flag file in the shared home for student07.
source ~/envs/charna/bin/activate
ROOT=~/charna/dna-xlstm
CKPT=${ROOT}/outputs/pretrain/hg38/full_charna_mlm/checkpoints/last.ckpt
FLAG=${ROOT}/outputs/full/moved_to_student07.flag
TARGET=3000
log() { echo "$(date "+%F %T") [student01] $*" >> ${ROOT}/outputs/full/queue.txt; }

while true; do
  STEP=$(python -c "import torch; print(torch.load('${CKPT}', map_location='cpu')['global_step'])" 2>/dev/null)
  if [ -n "${STEP}" ] && [ "${STEP}" -ge "${TARGET}" ]; then break; fi
  sleep 60
done
sleep 30  # let the checkpoint callback finish
log "checkpoint at step ${STEP} saved; stopping the run here to continue on student07"
kill $(pgrep -u "$USER" -f "[r]un_full_after_lr.sh") 2>/dev/null
kill $(pgrep -u "$USER" -f "[b]ash run_pretrain_charna.sh|[s]cripts_charna/train_charna.py") 2>/dev/null
while pgrep -u "$USER" -f "[s]cripts_charna/train_charna.py" > /dev/null; do sleep 10; done
STEP=$(python -c "import torch; print(torch.load('${CKPT}', map_location='cpu')['global_step'])" 2>/dev/null)
echo "${STEP}" > "${FLAG}"
log "stopped; last checkpoint step ${STEP}"
