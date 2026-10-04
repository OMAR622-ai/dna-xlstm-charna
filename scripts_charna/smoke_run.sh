#!/bin/bash
# Phase-2 smoke run: ~50M tokens (48 steps x 2^20) for GBST + BERT and Transformer++-2M with
# identical data, schedule (compressed to 48 steps), validation slice and logging.
# Records wall time and peak GPU memory of our processes. Run from scripts_pretrain/.
#   bash ../scripts_charna/smoke_run.sh

STEPS=${STEPS:-48}
OUT=${OUT:-../outputs/smoke}
mkdir -p "${OUT}"

poll_memory() {  # peak MiB used by this user's GPU processes, sampled every 5 s
  local peak=0
  while true; do
    local mine
    mine=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null |
      while IFS=', ' read -r pid mem; do
        if [ "$(ps -o user= -p "$pid" 2>/dev/null)" = "$USER" ]; then echo "$mem"; fi
      done | sort -n | tail -1)
    if [ -n "$mine" ] && [ "$mine" -gt "$peak" ]; then peak=$mine; echo "$peak" > "$1"; fi
    sleep 5
  done
}

for MODEL in charna llama; do
  NAME="smoke_${MODEL}"
  poll_memory "${OUT}/${NAME}.peak_mem_mib" &
  POLL=$!
  START=$(date +%s)
  MODEL=${MODEL} RUN_NAME=${NAME} MAX_STEPS=${STEPS} VAL_EVERY=12 LIMIT_VAL=64 CKPT_EVERY=100000 \
    bash run_pretrain_charna.sh > "${OUT}/${NAME}.log" 2>&1
  STATUS=$?
  END=$(date +%s)
  kill ${POLL} 2>/dev/null
  echo "${NAME} exit=${STATUS} wall_s=$(( END - START )) peak_mem_mib=$(cat "${OUT}/${NAME}.peak_mem_mib" 2>/dev/null)" | tee -a "${OUT}/summary.txt"
done
echo SMOKE_DONE >> "${OUT}/summary.txt"
