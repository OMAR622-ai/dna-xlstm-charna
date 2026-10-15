#!/bin/bash
# Move the checkpoints of finished (tested) NT runs to the local disk; leave a link behind.
STORE=/tmp/omohamed_nt_ckpts
mkdir -p ${STORE} && chmod 700 ${STORE}
cd ~/charna/dna-xlstm/outputs/downstream/nt_cv10_ep20
n=0
for m in */*/*/seed-*/metrics.csv; do
  run=$(dirname ${m})
  if grep -q "test/" ${m} && [ -d ${run}/checkpoints ] && [ ! -L ${run}/checkpoints ]; then
    rm -f ${run}/checkpoints/last.ckpt
    mkdir -p ${STORE}/${run} && mv ${run}/checkpoints ${STORE}/${run}/ && ln -s ${STORE}/${run}/checkpoints ${run}/checkpoints && n=$((n+1))
  fi
done
echo "moved ${n} runs"
du -sh ${STORE}
df -h ~ | tail -1
