#!/bin/bash
# Full pretraining queue: GBST-BERT (LR 5e-4, ChaRNABERT Table 2), then Transformer++-2M
# (LR 1e-2, Bio-xLSTM Table A3); 10k steps = 10.5B tokens each. LRs are the script defaults.
source ~/envs/charna/bin/activate
cd ~/charna/dna-xlstm/scripts_pretrain
mkdir -p ../outputs/full
for MODEL in charna llama; do
  NAME="full_${MODEL}_mlm"
  echo "$(date) start ${NAME}" >> ../outputs/full/queue.txt
  MODEL=${MODEL} RUN_NAME=${NAME} VAL_EVERY=2000 bash run_pretrain_charna.sh > ../outputs/full/${NAME}.log 2>&1
  STATUS=$?
  echo "$(date) end ${NAME} exit=${STATUS}" >> ../outputs/full/queue.txt
done
echo QUEUE_DONE >> ../outputs/full/queue.txt
