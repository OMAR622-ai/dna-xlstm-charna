#!/bin/bash
# Waits for the running NT queue (seeds 1-3) to finish, moves its checkpoints to the local disk,
# then runs seeds 4-10 so that every task has the 10 repeats of Bio-xLSTM Table 1.
LOG=~/charna/dna-xlstm/outputs/downstream/nt_queue/queue_log.txt
until grep -q NT_DONE ${LOG}; do sleep 300; done
bash ~/charna/dna-xlstm/scripts_charna/server/move_nt_ckpts.sh >> ${LOG} 2>&1
cd ~/charna/dna-xlstm/scripts_charna/server && bash run_nt_benchmark.sh "4 5 6 7 8 9 10"
