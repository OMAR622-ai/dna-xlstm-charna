#!/bin/bash
# Seeds 4-10 of the NT benchmark, so that every task has the 10 repeats of Bio-xLSTM Table 1.
# Settings: nt_tables/final_settings.txt (Table A5; splice acceptors/donors with the learning rate
# chosen by validation score in the splice learning-rate check). Waits for a running queue to finish.
# Start on the server (keeps running after logout):
#   setsid nohup bash ~/charna/dna-xlstm/scripts_charna/server/chain_nt_seeds_4_10.sh > ~/charna/chain_nt.out 2>&1 < /dev/null &
LOG=~/charna/dna-xlstm/outputs/downstream/nt_queue/queue_log.txt
until grep -q NT_DONE ${LOG}; do sleep 300; done
cd ~/charna/dna-xlstm/scripts_charna/server && \
  OUT_DIR=~/charna/dna-xlstm/outputs/downstream/nt_queue_seeds_4_10 NT_TABLE=nt_tables/final_settings.txt \
  bash run_nt_benchmark.sh "4 5 6 7 8 9 10"
