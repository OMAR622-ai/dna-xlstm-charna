#!/bin/bash
# After the learning-rate check: pick the GBST-BERT learning rate by validation loss at step 500
# (5e-4 from the small experiment, 1e-3, 2e-3, 5e-3), then full pretraining (10k steps = 10.5B
# bases) of GBST-BERT with that LR and of Transformer++-2M (LR 1e-2, Bio-xLSTM Table A3),
# then masked-base accuracy of both final models on held-out valid/test sequences.
source ~/envs/charna/bin/activate
ROOT=~/charna/dna-xlstm
OUT=${ROOT}/outputs/full
mkdir -p ${OUT}
log() { echo "$(date "+%F %T") $*" >> ${OUT}/queue.txt; }

until grep -q LR_CHECK_DONE ${ROOT}/outputs/lr_check/queue.txt 2>/dev/null; do sleep 120; done

BEST_LR=$(python - << "PY"
import csv, pathlib
root = pathlib.Path.home() / "charna/dna-xlstm/outputs/pretrain/hg38"
runs = {"5e-4": "small_charna_500", "1e-3": "small_charna_500_lr1e-3",
        "2e-3": "small_charna_500_lr2e-3", "5e-3": "small_charna_500_lr5e-3"}
best = None
for lr, name in runs.items():
    path = root / name / "metrics.csv"
    rows = list(csv.DictReader(open(path))) if path.exists() else []
    if rows and rows[-1]["global_step"] == "500":
        loss = float(rows[-1]["val/loss"])
        print(f"{lr}: val/loss {loss:.4f}", file=__import__("sys").stderr)
        if best is None or loss < best[1]:
            best = (lr, loss)
print(best[0])
PY
)
log "LR check done; best GBST-BERT LR by val/loss at step 500: ${BEST_LR}"

cd ${ROOT}/scripts_pretrain
log "start full pretrain charna (LR ${BEST_LR})"
MODEL=charna LR=${BEST_LR} RUN_NAME=full_charna_mlm VAL_EVERY=2000 bash run_pretrain_charna.sh > ${OUT}/full_charna_mlm.log 2>&1
STATUS=$?; log "end full pretrain charna exit=${STATUS}"

log "start full pretrain llama (LR 1e-2)"
MODEL=llama RUN_NAME=full_llama_mlm VAL_EVERY=2000 bash run_pretrain_charna.sh > ${OUT}/full_llama_mlm.log 2>&1
STATUS=$?; log "end full pretrain llama exit=${STATUS}"

cd ${ROOT}
for SPLIT in valid test; do
  CUDA_VISIBLE_DEVICES=0 python scripts_charna/evaluate_mlm.py \
    outputs/pretrain/hg38/full_charna_mlm:GBST-BERT outputs/pretrain/hg38/full_llama_mlm:Transformer++ \
    --split ${SPLIT} --n_seqs 2048 --csv ${OUT}/eval_${SPLIT}.csv --markdown ${OUT}/eval_${SPLIT}.md \
    > ${OUT}/eval_${SPLIT}.log 2>&1
  STATUS=$?; log "end evaluation ${SPLIT} exit=${STATUS}"
done
log FULL_DONE
