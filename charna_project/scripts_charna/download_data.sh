#!/bin/bash
# Download the full pretraining data (hg38, 3.1 GB unpacked) into data/hg38/ and check it,
# and pre-download the Nucleotide Transformer benchmark tasks (~350 MB, Hugging Face cache).
# The files are too large for GitHub; a small sample is in charna_project/datasets/hg38_sample.
# Sources: main README "Data Preparation" (hg38), InstaDeepAI/nucleotide_transformer_downstream_tasks (NT).
# Run from the repo root:  bash scripts_charna/download_data.sh [--no-nt]

set -euo pipefail
URL=https://ml.jku.at/research/Bio-xLSTM/downloads/DNA-xLSTM/data/hg38
mkdir -p data/hg38

if [ ! -f data/hg38/hg38.ml.fa ]; then
  curl -L --fail "${URL}/hg38.ml.fa.gz" -o data/hg38/hg38.ml.fa.gz
  gunzip data/hg38/hg38.ml.fa.gz
fi
[ -f data/hg38/human-sequences.bed ] || curl -L --fail "${URL}/human-sequences.bed" -o data/hg38/human-sequences.bed

# md5 sums of the files used for all runs in this project
md5sum -c - << "EOF"
c2cc5a203b71fff7fbf750f9276b3e8a  data/hg38/hg38.ml.fa
2c256f1c667fe3318865303248e70659  data/hg38/human-sequences.bed
EOF

if [ "${1:-}" != "--no-nt" ]; then
  python - << "EOF"
from datasets import load_dataset
tasks = ["H3", "H3K14ac", "H3K36me3", "H3K4me1", "H3K4me2", "H3K4me3", "H3K79me3", "H3K9ac", "H4", "H4ac",
         "enhancers", "enhancers_types", "promoter_all", "promoter_no_tata", "promoter_tata",
         "splice_sites_all", "splice_sites_acceptors", "splice_sites_donors"]
for task in tasks:
    for split in ("train", "test"):
        d = load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks",
                         data_files={split: f"{task}/{split}.parquet"}, split=split)
    print(f"NT {task}: cached")
EOF
fi
echo "data ready"
