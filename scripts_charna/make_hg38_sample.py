"""Build a small hg38 sample (a few real pretraining intervals) that fits in the git repository.

The full genome (hg38.ml.fa, 3.1 GB) is too large for GitHub. This script picks a fixed random
subset of the intervals in human-sequences.bed (per split), writes each interval as its own FASTA
record named <chr>_<start>_<end>, and writes a matching BED file with coordinates relative to
that record. HG38Dataset reads the sample exactly like the full data, so the tests, the overfit
check and short pretraining runs work on it.

  python scripts_charna/make_hg38_sample.py --fasta data/hg38/hg38.ml.fa --bed data/hg38/human-sequences.bed \
      --out charna_project/datasets/hg38_sample
"""

import argparse
import pathlib
import random

from pyfaidx import Fasta

COUNTS = {"train": 24, "valid": 4, "test": 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--bed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [line.split() for line in open(args.bed) if line.strip()]
    rng = random.Random(args.seed)
    chosen = []
    for split, count in COUNTS.items():
        candidates = [r for r in rows if r[3] == split]
        chosen += sorted(rng.sample(candidates, count), key=lambda r: (r[0], int(r[1])))

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    genome = Fasta(args.fasta)
    with open(out / "hg38_sample.fa", "w") as fa, open(out / "human-sequences_sample.bed", "w") as bed:
        for chrom, start, end, split in chosen:
            name = f"{chrom}_{start}_{end}"
            seq = str(genome[chrom][int(start):int(end)])
            fa.write(f">{name}\n")
            for i in range(0, len(seq), 60):
                fa.write(seq[i:i + 60] + "\n")
            bed.write(f"{name}\t0\t{len(seq)}\t{split}\n")
    print(f"wrote {len(chosen)} intervals ({', '.join(f'{k}: {v}' for k, v in COUNTS.items())}) to {out}")


if __name__ == "__main__":
    main()
