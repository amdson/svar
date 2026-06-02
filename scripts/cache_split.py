"""
cache_split.py
--------------
Build the dataset + aligned phenotypes once and write a single train/val split
file, so every training run (any embedding cache, any model) trains and
evaluates on the *same* sample points.

The split is stored by sample ID (plus the aligned, z-scored targets and trait
names for convenience), and is reloaded with
`crop_embed.data.loading.load_split(path, dataset)` — which re-resolves the IDs
against whatever dataset you hand it.

Example
-------
    python scripts/cache_split.py \\
        --test-split 0.30 --seed 42 \\
        --output splits/sativas413_seed42.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crop_embed.data.coords import FASTA_PATH
from crop_embed.data.loading import (
    DEFAULT_VCF_PATH,
    build_dataset,
    load_targets,
    make_split,
    save_split,
)
from crop_embed.data.preprocessing import DEFAULT_PHENO_PATH

parser = argparse.ArgumentParser(description="Cache a deterministic train/val split.")
parser.add_argument("--vcf-path", type=str, default=DEFAULT_VCF_PATH)
parser.add_argument("--fasta-path", type=str, default=str(FASTA_PATH))
parser.add_argument("--pheno-path", type=str, default=str(DEFAULT_PHENO_PATH))
parser.add_argument("--half-window", type=int, default=500)
parser.add_argument("--buffer", type=int, default=0)
parser.add_argument("--test-split", type=float, default=0.30,
                    help="Fraction of samples held out for validation.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=str, required=True,
                    help="Path to write the split .pt file.")
args = parser.parse_args()

dataset = build_dataset(args.vcf_path, args.fasta_path, args.half_window, args.buffer)
Y, trait_cols = load_targets(dataset, args.pheno_path)

train_idx, val_idx = make_split(dataset.samples, args.test_split, args.seed)
train_ids = [dataset.samples[i] for i in train_idx.tolist()]
val_ids = [dataset.samples[i] for i in val_idx.tolist()]
print(f"Split: {len(train_ids)} train / {len(val_ids)} val "
      f"(seed={args.seed}, test_split={args.test_split})")

save_split(
    args.output,
    train_ids=train_ids,
    val_ids=val_ids,
    trait_cols=trait_cols,
    targets=Y,
    sample_ids=list(dataset.samples),
    metadata={
        "vcf_path": args.vcf_path,
        "fasta_path": args.fasta_path,
        "pheno_path": str(args.pheno_path),
        "half_window": args.half_window,
        "buffer": args.buffer,
        "test_split": args.test_split,
        "seed": args.seed,
    },
)
print(f"Saved split to {args.output}")
