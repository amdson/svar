"""
demo_embed.py
-------------
Embed the sanity-checked sativas413_msu7.vcf with DNABERT-2 via SampleEmbedder.

Produces a .pt file with shape (n_samples, embedding_dim) and a matching
list of sample IDs, ready for downstream phenotype prediction.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

# crop_embed and DNABERT2_modules live in variant_cache/
sys.path.insert(0, str(Path(__file__).parent))

from DNABERT2_modules import load_dnabert2

from crop_embed.data.coords import FASTA_PATH
from crop_embed.data.vcf import load_snps_from_vcf
from crop_embed.dataset import UniqueWindowDataset
from crop_embed.embedder import SampleEmbedder
from crop_embed.partitioner import SNPWindowPartitioner

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Embed sativas413 VCF with DNABERT-2.")
parser.add_argument(
    "--snp-only",
    action="store_true",
    default=False,
    help="Pool only SNP-containing tokens instead of the full window. "
         "Produces a separate output file. Delete any existing checkpoint "
         "before switching modes.",
)
args = parser.parse_args()
SNP_ONLY = args.snp_only

# ── Paths ─────────────────────────────────────────────────────────────────────

VCF_PATH   = "/home/adickson/rice_data/sativas413_msu7.vcf"
MODEL_PATH = "zhihan1996/DNABERT-2-117M"
OUT_PATH        = "sativas413_embeddings_snponly.pt" if SNP_ONLY else "sativas413_embeddings.pt"
CHECKPOINT_PATH = OUT_PATH.replace(".pt", ".ckpt.pt")  # delete before switching modes

# ── Windowing parameters ──────────────────────────────────────────────────────

HALF_WINDOW      = 500   # 1 000 bp total; matches the smallest context_len from the original code
BUFFER           = 50    # min bases from a SNP to either window edge
BATCH_SIZE       = 64
MAX_LENGTH       = 512   # tokenizer truncation limit
CHECKPOINT_EVERY = 500   # save checkpoint every N batches (~32 k windows at batch_size=64)

# ── Build dataset ─────────────────────────────────────────────────────────────

print("Loading SNPs from VCF …")
snps_by_chrom, samples = load_snps_from_vcf(VCF_PATH)
n_snps = sum(len(v) for v in snps_by_chrom.values())
print(f"  {n_snps:,} SNPs | {len(snps_by_chrom)} chromosomes | {len(samples)} samples")

print("Building SNPWindowPartitioner …")
partitioner = SNPWindowPartitioner(snps_by_chrom, half_window=HALF_WINDOW, buffer=BUFFER)
stats = partitioner.snps_per_window_stats()
print(f"  {stats['n_windows']:,} windows, {stats['total_snps']:,} SNPs assigned "
      f"(mean {stats['mean']:.1f} SNPs/window)")

print("Building UniqueWindowDataset …")
dataset = UniqueWindowDataset(VCF_PATH, FASTA_PATH, partitioner)
print(f"  {len(dataset):,} unique windows to embed "
      f"(vs {len(partitioner) * len(samples):,} sample×window pairs)")

# ── Load model ────────────────────────────────────────────────────────────────

print(f"\nLoading DNABERT-2 from {MODEL_PATH} …")
_local = os.path.isdir(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False, local_files_only=_local)
model, _  = load_dnabert2(
    repo_id=MODEL_PATH,
    add_pooling_layer=False,
    config_overrides={"pad_token_id": tokenizer.pad_token_id},
)

# ── Embed unique windows ──────────────────────────────────────────────────────

print("Running fill_embedding_table …")
embedding_table = SampleEmbedder.fill_embedding_table(
    dataset,
    model,
    tokenizer,
    batch_size=BATCH_SIZE,
    max_length=MAX_LENGTH,
    checkpoint_path=CHECKPOINT_PATH,
    checkpoint_every=CHECKPOINT_EVERY,
    snp_only=SNP_ONLY,
)
print(f"  {len(embedding_table):,} unique windows embedded")

# ── Aggregate per sample ──────────────────────────────────────────────────────

print("Aggregating per-sample embeddings (mean pooling over windows) …")
embedder          = SampleEmbedder(dataset, embedding_table, aggregation="mean")
sample_embeddings = embedder.embed_all()

embedding_dim = next(iter(sample_embeddings.values())).shape[0]
print(f"  {len(sample_embeddings)} samples × {embedding_dim} dims")

# ── Save ──────────────────────────────────────────────────────────────────────

sample_ids  = list(sample_embeddings.keys())
emb_matrix  = torch.stack([sample_embeddings[s] for s in sample_ids])  # (N, D)

torch.save({"sample_ids": sample_ids, "embeddings": emb_matrix}, OUT_PATH)
print(f"\nSaved to {OUT_PATH}  —  shape {tuple(emb_matrix.shape)}")
