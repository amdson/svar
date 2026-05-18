"""
demo_train.py
-------------
Train a head model on rice phenotypes using per-window DNA embeddings, with
either cached embeddings (frozen embedder) or end-to-end through DNABERT-2.

Four supported configurations. For trainable-embedder runs (3 and 4), the
script does a two-phase pass by default: head-only for --head-only-steps first,
then end-to-end for --steps. Set --head-only-steps 0 to skip the warmup phase.

  # 1. Frozen-model, cached embeddings, linear head
  python scripts/demo_train.py \\
      --head linear \\
      --cache checkpoints/v1/sativas413_embeddings.ckpt.pt \\
      --steps 2000 --batch-size 32 --lr 1e-3

  # 2. Frozen-model, cached embeddings, attention head (with positional encoding)
  python scripts/demo_train.py \\
      --head attention --positional \\
      --cache checkpoints/v1/sativas413_embeddings.ckpt.pt \\
      --steps 2000 --batch-size 32 --lr 1e-3

  # 3. Trainable model (end-to-end), linear head — head warmup, then full
  python scripts/demo_train.py \\
      --head linear \\
      --head-only-steps 500 --head-only-lr 1e-3 \\
      --steps 500 --lr 1e-5 \\
      --batch-size 8 --precision bf16

  # 4. Trainable model (end-to-end), attention head with positional — same two-phase
  python scripts/demo_train.py \\
      --head attention --positional \\
      --head-only-steps 500 --head-only-lr 1e-3 \\
      --steps 500 --lr 1e-5 \\
      --batch-size 8 --precision bf16

Outputs (head state dict, embedder state dict if trainable, args, trait names)
are saved under {output-dir}/{run-name}/model.pt.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

# crop_embed and DNABERT2_modules live one level up
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crop_embed import (
    AttentionHead,
    CachedWindowEmbedder,
    LinearHead,
    SNPWindowPartitioner,
    UniqueWindowDataset,
    WindowEmbedder,
    train,
    window_position_features,
)
from crop_embed.data.coords import FASTA_PATH
from crop_embed.data.preprocessing import (
    DEFAULT_PHENO_PATH,
    align_targets_to_dataset,
    load_phenotypes,
    scale_phenotypes,
)
from crop_embed.data.vcf import load_snps_from_vcf
from DNABERT2_modules import load_dnabert2

# ── CLI ───────────────────────────────────────────────────────────────────────

DEFAULT_VCF_PATH = "/home/andrew.dickson/rice_data/sativas413_msu7_final.vcf"
DEFAULT_MODEL    = "zhihan1996/DNABERT-2-117M"

parser = argparse.ArgumentParser(description="Train a head on DNA window embeddings.")

# Embedder selection: cached (frozen) vs trainable
parser.add_argument("--cache", type=str, default=None,
                    help="Path to a fill_embedding_table checkpoint. If set, "
                         "the embedder is frozen and only the head trains.")
parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL,
                    help="HuggingFace repo or local dir for the DNA encoder "
                         "(only used when --cache is not given).")

# Head selection
parser.add_argument("--head", choices=["linear", "attention"], required=True)
parser.add_argument("--positional", action="store_true",
                    help="Add learnable per-chrom + sinusoidal position embeddings "
                         "to the windows before the head. Attention head only.")
parser.add_argument("--n-queries",  type=int, default=8)
parser.add_argument("--n-heads",    type=int, default=8)
parser.add_argument("--hidden-dim", type=int, default=None)

# Data paths and windowing
parser.add_argument("--vcf-path",   type=str, default=DEFAULT_VCF_PATH)
parser.add_argument("--fasta-path", type=str, default=FASTA_PATH)
parser.add_argument("--pheno-path", type=str, default=str(DEFAULT_PHENO_PATH))
parser.add_argument("--half-window", type=int, default=500)
parser.add_argument("--buffer",      type=int, default=0)
parser.add_argument("--max-length",  type=int, default=2048,
                    help="Tokenizer truncation length (trainable embedder only).")

# Training hyperparameters
parser.add_argument("--steps",      type=int,   default=1000,
                    help="Main training steps. With a trainable embedder this "
                         "is the end-to-end phase; with --cache it's the only phase.")
parser.add_argument("--lr",         type=float, default=1e-4)
parser.add_argument("--head-only-steps", type=int, default=500,
                    help="Head-only warmup steps before unfreezing the embedder. "
                         "Ignored when --cache is set (embedder has no params). "
                         "Set to 0 to skip the warmup phase entirely.")
parser.add_argument("--head-only-lr", type=float, default=1e-3)
parser.add_argument("--batch-size", type=int,   default=32)
parser.add_argument("--precision",  choices=["fp32", "bf16", "fp16"], default="fp32",
                    help="Mixed precision. fp16 uses GradScaler; bf16 doesn't need one.")
parser.add_argument("--log-every",  type=int,   default=50)
parser.add_argument("--seed",       type=int,   default=42)

# I/O
parser.add_argument("--output-dir", type=str, default="trained_heads")
parser.add_argument("--run-name",   type=str, default=None,
                    help="Subdirectory under --output-dir; default is auto-named "
                         "from --head / --cache / --positional.")

args = parser.parse_args()
torch.manual_seed(args.seed)

if args.positional and args.head != "attention":
    parser.error("--positional only applies to --head attention")

# ── Build dataset ─────────────────────────────────────────────────────────────

print("Loading SNPs from VCF …")
snps_by_chrom, samples = load_snps_from_vcf(args.vcf_path)
n_snps = sum(len(v) for v in snps_by_chrom.values())
print(f"  {n_snps:,} SNPs | {len(snps_by_chrom)} chromosomes | {len(samples)} samples")

print("Building SNPWindowPartitioner …")
partitioner = SNPWindowPartitioner(
    snps_by_chrom, half_window=args.half_window, buffer=args.buffer
)
stats = partitioner.snps_per_window_stats()
print(f"  {stats['n_windows']:,} windows "
      f"(mean {stats['mean']:.1f} SNPs/window)")

print("Building UniqueWindowDataset …")
dataset = UniqueWindowDataset(args.vcf_path, args.fasta_path, partitioner)
print(f"  {len(dataset):,} unique windows; {len(dataset.samples)} samples")

# ── Load and align phenotypes ─────────────────────────────────────────────────

print(f"Loading phenotypes from {args.pheno_path} …")
pheno_df, trait_cols = load_phenotypes(args.pheno_path)
Y_np = align_targets_to_dataset(dataset, pheno_df, trait_cols)
Y_np = scale_phenotypes(Y_np)                          # per-trait z-score, NaN-safe
Y    = torch.tensor(Y_np, dtype=torch.float32)
n_with_pheno = (~torch.isnan(Y).all(dim=1)).sum().item()
print(f"  {Y.shape[1]} traits; {n_with_pheno}/{Y.shape[0]} samples have phenotype data")

# ── Build embedder ────────────────────────────────────────────────────────────

if args.cache:
    print(f"Loading cached embeddings from {args.cache} …")
    embedder = CachedWindowEmbedder.from_checkpoint(args.cache)
    emb_dim  = embedder.cache.size(1)
    print(f"  {embedder.cache.size(0):,} cached fingerprints × {emb_dim} dims")
else:
    print(f"Loading DNABERT-2 from {args.model_path} …")
    _local    = os.path.isdir(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=False, local_files_only=_local,
    )
    model, _  = load_dnabert2(
        repo_id=args.model_path,
        add_pooling_layer=False,
        config_overrides={"pad_token_id": tokenizer.pad_token_id},
    )
    embedder = WindowEmbedder(model, tokenizer, max_length=args.max_length)
    emb_dim  = model.config.hidden_size
    print(f"  emb_dim = {emb_dim}")

# ── Build head ────────────────────────────────────────────────────────────────

n_traits = Y.shape[1]
if args.head == "linear":
    head = LinearHead(emb_dim=emb_dim, n_traits=n_traits)
else:
    head_kwargs = dict(
        emb_dim=emb_dim, n_traits=n_traits,
        n_queries=args.n_queries, n_heads=args.n_heads,
        hidden_dim=args.hidden_dim,
    )
    if args.positional:
        chroms, positions = window_position_features(dataset)
        head_kwargs.update(
            positional=True, window_chroms=chroms, window_positions=positions,
        )
    head = AttentionHead(**head_kwargs)

n_head_params = sum(p.numel() for p in head.parameters())
n_emb_params  = sum(p.numel() for p in embedder.parameters())
print(f"Head: {args.head}{' (+positional)' if args.positional else ''}  "
      f"— {n_head_params:,} params")
print(f"Embedder trainable params: {n_emb_params:,}")

# ── Train ─────────────────────────────────────────────────────────────────────

trainable_embedder = args.cache is None

# Phase 1: head-only warmup. Skip when embedder is already frozen (cached) or
# when the user opted out with --head-only-steps 0.
if trainable_embedder and args.head_only_steps > 0:
    print(f"\nPhase 1: head-only warmup — {args.head_only_steps} steps at lr={args.head_only_lr}")
    for p in embedder.parameters():
        p.requires_grad_(False)
    train(
        embedder, head, dataset, Y,
        batch_size=args.batch_size,
        lr=args.head_only_lr,
        steps=args.head_only_steps,
        precision=args.precision,
        log_every=args.log_every,
    )
    for p in embedder.parameters():
        p.requires_grad_(True)

# Phase 2: full training (or only phase for cached / opt-out).
phase_label = "Phase 2: full end-to-end" if (trainable_embedder and args.head_only_steps > 0) else "Training"
print(f"\n{phase_label} — {args.steps} steps at lr={args.lr}, precision={args.precision}")
train(
    embedder, head, dataset, Y,
    batch_size=args.batch_size,
    lr=args.lr,
    steps=args.steps,
    precision=args.precision,
    log_every=args.log_every,
)

# ── Save ──────────────────────────────────────────────────────────────────────

if args.run_name is None:
    mode_tag = "cached" if args.cache else "e2e"
    pos_tag  = "_pos"  if args.positional else ""
    args.run_name = f"{args.head}_{mode_tag}{pos_tag}"

out_dir = Path(args.output_dir) / args.run_name
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "model.pt"
torch.save({
    "head_state_dict":     head.state_dict(),
    "embedder_state_dict": embedder.state_dict(),
    "trait_cols":          trait_cols,
    "sample_ids":          dataset.samples,
    "args":                vars(args),
}, out_path)
print(f"\nSaved to {out_path}")
