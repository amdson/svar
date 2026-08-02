"""
embed_windows.py
-----------------
Precompute a FixedWindowEmbedder cache for a VCF + reference FASTA + a DNA
language model, and write it to disk in a format that demo_train.py's --cache
flag loads directly via FixedWindowEmbedder.from_file().

Two encoder backends are supported via ``--backend``:
  * ``dnabert2``  (default) — zhihan1996/DNABERT-2-117M and its 117M-param family.
  * ``plantcad``           — kuleshov-group/PlantCaduceus_* (Caduceus/Mamba).
                              Single-nucleotide tokenization, 512 bp max input,
                              and reverse-complement parameter sharing (the
                              cache stores RC-averaged (D//2)-dim embeddings).

The output is a torch-saved dict with:

  cache           : Tensor(n_unique_windows, emb_dim) — the precomputed table
  sample_fp_index : LongTensor(n_samples, n_windows)  — sample → window indexing
  metadata        : dict with everything needed to reconstruct the embedder
                    (backend, model_path, max_length, snp_only, output_layer,
                     emb_dim, vcf_path, fasta_path, half_window, buffer,
                     n_samples, n_windows, generated_at, ...)
  unique_fingerprints : list[(chrom, w_start, w_end, alt_positions)]
                        in cache-row order — lets downstream code verify the
                        dataset matches the cache without re-running the VCF
                        pipeline.
  sample_ids          : list[str] — ordered to match sample_fp_index rows.

The top-level `cache` and `sample_fp_index` keys keep this drop-in compatible
with FixedWindowEmbedder.from_file / demo_train.py --cache.

Examples
--------
    # DNABERT-2 (default)
    python scripts/generate_cache.py \\
        --half-window 500 --buffer 0 --max-length 2048 --batch-size 64 \\
        --output checkpoints/sativas413_embeddings.ckpt.pt
    python train_pipeline/embed_windows.py --half-window 500 --buffer 0 --max-length 2048 --batch-size 64 --output checkpoints/sativas413_embeddings.ckpt.pt

    # PlantCAD — note half_window <= 256 (PlantCAD's input cap is 512 bp)
    python scripts/generate_cache.py --backend plantcad \\
        --half-window 256 --max-length 512 --batch-size 16 \\
        --output checkpoints/plantcad/sativas413_pc_l32.ckpt.pt
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crop_embed import (
    CARBON_MODEL_PATHS,
    DEFAULT_MODEL_PATHS,
    FixedWindowEmbedder,
    SampleEmbedder,
    SNPWindowPartitioner,
    UniqueWindowDataset,
    load_encoder_model,
)
from crop_embed.data.coords import FASTA_PATH
from crop_embed.data.vcf import load_snps_from_vcf

# DEFAULT_VCF_PATH    = "/home/andrew.dickson/rice_data/sativas413_msu7_final.vcf"
DEFAULT_VCF_PATH    = "/home/andrew/rice_data/sativas413_msu7_final.vcf"

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Precompute and save a FixedWindowEmbedder cache."
)

# Model
parser.add_argument("--backend", choices=["dnabert2", "plantcad", "carbon"], default="dnabert2",
                    help="Encoder family. Picks default --model-path if --model-path "
                         "is left unset, and routes loading / forward semantics.")
parser.add_argument("--model-path", type=str, default=None,
                    help="HuggingFace repo or local dir for the DNA encoder. "
                         "Defaults to zhihan1996/DNABERT-2-117M for dnabert2, "
                         "kuleshov-group/PlantCaduceus_l32 for plantcad, and the "
                         "--carbon-size checkpoint for carbon (500M by default). "
                         "Overrides --carbon-size when set explicitly.")
parser.add_argument("--carbon-size", choices=list(CARBON_MODEL_PATHS), default="500M",
                    help="Which Carbon checkpoint to use (carbon backend only): "
                         f"{', '.join(CARBON_MODEL_PATHS)}. Ignored when --model-path "
                         "is given. Defaults to 500M.")
parser.add_argument("--max-length", type=int, default=2048,
                    help="Tokenizer truncation length. PlantCAD caps at 512.")
parser.add_argument("--snp-only", action="store_true",
                    help="Pool only SNP-containing tokens (falls back to "
                         "full-window pool on pure-reference windows). "
                         "Supported on dnabert2 and carbon; not plantcad "
                         "(Caduceus tokenizer returns no offset mapping).")
parser.add_argument("--output-layer", type=int, default=None, metavar="N",
                    help="Extract hidden states from encoder layer N (0-based; "
                         "negative counts from the end) instead of the final layer.")
parser.add_argument("--attn-impl", choices=["torch", "flash"], default="torch",
                    help="DNABERT-2 attention backend: 'torch' (default, pure-PyTorch "
                         "matmul; needs no flash-attn) or 'flash' (FlashAttention; "
                         "requires CUDA + the flash-attn package). dnabert2 backend only.")

# Data / windowing
parser.add_argument("--vcf-path",   type=str, default=DEFAULT_VCF_PATH)
parser.add_argument("--fasta-path", type=str, default=FASTA_PATH)
parser.add_argument("--half-window", type=int, default=500)
parser.add_argument("--buffer",      type=int, default=0)

# Compute
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--device", type=str, default=None,
                    help="torch device (default: cuda if available, else cpu).")

# I/O
parser.add_argument("--output", type=str, required=True,
                    help="Path to write the cache .pt file.")
parser.add_argument("--checkpoint-every", type=int, default=500,
                    help="Save a resume-able embedding-table checkpoint every N batches.")
parser.add_argument("--checkpoint-path", type=str, default=None,
                    help="Path for the intermediate {Fingerprint: Tensor} checkpoint. "
                         "Defaults to <output>.embtable.ckpt.pt next to --output.")
parser.add_argument("--reuse-windows-from", type=str, default=None,
                    help="Path to an existing embedding cache (.ckpt.pt) whose window "
                         "structure (unique_fingerprints + sample_fp_index) to reuse, "
                         "skipping the expensive VCF windowing / build_sample_window_map "
                         "pass (~130 GB / >1 h on soy). Windows are backbone-independent, "
                         "so this only re-embeds the same unique windows with the new "
                         "backbone. Still needs --vcf-path (per-position alt bytes).")

args = parser.parse_args()

if args.model_path is None:
    if args.backend == "carbon":
        args.model_path = CARBON_MODEL_PATHS[args.carbon_size]
    else:
        args.model_path = DEFAULT_MODEL_PATHS[args.backend]
if args.backend == "plantcad" and args.snp_only:
    parser.error("--snp-only is not supported with --backend plantcad")
if args.backend == "plantcad" and args.half_window > 256:
    print(
        f"WARNING: PlantCAD's max input is 512 bp; --half-window={args.half_window} "
        f"produces {2 * args.half_window} bp windows which will be truncated."
    )

device = torch.device(
    args.device if args.device is not None
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
print(f"Device: {device}  Backend: {args.backend}  Model: {args.model_path}")

out_path = Path(args.output)
out_path.parent.mkdir(parents=True, exist_ok=True)
ckpt_path = args.checkpoint_path or str(out_path) + ".embtable.ckpt.pt"

# ── Build (or reuse) the window dataset ──────────────────────────────────────

if args.reuse_windows_from:
    print(f"Reusing window structure from {args.reuse_windows_from} "
          f"(skipping VCF windowing / build_sample_window_map) …")
    dataset = UniqueWindowDataset.from_cached_windows(
        args.reuse_windows_from, args.fasta_path, vcf_path=args.vcf_path
    )
    src_hw = dataset.source_metadata.get("half_window")
    if src_hw is not None and src_hw != args.half_window:
        parser.error(
            f"--half-window {args.half_window} != source cache half_window {src_hw}; "
            f"the reused windows were built at hw={src_hw} (pass --half-window {src_hw})."
        )
    n_windows = int(dataset.sample_fp_index.shape[1])
    print(f"  {len(dataset):,} unique windows to embed "
          f"({dataset.sample_fp_index.shape[0]:,} samples × {n_windows:,} windows; "
          f"no sample×window enumeration)")
else:
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
    n_windows = len(partitioner)
    print(f"  {len(dataset):,} unique windows to embed "
          f"(vs {n_windows * len(dataset.samples):,} sample×window pairs)")

# ── Load model ────────────────────────────────────────────────────────────────

print(f"\nLoading {args.backend} encoder from {args.model_path} …")
model, tokenizer = load_encoder_model(
    args.backend, args.model_path, device=device, attn_impl=args.attn_impl
)

# ── Embed unique windows (with checkpoint/resume) ────────────────────────────

print(f"\nRunning fill_embedding_table → {ckpt_path}")
t0 = time.time()
embedding_table = SampleEmbedder.fill_embedding_table(
    dataset,
    model,
    tokenizer,
    batch_size=args.batch_size,
    max_length=args.max_length,
    device=device,
    checkpoint_path=ckpt_path,
    checkpoint_every=args.checkpoint_every,
    snp_only=args.snp_only,
    output_layer=args.output_layer,
    backend=args.backend,
)
elapsed = time.time() - t0
print(f"  {len(embedding_table):,} unique windows embedded in {elapsed:.1f}s")

# ── Build FixedWindowEmbedder and save with metadata ─────────────────────────

fixed   = FixedWindowEmbedder.from_embedding_table(embedding_table, dataset)
emb_dim = int(fixed.cache.shape[1])

metadata = {
    "generated_at":  datetime.now(timezone.utc).isoformat(),
    "generated_by":  "scripts/generate_cache.py",
    "elapsed_sec":   elapsed,
    "device":        str(device),

    # Embedder reconstruction
    "backend":       args.backend,
    "model_path":    args.model_path,
    "max_length":    args.max_length,
    "snp_only":      args.snp_only,
    "output_layer":  args.output_layer,
    "attn_impl":     args.attn_impl,
    "emb_dim":       emb_dim,

    # Dataset reconstruction
    "vcf_path":      args.vcf_path,
    "fasta_path":    args.fasta_path,
    "half_window":   args.half_window,
    "buffer":        args.buffer,

    # Shape / identity
    "n_samples":           len(dataset.samples),
    "n_windows":           n_windows,
    "n_unique_windows":    len(dataset.unique_fingerprints),
}

payload = {
    # Top-level keys keep FixedWindowEmbedder.from_file compatibility.
    "cache":               fixed.cache,
    "sample_fp_index":     fixed.sample_fp_index,
    # Extras for reconstruction + provenance.
    "metadata":            metadata,
    "unique_fingerprints": dataset.unique_fingerprints,
    "sample_ids":          list(dataset.samples),
}

tmp = str(out_path) + ".tmp"
torch.save(payload, tmp)
os.replace(tmp, out_path)
print(f"\nSaved cache to {out_path}")
print(f"  cache: {tuple(fixed.cache.shape)}  "
      f"sample_fp_index: {tuple(fixed.sample_fp_index.shape)}")
