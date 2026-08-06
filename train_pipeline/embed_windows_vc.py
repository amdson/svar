"""
embed_windows_vc.py
-------------------
Variant-cache analogue of ``embed_windows.py`` for the Carbon backbone. Produces
the *same* on-disk cache format (drop-in for FixedWindowEmbedder.from_file /
demo_train.py --cache), but computes the per-window embeddings with the variant
cache instead of a full Carbon forward per unique window.

Why
---
``embed_windows.py`` runs one full Carbon forward for every *unique fingerprint*
``(chrom, w_start, w_end, alt_positions)``. Many fingerprints share the same
reference window and differ only in which SNPs carry the alt allele. The variant
cache (CARBON_modules/variant_cache_layers.py) exploits exactly this: freeze the
reference window once, then recompute only the SNP token positions for every
variant haplotype against that frozen reference.

So this script groups fingerprints by their window ``(chrom, w_start, w_end)``
and, per group, does

  * one full reference forward  (load_carbon_local)            -> ref_h  (T, H)
  * one *batched* variant-cache forward (load_carbon_variant_cache) that
    recomputes the SNP token positions for all N haplotypes at once -> cache_var

instead of N independent full forwards. The pooled window embedding then follows
``model_dev/compare_variant_cache_embeddings.py``: take the reference hidden
states, overwrite the recomputed SNP positions, and pool over the real tokens.

The shared-positions trick
--------------------------
``VariantCacheCarbonEncoder.forward`` takes a *single* ``variant_cache_indices``
vector shared across the whole batch, but fingerprints in a group carry different
SNP sets. We feed the **union** of all SNP token positions in the group. A
fingerprint that has no SNP at a given union position simply keeps the reference
token there, which the cache reproduces exactly (the "identity probe" property
validated in model_dev/test_carbon_variant_cache.py) — so the extra positions are
no-ops for that haplotype.

Output (identical to embed_windows.py)
--------------------------------------
A torch-saved dict with ``cache``, ``sample_fp_index``, ``metadata``,
``unique_fingerprints`` and ``sample_ids`` (see embed_windows.py for the schema).

Example
-------
    python train_pipeline/embed_windows_vc.py \\
        --half-window 500 --max-length 2048 --vc-batch-size 16 \\
        --output checkpoints/sativas413_carbon_vc.ckpt.pt
"""

import argparse
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crop_embed import FixedWindowEmbedder, SNPWindowPartitioner, UniqueWindowDataset
from crop_embed.data.coords import FASTA_PATH
from crop_embed.data.vcf import load_snps_from_vcf
from crop_embed.embedder import (
    _CARBON_DNA_PREFIX,
    _CARBON_KMER,
    _CARBON_N_PREFIX_TOKENS,
    _build_snp_mask_kmer,
)
from CARBON_modules import load_carbon_local, load_carbon_variant_cache

# DEFAULT_VCF_PATH = "/home/andrew.dickson/rice_data/sativas413_msu7_final.vcf"
DEFAULT_VCF_PATH = "/home/andrew/rice_data/sativas413_msu7_final.vcf"


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Precompute a Carbon window-embedding cache via the variant cache."
    )
    # Model
    p.add_argument("--model-path", type=str, default="HuggingFaceBio/Carbon-500M",
                   help="HuggingFace repo or local dir for the Carbon checkpoint.")
    p.add_argument("--max-length", type=int, default=2048,
                   help="Tokenizer truncation length.")
    p.add_argument("--snp-only", action="store_true",
                   help="Pool only over SNP tokens (full-window pool on pure-"
                        "reference windows), matching WindowEmbedder(snp_only=True).")
    p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32",
                   help="Compute dtype. fp32 (default) matches the variant-cache "
                        "numerics validation; bf16 is faster/lighter.")

    # Data / windowing
    p.add_argument("--vcf-path", type=str, default=DEFAULT_VCF_PATH)
    p.add_argument("--fasta-path", type=str, default=str(FASTA_PATH))
    p.add_argument("--half-window", type=int, default=500)
    p.add_argument("--buffer", type=int, default=0)
    p.add_argument("--loader", choices=["pysam", "polars"], default="pysam",
                   help="VCF reader. 'polars' is ~2 orders of magnitude faster on "
                        "large plain biallelic-GT VCFs (arabidopsis: ~18s vs ~45min); "
                        "byte-identical SNPRecords. 'pysam' (default) for any VCF.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N window groups (smoke test / dev).")
    p.add_argument("--write-mode", choices=["tensor", "dict"], default="tensor",
                   help="'tensor' (default): scatter embeddings into a preallocated "
                        "(n_unique, emb_dim) fp16 table by fingerprint→row — RAM capped "
                        "at the cache size, single contiguous checkpoints (required for "
                        "large runs like arabidopsis, ~7.6M windows). 'dict': the legacy "
                        "{fingerprint: tensor} accumulator (fine for small caches).")

    # Compute
    p.add_argument("--vc-batch-size", type=int, default=16,
                   help="Haplotypes per variant-cache forward within a window "
                        "group (bounds the (N, heads, cs, ref) attention memory).")
    p.add_argument("--device", type=str, default=None,
                   help="torch device (default: cuda if available, else cpu).")

    # Validation
    p.add_argument("--sanity-windows", type=int, default=0,
                   help="If >0, for the first N variant windows also run the exact "
                        "full forward of each alt haplotype and report cosine "
                        "similarity vs the variant-cache embedding (correctness check).")

    # I/O
    p.add_argument("--output", type=str, required=True,
                   help="Path to write the cache .pt file.")
    p.add_argument("--checkpoint-every", type=int, default=500,
                   help="Save a resume-able embedding-table checkpoint every N groups.")
    p.add_argument("--checkpoint-path", type=str, default=None,
                   help="Path for the intermediate {Fingerprint: Tensor} checkpoint. "
                        "Defaults to <output>.embtable.ckpt.pt next to --output.")
    return p


# ── Pooling helpers (mirror crop_embed.embedder.WindowEmbedder) ────────────────

def _pool_full(hidden: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Mean over the real (attended) tokens. hidden (T, H); keep (T,) bool."""
    m = keep.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(0) / m.sum(0).clamp(min=1)


def _tokenize(tokenizer, seqs, max_length, device):
    """Tokenize Carbon DNA sequences (with the <dna> marker) -> (ids, attn) on device."""
    enc = tokenizer(
        [f"{_CARBON_DNA_PREFIX}{s}" for s in seqs],
        return_tensors="pt", padding="longest", truncation=True,
        max_length=max_length, add_special_tokens=False,
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


# ── Per-group embedding ────────────────────────────────────────────────────────

def embed_window_group(
    fps,
    dataset,
    exact_model,
    cache_model,
    tokenizer,
    max_length,
    snp_only,
    vc_batch_size,
    device,
):
    """Embed every fingerprint in one window group (shared (chrom, w_start, w_end)).

    Returns {fingerprint: Tensor(H,)} on CPU.
    """
    # Split off the pure-reference haplotype (alt_positions empty); the rest are
    # variant haplotypes recomputed through the cache.
    ref_fp = None
    var_fps = []
    for fp in fps:
        if fp[3]:
            var_fps.append(fp)
        else:
            ref_fp = fp

    chrom, w_start, w_end, _ = fps[0]

    # One full reference forward -> post-norm hidden states (T, H). Shared by the
    # pure-reference haplotype and the non-SNP positions of every variant.
    ref_seq = dataset.extract_sequence((chrom, w_start, w_end, ()))
    seqs = [ref_seq] + [dataset.extract_sequence(fp) for fp in var_fps]
    ids, attn = _tokenize(tokenizer, seqs, max_length, device)
    ref_ids = ids[0]                      # (T,)
    keep = attn[0].bool()                 # (T,) real-token mask (same for all, equal length)
    T = ref_ids.shape[0]

    ref_h = exact_model(
        input_ids=ref_ids.unsqueeze(0), attention_mask=attn[0].unsqueeze(0),
        output_hidden_states=True, logits_to_keep=1,
    ).hidden_states[-1][0].float()       # (T, H)

    out = {}
    if ref_fp is not None:
        out[ref_fp] = _pool_full(ref_h, keep).cpu()

    if not var_fps:
        return out

    # Per-fingerprint SNP token masks and their union (the shared recompute set).
    perfp_mask = _build_snp_mask_kmer(
        (len(var_fps), T), var_fps, k=_CARBON_KMER,
        n_prefix_tokens=_CARBON_N_PREFIX_TOKENS,
    ).to(device) & keep.unsqueeze(0)     # (N, T)
    union = torch.nonzero(perfp_mask.any(0), as_tuple=False).flatten()  # (cs,)

    if union.numel() == 0:
        # SNPs all fell outside the (possibly truncated) window — no recompute
        # possible; fall back to the reference embedding for every haplotype.
        for fp in var_fps:
            out[fp] = _pool_full(ref_h, keep).cpu()
        return out

    alt_ids = ids[1:]                          # (N, T)
    variant_input_ids = alt_ids[:, union]      # (N, cs) alt token ids at union positions

    for lo in range(0, len(var_fps), vc_batch_size):
        hi = min(lo + vc_batch_size, len(var_fps))
        cache_var = cache_model(
            ref_ids, variant_positions=union, attention_mask=attn[0],
            variant_input_ids=variant_input_ids[lo:hi], output_logits=False,
        ).last_hidden_state.float()            # (chunk, cs, H)

        for b, fp in enumerate(var_fps[lo:hi]):
            if snp_only:
                # Pool over this haplotype's own SNP tokens (a subset of `union`).
                own = perfp_mask[lo + b, union]            # (cs,) bool
                if own.any():
                    out[fp] = cache_var[b][own].mean(0).cpu()
                    continue
                # No SNP tokens survived truncation -> full-window fallback.
            cache_h = ref_h.clone()
            cache_h[union] = cache_var[b]
            out[fp] = _pool_full(cache_h, keep).cpu()

    return out


# ── Optional correctness check ─────────────────────────────────────────────────

def sanity_check(var_fps, emb_lookup, dataset, exact_model, tokenizer, max_length,
                 snp_only, device, n):
    """Compare variant-cache embeddings to the exact full-forward embeddings.
    ``emb_lookup(fp) -> Tensor`` fetches the stored embedding (dict.__getitem__ or
    a row lookup into the preallocated tensor)."""
    cos = []
    for fp in var_fps[:n]:
        chrom, w_start, w_end, _ = fp
        ids, attn = _tokenize(tokenizer, [dataset.extract_sequence(fp)],
                              max_length, device)
        keep = attn[0].bool()
        h = exact_model(input_ids=ids, attention_mask=attn,
                        output_hidden_states=True, logits_to_keep=1
                        ).hidden_states[-1][0].float()
        if snp_only:
            m = _build_snp_mask_kmer((1, ids.shape[1]), [fp], k=_CARBON_KMER,
                                     n_prefix_tokens=_CARBON_N_PREFIX_TOKENS
                                     ).to(device)[0] & keep
            exact_emb = h[m].mean(0) if m.any() else _pool_full(h, keep)
        else:
            exact_emb = _pool_full(h, keep)
        vc_emb = emb_lookup(fp).to(device)
        cos.append(F.cosine_similarity(vc_emb, exact_emb, dim=0).item())
    if cos:
        import statistics as st
        print(f"  [sanity] {len(cos)} windows  cosine(vc, exact): "
              f"mean={st.mean(cos):.6f}  min={min(cos):.6f}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    args = build_parser().parse_args()

    device = torch.device(
        args.device if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    print(f"Device: {device}  dtype: {dtype}  model: {args.model_path}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.checkpoint_path or str(out_path) + ".embtable.ckpt.pt"

    # ── Dataset ────────────────────────────────────────────────────────────────
    print(f"Loading SNPs from VCF (loader={args.loader}) …")
    if args.loader == "polars":
        from crop_embed.data.vcf_polars import load_snps_from_vcf as _load
    else:
        _load = load_snps_from_vcf
    snps_by_chrom, samples = _load(args.vcf_path)
    n_snps = sum(len(v) for v in snps_by_chrom.values())
    print(f"  {n_snps:,} SNPs | {len(snps_by_chrom)} chromosomes | {len(samples)} samples")

    print("Building SNPWindowPartitioner …")
    partitioner = SNPWindowPartitioner(
        snps_by_chrom, half_window=args.half_window, buffer=args.buffer)
    print(f"  {len(partitioner):,} windows")

    print("Building UniqueWindowDataset …")
    dataset = UniqueWindowDataset(args.vcf_path, args.fasta_path, partitioner,
                                  engine=args.loader)
    print(f"  {len(dataset):,} unique fingerprints to embed "
          f"(vs {len(partitioner) * len(dataset.samples):,} sample×window pairs)")

    # Group unique fingerprints by their reference window.
    groups: "OrderedDict[tuple, list]" = OrderedDict()
    for fp in dataset.unique_fingerprints:
        groups.setdefault(fp[:3], []).append(fp)
    print(f"  {len(groups):,} window groups "
          f"(mean {len(dataset) / max(1, len(groups)):.1f} haplotypes/window)")
    if args.limit is not None:
        groups = OrderedDict(list(groups.items())[: args.limit])
        print(f"  --limit: processing first {len(groups):,} window groups only")

    # ── Models ─────────────────────────────────────────────────────────────────
    print(f"\nLoading Carbon (exact + variant cache) from {args.model_path} …")
    exact_model, tokenizer = load_carbon_local(args.model_path, device=device, dtype=dtype)
    cache_model, _ = load_carbon_variant_cache(
        args.model_path, device=device, dtype=dtype, backend="efficient")
    exact_model.eval(); cache_model.eval()

    emb_dim = int(exact_model.config.hidden_size)
    n_uniq = len(dataset.unique_fingerprints)
    total_haplos = sum(len(v) for v in groups.values())

    # ── Embed (tensor mode: preallocated fp16 table; dict mode: legacy) ─────────
    sanity_pool = []  # variant fps gathered for the optional sanity check

    def _run_loop(process_group, is_done, save_ckpt, ckpt_desc):
        """Shared driver: iterate groups with a rate/ETA progress bar, checkpoint
        every N groups. process_group(fps) embeds+stores; is_done(fps)->bool skips
        already-embedded groups (resume)."""
        t0 = time.time()
        groups_since_save = done_haplos = 0
        pbar = tqdm(groups.items(), total=len(groups), desc="Embedding windows",
                    unit="win", smoothing=0.05)
        with torch.no_grad():
            for _key, fps in pbar:
                if not is_done(fps):
                    process_group(fps)
                    if args.sanity_windows and len(sanity_pool) < args.sanity_windows:
                        sanity_pool.extend(fp for fp in fps if fp[3])
                    groups_since_save += 1
                    if groups_since_save >= args.checkpoint_every:
                        save_ckpt(); groups_since_save = 0
                done_haplos += len(fps)
                dt = time.time() - t0
                pbar.set_postfix_str(
                    f"{done_haplos:,}/{total_haplos:,} haplos, "
                    f"{done_haplos/max(dt,1e-9):,.0f} haplo/s")
        save_ckpt()
        return time.time() - t0

    if args.write_mode == "tensor":
        fp_to_row = {fp: i for i, fp in enumerate(dataset.unique_fingerprints)}
        if os.path.exists(ckpt_path):
            saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            table, done = saved["table"], saved["done"]
            print(f"Resumed: {int(done.sum()):,}/{n_uniq:,} fingerprints already embedded")
        else:
            table = torch.empty((n_uniq, emb_dim), dtype=torch.float16)  # ~cache size, capped
            done = torch.zeros(n_uniq, dtype=torch.bool)

        def _save_checkpoint():
            tmp = ckpt_path + ".tmp"
            torch.save({"table": table, "done": done}, tmp)
            os.replace(tmp, ckpt_path)

        def _process(fps):
            out = embed_window_group(fps, dataset, exact_model, cache_model, tokenizer,
                                     args.max_length, args.snp_only, args.vc_batch_size, device)
            for fp, vec in out.items():
                r = fp_to_row[fp]
                table[r] = vec.to(torch.float16)
                done[r] = True

        def _is_done(fps):
            return bool(done[[fp_to_row[fp] for fp in fps]].all())

        print(f"\nEmbedding {len(groups):,} groups ({total_haplos:,} haplotypes) "
              f"[tensor/fp16, ~{n_uniq*emb_dim*2/1e9:.1f}GB] → {ckpt_path}")
        elapsed = _run_loop(_process, _is_done, _save_checkpoint, ckpt_path)
        print(f"  {int(done.sum()):,} fingerprints embedded in {elapsed:.1f}s")

        if args.sanity_windows and sanity_pool:
            print("\nRunning sanity check (variant cache vs exact full forward) …")
            with torch.no_grad():
                sanity_check(sanity_pool, lambda fp: table[fp_to_row[fp]].float(),
                             dataset, exact_model, tokenizer, args.max_length,
                             args.snp_only, device, args.sanity_windows)

        missing = int((~done).sum())
        if missing:
            print(f"\n{missing:,}/{n_uniq:,} fingerprints not embedded (partial/--limit "
                  f"run) — skipping final cache write. Resume checkpoint at {ckpt_path}; "
                  f"re-run without --limit to finish.")
            return 0
        cache_tensor = table
        sfi_tensor = dataset.sample_fp_index.clone()
    else:
        embedding_table = {}
        if os.path.exists(ckpt_path):
            embedding_table = torch.load(ckpt_path, weights_only=False)
            print(f"Resumed from checkpoint: {len(embedding_table):,} already embedded")

        def _save_checkpoint():
            tmp = ckpt_path + ".tmp"
            torch.save(embedding_table, tmp)
            os.replace(tmp, ckpt_path)

        def _process(fps):
            embedding_table.update(embed_window_group(
                fps, dataset, exact_model, cache_model, tokenizer,
                args.max_length, args.snp_only, args.vc_batch_size, device))

        print(f"\nEmbedding {len(groups):,} window groups [dict] → {ckpt_path}")
        elapsed = _run_loop(_process, lambda fps: all(fp in embedding_table for fp in fps),
                            _save_checkpoint, ckpt_path)
        print(f"  {len(embedding_table):,} fingerprints embedded in {elapsed:.1f}s")

        if args.sanity_windows and sanity_pool:
            print("\nRunning sanity check (variant cache vs exact full forward) …")
            with torch.no_grad():
                sanity_check(sanity_pool, embedding_table.__getitem__, dataset,
                             exact_model, tokenizer, args.max_length, args.snp_only,
                             device, args.sanity_windows)

        missing = sum(fp not in embedding_table for fp in dataset.unique_fingerprints)
        if missing:
            print(f"\n{missing:,}/{n_uniq:,} fingerprints not embedded (partial/--limit "
                  f"run) — skipping final cache write. Resume checkpoint at {ckpt_path}; "
                  f"re-run without --limit to finish.")
            return 0
        fixed = FixedWindowEmbedder.from_embedding_table(embedding_table, dataset)
        cache_tensor = fixed.cache
        sfi_tensor = fixed.sample_fp_index

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "train_pipeline/embed_windows_vc.py",
        "method":       "variant_cache",
        "elapsed_sec":  elapsed,
        "device":       str(device),

        "backend":      "carbon",
        "model_path":   args.model_path,
        "max_length":   args.max_length,
        "snp_only":     args.snp_only,
        "precision":    args.precision,
        "write_mode":   args.write_mode,
        "storage_dtype": str(cache_tensor.dtype),
        "emb_dim":      emb_dim,

        "vcf_path":     args.vcf_path,
        "fasta_path":   args.fasta_path,
        "half_window":  args.half_window,
        "buffer":       args.buffer,

        "n_samples":        len(dataset.samples),
        "n_windows":        len(partitioner),
        "n_unique_windows": len(dataset.unique_fingerprints),
    }

    payload = {
        "cache":               cache_tensor,
        "sample_fp_index":     sfi_tensor,
        "metadata":            metadata,
        "unique_fingerprints": dataset.unique_fingerprints,
        "sample_ids":          list(dataset.samples),
    }

    tmp = str(out_path) + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, out_path)
    print(f"\nSaved cache to {out_path}")
    print(f"  cache: {tuple(cache_tensor.shape)} {cache_tensor.dtype}  "
          f"sample_fp_index: {tuple(sfi_tensor.shape)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
