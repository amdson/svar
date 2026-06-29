"""
pretrained_eval/eval_plantcad.py
--------------------------------
Evaluate a *pretrained* PlantCAD (PlantCaduceus) checkpoint's masked-LM
log-likelihood loss on the rice dataset, reported both over **entire sequences**
(pseudo-perplexity over scored positions) and over **individual SNPs** (the
alt-allele tokens), as token-weighted mean NLL + perplexity + bits/nt.

Mirrors ``pretrained_eval/eval.py`` (the Carbon path) so the per-window CSV schema
is identical and ``view_losses.ipynb`` reads it unchanged (set ``K=1``). Windows are
the same unique variant windows the embedder / Carbon eval use
(``variant_mlm.build_variant_window_dataset``).

PlantCAD needs its own conda env (mamba-ssm; see PLANTCAD_HANDOFF.md §2) — run this
from that env. ``source env.sh`` first so weights/caches land on scratch.

Examples
--------
    # Smoke test on the first 16 variant windows (hw250 -> 500 bp <= 512 cap)
    python pretrained_eval/eval_plantcad.py --limit 16

    # Per-SNP only (fastest), full 30k windows
    python pretrained_eval/eval_plantcad.py --seq-mode none

    # Full pseudo-perplexity whole-sequence (expensive: ~L forwards/window)
    python pretrained_eval/eval_plantcad.py --seq-mode full --limit 200
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crop_embed.data.coords import FASTA_PATH as LEGACY_FASTA_PATH
from PlantCAD_modules import load_plantcad_mlm
from variant_mlm.data import build_variant_window_dataset, make_loader
from pretrained_eval.baselines import nats_per_token_to_bits_per_nt
from pretrained_eval.plantcad_loss import (
    PLANTCAD_BASES_PER_TOKEN, collect_per_window, plantcad_n_prefix_tokens)
from pretrained_eval.paths import per_window_csv

_REPO_ROOT = Path(__file__).resolve().parents[1]
_VCF_NAME = "sativas413_msu7_final.vcf"
_FASTA_NAME = "Oryza_sativa.IRGSP-1.0.dna_sm.toplevel.fa"


def _scratch_rice_dir() -> Path:
    scratch = os.environ.get("SVAR_SCRATCH", "/90daydata/small_grains/andrew.dickson")
    data_root = os.environ.get("DATA_ROOT", str(Path(scratch) / "datasets"))
    return Path(data_root) / "rice"


def _resolve_data_path(scratch_path: Path, *legacy: str) -> str:
    for cand in (scratch_path, *(Path(p) for p in legacy)):
        if cand.exists():
            return str(cand)
    return str(scratch_path)


_SCRATCH = _scratch_rice_dir()
DEFAULT_VCF_PATH = _resolve_data_path(_SCRATCH / _VCF_NAME, str(_REPO_ROOT / _VCF_NAME))
DEFAULT_FASTA_PATH = _resolve_data_path(_SCRATCH / _FASTA_NAME, LEGACY_FASTA_PATH)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained PlantCAD masked log-likelihood on rice "
                    "(whole-sequence pseudo-PPL + per-SNP).")
    parser.add_argument("--model-path", type=str, default="kuleshov-group/PlantCaduceus_l32",
                        help="HF repo or local dir for the PlantCAD checkpoint.")
    parser.add_argument("--vcf-path", type=str, default=DEFAULT_VCF_PATH)
    parser.add_argument("--fasta-path", type=str, default=DEFAULT_FASTA_PATH)
    parser.add_argument("--half-window", type=int, default=250,
                        help="Half window; full window 2*half_window bp. Default 250 "
                             "keeps the 500 bp window within PlantCAD's 512 bp cap.")
    parser.add_argument("--buffer", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=512,
                        help="Tokenizer truncation length (PlantCAD's hard cap is 512).")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Windows per data batch (per-SNP forward). Whole-seq "
                             "pseudo-PPL is internally re-batched by --sub-batch.")
    parser.add_argument("--sub-batch", type=int, default=128,
                        help="Masked copies per model forward in the pseudo-PPL loop.")
    parser.add_argument("--seq-mode", choices=["subsample", "full", "none"],
                        default="subsample",
                        help="Whole-sequence score: 'full' masks every position "
                             "(~L forwards/window), 'subsample' masks --subsample "
                             "positions/window (default), 'none' skips it (per-SNP only).")
    parser.add_argument("--subsample", type=int, default=128,
                        help="Positions/window scored when --seq-mode subsample.")
    parser.add_argument("--seq-seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N variant windows (smoke test).")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"],
                        default="auto",
                        help="Compute dtype. 'auto' picks bfloat16 on GPUs that "
                             "support it, else float16 (V100/Volta has no bf16 "
                             "hardware — bf16 there is emulated and slow).")
    parser.add_argument("--json-out", type=str, default=None,
                        help="Optional path to write the aggregate metrics as JSON.")
    parser.add_argument("--per-window-out", type=str, default=None,
                        help="Path for the per-window loss CSV (the notebook reads "
                             "this). Default: scratch (pretrained_eval.paths). "
                             "Pass 'none' to skip writing it.")
    args = parser.parse_args()

    if args.half_window * 2 > args.max_length:
        print(f"WARNING: full window {2 * args.half_window} bp exceeds --max-length "
              f"{args.max_length}; windows will be truncated.")

    device = torch.device(
        args.device if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.dtype == "auto":
        bf16_ok = device.type == "cuda" and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if bf16_ok else (
            torch.float16 if device.type == "cuda" else torch.float32)
    else:
        dtype = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}[args.dtype]
    print(f"Device: {device}  Model: {args.model_path}  half_window={args.half_window}  "
          f"dtype={str(dtype).replace('torch.', '')}")
    print(f"Whole-seq mode: {args.seq_mode}"
          + (f" (subsample={args.subsample} positions/window)"
             if args.seq_mode == "subsample" else ""))

    print("Loading PlantCAD …")
    model, tokenizer = load_plantcad_mlm(args.model_path, device=device, dtype=dtype)
    n_prefix = plantcad_n_prefix_tokens(tokenizer)
    print(f"  detected n_prefix_tokens={n_prefix}  mask_token_id={tokenizer.mask_token_id}")

    print("Building variant window dataset …")
    dataset, indices = build_variant_window_dataset(
        args.vcf_path, args.fasta_path, half_window=args.half_window,
        buffer=args.buffer)
    if args.limit is not None:
        indices = indices[: args.limit]
    print(f"  {len(indices):,} variant windows "
          f"(of {len(dataset.unique_fingerprints):,} unique)")

    loader = make_loader(dataset, indices, batch_size=args.batch_size, shuffle=False)
    records = collect_per_window(
        model, tokenizer, loader, max_length=args.max_length, device=device,
        seq_mode=args.seq_mode, subsample=args.subsample, seq_seed=args.seq_seed,
        sub_batch=args.sub_batch)
    df = pd.DataFrame(records)

    def _agg(nll_col: str, tok_col: str) -> dict:
        tok = int(df[tok_col].sum())
        mean = float(df[nll_col].sum()) / tok if tok else float("nan")
        return {"mean_nll": mean,
                "perplexity": math.exp(mean) if tok else float("nan"),
                "bits_per_nt": nats_per_token_to_bits_per_nt(mean, PLANTCAD_BASES_PER_TOKEN)
                               if tok else float("nan"),
                "n_tokens": tok}

    seq, snp = _agg("seq_nll_sum", "seq_tokens"), _agg("snp_nll_sum", "snp_tokens")
    print("\n── Pretrained PlantCAD masked loss on rice ──")
    print(f"  windows           : {len(df):,}")
    print(f"  whole-sequence    : mean NLL {seq['mean_nll']:.4f}  "
          f"ppl {seq['perplexity']:.4f}  {seq['bits_per_nt']:.4f} bits/nt  "
          f"({seq['n_tokens']:,} tokens)")
    print(f"  per-SNP           : mean NLL {snp['mean_nll']:.4f}  "
          f"ppl {snp['perplexity']:.4f}  {snp['bits_per_nt']:.4f} bits/nt  "
          f"({snp['n_tokens']:,} tokens)")

    if args.per_window_out != "none":
        csv_path = (Path(args.per_window_out) if args.per_window_out
                    else per_window_csv(args.model_path, args.half_window))
        df.to_csv(csv_path, index=False)
        print(f"\n  per-window CSV : {csv_path}  ({len(df):,} rows)")

    if args.json_out:
        out = {"sequence": seq, "snp": snp, "n_windows": len(df),
               "model_path": args.model_path, "half_window": args.half_window,
               "seq_mode": args.seq_mode, "subsample": args.subsample,
               "n_windows_evaluated": len(indices)}
        Path(args.json_out).write_text(json.dumps(out, indent=2))
        print(f"  metrics JSON   : {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
