"""
pretrained_eval/eval.py
-----------------------
Evaluate a *pretrained* Carbon checkpoint's autoregressive log-likelihood loss on
the rice dataset, reported both over **entire sequences** (every DNA token in a
window) and over **individual SNPs** (only the 6-mer tokens that carry an alt
allele). Reports token-weighted mean NLL and perplexity for each.

Windows are the same unique variant windows the embedder / variant-MLM eval use
(`variant_mlm.build_variant_window_dataset`), so the corpus is exactly the rice
VCF haplotype windows. One Carbon forward pass per batch produces both metrics.

Examples
--------
    # Smoke test on the first 50 variant windows
    python pretrained_eval/eval.py --limit 50

    # Full eval with the 3B checkpoint
    python pretrained_eval/eval.py --model-path HuggingFaceBio/Carbon-3B \
        --half-window 500 --batch-size 8
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
from CARBON_modules import load_carbon_lm
from variant_mlm.data import build_variant_window_dataset, make_loader
from pretrained_eval.loss import collect_per_window
from pretrained_eval.paths import per_window_csv

_REPO_ROOT = Path(__file__).resolve().parents[1]
_VCF_NAME = "sativas413_msu7_final.vcf"
_FASTA_NAME = "Oryza_sativa.IRGSP-1.0.dna_sm.toplevel.fa"

# Convenience aliases so `--model-path 3b` resolves to the published repo. Any
# other value is passed straight through (HF repo id or local dir).
_MODEL_ALIASES = {
    "500m": "HuggingFaceBio/Carbon-500M",
    "3b":   "HuggingFaceBio/Carbon-3B",
    "8b":   "HuggingFaceBio/Carbon-8B",
}


def _resolve_model_path(model_path: str) -> str:
    return _MODEL_ALIASES.get(model_path.lower(), model_path)


def _scratch_rice_dir() -> Path:
    """Migrated rice data lives under cluster scratch (see env.sh / datasets/common.mk):
    ``$DATA_ROOT/rice``, with ``DATA_ROOT`` defaulting to ``$SVAR_SCRATCH/datasets``."""
    scratch = os.environ.get("SVAR_SCRATCH", "/90daydata/small_grains/andrew.dickson")
    data_root = os.environ.get("DATA_ROOT", str(Path(scratch) / "datasets"))
    return Path(data_root) / "rice"


def _resolve_data_path(scratch_path: Path, *legacy: str) -> str:
    """Prefer the migrated scratch copy; fall back to legacy (repo / ~/rice_data)
    locations so the eval still runs before `make -C datasets/rice` populates scratch."""
    for cand in (scratch_path, *(Path(p) for p in legacy)):
        if cand.exists():
            return str(cand)
    return str(scratch_path)  # report the canonical location if nothing exists yet


# Scratch-first so the eval honours the 90day-storage migration; the legacy
# in-repo VCF and ~/rice_data FASTA are fallbacks until the rebuild lands.
_SCRATCH = _scratch_rice_dir()
DEFAULT_VCF_PATH = _resolve_data_path(_SCRATCH / _VCF_NAME, str(_REPO_ROOT / _VCF_NAME))
DEFAULT_FASTA_PATH = _resolve_data_path(_SCRATCH / _FASTA_NAME, LEGACY_FASTA_PATH)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained Carbon AR log-likelihood on rice "
                    "(whole-sequence + per-SNP).")
    parser.add_argument("--model-path", type=str, default="HuggingFaceBio/Carbon-500M",
                        help="HF repo or local dir for the Carbon checkpoint, or an "
                             "alias: '500m', '3b', '8b'. The 3B/8B checkpoints are "
                             "sharded (handled by the loader) and need more GPU "
                             "memory — drop --batch-size to ~4 for 3B.")
    parser.add_argument("--vcf-path", type=str, default=DEFAULT_VCF_PATH)
    parser.add_argument("--fasta-path", type=str, default=DEFAULT_FASTA_PATH)
    parser.add_argument("--half-window", type=int, default=500,
                        help="Half window size; full window is 2*half_window bp.")
    parser.add_argument("--buffer", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N variant windows (smoke test).")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16",
                        help="Compute dtype (Carbon's published default is bfloat16).")
    parser.add_argument("--json-out", type=str, default=None,
                        help="Optional path to write the aggregate metrics as JSON.")
    parser.add_argument("--per-window-out", type=str, default=None,
                        help="Path for the per-window loss CSV (the notebook reads "
                             "this). Default: scratch (pretrained_eval.paths). "
                             "Pass 'none' to skip writing it.")
    args = parser.parse_args()
    args.model_path = _resolve_model_path(args.model_path)

    device = torch.device(
        args.device if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    print(f"Device: {device}  Model: {args.model_path}  half_window={args.half_window}")

    print("Loading Carbon …")
    model, tokenizer = load_carbon_lm(args.model_path, device=device, dtype=dtype)

    print("Building variant window dataset …")
    dataset, indices = build_variant_window_dataset(
        args.vcf_path, args.fasta_path, half_window=args.half_window,
        buffer=args.buffer)
    if args.limit is not None:
        indices = indices[: args.limit]
    print(f"  {len(indices):,} variant windows "
          f"(of {len(dataset.unique_fingerprints):,} unique)")

    loader = make_loader(dataset, indices, batch_size=args.batch_size, shuffle=False)
    records = collect_per_window(model, tokenizer, loader,
                                 max_length=args.max_length, device=device)
    df = pd.DataFrame(records)

    # Token-weighted aggregate (identical to loss.evaluate, derived from the
    # per-window sums so the CSV and the printed summary always agree).
    def _agg(nll_col: str, tok_col: str) -> dict:
        tok = int(df[tok_col].sum())
        mean = float(df[nll_col].sum()) / tok if tok else float("nan")
        return {"mean_nll": mean,
                "perplexity": math.exp(mean) if tok else float("nan"),
                "n_tokens": tok}

    seq, snp = _agg("seq_nll_sum", "seq_tokens"), _agg("snp_nll_sum", "snp_tokens")
    print("\n── Pretrained Carbon autoregressive loss on rice ──")
    print(f"  windows           : {len(df):,}")
    print(f"  whole-sequence    : mean NLL {seq['mean_nll']:.4f}  "
          f"ppl {seq['perplexity']:.4f}  ({seq['n_tokens']:,} tokens)")
    print(f"  per-SNP           : mean NLL {snp['mean_nll']:.4f}  "
          f"ppl {snp['perplexity']:.4f}  ({snp['n_tokens']:,} tokens)")

    if args.per_window_out != "none":
        csv_path = (Path(args.per_window_out) if args.per_window_out
                    else per_window_csv(args.model_path, args.half_window))
        df.to_csv(csv_path, index=False)
        print(f"\n  per-window CSV : {csv_path}  ({len(df):,} rows)")

    if args.json_out:
        out = {"sequence": seq, "snp": snp, "n_windows": len(df),
               "model_path": args.model_path, "half_window": args.half_window,
               "n_windows_evaluated": len(indices)}
        Path(args.json_out).write_text(json.dumps(out, indent=2))
        print(f"  metrics JSON   : {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
