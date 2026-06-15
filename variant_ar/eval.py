"""
variant_ar/eval.py
------------------
Evaluate Carbon's autoregressive SNP-token log-likelihood loss over every unique
variant window in a VCF + reference FASTA. Reports token-weighted mean NLL and
perplexity. Purely a testing harness for the variant-cache work.

Reuses the same windowing as the embedder / variant-MLM eval
(`variant_mlm.build_variant_window_dataset`), so the windows are identical.

Example
-------
    python variant_ar/eval.py --half-window 500 --batch-size 8 --limit 50
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crop_embed.data.coords import FASTA_PATH
from CARBON_modules import load_carbon_local
from variant_mlm.data import build_variant_window_dataset, make_loader
from variant_ar.loss import evaluate

DEFAULT_VCF_PATH = str(Path(__file__).resolve().parents[1] / "sativas413_msu7_final.vcf")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Carbon autoregressive SNP-token NLL.")
    parser.add_argument("--model-path", type=str, default="HuggingFaceBio/Carbon-500M",
                        help="HF repo or local dir for the Carbon checkpoint.")
    parser.add_argument("--vcf-path", type=str, default=DEFAULT_VCF_PATH)
    parser.add_argument("--fasta-path", type=str, default=FASTA_PATH)
    parser.add_argument("--half-window", type=int, default=500,
                        help="Half window size; full window is 2*half_window bp.")
    parser.add_argument("--buffer", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N variant windows (smoke test).")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    print(f"Device: {device}  Model: {args.model_path}  half_window={args.half_window}")

    print("Loading Carbon …")
    model, tokenizer = load_carbon_local(args.model_path, device=device, dtype=dtype)

    print("Building variant window dataset …")
    dataset, indices = build_variant_window_dataset(
        args.vcf_path, args.fasta_path, half_window=args.half_window,
        buffer=args.buffer)
    if args.limit is not None:
        indices = indices[: args.limit]
    print(f"  {len(indices):,} variant windows "
          f"(of {len(dataset.unique_fingerprints):,} unique)")

    loader = make_loader(dataset, indices, batch_size=args.batch_size, shuffle=False)
    stats = evaluate(model, tokenizer, loader, max_length=args.max_length,
                     device=device)

    print("\n── Carbon autoregressive SNP-token loss ──")
    print(f"  windows     : {stats['n_windows']:,}")
    print(f"  SNP tokens  : {stats['n_tokens']:,}")
    print(f"  mean NLL    : {stats['mean_nll']:.4f}")
    print(f"  perplexity  : {stats['perplexity']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
