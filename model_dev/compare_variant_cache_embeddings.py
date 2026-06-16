"""
model_dev/compare_variant_cache_embeddings.py
---------------------------------------------
How do variant-cache embeddings diverge from the true (full-forward) embeddings
as a window accumulates more concurrent SNPs? No training — pure measurement.

For each variant window we compare three embeddings of the alt haplotype:

  true   : full Carbon forward of the alt sequence (ground truth).
  cache  : frozen reference-genome forward with the alt tokens recomputed at the
           SNP positions via the variant cache (the real product).
  ref    : the reference-genome forward with NO variant substitution (baseline —
           what you'd get if you ignored the variants entirely).

Two views:
  * per-variant-token : cosine(cache, true) at the recomputed SNP positions.
  * window-pooled      : cosine of the mean-pooled window vector. The cache pool
                         uses frozen reference hidden states at the non-variant
                         positions (which the true forward shifts but the cache
                         does not) + the recomputed SNP positions.

Results are binned by the number of variant tokens in the window, so you can see
the error "add up" with concurrent SNPs (1 SNP is exact; more drift more).

    python -m model_dev.compare_variant_cache_embeddings --half-window 250 --limit 300
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crop_embed.data.coords import FASTA_PATH
from crop_embed.embedder import _build_snp_mask_kmer
from CARBON_modules import load_carbon_local, load_carbon_variant_cache
from variant_mlm.data import build_variant_window_dataset
from variant_ar.train import make_ar_loader

CARBON_DNA_PREFIX = "<dna>"
CARBON_KMER = 6
CARBON_N_PREFIX_TOKENS = 1


def _cos(a, b):
    return F.cosine_similarity(a.reshape(-1), b.reshape(-1), dim=0).item()


def _relerr(approx, true):
    return (torch.linalg.norm(approx.reshape(-1) - true.reshape(-1)) /
            torch.linalg.norm(true.reshape(-1)).clamp_min(1e-12)).item()


def main() -> int:
    p = argparse.ArgumentParser(description="Compare variant-cache vs true embeddings.")
    p.add_argument("--model-path", type=str, default="HuggingFaceBio/Carbon-500M")
    p.add_argument("--vcf-path", type=str,
                   default=str(Path(__file__).resolve().parents[1] / "sativas413_msu7_final.vcf"))
    p.add_argument("--fasta-path", type=str, default=str(FASTA_PATH))
    p.add_argument("--half-window", type=int, default=250)
    p.add_argument("--buffer", type=int, default=0)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--limit", type=int, default=300,
                   help="Number of variant windows to scan.")
    p.add_argument("--min-snps", type=int, default=1,
                   help="Only evaluate windows with at least this many SNP tokens. "
                        "Counted from tokenization alone, so windows below the "
                        "threshold are skipped before any (expensive) model forward.")
    p.add_argument("--max-eval", type=int, default=None,
                   help="Stop after evaluating this many qualifying windows.")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--csv-out", type=str, default=None,
                   help="If set, write the per-window records (one row per window: "
                        "n_snp, var_cos, cache_pool_cos, ref_pool_cos, cache_relerr, "
                        "ref_relerr) to this CSV for plotting.")
    args = p.parse_args()

    device = torch.device(
        args.device if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float32
    print(f"Device: {device}  half_window={args.half_window}  limit={args.limit}")

    exact, tokenizer = load_carbon_local(args.model_path, device=device, dtype=dtype)
    cache, _ = load_carbon_variant_cache(args.model_path, device=device, dtype=dtype,
                                         backend="efficient")

    dataset, indices = build_variant_window_dataset(
        args.vcf_path, args.fasta_path, half_window=args.half_window, buffer=args.buffer)
    indices = indices[: args.limit]
    loader = make_ar_loader(dataset, indices, batch_size=16, shuffle=False)
    print(f"Scanning {len(indices):,} variant windows "
          f"(min_snps={args.min_snps})…")

    records = []  # (n_snp_tokens, var_cos, cache_pool_cos, ref_pool_cos,
                  #  cache_pool_relerr, ref_pool_relerr)
    exact.eval(); cache.eval()
    with torch.no_grad():
        from tqdm import tqdm
        for batch in tqdm(loader):
            for rs, asq, fp in zip(batch["ref_sequences"], batch["alt_sequences"],
                                   batch["fingerprints"]):
                enc = tokenizer(
                    [f"{CARBON_DNA_PREFIX}{rs}", f"{CARBON_DNA_PREFIX}{asq}"],
                    return_tensors="pt", padding="longest", truncation=True,
                    max_length=args.max_length, add_special_tokens=False)
                ids = enc["input_ids"].to(device)
                am0 = enc["attention_mask"][0].to(device)
                ref_ids, alt_ids = ids[0], ids[1]
                T = alt_ids.shape[0]

                snp = _build_snp_mask_kmer((1, T), [fp], k=CARBON_KMER,
                                           n_prefix_tokens=CARBON_N_PREFIX_TOKENS).to(device)[0]
                pos = torch.nonzero(snp & am0.bool(), as_tuple=False).flatten()
                # SNP count is known from tokenization alone — filter cheaply,
                # before spending any model forwards on this window.
                if pos.numel() < max(1, args.min_snps):
                    continue

                # True (alt) and reference full forwards; logits_to_keep=1 skips
                # the big vocab projection we don't need.
                alt_h = exact(input_ids=alt_ids.unsqueeze(0),
                              attention_mask=am0.unsqueeze(0),
                              output_hidden_states=True, logits_to_keep=1
                              ).hidden_states[-1][0]              # (T, H)
                ref_h = exact(input_ids=ref_ids.unsqueeze(0),
                              attention_mask=am0.unsqueeze(0),
                              output_hidden_states=True, logits_to_keep=1
                              ).hidden_states[-1][0]              # (T, H)

                # Cache: recompute the SNP positions against the frozen reference.
                cache_var = cache(ref_ids, variant_positions=pos, attention_mask=am0,
                                  variant_input_ids=alt_ids.index_select(0, pos)
                                  ).last_hidden_state[0]          # (cs, H)

                true_var = alt_h.index_select(0, pos)             # (cs, H)
                var_cos = F.cosine_similarity(cache_var, true_var, dim=1).mean().item()

                # Window-pooled embeddings over real tokens.
                m = am0.bool()
                true_pool = alt_h[m].mean(0)
                ref_pool = ref_h[m].mean(0)
                cache_h = ref_h.clone()
                cache_h[pos] = cache_var
                cache_pool = cache_h[m].mean(0)

                records.append((
                    int(pos.numel()),
                    var_cos,
                    _cos(cache_pool, true_pool),
                    _cos(ref_pool, true_pool),
                    _relerr(cache_pool, true_pool),
                    _relerr(ref_pool, true_pool),
                ))
                if args.max_eval is not None and len(records) >= args.max_eval:
                    break
            if args.max_eval is not None and len(records) >= args.max_eval:
                break

    if not records:
        print("No windows with variant tokens found.")
        return 1

    if args.csv_out:
        import csv
        with open(args.csv_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["n_snp", "var_cos", "cache_pool_cos", "ref_pool_cos",
                        "cache_relerr", "ref_relerr"])
            w.writerows(records)
        print(f"Wrote {len(records):,} records to {args.csv_out}")

    _report(records)
    return 0


def _report(records):
    import statistics as st

    def col(rows, j):
        return [r[j] for r in rows]

    by_n = defaultdict(list)
    for r in records:
        by_n[r[0]].append(r)

    print(f"\nEvaluated {len(records):,} windows.\n")
    print("per-variant-token cosine and window-pooled cosine, binned by #SNP tokens")
    print("(ref = reference-only baseline; cache = variant cache; vs true alt forward)\n")
    hdr = (f"{'#SNP':>4}  {'n_win':>6}  {'var_cos':>9}  "
           f"{'cache_pool_cos':>14}  {'ref_pool_cos':>12}  "
           f"{'cache_relerr':>12}  {'ref_relerr':>10}")
    print(hdr)
    print("-" * len(hdr))
    for n in sorted(by_n):
        rows = by_n[n]
        print(f"{n:>4}  {len(rows):>6}  "
              f"{st.mean(col(rows,1)):>9.6f}  "
              f"{st.mean(col(rows,2)):>14.6f}  "
              f"{st.mean(col(rows,3)):>12.6f}  "
              f"{st.mean(col(rows,4)):>12.4e}  "
              f"{st.mean(col(rows,5)):>10.4e}")

    print("\nOverall:")
    print(f"  variant-token cosine : mean={st.mean(col(records,1)):.6f}  "
          f"min={min(col(records,1)):.6f}")
    print(f"  window cache cosine  : mean={st.mean(col(records,2)):.6f}  "
          f"min={min(col(records,2)):.6f}")
    print(f"  window ref  cosine   : mean={st.mean(col(records,3)):.6f}  "
          f"min={min(col(records,3)):.6f}")
    print(f"  window cache relerr  : mean={st.mean(col(records,4)):.4e}")
    print(f"  window ref  relerr   : mean={st.mean(col(records,5)):.4e}")
    print("\nNote: cache vs ref window error is the cache's pooled-embedding gain "
          "over ignoring variants. If they are ~equal, mean-pooling has diluted "
          "the variant signal and/or the pooled error is dominated by downstream "
          "non-variant positions the cache does not recompute — the cache's "
          "accuracy lives at the per-variant-token level (var_cos).")


if __name__ == "__main__":
    sys.exit(main())
