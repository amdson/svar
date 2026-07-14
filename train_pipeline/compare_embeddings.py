"""
compare_embeddings.py
---------------------
Compare two window-embedding tables produced by embed_windows.py and report the
max L2 norm of the per-window difference vector.

Rows are aligned by window fingerprint (chrom, w_start, w_end, alt_positions)
rather than by position, so the comparison is correct even if the two files
order their cache rows differently.

    python train_pipeline/compare_embeddings.py a.ckpt.pt b.ckpt.pt
"""

import argparse

import torch


def _load_rows(path):
    """Return {fingerprint_key: embedding_tensor} for one cache file."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cache = payload["cache"]
    fingerprints = payload["unique_fingerprints"]
    if len(fingerprints) != cache.shape[0]:
        raise ValueError(
            f"{path}: {len(fingerprints)} fingerprints but {cache.shape[0]} cache rows"
        )
    return {_key(fp): cache[i] for i, fp in enumerate(fingerprints)}


def _key(fp):
    """Make a fingerprint hashable (alt_positions may be a list)."""
    chrom, w_start, w_end, alt_positions = fp
    return (chrom, w_start, w_end, tuple(alt_positions))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file_a")
    parser.add_argument("file_b")
    parser.add_argument("--csv-out", type=str, default=None,
                        help="If set, write one row per shared window "
                             "(n_snp, diff_norm, ref_norm, rel_diff) to this CSV "
                             "for plotting (presentation/plot_cache_vs_fullforward.py).")
    args = parser.parse_args()

    rows_a = _load_rows(args.file_a)
    rows_b = _load_rows(args.file_b)

    shared = rows_a.keys() & rows_b.keys()
    only_a = rows_a.keys() - rows_b.keys()
    only_b = rows_b.keys() - rows_a.keys()
    if not shared:
        raise SystemExit("No windows in common between the two files.")

    diff_norms = torch.stack([(rows_a[k] - rows_b[k]).norm() for k in shared])
    max_norm = diff_norms.max().item()

    print(f"shared windows: {len(shared):,}  "
          f"(only in A: {len(only_a):,}, only in B: {len(only_b):,})")
    print(f"max diff norm:  {max_norm:.6g}")
    print(f"mean diff norm: {diff_norms.mean().item():.6g}")
    print(f"mean overall norm: {torch.stack([v.norm() for v in rows_a.values()]).mean().item():.6g}")

    if args.csv_out:
        import csv
        # key = (chrom, w_start, w_end, alt_positions); #SNP = len(alt_positions).
        with open(args.csv_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["n_snp", "diff_norm", "ref_norm", "rel_diff"])
            for k in shared:
                ref_norm = rows_a[k].norm().item()
                diff = (rows_a[k] - rows_b[k]).norm().item()
                w.writerow([len(k[3]), diff, ref_norm,
                            diff / ref_norm if ref_norm else float("nan")])
        print(f"wrote per-window CSV: {args.csv_out}  ({len(shared):,} rows)")


if __name__ == "__main__":
    main()
