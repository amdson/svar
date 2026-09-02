#!/usr/bin/env python3
"""Stage 1 (spec §2): single streaming pass over all accession allc files,
accumulating per-site statistics into dense genome arrays.

Index layout: for nuclear chromosomes 1..5,
    idx = (chrom_offset[c] + (pos - 1)) * 2 + (strand == '-')
Positions are TAIR10 1-based (allc convention). Within one allc file the
(chrom, pos, strand) triple is unique, so `arr[idx] += v` fancy-indexing is a
safe, fast scatter-add (no np.add.at needed).

Accumulators (spec §2), all length 2 * genome_len:
    n_obs     int16   count of accessions with total >= MIN_COV
    sum_p     f64     sum of p_hat over those accessions
    sum_p2    f64     sum of p_hat^2
    sum_noise f64     sum of p_hat*(1-p_hat)/(total-1)
    sum_m     int32   sum of mc_count (cov-filtered rows)
    sum_total int32   sum of total    (cov-filtered rows)
    n_seen    int16   count of accessions with ANY row at the site (cov >= 1);
                      distinguishes "cytosine absent" from "not covered"
    context   int8    0=unset 1=CG 2=CHG 3=CHH (first non-N context seen)
    ctx_changed int8  1 if any accession's context disagreed (spec §3b signal)

Parallelism: file list is split into --workers chunks; each worker accumulates
privately and writes partial arrays to <out>/partials/chunkNN.npz (resumable);
the parent then reduces. Per-worker RSS ~12 GB.

Per-accession QC (spec §9): row counts, per-chrom row counts and max pos,
mean coverage, weighted mean methylation by context -> allc_qc.tsv.

CHH 100-bp window accumulators (spec §3c): per accession, CHH mc/total are
summed within 100-bp genome windows (strand-collapsed); windows with summed
total >= WIN_MIN_COV contribute one binomial observation p_w = m_w/t_w to
window-level n_obs/sum_p/sum_p2/sum_noise. Done in this same pass so the
binned CHH target needs no re-streaming.
"""
import argparse
import gzip
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

CHROMS = ["1", "2", "3", "4", "5"]
DROP_CHROMS = {"C", "M", "PT", "MT", "CHLOROPLAST", "MITOCHONDRIA"}

STAT_SPECS = [  # name, dtype
    ("n_obs", np.int16), ("sum_p", np.float64), ("sum_p2", np.float64),
    ("sum_noise", np.float64), ("sum_m", np.int32), ("sum_total", np.int32),
    ("n_seen", np.int16),
]

WIN = 100           # CHH bin width (bp), spec §3c
WIN_MIN_COV = 10    # min summed total for a window observation
WIN_SPECS = [
    ("win_n_obs", np.int16), ("win_sum_p", np.float64),
    ("win_sum_p2", np.float64), ("win_sum_noise", np.float64),
]


def genome_layout(fai_path):
    lens = {}
    with open(fai_path) as fh:
        for line in fh:
            f = line.split("\t")
            if f[0] in CHROMS:
                lens[f[0]] = int(f[1])
    assert set(lens) == set(CHROMS), f"missing chroms in fai: {lens.keys()}"
    offsets, off = {}, 0
    for c in CHROMS:
        offsets[c] = off
        off += lens[c]
    return lens, offsets, off  # off == genome_len


def normalize_chrom(series):
    s = series.astype(str).str.upper().str.removeprefix("CHR")
    return s


def context_codes(mc_class):
    """mc_class 3-char strings -> 1=CG 2=CHG 3=CHH 0=unknown(N)."""
    b = mc_class.astype("S3").view(np.uint8).reshape(-1, 3)
    second, third = b[:, 1], b[:, 2]
    G, N = ord("G"), ord("N")
    code = np.full(len(b), 3, dtype=np.int8)          # default CHH
    code[third == G] = 2                               # CHG
    code[second == G] = 1                              # CG (overrides)
    code[(second == N) | ((third == N) & (second != G))] = 0
    return code


def process_file(path, offsets, genome_len, min_cov, stats, context, ctx_changed,
                 wstats=None):
    """Accumulate one allc file into the provided arrays; return QC dict."""
    qc = {"file": os.path.basename(path)}
    try:
        with gzip.open(path, "rt") as fh:
            first = fh.readline()
        has_header = first.startswith("chrom")
        df = pd.read_csv(
            path, sep="\t", header=0 if has_header else None,
            names=["chrom", "pos", "strand", "mc_class", "mc", "total"],
            usecols=[0, 1, 2, 3, 4, 5],
            dtype={"chrom": str, "pos": np.int64, "strand": str,
                   "mc_class": str, "mc": np.int32, "total": np.int32},
        )
    except Exception as e:
        qc["error"] = f"{type(e).__name__}: {e}"
        return qc

    qc["n_rows_raw"] = len(df)
    df["chrom"] = normalize_chrom(df["chrom"])
    dropped = ~df["chrom"].isin(CHROMS)
    qc["n_rows_dropped_chrom"] = int(dropped.sum())
    df = df[~dropped]

    # per-chrom row counts and max pos (truncation check, spec §9)
    g = df.groupby("chrom")["pos"]
    qc["per_chrom_rows"] = g.size().to_dict()
    qc["per_chrom_maxpos"] = g.max().to_dict()

    chrom_off = df["chrom"].map(offsets).to_numpy(np.int64)
    strand_bit = (df["strand"].to_numpy() == "-").astype(np.int64)
    idx = (chrom_off + df["pos"].to_numpy(np.int64) - 1) * 2 + strand_bit
    if idx.size and (idx.min() < 0 or idx.max() >= 2 * genome_len):
        qc["error"] = "position out of genome bounds"
        return qc

    mc = df["mc"].to_numpy(np.float64)
    total = df["total"].to_numpy(np.float64)
    code = context_codes(df["mc_class"].to_numpy())

    # presence (any coverage)
    stats["n_seen"][idx] += 1

    # context consistency: first non-N seen sets it; later disagreement flags
    known = code != 0
    ki, kc = idx[known], code[known]
    cur = context[ki]
    unset = cur == 0
    context[ki[unset]] = kc[unset]
    mismatch = (~unset) & (cur != kc)
    ctx_changed[ki[mismatch]] = 1

    # coverage-filtered stats
    cov = total >= min_cov
    ci = idx[cov]
    m, t = mc[cov], total[cov]
    p = m / t
    stats["n_obs"][ci] += 1
    stats["sum_p"][ci] += p
    stats["sum_p2"][ci] += p * p
    stats["sum_noise"][ci] += p * (1.0 - p) / (t - 1.0)
    stats["sum_m"][ci] += m.astype(np.int32)
    stats["sum_total"][ci] += t.astype(np.int32)

    # CHH 100-bp windows (spec §3c): strand-collapsed genome bins
    if wstats is not None:
        chh = code == 3
        gpos = idx[chh] // 2          # strand-collapsed genome coordinate
        win = gpos // WIN
        n_win = (genome_len + WIN - 1) // WIN
        m_w = np.bincount(win, weights=mc[chh], minlength=n_win)
        t_w = np.bincount(win, weights=total[chh], minlength=n_win)
        wi = np.flatnonzero(t_w >= WIN_MIN_COV)
        pw = m_w[wi] / t_w[wi]
        wstats["win_n_obs"][wi] += 1
        wstats["win_sum_p"][wi] += pw
        wstats["win_sum_p2"][wi] += pw * pw
        wstats["win_sum_noise"][wi] += pw * (1.0 - pw) / (t_w[wi] - 1.0)

    qc["n_rows_cov"] = int(cov.sum())
    qc["mean_cov"] = float(total.mean()) if len(total) else 0.0
    for name, cval in (("CG", 1), ("CHG", 2), ("CHH", 3)):
        sel = cov & (code == cval)
        ts = total[sel].sum()
        qc[f"wmean_m{name}"] = float(mc[sel].sum() / ts) if ts > 0 else float("nan")
    return qc


def run_chunk(args):
    chunk_id, files, fai, min_cov, out_dir = args
    part = Path(out_dir) / "partials" / f"chunk{chunk_id:03d}.npz"
    qc_part = Path(out_dir) / "partials" / f"chunk{chunk_id:03d}_qc.json"
    if part.exists() and qc_part.exists():
        return str(part)
    lens, offsets, genome_len = genome_layout(fai)
    n = 2 * genome_len
    n_win = (genome_len + WIN - 1) // WIN
    stats = {name: np.zeros(n, dtype=dt) for name, dt in STAT_SPECS}
    wstats = {name: np.zeros(n_win, dtype=dt) for name, dt in WIN_SPECS}
    context = np.zeros(n, dtype=np.int8)
    ctx_changed = np.zeros(n, dtype=np.int8)
    qcs = []
    for i, f in enumerate(files):
        qcs.append(process_file(f, offsets, genome_len, min_cov, stats, context,
                                ctx_changed, wstats))
        print(f"[chunk {chunk_id}] {i+1}/{len(files)} {os.path.basename(f)}", flush=True)
    np.savez(part.with_suffix(".tmp.npz"), context=context, ctx_changed=ctx_changed,
             **stats, **wstats)
    os.replace(part.with_suffix(".tmp.npz"), part)
    qc_part.write_text(json.dumps(qcs))
    return str(part)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allc-dir", required=True)
    ap.add_argument("--accessions", required=True,
                    help="benchmark_accessions.tsv (defines the 811 files used)")
    ap.add_argument("--fai", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-cov", type=int, default=5)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    out = Path(a.out_dir)
    (out / "partials").mkdir(parents=True, exist_ok=True)

    acc = pd.read_csv(a.accessions, sep="\t", dtype=str)
    files = [str(Path(a.allc_dir) / f) for f in acc["allc_file"]]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        sys.exit(f"{len(missing)} allc files missing, e.g. {missing[:5]}")

    chunks = [(i, files[i::a.workers], a.fai, a.min_cov, str(out))
              for i in range(a.workers)]
    with Pool(a.workers) as pool:
        parts = pool.map(run_chunk, chunks)

    # reduce
    lens, offsets, genome_len = genome_layout(a.fai)
    n = 2 * genome_len
    n_win = (genome_len + WIN - 1) // WIN
    stats = {name: np.zeros(n, dtype=dt) for name, dt in STAT_SPECS}
    wstats = {name: np.zeros(n_win, dtype=dt) for name, dt in WIN_SPECS}
    context = np.zeros(n, dtype=np.int8)
    ctx_changed = np.zeros(n, dtype=np.int8)
    for p in parts:
        z = np.load(p)
        for name, _ in STAT_SPECS:
            stats[name] += z[name]
        for name, _ in WIN_SPECS:
            wstats[name] += z[name]
        pc = z["context"]
        disagree = (context != 0) & (pc != 0) & (context != pc)
        ctx_changed[disagree] = 1
        context[(context == 0)] = pc[(context == 0)]
        ctx_changed |= z["ctx_changed"]
        del z

    for name, _ in STAT_SPECS:
        np.save(out / f"{name}.npy", stats[name])
    for name, _ in WIN_SPECS:
        np.save(out / f"{name}.npy", wstats[name])
    np.save(out / "context.npy", context)
    np.save(out / "ctx_changed.npy", ctx_changed)

    # merge QC
    qcs = []
    for p in sorted((out / "partials").glob("chunk*_qc.json")):
        qcs.extend(json.loads(p.read_text()))
    qdf = pd.DataFrame(qcs)
    file2eco = dict(zip(acc["allc_file"], acc["ecotype_id"]))
    qdf.insert(0, "ecotype_id", qdf["file"].map(file2eco))
    qdf.to_csv(out / "allc_qc.tsv", sep="\t", index=False)

    meta = {"min_cov": a.min_cov, "n_accessions": len(files),
            "chrom_lengths": lens, "chrom_offsets": offsets,
            "genome_len": genome_len,
            "index": "(offset[chrom] + pos - 1) * 2 + (strand=='-')",
            "win": WIN, "win_min_cov": WIN_MIN_COV,
            "win_index": "(offset[chrom] + pos - 1) // WIN, strand-collapsed"}
    (out / "stage1_meta.json").write_text(json.dumps(meta, indent=2))
    print("Stage 1 accumulation complete:", out)


if __name__ == "__main__":
    main()
