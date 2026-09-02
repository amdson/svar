#!/usr/bin/env python3
"""Stage 3 (spec §4): second streaming pass — extract (mc_count, total) at the
Stage 2 retained sites into the final HDF5 dataset.

Also assigns the accession-axis split (spec §6): by admixture group, never
random. Defaults (config): test=italy_balkan_caucasus, val=asia, everything
else (incl. admixed and the one unlabelled accession) train.

Output dataset.h5:
  sites/      chrom,pos,strand,context,subtask,split_role,pos_split,
              mean_p,var_obs,noise,annotation_class,cg_density  (variable+invariant)
  counts      (n_sites, n_accessions, 2) int16  [mc, total]
  chh_bins/   window table + counts (n_bins, n_accessions, 2) int32
  accessions/ ecotype_id, admixture_group, tissue, gsm, source_series, acc_split
"""
import argparse
import gzip
import json
import os
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

CHROMS = ["1", "2", "3", "4", "5"]

# module-level state for workers (fork inheritance)
G = {}


def normalize_chrom(series):
    return series.astype(str).str.upper().str.removeprefix("CHR")


def worker(args):
    slot, path = args
    site_idx = G["site_idx"]          # sorted global strand-site indices
    win_idx = G["win_idx"]            # sorted global window indices (may be empty)
    offsets, glen, winL = G["offsets"], G["glen"], G["winL"]

    counts = np.zeros((len(site_idx), 2), dtype=np.int16)
    wcounts = np.zeros((len(win_idx), 2), dtype=np.int32)
    with gzip.open(path, "rt") as fh:
        has_header = fh.readline().startswith("chrom")
    df = pd.read_csv(path, sep="\t", header=0 if has_header else None,
                     names=["chrom", "pos", "strand", "mc_class", "mc", "total"],
                     usecols=[0, 1, 2, 3, 4, 5],
                     dtype={"chrom": str, "pos": np.int64, "strand": str,
                            "mc_class": str, "mc": np.int32, "total": np.int32})
    df["chrom"] = normalize_chrom(df["chrom"])
    df = df[df["chrom"].isin(CHROMS)]
    chrom_off = df["chrom"].map(offsets).to_numpy(np.int64)
    pos = df["pos"].to_numpy(np.int64)
    strand_bit = (df["strand"].to_numpy() == "-").astype(np.int64)
    idx = (chrom_off + pos - 1) * 2 + strand_bit
    mc = df["mc"].to_numpy(np.int32)
    total = df["total"].to_numpy(np.int32)

    loc = np.searchsorted(site_idx, idx)
    loc_ok = (loc < len(site_idx))
    hit = np.zeros(len(idx), dtype=bool)
    hit[loc_ok] = site_idx[loc[loc_ok]] == idx[loc_ok]
    counts[loc[hit], 0] = np.clip(mc[hit], 0, 32767)
    counts[loc[hit], 1] = np.clip(total[hit], 0, 32767)

    if len(win_idx):
        b = df["mc_class"].to_numpy().astype("S3").view(np.uint8).reshape(-1, 3)
        chh = (b[:, 1] != ord("G")) & (b[:, 2] != ord("G")) & \
              (b[:, 1] != ord("N")) & (b[:, 2] != ord("N"))
        gwin = (idx[chh] // 2) // winL
        m_w = np.bincount(gwin, weights=mc[chh], minlength=(glen + winL - 1) // winL)
        t_w = np.bincount(gwin, weights=total[chh], minlength=(glen + winL - 1) // winL)
        wcounts[:, 0] = m_w[win_idx]
        wcounts[:, 1] = t_w[win_idx]

    np.save(G["tmp"] / f"acc{slot:04d}_sites.npy", counts)
    np.save(G["tmp"] / f"acc{slot:04d}_wins.npy", wcounts)
    return slot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-dir", required=True)
    ap.add_argument("--stage2-dir", required=True)
    ap.add_argument("--allc-dir", required=True)
    ap.add_argument("--accessions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--acc-test-groups", default="italy_balkan_caucasus")
    ap.add_argument("--acc-val-groups", default="asia")
    a = ap.parse_args()

    meta = json.loads((Path(a.stage1_dir) / "stage1_meta.json").read_text())
    s2 = Path(a.stage2_dir)
    var_df = pd.read_parquet(s2 / "sites_variable.parquet")
    inv_df = pd.read_parquet(s2 / "sites_invariant.parquet")
    sites = pd.concat([var_df, inv_df], ignore_index=True)
    sites = sites.sort_values("site_idx").reset_index(drop=True)
    site_idx = sites["site_idx"].to_numpy(np.int64)
    assert (np.diff(site_idx) > 0).all(), "site_idx must be unique/sorted"

    bins_path = s2 / "chh_bins.parquet"
    if bins_path.exists():
        bins = pd.read_parquet(bins_path).sort_values("win_idx").reset_index(drop=True)
        win_idx = bins["win_idx"].to_numpy(np.int64)
    else:
        bins, win_idx = None, np.empty(0, np.int64)

    acc = pd.read_csv(a.accessions, sep="\t", dtype=str)
    test_g = set(a.acc_test_groups.split(","))
    val_g = set(a.acc_val_groups.split(","))
    acc["acc_split"] = ["test" if g in test_g else "val" if g in val_g else "train"
                        for g in acc["admixture_group"].fillna("")]
    print(acc.groupby("acc_split").size())

    tmp = Path(a.out).parent / "stage3_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    G.update(site_idx=site_idx, win_idx=win_idx, offsets=meta["chrom_offsets"],
             glen=meta["genome_len"], winL=meta["win"], tmp=tmp)

    jobs = [(i, os.path.join(a.allc_dir, f)) for i, f in enumerate(acc["allc_file"])]
    todo = [j for j in jobs if not (tmp / f"acc{j[0]:04d}_sites.npy").exists()]
    print(f"{len(todo)}/{len(jobs)} accessions to extract")
    if todo:
        with Pool(a.workers) as pool:
            for i, slot in enumerate(pool.imap_unordered(worker, todo)):
                print(f"done {i+1}/{len(todo)} (slot {slot})", flush=True)

    n_sites, n_acc = len(sites), len(acc)
    counts = np.zeros((n_sites, n_acc, 2), dtype=np.int16)
    wcounts = np.zeros((len(win_idx), n_acc, 2), dtype=np.int32)
    for i in range(n_acc):
        counts[:, i, :] = np.load(tmp / f"acc{i:04d}_sites.npy")
        if len(win_idx):
            wcounts[:, i, :] = np.load(tmp / f"acc{i:04d}_wins.npy")

    str_dt = h5py.string_dtype()
    with h5py.File(a.out, "w") as h5:
        gs = h5.create_group("sites")
        for col in ["chrom", "strand", "context", "subtask", "split_role",
                    "pos_split", "annotation_class"]:
            gs.create_dataset(col, data=sites[col].astype(str).to_numpy(), dtype=str_dt)
        for col, dt in [("pos", np.int64), ("mean_p", np.float32),
                        ("var_obs", np.float32), ("noise", np.float32),
                        ("n_obs", np.int16), ("cg_density", np.int16),
                        ("site_idx", np.int64)]:
            gs.create_dataset(col, data=sites[col].to_numpy(dt))
        h5.create_dataset("counts", data=counts, chunks=(min(n_sites, 4096), n_acc, 2),
                          compression="lzf")
        ga = h5.create_group("accessions")
        for col in ["ecotype_id", "admixture_group", "tissue", "gsm",
                    "source_series", "acc_split"]:
            ga.create_dataset(col, data=acc[col].fillna("").astype(str).to_numpy(),
                              dtype=str_dt)
        if bins is not None and len(win_idx):
            gb = h5.create_group("chh_bins")
            for col in ["chrom", "annotation_class", "pos_split"]:
                gb.create_dataset(col, data=bins[col].astype(str).to_numpy(), dtype=str_dt)
            for col, dt in [("start", np.int64), ("end", np.int64),
                            ("mean_p", np.float32), ("var_obs", np.float32),
                            ("noise", np.float32), ("n_obs", np.int16),
                            ("win_idx", np.int64)]:
                gb.create_dataset(col, data=bins[col].to_numpy(dt))
            gb.create_dataset("counts", data=wcounts,
                              chunks=(min(len(win_idx), 4096), n_acc, 2),
                              compression="lzf")
        h5.attrs["spec"] = ("targets GSE43857 only (22C, Salk); v1 SNVs-only; "
                            "TAIR10 coordinates; counts=[mc,total] int16; "
                            "MIN_COV=" + str(meta["min_cov"]))
    print("wrote", a.out, f"({n_sites:,} sites x {n_acc} accessions)")


if __name__ == "__main__":
    main()
