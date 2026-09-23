"""One frozen inference pass per backbone: per-site centre-base embeddings for
every unique accession window, saved as a reusable npz for CPU fitting.

  python -m training_meth.extract_features --backend carbon --n-sites 1000 \
      --contexts CG --pos-splits train val --out $SVAR_SCRATCH/runs/meth_feat_carbon.npz

npz layout (n_sites S, accessions A, feature dim D)
  feats      (U, D) float16   embeddings of all unique windows, sites concatenated
  site_ptr   (S+1,) int64     feats[site_ptr[s]:site_ptr[s+1]] belong to site s
  inverse    (S, A) int32     global row in feats for each accession
  ref_idx    (S,)   int32     global row of the pure-reference (TAIR10) window
  p, total   (S, A) float32 / int32   targets (p is nan below min_cov)
  geno       (G, A) int8, geno_ptr (S+1,), geno_pos (G,)   cis SNV dosage per site
  site_*     per-site metadata columns;  acc_id, acc_split, acc_group  (A,)
  meta       json string of the CLI args, backend, dim
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="carbon")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--pool-bp", type=int, default=0, help="average tokens covering centre +- this many bp")
    ap.add_argument("--no-bidir", action="store_true", help="skip the reverse-complement read-out")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--n-sites", type=int, default=500)
    ap.add_argument("--contexts", nargs="+", default=["CG"])
    ap.add_argument("--subtasks", nargs="+", default=None,
                    help="context_changing and/or invariant_context (default both)")
    ap.add_argument("--pos-splits", nargs="+", default=["train"])
    ap.add_argument("--annotation", nargs="+", default=None)
    ap.add_argument("--min-var-ratio", type=float, default=None, help="var_obs/noise floor")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--h5", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from training_meth.backbones import SiteEmbedder
    from training_meth.data import MethSiteSource, DEFAULT_H5

    src = MethSiteSource(a.h5 or DEFAULT_H5, window=a.window)
    idx = src.select_sites(a.n_sites, contexts=a.contexts, subtasks=a.subtasks, pos_splits=a.pos_splits,
                           annotation=a.annotation, seed=a.seed, min_var_ratio=a.min_var_ratio)
    print(f"{len(idx)} sites selected; contexts={a.contexts} subtasks={a.subtasks} "
          f"pos_splits={a.pos_splits}; accessions={src.n_acc}", flush=True)

    emb = SiteEmbedder(a.backend, a.model_path, layer=a.layer, pool_bp=a.pool_bp,
                       bidir=not a.no_bidir, batch_size=a.batch_size)
    print(f"backend={a.backend} path={emb.model_path} layer={a.layer} dim={emb.dim} "
          f"layout prefix={emb.prefix_tokens} chars/token={emb.chars_per_token}", flush=True)

    feats, site_ptr, inverse, ref_idx, P, T = [], [0], [], [], [], []
    geno, geno_ptr, geno_pos = [], [0], []
    pending_seqs, pending_sites = [], []
    t0 = time.time()

    def flush():
        if not pending_seqs:
            return
        F = emb.embed(pending_seqs).astype(np.float16)
        feats.append(F)
        pending_seqs.clear()

    n_uniq_total = 0
    for k, i in enumerate(idx):
        seqs, inv, ref = src.windows(int(i))
        base = n_uniq_total
        inverse.append(inv + base)
        ref_idx.append(ref + base)
        n_uniq_total += len(seqs)
        site_ptr.append(n_uniq_total)
        p, t = src.targets(int(i))
        P.append(p); T.append(t)
        pos, dos = src.genotypes(int(i))
        geno.append(dos); geno_pos.append(pos); geno_ptr.append(geno_ptr[-1] + len(pos))
        pending_seqs.extend(seqs)
        if len(pending_seqs) >= 4 * a.batch_size:
            flush()
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(idx)} sites, {n_uniq_total} unique windows, {time.time()-t0:.0f}s", flush=True)
    flush()

    feats = np.concatenate(feats) if feats else np.zeros((0, emb.dim), np.float16)
    assert feats.shape[0] == n_uniq_total
    s = src.sites.iloc[idx]
    out = dict(
        feats=feats, site_ptr=np.array(site_ptr, np.int64), inverse=np.stack(inverse).astype(np.int32),
        ref_idx=np.array(ref_idx, np.int32), p=np.stack(P).astype(np.float32), total=np.stack(T).astype(np.int32),
        geno=np.concatenate(geno, axis=0).astype(np.int8) if geno else np.zeros((0, src.n_acc), np.int8),
        geno_ptr=np.array(geno_ptr, np.int64), geno_pos=np.concatenate(geno_pos).astype(np.int64),
        site_index=idx.astype(np.int64),
        acc_id=src.acc.ecotype_id.to_numpy().astype(str), acc_split=src.acc.acc_split.to_numpy().astype(str),
        acc_group=src.acc.admixture_group.to_numpy().astype(str),
        meta=json.dumps({**vars(a), "dim": int(emb.dim), "model_path": emb.model_path,
                         "n_unique_windows": int(n_uniq_total), "seconds": time.time() - t0}),
    )
    for c in s.columns:
        v = s[c].to_numpy()
        out[f"site_{c}"] = v.astype(str) if v.dtype == object else v
    np.savez(a.out, **out)
    print(f"wrote {a.out}: {len(idx)} sites, {n_uniq_total} unique windows "
          f"({n_uniq_total/len(idx):.1f}/site), dim {emb.dim}, {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
