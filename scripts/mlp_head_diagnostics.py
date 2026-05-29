"""
mlp_head_diagnostics.py
-----------------------
A/B matrix to localize why the production crop_embed.MLPHead pipeline
underperforms sklearn MLPRegressor on the same mean-pooled features.

All runners use the same 10-fold CV over the same matched samples / traits,
so deltas isolate one knob at a time:

  sklearn_mlp                  reference (StandardScaler, per-trait, early stop)
  torch_pertrait_scaled_es     should ≈ sklearn (framework-only delta)
  torch_pertrait_unscaled_es   cost of no StandardScaler
  torch_pertrait_scaled_noes   cost of no early stopping
  torch_multi_scaled_es        cost of multitask training (with masked MSE)
  torch_multi_unscaled_noes    multitask + no scaling + no early stop (closest
                               hand-written analogue of crop_embed.train())
  crop_embed_unscaled          production: crop_embed.train() + per-window cache
  crop_embed_scaled            production code + per-dim cache StandardScaling

Each runner reports per-trait Pearson r (10-fold mean). The script prints
a Trait × Runner table and a MEAN row. Walk left-to-right to read off where
the gap appears.

Example
-------
    python scripts/mlp_head_diagnostics.py \\
        --cache checkpoints/v1/sativas413_embeddings.ckpt.pt
"""

import argparse
import contextlib
import io
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crop_embed import FixedWindowEmbedder, MLPHead, SampleDataset
from crop_embed import train as crop_train

DEFAULT_PHENO = "/home/andrew.dickson/rice_data/RiceDiversity_44K_Phenotypes_34traits_PLINK.txt"
DEFAULT_TRAITS = [
    "Alkali spreading value",
    "Amylose content",
    "Panicle number per plant",
    "Protein content",
    "Seed length",
    "Seed number per panicle",
]

# ── Data loading ─────────────────────────────────────────────────────────────


def load_cache(path: str):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cache_t = blob["cache"].float()
    sfp     = blob["sample_fp_index"].long()
    samples = blob.get("sample_ids")
    if samples is None:
        raise SystemExit("cache file missing 'sample_ids'; regenerate with generate_cache.py")
    return cache_t, sfp, samples


def load_phenos(path: str, want_traits: list[str], cache_samples: list[str]):
    df = pd.read_csv(path, sep="\t")
    df["NSFTVID"] = df["NSFTVID"].astype(int)
    all_traits = [c for c in df.columns if c not in ("HybID", "NSFTVID")]

    cache_id_to_ix = {
        int(s.rsplit("_", 1)[-1]): i for i, s in enumerate(cache_samples)
    }
    matched = (
        df[df["NSFTVID"].isin(cache_id_to_ix)].copy().reset_index(drop=True)
    )
    cache_row_order = np.asarray(
        [cache_id_to_ix[nid] for nid in matched["NSFTVID"]], dtype=np.int64
    )

    Y_raw    = matched[all_traits].values.astype(float)
    Y_scaled = np.full_like(Y_raw, np.nan)
    for j in range(Y_raw.shape[1]):
        col  = Y_raw[:, j]
        m    = ~np.isnan(col)
        if m.sum() < 2:
            continue
        mu, sd = col[m].mean(), col[m].std(ddof=0)
        Y_scaled[m, j] = (col[m] - mu) / (sd if sd > 0 else 1.0)

    trait_indices = [all_traits.index(t) for t in want_traits if t in all_traits]
    return Y_scaled, cache_row_order, trait_indices


def mean_pool(cache_t: torch.Tensor, sfp: torch.Tensor, batch: int = 8) -> np.ndarray:
    n_samples, _ = sfp.shape
    _, D = cache_t.shape
    out = torch.empty((n_samples, D), dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, n_samples, batch):
            j = min(i + batch, n_samples)
            out[i:j] = cache_t[sfp[i:j]].mean(dim=1)
    return out.numpy()


# ── CV harness ───────────────────────────────────────────────────────────────


def _folds(n: int, n_splits: int, seed: int):
    return list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.arange(n)))


def per_trait_cv(X, Y, trait_indices, fit_predict, n_splits, seed):
    """fit_predict(X_tr, y_tr, X_te) → y_pred (1-D). Returns {trait_idx: r}."""
    out = {}
    for j in trait_indices:
        y = Y[:, j]
        m = ~np.isnan(y)
        Xm, ym = X[m], y[m]
        rs = []
        for tr, te in _folds(Xm.shape[0], n_splits, seed):
            yp = fit_predict(Xm[tr], ym[tr], Xm[te])
            rs.append(pearsonr(ym[te], yp)[0])
        out[j] = float(np.mean(rs))
    return out


def multitask_cv(X_or_None, Y, trait_indices, fit_predict, n_splits, seed, n_rows):
    """
    Multitask CV. fit_predict(tr_idx, te_idx) → Y_pred[len(te_idx), n_traits].
    Per-trait Pearson r averaged across folds, masking NaN test entries.
    """
    per_trait = {j: [] for j in trait_indices}
    for tr, te in _folds(n_rows, n_splits, seed):
        Y_pred = fit_predict(tr, te)
        for j in trait_indices:
            m = ~np.isnan(Y[te, j])
            if m.sum() < 5:
                continue
            r, _ = pearsonr(Y[te][m, j], Y_pred[m, j])
            per_trait[j].append(r)
    return {j: float(np.mean(v)) if v else float("nan") for j, v in per_trait.items()}


# ── Models ───────────────────────────────────────────────────────────────────


class TorchMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x): return self.net(x)


def _train_torch(
    model: nn.Module,
    Xt: torch.Tensor, Yt: torch.Tensor,
    Xv: torch.Tensor | None, Yv: torch.Tensor | None,
    hp, masked: bool, early_stopping: bool,
) -> None:
    optim = torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)

    def loss_fn(pred, target):
        if masked:
            m = ~torch.isnan(target)
            return ((pred[m] - target[m]) ** 2).mean() if m.any() else pred.sum() * 0.0
        return ((pred - target) ** 2).mean()

    best_val, best_state, patience, no_improve = float("inf"), None, 10, 0
    for epoch in range(hp.max_epochs):
        model.train()
        perm = torch.randperm(Xt.shape[0])
        for i in range(0, Xt.shape[0], hp.batch_size):
            idx = perm[i : i + hp.batch_size]
            optim.zero_grad()
            loss = loss_fn(model(Xt[idx]), Yt[idx])
            loss.backward()
            optim.step()

        if not early_stopping:
            continue
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xv), Yv).item()
        if vl < best_val - 1e-5:
            best_val, best_state, no_improve = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    if early_stopping and best_state is not None:
        model.load_state_dict(best_state)


def _val_split(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    val_n = max(2, n // 10)
    val_ix = torch.randperm(n, generator=g)[:val_n]
    val_mask = torch.zeros(n, dtype=torch.bool)
    val_mask[val_ix] = True
    return val_mask, ~val_mask


# ── Runners ──────────────────────────────────────────────────────────────────


def run_sklearn_mlp(X, Y, trait_indices, hp):
    def fp(Xt, yt, Xe):
        scaler = StandardScaler()
        Xt = scaler.fit_transform(Xt)
        Xe = scaler.transform(Xe)
        mlp = MLPRegressor(
            hidden_layer_sizes=(hp.hidden_dim,),
            activation="relu", solver="adam",
            alpha=hp.weight_decay, batch_size=hp.batch_size,
            learning_rate_init=hp.lr, max_iter=hp.max_epochs,
            early_stopping=True, validation_fraction=0.1,
            random_state=hp.seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            mlp.fit(Xt, yt)
        return mlp.predict(Xe)
    return per_trait_cv(X, Y, trait_indices, fp, hp.n_splits, hp.seed)


def run_torch_pertrait(X, Y, trait_indices, hp, *, scale: bool, early_stopping: bool):
    def fp(Xt, yt, Xe):
        torch.manual_seed(hp.seed)
        if scale:
            sc = StandardScaler()
            Xt, Xe = sc.fit_transform(Xt), sc.transform(Xe)
        Xt_t = torch.tensor(Xt, dtype=torch.float32)
        yt_t = torch.tensor(yt, dtype=torch.float32).unsqueeze(-1)
        Xe_t = torch.tensor(Xe, dtype=torch.float32)

        if early_stopping:
            v_mask, t_mask = _val_split(Xt_t.shape[0], hp.seed)
            Xv, yv = Xt_t[v_mask], yt_t[v_mask]
            Xt_t, yt_t = Xt_t[t_mask], yt_t[t_mask]
        else:
            Xv = yv = None

        model = TorchMLP(Xt_t.shape[1], hp.hidden_dim, 1)
        _train_torch(model, Xt_t, yt_t, Xv, yv, hp, masked=False, early_stopping=early_stopping)
        model.eval()
        with torch.no_grad():
            return model(Xe_t).numpy().squeeze(-1)
    return per_trait_cv(X, Y, trait_indices, fp, hp.n_splits, hp.seed)


def run_torch_multitask(X, Y, trait_indices, hp, *, scale: bool, early_stopping: bool):
    n_traits = Y.shape[1]

    def fp(tr, te):
        torch.manual_seed(hp.seed)
        Xtr, Ytr, Xte = X[tr], Y[tr], X[te]
        if scale:
            sc = StandardScaler()
            Xtr, Xte = sc.fit_transform(Xtr), sc.transform(Xte)
        Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
        Ytr_t = torch.tensor(Ytr, dtype=torch.float32)
        Xte_t = torch.tensor(Xte, dtype=torch.float32)

        if early_stopping:
            v_mask, t_mask = _val_split(Xtr_t.shape[0], hp.seed)
            Xv, Yv = Xtr_t[v_mask], Ytr_t[v_mask]
            Xtr_t, Ytr_t = Xtr_t[t_mask], Ytr_t[t_mask]
        else:
            Xv = Yv = None

        model = TorchMLP(Xtr_t.shape[1], hp.hidden_dim, n_traits)
        _train_torch(model, Xtr_t, Ytr_t, Xv, Yv, hp, masked=True, early_stopping=early_stopping)
        model.eval()
        with torch.no_grad():
            return model(Xte_t).numpy()
    return multitask_cv(X, Y, trait_indices, fp, hp.n_splits, hp.seed, n_rows=X.shape[0])


def run_crop_embed(cache_t, sub_sfp, Y, trait_indices, hp, *, scale: bool):
    n_rows = sub_sfp.shape[0]
    n_traits = Y.shape[1]
    emb_dim = cache_t.shape[1]
    Y_t = torch.tensor(Y, dtype=torch.float32)

    if scale:
        mu = cache_t.mean(dim=0, keepdim=True)
        sd = cache_t.std(dim=0, keepdim=True).clamp(min=1e-6)
        cache_use = (cache_t - mu) / sd
    else:
        cache_use = cache_t

    fixed = FixedWindowEmbedder(cache_use, sub_sfp)
    stub_ds = SimpleNamespace(samples=[f"s{i}" for i in range(n_rows)])

    def fp(tr, te):
        head = MLPHead(emb_dim=emb_dim, n_traits=n_traits, hidden_dim=hp.hidden_dim, dropout=0.0)
        train_ds = SampleDataset(stub_ds, Y_t, torch.tensor(tr, dtype=torch.long))
        val_ds   = SampleDataset(stub_ds, Y_t, torch.tensor(te, dtype=torch.long))
        # Suppress train()'s periodic prints; loss curves not useful here.
        with contextlib.redirect_stdout(io.StringIO()):
            crop_train(
                fixed, head, train_ds, val_ds,
                batch_size=hp.batch_size, lr=hp.lr, weight_decay=hp.weight_decay,
                steps=hp.steps, log_every=10**9,
            )
        head.eval()
        with torch.no_grad():
            return head(fixed(torch.tensor(te, dtype=torch.long))).cpu().numpy()
    return multitask_cv(None, Y, trait_indices, fp, hp.n_splits, hp.seed, n_rows=n_rows)


# ── Main ─────────────────────────────────────────────────────────────────────


ALL_RUNNERS = [
    "sklearn_mlp",
    "torch_pertrait_scaled_es",
    "torch_pertrait_unscaled_es",
    "torch_pertrait_scaled_noes",
    "torch_multi_scaled_es",
    "torch_multi_unscaled_noes",
    "crop_embed_unscaled",
    "crop_embed_scaled",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=str, required=True)
    p.add_argument("--pheno", type=str, default=DEFAULT_PHENO)
    p.add_argument("--traits", nargs="+", default=DEFAULT_TRAITS)
    p.add_argument("--runners", nargs="+", choices=ALL_RUNNERS, default=ALL_RUNNERS)
    p.add_argument("--n-splits",    type=int,   default=10)
    p.add_argument("--hidden-dim",  type=int,   default=768)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--max-epochs",  type=int,   default=200, help="Sklearn / torch_* runners.")
    p.add_argument("--steps",       type=int,   default=1000, help="crop_embed.train() step budget.")
    p.add_argument("--batch-size",  type=int,   default=32)
    p.add_argument("--seed",        type=int,   default=42)
    args = p.parse_args()

    print(f"Cache : {args.cache}")
    cache_t, sfp, cache_samples = load_cache(args.cache)
    print(f"  cache {tuple(cache_t.shape)}  sample_fp_index {tuple(sfp.shape)}")

    Y, row_order, trait_indices = load_phenos(args.pheno, args.traits, cache_samples)
    print(f"Matched {len(row_order)} samples, {len(trait_indices)} traits requested")

    print("Mean-pooling window cache …")
    sample_vec = mean_pool(cache_t, sfp)
    X = sample_vec[row_order]                            # (n, D)
    Y = Y                                                # (n, all_traits)
    sub_sfp = sfp[torch.tensor(row_order, dtype=torch.long)]  # (n, n_windows), for crop_embed
    print(f"  X {X.shape}  Y {Y.shape}  sub_sfp {tuple(sub_sfp.shape)}")

    hp = SimpleNamespace(
        hidden_dim=args.hidden_dim, weight_decay=args.weight_decay,
        lr=args.lr, max_epochs=args.max_epochs, steps=args.steps,
        batch_size=args.batch_size, seed=args.seed, n_splits=args.n_splits,
    )

    runners = {
        "sklearn_mlp":                lambda: run_sklearn_mlp(X, Y, trait_indices, hp),
        "torch_pertrait_scaled_es":   lambda: run_torch_pertrait(X, Y, trait_indices, hp, scale=True,  early_stopping=True),
        "torch_pertrait_unscaled_es": lambda: run_torch_pertrait(X, Y, trait_indices, hp, scale=False, early_stopping=True),
        "torch_pertrait_scaled_noes": lambda: run_torch_pertrait(X, Y, trait_indices, hp, scale=True,  early_stopping=False),
        "torch_multi_scaled_es":      lambda: run_torch_multitask(X, Y, trait_indices, hp, scale=True,  early_stopping=True),
        "torch_multi_unscaled_noes":  lambda: run_torch_multitask(X, Y, trait_indices, hp, scale=False, early_stopping=False),
        "crop_embed_unscaled":        lambda: run_crop_embed(cache_t, sub_sfp, Y, trait_indices, hp, scale=False),
        "crop_embed_scaled":          lambda: run_crop_embed(cache_t, sub_sfp, Y, trait_indices, hp, scale=True),
    }

    results = {}
    for name in args.runners:
        print(f"\n>>> {name}")
        t0 = time.time()
        results[name] = runners[name]()
        print(f"    ({time.time() - t0:.1f}s)")

    # Build comparison table. trait_indices may be a subset of args.traits when
    # any requested trait is missing from the pheno file.
    name_by_idx = {trait_indices[k]: args.traits[k] for k in range(len(trait_indices))}
    rows = []
    for j in trait_indices:
        row = {"Trait": name_by_idx[j]}
        for name, r in results.items():
            row[name] = round(r.get(j, float("nan")), 4)
        rows.append(row)
    df = pd.DataFrame(rows)

    means = {"Trait": "MEAN"}
    for name, r in results.items():
        vals = [v for v in r.values() if not np.isnan(v)]
        means[name] = round(float(np.mean(vals)), 4) if vals else float("nan")
    df = pd.concat([df, pd.DataFrame([means])], ignore_index=True)

    print("\n=== Per-trait Pearson r (10-fold CV) ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
