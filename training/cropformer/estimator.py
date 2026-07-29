"""
training/cropformer/estimator.py
--------------------------------
Cropformer (Jie et al., jiekesen/Cropformer) as an sklearn-style regressor so it
drops into the shared harness like the snp_sklearn models: one trait at a time,
fit on train only, predict on val/test.

The architecture is the upstream one — a Conv1d over the SNP vector, a
(sequence-length-1) self-attention block with residual + LayerNorm, then a linear
head — reproduced here (with the upstream's missing `import torch.nn as nn` fixed
and the unused Lightning base dropped) so this module is self-contained and
committable; the external Cropformer/ checkout is gitignored and not imported.

Faithful preprocessing: top-`mic_k` SNPs by mutual information with the trait
(fit on the train partition only -> no leakage), then StandardScaler. A small
internal validation slice drives early stopping on Pearson r.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class _SelfAttention(nn.Module):
    """Upstream Cropformer core: Conv1d -> self-attention (seq len 1) -> linear."""

    def __init__(self, num_heads, input_size, hidden_size, output_dim=1,
                 kernel_size=3, hidden_dropout_prob=0.5, attention_probs_dropout_prob=0.5):
        super().__init__()
        self.attention_head_size = int(hidden_size / num_heads)
        self.query = nn.Linear(input_size, hidden_size)
        self.key = nn.Linear(input_size, hidden_size)
        self.value = nn.Linear(input_size, hidden_size)
        self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.dense = nn.Linear(hidden_size, input_size)
        self.LayerNorm = nn.LayerNorm(input_size, eps=1e-12)
        self.relu = nn.ReLU()
        self.out = nn.Linear(input_size, output_dim)
        self.cnn = nn.Conv1d(1, 1, kernel_size, stride=1, padding=1)

    def forward(self, x):
        h = self.cnn(x.view(x.size(0), 1, -1))
        q, k, v = self.query(h), self.key(h), self.value(h)
        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.attention_head_size)
        probs = self.attn_dropout(torch.softmax(scores, dim=-1))
        ctx = torch.matmul(probs, v)
        hs = self.out_dropout(self.dense(ctx))
        hs = self.LayerNorm(hs + h)
        return self.out(self.relu(hs.view(hs.size(0), -1)))


class CropformerRegressor(BaseEstimator, RegressorMixin):
    """sklearn-compatible single-trait Cropformer."""

    def __init__(self, mic_k=10000, num_heads=8, hidden_dim=128, kernel_size=3,
                 lr=1e-3, batch_size=32, max_epochs=100, patience=5, dropout=0.5,
                 val_frac=0.15, seed=42, device=None):
        self.mic_k = mic_k
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.dropout = dropout
        self.val_frac = val_frac
        self.seed = seed
        self.device = device

    # -- helpers -----------------------------------------------------------
    def _dev(self):
        return self.device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _select(self, X, y):
        V = X.shape[1]
        if self.mic_k and 0 < self.mic_k < V:
            mi = mutual_info_regression(X, y, random_state=self.seed)
            self.feat_idx_ = np.argsort(mi)[-self.mic_k:]
        else:
            self.feat_idx_ = np.arange(V)
        return X[:, self.feat_idx_]

    # -- sklearn API -------------------------------------------------------
    def fit(self, X, y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        dev = self._dev()

        Xs = self._select(X, y)
        self.scaler_ = StandardScaler().fit(Xs)
        Xs = self.scaler_.transform(Xs).astype(np.float32)
        input_size = Xs.shape[1]

        # Standardize the target so MSE is well-scaled (stabilizes training and
        # makes r2/mse comparable to the other baselines); undone in predict.
        # Pearson is scale/shift-invariant, so early-stopping is unaffected.
        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std()) or 1.0
        ys = ((y - self.y_mean_) / self.y_std_).astype(np.float32)

        # internal split for early stopping (guard tiny n)
        if len(y) >= 8 and self.val_frac > 0:
            xt, xv, yt, yv = train_test_split(Xs, ys, test_size=self.val_frac,
                                              random_state=self.seed)
        else:
            xt, yt, xv, yv = Xs, ys, Xs, ys

        model = _SelfAttention(self.num_heads, input_size, self.hidden_dim,
                               kernel_size=self.kernel_size,
                               hidden_dropout_prob=self.dropout,
                               attention_probs_dropout_prob=self.dropout).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.1, patience=10)
        lossf = nn.MSELoss()

        xt_t = torch.from_numpy(xt).to(dev)
        yt_t = torch.from_numpy(yt).view(-1, 1).to(dev)
        xv_t = torch.from_numpy(xv).to(dev)

        n = xt_t.size(0)
        best_r, best_state, bad = -np.inf, None, 0
        for _ in range(self.max_epochs):
            model.train()
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, self.batch_size):
                idx = perm[s:s + self.batch_size]
                opt.zero_grad()
                loss = lossf(model(xt_t[idx]), yt_t[idx])
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                pv = model(xv_t).cpu().numpy().ravel()
            r = _pearson(yv, pv)
            sched.step(r)
            if r > best_r:
                best_r, bad = r, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model_ = model
        self.input_size_ = input_size
        self.best_params_ = {"mic_k": int(len(self.feat_idx_)), "num_heads": self.num_heads,
                             "hidden_dim": self.hidden_dim, "lr": self.lr,
                             "best_val_r": round(float(best_r), 4)}
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)[:, self.feat_idx_]
        X = self.scaler_.transform(X).astype(np.float32)
        dev = self._dev()
        self.model_.eval()
        with torch.no_grad():
            out = self.model_(torch.from_numpy(X).to(dev)).cpu().numpy().ravel()
        return out * self.y_std_ + self.y_mean_        # undo target standardization


def _pearson(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    if y.std() == 0 or yhat.std() == 0:
        return 0.0
    return float(np.corrcoef(y, yhat)[0, 1])


def add_cropformer_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mic-k", type=int, default=10000,
                   help="top-K SNPs by mutual info with the trait (0 = use all)")
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.5)


def make_estimator(args) -> CropformerRegressor:
    return CropformerRegressor(
        mic_k=args.mic_k, num_heads=args.num_heads, hidden_dim=args.hidden_dim,
        kernel_size=args.kernel_size, lr=args.lr, batch_size=args.batch_size,
        max_epochs=args.max_epochs, patience=args.patience, dropout=args.dropout,
        seed=args.seed,
    )
