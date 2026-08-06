"""
training/snp_gated/model.py
---------------------------
GatedRidge — a sequence-conditioned ridge regression.

The classic RR-BLUP / ridge model gives every SNP a free effect β_j fit from
labels. GatedRidge keeps that linear predictor but multiplies each SNP column by
a per-SNP gate a_j predicted from that SNP's *window embedding* (a frozen Carbon
representation of its genomic neighborhood):

    ŷ_i = (X_i ⊙ a) β,     a_j = exp(φ(E_j))

φ is a small MLP over the (frozen) per-SNP embedding E_j. Because `a` rescales the
columns and β is L2-penalized, this is exactly *differential shrinkage*: the
original-units penalty on SNP j is λ·(β_j / a_j)², so a SNP whose neighborhood
Carbon flags as important (large a_j) is shrunk less. It is the learned,
sequence-driven generalization of annotation-informed GBLUP (GFBLUP / BayesRC).

Two design points make it a genuine *prior toward ordinary ridge*, not just a
lucky init:
  1. exp form + **zero-initialized last layer** ⇒ φ(E)=0 ⇒ a≡1 exactly at step 0,
     so training starts as plain ridge.
  2. a log-space penalty ``gate_penalty`` = mean (log a)² pulls a back toward 1
     throughout training; λ_gate→∞ pins a≡1 and recovers ridge exactly, λ_gate→0
     frees the gate. That single dial interpolates to the baseline.

The predictor stays linear in X (given a), so after training `a_j` is directly
readable as a Carbon-informed learned SNP importance.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GatedRidge(nn.Module):
    """Sequence-gated ridge:  ŷ = (X ⊙ a) β,  a_j = exp(φ(E_j)).

    Parameters
    ----------
    snp_emb   : (n_snp, D) frozen per-SNP embedding (its window's Carbon vector),
                aligned to the SNP-matrix column order. Registered as a buffer.
    n_traits  : regression outputs.
    hidden    : width of the gate MLP's single hidden layer.
    per_trait : if True the gate is per (SNP, trait) — a separate importance
                surface per trait; if False (default) one shared gate per SNP.
    """

    def __init__(
        self,
        snp_emb: torch.Tensor,
        n_traits: int,
        hidden: int = 256,
        per_trait: bool = False,
    ) -> None:
        super().__init__()
        n_snp, D = snp_emb.shape
        self.register_buffer("E", snp_emb.float())      # (n_snp, D) frozen
        self.per_trait = per_trait
        self.gate = nn.Sequential(
            nn.Linear(D, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_traits if per_trait else 1),
        )
        # Zero-init the last layer ⇒ φ(E)=0 ⇒ a=exp(0)=1 exactly at init.
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.beta = nn.Linear(n_snp, n_traits)          # ridge effects (rescaled units)

    def log_gate(self) -> torch.Tensor:
        """log a = φ(E): (n_snp,) shared, or (n_snp, n_traits) if per_trait."""
        z = self.gate(self.E)
        return z if self.per_trait else z.squeeze(-1)

    def gates(self) -> torch.Tensor:
        """The multiplicative gate a = exp(φ(E)); same shape as log_gate()."""
        return torch.exp(self.log_gate())

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: (B, n_snp) dosage → (B, n_traits)."""
        a = self.gates()
        if self.per_trait:
            # effect_{j,t} = a_{j,t} · β_{j,t}; β.weight is (T, n_snp), a is (n_snp, T).
            W = self.beta.weight * a.t()                # (T, n_snp)
            return X @ W.t() + self.beta.bias
        return self.beta(X * a)                         # shared gate: rescale columns

    def gate_penalty(self) -> torch.Tensor:
        """mean (log a)² — the prior pulling the gate toward 1 (ordinary ridge)."""
        return self.log_gate().pow(2).mean()
