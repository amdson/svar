# `variant_ll/` — the variant-cache viability benchmark

**One question, one number.** Fine-tune Carbon through the variant cache to predict
the allele an individual carries at each SNP site, and score it in **bits per SNP on
held-out individuals** against the naive independent-sites baseline (per-site allele
frequency). Everything below exists to make that number trustworthy and to make a
negative result *diagnostic* rather than ambiguous.

Measured on rice (`sativas413_msu7_final.vcf`, 33,291 SNPs × 383 accessions), the
bar is:

| model | bits / SNP (held-out individuals) |
|---|---|
| B0 uniform | 1.000 |
| **B1 per-site marginal allele frequency** (the baseline you asked for) | **0.748** |
| B2 nearest-upstream-SNP Markov, 20 kb window (LD headroom) | 0.543 |

If the fine-tuned cache lands below 0.748 it is passing real, individual-specific
information. If it lands near 0.543 it is doing genuinely well. If it lands at 0.748
it has learned site marginals and nothing else, and if it lands above 1.0 something
is broken. The **exact-forward control** (§7) tells you which of those is the cache's
fault versus the objective's.

---

## 1. Why not just reuse `variant_ar/`

`variant_ar/loss.py::snp_next_token_nll` already fine-tunes Carbon through the cache
on a next-token objective, and its docstring explains the choice: in a causal LM the
SNP token is predicted from positions strictly to its left, so it reasoned that
"predict the SNP token" carries no variant signal, and switched the target to the
token *after* each SNP.

That reasoning is right about the cache **as currently invoked** and wrong as a
general statement, and the difference is the whole benchmark:

* The logits that predict the token at position `p` come from position `p-1`. The
  cache only emits logits at positions it recomputes, and `variant_ar` only puts the
  SNP tokens themselves in the cache — so `p-1` is frozen at its reference value and
  is indeed allele-independent. **Fix: put `p-1` in the cache set too.** Then `p-1`
  attends to the recomputed upstream SNP tokens and its logits *do* depend on the
  individual's upstream alleles. That is linkage disequilibrium, and it is the only
  channel by which any model can beat a per-site marginal.
* The `variant_ar` target (the token *following* a SNP) is usually a pure-reference
  6-mer. Its NLL is dominated by how well Carbon knows the reference sequence, not by
  allele information, and there is no marginal-allele baseline it can be compared to.
  It cannot answer the question you are asking.

So: new sibling module, same spirit, different objective. `variant_ar/` stays as-is
(it is what figure `05_ar_curves.png` reports).

---

## 2. The objective

For each SNP site *j* in a window and each individual *i*:

```
loss_ij = -log2 P(allele_ij | reference window, alleles at sites < j for individual i)
```

**Renormalize over the token's candidates.** Carbon's DNA mode is a fixed
contiguous 6-mer split, so the token containing site *j* is a 6-mer whose
non-segregating bases are shared across alleles. Score the model by restricting its
4096-way token distribution to the 6-mers reachable by varying that token's
segregating sites, and renormalizing. The autoregressive target is just the 6-mer
with this haplotype's alleles filled in:

```
cand   = [id6(kmer with allele assignment c at the token's m sites)
          for c in range(2**m)]
logp   = logits[pred_row, cand] - logsumexp(logits[pred_row, cand])
target = sum(allele[b] << b for b in range(m))     # the true assignment
```

For the usual *m* = 1 that is the two-candidate REF/ALT case. See "Multi-SNP
tokens" in §5 for why *m* > 1 must be scored jointly like this.

This is not a convenience — it is what makes the comparison fair. Without it the LM
must spread mass over 4096 tokens while B1 puts all its mass on the alleles, and it
also must "pay" for the token's non-segregating bases that the baseline gets free.
Renormalizing cancels both, leaves a clean 2^m-class problem, and makes the units
**bits per SNP**, directly comparable to B1.

**The sum is a real haplotype log-likelihood.** Tokens are scored left-to-right,
each conditioned on the true upstream alleles, so `Σ loss` is a chain-rule
factorization of `P(haplotype_i | reference)` — a token carrying *m* sites
contributes their joint, which is exactly their chain-rule product. B1's
`Σ_j H(p_j)` is the same quantity under an independent-sites model. They are on the
same scale by construction. Divide by the *site* count, not the token count, to keep
the unit **bits per SNP**.

**Positions in the cache.** For every SNP token position `p` in the window:

```
cache_idx = unique( {p : SNP token positions} ∪ {p-1 : predictor positions} )
```

`torch.unique` (sorted, deduplicated) is required — `variant_cache_layers.forward`
does `keep[pos_q] = False`, which double-counts if `pos_q` has duplicates. Order does
not matter (the causal masks are built from actual positions), but uniqueness does.
Cache size is ≈ 2 × SNPs-per-window: about 6 tokens against a 3,334-token window at
`half_window=10000`.

**No leakage — worth stating because it is the kind of thing that silently ruins a
benchmark.** The frozen reference stream contains the *reference* allele at every
SNP site. Predictor position `p-1` cannot see any of them:

* Position `p` and everything downstream is excluded by the causal mask
  (`causal_c = pos_ref <= pos_q`), and the reference stream is itself causal, so the
  frozen state at position `m` depends only on positions `≤ m`.
* Upstream SNP tokens are in `cache_idx`, so `keep[pos_q] = False` removes their
  *reference* keys from the cross branch entirely; the only version of them the model
  sees is the recomputed one carrying the individual's true alleles.

**What the frozen-reference approximation actually costs here.** Information reaches
`p-1` from an upstream variant token via a direct attention edge at each layer (the
self branch). It cannot route *through* an intermediate non-cached position, because
those are pinned to reference values. With 28 layers and both the SNP tokens and the
predictors in the cache, the direct paths are the important ones — but that is a
hypothesis, and §7's exact control is what tests it.

---

## 3. The configuration that decides whether this experiment can work at all

Rice 44K is array data: **1 SNP per 11.2 kb**. At the repo's default
`half_window=500`, almost every window holds exactly one SNP, so almost every site
has *no upstream variant context* and the model's prediction is identical for every
individual. At that setting the benchmark is guaranteed to return "ties B1" no matter
how well the cache works.

Measured on the rice VCF, replicating `SNPWindowPartitioner`'s greedy assignment
(`buffer=0`), with B2 = nearest-upstream-SNP Markov table fit on train individuals
and scored on held-out ones (add-0.5 smoothing, 80/20 split):

| `half_window` | window bp | tokens | windows | mean SNPs/win | % site-calls with upstream context | **B2 bits/SNP** | unique haplotypes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 1,000 | 167 | 27,081 | 1.23 | 18.7% | 0.693 | 61,967 |
| 2,500 | 5,000 | 834 | 18,322 | 1.82 | 45.0% | 0.613 | — |
| 5,000 | 10,000 | 1,667 | 14,317 | 2.33 | 57.0% | 0.576 | 55,416 |
| **10,000** | **20,000** | **3,334** | **10,596** | **3.14** | **68.2%** | **0.543** | **54,666** |
| 20,000 | 40,000 | 6,667 | 8,854 | 3.72 | 73.4% | 0.527 | 54,804 |

Two things fall out of this table:

1. **Use `half_window=10000` as the headline setting** (`--max-length 8192`; Carbon's
   `max_position_embeddings` is 8192, so `half_window ≤ 24000` is the hard ceiling).
   `half_window=500` is a null-result trap — keep it only as a deliberate negative
   control, where the *expected* result is "ties B1".
2. **Cost is nearly flat in window size.** Unique haplotypes stay ~55k across the
   whole range while the LD signal more than triples, because bigger windows mean
   fewer windows with more haplotypes each. There is no compute argument for small
   windows here.

---

## 4. Splits — by individual, not by window

`variant_ar` splits by *window*, which is correct for its objective and fatal for
this one: B1 is a per-site frequency estimated from training individuals, so held-out
data must be **new individuals at the same sites**.

Reuse the committed split — `splits/rice_seed42.pt`, 269/57/57 by sample ID, built by
`training.common.splits.build_split`, identical to what every other model in
`training/` uses (no cross-model leakage):

```python
from training.common.splits import get_or_build_split
split = get_or_build_split("rice")            # .train_sample_ids / .val_sample_ids / .test_sample_ids
```

Map sample IDs to VCF column order via `UniqueWindowDataset.samples` /
`load_snps_from_vcf`'s returned `samples` list. Tune on val, report test once.

---

## 5. Data layer — batch by window, weight by multiplicity

The unit of work is a **window**, not a (sample, window) pair. All individuals share
the same *sites* in a window and differ only in *alleles*, so one reference forward
serves every haplotype in that window and they batch along the cache's `N` dimension
in a single call. `variant_ar` runs `N=1` and re-does the reference forward per
haplotype; do not copy that.

Within a window, individuals with the same allele pattern produce byte-identical
sequences (this is exactly `crop_embed/fingerprint.py`). Deduplicate and carry an
integer **multiplicity** so the loss is still the population average:

```
L = Σ_windows Σ_haplotypes  count[h] · Σ_sites  nll[h, site]
    ─────────────────────────────────────────────────────────
                Σ_windows Σ_haplotypes count[h] · n_sites
```

Counts must be computed **within the split** (train counts from train individuals
only) or you leak. At `half_window=10000` a full pass costs 10,596 reference forwards
+ ~54.7k haplotype recomputes instead of 269 × 10,596 ≈ 2.85M full forwards.

### `variant_ll/data.py`

```python
@dataclass
class WindowBatch:
    ref_ids:     LongTensor   # (T,)      tokenized all-reference window
    site_tok:    LongTensor   # (S,)      1 + (pos - w_start) // 6   (see _build_snp_mask_kmer)
    pred_row:    LongTensor   # (S,)      row of (site_tok - 1) within cache_idx
    cache_idx:   LongTensor   # (C,)      unique(site_tok ∪ site_tok-1), sorted
    cand_ids:    LongTensor   # (K, Q)    candidate 6-mer ids per token, right-padded
    cand_mask:   BoolTensor   # (K, Q)    real candidate slots (Q = 2**max m)
    tok_nsite:   LongTensor   # (K,)      m — sites in the token; the bits/SNP weight
    hap_ids:     LongTensor   # (N, C)    token ids at cache positions per haplotype
    hap_target:  LongTensor   # (N, K)    index of the true candidate, bit b = site b
    hap_allele:  BoolTensor   # (N, S)    1 = carries alt (baselines only)
    hap_count:   FloatTensor  # (N,)      multiplicity within this split
    has_upstream: BoolTensor  # (K,)      ≥1 SNP token earlier in this window  → the key diagnostic slice
```

Build it from machinery that already exists:

* `crop_embed.data.vcf.load_snps_from_vcf` → `SNPRecord(pos, ref_byte, alt_byte, gt_alts)`
* `crop_embed.partitioner.SNPWindowPartitioner` → `.windows`, `.window_snp_indices[w]`
  (the per-window site list — this is what you want, not a per-haplotype fingerprint)
* `crop_embed.dataset.UniqueWindowDataset.extract_sequence(fp)` with
  `fp = (chrom, start, end, frozenset())` for the reference window
* `crop_embed.embedder._build_snp_mask_kmer`'s index formula for `site_tok`

**Do not re-tokenize per haplotype.** Tokenize the reference window once, then patch
token ids at the SNP positions directly. The 6-mer block is contiguous and
base-4 ordered:

```python
KMER_BASE, ORDER = 151672, "ATCG"       # verified against all 4096 6-mers
def id6(kmer): 
    v = 0
    for ch in kmer: v = v * 4 + ORDER.index(ch)
    return KMER_BASE + v
```

Build the 4096-entry table once by tokenizing every 6-mer at startup and `assert` it
matches this formula — cheap insurance against a checkpoint/tokenizer change.

Tokenizer facts, all verified against `HuggingFaceBio/Carbon-500M`:

| fact | value |
|---|---|
| `<dna>` marker | id 151669, exactly 1 prefix token, supplied as literal text with `add_special_tokens=False` |
| DNA 6-mer ids | 151672 – 155767, contiguous, base-4 with alphabet order **`ATCG`** (not ACGT) |
| any 6-mer containing `N` | collapses to `<oov>` = 151671 |
| lowercase | uppercased by the tokenizer (soft-masked FASTA is fine) |
| trailing partial 6-mer | right-padded with `A`, still occupies one token |
| batch padding | right, pad id 151643, `attention_mask` correct |

Consequences to handle: **drop sites whose reference 6-mer contains `N`** (both
candidates collapse to `<oov>` and the site is unscoreable) and log how many were
dropped. Windows near a chromosome edge get `N`-padding from `extract_sequence`;
because `SNPWindowPartitioner` centres each window on its first SNP, no site token
ever lands at index 0 or 1, so the `<dna>` marker is never a predictor in practice.

**Multi-SNP tokens — score the token jointly.** When *m* segregating sites share a
6-mer, enumerate all 2^m candidate 6-mers and make the target the one carrying the
haplotype's alleles at all *m*. Rice sees this at ~0.05% of sites; arabidopsis chr4
at hw=500 sees **20.8%** — 10.1% of SNP-bearing tokens hold 2 sites, 0.9% hold 3,
max observed 5, so at most 32 candidates out of a 4096 vocab: one wider `gather`.

An earlier draft of this file proposed instead scoring each site with its co-token
neighbours pinned to that haplotype's *true* alleles — `cand_ids → (N, S, 2)`,
flipping only the focal site. **Do not do that.** It scores
`P(s_j | s_other = true, upstream)`, a pseudo-likelihood whose terms do not compose
into a joint, so `Σ loss` stops being `log P(haplotype)` and the comparison to B1's
`Σ_j H(p_j)` quietly loses its footing. It also hands the model the true allele at a
site 1–5 bp away — about the strongest LD anywhere in the genome — that B1 gets
nothing of, so it would flatter the model on exactly the sites it was introduced to
rescue.

The joint has neither problem: it is the exact chain-rule contribution of those *m*
sites and conditions only on strictly upstream tokens. It also needs no per-site
decomposition for the diagnostics, since `has_upstream` is a property of the token —
sites sharing a token are predicted simultaneously and give each other no context.

---

## 6. Loss and baselines

### `variant_ll/loss.py`

```python
def window_nll(model, batch, backend="cache", chunk=32):
    """Returns (nll_bits_sum, n_calls) weighted by hap_count. Keeps grad."""
    if backend == "cache":
        out = model(batch.ref_ids,
                    variant_positions=batch.cache_idx,
                    variant_input_ids=batch.hap_ids)        # logits (N, C, V)
        logits = out.logits.index_select(1, batch.pred_row) # (N, S, V)
    else:                                                   # "exact" control
        alt_ids = batch.ref_ids.expand(N, T).clone()
        alt_ids[:, batch.cache_idx] = batch.hap_ids
        logits = model(input_ids=alt_ids).logits.index_select(1, batch.site_tok - 1)

    sel  = logits.gather(-1, batch.cand_ids.expand(N, K, Q))     # (N, K, Q)
    sel  = sel.float().masked_fill(~batch.cand_mask, -inf)       # ragged over m
    logp = torch.log_softmax(sel, dim=-1) / math.log(2)          # bits
    nll  = -logp.gather(-1, batch.hap_target.unsqueeze(-1)).squeeze(-1)
    w    = batch.hap_count[:, None] * batch.tok_nsite[None, :]   # bits per *SNP*
    return (nll * w).sum(), w.sum()
```

Both backends consume the identical `cand_ids` / `hap_allele`, so their bits/SNP are
directly comparable — that is the whole point of the control.

### `variant_ll/baselines.py`

```python
def marginal(hap_train, count_train, alpha=0.5) -> p          # B1, add-alpha smoothed
def markov_upstream(hap_train, count_train, alpha=0.5) -> tbl  # B2, nearest upstream SNP in-window
```

Smoothing matters: sites monomorphic in train would otherwise assign probability 0 to
a held-out minor allele and produce an infinite baseline loss. Keep those sites (with
`alpha=0.5`, Jeffreys) and report how many there are. `pretrained_eval/baselines.py`
already owns the nats↔bits/nt conventions — follow them so numbers stay comparable
across the repo.

---

## 7. The controls that make a negative result mean something

Run all of these. They cost little and they are the difference between "it didn't
work" and "here is *why* it didn't work".

| control | what it isolates |
|---|---|
| **Zero-shot cache** (no fine-tuning) | where Carbon starts. Expect it to be *worse* than 1 bit — it is trained on reference genomes and will over-predict the REF allele. |
| **Exact-forward fine-tune** (same objective, full alt-window forward) | the ceiling the cache is approximating. `FT-exact ≪ FT-cache` ⇒ the frozen-reference approximation is what's broken. |
| **Sites with vs without upstream context** (`has_upstream`) | on no-upstream sites the model *cannot* beat B1; it can only match it by memorizing marginals. Beating B1 must come from the upstream slice or it isn't LD. |
| **Upstream-allele permutation** | shuffle upstream alleles across individuals within a window and re-evaluate. If loss doesn't degrade, the cache is passing *no* individual-specific information — the sharpest single test of viability. |
| **`half_window=500` negative control** | expected to tie B1 by construction (§3). If it doesn't tie, suspect a bug. |
| **Mixture with B1** (`λ·model + (1-λ)·B1`, λ tuned on val) | detects weak-but-real complementary signal when the model loses outright. Post-hoc from saved per-site probabilities; costs nothing. |

### Decision matrix

| FT-cache vs B1 | FT-exact vs B1 | read |
|---|---|---|
| better | better | **cache is viable** — proceed to phenotype work |
| ties | better | approximation is destroying the signal → fix the cache (more positions cached? multi-hop?) before anything else |
| ties | ties | objective/model/data is the problem, not the cache — the cache is faithfully reproducing a model that has nothing to say |
| worse | — | bug. Check leakage, candidate construction, position indexing |

---

## 8. Compute and memory — measured, not estimated

On this cluster's A100-80GB, `load_carbon_variant_cache("HuggingFaceBio/Carbon-500M")`,
`cs=8`, bf16:

| T (tokens) | N (haplotypes) | grad | time | peak |
|---:|---:|---|---:|---:|
| 3,334 | 1 | no | 277 ms | 3.0 GiB |
| 3,334 | 64 | no | 341 ms | 2.9 GiB |
| 3,334 | 1 | yes | 682 ms | 34.6 GiB → **5.1 GiB** |
| 3,334 | 64 | yes | 898 ms | 60.7 GiB → **7.9 GiB** |
| 3,334 | 256 | yes | 1,314 ms | OOM → **24.5 GiB** |

**Inference is already excellent**: going 1 → 64 haplotypes costs +23% time and *no*
extra memory. A full evaluation pass at `half_window=10000` (10,596 windows,
~5.2 haplotypes each) is roughly **50 minutes on one A100**.

**Training was the problem, from two distinct causes.** The right-hand column above
is after both fixes below; §8b is already applied to
`CARBON_modules/variant_cache_layers.py`, §8a is a flag the runner needs to set.

### 8a. The reference forward's retained activations — use gradient checkpointing

This is purely a memory-implementation problem, not a question about whether the
gradient is needed. `carbon_layers.eager_attention_forward` saves, per layer, a
`(1, 16, T, T)` score tensor *and* a float32 softmax copy. At T=3334 across 28 layers
that is `28 × 16 × 3334² × (2 B + 4 B) ≈ 30 GB` — the entire reason grad-on peaks at
34.6 GiB with a single haplotype.

Per-layer `torch.utils.checkpoint` on `forward_reference` fixes it exactly. Measured
at T=3334, bf16, gradient compared against unmodified eager on a mid-stack parameter
(`encoder.layers.3.mlp.down_proj.weight`):

| reference forward | peak (N=1) | peak (N=64) | gradient error |
|---|---:|---:|---|
| eager (current) | 34.6 GiB | 60.7 GiB | — (baseline) |
| **+ gradient checkpointing** | **5.0 GiB** | **29.9 GiB** | **0.0 — exact** |
| SDPA + additive mask | 7.2 GiB | 34.4 GiB | 6.2e-2 (kernel numerics) |
| detached reference | 2.9 GiB | 29.4 GiB | **9.6e-1** |

**Do not detach the reference stream.** It is tempting — the forward pass is
bit-identical to deployment, and parameters still get gradients through the per-layer
`input_layernorm → k_proj/v_proj` applied to `ref_input`. But what it drops (the
gradient flowing *through* the reference residual stream) turns out to be ~96% of the
gradient on mid-stack parameters. That is a different gradient, not a small bias.

Checkpointing costs ~1.4× time at N=64 (1,030 ms vs 740 ms). The N=1 measurement came
out at 6.4 s, which is anomalous against the N=64 figure and unexplained — re-measure
before relying on it; it does not change the recommendation.

SDPA is a reasonable second option (7.2 GiB, and *faster* than eager at 550 ms) and
composes with checkpointing, but it changes numerics slightly, so
`model_dev/test_carbon_variant_cache.py`'s identity probe would need its tolerance
revisited. Checkpointing is exact and leaves that test untouched.

### 8b. The reference K/V were silently expanded to batch N — **fixed**

`VariantCacheCarbonEncoder.forward` kept `k_r`/`v_r` at batch 1 and relied on
broadcasting (the docstring said so explicitly), but `torch.matmul` on batch dims
`(N,h) × (1,h)` expands the reference K/V and autograd **saves the expanded result**:
`2 × N × h × T × d × 2 B × 28 layers`, measured at 218 MB per haplotype at T=1667 and
399 MB at T=3334 (predicted 191 / 382 MB — within 10%). That defeated the design
intent of keeping the reference at batch 1, and it is why batching haplotypes OOM'd.

Fixed in `_cross_attention_with_lse`: the N scenarios fold into the *query* axis —
`q_v (N,h,cs,d) → (1,h,N·cs,d)` — so both matmul operands are batch-1 and nothing
expands. The head dim stays in the batch position (that is what lines up with the
reference K/V), and the `(1,1,cs,T)` bias broadcasts over an `(h,N,cs,T)` view of the
scores rather than being materialized. Self branch untouched (already `(N,h,·)` on
both sides).

Verified — peak memory, T=3334, bf16, gradient on:

| reference fwd | N | before | after |
|---|---:|---:|---:|
| eager | 64 | 61.1 GiB | 38.7 GiB |
| eager | 256 | OOM | 55.2 GiB |
| **checkpointed** | **64** | **30.3 GiB** | **7.9 GiB** |
| **checkpointed** | **256** | **OOM** | **24.5 GiB** |

Checkpointing (8a) and the fold (8b) attack different tensors and compose: together
they take N=64 from 60.7 GiB to 7.9 GiB (**7.7×**), and N=256 now fits where it
previously OOM'd. Throughput is unchanged (N=64: 917 → 895 ms).

Math is unchanged: forward logits are bit-identical, and gradients over *all*
parameters agree to 4e-7 relative in fp32 (pure reduction order). In bf16 the same
comparison reads ~5e-3, which is bf16 epsilon (2⁻⁸ ≈ 3.9e-3), not a discrepancy.
`python -m model_dev.test_carbon_variant_cache` passes unchanged (efficient vs
bruteforce 6.2e-6, identity probe 7.6e-6).

Inference is unaffected either way — under `no_grad` nothing is saved, so the
broadcast was already free (N=256: 4.04 GiB before and after).

**Still worth doing:** chunk `N` per window anyway, so a window with an unusually
large haplotype count can't blow up. With the fix the chunk can be much larger; hoist
the reference forward out of the chunk loop so it runs once per window.

### Practical training envelope

`half_window=10000`, bf16, checkpointed reference, N chunked at 32: ~1.0 s per window
step, 10,596 windows ⇒ **~3 h/epoch on one A100**. Three epochs plus val evaluation
fits in a single `gpu-a100` allocation. (Rice figures; re-derive for arabidopsis,
where far denser SNPs mean more haplotypes and a larger cache set per window.)

The exact-forward control is far more expensive (T=3334, B=1, grad: 486 ms and
37 GiB; B=4 OOMs). Run it zero-shot on everything, and fine-tune it only on a
subsample or at `half_window=5000`.

---

## 9. Module layout

```
variant_ll/
├── BENCHMARK.md      # this file
├── __init__.py
├── data.py           # WindowBatch + builder (§5)
├── loss.py           # window_nll, both backends (§6)
├── baselines.py      # B0/B1/B2 (§6)
├── eval.py           # bits/SNP + the §7 diagnostic slices
└── run.py            # CLI: fine-tune + evaluate
```

Reuse without modification: `crop_embed.data.vcf`, `crop_embed.partitioner`,
`crop_embed.dataset.UniqueWindowDataset`, `crop_embed.fingerprint`,
`crop_embed.embedder._build_snp_mask_kmer`, `CARBON_modules.load_carbon_variant_cache`
/ `load_carbon_local`, `crop_embed.MetricLogger` + `metrics_path_for`,
`training.common.splits`. Net-new is only `variant_ll/` itself.

Follow the repo's run conventions: JSONL metric sidecar next to `--output`, optional
wandb, artifacts under `$SVAR_SCRATCH/runs/` (`training/DESIGN.md`).

---

## 10. Runbook

```bash
source env.sh          # HF_HOME etc. → scratch, off the home quota

# 0. Baselines only — no GPU, seconds. Establishes the bar before any training.
python -m variant_ll.baselines --dataset rice --half-window 10000 --split splits/rice_seed42.pt

# 1. Smoke test — 50 windows, asserts shapes/indices and that zero-shot runs.
python -m variant_ll.run --dataset rice --half-window 10000 --limit 50 \
    --backend cache --epochs 0 --max-length 8192

# 2. Zero-shot on val, both backends. cache vs exact here is pure approximation error.
python -m variant_ll.run --dataset rice --half-window 10000 --epochs 0 \
    --backend cache --eval-split val --output $SVAR_SCRATCH/runs/vll_zeroshot_cache.pt
python -m variant_ll.run --dataset rice --half-window 10000 --epochs 0 \
    --backend exact --eval-split val --output $SVAR_SCRATCH/runs/vll_zeroshot_exact.pt

# 3. The actual benchmark.
python -m variant_ll.run --dataset rice --half-window 10000 --epochs 3 \
    --backend cache --precision bf16 --ckpt-reference --hap-chunk 32 \
    --lr 1e-5 --output $SVAR_SCRATCH/runs/vll_ft_cache.pt

# 4. Controls (§7).
#    --half-window 500                → expected to tie B1
#    --permute-upstream               → expected to degrade if the cache works
#    --backend exact --limit 2000     → the ceiling, on a subsample
```

`sbatch` per `run_overnight.sbatch`: `--partition=gpu-a100 --gres=gpu:1 --mem=64G`,
sourcing the `svar` conda env and `env.sh`.

**Report**: bits/SNP on val (and test, once) for {B0, B1, B2, zero-shot cache,
zero-shot exact, FT cache, FT exact}, each split into the `has_upstream` and
no-upstream slices, plus the permutation control and the mixture-λ.

---

## 11. Things that will bite

* **Dedup shrinks the effective training set.** ~54.7k unique window-haplotypes, not
  2.85M sample-window pairs. Multiplicity weighting restores the population *average*
  but adds no diversity, and a 500M model will overfit. Low LR, early-stop on val,
  and consider LoRA or freezing the lower layers.
* **`max_length`.** `variant_ar` defaults to 2048, which would silently truncate a
  3,334-token window. Set it ≥ T+1; Carbon caps at 8192.
* **2.83% of rice calls are `./.`** and `load_snps_from_vcf` maps missing → ref
  (`gt_alts` also keeps only the first allele; 0.05% hets). Both baseline and model
  see the same corrupted labels so the comparison stays fair, but it is a real noise
  floor and a real input corruption. Cleanest fix is a missing-mask on `SNPRecord`
  so those calls are excluded from scoring and from B1's frequency estimates — small,
  well-scoped, and worth doing if the result comes out marginal.
* **Rice array data is the cache's worst case.** 1 SNP per 11.2 kb means most of every
  window is invariant. If rice gives an ambiguous answer, the informative rerun is
  **arabidopsis 1001 Genomes** (`datasets/arabidopsis/`, recipe written, not yet
  built): resequencing density puts ~10 SNPs in a 1 kb window, which is both the
  regime the variant cache is *for* and where B1 has far more headroom to lose.
* **`model_dev/test_carbon_variant_cache.py` line 92** uses `dna_lo = 151669` as the
  start of the DNA block; the 4096-token block actually starts at 151672 (151669 is
  the `<dna>` marker, 151671 is `<oov>`). Harmless there — it only needed "some other
  DNA token" — but do not copy that constant into `variant_ll/`.
