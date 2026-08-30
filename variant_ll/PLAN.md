# First arabidopsis run — execution plan

Companion to [BENCHMARK.md](BENCHMARK.md), which defines the objective, the loss, and
the decision matrix. This is the concrete sequence for the first real run: one
chromosome, individual-level split, training and eval wired end to end.

**Standing recommendation: build and validate the whole pipeline on rice first.**
Rice is already built, tiny (33k SNPs, 383 accessions), and its baselines are already
measured. Every component below can be written and tested against it while the 19 GB
arabidopsis build runs, then pointed at arabidopsis by changing one dataset name. Do
not debug the data layer and the new dataset at the same time.

Rice is also more than plumbing — B1 is **not** its ceiling. B2, a two-parameter
per-site-pair table, already beats it, and by a margin that depends entirely on window
size:

| `half_window` | tokens | B1 | B2 | headroom |
|---:|---:|---:|---:|---:|
| 500 | 167 | 0.748 | 0.693 | 0.055 |
| 5,000 | 1,667 | 0.748 | 0.576 | **0.172** |
| 10,000 | 3,334 | 0.748 | 0.543 | **0.205** |

At hw=500, matching B1 really is about the best available: 81% of rice sites have no
upstream SNP in-window and are capped at B1 by construction. But that is the window
size talking, not the dataset. Split the duty:

* **hw=500** — plumbing. T=167, minutes per epoch, expect to tie B1.
* **hw=5000** — a cheap first real signal. 0.172 bits of *demonstrated* headroom, and
  T=1667 keeps it inside the comfortable memory regime. If the fine-tuned cache
  cannot get below 0.748 here, where a per-site-pair lookup reaches 0.576, that is a
  genuine negative result available before arabidopsis exists.

This benchmark needs **no phenotypes**, so it is unblocked the moment the final VCF
lands — it does not wait on AraPheno alignment.

---

## Chromosome choice: chr4

| chr | length | N bases | N % |
|---:|---:|---:|---:|
| 1 | 30,427,671 | 163,958 | **0.539%** |
| 2 | 19,698,289 | 2,506 | 0.013% |
| 3 | 23,459,830 | 5,966 | 0.025% |
| **4** | **18,585,056** | **3,030** | **0.016%** |
| 5 | 26,975,502 | 10,278 | 0.038% |

**Chr4**: smallest (fastest iteration) and effectively N-free. N content matters more
than usual here — any 6-mer containing `N` tokenizes to `<oov>`, which makes both
allele candidates identical and the site unscoreable. Chr1 has 30–40× the N of the
others and would silently drop windows.

---

## Phase 0 — characterize before configuring (CPU, ~30 min)

Arabidopsis is going to be **~300× denser than rice** (1001G at MAF 0.01 is roughly
1 SNP per 30 bp against rice's 1 per 11.2 kb). Every design constant changes. Measure
them; do not carry rice's numbers over.

Run the rice probe against chr4 and report:

- SNP count, density (bp/SNP), **missing-call rate** (per-variant and overall)
- **B0 / B1 / B2 in bits/SNP** at `half_window ∈ {250, 500, 1000, 2500}`
- % of site-calls with ≥1 upstream SNP in-window
- unique haplotypes per window (the dedup factor — expect it to be ~1 here, unlike rice)
- share of SNP-containing tokens holding ≥2 segregating sites
- share of sites whose reference 6-mer contains `N`
- cache fraction `cs/T` — see the tension below

**Gate:** `B1 − B2` must be a real gap (say ≥ 0.05 bits/SNP). That is the headroom
any model has to work with. If it isn't there at any window size, nothing downstream
can succeed and the problem is the data, not the cache.

---

## Phase 1 — set the config from those numbers

### Window size
Pick the **smallest** `half_window` with ≥90% upstream coverage. At 1 SNP/~30 bp that
is likely `hw = 250–500`, i.e. **T = 83–167 tokens** — trivially cheap, so iteration
is fast and large haplotype batches fit easily. This is the opposite of rice, where
the density forced 20 kb windows.

### Multi-SNP tokens — now a real case, not a rounding error
With λ = 6/D expected SNPs per 6-mer token:

| MAF filter | ~bp/SNP (D) | λ | tokens carrying a SNP | **SNP tokens with ≥2 sites** | cache fraction `cs/T` |
|---|---:|---:|---:|---:|---:|
| 0.01 | 33 | 0.18 | 17% | **8.6%** | ~0.30 |
| 0.05 | 74 | 0.08 | 7.8% | 4.0% | ~0.15 |
| LD-pruned | 200 | 0.03 | 3.0% | 1.5% | ~0.06 |

In rice this was 0.05%. **Measured** on chr4 at hw=500 (3,000 windows, 35,611
SNP-bearing tokens): 10.12% of tokens hold 2 sites, 0.93% hold 3, 0.06% hold 4,
0.01% hold 5 — **20.8% of sites**, not the ~9% predicted above. Dropping them would
bias the evaluation toward isolated SNPs, exactly the regime where the cache is most
accurate, i.e. it would flatter the result.

**Score the token jointly** — enumerate its 2^m candidate 6-mers and make the
autoregressive target the one carrying the haplotype's alleles at all m sites.
Implemented; see BENCHMARK.md §5, which also records why the earlier proposal here
(flip the focal site, pin its co-token neighbours to their true alleles) was wrong:
it is a pseudo-likelihood rather than a chain-rule factorization, and it leaks a
neighbouring allele 1–5 bp away that B1 never sees. Max m observed is 5, so at most
32 candidates against a 4096 vocab — one wider `gather`.

### Missing calls — a mask is now mandatory
`GENO=0.2` admits variants with up to 20% missing genotypes, and
`crop_embed/data/vcf.py` maps missing → ref. Rice's 2.8% overall was tolerable noise;
20% per-variant is not — it corrupts the scoring labels *and* the haplotype the model
is fed. Add a missing mask to `SNPRecord` and exclude those calls from both the loss
and B1's frequency estimates. Alternatively tighten `GENO` for this build, but the
mask is the right fix and it is small.

### The density/speedup tension — worth naming now
Cache speedup is bounded by `T/cs`. At `cs/T ≈ 0.30` the cache recomputes nearly a
third of every window and saves ~3×, not the ~400× in figure 10. That does **not**
invalidate this benchmark — it tests whether the approximation preserves signal, not
whether it is fast — but state it plainly rather than letting the two get conflated.

It also suggests the genuinely interesting sweep: denser variants mean *more* cached
positions, so the approximation gets **more** accurate as it gets **less** fast. There
is a density regime where it is both, and MAF / LD-pruning is the knob that locates
it. That is a follow-up experiment, not part of the go/no-go.

---

## Phase 2 — genotype-level split

`training/common/splits.py::build_split` partitions **phenotyped** samples. This
benchmark uses all genotyped accessions and no traits, so add a genotype-only path
that partitions VCF sample IDs → `splits/arabidopsis_geno_seed42.pt`, same 70/15/15
and seed 42 convention so it sits alongside the existing split files.

**Relatedness is a confound here.** 1001G has strong population structure (relicts,
admixed groups, near-duplicate accessions). A random split puts close relatives on
both sides, and a model can then win by haplotype matching rather than by
understanding sequence. B1 cannot exploit that, so the comparison would flatter the
model. Two cheap mitigations:

1. Record, for each test accession, the genetic distance to its nearest training
   accession. Report the distribution alongside the headline number.
2. Add a **structured split** as secondary eval (hold out a population group), and
   report both. If the random-split win vanishes under the structured split, that is
   the real finding.

Related: with dense SNPs, add a **nearest-neighbour haplotype-copying baseline** —
for each test individual and window, copy the allele from the training individual
with the most similar upstream alleles. That is essentially what imputation does, and
it is the honest "is this interesting" bar. B1 stays the go/no-go gate per
BENCHMARK.md; this is the second gate.

---

## Phase 3 — window sampling

**A chromosome is the sampling frame, not the training set.** At hw=500, chr4 gives
roughly 18k windows, and with ~1,135 accessions at this density nearly every accession
is a unique haplotype — so a full epoch is ~18M window-haplotype items. Too slow to
iterate on.

Draw **~2,000 windows** at random from chr4. That still yields ~2,000 × ~30 sites ×
170 held-out individuals ≈ 10M scored site-calls, far more than needed for a tight
interval on bits/SNP.

**Split on accessions only.** The same 2,000 windows are used for training and for
evaluation; the only held-out axis is the individual. This is what makes the
comparison to B1 meaningful — both the model and the baseline get to know these sites
from the training individuals, and both are scored on new individuals at those sites.

> **Why there is no held-out *window* set.** It was proposed and dropped: the
> comparison would be structurally unfair rather than merely hard. B1 is fit per-site
> from training individuals in any cell, so it gets that site's base rate for free
> even on a window the model never trained on — while the model would have to infer
> the base rate from sequence alone. It would lose there even with a perfect LD
> mechanism, so the result would say nothing about the mechanism. The
> memorization-vs-mechanism question is answered instead by the **permutation
> control** (§Phase 5, and BENCHMARK.md §7): shuffle upstream alleles across
> individuals within a window, and if the loss does not degrade, the model is reading
> a per-site table rather than the individual's haplotype. Same question, fair
> comparison, no extra machinery.

---

## Phase 4 — code, in dependency order

Shapes and signatures are in BENCHMARK.md §5–§6; this is the build order and what is
new versus changed.

| # | file | status | notes |
|---|---|---|---|
| 1 | `crop_embed/data/vcf.py` | **change** | missing mask on `SNPRecord` (Phase 1) |
| 2 | `training/common/splits.py` | **change** | genotype-only split path (Phase 2) |
| 3 | `variant_ll/data.py` | new | `WindowBatch`; chromosome subsetting; window sampling; joint per-token candidates |
| 4 | `variant_ll/baselines.py` | new | B0/B1/B2 + NN-haplotype copying |
| 5 | `variant_ll/loss.py` | new | `window_nll`, cache + exact backends |
| 6 | `CARBON_modules/variant_cache_layers.py` | **change** | `--ckpt-reference` flag on `forward_reference` (BENCHMARK.md §8a; §8b already landed in `dc46caa`) |
| 7 | `variant_ll/eval.py` | new | bits/SNP + the §7 diagnostic slices (`has_upstream`, permutation) |
| 8 | `variant_ll/run.py` | new | CLI, MetricLogger, run records |

Steps 1–5 and 7 are testable on rice today. Only step 3's chromosome subsetting and
the arabidopsis paths need the new VCF.

---

## Phase 5 — run sequence, with gates

| step | what | gate to pass |
|---|---|---|
| 0 | Phase 0 characterization | `B1 − B2` ≥ 0.05 bits |
| 1 | smoke test, 50 windows | shapes/indices assert clean; zero-shot runs |
| 2 | baselines on the real split | B1 and B2 reproduce Phase 0 within noise |
| 3 | zero-shot cache + exact | both finite; cache vs exact gap is the pure approximation error |
| 4 | fine-tune through the cache | val bits/SNP **< B1** |
| 5 | fine-tune exact (subsample) | tells you whether step 4's result is the cache's fault |
| 6 | controls (§7) | permutation degrades; upstream slice carries the win; hw-small ties B1 |

Stop and diagnose at the first gate that fails rather than running the whole sequence.

---

## Phase 6 — what "done" looks like

One table, in bits/SNP on held-out accessions, split by whether the site has upstream
context (the model can only beat B1 on the right-hand column — on the left it is
capped at B1 by construction):

```
                              all sites    no upstream ctx    has upstream ctx
B0 uniform                      1.000           1.000              1.000
B1 per-site marginal                ?               ?                  ?
B2 upstream Markov                  ?          (= B1)                  ?
B3 NN haplotype copy                ?               ?                  ?
Carbon zero-shot (cache)            ?               ?                  ?
Carbon zero-shot (exact)            ?               ?                  ?
Carbon FT (cache)                   ?               ?                  ?
Carbon FT (exact, subsample)        ?               ?                  ?
```

Val while iterating; test once, at the end. Plus: the permutation control, the
nearest-train-neighbour distance distribution, and the structured-split rerun of the
headline row.

---

## Watch-items

* **Don't let dense data hide a broken cache.** With ~30 SNPs per 1 kb window, LD is
  so strong that a model could beat B1 while still passing very little information
  through the cache. The permutation control and the cache-vs-exact gap are what
  distinguish those; they are not optional here.
* **Effective training set size.** Rice's fingerprint dedup collapsed 2.85M
  sample-windows to 55k. At arabidopsis density dedup buys almost nothing, so the
  training set is genuinely large — good for overfitting, but re-derive the per-epoch
  cost before committing to a schedule.
* **Re-derive, don't reuse.** Every number in BENCHMARK.md §3 and §8's timing
  envelope is rice-specific.
