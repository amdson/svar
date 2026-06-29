# Figure guide

Short explanation of every figure in [figs/](figs/). Figures 01–05 come from
[render_all.ipynb](render_all.ipynb); 06–07 from [vc_explainer.ipynb](vc_explainer.ipynb);
08–09 from [fingerprint_explainer.ipynb](fingerprint_explainer.ipynb); 10 from
[forward_calls.ipynb](forward_calls.ipynb); 11–12 from
[e2e_explainer.ipynb](e2e_explainer.ipynb).

## From `render_all.ipynb`

**[01_window_sweep.png](figs/01_window_sweep.png) — window length tradeoff.**
Three panels vs window length L (2 bp → 1 Mb, log x). *Unique windows* is roughly
flat (~53 k) until L≈10 kb then climbs to ~100 k; *mean tokens/window* grows
linearly (~L/6, Carbon 6-mer); *unique variants/window* rises 2 → 135. Takeaway:
short windows keep each window near-reference (few haplotypes, cheap to embed);
long windows pack more co-segregating SNPs, so distinct haplotypes — and tokens —
explode.

**[02_diff_curves.png](figs/02_diff_curves.png) — head-training diffs.** Train and
val mean MSE/PCC over training steps for the three window-aggregation variants
(absolute / centered / refdelta) at a fixed config (linear head, mean pool,
warm-started standardizer) on the `carbon500m_hw500` cache. Shows which variant
trains lower-error / higher-correlation and how train vs val diverge.

**[02_per_trait_pcc.png](figs/02_per_trait_pcc.png) — per-trait val PCC.** The same
three variants, but val Pearson correlation for individual traits (Seed length,
Flowering time at Arkansas, Amylose content). Surfaces per-trait differences the
mean curve hides.

**[03_umap_grid.png](figs/03_umap_grid.png) — preprocessing UMAPs.** 2-D UMAP of
the per-sample embedding under each preprocessing option, colored by inferred
subpopulation (SNP PC1 quartile). Panels span Carbon all/snp-only × full-forward
vs variant-cache, and the `sum+std` vs `center+ln` aggregation recipes. All
recover subpopulation structure; `center+ln` spreads it more, `snp-only` tightens
clusters, and variant-cache panels mirror full-forward — i.e. the cache preserves
the downstream structure.

**[04_vc_degradation.png](figs/04_vc_degradation.png) — cache fidelity vs #SNPs.**
From 3,000 scanned windows. Left: cosine to the true alt forward vs number of
concurrent SNPs — per-SNP-token cosine is ~1.0 at 1 SNP and drifts to ~0.999 by
9 SNPs (window counts annotated; the 9–10 SNP bins are sparse). Right: pooled-
embedding relative L2 error rises with #SNPs. Takeaway: the approximation is
near-exact for isolated SNPs and degrades gracefully as SNPs co-occur.

**[05_ar_curves.png](figs/05_ar_curves.png) — AR fine-tune curves.** Carbon fine-
tuned through the variant cache on the next-token SNP objective (90% train split).
Left: train vs val log-likelihood (−NLL) on SNP tokens; right: val perplexity.
Val improves every epoch (NLL 5.94 → 5.03, ppl 382 → 153) while noisy train LL
climbs higher — the gap reflects mild overfitting on the bounded (limit-1000)
demo subset.

## From `vc_explainer.ipynb`

**[06_vc_batch_decomposition.png](figs/06_vc_batch_decomposition.png) — batch
anatomy.** A reference + sample rows × token-position grid for one window. Almost
every column is *shared* (grey, identical to reference for all samples); only the
bracketed SNP columns *differ* (blue = reference allele, orange = alt allele).
Shows the decomposition the cache exploits: a handful of differing tokens against
a mostly-shared background.

**[07_vc_forward_comparison.png](figs/07_vc_forward_comparison.png) — standard vs
variant-cache forward.** Left: the standard pass runs all `B·T` tokens through
every layer (`work ∝ B·T·L`). Right: the variant cache forwards the reference
once into a per-layer cache, then recomputes only the differing tokens, which
attend back into the cached reference context via a log-sum-exp merge
(`work ∝ (T + B·V)·L`, with V ≪ T). The conceptual "why it's cheaper" companion
to figure 06.

## From `fingerprint_explainer.ipynb`

**[08_fingerprint_window.png](figs/08_fingerprint_window.png) — fingerprinting one
window.** Six samples × three SNP sites; each sample's pattern of ref (grey) / alt
(orange) calls maps to a fingerprint = its set of alt positions. Identical
patterns collapse, so 6 samples yield only 3 unique sequences to embed. Shows the
dedup key and why repeats are free.

**[09_fingerprint_dedup_scale.png](figs/09_fingerprint_dedup_scale.png) — why it's
fast.** Left: the `(sample × window)` fingerprint matrix is mostly the reference
fingerprint (grey), shared across all samples, with only a sprinkling of variant
fingerprints. Right: forward passes needed to embed the dataset, log scale —
naïve `n_samples × n_windows` = 8,861,854 vs dedup `n_unique` = 53,398, i.e.
**166× fewer** and exact (the embedded sequences are literally identical).

## From `forward_calls.ipynb`

**[10_forward_calls_stages.png](figs/10_forward_calls_stages.png) — full forwards
per stage.** Log-scale bars for the number of *full* LM forward passes at each
stage: full `n_samples × n_windows` = 8,861,854 → dedup `n_unique` = 53,398
(÷166, exact) → variant cache `n_windows` = 23,138 (÷2.3 more; **383× fewer than
naïve**). The dedup→cache step replaces the 30,260 variant-fingerprint forwards
with partial recomputes (only the SNP tokens), leaving one reference full-forward
per window. The capstone that ties the dedup and variant-cache stories together.

## From `e2e_explainer.ipynb`

**[11_e2e_window_aggregation.png](figs/11_e2e_window_aggregation.png) — window
aggregation.** The e2e forward pass: a batch of samples → `gather_batch` dedups
its windows → the encoder embeds the unique windows once into `E (n_unique × D)` →
each sample's vector is pooled from its windows' rows of `E` (`embedding_bag`,
sum/mean, via the `inverse` index) → head → trait predictions/loss.

**[12_e2e_gradient_caching.png](figs/12_e2e_gradient_caching.png) — gradient
caching (two-pass backward).** Why e2e fits in memory. Pass 1: embed under
`no_grad` (chunked) → `E` detached + `requires_grad`, run the head, `loss.backward()`
caches head grads and `E.grad = dL/dE` with no encoder graph kept. Pass 2:
recompute the encoder one chunk at a time WITH grad and `e_c.backward(E.grad[c])`
to accumulate encoder grads, then `opt.step()` updates head + encoder. Peak memory
= `E` (tiny) + a single chunk's encoder graph. (Does not use the variant cache.)
