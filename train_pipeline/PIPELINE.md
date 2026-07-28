# Genome → Prediction Pipeline

## The big idea

We predict crop phenotypes (traits) from genotypes. Rather than feeding a DNA
language model one giant genome, we exploit the fact that samples differ only at
SNPs. We cut the genome into small windows around SNP clusters, embed each
**unique** window sequence once with a frozen DNA LM (DNABERT-2 or PlantCAD),
pool those window embeddings into one vector per sample, and train a lightweight
regression head on top. The expensive embedding is computed once and cached;
only the cheap head is trained/retrained.

## The path from genome → prediction

### 1. VCF → SNPs — `crop_embed/data/vcf.py`
`load_snps_from_vcf` reads a biallelic VCF into `SNPRecord(pos, ref_byte,
alt_byte, gt_alts)` per chromosome. `gt_alts` is a compact per-sample byte flag:
1 = carries alt allele, 0 = ref/missing (only the first allele is checked, so
it's effectively haploid/first-allele).

### 2. SNPs → windows — `crop_embed/partitioner.py`
`SNPWindowPartitioner` greedily walks each chromosome left-to-right, opening a
`2*half_window` bp window (default 500 half → 1000 bp) at the leftmost
unassigned SNP and absorbing all nearby SNPs within the buffer. Every SNP lands
in exactly one window, never at an edge. This clusters co-located SNPs so one
window can cover several.

### 3. (sample, window) → fingerprint — `crop_embed/fingerprint.py`
Since the reference is fixed, a window's actual sequence for a sample is fully
determined by `(chrom, start, end, alt_positions)` — which SNPs in that window
carry the alt allele. This tuple is the **fingerprint**, the dedup key. Two
samples sharing a haplotype in a window share a fingerprint.

### 4. Dedup + materialize sequences — `crop_embed/dataset.py`
`UniqueWindowDataset` builds the set of *unique* fingerprints (far fewer than
samples × windows) and a `sample_fp_index` tensor `(n_samples, n_windows)`
mapping each sample's windows to fingerprint rows. `extract_sequence`
reconstructs the DNA string by copying the reference slice and patching in alt
bytes (N-padding at chromosome ends).

### 5. Windows → embeddings (the expensive, cached step) — `train_pipeline/embed_windows.py` + `crop_embed/embedder.py`
`WindowEmbedder` tokenizes each unique sequence, runs the frozen DNA LM, and
masked-mean-pools the hidden states into one vector per window (with `snp_only`
and `output_layer` options). `fill_embedding_table` runs this over all unique
windows with checkpoint/resume. The result is saved as a `.pt` cache: a `cache`
table `(n_unique_windows, D)` + `sample_fp_index` + metadata, loadable via
`FixedWindowEmbedder.from_file`.

### 6. Cache the split — `train_pipeline/cache_split.py`
One deterministic train/val split by sample ID, plus z-scored targets, so every
model trains/evaluates on identical points.

### 7. Windows → per-sample vector → trait prediction (the trained head) — `training/emb_nn/run.py` + `crop_embed/models/fp_head_model.py`
For a frozen cache the per-sample pooled vector never changes, so it's
**pre-pooled once** via `embedding_bag` (sum or mean over each sample's window
fingerprints) into `(n_samples, D)`. Then a head — `FPSumHeadModel` wrapping a
`LinearModel` or residual `MLPModel` — maps `(B, D) → (B, n_traits)`. Key pieces:

- A **learned standardizer** de-means/rescales the summed vector (warm-started
  from training-set stats) so early gradients learn trait structure instead of
  fighting the large summed magnitude.
- Trained with masked MSE (NaN targets ignored), AdamW with weight decay only on
  inner linear weights.
- **Reference-delta variant** (`--subtract-reference`, `FPRefDeltaSumHeadModel`):
  subtracts each window's variant-free reference embedding before pooling, so the
  head sees only departures from reference and pure-reference windows drop out.

### 8. End-to-end (optional) — `train_pipeline/train_end2end.py`
Unfreezes the DNA LM and fine-tunes backbone + head together (warm-started from a
cached head), using `BatchedWindowEmbedder` to embed each batch's unique windows
live instead of from cache.

### Inference
Inference uses the same chain: VCF → partition → fingerprint → embed (or cache
lookup) → pool → head. The saved head checkpoint carries `head_config`,
`trait_cols`, and cache/split paths to reconstruct everything.

## One thing worth flagging

`crop_embed/dataset.py:190` has a `TODO` questioning whether N-padding vs.
attention-mask padding is correct for windows that run off the chromosome end —
currently it hard-pads with `"N"`, which the embedder then pools over normally.
If you have windows near chromosome boundaries, that's a real modeling choice
still marked dubious in the code.
