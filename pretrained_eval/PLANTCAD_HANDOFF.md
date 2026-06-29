# Handoff: PlantCAD log-likelihood eval on rice (whole-sequence + per-SNP)

**Goal.** Extend the `pretrained_eval/` harness so PlantCAD (PlantCaduceus) produces
the same two numbers we already have for Carbon — whole-sequence and per-SNP
log-likelihood loss on the rice dataset — so the three models (Carbon, DNABERT2,
PlantCAD) can be compared **in bits/nt on identical windows**.

This is expected to be a real chunk of work, mostly because **PlantCAD needs its
own conda environment** (`mamba-ssm`, tight torch/CUDA pins). Read this whole doc
before starting; the conceptual differences from Carbon change the metric itself.

---

## 1. What already exists (lean on this)

The Carbon path is done and is your template for *structure*, but **not for the
math** (Carbon is autoregressive; PlantCAD is a masked LM — see §3).

| File | What it gives you |
|---|---|
| `pretrained_eval/loss.py` | Carbon AR per-window NLL (`_batch_row_nll`, `collect_per_window`). Whole-seq + per-SNP from one forward. |
| `pretrained_eval/eval.py` | CLI: builds the window dataset, runs the model, writes a **per-window CSV** + aggregate JSON. Scratch-first path resolution. |
| `pretrained_eval/paths.py` | `per_window_csv(model_path, half_window)` and `output_dir()` — scratch output locations. **Reuse verbatim** so the notebook finds PlantCAD output too. |
| `pretrained_eval/baselines.py` | `uniform_kmer_nll(k)`, `empirical_kmer_entropy(fasta, k=...)`, `nats_per_token_to_bits_per_nt(nll, bases_per_token)`. For PlantCAD use **k=1**. |
| `pretrained_eval/view_losses.ipynb` | The viewer. If PlantCAD writes the **same per-window CSV schema**, the notebook works unchanged (just set `MODEL_PATH`, `HALF_WINDOW`, `K=1`). |

**The masked-LM templates you actually want to copy:**

| File | Why it's the right model |
|---|---|
| `variant_mlm/eval.py`, `variant_mlm/data.py` | DNABERT2 **masked** variant-LM: mask the variant token(s), cross-entropy at masked positions, mean CE + perplexity + top-1. PlantCAD's per-SNP metric is the same shape. `VariantMLMCollator` shows the mask/label (`-100`) pattern. |
| `pseudo_perplexity.ipynb` | Salazar-style **pseudo-perplexity** for a masked model (DNABERT2): mask each position one at a time, average `-log P(true \| context)`. This is exactly the whole-sequence metric for PlantCAD. |
| `PlantCAD_Colab_Example.ipynb` (repo root) | Canonical PlantCAD usage: install, load, **masked scoring recipe** (cells 17–18), and the official `zero_shot_score.py` for VCF variant scoring (cell 26). |
| `PlantCAD_modules/loader.py` | `load_plantcad` / `load_plantcad_mlm` → `(AutoModelForMaskedLM, tokenizer)`; `average_rc_embeddings()` (embeddings only — **not** needed for logits). |

**Shared data pipeline (model-agnostic, reuse as-is):**
`variant_mlm.data.build_variant_window_dataset(vcf, fasta, half_window, buffer)`
→ `(UniqueWindowDataset, indices)` and `make_loader(...)` yielding
`{"sequences", "fingerprints"}`, where a fingerprint is
`(chrom, w_start, w_end, alt_positions)` with 0-based genomic `alt_positions`.

**Carbon reference numbers** (Carbon-500M, half_window=500 → 1000 bp windows,
30,260 variant windows; produced by `pretrained_eval/eval.py`):

| metric | mean NLL | perplexity | bits/nt | tokens |
|---|---|---|---|---|
| whole-sequence | 7.631 | 2060 | **1.835** | 5,053,420 |
| per-SNP | 8.109 | 3323 | **1.950** | 38,068 |
| empirical 6-mer baseline | 8.182 | — | 1.967 | — |
| uniform (ln 4096) | 8.318 | — | 2.000 | — |

---

## 2. Environment (the hard part) — build a dedicated env

PlantCAD's Caduceus/Mamba backbone needs `mamba-ssm` (+ `causal-conv1d`), which
build CUDA extensions and are tightly coupled to a specific torch/CUDA. **Do not
install these into the `svar` env** — it would risk the torch that Carbon and
DNABERT2 use. The `svar` env deliberately leaves `mamba-ssm` commented out
(`environment.yml`).

Upstream (PlantCaduceus Colab / `PlantCAD_modules/loader.py` docstring) pins:
`torch==2.3.1+cu121`, `mamba-ssm==2.2.2`, `triton==2.3.1`, `trust_remote_code=True`.

Suggested setup:

```bash
# 0. check the box: GPU arch + CUDA toolkit must support cu121 mamba-ssm
nvidia-smi              # driver / GPU
nvcc --version          # need a CUDA toolkit for building the extension (or use prebuilt wheels)

# 1. fresh env (keep it separate from `svar`)
conda create -n plantcad python=3.11 -y
conda activate plantcad

# 2. torch first, matching CUDA
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# 3. the Mamba stack (these compile; expect a slow install)
pip install mamba-ssm==2.2.2 causal-conv1d triton==2.3.1
#   if the build fails, try the project's prebuilt wheels matching torch2.3/cu121

# 4. the rest
pip install transformers pyfaidx pysam pandas numpy tqdm
pip install -e /home/andrew.dickson/svar      # so `crop_embed`, `variant_mlm`, `pretrained_eval` import

# 5. ROUTE CACHES TO SCRATCH (project rule — never fill the home quota)
source /home/andrew.dickson/svar/env.sh        # sets HF_HOME etc. under /90daydata

# 6. smoke test the install
python -c "
import torch; from mamba_ssm import Mamba
from transformers import AutoTokenizer, AutoModelForMaskedLM
tok = AutoTokenizer.from_pretrained('kuleshov-group/PlantCaduceus_l32')
m = AutoModelForMaskedLM.from_pretrained('kuleshov-group/PlantCaduceus_l32', trust_remote_code=True).to('cuda').eval()
ids = tok('acgt'*60, return_tensors='pt').input_ids.to('cuda')   # note: lowercase nucleotides
print('logits', m(input_ids=ids).logits.shape)   # (1, T, vocab)
"
```

**Likely failure modes:** mamba-ssm build errors (CUDA toolkit/arch mismatch — the
single biggest risk); `causal-conv1d` version skew; flash-attn-style ABI issues.
If building is a wall, look for the kuleshov-group prebuilt wheels or a container.

Data is already on scratch: `/90daydata/small_grains/andrew.dickson/datasets/rice/`
(FASTA + `sativas413_msu7_final.vcf`). `crop_embed.data.coords` resolves it
automatically when `env.sh` is sourced.

---

## 3. The metric is different from Carbon — read carefully

PlantCAD is a **bidirectional masked LM** (`AutoModelForMaskedLM`),
**single-nucleotide** tokenization, **512 bp max** input. Consequences:

1. **No autoregressive likelihood.** Score by **masking**, not next-token.
   - **Per-SNP (cheap, do this first):** mask each SNP position, read
     `-log P(true_allele | rest)`. One forward per window, mask all SNP tokens at
     once — identical to `variant_mlm`'s objective. This is also the canonical
     **PlantCaduceus zero-shot variant** score (the alt-vs-ref log-likelihood
     ratio); `PlantCaduceus/src/zero_shot_score.py` does exactly this over a VCF.
   - **Whole-sequence (expensive):** true masked likelihood is **pseudo-perplexity**
     — mask each position one at a time → **L forward passes per window**
     (`pseudo_perplexity.ipynb` shows the loop, batched). At 500 bp that's ~500×
     the per-SNP cost. Decide with the user: full pseudo-PPL, a **subsample** of
     positions per window, or report per-SNP only. Whatever you pick, **log it**
     (no silent truncation).

2. **Masked-scoring recipe (from the Colab, cells 17–18) — RC is handled for you:**
   ```python
   NT = list("acgt")                      # PlantCAD nucleotide tokens are LOWERCASE
   nt_ids = [tok.get_vocab()[c] for c in NT]
   input_ids[0, pos] = tok.mask_token_id
   logits = model(input_ids=input_ids).logits        # RC already combined in the head
   p = logits[0, pos, nt_ids].softmax(-1)            # 4-way distribution over a/c/g/t
   nll = -torch.log(p[NT.index(true_base.lower())])
   ```
   `outputs.logits` is directly usable — `average_rc_embeddings` is for *embeddings*
   only; do **not** apply it to logits.

3. **Single-nucleotide ⇒ bits/nt is trivial.** NLL is already per-nucleotide:
   `nats_per_token_to_bits_per_nt(nll, bases_per_token=1)` = `nll / ln 2`. This is
   the clean common unit vs Carbon (k=6) and DNABERT2 (mean tokens/base). The
   naive baseline is `empirical_kmer_entropy(fasta, k=1)` (base composition).

4. **512 bp cap ⇒ use `half_window=250`** (full window 500 bp ≤ 512). The repo
   already has `_w250` artifacts. **Fairness caveat:** the Carbon numbers above are
   half_window=500. For an apples-to-apples head-to-head, **rerun Carbon at
   half_window=250 too** (`python pretrained_eval/eval.py --half-window 250`), or
   note the window-size difference loudly.

5. **SNP → token index.** Single-nucleotide, so token = `n_prefix + (p - w_start)`.
   PlantCAD's tokenizer doesn't expose offset mappings (the embedder notes this),
   so compute analytically — `_build_snp_mask_kmer(..., k=1, n_prefix_tokens=?)`
   nearly works, but **verify `n_prefix_tokens`** (does PlantCAD prepend a BOS/CLS?
   tokenize a known string and inspect). Confirm with an assert that the token at
   the computed index decodes to the expected reference base.

---

## 4. Implementation plan (recommended shape)

Mirror the Carbon module so the **per-window CSV schema is identical** and the
existing notebook + baselines just work.

1. `pretrained_eval/plantcad_loss.py`
   - `plantcad_masked_nll(model, tokenizer, sequences, fingerprints, ...)` →
     per-row `{seq_nll, seq_tokens, snp_nll, snp_tokens}` (CPU tensors), matching
     `loss._batch_row_nll`'s return shape.
     - per-SNP: mask SNP positions, CE over the 4 nt-token logits.
     - whole-seq: pseudo-PPL loop (or chosen subsample); reuse the masking pattern.
   - `collect_per_window(...)` emitting the **same record keys** as
     `loss.collect_per_window` (`chrom,w_start,w_end,n_snps,seq_*,snp_*`).

2. `pretrained_eval/eval_plantcad.py` (copy `eval.py`)
   - `load_plantcad_mlm` instead of `load_carbon_lm`; default
     `--model-path kuleshov-group/PlantCaduceus_l32`, `--half-window 250`,
     `--max-length 512`.
   - Write the per-window CSV via `paths.per_window_csv(model_path, half_window)`
     (the slug keeps it separate from Carbon's file).

3. Notebook: open `view_losses.ipynb`, set `MODEL_PATH` to the PlantCAD repo,
   `HALF_WINDOW=250`, `K=1`. The baseline cell already parameterizes on `K`.

> Alternative: a `--backend {carbon,plantcad}` switch on the existing `eval.py`.
> A separate module is cleaner given the env split (you'll run PlantCAD from the
> `plantcad` env, Carbon from `svar`).

---

## 5. Validation / acceptance

- **Smoke:** `--limit 16`; assert ~500 tokens/window at hw250; per-SNP NLL finite;
  spot-check that masking the SNP and softmaxing over `acgt` gives sane probs.
- **Cross-check** per-SNP against `zero_shot_score.py` on a handful of the same
  variants (should agree up to sign/convention).
- **Head-to-head in bits/nt** on identical windows (hw250 for all models). Expect
  PlantCAD — bidirectional, purpose-built for plant genomes — to **beat Carbon**
  on per-SNP especially (it sees both flanks; Carbon only sees the left).
- Keep the **scratch discipline**: weights + outputs under `/90daydata`, nothing
  on the home quota, no data committed to git.

---

## 6. Open questions to settle with the user

1. **Whole-sequence cost:** full pseudo-PPL (~500 forwards/window) vs a
   per-window position subsample vs per-SNP-only? (Biggest runtime lever.)
2. **Checkpoint size:** `PlantCaduceus_l32` (default) vs a smaller `l20`/`l24`?
3. **Carbon parity:** OK to rerun Carbon at half_window=250 for a fair comparison,
   or compare at each model's native window and note it?

---

## 7. Quick reference — key paths

- Repo: `/home/andrew.dickson/svar` (`pip install -e .` exposes `crop_embed`, `variant_mlm`, `pretrained_eval`)
- Rice data (scratch): `/90daydata/small_grains/andrew.dickson/datasets/rice/`
- Env knobs: `source /home/andrew.dickson/svar/env.sh` (HF/torch caches → scratch)
- Carbon eval to mirror: `pretrained_eval/{loss,eval,paths,baselines}.py`
- Masked-LM templates: `variant_mlm/{eval,data}.py`, `pseudo_perplexity.ipynb`, `PlantCAD_Colab_Example.ipynb`
- PlantCAD loader: `PlantCAD_modules/loader.py` (`load_plantcad_mlm`, default `kuleshov-group/PlantCaduceus_l32`)
