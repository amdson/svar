#!/usr/bin/env bash
# setup_plantcad_env.sh — build the dedicated PlantCAD (PlantCaduceus) conda env.
#
# PlantCAD's Caduceus/Mamba backbone needs mamba-ssm (+ causal-conv1d), which
# build CUDA extensions tightly coupled to a specific torch/CUDA. Keep this OUT
# of the `svar` env (it would risk the torch Carbon + DNABERT2 use).
#
# Upstream pins (PlantCaduceus Colab / PlantCAD_modules/loader.py):
#   torch==2.3.1+cu121, mamba-ssm==2.2.2, triton==2.3.1, trust_remote_code=True
#
# IMPORTANT: every install uses the plantcad env's ABSOLUTE pip/python ($PIP/$PY),
# never a bare `pip` after `conda activate`. In a non-interactive script
# `conda activate` does not reliably switch the active interpreter, so a bare
# `pip` can silently install into base/svar instead — which is how an earlier run
# tried to build against base's CUDA-13 torch (major-version mismatch with the
# cuda/12.4 nvcc → hard error). Absolute paths make the target env unambiguous.
set -euo pipefail

ENV_NAME="${ENV_NAME:-plantcad}"
SVAR_REPO="${SVAR_REPO:-/home/andrew.dickson/svar}"

# Ensure `conda` is callable even in a non-login shell.
if ! command -v conda >/dev/null 2>&1; then
    for c in /apps/spack-managed-x86_64_v3-v1.1/gcc-11.5.0/miniconda3-*/etc/profile.d/conda.sh \
             "$HOME"/miniconda3/etc/profile.d/conda.sh; do
        # shellcheck disable=SC1090
        [ -f "$c" ] && source "$c" && break
    done
fi
command -v conda >/dev/null 2>&1 || { echo "FATAL: conda not on PATH"; exit 1; }

echo "── 0. preflight: GPU arch + CUDA toolkit must support cu121 mamba-ssm ──"
nvidia-smi || echo "  (no nvidia-smi — inference needs CUDA)"
# Atlas has no system nvcc; load a CUDA toolkit module so mamba-ssm can compile.
# 12.4 is the closest available to torch's cu121 wheel; same major (12) as the
# torch wheel, so torch's cpp_extension treats it as a minor mismatch (warn, OK).
module load cuda/12.4.0 2>/dev/null || echo "  (could not module-load cuda/12.4.0 — check 'module avail cuda')"
nvcc --version || { echo "  FATAL: still no nvcc — cannot build mamba-ssm"; exit 1; }
# V100 = Volta (sm_70). Pin the build target so kernels include sm_70 and the
# build is faster. (V100 also has NO bf16 hardware — eval defaults to fp16 there.)
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0}"
echo "  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

echo "── 1. env (separate from svar), reused if it already exists ──"
CONDA_BASE="$(conda info --base)"
ENV_PREFIX="$CONDA_BASE/envs/$ENV_NAME"
# fall back to the user conda envs dir if the env lives there
[ -d "$ENV_PREFIX" ] || ENV_PREFIX="$HOME/.conda/envs/$ENV_NAME"
if [ ! -x "$ENV_PREFIX/bin/python" ]; then
    conda create -n "$ENV_NAME" python=3.11 -y
    ENV_PREFIX="$(conda info --base)/envs/$ENV_NAME"
    [ -x "$ENV_PREFIX/bin/python" ] || ENV_PREFIX="$HOME/.conda/envs/$ENV_NAME"
fi
PY="$ENV_PREFIX/bin/python"
PIP="$PY -m pip"
echo "  env prefix: $ENV_PREFIX"
"$PY" --version
# sanity: confirm we are NOT about to touch the svar env
case "$ENV_PREFIX" in *"/envs/svar") echo "REFUSING: resolved to svar env"; exit 1;; esac

echo "── 2. torch first, matching CUDA (cu121) — into the plantcad env ──"
$PIP install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
"$PY" -c "import torch; print('  installed torch', torch.__version__, 'cuda', torch.version.cuda)"

echo "── 3. the Mamba stack (these COMPILE against the cuda module — slow) ──"
# CRITICAL: build with --no-build-isolation. Otherwise pip's PEP 517 isolated
# build env installs an *unpinned* (latest) torch as a build-require and compiles
# the CUDA extension against THAT (e.g. 2.12+cu130) instead of our cu121 torch —
# nvcc 12.4 vs cu130 is a major mismatch and hard-errors. --no-build-isolation
# reuses the env's own torch 2.3.1+cu121 (nvcc 12.4 vs 12.1 = minor → warn, OK),
# so we must pre-install the build deps ourselves:
$PIP install triton==2.3.1
$PIP install setuptools wheel ninja packaging numpy einops
# causal-conv1d first (mamba-ssm imports it). *_FORCE_BUILD forces a source
# compile (sm_70) rather than any cached wheel. Both honour TORCH_CUDA_ARCH_LIST.
CAUSAL_CONV1D_FORCE_BUILD=TRUE $PIP install --no-build-isolation causal-conv1d
MAMBA_FORCE_BUILD=TRUE $PIP install --no-build-isolation mamba-ssm==2.2.2
#   if the build still fails: try a prebuilt wheel for torch2.3/cu12/cp311 from
#   the mamba-ssm + causal-conv1d GitHub releases (must include sm_70), or a container.

echo "── 4. the rest ──"
$PIP install transformers pyfaidx pysam pandas numpy tqdm
$PIP install -e "$SVAR_REPO"   # exposes crop_embed, variant_mlm, pretrained_eval, PlantCAD_modules

echo "── 5. route caches to scratch (project rule — never fill the home quota) ──"
# shellcheck disable=SC1091
source "$SVAR_REPO/env.sh"

echo "── 6. smoke test the install ──"
"$PY" - <<'PY'
import torch
from mamba_ssm import Mamba
from transformers import AutoTokenizer, AutoModelForMaskedLM
print('torch', torch.__version__, 'cuda', torch.version.cuda,
      'bf16_supported', torch.cuda.is_bf16_supported())
tok = AutoTokenizer.from_pretrained('kuleshov-group/PlantCaduceus_l32')
m = AutoModelForMaskedLM.from_pretrained(
    'kuleshov-group/PlantCaduceus_l32', trust_remote_code=True,
    torch_dtype=torch.float16).to('cuda').eval()  # fp16: V100 has no bf16
ids = tok('acgt' * 60, return_tensors='pt').input_ids.to('cuda')  # lowercase nt
print('logits', m(input_ids=ids).logits.shape)  # (1, T, vocab)
print('mask_token_id', tok.mask_token_id, 'vocab a/c/g/t',
      [tok.get_vocab()[c] for c in 'acgt'])
PY

echo "── 7. eval smoke test (16 windows) ──"
"$PY" "$SVAR_REPO/pretrained_eval/eval_plantcad.py" --limit 16

echo "done. Activate with:  conda activate $ENV_NAME"
echo "  Full per-SNP run:   python pretrained_eval/eval_plantcad.py --seq-mode none"
echo "  Full whole-seq run: python pretrained_eval/eval_plantcad.py --seq-mode subsample"