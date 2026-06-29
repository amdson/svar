"""
crop_embed
----------
Pipeline for embedding crop genomes from SNP VCF files using a DNA language
model (e.g. DNABERT-2).

Typical usage
-------------
    from crop_embed.data import load_snps_from_vcf
    from crop_embed.partitioner import SNPWindowPartitioner
    from crop_embed.dataset import UniqueWindowDataset, SampleDataset
    from crop_embed.embedder import SampleEmbedder

    snps, samples = load_snps_from_vcf(vcf_path)
    partitioner   = SNPWindowPartitioner(snps, half_window=512, buffer=64)
    dataset       = UniqueWindowDataset(vcf_path, fasta_path, partitioner)

    embedding_table = SampleEmbedder.fill_embedding_table(dataset, model, tokenizer)
    embedder        = SampleEmbedder(dataset, embedding_table)
    sample_vecs     = embedder.embed_all()   # {sample_id: Tensor(D,)}
"""

# ── Scratch redirection ───────────────────────────────────────────────────────
# Keep model weights + caches off the home-directory quota. env.sh does this for
# shell/sbatch runs; this block is the safety net for interactive / notebook /
# direct-script runs that import crop_embed without sourcing env.sh. It must run
# *before* transformers/torch are imported (i.e. before the imports below), and
# only takes effect when the scratch base exists, so it's a no-op off-cluster.
# Already-set env vars always win (setdefault), so env.sh / manual overrides hold.
import os as _os
from pathlib import Path as _Path

_SVAR_SCRATCH = _os.environ.get("SVAR_SCRATCH", "/90daydata/small_grains/andrew.dickson")
if _Path(_SVAR_SCRATCH).is_dir():
    _os.environ.setdefault("HF_HOME", f"{_SVAR_SCRATCH}/hf_cache")
    _os.environ.setdefault("HF_HUB_CACHE", _os.environ["HF_HOME"] + "/hub")
    _os.environ.setdefault("TORCH_HOME", f"{_SVAR_SCRATCH}/torch_cache")
    _os.environ.setdefault("WANDB_DIR", f"{_SVAR_SCRATCH}/wandb")

from crop_embed.data.vcf import SNPRecord, load_snps_from_vcf
from crop_embed.dataset import SampleDataset, UniqueWindowDataset
from crop_embed.embedder import (
    BatchedWindowEmbedder,
    FixedWindowEmbedder,
    SampleEmbedder,
    WindowEmbedder,
)
from crop_embed.fingerprint import Fingerprint, build_sample_window_map, make_fingerprint
from crop_embed.partitioner import SNPWindowPartitioner, Window
from crop_embed.heads import AttentionHead, LinearHead, MLPHead, window_position_features
from crop_embed.train import masked_mse, train
from crop_embed.logging_utils import MetricLogger, metrics_path_for
from crop_embed.activation_tracker import ActivationTracker
from crop_embed.encoders import (
    CARBON_MODEL_PATHS,
    DEFAULT_MODEL_PATHS,
    load_encoder_model,
    build_window_embedder,
)

__all__ = [
    "SNPRecord",
    "load_snps_from_vcf",
    "SNPWindowPartitioner",
    "Window",
    "Fingerprint",
    "make_fingerprint",
    "build_sample_window_map",
    "UniqueWindowDataset",
    "SampleDataset",
    "SampleEmbedder",
    "WindowEmbedder",
    "BatchedWindowEmbedder",
    "FixedWindowEmbedder",
    "LinearHead",
    "MLPHead",
    "AttentionHead",
    "window_position_features",
    "masked_mse",
    "train",
    "MetricLogger",
    "metrics_path_for",
    "ActivationTracker",
    "CARBON_MODEL_PATHS",
    "DEFAULT_MODEL_PATHS",
    "load_encoder_model",
    "build_window_embedder",
]
