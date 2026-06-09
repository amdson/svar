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
    "DEFAULT_MODEL_PATHS",
    "load_encoder_model",
    "build_window_embedder",
]
