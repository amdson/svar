from crop_embed.data.vcf import SNPRecord, load_snps_from_vcf
from crop_embed.data.vcf_polars import (
    load_alt_matrix,
    load_snps_from_vcf as load_snps_from_vcf_polars,
)
from crop_embed.data.pgen import load_snps_from_pgen
from crop_embed.data.preprocessing import (
    load_vcf_sparse,
    load_phenotypes,
    align_samples,
    align_targets_to_dataset,
    scale_phenotypes,
    impute_phenotypes,
    reduce_snp_features,
    train_test_split_data,
    load_data,
)
from crop_embed.data.coords import (
    build_msu6_to_msu7_map,
    remap_vcf_coordinates,
    chrom_name_map,
    FASTA_PATH,
    VCF_PATH,
    FLANKING_PATH,
    SNP_INFO_PATH,
)
from crop_embed.data.loading import (
    DEFAULT_VCF_PATH,
    build_dataset,
    load_targets,
    make_split,
    save_split,
    load_split,
    prepare_data,
)

__all__ = [
    # vcf / pgen
    "SNPRecord", "load_snps_from_vcf", "load_snps_from_pgen",
    "load_snps_from_vcf_polars", "load_alt_matrix",
    # preprocessing
    "load_vcf_sparse", "load_phenotypes", "align_samples",
    "align_targets_to_dataset",
    "scale_phenotypes", "impute_phenotypes", "reduce_snp_features",
    "train_test_split_data", "load_data",
    # coords
    "build_msu6_to_msu7_map", "remap_vcf_coordinates", "chrom_name_map",
    "FASTA_PATH", "VCF_PATH", "FLANKING_PATH", "SNP_INFO_PATH",
    # loading / split caching
    "DEFAULT_VCF_PATH", "build_dataset", "load_targets", "make_split",
    "save_split", "load_split", "prepare_data",
]
