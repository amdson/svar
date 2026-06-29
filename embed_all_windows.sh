#!/usr/bin/env bash
# Sequentially embed 250 / 500 / 1000 / 2000 nt windows with both the variant-cache
# (embed_windows_vc.py) and the full-forward (embed_windows.py) Carbon embedders,
# each in full-window-mean and snp-only pooling. 16 caches total.
set -e

VCF=/home/andrew.dickson/rice_data/sativas413_msu7_final.vcf

# python train_pipeline/embed_windows.py --vcf-path "$VCF" --backend carbon --half-window 500 --buffer 0 --max-length 2048 --batch-size 64 --output checkpoints/manual/sativas413_carbon500m_hw500.ckpt.pt
# python train_pipeline/embed_windows.py --vcf-path "$VCF" --backend carbon --half-window 500 --buffer 0 --max-length 2048 --batch-size 64 --snp-only --output checkpoints/manual/sativas413_carbon500m_hw500_snponly.ckpt.pt
# python train_pipeline/embed_windows_vc.py --vcf-path "$VCF" --half-window 500  --max-length 2048 --vc-batch-size 32 --output checkpoints/manual/sativas413_carbon500m_vc_hw500.ckpt.pt
python train_pipeline/embed_windows_vc.py --vcf-path "$VCF" --half-window 500  --max-length 2048 --vc-batch-size 32 --snp-only --output checkpoints/manual/sativas413_carbon500m_vc_hw500_snponly.ckpt.pt


# # ── Variant cache, full-window mean ───────────────────────────────────────────
# python train_pipeline/embed_windows_vc.py --vcf-path "$VCF" --half-window 125  --max-length 2048 --vc-batch-size 32 --output checkpoints/carbon/sativas413_carbon_vc_w250_mean.ckpt.pt
# python train_pipeline/embed_windows_vc.py --vcf-path "$VCF" --half-window 250  --max-length 2048 --vc-batch-size 32 --output checkpoints/carbon/sativas413_carbon_vc_w500_mean.ckpt.pt
# python train_pipeline/embed_windows_vc.py --vcf-path "$VCF" --half-window 500  --max-length 2048 --vc-batch-size 32 --output checkpoints/carbon/sativas413_carbon_vc_w1000_mean.ckpt.pt
# python train_pipeline/embed_windows_vc.py --vcf-path "$VCF" --half-window 1000 --max-length 2048 --vc-batch-size 32 --output checkpoints/carbon/sativas413_carbon_vc_w2000_mean.ckpt.pt

# # ── Variant cache, snp-only ───────────────────────────────────────────────────
# python train_pipeline/embed_windows_vc.py --vcf-path "$VCF" --half-window 125  --max-length 2048 --vc-batch-size 32 --snp-only --output checkpoints/carbon/sativas413_carbon_vc_w250_snp.ckpt.pt
# python train_pipeline/embed_windows_vc.py --vcf-path "$VCF" --half-window 250  --max-length 2048 --vc-batch-size 32 --snp-only --output checkpoints/carbon/sativas413_carbon_vc_w500_snp.ckpt.pt
# python train_pipeline/embed_windows_vc.py --vcf-path "$VCF" --half-window 500  --max-length 2048 --vc-batch-size 32 --snp-only --output checkpoints/carbon/sativas413_carbon_vc_w1000_snp.ckpt.pt
# python train_pipeline/embed_windows_vc.py --vcf-path "$VCF" --half-window 1000 --max-length 2048 --vc-batch-size 32 --snp-only --output checkpoints/carbon/sativas413_carbon_vc_w2000_snp.ckpt.pt

# # ── Full forward, full-window mean ────────────────────────────────────────────
# python train_pipeline/embed_windows.py --backend carbon --vcf-path "$VCF" --half-window 125  --max-length 2048 --batch-size 64 --output checkpoints/carbon/sativas413_carbon_w250_mean.ckpt.pt
# python train_pipeline/embed_windows.py --backend carbon --vcf-path "$VCF" --half-window 250  --max-length 2048 --batch-size 64 --output checkpoints/carbon/sativas413_carbon_w500_mean.ckpt.pt
# python train_pipeline/embed_windows.py --backend carbon --vcf-path "$VCF" --half-window 500  --max-length 2048 --batch-size 64 --output checkpoints/carbon/sativas413_carbon_w1000_mean.ckpt.pt
# python train_pipeline/embed_windows.py --backend carbon --vcf-path "$VCF" --half-window 1000 --max-length 2048 --batch-size 64 --output checkpoints/carbon/sativas413_carbon_w2000_mean.ckpt.pt

# # ── Full forward, snp-only ────────────────────────────────────────────────────
# python train_pipeline/embed_windows.py --backend carbon --vcf-path "$VCF" --half-window 125  --max-length 2048 --batch-size 64 --snp-only --output checkpoints/carbon/sativas413_carbon_w250_snp.ckpt.pt
# python train_pipeline/embed_windows.py --backend carbon --vcf-path "$VCF" --half-window 250  --max-length 2048 --batch-size 64 --snp-only --output checkpoints/carbon/sativas413_carbon_w500_snp.ckpt.pt
# python train_pipeline/embed_windows.py --backend carbon --vcf-path "$VCF" --half-window 500  --max-length 2048 --batch-size 64 --snp-only --output checkpoints/carbon/sativas413_carbon_w1000_snp.ckpt.pt
# python train_pipeline/embed_windows.py --backend carbon --vcf-path "$VCF" --half-window 1000 --max-length 2048 --batch-size 64 --snp-only --output checkpoints/carbon/sativas413_carbon_w2000_snp.ckpt.pt
</content>
