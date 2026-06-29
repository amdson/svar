#!/usr/bin/env bash
# env.sh — route large/model-weight and cache writes to cluster scratch
# (/90daydata/small_grains/...) instead of the home-directory quota.
#
# WHY: home is quota-limited and was filling up with HuggingFace model weights
# (~/.cache/huggingface), torch downloads, generated embedding caches, and wandb
# run files. Everything below is redirected under one scratch base so it's easy
# to find and clean up.
#
# USAGE: source this before running anything that loads a model or writes a cache:
#     source env.sh
# The run_*.sbatch scripts source it automatically.
#
# Each var uses ${VAR:-default} so an already-exported value wins — you can
# override SVAR_SCRATCH (or any single cache dir) before sourcing.

export SVAR_SCRATCH="${SVAR_SCRATCH:-/90daydata/small_grains/andrew.dickson}"

# HuggingFace: model weights, tokenizers, datasets, hub blobs.
# HF_HOME is the modern single knob (covers the deprecated TRANSFORMERS_CACHE).
export HF_HOME="${HF_HOME:-$SVAR_SCRATCH/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

# torch.hub / torch model downloads.
export TORCH_HOME="${TORCH_HOME:-$SVAR_SCRATCH/torch_cache}"

# Weights & Biases run files (can grow large during sweeps).
export WANDB_DIR="${WANDB_DIR:-$SVAR_SCRATCH/wandb}"

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TORCH_HOME" "$WANDB_DIR" \
         "$SVAR_SCRATCH/checkpoints" "$SVAR_SCRATCH/trained_heads"
