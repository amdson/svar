"""
training/sweeps/example.py
--------------------------
Example sweep config for `python -m training.sweep --config <this file>`.

`SWEEP` is one block dict, or a list of blocks. Each block expands the Cartesian
product of `grid` (the swept axes), merges `fixed` (constant knobs), and shells
out to the block's `runner`. Axes are plain Python, so you can compute them:

    SVD = [100, 200, 500]
    MODELS = ["ridge", "krr"]

Values are scalars; a trait *set* is a single comma-joined string. `True` → a bare
flag (e.g. "sparse": True → --sparse); omit or set None/False to leave a flag off.

Run:   python -m training.sweep --config training/sweeps/example.py --dry-run
"""

# --- Block 1: sklearn on the SNP matrix, sweeping model × SVD width -----------
snp_sklearn = {
    "runner": "snp_sklearn",
    "fixed": {"dataset": "soy", "traits": "protein,oil", "seed": 42},
    "grid": {
        "model": ["ridge", "krr"],
        "svd": [200, 500],
    },
}

# --- Block 2: same suite over the raw SPARSE SNP matrix (SVD required) --------
snp_sparse = {
    "runner": "snp_sklearn",
    "name": "sparse",
    "fixed": {"dataset": "soy", "traits": "protein,oil", "sparse": True},
    "grid": {"model": ["ridge", "pls"], "svd": [200, 500]},
}

# --- Block 3: sklearn on pooled embeddings, sweeping backbone × window --------
# (needs the matching embedding caches to exist; see train_pipeline/embed_windows.py)
emb_sklearn = {
    "runner": "emb_sklearn",
    "fixed": {"dataset": "rice", "traits": "all", "model": "ridge"},
    "grid": {
        "backbone": ["carbon500m"],
        "half_window": [500, 1000],
    },
}

# --- Block 4: NN heads, GPU-pinned, with a templated output dir per point -----
emb_nn = {
    "runner": "emb_nn",
    "name": "head",
    "gpu": True,
    "fixed": {"dataset": "rice", "backbone": "carbon500m", "half_window": 500,
              "epochs": 15},
    "grid": {"head": ["linear", "mlp"], "lr": ["1e-3", "3e-4"]},
    "output_template": "trained_heads/sweep_example/{label}/model.pt",
}

# Enable the blocks you want to run. Start with the cheap CPU sklearn ones.
SWEEP = [snp_sklearn, snp_sparse]
