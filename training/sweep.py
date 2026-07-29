"""
training/sweep.py
-----------------
Generic sweep driver for the `training/` runners. You declare a grid as plain
Python (a config module exporting ``SWEEP``); this expands the Cartesian product,
shells out to the right runner per point, logs each point, and is **resumable**
(a ledger skips points already done). Results are NOT collected here — every
runner already appends to the run manifest, so ``run_record.load_runs()`` is the
one-read comparison surface after a sweep.

    python -m training.sweep --config training/sweeps/example.py
    python -m training.sweep --config training/sweeps/example.py --dry-run
    python -m training.sweep --config training/sweeps/example.py --jobs 3 --gpus 0,1,2
    python -m training.sweep --config training/sweeps/example.py --only ridge

Config contract
---------------
The config module defines ``SWEEP`` — one block dict, or a list of them. A block:

    {
      "runner": "snp_sklearn",          # snp_sklearn|emb_sklearn|emb_nn|e2e
      "grid":   {"model": ["ridge", "krr"], "svd": [200, 500]},  # swept axes
      "fixed":  {"dataset": "soy", "traits": "protein,oil"},     # constant knobs
      "gpu":    False,                  # pin a GPU from --gpus (NN runners)
      "output_template": None,          # e.g. "trained_e2e/sweep/{label}/model.pt"
      "name":   None,                   # optional label prefix / log subdir
    }

Each (key, value) becomes ``--key value`` (underscores → dashes). ``True`` → bare
flag; ``False``/``None`` → omitted. ``grid`` values are scalars (a trait *set* is
one string, e.g. ``"protein,oil"``). ``fixed`` is merged into every point; keys in
both are overridden by ``grid``. A ``label`` is built from the swept axes so logs
and templated outputs are distinct.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RUNNERS = {
    "snp_sklearn": "training.snp_sklearn.run",
    "emb_sklearn": "training.emb_sklearn.run",
    "emb_nn": "training.emb_nn.run",
    "e2e": "training.e2e.run",
}


def _load_config(path: str):
    spec = importlib.util.spec_from_file_location("_sweep_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "SWEEP"):
        raise SystemExit(f"{path}: config must define SWEEP (a block dict or list of them)")
    sweep = mod.SWEEP
    return sweep if isinstance(sweep, list) else [sweep]


def _sanitize(v) -> str:
    return str(v).replace("/", "-").replace(" ", "").replace(",", "+")


def _expand(block: dict) -> list[dict]:
    """Cartesian product of ``grid`` → list of point dicts, each with a `label`."""
    runner = block["runner"]
    if runner not in RUNNERS:
        raise SystemExit(f"unknown runner {runner!r}; choices: {list(RUNNERS)}")
    grid = block.get("grid", {}) or {}
    fixed = block.get("fixed", {}) or {}
    axes = list(grid.keys())
    combos = list(itertools.product(*[grid[a] for a in axes])) if axes else [()]

    points = []
    for combo in combos:
        swept = dict(zip(axes, combo))
        params = {**fixed, **swept}
        label_bits = [f"{k}{_sanitize(v)}" for k, v in swept.items()]
        label = "_".join(label_bits) if label_bits else "single"
        if block.get("name"):
            label = f"{block['name']}__{label}"
        points.append({"runner": runner, "params": params, "label": label,
                       "gpu": bool(block.get("gpu", False)),
                       "output_template": block.get("output_template")})
    return points


def _to_argv(params: dict) -> list[str]:
    argv: list[str] = []
    for k, v in params.items():
        flag = "--" + k.replace("_", "-")
        if v is True:
            argv.append(flag)
        elif v is False or v is None:
            continue
        else:
            argv += [flag, str(v)]
    return argv


def _build_command(point: dict) -> tuple[list[str], dict, str | None]:
    """Return (argv, params_used, output_path_or_None) for a point."""
    params = dict(point["params"])
    out = None
    if point["output_template"]:
        out = point["output_template"].format(label=point["label"], **point["params"])
        params["output"] = out
    module = RUNNERS[point["runner"]]
    argv = [sys.executable, "-m", module] + _to_argv(params)
    return argv, params, out


def _point_key(argv: list[str]) -> str:
    """Stable id for a point = hash of its resolved command (minus interpreter path)."""
    payload = json.dumps(argv[1:], sort_keys=False)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def _load_ledger(path: Path) -> set[str]:
    done: set[str] = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "ok":
                done.add(row.get("key"))
    return done


def _run_point(point: dict, out_dir: Path, log_dir: Path, ledger: Path,
               gpu: str | None, dry_run: bool, force: bool, done: set[str]) -> str:
    argv, _, out = _build_command(point)
    key = _point_key(argv)
    label = point["label"]

    if not force and key in done:
        print(f"[skip] {label} (done)")
        return "skip"
    if not force and out and Path(out).exists() and Path(out).stat().st_size > 0:
        print(f"[skip] {label} (output exists: {out})")
        return "skip"

    printable = " ".join(argv[2:])  # drop interpreter + -m for readability
    if dry_run:
        gtag = f"[gpu {gpu}] " if gpu else ""
        print(f"[dry] {gtag}{label}\n      {printable}")
        return "dry"

    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    log_path = log_dir / f"{label}.log"
    print(f"[start] {label}" + (f" (gpu {gpu})" if gpu else ""))
    t0 = time.time()
    with open(log_path, "w") as lf:
        lf.write(f"# {printable}\n\n")
        lf.flush()
        rc = subprocess.call(argv, stdout=lf, stderr=subprocess.STDOUT, env=env)
    dt = time.time() - t0
    status = "ok" if rc == 0 else "fail"
    with open(ledger, "a") as f:
        f.write(json.dumps({"key": key, "label": label, "status": status,
                            "rc": rc, "seconds": round(dt, 1),
                            "cmd": printable, "log": str(log_path)}) + "\n")
    mark = "done" if rc == 0 else f"FAIL rc={rc}"
    print(f"[{mark}] {label}  ({dt:.0f}s, log: {log_path})")
    return status


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="path to a Python config exporting SWEEP")
    p.add_argument("--out-dir", default=None,
                   help="sweep working dir for logs+ledger (default: logs/sweeps/<config-stem>)")
    p.add_argument("--jobs", type=int, default=1, help="concurrent points (worker pool size)")
    p.add_argument("--gpus", default="", help="comma-separated GPU ids to round-robin over GPU blocks")
    p.add_argument("--only", default=None, help="substring filter on point labels")
    p.add_argument("--dry-run", action="store_true", help="print commands, run nothing")
    p.add_argument("--list", action="store_true", help="list expanded points and exit")
    p.add_argument("--force", action="store_true", help="ignore ledger/output skips; re-run all")
    args = p.parse_args()

    blocks = _load_config(args.config)
    points: list[dict] = []
    for b in blocks:
        points.extend(_expand(b))
    if args.only:
        points = [pt for pt in points if args.only in pt["label"]]

    stem = Path(args.config).stem
    out_dir = Path(args.out_dir) if args.out_dir else Path("logs/sweeps") / stem
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ledger = out_dir / "ledger.jsonl"
    done = set() if args.force else _load_ledger(ledger)

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]

    print(f"Sweep '{stem}': {len(points)} points"
          + (f", {len(done)} already done" if done else "")
          + f"  (jobs={args.jobs}, gpus={gpus or 'none'})")
    print(f"  logs+ledger: {out_dir}/")

    if args.list:
        for pt in points:
            argv, _, out = _build_command(pt)
            print(f"  {pt['label']}\n    {' '.join(argv[2:])}")
        return

    # Assign a GPU per point (round-robin) only for GPU blocks; None otherwise.
    gpu_i = 0
    for pt in points:
        if pt["gpu"] and gpus:
            pt["_gpu"] = gpus[gpu_i % len(gpus)]
            gpu_i += 1
        else:
            pt["_gpu"] = None

    def work(pt):
        return _run_point(pt, out_dir, log_dir, ledger, pt["_gpu"],
                          args.dry_run, args.force, done)

    if args.jobs > 1 and not args.dry_run:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            results = list(ex.map(work, points))
    else:
        results = [work(pt) for pt in points]

    n_ok = results.count("ok")
    n_fail = results.count("fail")
    n_skip = results.count("skip")
    if not args.dry_run:
        print(f"\nSweep '{stem}' done: {n_ok} ok, {n_fail} fail, {n_skip} skipped."
              f"  Compare with run_record.load_runs().")


if __name__ == "__main__":
    main()
