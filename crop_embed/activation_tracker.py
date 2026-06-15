"""
activation_tracker.py
---------------------
Track internal activations (and module inputs) during training without editing
any model's `forward`. Uses PyTorch forward hooks: `register_forward_hook`
attaches a callback to a module that fires on every forward pass and sees that
module's inputs and output.

We attach a hook to every *leaf* submodule (a module with no children — the
actual Linear / GELU / LayerNorm / standardizer layers, not the Sequential /
ModuleList containers that wrap them) and reduce each tensor to a few scalars
(mean / std / absmax / frac_zero). Those scalars merge straight into the
MetricLogger row, so they land in the JSONL sidecar and wandb next to the
train/val metrics — see train_pipeline/train_head.py and train_end2end.py.

Why leaves, and why both input and output: a hook only fires when a module is
called via `module(...)` (i.e. its `__call__`/`forward`). train_head.py calls
`head.forward_postsum(x)` directly, which bypasses the top-level head's
`__call__`, so a hook on the head itself never fires — but the leaf submodules
are still invoked normally inside it. Recording each leaf's *input* as well as
its *output* means the network input (the summed embedding fed to the
standardizer / first Linear) is captured regardless of that bypass.

Cost control: hooks fire every batch, but we only reduce-and-store when the
tracker is *armed*. Arm it right before the forward on the steps you already
log; it no-ops on every other step, so overhead is negligible.

    from crop_embed import ActivationTracker

    tracker = ActivationTracker(head)          # after the model is built
    ...
    is_log = step % args.log_every == 0
    if is_log:
        tracker.arm()                          # next forward records stats
    pred = head.forward_postsum(x)             # (or head(E, inv) in e2e)
    ...
    if is_log:
        logger.log({"step": step, **train_m, **tracker.collect()})
    ...
    tracker.remove()                           # detach hooks when done (optional)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _first_tensor(obj):
    """Pull the first floating-point tensor out of a hook input/output, which may
    be a bare tensor or a (possibly nested) tuple/list (e.g. attention returns a
    tuple). Returns None if there isn't one."""
    if isinstance(obj, torch.Tensor):
        return obj if obj.is_floating_point() else None
    if isinstance(obj, (tuple, list)):
        for item in obj:
            t = _first_tensor(item)
            if t is not None:
                return t
    return None


class ActivationTracker:
    """
    Registers forward hooks on a model's leaf modules and, when armed, reduces
    each module's input and output to scalar summary stats keyed by module name.

    Parameters
    ----------
    model      : the module to instrument.
    prefix     : key prefix for the emitted stats (default "act"). Stats are
                 keyed "<prefix>/<module_name>/<in|out>/<stat>".
    leaf_only  : if True (default), hook only modules with no children — the real
                 compute layers, not the containers that hold them.
    record_input : if True (default), record module inputs in addition to outputs.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        prefix: str = "act",
        leaf_only: bool = True,
        record_input: bool = True,
    ) -> None:
        self.prefix = prefix
        self.record_input = record_input
        self._armed = False
        self._stats: dict[str, float] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

        for name, module in model.named_modules():
            if name == "":                       # the top-level module itself
                continue
            if leaf_only and any(module.children()):
                continue                         # skip containers; hook leaves
            self._handles.append(
                module.register_forward_hook(self._make_hook(name))
            )

    def _make_hook(self, name: str):
        def hook(module, inputs, output):
            if not self._armed:
                return
            if self.record_input:
                self._record(f"{name}/in", _first_tensor(inputs))
            self._record(f"{name}/out", _first_tensor(output))
        return hook

    @torch.no_grad()
    def _record(self, key: str, t: torch.Tensor | None) -> None:
        if t is None:
            return
        t = t.detach().float()
        base = f"{self.prefix}/{key}"
        self._stats[f"{base}/mean"]   = t.mean().item()
        self._stats[f"{base}/std"]    = t.std().item()
        self._stats[f"{base}/absmax"] = t.abs().max().item()
        # Fraction of exactly-zero entries: dead ReLU/GELU units, masked rows, etc.
        self._stats[f"{base}/frac_zero"] = (t == 0).float().mean().item()

    def arm(self) -> None:
        """Record stats on the next forward pass(es) until `collect()`."""
        self._armed = True
        self._stats = {}

    def collect(self) -> dict[str, float]:
        """Disarm and return the stats gathered since `arm()`."""
        self._armed = False
        return self._stats

    def remove(self) -> None:
        """Detach all hooks. Safe to call once at the end of training."""
        for h in self._handles:
            h.remove()
        self._handles = []
