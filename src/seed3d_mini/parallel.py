"""Data-parallel wrapper that averages gradients with ONE flat all-reduce.

Why this exists
---------------
``torch.nn.parallel.DistributedDataParallel`` was measured at **3.7-7.9 s/step**
on this machine while the *identical* unwrapped module runs at **0.23 s/step**
(`/tmp/ddp_test.py`; the same gap appears in the real training loop, where the
profiler attributes 98% of every step to ``fwd_bwd``).  That is not a model
problem: it is DDP's gradient bucketing interacting badly with this host's
interconnect, where every GPU except one negotiates **PCIe Gen 1** instead of
Gen 4 (~4 GB/s aggregate for NCCL all-reduce, measured).

For a 25M-parameter model the gradient is only ~100 MB, i.e. ~25 ms at that
bandwidth, so the cost is per-collective overhead rather than bytes.  This module
therefore:

* flattens all gradients into a single contiguous buffer,
* performs exactly **one** ``all_reduce`` per step,
* writes the averaged values back.

It is a drop-in replacement for DDP for the single-node case (no
``find_unused_parameters`` subtleties, no bucketing, no autograd hooks), and it is
what ``--data-parallel flat`` selects in the training scripts.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.distributed as dist

__all__ = ["FlatDataParallel", "wrap_data_parallel"]


class FlatDataParallel(torch.nn.Module):
    """Average gradients across ranks with a single flat all-reduce."""

    def __init__(self, module: torch.nn.Module, device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.module = module
        self.device = device
        self._params: List[torch.nn.Parameter] = [p for p in module.parameters() if p.requires_grad]
        self._sizes = [p.numel() for p in self._params]

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def sync_grads(self) -> None:
        """Average the gradients of all ranks in place (one collective)."""
        grads = [p.grad for p in self._params if p.grad is not None]
        if not grads or not dist.is_available() or not dist.is_initialized():
            return
        world = dist.get_world_size()
        if world == 1:
            return
        flat = torch.cat([g.reshape(-1) for g in grads])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world)
        offset = 0
        for g in grads:
            n = g.numel()
            g.copy_(flat[offset : offset + n].view_as(g))
            offset += n


def wrap_data_parallel(
    module: torch.nn.Module,
    *,
    mode: str = "flat",
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """Wrap ``module`` for data parallelism.

    ``mode``: ``"flat"`` (single flat all-reduce, see :class:`FlatDataParallel`),
    ``"torch"`` (stock DDP, kept for comparison/benchmarks) or ``"none"``.
    """
    if mode == "none" or not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return module
    if mode == "torch":
        return torch.nn.parallel.DistributedDataParallel(
            module, device_ids=[device.index] if isinstance(device, torch.device) else None
        )
    if mode == "flat":
        return FlatDataParallel(module, device=device)
    raise ValueError(f"unknown data-parallel mode {mode!r}")
