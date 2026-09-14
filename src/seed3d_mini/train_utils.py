"""Shared training utilities: torchrun/DDP plumbing, checkpointing, logging."""

from __future__ import annotations

import datetime
from contextlib import contextmanager
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist

__all__ = [
    "TrainContext",
    "init_distributed",
    "reduce_metrics",
    "save_checkpoint",
    "load_checkpoint",
    "MetricLogger",
    "cosine_with_warmup",
    "set_seed",
]


@dataclass
class TrainContext:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: torch.device = torch.device("cpu")
    distributed: bool = False

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed(device: Optional[str] = None) -> TrainContext:
    """Initialise torchrun/DDP if launched under ``torchrun``, else single process."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if distributed and device.startswith("cuda"):
        torch.cuda.set_device(local_rank)
        dev = torch.device("cuda", local_rank)
        if not dist.is_initialized():
            # datetime.timedelta, not torch.timedelta (the latter does not exist);
            # a long timeout keeps a slow first NCCL handshake from aborting the job
            dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600))
    else:
        dev = torch.device(device)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    return TrainContext(rank=rank, local_rank=local_rank, world_size=world_size, device=dev, distributed=distributed)


def set_seed(seed: int, rank: int = 0) -> None:
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def reduce_metrics(metrics: Dict[str, torch.Tensor], ctx: TrainContext) -> Dict[str, float]:
    """All-reduce scalar metrics across ranks (NCCL needs them on the GPU).

    Logging must never kill a training run, so a failure to reduce falls back to the
    local value instead of raising.
    """
    out: Dict[str, float] = {}
    for k, v in metrics.items():
        t = v.detach().float()
        if ctx.distributed:
            try:
                t = t.to(ctx.device, non_blocking=True)
                dist.all_reduce(t, op=dist.ReduceOp.AVG)
                t = t.cpu()
            except Exception:  # noqa: BLE001 - see docstring
                pass
        out[k] = float(t.item())
    return out


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    *,
    step: int = 0,
    epoch: int = 0,
    config: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = model.module if hasattr(model, "module") else model
    state = {
        "model": raw.state_dict(),
        "step": step,
        "epoch": epoch,
        "config": config or {},
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if extra:
        state.update(extra)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    *,
    strict: bool = True,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    raw = model.module if hasattr(model, "module") else model
    missing, unexpected = raw.load_state_dict(ckpt["model"], strict=strict)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    info = {k: v for k, v in ckpt.items() if k not in ("model", "optimizer")}
    info["missing_keys"] = list(missing)
    info["unexpected_keys"] = list(unexpected)
    return info


class MetricLogger:
    """Rank-0 stdout + JSONL logger."""

    def __init__(self, log_path: Optional[str | Path] = None, is_main: bool = True, use_wandb: bool = False):
        self.is_main = is_main
        self.path = Path(log_path) if log_path else None
        if self.path and is_main:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.wandb = None
        if use_wandb and is_main:
            try:
                import wandb  # noqa: F401

                self.wandb = wandb
            except Exception:
                self.wandb = None

    def log(self, step: int, metrics: Dict[str, float], prefix: str = "train") -> None:
        if not self.is_main:
            return
        msg = " ".join(f"{k}={v:.4g}" for k, v in metrics.items())
        elapsed = time.time() - self.t0
        print(f"[{prefix}] step {step} ({elapsed / 60:.1f} min) {msg}", flush=True)
        if self.path:
            with self.path.open("a") as f:
                f.write(json.dumps({"step": step, "prefix": prefix, "elapsed": elapsed, **metrics}) + "\n")
        if self.wandb is not None:
            self.wandb.log({f"{prefix}/{k}": v for k, v in metrics.items()}, step=step)

    def init_wandb(self, project: str, config: Dict, run_name: Optional[str] = None) -> None:
        if self.wandb is not None and self.is_main:
            self.wandb.init(project=project, config=config, name=run_name)


def cosine_with_warmup(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr_ratio: float = 0.05,
) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    if total_steps <= warmup_steps:
        return base_lr
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    cos = 0.5 * (1.0 + np.cos(np.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cos)


class StepProfiler:
    """Opt-in phase timer for the training loop (``SEED3D_PROFILE=1``).

    Distributed training has three very different latency sources -- the data
    loader, the GPU work, and the DDP gradient all-reduce -- and the aggregate
    ``step_time`` cannot tell them apart.  This records per-phase wall clock for a
    bounded number of steps and prints a summary, which is how the 20x gap between
    the single-GPU measurement and the 4-GPU run was located.
    """

    def __init__(self, enabled: bool, max_steps: int = 25) -> None:
        self.enabled = enabled
        self.max_steps = max_steps
        self.sums: Dict[str, float] = {}
        self.count = 0

    @contextmanager
    def phase(self, name: str):
        if not self.enabled or self.count >= self.max_steps:
            yield
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.sums[name] = self.sums.get(name, 0.0) + (time.perf_counter() - t0)

    def step_done(self) -> None:
        self.count += 1
        if self.enabled and self.count == self.max_steps:
            total = sum(self.sums.values())
            print("--- step profile (mean seconds over %d steps) ---" % self.count, flush=True)
            for k, v in sorted(self.sums.items(), key=lambda kv: -kv[1]):
                print(f"    {k:14s} {v / self.count:7.3f} s  ({100 * v / max(total, 1e-9):4.1f}%)", flush=True)
