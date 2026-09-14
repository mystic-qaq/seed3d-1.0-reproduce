"""Train Seed3D-DiT (stage 2): rectified flow over frozen VAE latents.

Report recipe implemented here (Sec. 2.1.2 + 4.1):
  * single-reference-image conditioning through the dual (DINOv2 + RADIO)
    encoder, with classifier-free-guidance dropout on the whole condition,
  * velocity-field regression along the OT path ``x_t = (1-t) eps + t z``,
  * logit-normal timestep sampling with a **length-aware shift**,
  * the three-stage schedule (PT at 256 tokens -> CT at 4096 -> SFT on a curated
    subset) is expressed here as a schedule over ``(max_latents, lr)`` so a mini
    run can traverse all three stages in one job.

    torchrun --nproc_per_node=4 scripts/train_dit.py --config configs/dit_mini.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seed3d_mini.data.dataset import build_latent_loader  # noqa: E402
from seed3d_mini.flow_matching import flow_matching_loss, sample_timesteps  # noqa: E402
from seed3d_mini.models.conditioner import ImageConditioner  # noqa: E402
from seed3d_mini.models.dit import Seed3DDiT  # noqa: E402
from seed3d_mini.train_utils import (  # noqa: E402
    MetricLogger,
    cosine_with_warmup,
    init_distributed,
    load_checkpoint,
    reduce_metrics,
    save_checkpoint,
    set_seed,
)


def build_modules(cfg: Dict, device: torch.device):
    dcfg = dict(cfg["dit"])
    dcfg.pop("_target", None)
    dit = Seed3DDiT(**dcfg)
    ccfg = dict(cfg["conditioner"])
    ccfg.pop("_target", None)
    cond = ImageConditioner(**ccfg)
    return dit.to(device), cond.to(device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    tcfg = cfg["training"]
    if args.steps:
        tcfg["steps"] = args.steps
    if args.batch_size:
        tcfg["batch_size"] = args.batch_size
    if args.lr:
        tcfg["lr"] = args.lr
    if args.out:
        tcfg["output_dir"] = args.out
    if args.dry_run:
        tcfg.update({"steps": 2, "batch_size": 2, "num_workers": 0, "log_every": 1, "save_every": 10000})
        tcfg["output_dir"] = str(ROOT / "artifacts" / "dryrun_dit")

    ctx = init_distributed()
    set_seed(int(tcfg.get("seed", 0)), ctx.rank)
    logger = MetricLogger(Path(tcfg["output_dir"]) / "train_dit.jsonl", is_main=ctx.is_main)
    if ctx.is_main:
        Path(tcfg["output_dir"]).mkdir(parents=True, exist_ok=True)
        print(json.dumps(cfg, indent=1, default=str), flush=True)

    dit, cond = build_modules(cfg, ctx.device)
    if ctx.is_main:
        trainable = sum(p.numel() for p in dit.parameters() if p.requires_grad)
        cond_p = sum(p.numel() for p in cond.parameters())
        print(f"DiT trainable: {trainable / 1e6:.2f}M ; conditioner (frozen encoders): {cond_p / 1e6:.2f}M", flush=True)
    if ctx.distributed:
        dit = torch.nn.parallel.DistributedDataParallel(dit, device_ids=[ctx.local_rank], find_unused_parameters=False)

    dcfg = cfg["data"]
    train_loader = build_latent_loader(
        dcfg["latent_shard_dir"],
        "train",
        batch_size=tcfg["batch_size"],
        num_workers=tcfg.get("num_workers", 8),
        image_size=dcfg.get("image_size", 224),
        max_latents=dcfg.get("max_latents"),
        seed=int(tcfg.get("seed", 0)),
    )

    opt = torch.optim.AdamW(
        [p for p in dit.parameters() if p.requires_grad],
        lr=tcfg["lr"],
        betas=tuple(tcfg.get("betas", (0.9, 0.99))),
        weight_decay=tcfg.get("weight_decay", 0.0),
        eps=1e-6,
    )
    use_amp = bool(tcfg.get("amp", True)) and ctx.device.type == "cuda"
    amp_dtype = torch.bfloat16 if tcfg.get("amp_dtype", "bf16") == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    start_step = 0
    if args.resume:
        info = load_checkpoint(args.resume, dit, opt, strict=False)
        start_step = int(info.get("step", 0))
        if ctx.is_main:
            print(f"resumed from {args.resume} at step {start_step}", flush=True)

    steps = int(tcfg["steps"])
    stage_schedule = cfg.get("stage_schedule")
    out_dir = Path(tcfg["output_dir"])
    log_every = int(tcfg.get("log_every", 50))
    save_every = int(tcfg.get("save_every", 2000))
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    accum = int(tcfg.get("grad_accum", 1))
    fm = cfg.get("flow_matching", {})
    cfg_drop_prob = float(tcfg.get("cfg_drop_prob", 0.1))

    def stage_params(step: int):
        if not stage_schedule:
            return tcfg["lr"], dcfg.get("max_latents")
        for st in stage_schedule:
            if step < st["until"]:
                return float(st.get("lr", tcfg["lr"])), st.get("max_latents", dcfg.get("max_latents"))
        last = stage_schedule[-1]
        return float(last.get("lr", tcfg["lr"])), last.get("max_latents", dcfg.get("max_latents"))

    model = dit
    model.train()
    cond.eval()
    step = start_step
    epoch = 0
    t0 = time.time()
    data_iter = iter(train_loader)
    while step < steps:
        lr_stage, max_latents = stage_params(step)
        opt.zero_grad(set_to_none=True)
        micro: Dict[str, float] = {}
        for _ in range(accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                if hasattr(train_loader.dataset, "set_epoch"):
                    train_loader.dataset.set_epoch(epoch)
                data_iter = iter(train_loader)
                batch = next(data_iter)
            latents = batch["latent"].to(ctx.device, non_blocking=True)
            images = batch["image"].to(ctx.device, non_blocking=True)
            if max_latents and latents.shape[1] > max_latents:
                latents = latents[:, :max_latents]
            n_tok = latents.shape[1]

            drop = torch.rand(latents.shape[0], device=ctx.device) < cfg_drop_prob
            with torch.no_grad():
                context = cond(images, drop_mask=drop)

            t = sample_timesteps(
                latents.shape[0],
                n_tok,
                device=ctx.device,
                dtype=torch.float32,
                mean=float(fm.get("logit_normal_mean", 0.0)),
                std=float(fm.get("logit_normal_std", 1.0)),
                base_tokens=int(fm.get("base_tokens", 256)),
                max_tokens=int(fm.get("max_tokens", 4096)),
                base_shift=float(fm.get("base_shift", 0.5)),
                max_shift=float(fm.get("max_shift", 1.15)),
            )
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out = flow_matching_loss(model, latents, t, model_kwargs={"context": context})
                loss = out["loss"].mean() / accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            micro["loss"] = micro.get("loss", 0.0) + float(loss.item())
            micro["t_mean"] = micro.get("t_mean", 0.0) + float(t.mean().item()) / accum
            micro["n_tokens"] = float(n_tok)

        lr = cosine_with_warmup(
            step,
            total_steps=steps,
            warmup_steps=int(tcfg.get("warmup_steps", max(1, steps // 50))),
            base_lr=lr_stage,
            min_lr_ratio=float(tcfg.get("min_lr_ratio", 0.05)),
        )
        for g in opt.param_groups:
            g["lr"] = lr
        if scaler.is_enabled():
            scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if scaler.is_enabled():
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()

        micro["lr"] = lr
        micro["grad_norm"] = float(grad_norm)
        micro["step_time"] = (time.time() - t0) / max(1, step - start_step + 1)
        if (step + 1) % log_every == 0 or step == start_step:
            logger.log(step, reduce_metrics({k: torch.tensor(v) for k, v in micro.items()}, ctx))

        if (step + 1) % save_every == 0 or step + 1 == steps:
            if ctx.is_main:
                save_checkpoint(
                    out_dir / f"dit_step{step + 1:07d}.pt", model, opt, step=step + 1, epoch=epoch, config=cfg
                )
                save_checkpoint(out_dir / "dit_latest.pt", model, opt, step=step + 1, epoch=epoch, config=cfg)
        step += 1

    if ctx.is_main:
        save_checkpoint(out_dir / "dit_final.pt", model, opt, step=steps, epoch=epoch, config=cfg)
        print(f"done in {(time.time() - t0) / 60:.1f} min -> {out_dir}", flush=True)
    if ctx.distributed:
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
