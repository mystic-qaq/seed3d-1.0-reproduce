"""Train Seed3D-VAE (stage 1 of the shape-generation pipeline).

Report recipe implemented here (Sec. 2.1.1 + 4.1):
  * TSDF reconstruction loss + KL regularisation,
  * KL warm-up: ``lambda_KL`` annealed from ~0 to its target (1e-4),
  * multi-scale latent length: the number of latent tokens ``M`` is drawn from
    ``{256, 512, ..., 4096}`` every step (the vector-set decoder is length
    agnostic, so this is a free generalisation win and is what lets the DiT be
    trained at a different token count than the VAE),
  * optional eikonal term (unit-norm decoder gradient) which turns the regressed
    TSDF into a proper continuous SDF usable for hierarchical iso-extraction.

Single-GPU and multi-GPU (torchrun) both work:

    torchrun --nproc_per_node=4 scripts/train_vae.py --config configs/vae_mini.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seed3d_mini.data.dataset import build_sdf_loader  # noqa: E402
from seed3d_mini.models.vae import Seed3DVAE  # noqa: E402
from seed3d_mini.parallel import wrap_data_parallel  # noqa: E402
from seed3d_mini.train_utils import (  # noqa: E402
    MetricLogger,
    StepProfiler,
    cosine_with_warmup,
    init_distributed,
    load_checkpoint,
    reduce_metrics,
    save_checkpoint,
    set_seed,
)


def build_model(cfg: Dict) -> Seed3DVAE:
    model_cfg = dict(cfg["model"])
    model_cfg.pop("_target", None)
    return Seed3DVAE(**model_cfg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=None, help="override training.steps")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--out", default=None, help="override training.output_dir")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="1 step, no checkpoint, tiny data")
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
        tcfg["steps"] = 2
        tcfg["batch_size"] = 2
        tcfg["num_workers"] = 0
        tcfg["log_every"] = 1
        tcfg["save_every"] = 10_000
        tcfg["output_dir"] = str(ROOT / "artifacts" / "dryrun_vae")

    ctx = init_distributed()
    set_seed(int(tcfg.get("seed", 0)), ctx.rank)
    logger = MetricLogger(Path(tcfg["output_dir"]) / "train_vae.jsonl", is_main=ctx.is_main)
    if ctx.is_main:
        Path(tcfg["output_dir"]).mkdir(parents=True, exist_ok=True)
        print(json.dumps({k: v for k, v in cfg.items() if k != "model"}, indent=1, default=str), flush=True)
        print("model config:", json.dumps(cfg["model"], indent=1), flush=True)

    # fail loudly (and early) if the data/model point-count contract is broken --
    # the encoder splits the cloud into [pc_size, pc_sharpedge_size] and a silent
    # mismatch only shows up as a confusing tensor-split error mid-training
    if cfg["model"]["pc_size"] != cfg["data"]["pc_size"] or cfg["model"].get(
        "pc_sharpedge_size", 0
    ) != cfg["data"].get("pc_sharpedge_size", 0):
        raise SystemExit(
            "config mismatch: model.pc_size/pc_sharpedge_size "
            f"({cfg['model']['pc_size']}/{cfg['model'].get('pc_sharpedge_size', 0)}) must equal "
            f"data.pc_size/pc_sharpedge_size ({cfg['data']['pc_size']}/{cfg['data'].get('pc_sharpedge_size', 0)})"
        )

    model = build_model(cfg).to(ctx.device)
    if ctx.is_main:
        print(f"VAE parameters: {model.num_parameters() / 1e6:.2f}M", flush=True)
    dp_mode = str(tcfg.get("data_parallel", "flat"))
    model = wrap_data_parallel(model, mode=dp_mode, device=ctx.device)
    if ctx.is_main and ctx.distributed:
        print(f"data parallel: {dp_mode}", flush=True)

    dcfg = cfg["data"]
    train_loader = build_sdf_loader(
        dcfg["shard_dir"],
        "train",
        batch_size=tcfg["batch_size"],
        num_workers=tcfg.get("num_workers", 8),
        pc_size=dcfg["pc_size"],
        pc_sharpedge_size=dcfg.get("pc_sharpedge_size", 0),
        num_queries=dcfg["num_queries"],
        max_samples=args.max_samples,
        seed=int(tcfg.get("seed", 0)),
    )
    val_loader = None
    if dcfg.get("val_shards", True):
        try:
            val_loader = build_sdf_loader(
                dcfg["shard_dir"],
                "val",
                batch_size=tcfg["batch_size"],
                num_workers=max(2, tcfg.get("num_workers", 8) // 2),
                pc_size=dcfg["pc_size"],
                pc_sharpedge_size=dcfg.get("pc_sharpedge_size", 0),
                num_queries=dcfg["num_queries"],
                max_samples=args.max_samples,
                seed=int(tcfg.get("seed", 0)) + 1,
            )
        except FileNotFoundError:
            val_loader = None

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["lr"],
        betas=tuple(tcfg.get("betas", (0.9, 0.99))),
        weight_decay=tcfg.get("weight_decay", 0.01),
        eps=1e-6,
    )
    use_amp = bool(tcfg.get("amp", True)) and ctx.device.type == "cuda"
    amp_dtype = torch.bfloat16 if tcfg.get("amp_dtype", "bf16") == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    prof = StepProfiler(enabled=os.environ.get("SEED3D_PROFILE") == "1")

    start_step = 0
    if args.resume:
        info = load_checkpoint(args.resume, model, opt, strict=False, map_location="cpu")
        start_step = int(info.get("step", 0))
        if ctx.is_main:
            print(f"resumed from {args.resume} at step {start_step}", flush=True)

    steps = int(tcfg["steps"])
    out_dir = Path(tcfg["output_dir"])
    kl_target = float(tcfg.get("lambda_kl", cfg["model"].get("lambda_kl", 1e-4)))
    kl_warmup = int(tcfg.get("kl_warmup_steps", max(1, steps // 20)))
    eik_target = float(cfg["model"].get("lambda_eikonal", 0.0))
    eik_warmup = int(tcfg.get("eikonal_warmup_steps", max(1, steps // 10)))
    log_every = int(tcfg.get("log_every", 50))
    save_every = int(tcfg.get("save_every", 2000))
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    accum = int(tcfg.get("grad_accum", 1))

    model.train()
    step = start_step
    epoch = 0
    t0 = time.time()
    data_iter = iter(train_loader)
    while step < steps:
        # ---- gradient accumulation micro-steps ---------------------------- #
        opt.zero_grad(set_to_none=True)
        micro: Dict[str, float] = {}
        for _ in range(accum):
            with prof.phase("data"):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    epoch += 1
                    if hasattr(train_loader.dataset, "set_epoch"):
                        train_loader.dataset.set_epoch(epoch)
                    data_iter = iter(train_loader)
                    batch = next(data_iter)

            with prof.phase("h2d"):
                pc = batch["pc"].to(ctx.device, non_blocking=True)
                feats = batch["feats"].to(ctx.device, non_blocking=True)
                q = batch["query"].to(ctx.device, non_blocking=True)
                sdf = batch["sdf"].to(ctx.device, non_blocking=True)
                w = batch["weight"].to(ctx.device, non_blocking=True)

            lam_kl = kl_target * min(1.0, (step + 1) / max(kl_warmup, 1))
            # The eikonal term is worth ~1e4 at initialisation (a random field has
            # huge gradients), so it is warmed up as well; otherwise it dominates
            # the reconstruction loss for the first few thousand steps.
            if eik_target > 0:
                raw = model.module if hasattr(model, "module") else model
                raw.lambda_eikonal = eik_target * min(1.0, (step + 1) / max(eik_warmup, 1))
            with prof.phase("fwd_bwd"):
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    out = model(pc, q, sdf, feats=feats, query_weights=w, lambda_kl=lam_kl)
                    loss = out.loss / accum
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            for k, v in out.loss_dict.items():
                micro[k] = micro.get(k, 0.0) + float(v.item()) / accum
            micro["total"] = micro.get("total", 0.0) + float(loss.item())

        lr = cosine_with_warmup(
            step,
            total_steps=steps,
            warmup_steps=int(tcfg.get("warmup_steps", max(1, steps // 50))),
            base_lr=tcfg["lr"],
            min_lr_ratio=float(tcfg.get("min_lr_ratio", 0.05)),
        )
        for g in opt.param_groups:
            g["lr"] = lr
        with prof.phase("optim"):
            if scaler.is_enabled():
                scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if hasattr(model, "sync_grads"):
                model.sync_grads()
            if scaler.is_enabled():
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()

        micro["lr"] = lr
        micro["grad_norm"] = float(grad_norm)
        micro["lambda_kl"] = lam_kl
        micro["step_time"] = (time.time() - t0) / max(1, (step - start_step + 1))

        with prof.phase("log"):
            if (step + 1) % log_every == 0 or step == start_step:
                logger.log(step, reduce_metrics({k: torch.tensor(v) for k, v in micro.items()}, ctx))
        prof.step_done()

        if val_loader is not None and (step + 1) % int(tcfg.get("val_every", 500)) == 0:
            model.eval()
            with torch.no_grad():
                vbatch = next(iter(val_loader))
                pc = vbatch["pc"].to(ctx.device)
                feats = vbatch["feats"].to(ctx.device)
                q = vbatch["query"].to(ctx.device)
                sdf = vbatch["sdf"].to(ctx.device)
                w = vbatch["weight"].to(ctx.device)
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    vout = model(pc, q, sdf, feats=feats, query_weights=w, lambda_kl=kl_target, compute_eikonal=False)
                vmetrics = {f"val_{k}": v for k, v in vout.loss_dict.items()}
                vmetrics["val_total"] = vout.loss.detach()
                # near-surface / far-field split tells us whether the surface band
                # is actually being learned (the far field is trivially easy)
                err = (vout.sdf_pred - sdf).abs()
                near = (sdf.abs() < 0.99 * sdf.abs().max()).float()
                vmetrics["val_near_l1"] = (err * near).sum() / near.sum().clamp_min(1.0)
                vmetrics["val_far_l1"] = (err * (1 - near)).sum() / (1 - near).sum().clamp_min(1.0)
            logger.log(step, reduce_metrics(vmetrics, ctx), prefix="val")
            model.train()

        if (step + 1) % save_every == 0 or step + 1 == steps:
            if ctx.is_main:
                save_checkpoint(
                    out_dir / f"vae_step{step + 1:07d}.pt",
                    model,
                    opt,
                    step=step + 1,
                    epoch=epoch,
                    config=cfg,
                )
                save_checkpoint(out_dir / "vae_latest.pt", model, opt, step=step + 1, epoch=epoch, config=cfg)
        step += 1

    if ctx.is_main:
        save_checkpoint(out_dir / "vae_final.pt", model, opt, step=steps, epoch=epoch, config=cfg)
        print(f"done in {(time.time() - t0) / 60:.1f} min -> {out_dir}", flush=True)
    if ctx.distributed:
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
