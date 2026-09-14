"""Image -> mesh inference for the mini Seed3D reproduction.

Mirrors the report's geometry stage (Fig. 7, first two stages):

    reference image -> Seed3D-DiT (40-step deterministic ODE, CFG)
                    -> latent token set Z
                    -> Seed3D-VAE decoder over a grid (coarse-to-fine)
                    -> marching cubes -> mesh

Examples
--------
    # render a reference view of a ground-truth asset and reconstruct it
    python scripts/infer.py --image auto --mesh-from-shard data/shards/train-00000.npz --index 3

    # a real image
    python scripts/infer.py --image photo.png --out artifacts/meshes/photo.glb
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seed3d_mini.flow_matching import euler_sample, heun_sample  # noqa: E402
from seed3d_mini.models.conditioner import ImageConditioner  # noqa: E402
from seed3d_mini.models.dit import Seed3DDiT  # noqa: E402
from seed3d_mini.models.vae import Seed3DVAE  # noqa: E402
from seed3d_mini.volume_decode import latents_to_mesh  # noqa: E402


def load_dit(config_path: str, ckpt_path: str, device: torch.device):
    cfg = yaml.safe_load(Path(config_path).read_text())
    dcfg = dict(cfg["dit"]); dcfg.pop("_target", None)
    ccfg = dict(cfg["conditioner"]); ccfg.pop("_target", None)
    dit = Seed3DDiT(**dcfg)
    cond = ImageConditioner(**ccfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = dit.load_state_dict(ckpt["model"], strict=False)
    dit = dit.to(device).eval()
    cond = cond.to(device).eval()
    print(
        f"DiT step={ckpt.get('step')} params={sum(p.numel() for p in dit.parameters())/1e6:.2f}M "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    return dit, cond, cfg


def load_vae(config_path: str, ckpt_path: str, device: torch.device):
    cfg = yaml.safe_load(Path(config_path).read_text())
    mcfg = dict(cfg["model"]); mcfg.pop("_target", None)
    vae = Seed3DVAE(**mcfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    vae.load_state_dict(ckpt["model"])
    vae = vae.to(device).eval()
    print(f"VAE step={ckpt.get('step')} params={vae.num_parameters()/1e6:.2f}M")
    return vae, cfg


def preprocess_image(image: np.ndarray, size: int) -> torch.Tensor:
    """``HWC uint8/float`` -> ``[1, 3, size, size]`` float in ``[0, 1]``."""
    from PIL import Image
    import torch.nn.functional as F

    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(image).convert("RGB").resize((size, size), Image.BILINEAR)
    t = torch.from_numpy(np.asarray(img)).float().permute(2, 0, 1) / 255.0
    return t.unsqueeze(0).contiguous()


def image_from_shard(shard: str, index: int) -> np.ndarray:
    with np.load(shard, allow_pickle=False) as f:
        imgs = f["image"]
        im = imgs[index]
        if im.ndim == 4:
            im = im[0]
        if im.shape[0] == 3:
            im = np.transpose(im, (1, 2, 0))
    return im


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="path to an image, or 'auto' (render from shard)")
    ap.add_argument("--mesh-from-shard", default=None, help="shard used for the 'auto' reference view")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--vae-config", default=str(ROOT / "configs" / "vae_mini.yaml"))
    ap.add_argument("--vae", required=True)
    ap.add_argument("--dit-config", default=str(ROOT / "configs" / "dit_mini.yaml"))
    ap.add_argument("--dit", required=True)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "meshes" / "infer.glb"))
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--resolution", type=int, default=192)
    ap.add_argument("--coarse", type=int, default=64)
    ap.add_argument("--mode", default="hierarchical", choices=["hierarchical", "dense"])
    ap.add_argument("--sampler", default="euler", choices=["euler", "heun"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-latent", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dit, cond, dcfg = load_dit(args.dit_config, args.dit, device)
    vae, vcfg = load_vae(args.vae_config, args.vae, device)
    image_size = int(dcfg["data"].get("image_size", 224))
    fm = dcfg.get("flow_matching", {})

    if args.image in (None, "auto"):
        if not args.mesh_from_shard:
            ap.error("--mesh-from-shard is required when --image auto")
        img = image_from_shard(args.mesh_from_shard, args.index)
    else:
        from PIL import Image

        img = np.asarray(Image.open(args.image).convert("RGB"))
    images = preprocess_image(img, image_size).to(device)

    t0 = time.time()
    with torch.no_grad():
        context = cond(images)
        uncond = cond.unconditional(images.shape[0], device, context.dtype)
        shape = (1, int(dcfg["dit"]["num_latents"]), int(dcfg["dit"]["in_channels"]))
        sampler = euler_sample if args.sampler == "euler" else heun_sample
        gen = torch.Generator(device=device).manual_seed(args.seed)
        latents = sampler(
            dit,
            shape,
            num_steps=args.steps,
            device=device,
            dtype=torch.float32,
            model_kwargs={"context": context},
            uncond_kwargs={"context": uncond},
            guidance_scale=args.guidance,
            num_tokens=shape[1],
            base_tokens=int(fm.get("base_tokens", 256)),
            max_tokens=int(fm.get("max_tokens", 4096)),
            base_shift=float(fm.get("base_shift", 0.5)),
            max_shift=float(fm.get("max_shift", 1.15)),
            generator=gen,
        )
        t_sample = time.time() - t0
        mesh = latents_to_mesh(
            latents,
            lambda q, z: vae.decode(q, z),
            mode=args.mode,
            resolution=args.resolution,
            coarse_resolution=args.coarse,
            device=device,
        )
    t_total = time.time() - t0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out)
    watertight = bool(mesh.is_watertight)
    meta = {
        "out": str(out),
        "vertices": int(mesh.vertices.shape[0]),
        "faces": int(mesh.faces.shape[0]),
        "watertight": watertight,
        "volume": float(mesh.volume) if watertight else None,
        "sampling_seconds": round(t_sample, 2),
        "total_seconds": round(t_total, 2),
        "steps": args.steps,
        "guidance": args.guidance,
        "resolution": args.resolution,
        "mode": args.mode,
        "seed": args.seed,
    }
    print(json.dumps(meta, indent=1))
    (out.with_suffix(".json")).write_text(json.dumps(meta, indent=1))
    if args.save_latent:
        np.save(out.with_suffix(".latent.npy"), latents.float().cpu().numpy())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
