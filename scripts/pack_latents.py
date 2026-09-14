"""Pack frozen-VAE latents for Seed3D-DiT training.

Runs the trained VAE encoder over the SDF shards once and stores

    latent : [N, M, d]   posterior sample in the VAE's *raw* latent space
                         (the space the DiT is trained in; the decoder applies
                         post_kl + its transformer trunk at decode time)
    image  : [N, 3, H, W] uint8 reference view rendered from the same frame
    mu, logvar : [N, M, d]  (kept for the latent statistics / provenance)

so that DiT training never has to run the encoder (and stays cheap).

    python scripts/pack_latents.py --vae checkpoints/vae_mini/vae_final.pt \
        --config configs/vae_mini.yaml --out data/latents_mini
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seed3d_mini.models.vae import Seed3DVAE  # noqa: E402


def render_reference_views(
    sample_ids: np.ndarray,
    raw_dir: str,
    resolution: int,
    *,
    workers: int = 8,
) -> np.ndarray:
    """Render one reference view per sample from the *source* mesh.

    Uses the same canonical frame and the same rasteriser as the preprocessing
    pipeline (`normalize_mesh` + `render_mesh_views` with the report's elevation
    range), so the conditioning distribution matches.  The raw mesh - not the
    remeshed one - is rendered: that is the surface a user would photograph.

    Rendering is CPU-only and independent per sample, so it is threaded; the raw
    index is built once (a per-sample ``rglob`` over 1278 files would dominate).
    """
    import concurrent.futures as cf

    from seed3d_mini.mesh_ops import load_mesh, normalize_mesh
    from seed3d_mini.render.rasterizer import render_mesh_views

    raw_root = Path(raw_dir)
    index = {p.stem: p for p in raw_root.rglob("*.off")}
    if not index:
        raise FileNotFoundError(f"no .off files under {raw_root}; pass --raw-dir")

    done = {"n": 0}
    lock = __import__("threading").Lock()

    def one(sid: str) -> np.ndarray:
        name = str(sid).split("/")[-1]
        path = index.get(name)
        if path is None:
            raise FileNotFoundError(f"raw mesh {name} not found under {raw_root}")
        mesh = normalize_mesh(load_mesh(str(path)))
        # The rasteriser is a per-triangle python loop, so a 100k-face CAD mesh costs
        # seconds per view; decimating to 20k faces is visually indistinguishable at
        # 256-512 px and ~5x faster, which matters because this is CPU-only work.
        if mesh.faces.shape[0] > 20_000:
            try:
                mesh = mesh.simplify_quadric_decimation(face_count=20_000)
            except Exception:
                pass
        views = render_mesh_views(mesh, n_views=1, resolution=resolution, seed=abs(hash(sid)) % (2**31))
        with lock:
            done["n"] += 1
            if done["n"] % 64 == 0:
                print(f"    rendered {done['n']}/{len(sample_ids)}", flush=True)
        return (views["rgb"][0] * 255).astype(np.uint8)

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(one, [str(s) for s in sample_ids]))
    return np.stack(out, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True, help="trained VAE checkpoint (.pt)")
    ap.add_argument("--config", required=True, help="yaml used to build the VAE")
    ap.add_argument("--shard-dir", default=str(ROOT / "data" / "shards"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-latents", type=int, default=None, help="override M at pack time")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument(
        "--render", action="store_true",
        help="render the reference view from the source mesh when the shard has no 'image'"
    )
    ap.add_argument("--raw-dir", default=str(ROOT / "data" / "raw" / "modelnet40"))
    ap.add_argument("--render-resolution", type=int, default=256)
    ap.add_argument("--render-workers", type=int, default=8)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    model_cfg = dict(cfg["model"])
    model_cfg.pop("_target", None)
    model = Seed3DVAE(**model_cfg)
    ckpt = torch.load(args.vae, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"loaded VAE from {args.vae} (step {ckpt.get('step')}), {model.num_parameters() / 1e6:.2f}M params")

    dcfg = cfg["data"]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    stats: Dict[str, float] = {}

    with torch.no_grad():
        for split in args.splits.split(","):
            for shard in sorted(Path(args.shard_dir).glob(f"{split}-*.npz")):
                data = dict(np.load(shard, allow_pickle=False))
                n = data["surface"].shape[0]
                latents: List[np.ndarray] = []
                mus: List[np.ndarray] = []
                logvars: List[np.ndarray] = []
                t0 = time.time()
                for s in range(0, n, args.batch_size):
                    e = min(s + args.batch_size, n)
                    pc = torch.from_numpy(data["surface"][s:e]).to(device)
                    feats = torch.from_numpy(data["surface_normal"][s:e]).to(device)
                    if "edge" in data and data["edge"].shape[1] > 0:
                        pc = torch.cat([pc, torch.from_numpy(data["edge"][s:e]).to(device)], dim=1)
                        feats = torch.cat([feats, torch.from_numpy(data["edge_normal"][s:e]).to(device)], dim=1)
                    # the encoder splits the cloud into
                    # [pc_size, pc_sharpedge_size]: trim/pad so a shard whose cloud
                    # size differs from the model config still works
                    want = model.encoder.pc_size + model.encoder.pc_sharpedge_size
                    if pc.shape[1] > want:
                        pc, feats = pc[:, :want], feats[:, :want]
                    elif pc.shape[1] < want:
                        pad = want - pc.shape[1]
                        pc = torch.cat([pc, pc.new_zeros(pc.shape[0], pad, 3)], dim=1)
                        feats = torch.cat([feats, feats.new_zeros(feats.shape[0], pad, feats.shape[-1])], dim=1)
                    z, post = model.encode(pc, feats, num_latents=args.num_latents, sample=True)
                    latents.append(z.float().cpu().numpy())
                    mus.append(post.mean.float().cpu().numpy())
                    logvars.append(post.logvar.float().cpu().numpy())
                latent = np.concatenate(latents, 0).astype(np.float32)
                payload = {
                    "latent": latent,
                    "mu": np.concatenate(mus, 0).astype(np.float32),
                    "logvar": np.concatenate(logvars, 0).astype(np.float32),
                    "category": data["category"],
                    "sample_id": data["sample_id"],
                }
                if "image" in data:
                    payload["image"] = data["image"]
                elif args.render:
                    # The reference view is part of the *conditioning* data, not the
                    # VAE data, so it is rendered here from the source mesh when the
                    # SDF shards did not carry one (``prepare_data.py --num-views 0``).
                    payload["image"] = render_reference_views(
                        data["sample_id"], args.raw_dir, args.render_resolution, workers=args.render_workers
                    )
                out_path = out_root / shard.name
                np.savez(out_path, **payload)
                std = float(latent.std())
                stats.setdefault("latent_std_sum", 0.0)
                stats["latent_std_sum"] += std
                print(
                    f"  {shard.name}: {n} samples -> {out_path} "
                    f"latent {latent.shape} std={std:.4f} ({time.time() - t0:.1f}s)",
                    flush=True,
                )

    arr = np.load(sorted(out_root.glob("train-*.npz"))[0])["latent"]
    all_std = float(arr.std())
    (out_root / "latent_stats.json").write_text(
        f'{{"latent_std": {all_std}, "num_latents": {arr.shape[1]}, "embed_dim": {arr.shape[2]}, '
        f'"vae_ckpt": "{args.vae}", "step": {ckpt.get("step", 0)}}}'
    )
    print(f"[pack_latents] global latent std = {all_std:.4f}; z_scale_factor = {1.0 / all_std:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
