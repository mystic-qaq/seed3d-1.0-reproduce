"""Evaluation for the mini Seed3D reproduction.

Three levels, cheapest first:

1. **VAE reconstruction** -- encode the ground-truth surface samples, decode a
   volume, extract a mesh, and compare it against the ground-truth mesh:
   Chamfer-L1/L2 (both directions), F-score at 1%/2% of the object scale,
   volumetric IoU and normal consistency.  This isolates the codec.
2. **DiT generation** -- sample the latent set from the reference image and run
   the same metrics.  This isolates the generator (VAE error is included).
3. **Latent-space diagnostics** -- reconstruction error of the DiT auto-encoder
   path (x -> velocity field), which is cheap and tracks training progress.

    python scripts/evaluate.py --vae checkpoints/vae_mini/vae_final.pt \
        --config configs/vae_mini.yaml --level vae --n 20 --out artifacts/eval
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seed3d_mini.mesh_ops import MeshField, load_mesh, normalize_mesh, remesh_watertight, RemeshConfig  # noqa: E402
from seed3d_mini.volume_decode import latents_to_mesh  # noqa: E402


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def sample_mesh_points(mesh, n: int, rng: np.random.Generator) -> np.ndarray:
    import trimesh

    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=int(rng.integers(1 << 31)))
    return np.asarray(pts, dtype=np.float64)


def chamfer(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    from scipy.spatial import cKDTree

    tp, tg = cKDTree(pred), cKDTree(gt)
    d_pg, _ = tg.query(pred, k=1, workers=-1)
    d_gp, _ = tp.query(gt, k=1, workers=-1)
    return float(d_pg.mean() + d_gp.mean()), float((d_pg ** 2).mean() + (d_gp ** 2).mean())


def fscore(pred: np.ndarray, gt: np.ndarray, threshold: float) -> float:
    from scipy.spatial import cKDTree

    d_pg, _ = cKDTree(gt).query(pred, k=1, workers=-1)
    d_gp, _ = cKDTree(pred).query(gt, k=1, workers=-1)
    precision = float((d_pg < threshold).mean())
    recall = float((d_gp < threshold).mean())
    return 2 * precision * recall / max(precision + recall, 1e-9)


def voxel_iou(pred_mesh, gt_mesh, resolution: int = 48, bound: float = 1.0) -> float:
    """Volumetric IoU on an occupancy grid (the report's simulation-readiness proxy)."""
    lin = np.linspace(-bound, bound, resolution)
    grid = np.stack(np.meshgrid(lin, lin, lin, indexing="ij"), axis=-1).reshape(-1, 3)
    occ_p = pred_mesh.contains(grid)
    occ_g = gt_mesh.contains(grid)
    inter = np.logical_and(occ_p, occ_g).sum()
    union = np.logical_or(occ_p, occ_g).sum()
    return float(inter / max(union, 1))


def surface_to_mesh(points: np.ndarray, normals: np.ndarray, resolution: int = 96):
    """Rebuild a watertight mesh from the stored ground-truth surface samples.

    Used as the IoU reference: the shards deliberately store *samples*, not meshes,
    so that the training data is small; for evaluation we voxelise them back.
    """
    import trimesh
    from seed3d_mini.mesh_ops import _sdf_grids, _surface_nets

    cloud = trimesh.Trimesh(vertices=points.astype(np.float64), process=False)
    lo = np.minimum(points.min(axis=0), -1.0)
    hi = np.maximum(points.max(axis=0), 1.0)
    extent = float(np.max(hi - lo))
    voxel = extent / resolution
    shape = np.maximum(np.ceil((hi - lo) / voxel).astype(int) + 3, 4)
    origin = lo - voxel
    dist, inside = _sdf_grids(cloud, shape, origin, voxel, trim_faces=200000)
    field = np.where(inside, -1.0, 1.0) * np.minimum(dist, 1.5 * voxel).astype(np.float32)
    verts, faces = _surface_nets(field, level=0.0)
    return trimesh.Trimesh(vertices=verts * voxel + origin, faces=faces, process=True)


def normal_consistency(pred_mesh, gt_mesh, n: int, rng: np.random.Generator) -> float:
    import trimesh

    pts, fidx = trimesh.sample.sample_surface(pred_mesh, n, seed=int(rng.integers(1 << 31)))
    pn = pred_mesh.face_normals[fidx]
    _, dist, gt_fidx = trimesh.proximity.closest_point(gt_mesh, pts)
    gn = gt_mesh.face_normals[gt_fidx]
    cos = np.abs((pn * gn).sum(axis=-1))
    return float(cos.mean())


# --------------------------------------------------------------------------- #
def _fit_cloud(vae, points: np.ndarray, normals: np.ndarray, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Trim/pad a surface cloud to the exact size the encoder expects.

    The encoder splits the cloud into ``[pc_size, pc_sharpedge_size]``, so an
    evaluator pointed at a shard with a different cloud size must adapt rather than
    crash (the shipped shards match the config, but this keeps the script reusable).
    """
    want = int(vae.encoder.pc_size) + int(vae.encoder.pc_sharpedge_size)
    pc = torch.from_numpy(points)[None].to(device)
    feats = torch.from_numpy(normals)[None].to(device)
    if pc.shape[1] > want:
        pc, feats = pc[:, :want], feats[:, :want]
    elif pc.shape[1] < want:
        pad = want - pc.shape[1]
        pc = torch.cat([pc, pc.new_zeros(1, pad, 3)], dim=1)
        feats = torch.cat([feats, feats.new_zeros(1, pad, feats.shape[-1])], dim=1)
    return pc, feats


def load_vae(config_path: str, ckpt: str, device: torch.device):
    from seed3d_mini.models.vae import Seed3DVAE

    cfg = yaml.safe_load(Path(config_path).read_text())
    mcfg = dict(cfg["model"])
    mcfg.pop("_target", None)
    model = Seed3DVAE(**mcfg)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    return model.to(device).eval(), cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="vae", choices=["vae", "dit", "both"])
    ap.add_argument("--vae-config", default=str(ROOT / "configs" / "vae_mini.yaml"))
    ap.add_argument("--vae", default=None)
    ap.add_argument("--dit-config", default=str(ROOT / "configs" / "dit_mini.yaml"))
    ap.add_argument("--dit", default=None)
    ap.add_argument("--shard", default=None, help="npz shard with ground-truth surface points")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--resolution", type=int, default=160)
    ap.add_argument("--coarse", type=int, default=64)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--index-offset", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "eval"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-meshes", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard = args.shard or str(sorted((ROOT / "data" / "shards").glob("val-*.npz"))[0])
    data = dict(np.load(shard, allow_pickle=False))
    n = min(args.n, data["surface"].shape[0])
    print(f"evaluating {n} samples from {shard} on {device}")

    vae, vcfg = None, None
    if args.level in ("vae", "both") and args.vae:
        vae, vcfg = load_vae(args.vae_config, args.vae, device)
        print(f"VAE: {vae.num_parameters() / 1e6:.2f}M params")

    dit = cond = None
    image_size = 512
    if args.level in ("dit", "both") and args.dit:
        from seed3d_mini.models.conditioner import ImageConditioner
        from seed3d_mini.models.dit import Seed3DDiT

        dcfg = yaml.safe_load(Path(args.dit_config).read_text())
        dc = dict(dcfg["dit"]); dc.pop("_target", None)
        cc = dict(dcfg["conditioner"]); cc.pop("_target", None)
        dit = Seed3DDiT(**dc)
        dit.load_state_dict(torch.load(args.dit, map_location="cpu", weights_only=False)["model"])
        dit = dit.to(device).eval()
        cond = ImageConditioner(**cc).to(device).eval()
        image_size = int(dcfg["data"].get("image_size", 512))
        print(f"DiT: {sum(p.numel() for p in dit.parameters()) / 1e6:.2f}M params")

    rng = np.random.default_rng(0)
    rows: List[Dict] = []
    for i in range(n):
        idx = args.index_offset + i
        gt_pts = data["surface"][idx].astype(np.float32)
        gt_nrm = data["surface_normal"][idx].astype(np.float32)
        morph = data["query"][idx].astype(np.float32)  # noqa: F841 (kept for parity with the dataset)
        row: Dict = {"index": int(idx), "category": str(data["category"][idx])}
        gt_mesh = None
        t0 = time.time()

        with torch.no_grad():
            if vae is not None:
                pc, feats = _fit_cloud(vae, gt_pts, gt_nrm, device)
                z, _ = vae.encode(pc, feats, sample=False)
                mesh = latents_to_mesh(
                    z, lambda q, zz: vae.decode(q, zz), mode="hierarchical",
                    resolution=args.resolution, coarse_resolution=args.coarse, device=device,
                )
            elif dit is not None:
                from seed3d_mini.flow_matching import euler_sample

                im = data["image"][idx]
                if im.ndim == 4:
                    im = im[0]
                t = torch.from_numpy(np.ascontiguousarray(im)).float()
                if t.max() > 1.5:
                    t = t / 255.0
                if t.shape[0] != 3:
                    t = t.permute(2, 0, 1)
                t = torch.nn.functional.interpolate(
                    t[None], size=(image_size, image_size), mode="bilinear", align_corners=False
                ).to(device)
                ctx = cond(t)
                unc = cond.unconditional(1, device, ctx.dtype)
                z = euler_sample(
                    dit, (1, int(dcfg["dit"]["num_latents"]), int(dcfg["dit"]["in_channels"])),
                    num_steps=args.steps, device=device, model_kwargs={"context": ctx},
                    uncond_kwargs={"context": unc}, guidance_scale=args.guidance,
                )
                mesh = latents_to_mesh(
                    z, lambda q, zz: vae.decode(q, zz), mode="hierarchical",
                    resolution=args.resolution, coarse_resolution=args.coarse, device=device,
                )
            else:
                raise SystemExit("no model selected")

        row["seconds"] = round(time.time() - t0, 2)
        row["pred_vertices"] = int(mesh.vertices.shape[0])
        row["pred_faces"] = int(mesh.faces.shape[0])
        row["pred_watertight"] = bool(mesh.is_watertight)

        # metrics on surface samples (resolution-independent)
        pred_pts = sample_mesh_points(mesh, 8192, rng)
        scale = 1.0  # both frames are canonical [-1, 1]
        cd, cd2 = chamfer(pred_pts, gt_pts.astype(np.float64))
        row["chamfer_l1"] = cd
        row["chamfer_l2"] = cd2
        row["fscore_1pct"] = fscore(pred_pts, gt_pts.astype(np.float64), 0.01 * scale)
        row["fscore_2pct"] = fscore(pred_pts, gt_pts.astype(np.float64), 0.02 * scale)
        try:
            if gt_mesh is None:
                gt_mesh = surface_to_mesh(gt_pts, gt_nrm)
            row["iou"] = voxel_iou(mesh, gt_mesh, resolution=48)
            row["normal_consistency"] = normal_consistency(mesh, gt_mesh, 4096, rng)
        except Exception as exc:  # noqa: BLE001
            row["iou"] = float("nan")
            row["iou_error"] = f"{type(exc).__name__}: {exc}"[:160]
        rows.append(row)
        print(
            f"[{i + 1}/{n}] idx={idx} {row['category']:9s} CD-L1={cd:.4f} F@1%={row['fscore_1pct']:.3f} "
            f"watertight={row['pred_watertight']} ({row['seconds']}s)",
            flush=True,
        )
        if args.save_meshes:
            mesh.export(out_dir / f"pred_{idx:04d}_{row['category']}.glb")

    summary = {
        "n": len(rows),
        "shard": str(shard),
        "level": args.level,
        "resolution": args.resolution,
        "iou_note": "nan when the GT mesh could not be rebuilt from the stored samples",
        "chamfer_l1": float(np.nanmean([r["chamfer_l1"] for r in rows])),
        "chamfer_l2": float(np.nanmean([r["chamfer_l2"] for r in rows])),
        "fscore_1pct": float(np.nanmean([r["fscore_1pct"] for r in rows])),
        "fscore_2pct": float(np.nanmean([r["fscore_2pct"] for r in rows])),
        "iou": float(np.nanmean([r["iou"] for r in rows])) if np.isfinite([r["iou"] for r in rows]).any() else float("nan"),
        "normal_consistency": float(np.nanmean([r.get("normal_consistency", float("nan")) for r in rows])),
        "watertight_rate": float(np.mean([r["pred_watertight"] for r in rows])),
        "seconds_per_sample": float(np.mean([r["seconds"] for r in rows])),
    }
    by_cat: Dict[str, List[float]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["chamfer_l1"])
    summary["chamfer_l1_by_category"] = {k: float(np.mean(v)) for k, v in by_cat.items()}

    (out_dir / f"eval_{args.level}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print("\n=== summary ===")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
