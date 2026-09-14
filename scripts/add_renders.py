"""Backfill reference-view images into existing data shards.

The SDF shards only need geometry; the *reference image* belongs to the DiT's
conditioning data.  Keeping the two separable means the same geometry shards can
feed different conditioning sets, but it also means the images have to be added
before DiT training.  This script renders them, in parallel processes (rendering
is pure-python per-triangle work, so threads do not help), and rewrites the shards
in place with an added ``image`` array.

    python scripts/add_renders.py --shard-dir data/shards --workers 32 --resolution 256
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# module-level so the process pool can pickle it
_INDEX: Dict[str, Path] = {}


def _init_worker(raw_dir: str) -> None:
    global _INDEX
    _INDEX = {p.stem: p for p in Path(raw_dir).rglob("*.off")}


def _render_one(job: Tuple[str, int, int]) -> np.ndarray:
    """Render a single reference view for sample ``sid`` (runs in a worker)."""
    sid, resolution, max_faces = job
    name = str(sid).split("/")[-1]
    path = _INDEX.get(name)
    if path is None:
        raise FileNotFoundError(f"raw mesh {name} not found")
    from seed3d_mini.mesh_ops import load_mesh, normalize_mesh
    from seed3d_mini.render.rasterizer import render_mesh_views

    mesh = normalize_mesh(load_mesh(str(path)))
    if max_faces and mesh.faces.shape[0] > max_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=max_faces)
        except Exception:
            pass
    views = render_mesh_views(mesh, n_views=1, resolution=resolution, seed=abs(hash(sid)) % (2**31))
    return (views["rgb"][0] * 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", default=str(ROOT / "data" / "shards"))
    ap.add_argument("--raw-dir", default=str(ROOT / "data" / "raw" / "modelnet40"))
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-faces", type=int, default=6000)
    ap.add_argument("--splits", default="train,val")
    args = ap.parse_args()

    shard_dir = Path(args.shard_dir)
    for split in args.splits.split(","):
        for shard in sorted(shard_dir.glob(f"{split}-*.npz")):
            data = dict(np.load(shard, allow_pickle=False))
            if "image" in data:
                print(f"{shard.name}: images already present, skipping")
                continue
            ids = [str(s) for s in data["sample_id"]]
            t0 = time.time()
            jobs = [(s, args.resolution, args.max_faces) for s in ids]
            with cf.ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(args.raw_dir,)) as ex:
                images: List[np.ndarray] = list(ex.map(_render_one, jobs, chunksize=4))
            data["image"] = np.stack(images, axis=0)
            tmp = shard.with_suffix(".tmp.npz")
            np.savez(tmp, **data)
            tmp.replace(shard)
            print(
                f"{shard.name}: rendered {len(images)} views in {time.time() - t0:.1f}s -> {shard}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
