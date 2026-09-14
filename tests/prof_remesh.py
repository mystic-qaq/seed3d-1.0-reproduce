"""Profiling helper: time each remeshing stage on a real asset (dev tool)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seed3d_mini.mesh_ops import (  # noqa: E402
    InsideTest,
    MeshField,
    RemeshConfig,
    TriangleDistance,
    _sample_triangles,
    _sdf_grids,
    load_mesh,
    mesh_stats,
    normalize_mesh,
    remesh_watertight,
)

PATH = sys.argv[1] if len(sys.argv) > 1 else str(
    ROOT / "data" / "raw" / "modelnet40" / "ModelNet40" / "airplane" / "train" / "airplane_0001.off"
)
RES = int(sys.argv[2]) if len(sys.argv) > 2 else 96


def timed(label, fn):
    t = time.time()
    out = fn()
    print(f"  {label:28s} {time.time() - t:7.2f}s", flush=True)
    return out


def main() -> int:
    m = normalize_mesh(load_mesh(PATH))
    print(f"mesh: {m.faces.shape[0]} faces, watertight={m.is_watertight}", flush=True)
    print("stats:", mesh_stats(m), flush=True)

    work = m
    if work.faces.shape[0] > 60000:
        work = timed("decimate", lambda: work.simplify_quadric_decimation(face_count=60000))
        print(f"  -> {work.faces.shape[0]} faces", flush=True)

    tri = TriangleDistance(work, measured=True)
    rng = np.random.default_rng(0)
    samples = timed("sample triangles", lambda: _sample_triangles(work, 200000, rng=rng))
    pts = samples[:20000]
    timed("TriangleDistance 20k", lambda: tri.distance(pts))
    tester = InsideTest(work)
    timed("InsideTest 20k", lambda: tester.inside(pts))

    lo = np.minimum(m.bounds[0], -1.0)
    hi = np.maximum(m.bounds[1], 1.0)
    voxel = float(np.max(hi - lo)) / RES
    shape = np.maximum(np.ceil((hi - lo) / voxel).astype(int) + 3, 4)
    origin = lo - voxel
    timed("sdf grids", lambda: _sdf_grids(work, shape, origin, voxel))

    remeshed = timed("full remesh", lambda: remesh_watertight(m, RemeshConfig(resolution=RES)))
    print("  ->", mesh_stats(remeshed), flush=True)

    field = MeshField(remeshed, truncation=0.1)
    q = rng.uniform(-1, 1, size=(16384, 3))
    timed("MeshField.sdf 16k", lambda: field.sdf(q))
    timed("sample surface 8k", lambda: __import__("seed3d_mini.mesh_ops", fromlist=["x"]).sample_surface_points(remeshed, 8192, rng=rng))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
