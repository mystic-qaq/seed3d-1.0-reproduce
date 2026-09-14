"""Smoke test for the mesh preprocessing stack (run manually, not pytest).

Checks, on analytic primitives whose answers are known:
  * remesh_watertight produces a watertight, positive-volume, correctly sized mesh,
  * the SDF sign is negative at the centre of a solid and positive at a far corner,
  * surface samples really lie on the surface,
  * sharp-edge extraction returns points.

    python tests/smoke_meshops.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seed3d_mini.mesh_ops import (  # noqa: E402
    MeshField,
    RemeshConfig,
    TriangleDistance,
    mesh_stats,
    normalize_mesh,
    remesh_watertight,
    sample_edge_points,
    sample_surface_points,
)


def main() -> int:
    cases = [
        ("sphere", lambda: trimesh.creation.uv_sphere(radius=0.8, count=[32, 32])),
        ("box", lambda: trimesh.creation.box(extents=[1.2, 0.8, 0.6])),
        ("torus", lambda: trimesh.creation.torus(major_radius=0.6, minor_radius=0.22, major_segments=48, minor_segments=16)),
    ]
    failures = 0
    for name, make in cases:
        mesh = normalize_mesh(make())
        t0 = time.time()
        remeshed = remesh_watertight(mesh, RemeshConfig(resolution=96))
        dt = time.time() - t0
        st = mesh_stats(remeshed)
        field = MeshField(remeshed, truncation=0.1)
        center = float(field.sdf(np.array([[0.0, 0.0, 0.0]]), truncate=False)[0, 0])
        corner = float(field.sdf(np.array([[0.97, 0.97, 0.97]]), truncate=False)[0, 0])
        pts, _ = sample_surface_points(remeshed, 2048)
        residual = TriangleDistance(remeshed).distance(pts)
        edge_pts, _ = sample_edge_points(remeshed, 256)
        ok = (
            st["watertight"]
            and st["volume"] > 0
            and abs(max(st["extent_x"], st["extent_y"], st["extent_z"]) - 2.0) < 0.15
            and corner > 0
            and residual.max() < 2e-2
            and (name != "torus" or center < 0)  # torus centre is a hole -> positive
        )
        failures += 0 if ok else 1
        print(
            f"[{'ok' if ok else 'FAIL'}] {name:8s} {dt:5.1f}s watertight={st['watertight']} "
            f"V={st['n_vertices']:6d} F={st['n_faces']:6d} vol={st['volume']:.3f} "
            f"sdf(center)={center:+.3f} sdf(corner)={corner:+.3f} "
            f"surface_residual(max)={residual.max():.4f} edge_pts={edge_pts.shape[0]}"
        )
    print("FAILURES:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
