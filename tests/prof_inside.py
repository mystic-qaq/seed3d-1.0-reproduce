"""Compare inside/outside backends on real ModelNet40 assets (dev tool).

The remesher needs a robust sign field for meshes that are *not* watertight
(ModelNet40 is full of non-manifold, self-intersecting CAD shells).  This script
measures, per backend: wall-clock time, agreement with a high-quality reference,
and whether it survives non-manifold input.

    python tests/prof_inside.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seed3d_mini.mesh_ops import InsideTest, TriangleDistance, load_mesh, normalize_mesh  # noqa: E402

RAW = ROOT / "data" / "raw" / "modelnet40" / "ModelNet40"
ASSETS = [
    RAW / "airplane" / "train" / "airplane_0001.off",
    RAW / "car" / "train" / "car_0001.off",
    RAW / "ship" / "train" / "ship_0001.off",
]


def timed(fn):
    t = time.time()
    out = fn()
    return out, time.time() - t


def main() -> int:
    rng = np.random.default_rng(0)
    for path in ASSETS:
        if not path.exists():
            print(f"skip missing {path}")
            continue
        mesh = normalize_mesh(load_mesh(str(path)))
        print(f"\n=== {path.name}: {mesh.faces.shape[0]} faces watertight={mesh.is_watertight}")
        rep = mesh.copy()
        import trimesh

        trimesh.repair.fill_holes(rep)
        trimesh.repair.fix_normals(rep)
        print(f"    after repair: watertight={rep.is_watertight} consistent={rep.is_winding_consistent}")

        pts = rng.uniform(-1, 1, size=(8192, 3))
        # reference: trimesh's own winding-number test on the repaired mesh
        (ref, t_ref) = timed(lambda: rep.contains(pts) if rep.is_watertight else mesh.contains(pts))
        print(f"    trimesh.contains      {t_ref:6.2f}s  inside={ref.mean():.3f}")

        for rays, cand in ((3, 512), (5, 1024)):
            tester = InsideTest(mesh, num_rays=rays, candidate_faces=cand)
            (got, t_got) = timed(lambda: tester.inside(pts))
            agree = float((got == ref).mean())
            print(f"    InsideTest(r={rays},c={cand}) {t_got:6.2f}s  agree={agree:.3f}")

        tri = TriangleDistance(mesh)
        (d, t_d) = timed(lambda: tri.distance(pts))
        on_surface = d < 5e-3
        print(f"    TriangleDistance      {t_d:6.2f}s  on-surface pts={on_surface.mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
