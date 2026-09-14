"""Compare remeshing strategies on real ModelNet40 assets (dev tool).

The raw meshes are *open, non-manifold CAD shells*, so "make it watertight and
correctly filled" is the hard part of the data pipeline (report Sec. 3.1,
"mesh remeshing").  This script runs each candidate strategy over a sample of real
assets and reports, per strategy: the watertight rate, the interior-volume
fraction (a solid object should fill a plausible share of the unit cube) and the
wall-clock cost.

    python tests/prof_remesh_strategies.py --limit 24
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seed3d_mini.mesh_ops import (  # noqa: E402
    _clean_occupancy,
    _flood_fill_outside,
    _sdf_grids,
    _surface_nets,
    clean_mesh,
    is_watertight,
    load_mesh,
    normalize_mesh,
    repair_mesh,
)

RAW = ROOT / "data" / "raw" / "modelnet40" / "ModelNet40"


def grid_of(mesh, resolution):
    lo = np.minimum(mesh.bounds[0], -1.0)
    hi = np.maximum(mesh.bounds[1], 1.0)
    extent = float(np.max(hi - lo))
    voxel = extent / float(resolution)
    shape = np.maximum(np.ceil((hi - lo) / voxel).astype(int) + 3, 4)
    origin = lo - voxel + 0.5 * voxel
    return shape, origin, voxel


def extract(field, shape, origin, voxel) -> trimesh.Trimesh:
    verts, faces = _surface_nets(field.astype(np.float32), level=0.0)
    mesh = trimesh.Trimesh(vertices=verts * voxel + origin, faces=faces, process=True)
    mesh.merge_vertices()
    mesh = clean_mesh(mesh)
    return mesh


def repair_to_watertight(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    out = mesh.copy()
    trimesh.repair.fill_holes(out)
    trimesh.repair.fix_normals(out)
    out.merge_vertices()
    out = clean_mesh(out)
    if not out.is_watertight:
        # last resort: a slightly larger hole fill via laplacian-consistent repair
        trimesh.repair.fill_holes(out)
    return out


# --------------------------------------------------------------------------- #
# strategies
# --------------------------------------------------------------------------- #
def strategy_band_flood(mesh, resolution=96, band_voxels=2.5, cleanup=1, **kw):
    shape, origin, voxel = grid_of(mesh, resolution)
    dist, _ = _sdf_grids(mesh, shape, origin, voxel, refine=False)
    outside = _flood_fill_outside(dist, band=band_voxels * voxel)
    inside = _clean_occupancy(~outside, iterations=cleanup) if cleanup else ~outside
    field = np.where(inside, -1.0, 1.0)
    return extract(field, shape, origin, voxel)


def strategy_band_flood_fillholes(mesh, resolution=96, band_voxels=2.5, cleanup=0, **kw):
    m = strategy_band_flood(mesh, resolution, band_voxels, cleanup)
    return repair_to_watertight(m)


def strategy_noband_nocleanup(mesh, resolution=96, band_voxels=1.2, cleanup=0, **kw):
    return strategy_band_flood(mesh, resolution, band_voxels, cleanup)


def strategy_repaired_flood(mesh, resolution=96, band_voxels=2.5, cleanup=0, **kw):
    m = repair_mesh(mesh.copy())
    return strategy_band_flood(m, resolution, band_voxels, cleanup)


STRATEGIES: Dict[str, Callable] = {
    "flood+clean1": strategy_band_flood,
    "flood+fillholes": strategy_band_flood_fillholes,
    "flood(b1.2)": strategy_noband_nocleanup,
    "repair+flood": strategy_repaired_flood,
}

# quick A/B of closing strength for the shipped strategy
for it in (1, 2):
    STRATEGIES[f"flood+clean{it}"] = (lambda it: (lambda mesh, resolution=96, **kw: strategy_band_flood(
        mesh, resolution=resolution, band_voxels=2.5, cleanup=it)))(it)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--resolution", type=int, default=96)
    args = ap.parse_args()

    assets: List[Path] = []
    for cat in ("airplane", "car", "ship"):
        d = RAW / cat / "train"
        if d.exists():
            assets.extend(sorted(d.glob("*.off"))[: args.limit // 3 + 1])
    assets = assets[: args.limit]
    print(f"assets: {len(assets)}")

    rows: Dict[str, List[Dict]] = {k: [] for k in STRATEGIES}
    for path in assets:
        try:
            mesh = normalize_mesh(load_mesh(str(path)))
        except Exception as exc:  # noqa: BLE001
            print(f"  load fail {path.name}: {exc}")
            continue
        for name, fn in STRATEGIES.items():
            t0 = time.time()
            try:
                out = fn(mesh, resolution=args.resolution)
                dt = time.time() - t0
                rows[name].append(
                    {
                        "id": path.stem,
                        "watertight": bool(is_watertight(out)),
                        "vol": float(out.volume) if out.is_watertight else float("nan"),
                        "faces": int(out.faces.shape[0]),
                        "seconds": dt,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                rows[name].append({"id": path.stem, "watertight": False, "vol": float("nan"),
                                   "faces": 0, "seconds": time.time() - t0, "error": str(exc)[:80]})

    print(f"\n{'strategy':18s} {'watertight':>10s} {'median vol':>11s} {'median faces':>13s} {'median s':>9s}")
    for name, rs in rows.items():
        wt = np.mean([r["watertight"] for r in rs]) if rs else 0.0
        vols = [r["vol"] for r in rs if np.isfinite(r["vol"])]
        faces = [r["faces"] for r in rs if r["faces"]]
        secs = [r["seconds"] for r in rs]
        print(
            f"{name:18s} {wt:10.2f} {(np.median(vols) if vols else float('nan')):11.4f} "
            f"{(int(np.median(faces)) if faces else 0):13d} {(np.median(secs) if secs else 0):9.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
