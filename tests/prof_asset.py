"""Per-asset preprocessing cost breakdown (dev tool)."""
import sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from seed3d_mini.mesh_ops import (load_mesh, normalize_mesh, remesh_watertight, RemeshConfig,
                                  MeshField, sample_surface_points, sample_edge_points)

path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT/"data/raw/modelnet40/ModelNet40/airplane/train/airplane_0001.off")
res = int(sys.argv[2]) if len(sys.argv) > 2 else 96
m = normalize_mesh(load_mesh(path))
print(f"input: F={m.faces.shape[0]} wt={m.is_watertight}", flush=True)
t=time.time(); r = remesh_watertight(m, RemeshConfig(resolution=res))
print(f"remesh({res}): {time.time()-t:5.1f}s F={r.faces.shape[0]} wt={r.is_watertight} vol={r.volume:.4f}", flush=True)
f = MeshField(r, truncation=0.1); rng = np.random.default_rng(0)
t=time.time(); pts, sdf, w = f.build_query_cloud(16384, rng=rng)
print(f"query cloud 16k: {time.time()-t:5.1f}s near={float((np.abs(sdf)<0.099).mean()):.2f}", flush=True)
t=time.time(); sp, sn = sample_surface_points(r, 8192, rng=rng)
print(f"surface 8k: {time.time()-t:5.2f}s", flush=True)
t=time.time(); ep, en = sample_edge_points(r, 1024, rng=rng)
print(f"edge 1k: {time.time()-t:5.1f}s", flush=True)
