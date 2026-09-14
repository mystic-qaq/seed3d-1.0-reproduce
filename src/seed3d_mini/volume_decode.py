"""Latent -> mesh: volume decoding and marching-cubes extraction.

Inference path of the report's geometry stage.  Two decoders are provided:

* :func:`dense_volume_decode` -- evaluate the VAE's TSDF head on a full
  ``R^3`` grid.  Simple, exact, and what the quality/ablation numbers in the
  README are produced with.
* :func:`hierarchical_volume_decode` -- the report's quantisation + spatial
  filtering trick: a cheap coarse pass locates the zero-crossing band, a
  conservative band is refined around it at the target resolution, and the far
  field is filled with the (known) saturation value.  This is what makes 256^3
  or 384^3 extraction affordable on a single 24 GB GPU.

Both return a :class:`VolumeGrid` holding the scalar field plus the world-space
grid geometry, so extraction is decoupled from evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = [
    "VolumeGrid",
    "dense_volume_decode",
    "hierarchical_volume_decode",
    "extract_mesh",
    "latents_to_mesh",
    "query_field",
]


@dataclass
class VolumeGrid:
    values: np.ndarray          # [R, R, R] float32 TSDF samples, C-order (x, y, z)
    bound: float                # grid spans [-bound, bound]^3
    resolution: int

    @property
    def voxel_size(self) -> float:
        return 2.0 * self.bound / (self.resolution - 1)

    def extract(self, level: float = 0.0, bounds: Optional[float] = None):
        """Marching cubes on the grid; returns a ``trimesh.Trimesh``."""
        from skimage import measure

        verts, faces, _, _ = measure.marching_cubes(
            np.ascontiguousarray(self.values, dtype=np.float32), level=float(level)
        )
        verts = verts / (self.resolution - 1) * (2.0 * self.bound) - self.bound
        import trimesh

        mesh = trimesh.Trimesh(vertices=verts.astype(np.float64), faces=faces.astype(np.int64), process=True)
        keep = mesh.nondegenerate_faces()
        if keep is not None and len(keep) and not keep.all():
            mesh.update_faces(keep)
        mesh.remove_unreferenced_vertices()
        trimesh.repair.fix_normals(mesh)
        return mesh


def _grid_points(resolution: int, bound: float, device, dtype) -> torch.Tensor:
    lin = torch.linspace(-bound, bound, resolution, device=device, dtype=dtype)
    g = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), dim=-1)
    return g.reshape(-1, 3)


@torch.no_grad()
def query_field(
    queries: torch.Tensor,
    latents: torch.Tensor,
    decode_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    chunk: int = 32768,
) -> torch.Tensor:
    """Chunked evaluation of a (continuous) decoder on flat query points.

    Args:
        queries: ``[N, 3]``
        latents: ``[B, M, d]`` (batch is broadcast against queries)
    Returns:
        ``[B, N, 1]``
    """
    outs: List[torch.Tensor] = []
    for s in range(0, queries.shape[0], chunk):
        e = min(s + chunk, queries.shape[0])
        outs.append(decode_fn(queries[s:e].unsqueeze(0).expand(latents.shape[0], -1, -1), latents))
    return torch.cat(outs, dim=1)


@torch.no_grad()
def dense_volume_decode(
    latents: torch.Tensor,
    decode_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    resolution: int = 128,
    bound: float = 1.0,
    chunk: int = 32768,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    verbose: bool = False,
) -> VolumeGrid:
    """Evaluate the decoder on the full ``resolution^3`` grid."""
    device = device or latents.device
    pts = _grid_points(resolution, bound, device, dtype)
    out = query_field(pts, latents.to(device), decode_fn, chunk=chunk)
    values = out[0, :, 0].reshape(resolution, resolution, resolution).float().cpu().numpy()
    return VolumeGrid(values=values, bound=bound, resolution=resolution)


@torch.no_grad()
def hierarchical_volume_decode(
    latents: torch.Tensor,
    decode_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    resolution: int = 256,
    bound: float = 1.0,
    coarse_resolution: int = 64,
    band: float = 0.1,
    band_voxels: float = 1.5,
    far_value: Optional[float] = None,
    chunk: int = 32768,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> VolumeGrid:
    """Coarse-to-fine decoding with a conservative zero-crossing band.

    Stage 1 evaluates a ``coarse_resolution^3`` grid.  Stage 2 evaluates the fine
    sub-grid region whose *coarse* value could hide a zero crossing -- i.e.
    ``|value| <= band`` (the field's saturation/truncation value) dilated by one
    coarse cell, plus ``band_voxels`` fine voxels of margin.  Every remaining
    voxel is provably in the saturated far field, so it is filled with
    ``far_value`` (defaults to ``band``) instead of being evaluated.  This is the
    report's "coarse SDF evaluation, prune inactive cells, full-precision
    evaluation for active cells" strategy.

    The coarse nodes are exact sub-grid points of the fine grid, so the two
    resolutions agree on the nodes and no interpolation error is introduced.
    """
    device = device or latents.device
    latents = latents.to(device)
    if resolution % coarse_resolution != 0 and (resolution - 1) % (coarse_resolution - 1) != 0:
        raise ValueError(
            f"resolution {resolution} and coarse_resolution {coarse_resolution} must be nested"
        )
    stride = (resolution - 1) // (coarse_resolution - 1)
    voxel = 2.0 * bound / (resolution - 1)
    if far_value is None:
        far_value = float(band)

    coarse_pts = _grid_points(coarse_resolution, bound, device, dtype)
    coarse = query_field(coarse_pts, latents, decode_fn, chunk=chunk)
    coarse_vals = coarse[0, :, 0].reshape(coarse_resolution, coarse_resolution, coarse_resolution)

    # A coarse node can hide a zero crossing only if |value| <= band; dilating by
    # one coarse cell plus band_voxels of fine margin makes the active region a
    # strict superset of the true surface band.
    threshold = float(band) + (stride + band_voxels) * voxel
    near = coarse_vals.abs() <= threshold
    dil = near.clone()
    for dim in range(3):
        dil = dil | torch.roll(near, 1, dims=dim) | torch.roll(near, -1, dims=dim)

    values = torch.full(
        (resolution, resolution, resolution), float(far_value), device=device, dtype=torch.float32
    )
    # always keep the (exact, already computed) coarse nodes
    idx = torch.arange(coarse_resolution, device=device) * stride
    idx_np = idx.cpu().numpy()
    values[np.ix_(idx_np, idx_np, idx_np)] = coarse_vals.float()

    active = torch.nonzero(dil)  # [C, 3] coarse indices to refine
    if active.numel():
        fine = torch.arange(resolution, device=device)
        lo_all = np.maximum(active.min(dim=0).values.cpu().numpy() * stride - stride, 0)
        hi_all = np.minimum(
            active.max(dim=0).values.cpu().numpy() * stride + stride, resolution - 1
        )
        grids = [fine[int(lo_all[d]): int(hi_all[d]) + 1] for d in range(3)]
        mesh = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).reshape(-1, 3)
        pts = mesh / (resolution - 1) * (2 * bound) - bound
        vals = query_field(pts, latents, decode_fn, chunk=chunk)[0, :, 0].float()
        values[
            int(lo_all[0]): int(hi_all[0]) + 1,
            int(lo_all[1]): int(hi_all[1]) + 1,
            int(lo_all[2]): int(hi_all[2]) + 1,
        ] = vals.reshape(
            int(hi_all[0] - lo_all[0]) + 1, int(hi_all[1] - lo_all[1]) + 1, int(hi_all[2] - lo_all[2]) + 1
        )

    return VolumeGrid(values=values.cpu().numpy(), bound=bound, resolution=resolution)


def extract_mesh(grid: VolumeGrid, level: float = 0.0, *, keep_largest: bool = False):
    """Extract + clean the iso-surface (``level = 0`` for a TSDF zero crossing)."""
    import trimesh

    mesh = grid.extract(level=level)
    if keep_largest and len(mesh.split(only_watertight=False)) > 1:
        comps = mesh.split(only_watertight=False)
        mesh = max(comps, key=lambda c: abs(c.volume))
    return mesh


@torch.no_grad()
def latents_to_mesh(
    latents: torch.Tensor,
    decode_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    mode: str = "hierarchical",
    resolution: int = 256,
    bound: float = 1.0,
    level: float = 0.0,
    coarse_resolution: int = 64,
    band: float = 0.1,
    chunk: int = 32768,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    return_grid: bool = False,
):
    """Seed3D-VAE decoder + iso-surface extraction (the report's DMC stage)."""
    if mode == "dense":
        grid = dense_volume_decode(
            latents, decode_fn, resolution=resolution, bound=bound, chunk=chunk, device=device, dtype=dtype
        )
    elif mode == "hierarchical":
        grid = hierarchical_volume_decode(
            latents,
            decode_fn,
            resolution=resolution,
            bound=bound,
            coarse_resolution=coarse_resolution,
            band=band,
            chunk=chunk,
            device=device,
            dtype=dtype,
        )
    else:
        raise ValueError(f"unknown decode mode {mode!r}")
    mesh = extract_mesh(grid, level=level)
    if return_grid:
        return mesh, grid
    return mesh
