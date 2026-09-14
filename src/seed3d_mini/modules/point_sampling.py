"""Point-set utilities: farthest-point sampling, radius queries, KNN, and the
encoder-side "query / data" construction used by the vector-set encoder.

Everything here has a pure-PyTorch implementation so the reproduction runs on a
bare machine (no ``torch_cluster`` / ``pytorch3d`` CUDA extensions required).
Both the deterministic and the stochastic variants are provided:

* ``farthest_point_sample(..., random_start=...)`` implements the runtime-seeded
  variant used by the reference vector-set encoders ("random query points"), and
  with ``random_start=False`` it is the classic deterministic FPS.
* ``radius_neighbors`` is a chunked exact ball query, used for sharp-edge point
  extraction (a point is "sharp" when its ball neighbourhood is sparse).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

__all__ = [
    "farthest_point_sample",
    "gather_points",
    "knn_points",
    "radius_neighbors",
    "voxel_grid_unique",
    "sample_encoder_inputs",
]


@torch.no_grad()
def farthest_point_sample(
    points: torch.Tensor,
    n_sample: int,
    random_start: bool = True,
) -> torch.Tensor:
    """Farthest point sampling on a *single* point set.

    Args:
        points: ``[N, C]`` float tensor (usually ``[N, 3]``).
        n_sample: number of points to select (clamped to ``N``).
        random_start: pick the seed uniformly at random instead of always index 0.

    Returns:
        ``[n_sample]`` long tensor of indices into ``points``.
    """
    if points.dim() != 2:
        raise ValueError(f"expected [N, C] point set, got {tuple(points.shape)}")
    n = points.shape[0]
    n_sample = min(int(n_sample), n)
    device = points.device
    pts = points.float()
    indices = torch.empty(n_sample, dtype=torch.long, device=device)

    if random_start:
        start = torch.randint(0, n, (1,), device=device).item()
    else:
        start = 0

    min_dist = torch.full((n,), float("inf"), device=device, dtype=torch.float32)
    current = start
    for i in range(n_sample):
        indices[i] = current
        dist = torch.sum((pts - pts[current]) ** 2, dim=-1)
        min_dist = torch.minimum(min_dist, dist)
        if i + 1 < n_sample:
            # mark the selected point as used, then take the farthest remaining one
            min_dist[current] = -1.0
            current = int(torch.argmax(min_dist).item())
    return indices


@torch.no_grad()
def farthest_point_sample_batch(
    points: torch.Tensor,
    n_sample: int,
    random_start: bool = True,
) -> torch.Tensor:
    """Batched wrapper around :func:`farthest_point_sample`.

    Args:
        points: ``[B, N, C]``.
    Returns:
        ``[B, n_sample]`` long tensor.
    """
    outs = [
        farthest_point_sample(points[b], n_sample=n_sample, random_start=random_start)
        for b in range(points.shape[0])
    ]
    return torch.stack(outs, dim=0)


def gather_points(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather ``[B, N, C]`` points with ``[B, S]`` indices -> ``[B, S, C]``."""
    b = torch.arange(points.shape[0], device=points.device)[:, None]
    return points[b, indices]


@torch.no_grad()
def radius_neighbors(
    points: torch.Tensor,
    query: torch.Tensor,
    radius: float,
    max_neighbors: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact ball query around each query point.

    Args:
        points: ``[N, 3]`` reference points.
        query:  ``[M, 3]`` query points.
        radius: ball radius.
        max_neighbors: cap on returned neighbours per query.

    Returns:
        ``(idx, valid)`` with shapes ``[M, max_neighbors]``; ``idx`` is clamped to
        a valid range and ``valid`` is a boolean mask (first ``n_i`` entries are
        the true neighbours, the rest are padding at index 0).
    """
    pts = points.float()
    q = query.float()
    m = q.shape[0]
    r2 = float(radius) ** 2
    # chunk over queries to bound peak memory at M*N floats
    chunk = max(1, int(2 ** 22 // max(points.shape[0], 1)))
    idx_out = torch.zeros(m, max_neighbors, dtype=torch.long, device=points.device)
    valid_out = torch.zeros(m, max_neighbors, dtype=torch.bool, device=points.device)
    for s in range(0, m, chunk):
        e = min(s + chunk, m)
        d2 = torch.cdist(q[s:e], pts, compute_mode="donot_use_mm_for_euclid_dist") ** 2
        in_ball = d2 <= r2
        # keep the nearest `max_neighbors` of the ball
        d2_masked = torch.where(in_ball, d2, torch.full_like(d2, float("inf")))
        k = min(max_neighbors, points.shape[0])
        vals, idx = torch.topk(d2_masked, k=k, dim=-1, largest=False)
        valid = torch.isfinite(vals)
        idx = torch.where(valid, idx, torch.zeros_like(idx))
        idx_out[s:e, :k] = idx
        valid_out[s:e, :k] = valid
    return idx_out, valid_out


@torch.no_grad()
def knn_points(points: torch.Tensor, query: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the ``k`` nearest ``points`` for every ``query`` point."""
    idx, _ = radius_neighbors(points, query, radius=float("inf"), max_neighbors=k)
    return idx


@torch.no_grad()
def voxel_grid_unique(points: torch.Tensor, resolution: int = 512) -> torch.Tensor:
    """Greedy voxel-grid downsampling used for sharp-edge candidate cloud.

    Keeps one point per occupied voxel of a ``resolution^3`` grid spanning the
    unit cube ``[-1, 1]``.
    """
    p = points.float()
    q = torch.clamp(((p + 1.0) * 0.5 * resolution).floor().long(), 0, resolution - 1)
    keys = (q[:, 0] * resolution + q[:, 1]) * resolution + q[:, 2]
    uniq, inverse = torch.unique(keys, return_inverse=True)
    first = torch.full((uniq.shape[0],), points.shape[0], dtype=torch.long, device=points.device)
    first = first.scatter_reduce(0, inverse, torch.arange(points.shape[0], device=points.device), reduce="amin")
    return first[first < points.shape[0]]


@torch.no_grad()
def sample_encoder_inputs(
    pc: torch.Tensor,
    feats: torch.Tensor,
    num_latents: int,
    pc_size: int,
    pc_sharpedge_size: int = 0,
    downsample_ratio: int = 20,
    random_start: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the (query, data) pair feeding the vector-set encoder.

    The encoder does not run on the raw surface samples; it runs on

    * ``data``  -- a random subset of the surface points (the "keys/values"),
    * ``query`` -- ``num_latents`` points obtained by farthest-point sampling of
      the *same* surface points (the "learnable-latent stand-ins").

    Both are lifted to the same ``fourier(pos) || feat`` representation by the
    caller.  ``pc`` is split into ``[pc_size, pc_sharpedge_size]`` blocks so that
    a caller can bias the query budget towards sharp edges, exactly as in the
    reference vector-set encoder.

    Args:
        pc: ``[B, N, 3]`` surface points, ordered as ``random || sharpedge``.
        feats: ``[B, N, F]`` per-point features (normals / curvature / ...).
        num_latents: number of latent query points ``M``.
        pc_size: size of the random block.
        pc_sharpedge_size: size of the sharp-edge block (may be 0).
        downsample_ratio: how many surface points back every latent query.

    Returns:
        ``(query, data, query_feats, data_feats)`` with shapes
        ``[B, M, 3]``, ``[B, M*ratio, 3]``, ``[B, M, F]``, ``[B, M*ratio, F]``.
    """
    b, n, d = pc.shape
    if pc_size + pc_sharpedge_size > n:
        raise ValueError(
            f"pc has {n} points but pc_size+pc_sharpedge_size={pc_size + pc_sharpedge_size}"
        )

    num_query_random = int(round(pc_size / max(pc_size + pc_sharpedge_size, 1) * num_latents))
    num_query_sharp = num_latents - num_query_random

    random_pc, sharp_pc = torch.split(pc, [pc_size, pc_sharpedge_size], dim=1)
    random_feats, sharp_feats = torch.split(feats, [pc_size, pc_sharpedge_size], dim=1)

    def _sample_block(block_pc, block_feats, n_query):
        """Pick n_query*ratio inputs from the block, then FPS n_query of them."""
        if n_query <= 0 or block_pc.shape[1] == 0:
            z = block_pc.new_zeros(b, 0, d)
            zf = block_feats.new_zeros(b, 0, block_feats.shape[-1])
            return z, zf, z, zf
        n_input = min(int(n_query * downsample_ratio), block_pc.shape[1])
        perm = torch.randperm(block_pc.shape[1], device=block_pc.device)[:n_input]
        inp_pc = block_pc[:, perm, :]
        inp_feats = block_feats[:, perm, :]
        idx = farthest_point_sample_batch(inp_pc, n_sample=min(n_query, n_input), random_start=random_start)
        q_pc = gather_points(inp_pc, idx)
        q_feats = gather_points(inp_feats, idx)
        return q_pc, q_feats, inp_pc, inp_feats

    q_r, qf_r, i_r, if_r = _sample_block(random_pc, random_feats, num_query_random)
    q_s, qf_s, i_s, if_s = _sample_block(sharp_pc, sharp_feats, num_query_sharp)

    query = torch.cat([q_r, q_s], dim=1)
    data = torch.cat([i_r, i_s], dim=1)
    query_feats = torch.cat([qf_r, qf_s], dim=1)
    data_feats = torch.cat([if_r, if_s], dim=1)
    return query, data, query_feats, data_feats
