"""A dependency-free differentiable-ish mesh rasteriser for reference images.

The report renders reference images with Blender Cycles (elevation in
``[-30, 70]`` degrees, stochastic lighting).  Blender is not installable here, so
this module implements the same *contract* with a small PyTorch z-buffer
rasteriser:

``render_views(mesh, views, resolution, ...) -> {"rgb", "normal", "mask", "cam_origin"}``

* ``rgb``      -- shaded image, headlight Lambert + ambient, light jitter and
                  optional albedo, in ``[0, 1]``, shape ``[V, 3, H, W]``
* ``normal``   -- world-space surface normals, ``[-1, 1]``, shape ``[V, 3, H, W]``
* ``mask``     -- object silhouette ``[V, 1, H, W]``
* ``cam_origin`` -- camera position per view, ``[V, 3]``

Rendering is perspective with the object normalised to ``[-1, 1]^3``.  The
rasteriser is a straightforward per-triangle bounding-box scatter with a depth
buffer; each view is rendered independently so the loop is embarrassingly
parallel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["View", "sample_views", "render_views", "render_mesh_views", "look_at"]


@dataclass
class View:
    azimuth_deg: float
    elevation_deg: float
    distance: float = 2.6


def look_at(origin: torch.Tensor, target: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """World->camera rotation for a right-handed camera looking down ``-z``."""
    f = F.normalize(target - origin, dim=-1)
    r = F.normalize(torch.cross(f, up, dim=-1), dim=-1)
    u = torch.cross(r, f, dim=-1)
    return torch.stack([r, u, -f], dim=0)  # [3, 3]


def sample_views(
    n_views: int,
    *,
    elevation_range: Tuple[float, float] = (-30.0, 70.0),
    rng: Optional[np.random.Generator] = None,
    distance: float = 2.6,
    fixed: bool = False,
) -> List[View]:
    """Random reference viewpoints (report: elevation sampled from [-30, 70])."""
    rng = rng or np.random.default_rng()
    if fixed:
        azis = np.linspace(0, 360, n_views, endpoint=False)
        eles = np.linspace(elevation_range[0], elevation_range[1], n_views)
        return [View(float(a), float(e), distance) for a, e in zip(azis, eles)]
    azis = rng.uniform(0.0, 360.0, size=n_views)
    eles = rng.uniform(elevation_range[0], elevation_range[1], size=n_views)
    return [View(float(a), float(e), distance) for a, e in zip(azis, eles)]


def _view_matrix(view: View, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    az = math.radians(view.azimuth_deg)
    el = math.radians(view.elevation_deg)
    origin = torch.tensor(
        [
            view.distance * math.cos(el) * math.cos(az),
            view.distance * math.sin(el),
            view.distance * math.cos(el) * math.sin(az),
        ],
        device=device,
        dtype=dtype,
    )
    up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    R = look_at(origin, torch.zeros(3, device=device, dtype=dtype), up)
    return origin, R


def _rasterize(
    verts_cam: torch.Tensor,   # [F, 3, 3] in camera space (-z forward)
    faces: torch.Tensor,       # [F, 3]
    normals_cam: torch.Tensor,  # [F, 3]
    face_colors: torch.Tensor,  # [F, 3]
    resolution: int,
    fov_deg: float,
) -> Dict[str, torch.Tensor]:
    device = verts_cam.device
    dtype = verts_cam.dtype
    focal = 0.5 * resolution / math.tan(math.radians(fov_deg) * 0.5)
    depth = torch.full((resolution, resolution), float("inf"), device=device, dtype=dtype)
    img = torch.zeros(resolution, resolution, 3, device=device, dtype=dtype)
    nrm = torch.zeros(resolution, resolution, 3, device=device, dtype=dtype)
    mask = torch.zeros(resolution, resolution, device=device, dtype=dtype)

    z = -verts_cam[..., 2]  # positive in front of the camera
    valid = z > 0.05
    uvw = torch.empty_like(verts_cam)
    uvw[..., 0] = verts_cam[..., 0] * focal / z.clamp(min=1e-4) + resolution * 0.5
    uvw[..., 1] = -verts_cam[..., 1] * focal / z.clamp(min=1e-4) + resolution * 0.5
    uvw[..., 2] = z

    tri_valid = valid.all(dim=1)
    if tri_valid.any():
        # frustum cull: skip triangles fully outside the image square
        P_all = uvw[tri_valid]
        inside_frustum = (
            (P_all[..., 0].max(dim=1).values >= 0)
            & (P_all[..., 0].min(dim=1).values <= resolution)
            & (P_all[..., 1].max(dim=1).values >= 0)
            & (P_all[..., 1].min(dim=1).values <= resolution)
        )
        idx = torch.nonzero(tri_valid).squeeze(-1)[inside_frustum]
    else:
        idx = torch.nonzero(tri_valid).squeeze(-1)
    if idx.numel() == 0:
        return {"color": img, "normal": nrm, "mask": mask}

    P = uvw[idx]                                # [T, 3, 3] (u, v, depth)
    C = face_colors[idx]
    N = normals_cam[idx]

    xmin = torch.floor(P[..., 0].min(dim=1).values).clamp(0, resolution - 1).long()
    xmax = torch.ceil(P[..., 0].max(dim=1).values).clamp(0, resolution - 1).long()
    ymin = torch.floor(P[..., 1].min(dim=1).values).clamp(0, resolution - 1).long()
    ymax = torch.ceil(P[..., 1].max(dim=1).values).clamp(0, resolution - 1).long()

    area = (P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1]) - (P[:, 2, 0] - P[:, 0, 0]) * (
        P[:, 1, 1] - P[:, 0, 1]
    )
    keep = area.abs() > 1e-12
    P, C, N, xmin, xmax, ymin, ymax, area = (
        P[keep], C[keep], N[keep], xmin[keep], xmax[keep], ymin[keep], ymax[keep], area[keep]
    )
    if P.shape[0] == 0:
        return {"color": img, "normal": nrm, "mask": mask}

    eps = 1e-9
    for i in range(P.shape[0]):
        x0, x1 = int(xmin[i]), int(xmax[i])
        y0, y1 = int(ymin[i]), int(ymax[i])
        if x1 < x0 or y1 < y0:
            continue
        ys_f, xs_f = torch.meshgrid(
            torch.arange(y0, y1 + 1, device=device, dtype=dtype),
            torch.arange(x0, x1 + 1, device=device, dtype=dtype),
            indexing="ij",
        )
        ys, xs = ys_f.long(), xs_f.long()
        px = xs_f + 0.5
        py = ys_f + 0.5
        a = P[i, 0]
        b = P[i, 1]
        c = P[i, 2]
        d = area[i]
        w0 = ((b[0] - px) * (c[1] - py) - (c[0] - px) * (b[1] - py)) / d
        w1 = ((c[0] - px) * (a[1] - py) - (a[0] - px) * (c[1] - py)) / d
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -eps) & (w1 >= -eps) & (w2 >= -eps)
        if not inside.any():
            continue
        depth_px = w0 * a[2] + w1 * b[2] + w2 * c[2]
        sub_depth = depth[ys, xs]
        better = inside & (depth_px < sub_depth)
        if not better.any():
            continue
        ys_b, xs_b = ys[better], xs[better]
        depth[ys_b, xs_b] = depth_px[better]
        w0b, w1b, w2b = w0[better], w1[better], w2[better]
        img[ys_b, xs_b] = w0b[:, None] * C[i, 0] + w1b[:, None] * C[i, 1] + w2b[:, None] * C[i, 2]
        n = w0b[:, None] * N[i, 0] + w1b[:, None] * N[i, 1] + w2b[:, None] * N[i, 2]
        nrm[ys_b, xs_b] = F.normalize(n, dim=-1)
        mask[ys_b, xs_b] = 1.0
    return {"color": img, "normal": nrm, "mask": mask}


def render_views(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    face_normals: torch.Tensor,
    views: Sequence[View],
    *,
    resolution: int = 256,
    fov_deg: float = 40.0,
    albedo: Optional[torch.Tensor] = None,
    light_jitter: float = 0.6,
    ambient: float = 0.25,
    device: Optional[torch.device] = None,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Render a batch of views of one mesh.

    Shading is a headlight-style Lambert term with the light direction jittered
    per view (the report's "stochastic illumination": 30% point lights / 70% HDR
    environments -- here both are approximated by a jittered directional light).
    """
    device = device or vertices.device
    vertices = vertices.to(device)
    faces = faces.to(device)
    face_normals = face_normals.to(device)
    V = len(views)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    tri_verts = vertices[faces]                 # [F, 3, 3]
    tri_normals = face_normals                  # [F, 3]
    if albedo is None:
        base = torch.full((faces.shape[0], 3), 0.72, device=device)
    else:
        base = albedo[faces].mean(dim=1).to(device)

    colors, normals, masks, origins = [], [], [], []
    for v in views:
        origin, R = _view_matrix(v, device, vertices.dtype)
        cam_v = (tri_verts - origin) @ R.transpose(0, 1)
        cam_n = tri_normals @ R.transpose(0, 1)
        # backface cull against the view direction (meshes here are closed shells)
        centroid = cam_v.mean(dim=1)
        facing = (cam_n * F.normalize(centroid, dim=-1)).sum(dim=-1) < 0.0
        cam_v, cam_n = cam_v[facing], cam_n[facing]
        if cam_v.shape[0] == 0:
            colors.append(torch.zeros(3, resolution, resolution, device=device))
            normals.append(torch.zeros(3, resolution, resolution, device=device))
            masks.append(torch.zeros(1, resolution, resolution, device=device))
            origins.append(origin)
            continue
        base_f = base[facing] if base.shape[0] == faces.shape[0] else base
        # jittered directional light in camera space (headlight-ish)
        l = torch.tensor(
            [
                float(torch.empty(1).uniform_(-light_jitter, light_jitter, generator=gen)),
                0.6 + float(torch.empty(1).uniform_(-light_jitter, light_jitter, generator=gen)),
                0.8 + float(torch.empty(1).uniform_(-light_jitter / 2, light_jitter / 2, generator=gen)),
            ],
            device=device,
        )
        l = F.normalize(l, dim=-1)
        lam = (cam_n @ l).clamp(min=0.0)
        # two-sided shading keeps the image informative for back-facing regions
        shade = ambient + (1.0 - ambient) * lam
        face_colors = base_f * shade[:, None]
        out = _rasterize(cam_v, faces[: cam_v.shape[0]], cam_n, face_colors, resolution, fov_deg)
        colors.append(out["color"].permute(2, 0, 1))
        normals.append(out["normal"].permute(2, 0, 1))
        masks.append(out["mask"].unsqueeze(0))
        origins.append(origin)

    return {
        "rgb": torch.stack(colors, 0).clamp(0, 1),
        "normal": torch.stack(normals, 0),
        "mask": torch.stack(masks, 0),
        "cam_origin": torch.stack(origins, 0),
    }


def render_mesh_views(
    mesh,
    n_views: int = 4,
    *,
    resolution: int = 256,
    seed: int = 0,
    fixed: bool = False,
    elevation_range: Tuple[float, float] = (-30.0, 70.0),
    distance: float = 2.6,
    fov_deg: float = 40.0,
    device: Optional[torch.device] = None,
) -> Dict[str, np.ndarray]:
    """Convenience wrapper: ``trimesh.Trimesh`` -> numpy image dict (CHW, float32)."""
    verts = torch.as_tensor(np.asarray(mesh.vertices), dtype=torch.float32)
    faces = torch.as_tensor(np.asarray(mesh.faces), dtype=torch.int64)
    fn = torch.as_tensor(np.asarray(mesh.face_normals), dtype=torch.float32)
    rng = np.random.default_rng(seed)
    views = sample_views(
        n_views, elevation_range=elevation_range, rng=rng, distance=distance, fixed=fixed
    )
    out = render_views(
        verts,
        faces,
        fn,
        views,
        resolution=resolution,
        fov_deg=fov_deg,
        device=device,
        seed=seed,
    )
    return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in out.items()}
