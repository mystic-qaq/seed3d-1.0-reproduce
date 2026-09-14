"""Invariants of the data-preprocessing stack (fast, CPU-only).

These are the properties the VAE's supervision depends on, so they are asserted
rather than eyeballed:

* remeshing an open shell yields a closed, positive-volume solid,
* the extracted solid is close to the input surface (it is a voxelisation, not a
  different object),
* TSDF queries are truncated, negative inside and positive outside,
* near-surface query signs agree with an exact ray-parity reference.

    python tests/test_mesh_ops.py          # or: pytest tests/test_mesh_ops.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seed3d_mini.mesh_ops import (  # noqa: E402
    MeshField,
    RemeshConfig,
    TriangleDistance,
    is_watertight,
    mesh_stats,
    normalize_mesh,
    remesh_watertight,
    sample_edge_points,
    sample_surface_points,
)


def _box() -> trimesh.Trimesh:
    return normalize_mesh(trimesh.creation.box(extents=[1.2, 0.8, 0.6]))


def _open_box() -> trimesh.Trimesh:
    m = _box()
    return trimesh.Trimesh(vertices=m.vertices, faces=m.faces[:-2], process=True)


@pytest.mark.parametrize("mesh_fn", [_box, _open_box, lambda: normalize_mesh(trimesh.creation.uv_sphere(radius=0.8, count=[24, 24]))])
def test_remesh_is_watertight_solid(mesh_fn):
    mesh = mesh_fn()
    out = remesh_watertight(mesh, RemeshConfig(resolution=48))
    stats = mesh_stats(out)
    assert stats["watertight"], stats
    assert stats["volume"] > 0
    assert stats["euler_number"] == 2  # genus-0 closed surface
    # the remesh must stay in the canonical frame and keep the object's scale
    assert np.allclose(out.bounds[0].min(), -1.0, atol=0.15)
    assert abs(max(out.extents) - 2.0) < 0.15


def test_remesh_tracks_input_surface():
    mesh = _box()
    out = remesh_watertight(mesh, RemeshConfig(resolution=64))
    pts, _ = sample_surface_points(mesh, 2048)
    d = TriangleDistance(out).distance(pts)
    assert d.mean() < 0.05, d.mean()
    assert np.percentile(d, 95) < 0.1


def test_tsdf_signs_and_truncation():
    out = remesh_watertight(_box(), RemeshConfig(resolution=64))
    field = MeshField(out, truncation=0.1)
    assert field.sdf(np.array([[0.0, 0.0, 0.0]]), truncate=False)[0, 0] < 0  # centre is inside
    assert field.sdf(np.array([[0.98, 0.98, 0.98]]), truncate=False)[0, 0] > 0  # corner is outside
    q = np.array([[0.0, 0.0, 0.0], [0.99, 0.99, 0.99]])
    np.testing.assert_allclose(np.abs(field.sdf(q)).max(), 0.1, atol=1e-6)


def test_query_cloud_signs_match_reference():
    out = remesh_watertight(_box(), RemeshConfig(resolution=64))
    field = MeshField(out, truncation=0.1)
    rng = np.random.default_rng(0)
    pts, sdf, w = field.build_query_cloud(1024, rng=rng, near_ratio=0.7, jitter=0.012)
    assert pts.shape == (1024, 3) and sdf.shape == (1024, 1) and w.shape == (1024, 1)
    assert np.abs(sdf).max() <= 0.1 + 1e-6
    ref_inside = out.contains(pts.astype(np.float64))
    agree = float(((sdf[:, 0] < 0) == ref_inside).mean())
    assert agree > 0.9, agree  # near-surface sign heuristic + exact far-field test


def test_surface_and_edge_sampling():
    out = remesh_watertight(_box(), RemeshConfig(resolution=64))
    rng = np.random.default_rng(0)
    pts, nrm = sample_surface_points(out, 1024, rng=rng)
    assert pts.shape == (1024, 3) and nrm.shape == (1024, 3)
    np.testing.assert_allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-4)
    assert TriangleDistance(out).distance(pts).max() < 2e-3  # samples lie *on* the surface
    edge, enorm = sample_edge_points(out, 256, rng=rng)
    assert edge.shape[0] == 256 and enorm.shape == (256, 3)


def test_normalize_to_canonical_frame():
    mesh = trimesh.creation.box(extents=[10.0, 3.0, 1.0])
    mesh.apply_translation([100.0, -50.0, 7.0])
    out = normalize_mesh(mesh)
    assert abs(max(out.extents) - 2.0) < 1e-6
    np.testing.assert_allclose((out.bounds[0] + out.bounds[1]) / 2.0, 0.0, atol=1e-6)
    assert is_watertight(out)
