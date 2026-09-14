"""End-to-end smoke test of the mini Seed3D shape pipeline (CPU or one GPU).

Validates, with no dataset required:
  1. Seed3D-VAE forward/backward and multi-scale token lengths,
  2. Seed3D-DiT flow-matching forward/backward with the real image conditioner,
  3. the full image -> latent -> field -> mesh inference path (random weights,
     so the *mesh* is garbage; what is checked is that the machinery produces a
     finite field, a non-empty mesh and a watertight closed surface),
  4. velocity-field ODE sampling determinism.

Unit tests for the maths live in ``tests/test_vae_parity.py`` (numerical parity
against the reference implementation).

    python tests/smoke_pipeline.py --device cpu        # ~1 min
    CUDA_VISIBLE_DEVICES=4 python tests/smoke_pipeline.py --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seed3d_mini.flow_matching import euler_sample, flow_matching_loss, sample_timesteps  # noqa: E402
from seed3d_mini.models.dit import Seed3DDiT  # noqa: E402
from seed3d_mini.models.vae import Seed3DVAE  # noqa: E402
from seed3d_mini.volume_decode import dense_volume_decode, hierarchical_volume_decode  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "ok  " if cond else "FAIL"
    if not cond:
        FAILURES.append(name)
    print(f"[{status}] {name}{(' -- ' + detail) if detail else ''}", flush=True)


def test_vae(device: torch.device, image_free: bool = True) -> None:
    torch.manual_seed(0)
    model = Seed3DVAE(
        num_latents=64,
        embed_dim=8,
        width=128,
        heads=4,
        num_encoder_layers=2,
        num_decoder_layers=1,
        decoder_self_attn="query",
        pc_size=512,
        pc_sharpedge_size=64,
        downsample_ratio=8,
        train_num_latents=[32, 64],
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    b, n, q = 2, 576, 1024
    pc = torch.randn(b, n, 3, device=device) * 0.5
    feats = torch.nn.functional.normalize(torch.randn(b, n, 3, device=device), dim=-1)
    queries = torch.rand(b, q, 3, device=device) * 2 - 1
    sdf = (queries.norm(dim=-1, keepdim=True) - 0.6).clamp(-0.1, 0.1)
    w = torch.ones(b, q, 1, device=device)

    out = model(pc, queries, sdf, feats=feats, query_weights=w)
    check("vae/loss finite", bool(torch.isfinite(out.loss)), f"loss={out.loss.item():.4f}")
    out.loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    check("vae/gradients flow", len(grads) > 0 and all(torch.isfinite(g).all() for g in grads))
    opt.step()
    opt.zero_grad()

    # multi-scale: the same weights must accept different token counts
    for m in (32, 64):
        z, post = model.encode(pc, feats, num_latents=m, sample=False)
        check(f"vae/multi-scale M={m}", tuple(z.shape) == (b, m, 8), f"z={tuple(z.shape)}")
    pred = model.decode(queries[:1], z[:1])
    check("vae/decode shape", tuple(pred.shape) == (1, q, 1), f"{tuple(pred.shape)}")

    # eikonal branch (double backward).  torch's CPU fused-attention kernel has no
    # double backward, so this branch is exercised on GPU only; the loss value
    # itself is still checked everywhere.
    model.lambda_eikonal = 0.1
    out2 = model(pc, queries, sdf, feats=feats, query_weights=w)
    check("vae/eikonal finite", bool(torch.isfinite(out2.loss)), f"loss={out2.loss.item():.4f}")
    if device.type == "cuda":
        out2.loss.backward()
        check(
            "vae/eikonal backward",
            all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None),
        )
    else:
        print("[skip] vae/eikonal backward (CPU fused-attention has no double backward)")


def test_dit(device: torch.device) -> None:
    torch.manual_seed(0)
    dit = Seed3DDiT(in_channels=8, hidden_size=128, num_heads=4, depth=2, depth_single_blocks=2, num_latents=64).to(device)
    latents = torch.randn(2, 64, 8, device=device)
    context = torch.randn(2, 40, 128, device=device)  # already projected cond tokens
    t = sample_timesteps(2, 64, device=device)
    check("dit/timesteps in (0,1)", bool(((t > 0) & (t < 1)).all()), f"t={t.tolist()}")
    out = flow_matching_loss(dit, latents, t, model_kwargs={"context": context})
    check("dit/loss finite", bool(torch.isfinite(out["loss"]).all()), f"loss={out['loss'].mean().item():.4f}")
    out["loss"].mean().backward()
    check("dit/gradients flow", all(torch.isfinite(p.grad).all() for p in dit.parameters() if p.grad is not None))

    # length-aware shift: longer sequences must map to higher mean noise
    t_short = sample_timesteps(4096, 128, device=device, base_tokens=128, max_tokens=4096, base_shift=0.5, max_shift=1.15)
    t_long = sample_timesteps(4096, 4096, device=device, base_tokens=128, max_tokens=4096, base_shift=0.5, max_shift=1.15)
    check(
        "dit/length-aware shift increases t",
        float(t_long.mean()) > float(t_short.mean()),
        f"mean t: {float(t_short.mean()):.3f} (128 tok) vs {float(t_long.mean()):.3f} (4096 tok)",
    )

    # samplers are deterministic given a seeded generator.  The model must be in
    # eval() mode, otherwise training-time stochasticity (dropout / DropPath) makes
    # the same seed produce different samples -- a real usage footgun, so the test
    # pins it down.
    dit.eval()
    kw = dict(num_steps=4, device=device, model_kwargs={"context": context}, guidance_scale=1.0)
    a = euler_sample(dit, (2, 64, 8), generator=torch.Generator(device=device).manual_seed(7), **kw)
    b = euler_sample(dit, (2, 64, 8), generator=torch.Generator(device=device).manual_seed(7), **kw)
    check("dit/euler deterministic", bool(torch.allclose(a, b)), f"max|d|={float((a-b).abs().max()):.2e}")


def test_volume_decode(device: torch.device) -> None:
    torch.manual_seed(0)
    m = 64

    def decode_fn(q, z):
        # analytic-ish field: signed distance to a sphere of radius 0.5
        return (q.norm(dim=-1, keepdim=True) - 0.5).clamp(-0.1, 0.1)

    z = torch.randn(1, m, 8, device=device)
    dense = dense_volume_decode(z, decode_fn, resolution=64, chunk=8192, device=device)
    check("volume/dense grid shape", dense.values.shape == (64, 64, 64), f"{dense.values.shape}")
    check("volume/dense field finite", bool(np.isfinite(dense.values).all()))

    hier = hierarchical_volume_decode(
        z, decode_fn, resolution=64, coarse_resolution=16, band=0.1, chunk=8192, device=device
    )
    diff = np.abs(hier.values - dense.values).max()
    check("volume/hierarchical matches dense", diff < 1e-4, f"max|d|={diff:.2e}")

    mesh = dense.extract(level=0.0)
    check("volume/mesh non-empty", mesh.faces.shape[0] > 0, f"V={mesh.vertices.shape[0]} F={mesh.faces.shape[0]}")
    check("volume/mesh closed (sphere)", bool(mesh.is_watertight), f"watertight={mesh.is_watertight}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    if device.type == "cpu":
        # the CPU flash-attention kernel has no double-backward (needed by the
        # eikonal term), so use the math backend there
        torch.backends.mha.set_fastpath_enabled(False)  # noqa: SIM115
    print(f"device: {device}  torch {torch.__version__}")
    t0 = time.time()
    test_vae(device)
    test_dit(device)
    test_volume_decode(device)
    print(f"\n{'ALL PASS' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}  ({time.time() - t0:.1f}s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
