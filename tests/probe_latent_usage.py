"""Check how shape-dependent the VAE's decoded field is (dev tool).

The failure this guards against is subtle: a VAE can report a tiny reconstruction
loss while its decoder ignores the latent completely, because a *truncated* SDF is
saturated over most of the volume and a constant prediction is nearly optimal
there.  Such a model loses nothing on the training curves and produces garbage
meshes at inference time.

The test: encode two different shapes and compare the decoded fields.

    python tests/probe_latent_usage.py --vae checkpoints/vae_probe/vae_latest.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seed3d_mini.models.vae import Seed3DVAE  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs" / "vae_mini.yaml"))
    ap.add_argument("--shard", default=str(ROOT / "data" / "shards" / "val-00000.npz"))
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    mcfg = dict(cfg["model"]); mcfg.pop("_target", None)
    ck = torch.load(args.vae, map_location="cpu", weights_only=False)
    cfg_used = ck.get("config") or {}
    if not cfg_used:
        print("WARNING: checkpoint has no embedded config; using --config (may mismatch)")
    else:
        mcfg = dict(cfg_used["model"]); mcfg.pop("_target", None)
    model = Seed3DVAE(**mcfg)
    model.load_state_dict(ck["model"])
    device = torch.device(args.device)
    model = model.to(device).eval()

    data = dict(np.load(args.shard, allow_pickle=False))
    ps, pe = model.encoder.pc_size, model.encoder.pc_sharpedge_size

    def encode(i: int) -> torch.Tensor:
        pc = torch.from_numpy(
            np.concatenate([data["surface"][i : i + 1, :ps], data["edge"][i : i + 1, :pe]], axis=1)
        ).to(device)
        ft = torch.from_numpy(
            np.concatenate([data["surface_normal"][i : i + 1, :ps], data["edge_normal"][i : i + 1, :pe]], axis=1)
        ).to(device)
        with torch.no_grad():
            return model.encode(pc, ft, sample=False)[0]

    lin = torch.linspace(-1, 1, 32, device=device)
    grid = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), dim=-1).reshape(1, -1, 3)

    def field(z: torch.Tensor) -> torch.Tensor:
        # decode() returns world units (see Seed3DVAE.decode); that is the field the
        # meshes are extracted from, so it is the right thing to compare
        with torch.no_grad():
            return model.decode(grid, z)[0, :, 0]

    zs = [encode(i) for i in range(args.pairs)]
    inter = float(torch.stack([z.std() for z in zs]).mean())
    print(f"step {ck.get('step')}: latent std within-sample {inter:.4f}")
    zero = field(torch.zeros_like(zs[0]))
    print(f"  decode(zero): std {zero.std():.4f}")
    ok = True
    for i in range(1, len(zs)):
        fi, f0 = field(zs[i]), field(zs[0])
        d = (fi - f0).abs()
        # a shape-dependent decoder must move the field by much more than noise
        ratio = float(d.mean() / max(float(zero.std()), 1e-6))
        print(
            f"  shape0 vs shape{i}: mean|dF| {float(d.mean()):.5f}  max {float(d.max()):.4f}  "
            f"F(shape{i}) std {float(fi.std()):.4f}  ratio-to-noise {ratio:.2f}"
        )
        # the world-unit truncation band is +-0.1, so a shape-dependent field must
        # move by a meaningful fraction of it
        ok = ok and float(d.mean()) > 0.01
    print("VERDICT:", "latent is used by the decoder" if ok else "DECODER IGNORES THE LATENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
