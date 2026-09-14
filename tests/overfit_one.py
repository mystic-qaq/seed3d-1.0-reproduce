"""Single-sample overfit test: can this VAE represent *one* shape at all?

If a model cannot overfit a single asset in a few hundred steps, no amount of
dataset or compute will make it work, and the failure is architectural.  If it can,
then a plateau on the full dataset is a *training-budget* problem and more steps
(or more data) fix it.

    python tests/overfit_one.py --steps 150
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seed3d_mini.models.vae import Seed3DVAE  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "vae_mini.yaml"))
    ap.add_argument("--shard", default=str(ROOT / "data" / "shards" / "val-00000.npz"))
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    mcfg = dict(cfg["model"]); mcfg.pop("_target", None)
    device = torch.device(args.device)
    torch.manual_seed(0)
    model = Seed3DVAE(**mcfg).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    data = dict(np.load(args.shard, allow_pickle=False))
    i = args.index
    ps, pe = model.encoder.pc_size, model.encoder.pc_sharpedge_size
    pc = torch.from_numpy(np.concatenate([data["surface"][i : i + 1, :ps], data["edge"][i : i + 1, :pe]], axis=1)).to(device)
    ft = torch.from_numpy(
        np.concatenate([data["surface_normal"][i : i + 1, :ps], data["edge_normal"][i : i + 1, :pe]], axis=1)
    ).to(device)
    q = torch.from_numpy(data["query"][i : i + 1]).to(device)
    sdf = torch.from_numpy(data["sdf"][i : i + 1]).to(device)
    w = torch.from_numpy(data["weight"][i : i + 1]).to(device)
    const = float(((sdf - sdf.mean()) ** 2).mean())
    print(f"target: {q.shape[1]} queries, |sdf| in [{sdf.min():.3f}, {sdf.max():.3f}], constant-mean MSE {const:.5f}")

    t0 = time.time()
    for step in range(args.steps):
        out = model(pc, q, sdf, feats=ft, query_weights=w, compute_eikonal=False)
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        opt.step()
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            pred = out.sdf_pred.detach()
            print(
                f"  step {step:4d} loss {out.loss.item():.5f} recon {out.loss_dict['recon'].item():.5f} "
                f"pred std {pred.std().item():.5f} range [{pred.min().item():.4f}, {pred.max().item():.4f}] "
                f"in-band MAE {float((pred - sdf).abs()[(sdf.abs() < 0.099)].mean()):.5f}",
                flush=True,
            )

    with torch.no_grad():
        z, _ = model.encode(pc, ft, sample=False)
        # does the decoder depend on the latent at all?  compare a real latent with
        # a heavily perturbed one on the same queries
        p0 = model.decode(q, z)
        p1 = model.decode(q, z + 5.0 * torch.randn_like(z))
    print(
        f"\nafter {args.steps} steps: recon {out.loss_dict['recon'].item():.5f} (constant {const:.5f}), "
        f"latent sensitivity mean|d| {float((p0 - p1).abs().mean()):.5f}"
    )
    ok = out.loss_dict["recon"].item() < const
    print("VERDICT:", "architecture can represent a shape" if ok else "CANNOT EVEN OVERFIT ONE SHAPE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
