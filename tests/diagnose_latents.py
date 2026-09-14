"""Diagnose the trained VAE's latent space (dev tool).

The DiT stage can only be as good as the latent space it models, so this script
answers the three questions that actually matter:

1. **Is there variance collapse?** per-dimension std of the latent tokens. One dead
   dimension = one dimension of shape information the generator cannot represent
   (this is what PCA-style latent rescaling, as in Hunyuan3D's ``z_scale_factor``,
   is meant to fix).
2. **How far is the posterior from the prior?** per-dimension mean/std of
   ``mu``/``logvar``. The report regularises with ``lambda_KL = 1e-4``, which is far
   weaker than the 1.0 of a vanilla VAE; if KL grows without bound the latent
   distribution becomes heavy-tailed and the flow model has to work harder.
3. **How good is the decoder?** reconstruction metrics of latents sampled from the
   posterior (see ``scripts/evaluate.py`` for the full picture).

    python tests/diagnose_latents.py --vae checkpoints/vae_mini/vae_latest.pt
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
from seed3d_mini.train_utils import load_checkpoint  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default=str(ROOT / "checkpoints" / "vae_mini" / "vae_latest.pt"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "vae_mini.yaml"))
    ap.add_argument("--shard", default=None)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    mcfg = dict(cfg["model"]); mcfg.pop("_target", None)
    model = Seed3DVAE(**mcfg)
    info = load_checkpoint(args.vae, model, strict=False)
    device = torch.device(args.device)
    model = model.to(device).eval()
    print(f"VAE step {info.get('step')} ({model.num_parameters() / 1e6:.2f}M params) on {device}")

    shard = args.shard or str(sorted((ROOT / "data" / "shards").glob("val-*.npz"))[0])
    data = dict(np.load(shard, allow_pickle=False))
    n = min(args.n, data["surface"].shape[0])
    pc_size = int(mcfg.get("pc_size", 4096))
    sharp = int(mcfg.get("pc_sharpedge_size", 0))

    zs, mus, logvars = [], [], []
    with torch.no_grad():
        for i in range(n):
            pc = torch.from_numpy(data["surface"][i : i + 1, :pc_size]).to(device)
            feats = torch.from_numpy(data["surface_normal"][i : i + 1, :pc_size]).to(device)
            z, post = model.encode(pc, feats, sample=False)
            zs.append(z.float().cpu())
            mus.append(post.mean.float().cpu())
            logvars.append(post.logvar.float().cpu())

    z = torch.cat(zs)
    mu = torch.cat(mus)
    logvar = torch.cat(logvars)
    dim_std = z.reshape(-1, z.shape[-1]).std(dim=0)
    print("\nlatent token set: %s (M=%d, d=%d)" % (tuple(z.shape), z.shape[1], z.shape[2]))
    print("per-dim latent std      :", np.round(dim_std.numpy(), 4))
    print("  -> dead dims (std<0.05):", int((dim_std < 0.05).sum()), "/", dim_std.numel())
    print("global latent std       : %.4f  (z_scale_factor = %.4f)" % (z.std(), 1.0 / max(float(z.std()), 1e-6)))
    print("posterior mu  mean/std  : %.4f / %.4f" % (mu.mean(), mu.std()))
    print("posterior logvar mean   : %.4f  (std %.4f)" % (logvar.mean(), logvar.std()))
    kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=-1).mean()
    print("KL per sample (summed over tokens*dims): %.2f" % kl)
    print(
        "\nreading: dead dims mean variance collapse (DiT cannot model what the\n"
        "encoder threw away); a large KL means the posterior drifted from N(0, I)\n"
        "and the flow model has to fit a heavy-tailed distribution -- raise\n"
        "training.lambda_kl or extend kl_warmup_steps if both happen together."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
