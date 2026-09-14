"""Rewrite query-loss weights in existing shards (no geometry recomputation).

The supervision weights are a *pure function of the stored TSDF target* under the
current scheme:

* points whose |tsdf| is saturated (``>= 0.95 * truncation``) are far field ->
  ``far_weight``,
* everything else is inside the truncation band and carries the geometry -> 1.0.

So the fix for the "VAE collapsed to a constant field" bug (far field dominating
the MSE) does not require re-running the whole preprocessing pipeline: the weights
can be recomputed from the shards that already exist.

    python scripts/reweight_shards.py --shard-dir data/shards --far-weight 0.03
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", default=str(ROOT / "data" / "shards"))
    ap.add_argument("--truncation", type=float, default=0.1)
    ap.add_argument("--near-weight", type=float, default=1.0)
    # The far field is *saturated*, so its own MSE is large in absolute terms:
    # measured on real shards, a constant far-field prediction costs 7.2e-3 while
    # the entire in-band signal costs 2.9e-3.  Any far-field weight above ~1e-2
    # therefore makes "output the saturated value" the loss-minimising solution and
    # the decoder stops using its latent (confirmed with
    # tests/probe_latent_usage.py).  Keep it small; it is only there to anchor the
    # sign far from the surface.
    ap.add_argument("--far-weight", type=float, default=0.002)
    ap.add_argument("--splits", default="train,val")
    args = ap.parse_args()

    shard_dir = Path(args.shard_dir)
    for split in args.splits.split(","):
        for shard in sorted(shard_dir.glob(f"{split}-*.npz")):
            data = dict(np.load(shard, allow_pickle=False))
            sdf = data["sdf"]
            in_band = np.abs(sdf) < 0.95 * args.truncation
            old = data.get("weight")
            new = np.where(in_band, args.near_weight, args.far_weight).astype(np.float32)
            data["weight"] = new
            tmp = shard.with_suffix(".tmp.npz")
            np.savez(tmp, **data)
            tmp.replace(shard)
            frac = float(in_band.mean())
            old_mean = float(old.mean()) if old is not None else float("nan")
            print(
                f"{shard.name}: band fraction {frac:.3f}, weight mean {old_mean:.3f} -> {new.mean():.3f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
