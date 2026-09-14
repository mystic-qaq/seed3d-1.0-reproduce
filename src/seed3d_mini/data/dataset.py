"""Dataset/Dataloader layer for the mini Seed3D reproduction.

Two datasets:

* :class:`SDFShardDataset` -- what ``train_vae.py`` consumes.  Each sample is
  ``(surface points + normals) || (edge points + normals)`` for the encoder and a
  mixed near-surface / uniform query cloud with (T)SDF targets and per-query
  weights for the decoder.
* :class:`LatentShardDataset` -- what ``train_dit.py`` consumes: a VAE latent
  token set plus the reference image rendered from the same canonical frame.

Shards are ``.npz`` files written by ``scripts/prepare_data.py`` /
``scripts/pack_latents.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["SDFShardDataset", "LatentShardDataset", "build_sdf_loader", "build_latent_loader"]


class SDFShardDataset(Dataset):
    def __init__(
        self,
        shard_dir: str,
        split: str = "train",
        *,
        pc_size: int = 4096,
        pc_sharpedge_size: int = 1024,
        num_queries: int = 8192,
        #: fraction of the per-step query budget drawn from the near-surface block
        #: (the shard stores ``near_ratio`` of its cloud as near-surface points)
        near_query_ratio: float = 0.9,
        #: fraction of the shard's query cloud that is near-surface (must match
        #: ``prepare_data.PrepConfig.near_ratio``)
        near_ratio: float = 0.7,
        augment: bool = True,
        seed: int = 0,
        max_samples: Optional[int] = None,
        categories: Optional[Sequence[str]] = None,
    ) -> None:
        self.shard_dir = Path(shard_dir)
        self.split = split
        self.pc_size = pc_size
        self.pc_sharpedge_size = pc_sharpedge_size
        self.num_queries = num_queries
        self.near_query_ratio = float(near_query_ratio)
        self.near_ratio = float(near_ratio)
        self.augment = augment and split == "train"
        self.seed = seed
        self.epoch = 0

        self.shards = sorted(self.shard_dir.glob(f"{split}-*.npz"))
        if not self.shards:
            raise FileNotFoundError(f"no {split}-*.npz shards in {self.shard_dir}")
        self.index: List[tuple] = []  # (shard_idx, sample_idx)
        self.categories: List[str] = []
        for si, sp in enumerate(self.shards):
            with np.load(sp, allow_pickle=False) as f:
                n = f["surface"].shape[0]
                cats = [str(c) for c in f["category"]] if "category" in f else ["unknown"] * n
            for i in range(n):
                if categories is not None and cats[i] not in categories:
                    continue
                self.index.append((si, i))
                self.categories.append(cats[i])
        if max_samples:
            self.index = self.index[:max_samples]
            self.categories = self.categories[:max_samples]
        self._cache: Dict[int, Dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.index)

    def _shard(self, si: int) -> Dict[str, np.ndarray]:
        """Cache the whole shard, with no eviction.

        There are only a handful of shards and every step needs a different sample
        at a different offset, so an LRU policy makes each DataLoader worker
        re-decompress a whole shard every couple of steps.  Measured on a contended
        machine that showed up as ~25 s/step with the GPUs still at 100% utilisation
        (the ranks were waiting on the loader).  Keeping the arrays resident costs a
        few hundred MB per worker and removes it.
        """
        cached = self._cache.get(si)
        if cached is None:
            cached = dict(np.load(self.shards[si], allow_pickle=False))
            self._cache[si] = cached
        return cached

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        si, i = self.index[idx]
        shard = self._shard(si)
        rng = np.random.default_rng((self.seed * 1_000_003 + idx + self.epoch * 7919) % (2**32))

        surf = shard["surface"][i]
        surf_n = shard["surface_normal"][i]
        edge = shard["edge"][i] if "edge" in shard else np.zeros((0, 3), np.float32)
        edge_n = shard["edge_normal"][i] if "edge_normal" in shard else np.zeros((0, 3), np.float32)

        def pick(pts, nrm, n):
            if pts.shape[0] == 0 or n <= 0:
                return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
            n = min(n, pts.shape[0])
            sel = rng.choice(pts.shape[0], size=n, replace=False)
            return pts[sel], nrm[sel]

        s_p, s_n = pick(surf, surf_n, self.pc_size)
        e_p, e_n = pick(edge, edge_n, self.pc_sharpedge_size)

        # augmentation: random rotation about Y (shape prior is orientation
        # canonicalised, so only yaw is free) + small jitter
        if self.augment:
            theta = rng.uniform(0, 2 * math.pi)
            c, s = math.cos(theta), math.sin(theta)
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
            s_p = s_p @ R.T
            e_p = e_p @ R.T
            s_n = s_n @ R.T
            e_n = e_n @ R.T

        pc = np.concatenate([s_p, e_p], axis=0).astype(np.float32)
        feats = np.concatenate([s_n, e_n], axis=0).astype(np.float32)
        # pad the sharp-edge block so the encoder can split it deterministically
        if feats.shape[0] < self.pc_size + self.pc_sharpedge_size:
            pad = self.pc_size + self.pc_sharpedge_size - feats.shape[0]
            pc = np.concatenate([pc, np.zeros((pad, 3), np.float32)], axis=0)
            feats = np.concatenate([feats, np.zeros((pad, 3), np.float32)], axis=0)

        q = shard["query"][i]
        sdf = shard["sdf"][i]
        w = shard["weight"][i]
        # Stratified query subsampling.  The shard stores a *coverage* cloud (16k
        # points: 70% near-surface, 30% uniform volume); sampling it uniformly per
        # step would waste most of the decoder budget on the far field, which is
        # both easy and already down-weighted.  The near-surface block comes first,
        # so keeping most of the draws from it (and a small far-field share to
        # anchor the sign) cuts the decoder cost per step for free.
        total = q.shape[0]
        n_near_total = int(round(total * self.near_ratio))
        n_q = min(self.num_queries, total)
        n_near = min(int(round(n_q * self.near_query_ratio)), n_near_total)
        n_far = n_q - n_near
        idx_near = rng.choice(n_near_total, size=n_near, replace=False)
        idx_far = rng.choice(total - n_near_total, size=n_far, replace=False) + n_near_total
        sel = np.concatenate([idx_near, idx_far])

        out = {
            "pc": torch.from_numpy(pc),
            "feats": torch.from_numpy(feats),
            "query": torch.from_numpy(q[sel]),
            "sdf": torch.from_numpy(sdf[sel]),
            "weight": torch.from_numpy(w[sel]),
            "category": self.categories[idx],
        }
        return out


class LatentShardDataset(Dataset):
    """VAE latents + reference image (single-image -> shape conditioning)."""

    def __init__(
        self,
        shard_dir: str,
        split: str = "train",
        *,
        image_size: int = 224,
        max_latents: Optional[int] = None,
        flip: bool = True,
        seed: int = 0,
        max_samples: Optional[int] = None,
        categories: Optional[Sequence[str]] = None,
    ) -> None:
        self.shard_dir = Path(shard_dir)
        self.split = split
        self.image_size = image_size
        self.max_latents = max_latents
        self.flip = flip and split == "train"
        self.seed = seed
        self.epoch = 0

        self.shards = sorted(self.shard_dir.glob(f"{split}-*.npz"))
        if not self.shards:
            raise FileNotFoundError(f"no {split}-*.npz latent shards in {self.shard_dir}")
        self.index: List[tuple] = []
        self.categories: List[str] = []
        for si, sp in enumerate(self.shards):
            with np.load(sp, allow_pickle=False) as f:
                n = f["latent"].shape[0]
                cats = [str(c) for c in f["category"]] if "category" in f else ["unknown"] * n
            for i in range(n):
                if categories is not None and cats[i] not in categories:
                    continue
                self.index.append((si, i))
                self.categories.append(cats[i])
        if max_samples:
            self.index = self.index[:max_samples]
            self.categories = self.categories[:max_samples]
        self._cache: Dict[int, Dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.index)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _shard(self, si: int) -> Dict[str, np.ndarray]:
        """Cache the whole latent shard (see ``SDFShardDataset._shard``)."""
        cached = self._cache.get(si)
        if cached is None:
            cached = dict(np.load(self.shards[si], allow_pickle=False))
            self._cache[si] = cached
        return cached

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        si, i = self.index[idx]
        shard = self._shard(si)
        latent = shard["latent"][i]
        if self.max_latents and latent.shape[0] > self.max_latents:
            # deterministic subsample so the model can be trained at a shorter
            # sequence length than the VAE was trained with
            step = latent.shape[0] // self.max_latents
            latent = latent[::step][: self.max_latents]
        rng = np.random.default_rng((self.seed * 7919 + idx + self.epoch * 104729) % (2**32))

        img = shard.get("image")
        if img is None:
            raise KeyError("latent shard has no 'image' field; re-run pack_latents.py with --render")
        im = img[i]
        if im.ndim == 4:  # multiple views -> pick one (single-image task)
            im = im[rng.integers(im.shape[0])]
        t = torch.from_numpy(np.ascontiguousarray(im)).float()
        if t.max() > 1.5:
            t = t / 255.0
        if t.shape[0] != 3 and t.shape[-1] == 3:
            t = t.permute(2, 0, 1)
        if self.flip and rng.random() < 0.5:
            t = torch.flip(t, dims=[-1])
        if t.shape[-1] != self.image_size:
            t = torch.nn.functional.interpolate(
                t.unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
            ).squeeze(0)

        return {
            "latent": torch.from_numpy(latent.astype(np.float32)),
            "image": t,
            "category": self.categories[idx],
        }


# --------------------------------------------------------------------------- #
def build_sdf_loader(
    shard_dir: str,
    split: str,
    *,
    batch_size: int,
    num_workers: int = 8,
    **kwargs,
) -> torch.utils.data.DataLoader:
    ds = SDFShardDataset(shard_dir, split, **kwargs)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        drop_last=(split == "train"),
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )


def build_latent_loader(
    shard_dir: str,
    split: str,
    *,
    batch_size: int,
    num_workers: int = 8,
    **kwargs,
) -> torch.utils.data.DataLoader:
    ds = LatentShardDataset(shard_dir, split, **kwargs)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        drop_last=(split == "train"),
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
