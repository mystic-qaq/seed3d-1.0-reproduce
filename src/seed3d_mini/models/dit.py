"""Seed3D-DiT: rectified-flow diffusion transformer over VecSet latents.

Faithful to Sec. 2.1.2 of the Seed3D 1.0 technical report:

* **Dual image encoder conditioning** -- DINOv2 and RADIO features are extracted
  and fused channel-wise (RADIO supplies the geometry-aware half that resolves
  single-view depth ambiguity),
* **FLUX-style hybrid backbone** -- ``depth`` double-stream blocks (modality
  specific LayerNorm / QKV / MLP with joint attention over the concatenated
  token sequence) followed by ``depth_single_blocks`` single-stream blocks
  (parallel attention+MLP on the shape stream alone),
* velocity prediction for a rectified-flow objective (see ``flow_matching.py``),
* no positional encoding on the latent tokens (the VecSet is a *set*), matching
  the reference vector-set DiTs.

Two additions relative to the reference shape DiT, needed for a small model:

* ``input_size`` is only used for sequence-length-aware bookkeeping; the stream
  dims stay rectangular,
* ``qk_norm`` uses RMSNorm, which is what makes small attention heads train
  stably without warmup tricks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

__all__ = ["Seed3DDiT", "DiTConfig"]

scaled_dot_product_attention = nn.functional.scaled_dot_product_attention


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0, time_factor: float = 1000.0):
    """Sinusoidal embedding of continuous timesteps (FLUX-style: ``t * 1000``)."""
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb.to(t.dtype) if torch.is_floating_point(t) else emb


class GELU(nn.Module):
    def __init__(self, approximate: str = "tanh") -> None:
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.gelu(x.contiguous(), approximate=self.approximate)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x * rrms).to(dtype) * self.scale


class QKNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        return self.query_norm(q).to(v), self.key_norm(k).to(v)


def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    x = scaled_dot_product_attention(q, k, v)
    return rearrange(x, "b h l d -> b l (h d)")


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "b l (k h d) -> k b h l d", k=3, h=self.num_heads)
        q, k = self.norm(q, k, v)
        return self.proj(_attention(q, k, v))


@dataclass
class ModulationOut:
    shift: torch.Tensor
    scale: torch.Tensor
    gate: torch.Tensor


class Modulation(nn.Module):
    """AdaLN modulation: one linear produces shift/scale/gate (x2 for double blocks)."""

    def __init__(self, dim: int, double: bool) -> None:
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: torch.Tensor):
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
        return (ModulationOut(*out[:3]), ModulationOut(*out[3:]) if self.is_double else None)


class DoubleStreamBlock(nn.Module):
    """Modality-specific streams with joint attention (FLUX double block).

    In our case the two modalities are *shape latents* and *image condition
    tokens*; each keeps its own norms, QKV and MLP, and they exchange
    information through attention over the concatenated sequence.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = False) -> None:
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, img: torch.Tensor, txt: torch.Tensor, vec: torch.Tensor):
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        img_modulated = (1 + img_mod1.scale) * self.img_norm1(img) + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "b l (k h d) -> k b h l d", k=3, h=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = (1 + txt_mod1.scale) * self.txt_norm1(txt) + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "b l (k h d) -> k b h l d", k=3, h=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn = _attention(q, k, v)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1]:]

        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        return img, txt


class SingleStreamBlock(nn.Module):
    """Parallel attention + MLP with a single fused input projection."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)
        self.norm = QKNorm(head_dim)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
        q, k, v = rearrange(qkv, "b l (k h d) -> k b h l d", k=3, h=self.num_heads)
        q, k = self.norm(q, k, v)
        attn = _attention(q, k, v)
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), dim=2))
        return x + mod.gate * output


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        return self.linear(x)


@dataclass
class DiTConfig:
    in_channels: int = 64
    hidden_size: int = 512
    num_heads: int = 8
    depth: int = 8
    depth_single_blocks: int = 12
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    num_latents: int = 512
    time_factor: float = 1000.0


class Seed3DDiT(nn.Module):
    """Rectified-flow DiT over VecSet latents, conditioned on fused image tokens."""

    def __init__(
        self,
        *,
        in_channels: int = 64,
        hidden_size: int = 512,
        num_heads: int = 8,
        depth: int = 8,
        depth_single_blocks: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        num_latents: int = 512,
        time_factor: float = 1000.0,
        guidance_embed: bool = False,
        context_in_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}")
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.depth = depth
        self.depth_single_blocks = depth_single_blocks
        self.num_latents = num_latents
        self.time_factor = time_factor
        self.guidance_embed = guidance_embed

        self.latent_in = nn.Linear(in_channels, hidden_size, bias=True)
        # The condition sequence arrives in the conditioner's own width
        # (``cond_dim``); project it into the backbone stream.  The reference DiT has
        # the same ``cond_in`` layer, so this is not an extra assumption.
        self.context_in = (
            nn.Linear(int(context_in_dim), hidden_size)
            if context_in_dim is not None and int(context_in_dim) != hidden_size
            else nn.Identity()
        )
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=hidden_size)
        self.guidance_in = MLPEmbedder(in_dim=256, hidden_dim=hidden_size) if guidance_embed else None

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias)
                for _ in range(depth)
            ]
        )
        self.single_blocks = nn.ModuleList(
            [SingleStreamBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth_single_blocks)]
        )
        self.final_layer = LastLayer(hidden_size, self.out_channels)

    # ``context`` is the (already projected) fused image-token sequence [B, T, hidden]
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        latent = self.latent_in(x)
        vec = self.time_in(timestep_embedding(t, 256, time_factor=self.time_factor).to(dtype=latent.dtype))
        if self.guidance_embed:
            if guidance is None:
                raise ValueError("guidance_embed=True but no guidance strength was passed")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256, time_factor=self.time_factor))

        if context is None:
            cond = latent.new_zeros(latent.shape[0], 1, self.hidden_size)
        else:
            cond = self.context_in(context)

        for block in self.double_blocks:
            latent, cond = block(img=latent, txt=cond, vec=vec)

        latent = torch.cat((cond, latent), dim=1)
        for block in self.single_blocks:
            latent = block(latent, vec=vec)
        latent = latent[:, cond.shape[1]:, ...]
        return self.final_layer(latent, vec)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_dit_from_config(cfg: dict) -> Seed3DDiT:
    return Seed3DDiT(**cfg)
