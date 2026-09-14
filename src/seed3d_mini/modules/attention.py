"""Attention primitives shared by Seed3D-VAE and Seed3D-DiT.

The vector-set architecture used by Seed3D 1.0 (Sec. 2.1.1 of the technical
report) is a permutation-invariant set-to-set transformer.  None of the blocks
below injects positional information into the latent tokens: the latent set is
"position-encoding-free", which is exactly the property that lets the same
decoder consume 256, 512, ..., 4096 tokens (multi-scale VAE training).

This file is an independent implementation written for the reproduction; the
op/parameter naming intentionally mirrors the widely used open-source
vector-set implementations (Hunyuan3D-2.1 / 3DShape2VecSet lineage) so that
pretrained checkpoints can be mapped onto it for numerical parity checks.  See
THIRD_PARTY_NOTICES.md for attribution and licenses.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

__all__ = [
    "scaled_dot_product_attention",
    "use_math_sdpa",
    "set_math_sdpa",
    "FourierEmbedder",
    "DropPath",
    "MLP",
    "MultiheadAttention",
    "ResidualAttentionBlock",
    "Transformer",
    "MultiheadCrossAttention",
    "ResidualCrossAttentionBlock",
]

# The eikonal regulariser needs a *double* backward through attention, which
# torch's fused (flash / memory-efficient) SDPA kernels do not implement.  Setting
# this flag routes attention through the math backend, which does.  It is toggled
# by ``use_math_sdpa()`` around the eikonal forward pass only.
_FORCE_MATH_SDPA = False


def set_math_sdpa(enabled: bool) -> None:
    global _FORCE_MATH_SDPA
    _FORCE_MATH_SDPA = bool(enabled)


def scaled_dot_product_attention(query, key, value, *args, **kwargs):
    """SDPA wrapper that can be forced onto the double-differentiable math backend."""
    if _FORCE_MATH_SDPA:
        return _math_sdpa(query, key, value, *args, **kwargs)
    return nn.functional.scaled_dot_product_attention(query, key, value, *args, **kwargs)


def _math_sdpa(query, key, value, *args, **kwargs):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.MATH):
        return nn.functional.scaled_dot_product_attention(query, key, value, *args, **kwargs)


class use_math_sdpa:
    """Context manager: force the double-differentiable attention backend."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def __enter__(self):
        global _FORCE_MATH_SDPA
        self.prev = _FORCE_MATH_SDPA
        _FORCE_MATH_SDPA = self.enabled
        return self

    def __exit__(self, *exc):
        global _FORCE_MATH_SDPA
        _FORCE_MATH_SDPA = self.prev
        return False


class FourierEmbedder(nn.Module):
    """Sine/cosine positional encoding.

    For an input ``x`` of shape ``[..., C]`` this returns

    ``[x, sin(pi * 2^0 * x), ..., sin(pi * 2^(F-1) * x), cos(...), ...]``

    i.e. ``include_input=True`` concatenates the raw coordinates first, which
    matters for surface reconstruction (raw coordinates carry the absolute
    location that the frequencies alone can alias).
    """

    def __init__(
        self,
        num_freqs: int = 8,
        logspace: bool = True,
        input_dim: int = 3,
        include_input: bool = True,
        include_pi: bool = True,
    ) -> None:
        super().__init__()
        if logspace:
            frequencies = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        else:
            frequencies = torch.linspace(1.0, 2.0 ** (num_freqs - 1), num_freqs, dtype=torch.float32)
        if include_pi:
            frequencies = frequencies * math.pi

        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs
        self.out_dim = self.get_dims(input_dim)

    def get_dims(self, input_dim: int) -> int:
        temp = 1 if (self.include_input or self.num_freqs == 0) else 0
        return input_dim * (self.num_freqs * 2 + temp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_freqs == 0:
            return x
        embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)
        if self.include_input:
            return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)


class DropPath(nn.Module):
    """Stochastic depth (per-sample residual dropout)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def extra_repr(self) -> str:
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"


class MLP(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        expand_ratio: int = 4,
        output_width: Optional[int] = None,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.c_fc = nn.Linear(width, width * expand_ratio)
        self.c_proj = nn.Linear(width * expand_ratio, output_width if output_width is not None else width)
        self.gelu = nn.GELU()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop_path(self.c_proj(self.gelu(self.c_fc(x))))


class MultiheadAttention(nn.Module):
    """Standard self attention over a token sequence (no positional term)."""

    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.c_qkv = nn.Linear(width, width * 3, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.q_norm = self._make_norm(qk_norm)
        self.k_norm = self._make_norm(qk_norm)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def _make_norm(self, enabled: bool) -> nn.Module:
        return nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1e-6) if enabled else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, n_ctx, _ = x.shape
        qkv = self.c_qkv(x).view(bs, n_ctx, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # 3 x B x H x N x D
        q, k = self.q_norm(q), self.k_norm(k)
        out = scaled_dot_product_attention(q, k, v)  # B x H x N x D
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.drop_path(self.c_proj(out))


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = MultiheadAttention(
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            drop_path_rate=drop_path_rate,
        )
        self.ln_1 = nn.LayerNorm(width, elementwise_affine=True, eps=1e-6)
        self.mlp = MLP(width=width, drop_path_rate=drop_path_rate)
        self.ln_2 = nn.LayerNorm(width, elementwise_affine=True, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """Plain pre-norm transformer trunk (used by the VAE encoder/decoder)."""

    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    drop_path_rate=drop_path_rate,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.resblocks:
            x = block(x)
        return x


class MultiheadCrossAttention(nn.Module):
    """Cross attention with a *fused* KV projection (``c_kv``).

    Keeping the fused KV projection matters: it is what the reference
    checkpoints store, so parity checks can copy weights 1:1.
    """

    def __init__(
        self,
        *,
        width: int,
        heads: int,
        qkv_bias: bool = True,
        n_data: Optional[int] = None,
        data_width: Optional[int] = None,
        qk_norm: bool = False,
        kv_cache: bool = False,
    ) -> None:
        super().__init__()
        self.n_data = n_data
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.data_width = width if data_width is None else data_width
        self.c_q = nn.Linear(width, width, bias=qkv_bias)
        self.c_kv = nn.Linear(self.data_width, width * 2, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.q_norm = self._make_norm(qk_norm)
        self.k_norm = self._make_norm(qk_norm)
        self.kv_cache = kv_cache
        self.data = None

    def _make_norm(self, enabled: bool) -> nn.Module:
        return nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1e-6) if enabled else nn.Identity()

    def forward(self, x: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        bs, n_ctx, _ = x.shape
        q = self.c_q(x).view(bs, n_ctx, self.heads, self.head_dim)
        if self.kv_cache and self.data is not None:
            kv = self.data
        else:
            kv = self.c_kv(data)
            if self.kv_cache:
                self.data = kv
        kv = kv.view(bs, -1, self.heads, 2, self.head_dim)
        k, v = kv[..., 0, :], kv[..., 1, :]
        q, k, v = (rearrange(t, "b n h d -> b h n d") for t in (q, k, v))
        q, k = self.q_norm(q), self.k_norm(k)
        out = scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.c_proj(out)


class ResidualCrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention + MLP residual block."""

    def __init__(
        self,
        *,
        n_data: Optional[int] = None,
        width: int,
        heads: int,
        mlp_expand_ratio: int = 4,
        data_width: Optional[int] = None,
        qkv_bias: bool = True,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        if data_width is None:
            data_width = width
        self.attn = MultiheadCrossAttention(
            n_data=n_data,
            width=width,
            heads=heads,
            data_width=data_width,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
        )
        self.ln_1 = nn.LayerNorm(width, elementwise_affine=True, eps=1e-6)
        self.ln_2 = nn.LayerNorm(data_width, elementwise_affine=True, eps=1e-6)
        self.ln_3 = nn.LayerNorm(width, elementwise_affine=True, eps=1e-6)
        self.mlp = MLP(width=width, expand_ratio=mlp_expand_ratio)

    def forward(self, x: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), self.ln_2(data))
        x = x + self.mlp(self.ln_3(x))
        return x
