"""Rectified-flow / flow-matching utilities for Seed3D-DiT.

Report (Sec. 2.1.2) specifies:

* rectified-flow training with **velocity-field prediction**,
* timesteps sampled from a **logit-normal** distribution,
* a **length-aware timestep shift**: longer latent sequences need more noise to
  be disrupted, so the noise schedule is re-scaled as a function of the
  sequence length (the ``shift`` formulation of Esser et al. / SD3, Eq. 23),
* deterministic ODE sampling at inference time.

The probability path is the standard (conditional) optimal-transport path
``x_t = (1 - t) x_0 + t x_1``, ``u_t = x_1 - x_0`` with ``t in [0, 1]``,
``t = 1`` meaning "clean data".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

__all__ = [
    "logit_normal_timesteps",
    "length_aware_shift",
    "sample_timesteps",
    "flow_matching_loss",
    "euler_sample",
    "heun_sample",
]


def length_aware_shift(
    num_tokens: int,
    *,
    base_tokens: int = 256,
    max_tokens: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Timestep-shift parameter ``mu`` for a latent sequence of length ``num_tokens``.

    ``mu`` grows logarithmically with the sequence length, from ``base_shift`` at
    ``base_tokens`` to ``base_shift + max_shift`` at ``max_tokens`` (the SD3/FLUX
    resolution-shift rule, with sequence length playing the role of resolution).
    The caller then applies ``t' = mu * t / (1 + (mu - 1) * t)``, which pushes
    probability mass towards higher noise -- longer token sets need more noise to
    be disrupted.

    The value is **clamped to** ``[base_shift, base_shift + max_shift]``: the
    affine fit in log-token space goes to zero (and then negative) for sequences
    shorter than ``base_tokens``, and ``mu < 1`` would invert the shift and collapse
    every sampled timestep onto ``t ~ 1`` (measured: ``t_mean = 1.0`` for a 64-token
    model with ``base_tokens = 128``, i.e. no diffusion signal at all).
    """
    if max_tokens <= base_tokens or max_shift == base_shift:
        return float(base_shift)
    import math

    alpha = math.log((base_shift + max_shift) / base_shift) / math.log(max_tokens / base_tokens)
    beta = base_shift - alpha * base_tokens
    mu = alpha * num_tokens + beta
    return float(min(max(mu, base_shift), base_shift + max_shift))


def logit_normal_timesteps(
    batch_size: int,
    *,
    device,
    dtype=torch.float32,
    mean: float = 0.0,
    std: float = 1.0,
    shift: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Draw ``t ~ sigmoid(N(mean, std))``, optionally shifted by ``shift``."""
    z = torch.randn(batch_size, device=device, dtype=dtype, generator=generator) * std + mean
    t = torch.sigmoid(z)
    if shift != 1.0:
        t = shift * t / (1.0 + (shift - 1.0) * t)
    return t


def sample_timesteps(
    batch_size: int,
    num_tokens: int,
    *,
    device,
    dtype=torch.float32,
    mean: float = 0.0,
    std: float = 1.0,
    base_tokens: int = 256,
    max_tokens: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    sampling: str = "logit_normal",
    uniform_eps: float = 1e-5,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample training timesteps with a length-aware shift.

    ``mu`` comes from :func:`length_aware_shift` (SD3/FLUX resolution-shift rule
    with sequence length in place of resolution) and the raw timesteps are drawn
    logit-normally (or uniformly, for evaluation), then shifted by
    ``t' = mu t / (1 + (mu - 1) t)``.

    Pass ``generator`` to make *sampling* reproducible -- without it the global RNG
    is advanced, which silently breaks seed-based reproducibility of the ODE
    sampler even when the initial noise is drawn from an explicit generator.
    """
    if sampling == "uniform":
        t = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
        t = t.clamp(min=uniform_eps, max=1 - uniform_eps)
    else:
        t = logit_normal_timesteps(
            batch_size, device=device, dtype=dtype, mean=mean, std=std, generator=generator
        )

    mu = length_aware_shift(
        num_tokens,
        base_tokens=base_tokens,
        max_tokens=max_tokens,
        base_shift=base_shift,
        max_shift=max_shift,
    )
    if mu != 1.0:
        t = mu * t / (1.0 + (mu - 1.0) * t)
    return t.clamp(min=uniform_eps, max=1.0 - uniform_eps)


def flow_matching_loss(
    model: Callable[..., torch.Tensor],
    latents: torch.Tensor,
    t: torch.Tensor,
    *,
    model_kwargs: Optional[dict] = None,
    noise: Optional[torch.Tensor] = None,
) -> dict:
    """Standard rectified-flow velocity-matching loss.

    ``x_t = (1 - t) * eps + t * z`` and the regression target is
    ``u_t = z - eps`` (constant velocity along the OT path).
    """
    model_kwargs = dict(model_kwargs or {})
    if noise is None:
        noise = torch.randn_like(latents)
    t_b = t.view(-1, *([1] * (latents.ndim - 1)))
    x_t = (1.0 - t_b) * noise + t_b * latents
    target = latents - noise
    pred = model(x_t, t, **model_kwargs)
    loss = (pred - target) ** 2
    loss = loss.flatten(1).mean(dim=1)
    return {"loss": loss, "pred": pred, "x_t": x_t, "target": target}


@torch.no_grad()
def euler_sample(
    model: Callable[..., torch.Tensor],
    shape,
    *,
    num_steps: int = 50,
    device=None,
    dtype=torch.float32,
    model_kwargs: Optional[dict] = None,
    guidance_scale: float = 1.0,
    uncond_kwargs: Optional[dict] = None,
    num_tokens: Optional[int] = None,
    base_tokens: int = 256,
    max_tokens: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    generator: Optional[torch.Generator] = None,
    callback: Optional[Callable[[int, torch.Tensor], None]] = None,
    t_start: float = 0.0,
) -> torch.Tensor:
    """Deterministic Euler integration of the learned velocity field.

    Integrates from ``t = t_start`` (pure noise) to ``t = 1`` (clean latent) in
    ``num_steps`` uniform steps; classifier-free guidance is applied on the
    velocity if ``guidance_scale != 1``.  The same length-aware shift used in
    training is applied to the evaluation timesteps so train/test schedules
    agree.
    """
    model_kwargs = dict(model_kwargs or {})
    x = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    n_tok = int(num_tokens or shape[1])
    ts = sample_timesteps(
        num_steps,
        n_tok,
        device=device,
        dtype=dtype,
        sampling="uniform",
        base_tokens=base_tokens,
        max_tokens=max_tokens,
        base_shift=base_shift,
        max_shift=max_shift,
        generator=generator,
    ).sort().values
    ts = torch.cat([torch.full((1,), t_start, device=device, dtype=dtype), ts])
    dt = ts[1:] - ts[:-1]

    for i in range(num_steps):
        t = ts[i].expand(shape[0])
        v = model(x, t, **model_kwargs)
        if guidance_scale != 1.0:
            if uncond_kwargs is None:
                raise ValueError("guidance_scale != 1 requires uncond_kwargs")
            v_uncond = model(x, t, **uncond_kwargs)
            v = v_uncond + guidance_scale * (v - v_uncond)
        x = x + dt[i] * v
        if callback is not None:
            callback(i, x)
    return x


@torch.no_grad()
def heun_sample(
    model: Callable[..., torch.Tensor],
    shape,
    *,
    num_steps: int = 25,
    device=None,
    dtype=torch.float32,
    model_kwargs: Optional[dict] = None,
    guidance_scale: float = 1.0,
    uncond_kwargs: Optional[dict] = None,
    num_tokens: Optional[int] = None,
    base_tokens: int = 256,
    max_tokens: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Second-order (Heun) sampler; costs 2 NFE per step but is more accurate."""
    model_kwargs = dict(model_kwargs or {})
    x = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    n_tok = int(num_tokens or shape[1])
    ts = sample_timesteps(
        num_steps + 1,
        n_tok,
        device=device,
        dtype=dtype,
        sampling="uniform",
        base_tokens=base_tokens,
        max_tokens=max_tokens,
        base_shift=base_shift,
        max_shift=max_shift,
        generator=generator,
    ).sort().values

    def _v(x_in, t_scalar):
        t = t_scalar.expand(shape[0])
        v = model(x_in, t, **model_kwargs)
        if guidance_scale != 1.0:
            v_u = model(x_in, t, **uncond_kwargs)
            v = v_u + guidance_scale * (v - v_u)
        return v

    for i in range(num_steps):
        t0, t1 = ts[i], ts[i + 1]
        dt = t1 - t0
        k1 = _v(x, t0)
        x_euler = x + dt * k1
        k2 = _v(x_euler, t1)
        x = x + 0.5 * dt * (k1 + k2)
    return x
