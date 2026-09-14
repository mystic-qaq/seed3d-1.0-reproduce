"""Seed3D-VAE: a vector-set (VecSet) variational auto-encoder for TSDF fields.

Faithful to Sec. 2.1.1 of the Seed3D 1.0 technical report:

* dual cross-attention encoder -> self-attention decoder architecture,
* Fourier positional encoding of points, concatenated with surface normals,
* a *TSDF* (not binary occupancy) regression head,
* multi-scale training: latent length ``M`` is randomly drawn from
  ``{256, 512, ..., 4096}``; the decoder is position-encoding-free and
  permutation invariant, so it generalizes to lengths unseen during training,
* ``L_VAE = L_recon + lambda_KL * L_KL`` with a KL warm-up schedule
  (``lambda_KL`` annealed up to ``1e-4``).

Additionally this implementation exposes the pieces needed by a *mini* version:

* ``sdf_head=True`` adds the eikonal (|grad f| = 1) regularizer that makes the
  decoded field a usable continuous SDF for hierarchical / gradient-based
  iso-surface extraction,
* the near-surface and far-field query clouds are constructed by the dataset,
  so the model itself stays agnostic to how the field is supervised.

Reference implementations studied while writing this file (see
THIRD_PARTY_NOTICES.md): 3DShape2VecSet, Dora-VAE, Hunyuan3D-2.1
(``hy3dshape/models/autoencoders``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.attention import (
    FourierEmbedder,
    ResidualAttentionBlock,
    ResidualCrossAttentionBlock,
    Transformer,
    use_math_sdpa,
)
from ..modules.point_sampling import sample_encoder_inputs

__all__ = ["DiagonalGaussianDistribution", "VecSetEncoder", "TSDFDecoder", "Seed3DVAE", "VAEOutput"]


class DiagonalGaussianDistribution:
    """Diagonal Gaussian posterior over the latent token set."""

    def __init__(self, parameters: Union[torch.Tensor, List[torch.Tensor]], deterministic: bool = False):
        if isinstance(parameters, (list, tuple)):
            self.mean, self.logvar = parameters[0], parameters[1]
        else:
            self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self) -> torch.Tensor:
        return self.mean + self.std * torch.randn_like(self.mean)

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self) -> torch.Tensor:
        """KL( q(z|x) || N(0, I) ), averaged over the whole token set."""
        if self.deterministic:
            return self.mean.new_zeros(())
        return 0.5 * torch.mean(self.mean.pow(2) + self.var - 1.0 - self.logvar)


class VecSetEncoder(nn.Module):
    """Surface point cloud -> latent token set ``Z in R^{M x d}``.

    ``CrossAttn(PE(P) || n_P)`` followed by ``L_e`` self-attention blocks, where
    the cross-attention *queries* are FPS-selected surface points (the latent
    slots) and the keys/values are a random subset of the surface points.
    """

    def __init__(
        self,
        *,
        num_latents: int,
        width: int,
        heads: int,
        layers: int = 8,
        embed_dim: int = 64,
        pc_size: int = 5120,
        pc_sharpedge_size: int = 0,
        point_feats: int = 3,
        downsample_ratio: int = 20,
        num_freqs: int = 8,
        include_pi: bool = True,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        use_ln_post: bool = True,
        normal_pe: bool = False,
    ) -> None:
        super().__init__()
        self.num_latents = num_latents
        self.pc_size = pc_size
        self.pc_sharpedge_size = pc_sharpedge_size
        self.downsample_ratio = downsample_ratio
        self.point_feats = point_feats
        self.normal_pe = normal_pe

        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)
        self.pos_dim = self.fourier_embedder.out_dim
        feat_dim = point_feats
        if normal_pe:
            feat_dim = self.pos_dim + max(point_feats - 3, 0)

        self.input_proj = nn.Linear(self.pos_dim + feat_dim, width)
        self.cross_attn = ResidualCrossAttentionBlock(
            width=width, heads=heads, qkv_bias=qkv_bias, qk_norm=qk_norm
        )
        self.self_attn = (
            Transformer(
                n_ctx=num_latents,
                width=width,
                layers=layers,
                heads=heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
            )
            if layers > 0
            else None
        )
        self.ln_post = nn.LayerNorm(width) if use_ln_post else None
        self.pre_kl = nn.Linear(width, embed_dim * 2)

    def _embed(self, pc: torch.Tensor, feats: Optional[torch.Tensor]) -> torch.Tensor:
        pos = self.fourier_embedder(pc)
        if feats is None or self.point_feats == 0:
            return pos
        if self.normal_pe:
            feats = torch.cat([self.fourier_embedder(feats[..., :3]), feats[..., 3:]], dim=-1)
        return torch.cat([pos, feats], dim=-1)

    def forward(
        self,
        pc: torch.Tensor,
        feats: Optional[torch.Tensor] = None,
        *,
        num_latents: Optional[int] = None,
        sample: bool = True,
        generator: Optional[torch.Generator] = None,
    ):
        """Args:
            pc: ``[B, N, 3]`` surface points ordered as ``random || sharpedge``.
            feats: ``[B, N, F]`` point features (normals by default).
        Returns:
            ``(latents, posterior)`` with ``latents`` of shape ``[B, M, embed_dim]``.
        """
        m = int(num_latents or self.num_latents)
        query, data, query_feats, data_feats = sample_encoder_inputs(
            pc,
            feats if feats is not None else pc.new_zeros(*pc.shape[:2], self.point_feats),
            num_latents=m,
            pc_size=self.pc_size,
            pc_sharpedge_size=self.pc_sharpedge_size,
            downsample_ratio=self.downsample_ratio,
        )
        q = self.input_proj(self._embed(query, query_feats))
        kv = self.input_proj(self._embed(data, data_feats))
        latents = self.cross_attn(q, kv)
        if self.self_attn is not None:
            latents = self.self_attn(latents)
        if self.ln_post is not None:
            latents = self.ln_post(latents)

        posterior = DiagonalGaussianDistribution(self.pre_kl(latents))
        if sample:
            if generator is not None:
                eps = torch.randn(posterior.mean.shape, generator=generator, device=posterior.mean.device)
                z = posterior.mean + posterior.std * eps
            else:
                z = posterior.sample()
        else:
            z = posterior.mode()
        return z, posterior


class TSDFDecoder(nn.Module):
    """Latent token set + query points -> truncated signed distance value.

    ``d_hat(x) = MLP( CrossAttn( SelfAttn^{(j)}(PE(x)), Z ) )`` with ``j = 1..L_k``
    self-attention layers over the query tokens before they attend to ``Z``
    (report Eq. 2).

    Where those ``L_k`` self-attention layers live is the one place where the
    report and its own reference implementations disagree:

    * ``decoder_self_attn="query"`` (default, literal reading of Eq. 2) --
      the query tokens are refined before cross-attending to ``Z``.  This costs
      ``O(Q^2)`` in the number of decoded points, so the volume decoder evaluates
      queries in chunks (see :mod:`seed3d_mini.volume_decode`).
    * ``decoder_self_attn="latent"`` -- the ``L_k`` layers refine the *latent
      set* instead, which is what the cited vector-set implementations
      (Dora / Hunyuan3D-2.1) do and what their released checkpoints contain.
    * ``decoder_self_attn="none"`` -- single cross-attention decoder (as in
      3DShape2VecSet's original decoder).

    All three are numerically identical when ``L_k = 0``; the parity test in
    ``tests/test_vae_parity.py`` confirms the checkpoint weight mapping for the
    latent-side variant (which is why it exists here as a switch rather than as a
    hard-coded choice).
    """

    def __init__(
        self,
        *,
        num_latents: int,
        width: int,
        heads: int,
        embed_dim: int = 64,
        num_layers: int = 1,
        self_attn_mode: str = "query",
        mlp_expand_ratio: int = 4,
        out_channels: int = 1,
        num_freqs: int = 8,
        include_pi: bool = True,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        enable_ln_post: bool = True,
        tanh_output: bool = False,
        truncation: float = 0.1,
        query_skip: bool = True,
        query_head_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.query_skip = bool(query_skip)
        if self_attn_mode not in ("query", "latent", "none"):
            raise ValueError(f"self_attn_mode must be query|latent|none, got {self_attn_mode!r}")
        self.enable_ln_post = enable_ln_post
        self.tanh_output = tanh_output
        self.truncation = truncation
        self.self_attn_mode = self_attn_mode
        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)
        self.query_proj = nn.Linear(self.fourier_embedder.out_dim, width)
        self.latents_proj = nn.Linear(embed_dim, width) if embed_dim != width else nn.Identity()
        if not enable_ln_post:
            qk_norm = False

        trunk = (
            Transformer(
                n_ctx=0, width=width, layers=num_layers, heads=heads, qkv_bias=qkv_bias, qk_norm=qk_norm
            )
            if num_layers > 0
            else None
        )
        self.self_attn = trunk if self_attn_mode == "query" else None
        self.latent_self_attn = trunk if self_attn_mode == "latent" else None
        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            n_data=num_latents,
            width=width,
            heads=heads,
            mlp_expand_ratio=mlp_expand_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
        )
        self.ln_post = nn.LayerNorm(width) if enable_ln_post else None
        self.output_proj = nn.Linear(width, out_channels)
        # direct coordinate -> field head (see the note in ``forward``)
        self.query_head = (
            nn.Sequential(
                nn.Linear(self.fourier_embedder.out_dim, query_head_hidden),
                nn.GELU(),
                nn.Linear(query_head_hidden, query_head_hidden),
                nn.GELU(),
                nn.Linear(query_head_hidden, out_channels),
            )
            if self.query_skip
            else None
        )

    def forward(
        self,
        queries: torch.Tensor,
        latents: torch.Tensor,
        *,
        query_embeddings: Optional[torch.Tensor] = None,
        query_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
            queries: ``[B, Q, 3]`` query coordinates.
            latents: ``[B, M, d]`` latent token set.
            query_embeddings: pre-projected query tokens (``[B, Q, width]``).
            query_features: the *raw* Fourier features of the queries
                (``[B, Q, fourier_dim]``), used by the skip head.
        Returns:
            ``[B, Q, out_channels]`` predicted (truncated) signed distance.

        The head reads the **raw query features directly** in addition to the
        cross-attention path.  This skip connection is what makes the decoder
        trainable in practice: measured on a single asset, this decoder fitted the
        field with correlation **-0.05** while a two-layer MLP on the *same*
        Fourier features reached **+0.86** -- i.e. a pure cross-attention decoder of
        this shape could not even represent a coordinate function, regardless of
        the latent.  With the skip, the network can always fall back on the
        coordinate function and use the latent as a residual modulation, which is
        also how the strong implicit decoders are built.
        """
        if query_embeddings is None:
            query_features = self.fourier_embedder(queries).to(latents.dtype)
            query_embeddings = self.query_proj(query_features)
        elif query_features is None:
            # caller supplied projected tokens only: recover the raw features for
            # the skip head (cheap, and keeps the public signature convenient)
            query_features = self.fourier_embedder(queries).to(latents.dtype)
        latents = self.latents_proj(latents)
        if self.latent_self_attn is not None:
            latents = self.latent_self_attn(latents)
        x = query_embeddings
        if self.self_attn is not None:
            x = self.self_attn(x)
        x = self.cross_attn_decoder(x, latents)
        if self.ln_post is not None:
            x = self.ln_post(x)
        d = self.output_proj(x)
        if self.query_skip:
            d = d + self.query_head(query_features)
        if self.tanh_output:
            d = torch.tanh(d) * self.truncation
        return d


@dataclass
class VAEOutput:
    """Container returned by :meth:`Seed3DVAE.forward`."""

    loss: torch.Tensor
    loss_dict: Dict[str, torch.Tensor] = field(default_factory=dict)
    sdf_pred: Optional[torch.Tensor] = None


class Seed3DVAE(nn.Module):
    """Vector-set VAE with a TSDF decoder head.

    The class deliberately separates *encoding* (mesh -> latent) from *decoding*
    (latent -> continuous TSDF field).  Iso-surface extraction lives in
    :mod:`seed3d_mini.volume_decode` because it is a pure inference concern.
    """

    def __init__(
        self,
        *,
        num_latents: int = 512,
        embed_dim: int = 64,
        width: int = 512,
        heads: int = 8,
        num_encoder_layers: int = 8,
        num_decoder_layers: int = 1,
        decoder_self_attn: str = "query",
        pc_size: int = 5120,
        pc_sharpedge_size: int = 0,
        point_feats: int = 3,
        downsample_ratio: int = 20,
        num_freqs: int = 8,
        include_pi: bool = True,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        use_ln_post: bool = True,
        mlp_expand_ratio: int = 4,
        tanh_output: bool = False,
        truncation: float = 0.1,
        query_skip: bool = True,
        lambda_kl: float = 1e-4,
        lambda_eikonal: float = 0.0,
        target_scale: float = 0.1,
        # multi-scale training: draw M from this list every step
        train_num_latents: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.num_latents = num_latents
        self.embed_dim = embed_dim
        self.lambda_kl = lambda_kl
        self.lambda_eikonal = lambda_eikonal
        #: divide the TSDF target by this (its saturation value) before the MSE
        self.target_scale = float(target_scale)
        self.train_num_latents = list(train_num_latents) if train_num_latents else [num_latents]

        self.encoder = VecSetEncoder(
            num_latents=num_latents,
            width=width,
            heads=heads,
            layers=num_encoder_layers,
            embed_dim=embed_dim,
            pc_size=pc_size,
            pc_sharpedge_size=pc_sharpedge_size,
            point_feats=point_feats,
            downsample_ratio=downsample_ratio,
            num_freqs=num_freqs,
            include_pi=include_pi,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            use_ln_post=use_ln_post,
        )
        self.decoder = TSDFDecoder(
            num_latents=num_latents,
            width=width,
            heads=heads,
            embed_dim=embed_dim,
            num_layers=num_decoder_layers,
            self_attn_mode=decoder_self_attn,
            mlp_expand_ratio=mlp_expand_ratio,
            num_freqs=num_freqs,
            include_pi=include_pi,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            tanh_output=tanh_output,
            truncation=truncation,
            query_skip=query_skip,
        )

    # ------------------------------------------------------------------ #
    # encode / decode
    # ------------------------------------------------------------------ #
    def encode(
        self,
        pc: torch.Tensor,
        feats: Optional[torch.Tensor] = None,
        *,
        num_latents: Optional[int] = None,
        sample: bool = True,
        generator: Optional[torch.Generator] = None,
    ):
        return self.encoder(pc, feats, num_latents=num_latents, sample=sample, generator=generator)

    def decode(self, queries: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """Decode a (truncated) signed distance in *world* units.

        The network is trained on ``tsdf / target_scale``; this multiplies back so
        every caller (volume decoding, marching cubes, evaluation) sees a field whose
        zero level is the surface and whose units are the canonical frame.
        """
        return self.decoder(queries, latents) * self.target_scale

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
    def forward(
        self,
        pc: torch.Tensor,
        query_points: torch.Tensor,
        query_sdf: torch.Tensor,
        *,
        feats: Optional[torch.Tensor] = None,
        query_weights: Optional[torch.Tensor] = None,
        num_latents: Optional[int] = None,
        lambda_kl: Optional[float] = None,
        compute_eikonal: Optional[bool] = None,
        **kwargs,
    ) -> VAEOutput:
        """One VAE training step.

        Args:
            pc: ``[B, N, 3]`` surface samples (``random || sharpedge`` blocks).
            query_points: ``[B, Q, 3]`` supervision coordinates.
            query_sdf: ``[B, Q, 1]`` ground-truth TSDF at those coordinates.
            query_weights: optional ``[B, Q, 1]`` per-query loss weights.  The
                dataset uses this to up-weight the near-surface band (where the
                field has to be sharp) against the far field.
        """
        if num_latents is None:
            if self.training and len(self.train_num_latents) > 1:
                idx = torch.randint(len(self.train_num_latents), (1,)).item()
                num_latents = self.train_num_latents[idx]
            else:
                num_latents = self.num_latents

        latents, posterior = self.encode(pc, feats, num_latents=num_latents, sample=True)
        # raw network output == tsdf / target_scale (see ``decode``)
        pred = self.decoder(query_points, latents)

        # Target normalisation.  A *truncated* TSDF lives in [-tau, tau] with
        # tau = 0.1, so the constant-mean predictor already reaches MSE ~4e-4 and is
        # a strong local optimum: the first mini run converged exactly there (recon
        # 2e-4, decoded fields that differed by 0.001 between different shapes,
        # useless meshes).  Dividing the target by its saturation value maps it onto
        # [-1, 1], where predicting the mean costs ~0.09 -- i.e. the degenerate
        # solution is no longer competitive, and the reported loss is directly
        # interpretable as a fraction of the truncation band.
        scale = float(self.target_scale)
        if scale != 1.0:
            query_sdf = query_sdf / scale

        if query_weights is None:
            loss_recon = F.mse_loss(pred, query_sdf)
        else:
            loss_recon = (query_weights * (pred - query_sdf) ** 2).sum() / query_weights.sum().clamp_min(1e-6)

        kl = posterior.kl()
        lam = self.lambda_kl if lambda_kl is None else lambda_kl
        loss = loss_recon + lam * kl

        loss_dict = {"recon": loss_recon.detach(), "kl": kl.detach(), "num_latents": pred.new_tensor(num_latents)}

        do_eikonal = self.lambda_eikonal > 0.0 if compute_eikonal is None else compute_eikonal
        if do_eikonal and self.lambda_eikonal > 0.0:
            # Only the *untruncated* part of the target is a true distance field, so
            # the regularizer is restricted to query points whose |target| is below
            # the clamping value; elsewhere the target is flat by construction and
            # |grad| = 1 would be wrong.
            loss_eik, grad_norm = self.eikonal_loss(query_points, latents, sdf_target=query_sdf)
            loss = loss + self.lambda_eikonal * loss_eik
            loss_dict["eikonal"] = loss_eik.detach()
            loss_dict["grad_norm"] = grad_norm.detach()

        loss_dict["pred_std"] = pred.detach().std()
        return VAEOutput(loss=loss, loss_dict=loss_dict, sdf_pred=pred * scale)

    def eikonal_loss(
        self,
        query_points: torch.Tensor,
        latents: torch.Tensor,
        *,
        sdf_target: Optional[torch.Tensor] = None,
        truncation: Optional[float] = None,
        max_points: int = 512,
        margin: float = 0.9,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Eikonal regularizer: the decoder's spatial gradient should have unit norm.

        Uses ``torch.autograd.grad`` (double backward through the query branch);
        the latent branch is detached so the regularizer only shapes the field.

        Two practical details:

        * **Selection.** A *truncated* SDF is flat outside the band, so imposing
          ``|grad| = 1`` there would be wrong.  When ``sdf_target`` is given, only
          points with ``|target| < margin * truncation`` (the genuine distance
          region) contribute.
        * **Cost.** The double backward needs the *math* attention backend (fused
          CUDA kernels have no second derivative), which materialises a dense
          ``Q x Q`` attention matrix.  The regularizer therefore runs on a random
          subset of the query cloud (``max_points``): still unbiased, and its cost
          no longer depends on the decoder batch size.
        """
        if sdf_target is not None and sdf_target.shape[1] == query_points.shape[1]:
            trunc = float(truncation if truncation is not None else getattr(self.decoder, "truncation", 0.1))
            keep = (sdf_target.detach().abs() < margin * trunc).squeeze(-1)  # [B, Q]
            if keep.sum() < 32:
                keep = torch.ones_like(keep)
        else:
            keep = None

        if keep is not None:
            # sample per batch element so every sample contributes
            idx = torch.multinomial(keep.float() + 1e-6, num_samples=min(max_points, keep.shape[1]), replacement=False)
            q_sel = torch.gather(query_points.detach(), 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        else:
            n = query_points.shape[1]
            if max_points and n > max_points:
                idx = torch.randperm(n, device=query_points.device)[:max_points]
                q_sel = query_points.detach()[:, idx]
            else:
                q_sel = query_points.detach()

        q = q_sel.contiguous().requires_grad_(True)
        with use_math_sdpa():
            pred = self.decode(q, latents.detach())
        grad = torch.autograd.grad(
            outputs=pred.sum(),
            inputs=q,
            create_graph=True,
            retain_graph=True,
        )[0]
        grad_norm = grad.norm(dim=-1, keepdim=True)
        # the network predicts tsdf / target_scale, so an ideal distance field has
        # |grad| = 1 / target_scale in these units
        want = 1.0 / float(self.target_scale)
        loss = ((grad_norm - want) ** 2).mean() / (want ** 2)
        return loss, grad_norm.mean()

    # ------------------------------------------------------------------ #
    # convenience constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, cfg: Dict) -> "Seed3DVAE":
        return cls(**{k: v for k, v in cfg.items() if not k.startswith("_")})

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
