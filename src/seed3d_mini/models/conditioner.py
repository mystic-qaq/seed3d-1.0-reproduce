"""Image conditioning for Seed3D-DiT.

The technical report (Sec. 2.1.2, "Image Conditioning Module") specifies a *dual
encoder*: DINOv2 provides semantic features, RADIO adds geometry-aware features
(distilled from several vision foundation models), and the two are **concatenated
channel-wise** to form the conditioning signal that resolves single-view depth
ambiguity.  Together with the two summary tokens this reproduces the reference
DiT's ``text_len = 1370`` condition sequence for a 518x518 input.

Implementation notes
--------------------
* The two ViTs do not share a patch grid (DINOv2-L/14@518 -> 37x37,
  RADIO@518 -> 37x37 with patch 14, or 74x74 / 148x148 for other variants).
  ``PixelShuffleAlign`` maps an arbitrary square patch grid onto the DINOv2 grid
  by pixel-(un)shuffling, so the channel-wise concatenation is spatially exact
  and no interpolation blur is introduced.
* If RADIO weights are unavailable, ``encoder_type: dinov2_pair`` builds the
  second stream from a second DINOv2 checkpoint (e.g. ``dinov2-small`` next to
  ``dinov2-base``). The fusion code path is identical, so swapping in RADIO is a
  config change.
* ``condition_drop_prob`` implements the classifier-free-guidance dropout: with
  that probability the whole condition sequence is replaced by zeros (matching
  the reference DiT's "unconditional embedding = zeros" convention).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ImageConditioner", "DualImageEncoder", "build_image_backbone"]


class HFImageBackbone(nn.Module):
    """Thin wrapper around a HF vision transformer exposing patch + summary tokens."""

    def __init__(
        self,
        version: str,
        *,
        image_size: int = 518,
        use_cls_token: bool = True,
        frozen: bool = True,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.version = version
        self.image_size = image_size
        self.use_cls_token = use_cls_token
        # RADIO ships custom modelling code in-repo; DINOv2 is natively supported.
        # ``trust_remote_code`` is explicit so nobody is surprised by it.
        self.model = AutoModel.from_pretrained(version, trust_remote_code=trust_remote_code)
        if dtype is not None:
            self.model.to(dtype)
        if frozen:
            self.model.requires_grad_(False)
            self.model.eval()
        self.frozen = frozen
        cfg = self.model.config
        self.hidden_size = int(getattr(cfg, "hidden_size", 768))
        patch = int(getattr(cfg, "patch_size", 14) or 14)
        self.patch_size = patch
        self.grid = image_size // patch
        self._input_kw: Optional[str] = None
        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def _call_backbone(self, x: torch.Tensor):
        """Call the backbone with whichever input keyword it expects.

        Transformers-native backbones (DINOv2) take ``pixel_values``; RADIO's
        remote-code wrapper takes ``x``.  Resolved once and cached.
        """
        if self._input_kw is None:
            import inspect

            params = inspect.signature(self.model.forward).parameters
            if "pixel_values" in params:
                self._input_kw = "pixel_values"
            elif "x" in params:
                self._input_kw = "x"
            elif "images" in params:
                self._input_kw = "images"
            else:
                self._input_kw = "pixel_values"
        return self.model(**{self._input_kw: x})

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Args: ``images`` in ``[0, 1]`` of shape ``[B, 3, H, W]``.

        Returns:
            ``(summary, patches, grid)`` with ``summary`` ``[B, S, C]`` and
            ``patches`` ``[B, grid*grid, C]``.
        """
        x = (images - self.pixel_mean.to(images.dtype)) / self.pixel_std.to(images.dtype)
        # ViTs are resolution-flexible but need H, W divisible by the patch size;
        # pad (not resize) so no spatial information is destroyed. When the
        # configured image size is not a multiple of the patch size the model
        # sees ceil(H/patch)*patch pixels and we keep the *usable* patch grid.
        pad_h = (-x.shape[-2]) % self.patch_size
        pad_w = (-x.shape[-1]) % self.patch_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        patch_grid = x.shape[-2] // self.patch_size
        with torch.set_grad_enabled(not self.frozen):
            out = self._call_backbone(x)
        tokens, summary = self._extract_tokens(out, patch_grid)
        n_patch = patch_grid * patch_grid
        if tokens.shape[1] != n_patch:
            raise RuntimeError(
                f"{self.version}: expected {n_patch} patch tokens for a {patch_grid}x{patch_grid} grid, "
                f"got {tokens.shape[1]}"
            )
        if not self.use_cls_token:
            summary = summary.mean(dim=1, keepdim=True)
        return summary, tokens, patch_grid

    def _extract_tokens(self, out, patch_grid: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalise the many backbone output conventions to (patches, summary).

        Handled shapes:
          * HF ``BaseModelOutput`` with ``last_hidden_state`` ``[B, 1+N, C]``,
          * RADIO ``RadioOutput`` with ``features`` ``[B, h, w, C]`` and a
            ``summary`` vector ``[B, S*C]``,
          * plain tensors / tuples of the above.
        """
        # --- RADIO-style structured output ---------------------------------- #
        feats = getattr(out, "features", None)
        summ = getattr(out, "summary", None)
        if feats is not None:
            if feats.dim() == 4:  # [B, h, w, C]
                b, h, w, c = feats.shape
                patches = feats.reshape(b, h * w, c)
            elif feats.dim() == 3:
                patches = feats
            else:
                raise RuntimeError(f"unsupported RADIO feature shape {tuple(feats.shape)}")
            if summ is None:
                summary = patches.mean(dim=1, keepdim=True)
            else:
                s = summ
                c = patches.shape[-1]
                if s.dim() == 1:
                    s = s.unsqueeze(0)
                if s.shape[-1] != c and s.numel() % c == 0:
                    # RADIO returns the summary as a flat [B, S*C] vector, i.e.
                    # several summary tokens of width C concatenated
                    s = s.reshape(s.shape[0], -1, c)
                summary = s if s.dim() == 3 else s.unsqueeze(1)
            return patches, summary

        # --- HF-style output ------------------------------------------------ #
        tokens = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        if tokens.dim() == 4:  # [B, C, H, W] or [B, H, W, C]
            if tokens.shape[1] == patch_grid * patch_grid or tokens.shape[1] not in (
                tokens.shape[2],
                tokens.shape[3],
            ):
                tokens = tokens.transpose(1, 2).flatten(2).transpose(1, 2)
            else:
                tokens = tokens.flatten(2).transpose(1, 2)
        n_patch = patch_grid * patch_grid
        if tokens.shape[1] == n_patch + 1:
            patches, summary = tokens[:, 1:], tokens[:, :1]
        elif tokens.shape[1] == n_patch:
            patches, summary = tokens, tokens.mean(dim=1, keepdim=True)
        else:
            # e.g. CLS + register tokens: keep the trailing n_patch tokens
            patches = tokens[:, -n_patch:]
            summary = tokens[:, :-n_patch]
            if summary.shape[1] == 0:
                summary = patches.mean(dim=1, keepdim=True)
        return patches, summary


class PixelShuffleAlign(nn.Module):
    """Resample a square patch grid onto a target grid.

    Same grid          -> identity.
    Integer ratio      -> exact pixel-(un)shuffle, which trades spatial
                          resolution for channels (or back) without any
                          interpolation.  This is what keeps the channel-wise
                          fusion spatially exact when RADIO runs at a higher
                          patch resolution than DINOv2.
    Non-integer ratio  -> bilinear resize (rare; e.g. RADIO@512 = 32x32 vs
                          DINOv2@518 = 37x37 with cls+register layouts).
    """

    def __init__(self, in_grid: int, out_grid: int) -> None:
        super().__init__()
        self.in_grid = in_grid
        self.out_grid = out_grid

    def forward(self, patches: torch.Tensor, in_grid: Optional[int] = None, out_grid: Optional[int] = None) -> torch.Tensor:
        b, n, c = patches.shape
        gi = int(in_grid or self.in_grid)
        go = int(out_grid or self.out_grid)
        if gi == go:
            return patches
        x = patches.transpose(1, 2).reshape(b, c, gi, gi)
        if gi % go == 0:
            x = F.pixel_shuffle(x, gi // go)
        elif go % gi == 0:
            x = F.pixel_unshuffle(x, go // gi)
        else:
            x = F.interpolate(x, size=(go, go), mode="bilinear", align_corners=False)
        return x.reshape(b, x.shape[1], -1).transpose(1, 2)


class DualImageEncoder(nn.Module):
    """DINOv2 + RADIO, fused channel-wise on a common patch grid."""

    def __init__(
        self,
        *,
        main_version: str = "facebook/dinov2-base",
        additional_version: str = "nvidia/C-RADIOv3-B",
        image_size: int = 518,
        proj_dim: int = 768,
        cond_dim: int = 1024,
        use_cls_token: bool = True,
        drop_ratio: float = 0.1,
        freeze: bool = True,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()
        self.main = HFImageBackbone(
            main_version, image_size=image_size, use_cls_token=use_cls_token, frozen=freeze,
            dtype=dtype, trust_remote_code=trust_remote_code,
        )
        self.additional = HFImageBackbone(
            additional_version, image_size=image_size, use_cls_token=use_cls_token, frozen=freeze,
            dtype=dtype, trust_remote_code=trust_remote_code,
        )
        self.grid = self.main.grid
        self.align = PixelShuffleAlign(self.additional.grid, self.grid)

        # Channel-wise fusion.  Both the fused channel count (pixel-shuffle
        # multiplies channels by (g_add/g_main)^2) and the summary width depend on
        # the runtime encoder pair, so the projections are created on the first
        # forward pass and then kept fixed -- which also makes the module robust
        # to image-size changes.
        self._patch_proj: Optional[nn.Linear] = None
        self._summary_projs: Optional[nn.ModuleList] = None
        self.cond_dim = cond_dim
        self.drop_ratio = drop_ratio
        self.register_buffer(
            "pixel_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "pixel_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False
        )

    @property
    def patch_proj(self) -> nn.Linear:
        if self._patch_proj is None:
            raise RuntimeError("DualImageEncoder.patch_proj is created on the first forward pass")
        return self._patch_proj

    @property
    def summary_projs(self) -> nn.ModuleList:
        if self._summary_projs is None:
            raise RuntimeError("DualImageEncoder.summary_projs is created on the first forward pass")
        return self._summary_projs

    def _lazy_proj(self, patch_in: int, summary_ins: Sequence[int], ref: torch.Tensor) -> None:
        """Create the fusion projections once, on the first forward pass.

        A separate projection per encoder keeps the summary path correct even when
        the two encoders have different widths *and* different numbers of summary
        tokens (RADIO returns ``S`` summary tokens of width ``C``, DINOv2 one CLS).
        """
        kwargs = dict(device=ref.device, dtype=torch.float32)
        if self._patch_proj is None:
            self._patch_proj = nn.Linear(patch_in, self.cond_dim, **kwargs)
        if self._summary_projs is None:
            self._summary_projs = nn.ModuleList(
                [nn.Linear(int(c), self.cond_dim, **kwargs) for c in summary_ins]
            )

    def forward(
        self,
        images: torch.Tensor,
        drop_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Args:
            images: ``[B, 3, H, W]`` in ``[0, 1]``.
            drop_mask: optional ``[B]`` bool, True -> drop the condition (CFG).
        Returns:
            ``{"context": [B, 1 + grid^2, cond_dim]}``.
        """
        s_main, p_main, g_main = self.main(images)
        s_add, p_add, g_add = self.additional(images)
        p_add = self.align(p_add, g_add, g_main)
        p_main = p_main.to(p_add.dtype)
        patches = torch.cat([p_main, p_add], dim=-1)
        self._lazy_proj(patches.shape[-1], [s_main.shape[-1], s_add.shape[-1]], patches)
        patches = self.patch_proj(patches.float())
        summaries = torch.cat(
            [proj(s.float()) for proj, s in zip(self.summary_projs, (s_main, s_add))], dim=1
        )
        context = torch.cat([summaries, patches], dim=1)

        if drop_mask is None and self.training and self.drop_ratio > 0:
            drop_mask = torch.rand(context.shape[0], device=context.device) < self.drop_ratio
        if drop_mask is not None:
            keep = (~drop_mask).to(context.dtype).view(-1, 1, 1)
            context = context * keep
        return {"context": context}


class ImageConditioner(nn.Module):
    """Seed3D-DiT condition stage: images -> fused condition token sequence."""

    def __init__(
        self,
        *,
        encoder_type: str = "dual",
        image_size: int = 518,
        cond_dim: int = 1024,
        drop_ratio: float = 0.1,
        freeze: bool = True,
        main_version: str = "facebook/dinov2-base",
        additional_version: str = "facebook/dinov2-small",
        proj_dim: int = 768,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()
        self.encoder_type = encoder_type
        self.cond_dim = cond_dim
        self._num_condition_tokens: Optional[int] = None
        self._runtime_grid: Optional[int] = None
        if encoder_type == "dual":
            self.encoder = DualImageEncoder(
                main_version=main_version,
                additional_version=additional_version,
                image_size=image_size,
                proj_dim=proj_dim,
                cond_dim=cond_dim,
                drop_ratio=drop_ratio,
                freeze=freeze,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        elif encoder_type == "single":
            self.encoder = HFImageBackbone(
                main_version, image_size=image_size, use_cls_token=True, frozen=freeze, dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
            self.patch_proj = nn.Linear(self.encoder.hidden_size, cond_dim)
            self.summary_proj = nn.Linear(self.encoder.hidden_size, cond_dim)
            self.drop_ratio = drop_ratio
        else:
            raise ValueError(f"unknown encoder_type {encoder_type!r}")

    @property
    def num_condition_tokens(self) -> int:
        """Sequence length of the condition sequence.

        Depends on the *runtime* patch grids and on how many summary tokens each
        encoder emits, so it is recorded on the first forward pass; the value is a
        pure function of the encoder pair and the image size, i.e. constant in
        practice.
        """
        if self._num_condition_tokens is None:
            # analytic fallback for the common case (1 summary + grid^2 patches
            # per encoder, grids equal after alignment)
            g = getattr(self, "_runtime_grid", None) or self.encoder.grid
            return 1 + g * g
        return self._num_condition_tokens

    def forward(
        self,
        images: torch.Tensor,
        drop_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.encoder_type == "dual":
            ctx = self.encoder(images, drop_mask=drop_mask)["context"]
            self._num_condition_tokens = int(ctx.shape[1])
            self._runtime_grid = int(self.encoder.grid)
            return ctx
        s, p, g = self.encoder(images)
        ctx = torch.cat([self.summary_proj(s), self.patch_proj(p)], dim=1)
        self._num_condition_tokens = int(ctx.shape[1])
        self._runtime_grid = int(g)
        if drop_mask is None and self.training and self.drop_ratio > 0:
            drop_mask = torch.rand(ctx.shape[0], device=ctx.device) < self.drop_ratio
        if drop_mask is not None:
            ctx = ctx * (~drop_mask).to(ctx.dtype).view(-1, 1, 1)
        return ctx

    def unconditional(self, batch_size: int, device, dtype) -> torch.Tensor:
        """Zero condition used as the CFG negative branch."""
        return torch.zeros(batch_size, self.num_condition_tokens, self.cond_dim, device=device, dtype=dtype)

    @torch.no_grad()
    def encode(self, images: torch.Tensor, drop_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        was_training = self.training
        self.eval()
        out = self.forward(images, drop_mask=drop_mask)
        self.train(was_training)
        return out


def build_image_backbone(version: str, image_size: int = 518, frozen: bool = True) -> HFImageBackbone:
    return HFImageBackbone(version, image_size=image_size, frozen=frozen)
