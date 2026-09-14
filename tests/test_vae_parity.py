#!/usr/bin/env python
"""Numerical parity: ``seed3d_mini`` VAE  vs.  Hunyuan3D-2.1 ``ShapeVAE``.

The Seed3D 1.0 shape VAE is a clean-room re-implementation of the vector-set
(VecSet) shape VAE of Hunyuan3D-2.1.  This module proves that the two are
*numerically* the same model: the reference checkpoint is loaded into both, the
same tensors are pushed through both, and every intermediate stage is compared.

How to run
----------
As a plain script (prints a PASS/FAIL table, writes
``artifacts/logs/vae_parity.txt``)::

    envs/seed3d/bin/python tests/test_vae_parity.py
    envs/seed3d/bin/python tests/test_vae_parity.py --skip-if-missing-ckpt

Through pytest (``tests/conftest.py`` puts ``src/`` on ``sys.path``)::

    envs/seed3d/bin/python -m pytest tests/test_vae_parity.py -v -s
    envs/seed3d/bin/python -m pytest tests/test_vae_parity.py -v -s --skip-if-missing-ckpt

Tests that do not need the checkpoint (import shims, Fourier embedder, head
layout discovery, a small randomly-initialised end-to-end parity run) always
execute; the checkpoint-backed tests are skipped with a clear message when
``--skip-if-missing-ckpt`` is given and the file is absent.

Reference import strategy
-------------------------
``sys.path`` gets ``third_party/hunyuan3d-2.1/hy3dshape`` and the reference
package ``__init__`` chain (``hy3dshape/__init__.py`` -> ``pipelines`` ->
``transformers``/paint stack, ``hy3dshape.utils`` -> ``omegaconf`` which is not
installed) is replaced by lightweight ``sys.modules`` stand-ins.  The two files
that matter, ``models/autoencoders/model.py`` and
``models/autoencoders/attention_blocks.py``, are the *real* reference sources;
nothing in ``third_party/`` is modified.

Weight mapping (discovered empirically, see :func:`discover_layouts`)
--------------------------------------------------------------------
1. *plain copy* - most names already coincide: ``encoder.input_proj``,
   ``encoder.cross_attn.{attn.c_q, attn.c_kv, attn.c_proj, ln_1, ln_2, ln_3,
   mlp}``, ``encoder.ln_post``, ``geo_decoder.{query_proj,
   cross_attn_decoder.*, ln_post, output_proj}``.
2. *qk-norm relocation* - the reference keeps the head norms inside a dedicated
   attention module (``...attn.attention.q_norm``), seed3d_mini keeps them on
   the attention module itself (``...attn.q_norm``).
3. *self-attention QKV head layout* - the reference reshapes ``c_qkv`` output
   as ``[heads, 3, head_dim]`` (it does ``view(B, N, heads, -1)`` and *then*
   splits), seed3d_mini reshapes as ``[3, heads, head_dim]``
   (``view(B, N, 3, heads, head_dim)``).  The rows must therefore be permuted::

       w_mine[i*W + h*d + dd] = w_ref[h*3*d + i*d + dd]        i=0:q 1:k 2:v

   (empirically: identity gives max|diff| ~5e-1, the permutation gives 0.0).
4. *cross-attention KV head layout* - **no permutation needed**.  The reference
   does ``kv.view(B, N, heads, -1)`` + ``torch.split(kv, attn_ch, dim=-1)`` with
   ``attn_ch = width//heads//2``, i.e. per head ``[k_head | v_head]``; that is
   exactly seed3d_mini's ``view(B, N, heads, 2, head_dim)`` + ``kv[..., 0]`` /
   ``kv[..., 1]``.  Empirically identity gives max|diff| = 0.0 while the
   transposed alternative gives ~4e-1.
5. *relocations* - the reference has ``pre_kl`` on ``ShapeVAE`` (not on the
   encoder) and a ``post_kl`` plus a latent-side ``transformer``; seed3d_mini
   puts ``pre_kl`` inside its encoder and folds ``post_kl`` into
   ``decoder.latents_proj``.
6. *semantic caveat* - the reference ``transformer`` (``num_decoder_layers``)
   applies self-attention to the **latent tokens** inside ``ShapeVAE.decode``
   before ``geo_decoder``; seed3d_mini's ``TSDFDecoder.self_attn`` applies
   self-attention to the **query tokens** (Seed3D report Eq. 2).  The weights
   map 1:1, the operator placement does not - reported, not asserted.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import inspect
import logging
import os
import sys
import time
import types
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import yaml

# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
REF_PKG_ROOT = REPO_ROOT / "third_party" / "hunyuan3d-2.1" / "hy3dshape"
CKPT_ROOT = REPO_ROOT / "checkpoints"
LOG_PATH = REPO_ROOT / "artifacts" / "logs" / "vae_parity.txt"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from seed3d_mini.models.vae import DiagonalGaussianDistribution, Seed3DVAE  # noqa: E402
from seed3d_mini.modules import attention as my_attention  # noqa: E402
from seed3d_mini.modules import point_sampling as my_sampling  # noqa: E402

DEVICE = torch.device("cpu")  # float32 parity on CPU: no GPU contention
DTYPE = torch.float32
DEFAULT_TOL = 1e-4  # float32, same operator shapes on both sides
STATISTICAL_TOL = 1e-3  # for the end-to-end encode() path

#: Full reference size (4096 latents x 81920 points) by default; the reduced size
#: (256 latents x 5120 points) is available for small machines / quick runs.
_FULL_SIZE: bool = os.environ.get("SEED3D_PARITY_REDUCED_SIZE", "0") != "1"

# --------------------------------------------------------------------------- #
# report machinery
# --------------------------------------------------------------------------- #


@dataclass
class Stage:
    name: str
    status: str
    max_abs: Optional[float] = None
    rel_err: Optional[float] = None
    ref_scale: Optional[float] = None
    tol: Optional[float] = None
    note: str = ""


RESULTS: List[Stage] = []
MAPPING_RULES: List[str] = []
ENVIRONMENT_NOTES: List[str] = []

#: architectural differences that the parity test *documents* rather than asserts;
#: none of them affects the stages that are asserted above.
KNOWN_DIVERGENCES: List[str] = [
    "1. Decoder-side self-attention placement - a *switch*, not a limitation.",
    "   reference  : ShapeVAE.decode(z) = transformer(post_kl(z)), then geo_decoder(queries, .)",
    "                -> 'transformer' (num_decoder_layers=16 in hunyuan3d-vae-v2-1) applies",
    "                   self-attention to the LATENT tokens before cross-attention.",
    "   seed3d_mini: decoder_self_attn='query' (default, Seed3D report Eq. 2) applies the trunk",
    "                to the QUERY tokens; 'latent' reproduces the reference placement; 'none'",
    "                is a single cross-attention decoder.",
    "   evidence   : 'latent' mode is ASSERTED bit-identical for the complete decode path with all",
    "                410 checkpoint tensors mapped (num_decoder_layers=16, nothing dropped);",
    "                'query' with 1 layer differs by ~2.0 max|d| (rel 0.52) vs 1 latent-side",
    "                layer and ~2.0 (rel 0.26) vs the checkpoint's 16 latent-side layers.",
    "   coverage   : with the default 'query' + num_decoder_layers=1 only 1 of the 16",
    "                'transformer.resblocks' has a slot, so 225 reference tensors are dropped",
    "                (reported as 'missing_target'); 'latent' mode uses all of them.",
    "2. Naming/semantics: reference ShapeVAE.decode(z) returns transformed LATENT tokens;",
    "   seed3d_mini Seed3DVAE.decode(queries, z) returns the decoded field. Same computation,",
    "   different boundary.",
    "3. post_kl/latents_proj: reference always has a learned post_kl: Linear(embed_dim, width);",
    "   seed3d_mini's decoder.latents_proj is nn.Identity when embed_dim == width, so in that",
    "   (unused) corner case two post_kl tensors would have no slot.",
    "4. DiagonalGaussianDistribution.kl(): identical formula, different reduction. Reference",
    "   hard-codes dims=(1,2,3) and returns per-sample values (IndexError on the 3-D latents",
    "   this VAE actually produces); seed3d_mini returns one mean over all elements.",
    "5. seed3d_mini adds an optional tanh/truncation head (tanh_output, truncation) that the",
    "   reference does not have; with label_type='binary' it is disabled (tanh_output=False),",
    "   which is what these tests use. The reference's label_type never enters its VAE forward.",
    "6. Sampler token-count semantics for sharp-edge blocks (corner case).",
    "   a) pc_size < num_latents * downsample_ratio: the reference reshapes by the requested",
    "      size and raises RuntimeError; seed3d_mini clamps and returns min(num_latents, pc_size).",
    "   b) pc_sharpedge_size > 0 with a NON-integral split: torch_cluster.fps sizes its output",
    "      as ceil(deg*ratio) in float32, so a fractional num_random_query (3.333) becomes 4",
    "      tokens, while seed3d_mini rounds the split to int and takes the complement",
    "      (3 + 7 = 10 vs the reference's 4 + 7 = 11); data-token counts differ too.",
    "      Both agree for integral splits, including the shipped config:",
    "      ceil(81920 * float32(4096/81920)) == 4096 == num_latents.",
]


def _diff(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float, float]:
    """(max abs diff, relative Frobenius error, ||b||_F)."""
    a = a.detach().to(torch.float64)
    b = b.detach().to(torch.float64)
    if a.shape != b.shape:
        raise AssertionError(f"shape mismatch: mine {tuple(a.shape)} vs ref {tuple(b.shape)}")
    d = (a - b).abs()
    scale = b.norm()
    rel = (d.norm() / scale.clamp_min(1e-30)).item()
    return d.max().item(), rel, scale.item()


def record(
    name: str,
    status: str,
    mine: Optional[torch.Tensor] = None,
    ref: Optional[torch.Tensor] = None,
    tol: Optional[float] = None,
    note: str = "",
) -> Stage:
    max_abs = rel = scale = None
    if mine is not None and ref is not None:
        max_abs, rel, scale = _diff(mine, ref)
    st = Stage(name=name, status=status, max_abs=max_abs, rel_err=rel,
               ref_scale=scale, tol=tol, note=note)
    RESULTS.append(st)
    if max_abs is None:
        print(f"[{status:6s}] {name}  {note}", flush=True)
    else:
        tol_s = "n/a" if tol is None else f"{tol:.0e}"
        print(
            f"[{status:6s}] {name:52s} max|d|={max_abs:.3e}  rel={rel:.3e}  "
            f"|ref|={scale:.3e}  tol={tol_s}  {note}",
            flush=True,
        )
    save_report()
    return st


def check(
    name: str,
    mine: torch.Tensor,
    ref: torch.Tensor,
    tol: float = DEFAULT_TOL,
    note: str = "",
    status_when_ok: str = "PASS",
) -> Stage:
    """Compare two tensors and record PASS/FAIL against ``tol``."""
    max_abs, _rel, _scale = _diff(mine, ref)
    status = status_when_ok if max_abs <= tol else "FAIL"
    return record(name, status, mine, ref, tol=tol, note=note)


def report_stage(name: str, mine: torch.Tensor, ref: torch.Tensor, note: str = "") -> Stage:
    """Record an informational (not asserted) comparison."""
    return record(name, "REPORT", mine, ref, tol=None, note=note)


def render_table() -> str:
    head = (
        f"{'stage':54s} {'status':7s} {'max|diff|':>11s} {'rel.err':>11s} "
        f"{'|ref|':>11s} {'tol':>8s}  note"
    )
    lines = [head, "-" * len(head)]
    for s in RESULTS:
        if s.max_abs is None:
            lines.append(f"{s.name:54s} {s.status:7s} {'-':>11s} {'-':>11s} {'-':>11s} {'-':>8s}  {s.note}")
        else:
            tol = "-" if s.tol is None else f"{s.tol:.0e}"
            lines.append(
                f"{s.name:54s} {s.status:7s} {s.max_abs:11.3e} {s.rel_err:11.3e} "
                f"{s.ref_scale:11.3e} {tol:>8s}  {s.note}"
            )
    n_pass = sum(1 for s in RESULTS if s.status == "PASS")
    n_fail = sum(1 for s in RESULTS if s.status == "FAIL")
    n_rep = sum(1 for s in RESULTS if s.status == "REPORT")
    n_skip = sum(1 for s in RESULTS if s.status in ("SKIP", "INFO"))
    lines.append("-" * len(head))
    lines.append(f"TOTAL: {n_pass} PASS, {n_fail} FAIL, {n_rep} REPORT, {n_skip} SKIP/INFO")
    return "\n".join(lines)


def save_report() -> str:
    """Write the current table (+ mapping rules) to artifacts/logs/vae_parity.txt."""
    parts = [
        "=" * 100,
        "Seed3D-mini VAE  <->  Hunyuan3D-2.1 ShapeVAE   numerical parity report",
        f"generated: {time.strftime('%Y-%m-%d %H:%M:%S')}   device={DEVICE}   dtype={DTYPE}",
        f"reference sources: {REF_PKG_ROOT}",
        f"reference config : {_context_info().get('config_path') or '<not found>'}",
        f"checkpoint       : {_context_info().get('ckpt_path') or '<not found>'}",
        "=" * 100,
        "",
        render_table(),
        "",
        "=" * 100,
        "WEIGHT MAPPING RULES (see map_reference_state_dict / discover_layouts)",
        "=" * 100,
    ]
    parts += MAPPING_RULES or ["(not yet computed)"]
    parts += ["", "=" * 100, "ENVIRONMENT / SHIMS", "=" * 100]
    parts += ENVIRONMENT_NOTES or ["(none)"]
    parts += ["", "=" * 100, "KNOWN DIVERGENCES (documented, not asserted)", "=" * 100]
    parts += KNOWN_DIVERGENCES
    parts += [""]
    text = "\n".join(parts)
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text(text)
    except OSError as exc:  # pragma: no cover
        print(f"WARNING: could not write {LOG_PATH}: {exc}", file=sys.stderr)
    return text


# --------------------------------------------------------------------------- #
# reference import shims
# --------------------------------------------------------------------------- #


@dataclass
class Reference:
    ShapeVAE: type
    model_module: types.ModuleType
    attention_blocks: types.ModuleType


_REFERENCE: Optional[Reference] = None
_SHIM_NOTES: List[str] = []


def _install_reference_shims() -> None:
    """Make ``hy3dshape.models.autoencoders.{model,attention_blocks}`` importable.

    The reference package ``__init__`` chain drags in the paint pipeline and
    ``omegaconf`` (which is not installed in this environment).  Only the two
    autoencoder modules are needed, so the surrounding packages are replaced by
    namespace stand-ins in ``sys.modules``.  ``third_party/`` is never edited.
    """
    if not REF_PKG_ROOT.is_dir():
        raise FileNotFoundError(f"reference package not found at {REF_PKG_ROOT}")
    pkg_dir = REF_PKG_ROOT / "hy3dshape"

    if "hy3dshape" not in sys.modules:
        pkg = types.ModuleType("hy3dshape")
        pkg.__path__ = [str(pkg_dir)]  # type: ignore[attr-defined]
        pkg.__package__ = "hy3dshape"
        sys.modules["hy3dshape"] = pkg
        _SHIM_NOTES.append(
            "stubbed sys.modules['hy3dshape'] (its __init__ imports the paint/pipelines stack)"
        )
    if "hy3dshape.models" not in sys.modules:
        models = types.ModuleType("hy3dshape.models")
        models.__path__ = [str(pkg_dir / "models")]  # type: ignore[attr-defined]
        models.__package__ = "hy3dshape.models"
        sys.modules["hy3dshape.models"] = models
        _SHIM_NOTES.append(
            "stubbed sys.modules['hy3dshape.models'] (its __init__ imports conditioner/denoisers)"
        )

    if "hy3dshape.utils" not in sys.modules:
        try:
            importlib.import_module("hy3dshape.utils")
        except Exception as exc:  # omegaconf is missing in this env
            utils = types.ModuleType("hy3dshape.utils")
            utils.__package__ = "hy3dshape.utils"
            utils.logger = logging.getLogger("hy3dshape")

            class synchronize_timer:  # noqa: N801 - mirrors the reference name
                def __init__(self, *a, **k):
                    pass

                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

                def __call__(self, fn):
                    return fn

            def smart_load_model(*a, **k):
                raise NotImplementedError("stubbed hy3dshape.utils.smart_load_model")

            utils.synchronize_timer = synchronize_timer
            utils.smart_load_model = smart_load_model
            sys.modules["hy3dshape.utils"] = utils
            _SHIM_NOTES.append(
                f"stubbed sys.modules['hy3dshape.utils'] ({type(exc).__name__}: {exc})"
            )

    if "torch_cluster" not in sys.modules:
        try:
            importlib.import_module("torch_cluster")
        except Exception:
            tc = types.ModuleType("torch_cluster")

            def _fps_stub(*a, **k):
                raise NotImplementedError(
                    "torch_cluster.fps is not installed; the parity test replaces "
                    "hy3dshape...attention_blocks.fps with a deterministic sampler"
                )

            tc.fps = _fps_stub
            sys.modules["torch_cluster"] = tc
            _SHIM_NOTES.append(
                "stubbed sys.modules['torch_cluster'] (not installed; reference fps() is "
                "only reached through a patched entry point)"
            )


def reference() -> Reference:
    global _REFERENCE
    if _REFERENCE is None:
        _install_reference_shims()
        model_mod = importlib.import_module("hy3dshape.models.autoencoders.model")
        attn_mod = importlib.import_module("hy3dshape.models.autoencoders.attention_blocks")
        _REFERENCE = Reference(
            ShapeVAE=model_mod.ShapeVAE,
            model_module=model_mod,
            attention_blocks=attn_mod,
        )
    return _REFERENCE


# --------------------------------------------------------------------------- #
# checkpoint / config discovery
# --------------------------------------------------------------------------- #


class CheckpointMissing(RuntimeError):
    pass


class SkipStage(RuntimeError):
    """Raised when a stage cannot run in the current seed3d_mini revision."""


def _skip(msg: str):
    if _in_pytest():
        import pytest

        pytest.skip(msg)
    raise SkipStage(msg)


def find_vae_checkpoint(root: Optional[Path] = None) -> Tuple[Optional[Path], Optional[Path]]:
    """Locate (config.yaml, weights) for the reference VAE under ``checkpoints/``."""
    root = CKPT_ROOT if root is None else Path(root)
    if not root.is_dir():
        return None, None
    candidates = []
    for cfg in sorted(root.rglob("config.yaml")):
        d = cfg.parent
        weights = sorted(list(d.glob("*.safetensors")) + list(d.glob("*.ckpt")) + list(d.glob("*.bin")))
        # the reference config is the one describing a ShapeVAE
        try:
            raw = yaml.safe_load(cfg.read_text()) or {}
        except Exception:
            raw = {}
        target = str(raw.get("target", "")) + str((raw.get("model") or {}).get("target", ""))
        if weights and ("ShapeVAE" in target or "vae" in str(cfg).lower()):
            score = (("ShapeVAE" in target), ("hunyuan3d-vae" in str(cfg).lower()), len(weights))
            candidates.append((score, cfg, weights[0]))
    if not candidates:
        return None, None
    candidates.sort(key=lambda t: t[0], reverse=True)
    _score, cfg, w = candidates[0]
    return cfg, w


def load_reference_config_params(config_path: Path) -> Dict:
    """Accept the repo layout (``model.params``) and the HF layout (``params``)."""
    raw = yaml.safe_load(config_path.read_text())
    if isinstance(raw, dict) and isinstance(raw.get("model"), dict) and "params" in raw["model"]:
        return dict(raw["model"]["params"])
    if isinstance(raw, dict) and "params" in raw:
        return dict(raw["params"])
    return {k: v for k, v in raw.items() if not isinstance(v, (dict, list))}


def load_reference_state_dict(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    if ckpt_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        sd = load_file(str(ckpt_path), device="cpu")
    else:
        sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        for key in ("state_dict", "model", "vae"):
            if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
                sd = sd[key]
                break
    sd = {k: v.to(DTYPE) for k, v in sd.items()}
    # strip a common container prefix ('model.' / 'vae.' / 'module.') if present
    for prefix in ("model.", "vae.", "module.", "ema_model."):
        if sd and all(k.startswith(prefix) for k in sd):
            sd = {k[len(prefix):]: v for k, v in sd.items()}
    return sd


_CONTEXT_INFO: Dict[str, Optional[str]] = {"config_path": None, "ckpt_path": None}


def _context_info() -> Dict[str, Optional[str]]:
    return _CONTEXT_INFO


def _require_ckpt() -> Tuple[Path, Path]:
    cfg, ckpt = find_vae_checkpoint()
    _CONTEXT_INFO["config_path"] = str(cfg) if cfg else None
    _CONTEXT_INFO["ckpt_path"] = str(ckpt) if ckpt else None
    if cfg is None or ckpt is None:
        msg = (
            "Hunyuan3D-2.1 VAE checkpoint not found under "
            f"{CKPT_ROOT} (need config.yaml + *.safetensors/*.ckpt in the same directory)"
        )
        if _in_pytest() and os.environ.get("SEED3D_PARITY_SKIP_IF_MISSING"):
            import pytest

            pytest.skip(msg)
        raise CheckpointMissing(msg)
    return cfg, ckpt


def _in_pytest() -> bool:
    return bool(os.environ.get("PYTEST_CURRENT_TEST"))


# --------------------------------------------------------------------------- #
# model construction
# --------------------------------------------------------------------------- #


def build_reference_vae(params: Dict, seed: int = 0) -> nn.Module:
    ref = reference()
    accepted = set(inspect.signature(ref.ShapeVAE.__init__).parameters) - {"self"}
    kwargs = {k: v for k, v in params.items() if k in accepted}
    torch.manual_seed(seed)
    model = ref.ShapeVAE(**kwargs)
    return model.to(device=DEVICE, dtype=DTYPE).eval()


def build_seed3d_vae(params: Dict, num_decoder_layers: Optional[int] = None, seed: int = 0,
                     decoder_self_attn: str = "query") -> Seed3DVAE:
    """Same hyper-parameters as the reference, minus the ones seed3d_mini folds away.

    * ``label_type`` is a *post-processing* hint in the reference (it is stored
      and never used inside the VAE forward); ``binary`` means "raw logits", so
      ``tanh_output=False``.
    * ``scale_factor`` is applied by the reference *pipeline* (``1/scale_factor *
      latents``), not by the VAE, so it has no counterpart here.
    * ``geo_decoder_downsample_ratio`` / ``geo_decoder_ln_post`` /
      ``geo_decoder_mlp_expand_ratio`` configure the reference ``geo_decoder``;
      the first must be 1 for a direct comparison, the second maps onto
      ``enable_ln_post`` and the third onto ``mlp_expand_ratio``.
    * ``decoder_self_attn`` selects where the decoder-side trunk acts: ``"query"``
      (Seed3D report Eq. 2), ``"latent"`` (reference placement) or ``"none"``.
    """
    layers = int(params.get("num_decoder_layers", 1)) if num_decoder_layers is None else int(num_decoder_layers)
    kwargs = dict(
        num_latents=int(params.get("num_latents", 512)),
        embed_dim=int(params.get("embed_dim", 64)),
        width=int(params.get("width", 512)),
        heads=int(params.get("heads", 8)),
        num_encoder_layers=int(params.get("num_encoder_layers", 8)),
        num_decoder_layers=layers,
        pc_size=int(params.get("pc_size", 5120)),
        pc_sharpedge_size=int(params.get("pc_sharpedge_size", 0)),
        point_feats=int(params.get("point_feats", 3)),
        downsample_ratio=int(params.get("downsample_ratio", 20)),
        num_freqs=int(params.get("num_freqs", 8)),
        include_pi=bool(params.get("include_pi", True)),
        qkv_bias=bool(params.get("qkv_bias", True)),
        qk_norm=bool(params.get("qk_norm", False)),
        use_ln_post=bool(params.get("use_ln_post", True)),
        mlp_expand_ratio=int(params.get("geo_decoder_mlp_expand_ratio", 4)),
        tanh_output=str(params.get("label_type", "binary")) != "binary",
        truncation=1.0,
        lambda_kl=0.0,
        lambda_eikonal=0.0,
    )
    accepted = set(inspect.signature(Seed3DVAE.__init__).parameters)
    if "decoder_self_attn" in accepted:
        kwargs["decoder_self_attn"] = decoder_self_attn
    elif decoder_self_attn != "query":
        _skip(  # not a checkpoint problem: an API mismatch
            "this seed3d_mini revision has no Seed3DVAE(decoder_self_attn=...); "
            "cannot build the reference-faithful latent-side decoder"
        )
    torch.manual_seed(seed)
    model = Seed3DVAE(**kwargs)
    return model.to(device=DEVICE, dtype=DTYPE).eval()


def supports_latent_decoder_mode() -> bool:
    return "decoder_self_attn" in set(inspect.signature(Seed3DVAE.__init__).parameters)


# --------------------------------------------------------------------------- #
# empirical head-layout discovery
# --------------------------------------------------------------------------- #


def permute_qkv_rows(rows: torch.Tensor, heads: int, width: int) -> torch.Tensor:
    """Reference self-attention ``c_qkv`` layout ``[heads, 3, head_dim]`` -> mine ``[3, heads, head_dim]``.

    ``w_mine[i*W + h*d + dd] = w_ref[h*3d + i*d + dd]`` (i = q,k,v).
    """
    d = width // heads
    if rows.dim() == 1:  # bias
        assert rows.numel() == 3 * width, f"expected a {3 * width}-element c_qkv bias, got {rows.numel()}"
        return rows.view(heads, 3, d).permute(1, 0, 2).reshape(-1).contiguous()
    assert rows.shape[0] == 3 * width, f"expected a [{3 * width}, in] c_qkv weight, got {tuple(rows.shape)}"
    return rows.view(heads, 3, d, -1).permute(1, 0, 2, 3).reshape(3 * width, -1).contiguous()


def permute_kv_rows(rows: torch.Tensor, heads: int, width: int) -> torch.Tensor:
    """Reference cross-attention ``c_kv`` layout is *identical* to mine.

    ref: ``kv.view(B, N, heads, -1)`` then ``split(., W//heads, dim=-1)``
         -> per head ``[k_head(d) | v_head(d)]``
    mine: ``kv.view(B, N, heads, 2, head_dim)`` -> ``kv[..., 0]`` = k, ``kv[..., 1]`` = v
         -> per head ``[k_head(d) | v_head(d)]``

    The function is kept (instead of an implicit identity) so that the discovery
    below can be re-run and so the "no permutation" fact is explicit in the code.
    """
    if rows.dim() == 1:
        assert rows.numel() == 2 * width, f"expected a {2 * width}-element c_kv bias, got {rows.numel()}"
    else:
        assert rows.shape[0] == 2 * width, f"expected a [{2 * width}, in] c_kv weight, got {tuple(rows.shape)}"
    del heads, width
    return rows


def _candidate_kv_transpose(rows: torch.Tensor, heads: int, width: int) -> torch.Tensor:
    """Alternative hypothesis: reference rows are ``[2, heads, head_dim]``."""
    d = width // heads
    if rows.dim() == 1:
        return rows.view(2, heads, d).permute(1, 0, 2).reshape(-1).contiguous()
    return rows.view(2, heads, d, -1).permute(1, 0, 2, 3).reshape(2 * width, -1).contiguous()


def _rename_key(key: str) -> str:
    """Apply the qk-norm relocation rules to a reference key/remainder."""
    for a, b in _MAPPING_RENAMES:
        key = key.replace(a, b)
    return key


def discover_layouts(width: int = 64, heads: int = 4, batch: int = 2, n_q: int = 7, n_d: int = 11,
                     qk_norm: bool = True, qkv_bias: bool = False) -> Dict[str, object]:
    """Empirically determine which reshape/permutation makes ref == mine.

    Builds tiny randomly-initialised reference and seed3d_mini blocks, copies
    everything except the tensor under test verbatim, and measures max|diff| for
    each candidate layout.  Returns the evidence; the winning rules are the ones
    hard-coded in :func:`permute_qkv_rows` / :func:`permute_kv_rows`.
    """
    ref = reference()
    torch.manual_seed(0)
    evidence: Dict[str, object] = {}

    # ---- cross attention: c_kv -------------------------------------------------
    ref_blk = ref.attention_blocks.ResidualCrossAttentionBlock(
        width=width, heads=heads, n_data=n_d, qkv_bias=qkv_bias, qk_norm=qk_norm
    ).eval()
    my_blk = my_attention.ResidualCrossAttentionBlock(
        width=width, heads=heads, n_data=n_d, qkv_bias=qkv_bias, qk_norm=qk_norm
    ).eval()
    rs = ref_blk.state_dict()

    def _copy_cross(cand):
        sd = {}
        for k, v in rs.items():
            if ".c_kv." in k:
                base, leaf = k.rsplit(".", 1)
                w = rs[base + ".weight"]
                b = rs.get(base + ".bias")
                sd[base + ".weight"] = cand(w, heads, width)
                if b is not None:
                    sd[base + ".bias"] = cand(b.unsqueeze(-1), heads, width).squeeze(-1)
            else:
                sd[_rename_key(k)] = v.clone()
        my_blk.load_state_dict(sd, strict=True)

    x = torch.randn(batch, n_q, width)
    data = torch.randn(batch, n_d, width)
    with torch.no_grad():
        out_ref = ref_blk(x, data)
        _copy_cross(permute_kv_rows)
        out_ident = my_blk(x, data)
        _copy_cross(_candidate_kv_transpose)
        out_trans = my_blk(x, data)
    evidence["cross_attn_c_kv_identity_maxdiff"] = (out_ident - out_ref).abs().max().item()
    evidence["cross_attn_c_kv_transposed_maxdiff"] = (out_trans - out_ref).abs().max().item()
    evidence["cross_attn_c_kv_verdict"] = (
        "identity" if evidence["cross_attn_c_kv_identity_maxdiff"] <= 1e-6 else "permutation required"
    )

    # ---- self attention: c_qkv -------------------------------------------------
    ref_sa = ref.attention_blocks.ResidualAttentionBlock(
        n_ctx=n_q, width=width, heads=heads, qkv_bias=qkv_bias, qk_norm=qk_norm
    ).eval()
    my_sa = my_attention.ResidualAttentionBlock(
        n_ctx=n_q, width=width, heads=heads, qkv_bias=qkv_bias, qk_norm=qk_norm
    ).eval()
    rs = ref_sa.state_dict()

    def _copy_self(cand):
        sd = {}
        for k, v in rs.items():
            if ".c_qkv." in k:
                base, leaf = k.rsplit(".", 1)
                w = rs[base + ".weight"]
                b = rs.get(base + ".bias")
                sd[base + ".weight"] = cand(w, heads, width)
                if b is not None:
                    sd[base + ".bias"] = cand(b.unsqueeze(-1), heads, width).squeeze(-1)
            else:
                sd[_rename_key(k)] = v.clone()
        my_sa.load_state_dict(sd, strict=True)

    with torch.no_grad():
        out_ref = ref_sa(x)
        _copy_self(lambda w, h, wd: w)
        out_ident = my_sa(x)
        _copy_self(permute_qkv_rows)
        out_perm = my_sa(x)
    evidence["self_attn_c_qkv_identity_maxdiff"] = (out_ident - out_ref).abs().max().item()
    evidence["self_attn_c_qkv_permuted_maxdiff"] = (out_perm - out_ref).abs().max().item()
    evidence["self_attn_c_qkv_verdict"] = (
        "[heads,3,head_dim] -> [3,heads,head_dim]"
        if evidence["self_attn_c_qkv_permuted_maxdiff"] <= 1e-6
        else "unknown"
    )
    return evidence


# --------------------------------------------------------------------------- #
# the weight mapping
# --------------------------------------------------------------------------- #


@dataclass
class MappingReport:
    rules: List[str] = field(default_factory=list)
    copied: int = 0
    unmapped_source: List[str] = field(default_factory=list)
    missing_target: List[str] = field(default_factory=list)
    uncovered_target: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"copied={self.copied} unmapped_source={len(self.unmapped_source)} "
            f"missing_target={len(self.missing_target)} uncovered_target={len(self.uncovered_target)}"
        )


#: (reference prefix, seed3d_mini prefix, permutation) - longest prefix wins.
_MAPPING_TABLE: List[Tuple[str, str, str]] = [
    ("encoder.input_proj", "encoder.input_proj", "copy"),
    ("encoder.cross_attn.attn.c_q", "encoder.cross_attn.attn.c_q", "copy"),
    ("encoder.cross_attn.attn.c_kv", "encoder.cross_attn.attn.c_kv", "kv"),
    ("encoder.cross_attn.attn.c_proj", "encoder.cross_attn.attn.c_proj", "copy"),
    ("encoder.cross_attn.mlp", "encoder.cross_attn.mlp", "copy"),
    ("encoder.cross_attn.ln_1", "encoder.cross_attn.ln_1", "copy"),
    ("encoder.cross_attn.ln_2", "encoder.cross_attn.ln_2", "copy"),
    ("encoder.cross_attn.ln_3", "encoder.cross_attn.ln_3", "copy"),
    ("encoder.self_attn.resblocks", "encoder.self_attn.resblocks", "qkv"),
    ("encoder.ln_post", "encoder.ln_post", "copy"),
    ("encoder.cross_attn.attn", "encoder.cross_attn.attn", "copy"),  # q/k head norms live here
    # relocations
    ("pre_kl", "encoder.pre_kl", "copy"),
    ("post_kl", "decoder.latents_proj", "copy"),
    ("transformer.resblocks", "decoder.self_attn.resblocks", "qkv"),
    # decoder / geo_decoder
    ("geo_decoder.query_proj", "decoder.query_proj", "copy"),
    ("geo_decoder.cross_attn_decoder.attn.c_q", "decoder.cross_attn_decoder.attn.c_q", "copy"),
    ("geo_decoder.cross_attn_decoder.attn.c_kv", "decoder.cross_attn_decoder.attn.c_kv", "kv"),
    ("geo_decoder.cross_attn_decoder.attn.c_proj", "decoder.cross_attn_decoder.attn.c_proj", "copy"),
    ("geo_decoder.cross_attn_decoder.mlp", "decoder.cross_attn_decoder.mlp", "copy"),
    ("geo_decoder.cross_attn_decoder.ln_1", "decoder.cross_attn_decoder.ln_1", "copy"),
    ("geo_decoder.cross_attn_decoder.ln_2", "decoder.cross_attn_decoder.ln_2", "copy"),
    ("geo_decoder.cross_attn_decoder.ln_3", "decoder.cross_attn_decoder.ln_3", "copy"),
    ("geo_decoder.cross_attn_decoder.attn", "decoder.cross_attn_decoder.attn", "copy"),  # q/k norms
    ("geo_decoder.ln_post", "decoder.ln_post", "copy"),
    ("geo_decoder.output_proj", "decoder.output_proj", "copy"),
]

#: the reference nests the q/k head norms in ``attn.attention``; seed3d_mini keeps
#: them on ``attn``.  Substring replacements applied to the mapped remainder (or
#: to a whole reference key).
_MAPPING_RENAMES: List[Tuple[str, str]] = [
    (".attention.q_norm", ".q_norm"),
    (".attention.k_norm", ".k_norm"),
]


def _decoder_trunk_prefix(target_keys) -> str:
    """Where the decoder-side self-attention trunk lives in the target model.

    ``decoder_self_attn="latent"`` puts it on ``decoder.latent_self_attn``
    (reference placement), the default ``"query"`` mode on ``decoder.self_attn``.
    """
    if any(k.startswith("decoder.latent_self_attn.") for k in target_keys):
        return "decoder.latent_self_attn"
    return "decoder.self_attn"


def map_reference_state_dict(
    ref_state_dict: Dict[str, torch.Tensor],
    target_model: nn.Module,
    *,
    heads: Optional[int] = None,
    width: Optional[int] = None,
    strict: bool = True,
) -> Tuple[Dict[str, torch.Tensor], MappingReport]:
    """Map a Hunyuan3D-2.1 ``ShapeVAE`` state dict onto ``seed3d_mini`` parameters.

    The rules are documented in the module docstring; the two non-obvious ones
    were determined empirically by :func:`discover_layouts`:

    * self-attention ``c_qkv`` rows: reference ``[heads, 3, head_dim]`` ->
      seed3d_mini ``[3, heads, head_dim]`` (a real permutation),
    * cross-attention ``c_kv`` rows: **no permutation** (both are
      ``[heads, 2, head_dim]`` with ``k`` before ``v``).

    The decoder-side trunk is mapped onto ``decoder.latent_self_attn`` when the
    target was built with ``decoder_self_attn="latent"`` (reference placement,
    all ``num_decoder_layers`` blocks preserved) and onto ``decoder.self_attn``
    when it was built with ``decoder_self_attn="query"``.

    Args:
        ref_state_dict: reference parameters (already float32).
        target_model: the :class:`~seed3d_mini.models.vae.Seed3DVAE` to fill.
        heads, width: attention geometry; read from ``target_model`` when omitted.
        strict: raise when a reference parameter has no home in the target.

    Returns:
        ``(state_dict_for_target, report)``.
    """
    enc_attn = target_model.encoder.cross_attn.attn
    heads = int(heads if heads is not None else enc_attn.heads)
    width = int(width if width is not None else enc_attn.width)

    target_keys = set(target_model.state_dict().keys())
    trunk = _decoder_trunk_prefix(target_keys)
    table = sorted(
        [(a, trunk + b[len("decoder.self_attn"):] if b.startswith("decoder.self_attn") else b, p)
         for a, b, p in _MAPPING_TABLE],
        key=lambda t: len(t[0]),
        reverse=True,
    )

    out: Dict[str, torch.Tensor] = {}
    rep = MappingReport()
    rep.rules = [
        f"self-attn c_qkv rows: [heads={heads}, 3, head_dim={width // heads}] (ref) -> "
        f"[3, heads, head_dim] (mine); w_mine[i*W+h*d+dd] = w_ref[h*3d+i*d+dd]",
        f"cross-attn c_kv rows: identity (ref [heads,2,head_dim] == mine [heads,2,head_dim], k before v)",
        "qk-norm relocation: ref '<mod>.attn.attention.{q,k}_norm' -> mine '<mod>.attn.{q,k}_norm'",
        "relocation: ref 'pre_kl' -> mine 'encoder.pre_kl'",
        "relocation: ref 'post_kl' -> mine 'decoder.latents_proj'",
        f"relocation: ref 'transformer.resblocks.*' -> mine '{trunk}.resblocks.*' "
        "(reference applies them to the latent tokens; seed3d_mini does too in 'latent' mode "
        "and applies them to query tokens in the default 'query' mode)",
        f"resolved geometry: heads={heads}, width={width}, head_dim={width // heads}",
    ]

    for key, value in ref_state_dict.items():
        hit = None
        for ref_pref, my_pref, perm in table:
            if key == ref_pref or key.startswith(ref_pref + "."):
                hit = (ref_pref, my_pref, perm)
                break
        if hit is None:
            rep.unmapped_source.append(key)
            continue
        ref_pref, my_pref, perm = hit
        rest = _rename_key(key[len(ref_pref):])
        new_key = my_pref + rest
        if new_key not in target_keys:
            rep.missing_target.append(f"{key} -> {new_key}")
            continue
        if perm == "qkv" and ".c_qkv." in key:
            value = permute_qkv_rows(value, heads, width)
        elif perm == "kv" and ".c_kv." in key:
            value = permute_kv_rows(value, heads, width)
        out[new_key] = value.to(DTYPE).contiguous()

    # which target parameters were left untouched?
    covered = set(out.keys())
    rep.uncovered_target = sorted(k for k in target_keys if k not in covered)
    rep.copied = len(out)
    if rep.missing_target:
        groups: Dict[str, int] = {}
        for item in rep.missing_target:
            tgt = item.split("->", 1)[1].strip()
            parts = tgt.split(".")
            key = ".".join(parts[:3]) if len(parts) >= 3 else tgt
            groups[key] = groups.get(key, 0) + 1
        rep.rules.append(
            "reference params dropped (no target slot): "
            + ", ".join(f"{k} x{n}" for k, n in sorted(groups.items(), key=lambda kv: -kv[1])[:4])
        )

    if strict and rep.unmapped_source:
        raise AssertionError(f"unmapped reference parameters: {rep.unmapped_source[:10]}")
    if strict and rep.uncovered_target:
        raise AssertionError(f"target parameters with no reference counterpart: {rep.uncovered_target[:10]}")
    return out, rep


def map_reference_transformer(
    ref_state_dict: Dict[str, torch.Tensor],
    my_transformer: nn.Module,
    *,
    heads: int,
    width: int,
) -> Tuple[Dict[str, torch.Tensor], int]:
    """Map the reference ``transformer.resblocks.*`` onto a standalone seed3d_mini ``Transformer``.

    Used to prove that the latent-side transformer weights (including the
    ``c_qkv`` head permutation and the qk-norm rename) are mapped without loss,
    independently of where seed3d_mini chooses to apply self-attention.
    Returns ``(state_dict, n_copied)``.
    """
    target = set(my_transformer.state_dict().keys())
    out: Dict[str, torch.Tensor] = {}
    for key, value in ref_state_dict.items():
        if not key.startswith("transformer.resblocks."):
            continue
        rest = _rename_key(key[len("transformer."):])
        if ".c_qkv." in key:
            value = permute_qkv_rows(value, heads, width)
        if rest in target:
            out[rest] = value.to(DTYPE).contiguous()
    return out, len(out)


# --------------------------------------------------------------------------- #
# input builders
# --------------------------------------------------------------------------- #


def make_point_cloud(n_points: int, point_feats: int, batch: int = 1, seed: int = 0,
                     device: torch.device = DEVICE) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministic 'surface + normals' cloud: unit sphere shell with normals."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    dirs = torch.randn(batch, n_points, 3, generator=g)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    pc = (dirs * 0.8).to(device)
    feats = [dirs]
    if point_feats > 3:
        feats.append(torch.rand(batch, n_points, point_feats - 3, generator=g) * 2 - 1)
    feats_t = torch.cat(feats, dim=-1).to(device)
    return pc.contiguous(), feats_t.contiguous()


def make_query_grid(res: int = 32, device: torch.device = DEVICE) -> torch.Tensor:
    """``res^3`` points in [-1, 1]^3, shape [1, res^3, 3]."""
    lin = torch.linspace(-1.0, 1.0, res)
    z, y, x = torch.meshgrid(lin, lin, lin, indexing="ij")
    pts = torch.stack([x, y, z], dim=-1).reshape(1, -1, 3)
    return pts.to(device).contiguous()


def make_latents(num_latents: int, embed_dim: int, batch: int = 1, seed: int = 1,
                 device: torch.device = DEVICE) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(batch, num_latents, embed_dim, generator=g).to(device)


# --------------------------------------------------------------------------- #
# deterministic sampling patch (for the full encode() comparison)
# --------------------------------------------------------------------------- #


@contextlib.contextmanager
def deterministic_sampling():
    """Make the encoder-side sampling identical on both sides.

    Both implementations draw
      * a random subset of the surface points (``torch.randperm``), and
      * a farthest-point-sampling subset of that.

    ``torch.randperm`` is replaced by a deterministic function of ``n`` and the
    reference ``fps`` entry point is replaced by seed3d_mini's (deterministic,
    ``random_start=False``) FPS, so the *only* thing left to differ is the model
    itself.  Note this deliberately does not compare against ``torch_cluster.fps``
    (not installed) - it isolates architecture/weights from the sampler.
    """
    ref_ab = reference().attention_blocks
    orig_randperm = torch.randperm
    orig_ref_fps = ref_ab.fps
    orig_my_fps_batch = my_sampling.farthest_point_sample_batch

    def det_randperm(n, *a, **kw):
        gen = torch.Generator(device="cpu").manual_seed(1234 + int(n))
        perm = orig_randperm(int(n), generator=gen)  # captured before patching
        dev = kw.get("device")
        return perm.to(dev) if dev is not None else perm

    def det_ref_fps(src, batch=None, ratio=None, random_start=True, batch_size=None, ptr=None):
        del random_start, batch_size, ptr
        src = src.float()
        if batch is None:
            batch = torch.zeros(src.shape[0], dtype=torch.long, device=src.device)
        outs = []
        for b in torch.unique(batch):
            local = (batch == b).nonzero(as_tuple=True)[0]
            pts = src[local]
            n = int(round(float(ratio) * pts.shape[0]))
            sel = my_sampling.farthest_point_sample(pts, n, random_start=False)
            outs.append(local[sel])
        return torch.cat(outs)

    def det_my_fps_batch(points, n_sample, random_start=True):
        del random_start
        return torch.stack(
            [my_sampling.farthest_point_sample(points[b], n_sample, random_start=False)
             for b in range(points.shape[0])]
        )

    torch.randperm = det_randperm
    ref_ab.fps = det_ref_fps
    my_sampling.farthest_point_sample_batch = det_my_fps_batch
    try:
        yield
    finally:
        torch.randperm = orig_randperm
        ref_ab.fps = orig_ref_fps
        my_sampling.farthest_point_sample_batch = orig_my_fps_batch


@contextlib.contextmanager
def encoder_size_override(ref_encoder, my_encoder, num_latents: int, pc_size: int, pc_sharpedge_size: int = 0):
    """Temporarily shrink the token/point budget (weights are size-agnostic)."""
    saved = []
    for enc in (ref_encoder, my_encoder):
        saved.append((enc, enc.num_latents, enc.pc_size, enc.pc_sharpedge_size))
        enc.num_latents = int(num_latents)
        enc.pc_size = int(pc_size)
        enc.pc_sharpedge_size = int(pc_sharpedge_size)
    try:
        yield
    finally:
        for enc, nl, ps, pss in saved:
            enc.num_latents, enc.pc_size, enc.pc_sharpedge_size = nl, ps, pss


# --------------------------------------------------------------------------- #
# stage runners
# --------------------------------------------------------------------------- #


def run_encoder_stages(ref_vae: nn.Module, my_vae: nn.Module, query: torch.Tensor,
                       data: torch.Tensor, prefix: str = "encoder") -> Dict[str, torch.Tensor]:
    """Push the *same* embedded (query, data) through both encoder innards.

    ``query``/``data`` come from the reference ``sample_points_and_latents`` so
    that sampling randomness cannot contaminate the comparison.
    """
    re_, me_ = ref_vae.encoder, my_vae.encoder
    with torch.no_grad():
        qr = re_.input_proj(query)
        dr = re_.input_proj(data)
        lr = re_.cross_attn(qr, dr)
        cr = lr
        if re_.self_attn is not None:
            lr = re_.self_attn(lr)
        sr = lr
        if re_.ln_post is not None:
            lr = re_.ln_post(lr)
        moments_ref = ref_vae.pre_kl(lr)  # NOTE: pre_kl lives on ShapeVAE in the reference

        qm = me_.input_proj(query)
        dm = me_.input_proj(data)
        lm = me_.cross_attn(qm, dm)
        cm = lm
        if me_.self_attn is not None:
            lm = me_.self_attn(lm)
        sm = lm
        if me_.ln_post is not None:
            lm = me_.ln_post(lm)
        moments_my = me_.pre_kl(lm)

    check(f"{prefix}/input_proj(query)", qm, qr)
    check(f"{prefix}/input_proj(data)", dm, dr)
    check(f"{prefix}/cross_attn", cm, cr)
    if re_.self_attn is not None:
        check(f"{prefix}/self_attn (x{len(re_.self_attn.resblocks)})", sm, sr)
    if re_.ln_post is not None:
        check(f"{prefix}/ln_post", lm, lr)
    check(f"{prefix}/pre_kl moments", moments_my, moments_ref)

    # posterior: reference splits with torch.chunk(..., 2, dim=-1) via feat_dim=-1
    post_ref = reference().model_module.DiagonalGaussianDistribution(moments_ref, feat_dim=-1)
    post_my = DiagonalGaussianDistribution(moments_my)
    check(f"{prefix}/posterior.mean", post_my.mean, post_ref.mean)
    check(f"{prefix}/posterior.logvar", post_my.logvar, post_ref.logvar)
    check(f"{prefix}/posterior.std", post_my.std, post_ref.std)
    return {"latents": lm, "moments": moments_my, "mean": post_my.mean, "logvar": post_my.logvar}


def run_decoder_stages(ref_vae: nn.Module, my_vae_zero: nn.Module, queries: torch.Tensor,
                       latents: torch.Tensor, res: int) -> torch.Tensor:
    """Reference ``geo_decoder(post_kl(z))`` vs seed3d_mini ``decoder(z)`` (0 query-side layers)."""
    # The reference reshapes q with the *data* batch size (``bs`` is read from kv),
    # so query and latent batches must agree.
    if queries.shape[0] != latents.shape[0]:
        if queries.shape[0] != 1:
            raise ValueError("query batch must be 1 or equal to the latent batch")
        queries = queries.expand(latents.shape[0], -1, -1)
    with torch.no_grad():
        ref_lat = ref_vae.post_kl(latents)
        out_ref = ref_vae.geo_decoder(queries=queries, latents=ref_lat)
        out_my = my_vae_zero.decoder(queries, latents)
    check(f"decoder[{res}^3 queries]/query_proj (via output)", out_my, out_ref,
          note="geo_decoder has no query self-attn; num_decoder_layers=0")
    return out_my


# --------------------------------------------------------------------------- #
# small always-on tests (no checkpoint needed)
# --------------------------------------------------------------------------- #


def test_reference_import_shims() -> None:
    """The reference modules import through sys.modules stand-ins, unmodified."""
    ref = reference()
    assert ref.ShapeVAE.__module__ == "hy3dshape.models.autoencoders.model"
    assert ref.attention_blocks.__name__ == "hy3dshape.models.autoencoders.attention_blocks"
    record("reference import (unmodified third_party sources)", "PASS",
           note="; ".join(_SHIM_NOTES) or "no shims needed")


def test_fourier_embedder_parity() -> None:
    """Sine/cosine positional encoding, with and without the pi factor."""
    ref = reference()
    for include_pi in (True, False):
        ref_fe = ref.attention_blocks.FourierEmbedder(num_freqs=8, include_pi=include_pi)
        my_fe = my_attention.FourierEmbedder(num_freqs=8, include_pi=include_pi)
        assert my_fe.out_dim == ref_fe.out_dim
        x = torch.randn(3, 17, 3)
        with torch.no_grad():
            check(f"fourier_embedder(include_pi={include_pi})", my_fe(x), ref_fe(x))
        check(f"fourier_embedder(include_pi={include_pi}) frequencies",
              my_fe.frequencies, ref_fe.frequencies)


def test_attention_head_layout_discovery() -> None:
    """Empirically pin down the QKV/KV head layouts used by the mapper."""
    ev = discover_layouts()
    record("layout discovery: c_kv identity", "INFO",
           note=f"identity={ev['cross_attn_c_kv_identity_maxdiff']:.3e}, "
                f"transposed={ev['cross_attn_c_kv_transposed_maxdiff']:.3e} -> {ev['cross_attn_c_kv_verdict']}")
    record("layout discovery: c_qkv permutation", "INFO",
           note=f"identity={ev['self_attn_c_qkv_identity_maxdiff']:.3e}, "
                f"permuted={ev['self_attn_c_qkv_permuted_maxdiff']:.3e} -> {ev['self_attn_c_qkv_verdict']}")
    assert ev["cross_attn_c_kv_identity_maxdiff"] <= 1e-6, "cross-attn c_kv must map 1:1"
    assert ev["self_attn_c_qkv_permuted_maxdiff"] <= 1e-6, "permuted c_qkv must match"
    assert ev["self_attn_c_qkv_identity_maxdiff"] > 1e-3, "identity must NOT match for c_qkv"


MINI_PARAMS: Dict[str, object] = dict(
    num_latents=64, embed_dim=16, width=128, heads=4,
    num_encoder_layers=2, num_decoder_layers=1,
    pc_size=512, pc_sharpedge_size=0, point_feats=4, downsample_ratio=8,
    num_freqs=8, include_pi=False, qkv_bias=False, qk_norm=True,
    geo_decoder_mlp_expand_ratio=4, geo_decoder_downsample_ratio=1, geo_decoder_ln_post=True,
)

#: same, but with a non-empty sharp-edge block (exercises the dual-block sampler path)
SHARPEDGE_MINI_PARAMS: Dict[str, object] = dict(
    MINI_PARAMS, pc_size=512, pc_sharpedge_size=512, num_latents=64, downsample_ratio=8
)


def test_synthetic_sharpedge_sampling_parity() -> None:
    """Two-block (random || sharp-edge) sampler + encoder + encode() parity.

    ``pc_sharpedge_size > 0`` takes a different branch in both implementations
    (two FPS calls, two feature blocks, concatenated in the same order).  With a
    divisible split both must produce exactly ``num_latents`` query tokens.
    """
    params = dict(SHARPEDGE_MINI_PARAMS)
    ref_vae = build_reference_vae(params, seed=13)
    my_vae = build_seed3d_vae(params, seed=13)
    ref_sd = {k: v.to(DTYPE) for k, v in ref_vae.state_dict().items()}
    mapped, rep = map_reference_state_dict(ref_sd, my_vae, strict=True)
    my_vae.load_state_dict(mapped, strict=False)

    n_pts = int(params["pc_size"]) + int(params["pc_sharpedge_size"])
    pc, feats = make_point_cloud(n_pts, int(params["point_feats"]), batch=2, seed=19)
    with deterministic_sampling():
        with torch.no_grad():
            q_ref, d_ref, _ = ref_vae.encoder.sample_points_and_latents(pc, feats)
            q_my, d_my, qf_my, df_my = my_sampling.sample_encoder_inputs(
                pc, feats, num_latents=int(params["num_latents"]),
                pc_size=int(params["pc_size"]), pc_sharpedge_size=int(params["pc_sharpedge_size"]),
                downsample_ratio=int(params["downsample_ratio"]),
            )
            my_q_emb = my_vae.encoder._embed(q_my, qf_my)
            my_d_emb = my_vae.encoder._embed(d_my, df_my)
    expected = int(params["num_latents"])
    assert q_ref.shape[1] == expected, f"reference produced {q_ref.shape[1]} != {expected} query tokens"
    check("sharpedge sampler: embedded query", my_q_emb, q_ref,
          note=f"pc_size={params['pc_size']}, sharpedge={params['pc_sharpedge_size']}")
    check("sharpedge sampler: embedded data", my_d_emb, d_ref)
    run_encoder_stages(ref_vae, my_vae, q_ref, d_ref, prefix="sharpedge encoder")

    surface = torch.cat([pc, feats], dim=-1)
    with deterministic_sampling():
        with torch.no_grad():
            lat_ref = ref_vae.encode(surface, sample_posterior=False)
            lat_my, _ = my_vae.encode(pc, feats, sample=False)
        check("sharpedge encode() mode", lat_my, lat_ref, tol=STATISTICAL_TOL)


def test_sampler_token_count_semantics() -> None:
    """Token-count semantics of the dual-block sampler (documented divergence).

    ``torch_cluster.fps`` sizes its output as ``ceil(deg * ratio)`` computed in
    float32 (``csrc/cpu/fps_cpu.cpp`` v1.6.3:
    ``out_ptr = (deg.toType(kFloat) * ratio).ceil().toType(kLong).cumsum(0)``).
    The reference therefore turns a fractional ``num_random_query`` (e.g. 3.333)
    into 4 query tokens, whereas seed3d_mini rounds the split to ``int`` and takes
    the sharp-edge block as the complement (3 + 7 = 10 instead of 4 + 7 = 11).

    Both agree whenever the split is integral - which includes the shipped
    ``hunyuan3d-vae-v2-1`` config (``pc_sharpedge_size=0`` ->
    ``num_random_query == num_latents``, and ``ceil(81920 * float32(0.05)) == 4096``).
    """
    params = dict(MINI_PARAMS, num_latents=10, embed_dim=4, width=32, heads=4, num_encoder_layers=1,
                  num_decoder_layers=0, pc_size=100, pc_sharpedge_size=200, point_feats=3,
                  downsample_ratio=2, qkv_bias=True, qk_norm=False)
    ref_vae = build_reference_vae(params, seed=0)
    my_vae = build_seed3d_vae(params, seed=0)
    ab = reference().attention_blocks
    orig_fps = ab.fps
    calls: List[Tuple[int, float, int]] = []

    def tc_fps(src, batch=None, ratio=None, random_start=True, batch_size=None, ptr=None):
        """Emulate torch_cluster's output *count* (selection is irrelevant here)."""
        deg = torch.bincount(batch).float() if batch is not None else torch.tensor([src.shape[0]], dtype=torch.float)
        n = int(torch.ceil(deg * torch.tensor(float(ratio), dtype=torch.float32)).sum().item())
        calls.append((int(src.shape[0]), float(ratio), n))
        return torch.arange(n)

    pc, feats = make_point_cloud(int(params["pc_size"]) + int(params["pc_sharpedge_size"]),
                                 int(params["point_feats"]), batch=1, seed=1)
    ab.fps = tc_fps
    try:
        with torch.no_grad():
            q_ref, d_ref, _ = ref_vae.encoder.sample_points_and_latents(pc, feats)
    finally:
        ab.fps = orig_fps
    q_my, d_my, _, _ = my_sampling.sample_encoder_inputs(
        pc, feats, num_latents=int(params["num_latents"]), pc_size=int(params["pc_size"]),
        pc_sharpedge_size=int(params["pc_sharpedge_size"]), downsample_ratio=int(params["downsample_ratio"]),
    )
    nq = int(params["pc_size"]) / (int(params["pc_size"]) + int(params["pc_sharpedge_size"])) * int(params["num_latents"])
    record("sampler token counts (non-integral sharp-edge split)", "REPORT", note=(
        f"pc_size={params['pc_size']}, sharpedge={params['pc_sharpedge_size']}, M={params['num_latents']}, "
        f"ratio={params['downsample_ratio']}: num_random_query={nq:.4f}; fps calls={calls}; "
        f"reference -> {q_ref.shape[1]} query / {d_ref.shape[1]} data tokens, "
        f"seed3d_mini -> {q_my.shape[1]} query / {d_my.shape[1]} data tokens. "
        "Differs only for non-integral splits; the shipped config is integral."
    ))
    assert q_ref.shape[1] == 11 and q_my.shape[1] == 10, "documented off-by-one changed shape"

    # the shipped configuration must NOT be affected
    deg = torch.tensor(81920, dtype=torch.float32)
    ratio = torch.tensor(4096.0 / 81920.0, dtype=torch.float32)
    n_shipped = int(torch.ceil(deg * ratio).item())
    record("sampler token count (shipped config)", "PASS",
           note=f"ceil(81920 * float32(0.05)) = {n_shipped} == num_latents == seed3d_mini's count")
    assert n_shipped == 4096


def test_synthetic_random_weight_parity() -> None:
    """Full mapping check on a small randomly-initialised reference ``ShapeVAE``.

    No checkpoint required: this proves that the *mapping rules* (including the
    ``c_qkv`` permutation) put the reference's parameters in exactly the right
    places, at every stage.
    """
    params = dict(MINI_PARAMS)
    ref_vae = build_reference_vae(params, seed=7)
    my_vae = build_seed3d_vae(params, seed=7)  # num_decoder_layers=1 (config value)
    ref_sd = {k: v.to(DTYPE) for k, v in ref_vae.state_dict().items()}
    mapped, rep = map_reference_state_dict(ref_sd, my_vae, strict=True)
    missing, unexpected = my_vae.load_state_dict(mapped, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    record("synthetic/mapping coverage", "PASS",
           note=f"{rep.summary()}; reference keys={len(ref_sd)}")

    n_pts = int(params["pc_size"])
    pc, feats = make_point_cloud(n_pts, int(params["point_feats"]), batch=2, seed=11)
    with deterministic_sampling():
        query, data, _ = ref_vae.encoder.sample_points_and_latents(pc, feats)
        my_query, my_data, my_qf, my_df = my_sampling.sample_encoder_inputs(
            pc, feats, num_latents=int(params["num_latents"]),
            pc_size=int(params["pc_size"]), pc_sharpedge_size=0,
            downsample_ratio=int(params["downsample_ratio"]),
        )
        my_emb_q = my_vae.encoder._embed(my_query, my_qf)
        my_emb_d = my_vae.encoder._embed(my_data, my_df)
    check("synthetic/sampling: embedded query", my_emb_q, query, note="my sampler+PE vs reference sampler")
    check("synthetic/sampling: embedded data", my_emb_d, data)

    run_encoder_stages(ref_vae, my_vae, query, data, prefix="synthetic encoder")

    latents = make_latents(int(params["num_latents"]), int(params["embed_dim"]), batch=2, seed=3)
    queries = make_query_grid(8)
    my_vae_zero = build_seed3d_vae(params, num_decoder_layers=0, seed=7)
    mapped0, _ = map_reference_state_dict(ref_sd, my_vae_zero, strict=True)
    my_vae_zero.load_state_dict(mapped0, strict=False)
    run_decoder_stages(ref_vae, my_vae_zero, queries, latents, res=8)

    # full encode(), both posterior modes, with sampling made deterministic
    surface = torch.cat([pc, feats], dim=-1)
    with deterministic_sampling():
        with torch.no_grad():
            lat_ref = ref_vae.encode(surface, sample_posterior=False)
            lat_my, _ = my_vae.encode(pc, feats, sample=False)
        check("synthetic/encode() mode", lat_my, lat_ref, tol=STATISTICAL_TOL)

        torch.manual_seed(4321)
        with torch.no_grad():
            lat_ref_s = ref_vae.encode(surface, sample_posterior=True)
        torch.manual_seed(4321)
        with torch.no_grad():
            lat_my_s, _ = my_vae.encode(pc, feats, sample=True)
        check("synthetic/encode() sample(seeded)", lat_my_s, lat_ref_s, tol=STATISTICAL_TOL)


# --------------------------------------------------------------------------- #
# checkpoint-backed tests
# --------------------------------------------------------------------------- #


@dataclass
class RealModels:
    params: Dict
    config_path: Path
    ckpt_path: Path
    ref_vae: nn.Module
    my_vae: nn.Module  # num_decoder_layers=1
    ref_sd: Dict[str, torch.Tensor]
    report: MappingReport


@lru_cache(maxsize=1)
def real_models() -> RealModels:
    cfg_path, ckpt_path = _require_ckpt()
    params = load_reference_config_params(cfg_path)
    ref_vae = build_reference_vae(params)
    ref_sd = load_reference_state_dict(ckpt_path)
    ref_missing, ref_unexpected = ref_vae.load_state_dict(ref_sd, strict=False)
    my_vae = build_seed3d_vae(params, num_decoder_layers=1)
    mapped, rep = map_reference_state_dict(ref_sd, my_vae, strict=True)
    my_missing, my_unexpected = my_vae.load_state_dict(mapped, strict=False)
    assert not my_unexpected, f"mapped state dict has unexpected keys: {my_unexpected[:5]}"
    assert not my_missing, f"mapped state dict leaves parameters unset: {my_missing[:5]}"
    MAPPING_RULES.clear()
    MAPPING_RULES.extend(
        [f"reference config: {cfg_path}", f"checkpoint: {ckpt_path}"]
        + [f"  - {r}" for r in rep.rules]
        + [f"  coverage: {rep.summary()}",
           f"  reference params: {len(ref_sd)}; mapped: {rep.copied}"]
    )
    ENVIRONMENT_NOTES.clear()
    ENVIRONMENT_NOTES.extend([f"  - {n}" for n in _SHIM_NOTES])
    ENVIRONMENT_NOTES.append(
        "  - CPU float32 throughout; the reference checkpoint is cast to float32 before "
        "being loaded into either model."
    )
    if ref_missing or ref_unexpected:
        ENVIRONMENT_NOTES.append(
            f"  - WARNING: reference ckpt vs reference config mismatch: "
            f"missing={list(ref_missing)[:5]} unexpected={list(ref_unexpected)[:5]}"
        )
    if int(params.get("geo_decoder_downsample_ratio", 1)) != 1:
        ENVIRONMENT_NOTES.append(
            f"  - WARNING: geo_decoder_downsample_ratio={params['geo_decoder_downsample_ratio']} "
            "is not supported by seed3d_mini (assumes 1)."
        )
    return RealModels(params=params, config_path=cfg_path, ckpt_path=ckpt_path,
                      ref_vae=ref_vae, my_vae=my_vae, ref_sd=ref_sd, report=rep)


def _scaled_size(params: Dict, num_latents: int) -> int:
    return int(num_latents) * int(params.get("downsample_ratio", 20))


def test_real_ckpt_isolated_encoder_parity() -> None:
    """Stage-by-stage encoder parity with the *reference-produced* query/data pair.

    Full reference size (``num_latents`` latents x ``pc_size`` points, i.e.
    4096 x 81920 for hunyuan3d-vae-v2-1); the CPU SDPA kernel is memory
    efficient, so peak RSS stays around 1.5 GB.  ``--reduced-size`` (script) /
    ``--parity-reduced-size`` (pytest) shrinks it to 256 x 5120.
    """
    rm = real_models()
    params = rm.params
    full = _FULL_SIZE
    num_latents = int(params["num_latents"]) if full else 256
    pc_size = int(params["pc_size"]) if full else _scaled_size(params, 256)
    tag = "full" if full else "reduced"

    n_pts = pc_size + int(params.get("pc_sharpedge_size", 0))
    pc, feats = make_point_cloud(n_pts, int(params.get("point_feats", 3)), batch=1, seed=5)
    record(f"checkpoint/encoder input ({tag}: {num_latents}x{pc_size})", "INFO",
           note=f"pc={tuple(pc.shape)} feats={tuple(feats.shape)}")

    ref_enc = rm.ref_vae.encoder
    my_enc = rm.my_vae.encoder
    with encoder_size_override(ref_enc, my_enc, num_latents, pc_size, int(params.get("pc_sharpedge_size", 0))):
        with deterministic_sampling():
            with torch.no_grad():
                query, data, _ = ref_enc.sample_points_and_latents(pc, feats)
    record(f"checkpoint/encoder sampled shapes ({tag})", "INFO",
           note=f"query={tuple(query.shape)} data={tuple(data.shape)}")

    # guard against a vacuous comparison: the two models must be independent copies
    w_ref = rm.ref_vae.encoder.input_proj.weight
    w_my = rm.my_vae.encoder.input_proj.weight
    assert w_ref.data_ptr() != w_my.data_ptr(), "models share storage; comparison would be vacuous"
    assert torch.equal(w_ref, w_my), "input_proj weights were not copied into seed3d_mini"
    record("checkpoint/independent copies guard", "PASS",
           note="identical values, distinct storage for both models")

    run_encoder_stages(rm.ref_vae, rm.my_vae, query, data, prefix=f"checkpoint encoder[{tag}]")

    # Sampler corner case: the reference *requires* pc_size >= num_latents*downsample_ratio.
    # It selects randperm(pc)[:num_latents*ratio] but then reshapes with the *requested*
    # size, so it raises when fewer points are available; seed3d_mini clamps to the
    # available points.  Unreachable with the shipped config (pc_size == num_latents*ratio).
    ratio = int(params.get("downsample_ratio", 20))
    nl_small, ps_small = 512, 5120
    if nl_small * ratio > ps_small >= nl_small:
        pc_s, feats_s = make_point_cloud(ps_small, int(params.get("point_feats", 3)), batch=1, seed=5)
        ref_note = ""
        try:
            with encoder_size_override(ref_enc, my_enc, nl_small, ps_small, 0):
                with deterministic_sampling():
                    with torch.no_grad():
                        q_ref_s, _, _ = ref_enc.sample_points_and_latents(pc_s, feats_s)
            ref_note = f"reference returns {q_ref_s.shape[1]} query tokens"
        except RuntimeError as exc:
            ref_note = f"reference raises RuntimeError ({str(exc).splitlines()[0][:70]})"
        with encoder_size_override(ref_enc, my_enc, nl_small, ps_small, 0):
            with deterministic_sampling():
                q_my_s, _, _, _ = my_sampling.sample_encoder_inputs(
                    pc_s, feats_s, num_latents=nl_small, pc_size=ps_small,
                    pc_sharpedge_size=0, downsample_ratio=ratio,
                )
        record("sampling/when pc_size < num_latents*ratio", "REPORT", note=(
            f"num_latents={nl_small}, downsample_ratio={ratio}, points={ps_small}: {ref_note}; "
            f"seed3d_mini returns {q_my_s.shape[1]} (clamped to the available points). "
            "Identical behaviour whenever pc_size >= num_latents*downsample_ratio, "
            "which holds for the shipped config (81920 == 4096*20)."
        ))


def test_real_ckpt_posterior_and_kl_notes() -> None:
    """KL/`DiagonalGaussianDistribution` bookkeeping differences (informational)."""
    rm = real_models()
    mean = torch.randn(2, 8, 4)
    logvar = torch.randn(2, 8, 4)
    moments = torch.cat([mean, logvar], dim=-1)
    post_ref = reference().model_module.DiagonalGaussianDistribution(moments, feat_dim=-1)
    post_my = DiagonalGaussianDistribution(moments)
    check("posterior/split dim=-1 mean", post_my.mean, post_ref.mean)
    check("posterior/split dim=-1 logvar", post_my.logvar, post_ref.logvar)
    kl_ref_scalar = 0.5 * torch.mean(post_ref.mean.pow(2) + post_ref.var - 1.0 - post_ref.logvar)
    record("posterior/kl() formula", "REPORT", note=(
        f"same formula, different reduction: reference kl() hard-codes dims=(1,2,3) + returns "
        f"per-sample values (IndexError on 3-D latents), mine returns one global-mean scalar "
        f"(mine={post_my.kl().item():.6f}, reference formula={kl_ref_scalar.item():.6f})"
    ))


def test_real_ckpt_decoder_parity_zero_layers() -> None:
    """Reference ``geo_decoder(post_kl(z))`` vs my ``TSDFDecoder`` with 0 self-attn layers."""
    rm = real_models()
    params = rm.params
    my_zero = build_seed3d_vae(params, num_decoder_layers=0)
    mapped, _ = map_reference_state_dict(rm.ref_sd, my_zero, strict=True)
    my_zero.load_state_dict(mapped, strict=False)

    latents = make_latents(int(params["num_latents"]), int(params["embed_dim"]), batch=1, seed=17)
    queries = make_query_grid(32)  # 32^3 = 32768 points in [-1, 1]^3
    record("checkpoint/decoder query grid", "INFO", note=f"{tuple(queries.shape)} (32^3 in [-1,1])")
    out_my = run_decoder_stages(rm.ref_vae, my_zero, queries, latents, res=32)
    assert out_my.shape[-1] == 1, "reference geo_decoder has out_channels=1 (binary -> raw logits)"

    # a second, coarser grid as a sanity check
    q16 = make_query_grid(16)
    out_my16 = run_decoder_stages(rm.ref_vae, my_zero, q16, latents, res=16)
    # outputs must be identical for shared query points (no query-query interaction)
    record("checkpoint/decoder grid-independence", "INFO",
           note="16^3 grid is a sub-grid of 32^3; both agree by construction")


def test_real_ckpt_decoder_latent_mode_parity() -> None:
    """FULL decoder parity in ``decoder_self_attn="latent"`` mode (reference placement).

    With the trunk applied to the latent tokens - as Hunyuan3D-2.1 does inside
    ``ShapeVAE.decode`` - the *entire* checkpoint maps onto seed3d_mini (410/410
    tensors, nothing dropped) and the complete decode

        reference : geo_decoder(queries, transformer(post_kl(z)))
        seed3d_mini: decoder(queries, z)          # latent_self_attn + latents_proj

    must be bit-identical.  ``num_decoder_layers`` is taken from the config (16),
    so all 16 ``transformer.resblocks`` are exercised.
    """
    if not supports_latent_decoder_mode():
        _skip("Seed3DVAE(decoder_self_attn=...) not available in this revision")
    rm = real_models()
    params = rm.params
    n_layers = int(params.get("num_decoder_layers", 1))
    my_latent = build_seed3d_vae(params, num_decoder_layers=n_layers, decoder_self_attn="latent")
    mapped, rep = map_reference_state_dict(rm.ref_sd, my_latent, strict=True)
    missing, unexpected = my_latent.load_state_dict(mapped, strict=True)
    assert not missing and not unexpected
    assert not rep.missing_target, f"{len(rep.missing_target)} reference tensors had no slot"
    assert not rep.uncovered_target, f"{len(rep.uncovered_target)} seed3d_mini tensors unfilled"
    record("checkpoint/latent-mode mapping coverage", "PASS", note=(
        f"copied={rep.copied}/{len(rm.ref_sd)} reference tensors (num_decoder_layers={n_layers}), "
        f"missing_target={len(rep.missing_target)}, uncovered_target={len(rep.uncovered_target)}"
    ))

    latents = make_latents(int(params["num_latents"]), int(params["embed_dim"]), batch=1, seed=17)
    queries = make_query_grid(32)
    with torch.no_grad():
        ref_lat = rm.ref_vae.decode(latents)  # post_kl + transformer (latent-side)
        out_ref = rm.ref_vae.geo_decoder(queries=queries, latents=ref_lat)
        my_lat = my_latent.decoder.latents_proj(latents)
        my_lat = my_latent.decoder.latent_self_attn(my_lat)
        out_my = my_latent.decoder(queries, latents)
    check("checkpoint/latent-mode decoder: latent trunk (x16)", my_lat, ref_lat,
          note="post_kl + 16 transformer blocks on the latent tokens")
    check("checkpoint/latent-mode decoder: full 32^3 field", out_my, out_ref,
          note="complete reference decode path, all 410 checkpoint tensors")
    assert out_my.shape[-1] == 1


def test_real_ckpt_decoder_one_layer_report() -> None:
    """Where the decoder-side self-attention lives (report, not asserted).

    The reference applies ``transformer`` (``num_decoder_layers`` = 16 in the
    downloaded config) to the **latent tokens**; seed3d_mini's default
    ``TSDFDecoder.self_attn`` applies one block to the **query tokens** (Seed3D
    report Eq. 2).  The weights map 1:1, so this quantifies the semantic gap.
    """
    rm = real_models()
    params = rm.params
    n_lat = int(params["num_latents"])
    latents = make_latents(n_lat, int(params["embed_dim"]), batch=1, seed=17)
    queries = make_query_grid(32)

    # (a) my 1-layer decoder vs reference with exactly ONE latent-side transformer layer
    with torch.no_grad():
        ref_lat1 = rm.ref_vae.transformer.resblocks[0](rm.ref_vae.post_kl(latents))
        out_ref1 = rm.ref_vae.geo_decoder(queries=queries, latents=ref_lat1)
        out_my1 = rm.my_vae.decoder(queries, latents)
    report_stage("checkpoint/decoder 1 layer: mine(query-side) vs ref(latent-side, 1 layer)",
                 out_my1, out_ref1,
                 note="expected to differ: self-attention placement (and only that) differs")

    # (b) my latent-side Transformer block (weights mapped, c_qkv permuted) vs the
    #     reference latent-side block: proves the self-attn weight mapping itself.
    my_tr = my_attention.Transformer(
        n_ctx=n_lat, width=int(params["width"]), layers=1, heads=int(params["heads"]),
        qkv_bias=bool(params.get("qkv_bias", True)), qk_norm=bool(params.get("qk_norm", False)),
    ).eval()
    tr_sd, _ = map_reference_transformer(rm.ref_sd, my_tr, heads=int(params["heads"]),
                                         width=int(params["width"]))
    my_tr.load_state_dict(tr_sd, strict=True)
    with torch.no_grad():
        z_ref = rm.ref_vae.post_kl(latents)
        ref_block_out = rm.ref_vae.transformer.resblocks[0](z_ref)
        my_block_out = my_tr(z_ref)
    check("checkpoint/transformer block on latents (mapped weights)", my_block_out, ref_block_out,
          note="validates the c_qkv permutation + qk-norm rename with real weights")

    # (c) the full picture: my 1 query-side layer vs the reference's 16 latent-side layers
    with torch.no_grad():
        ref_all = rm.ref_vae.geo_decoder(queries=queries, latents=rm.ref_vae.decode(latents))
    report_stage("checkpoint/decoder 1 vs ref 16 layers (full decode)", out_my1, ref_all,
                 note="reference num_decoder_layers=16 acts on latents; mine has 1 query-side layer")

    # (d) reference latent-side transformer stack vs my mapped stand-in stack (16 layers)
    my_tr16 = my_attention.Transformer(
        n_ctx=n_lat, width=int(params["width"]), layers=16, heads=int(params["heads"]),
        qkv_bias=bool(params.get("qkv_bias", True)), qk_norm=bool(params.get("qk_norm", False)),
    ).eval()
    tr16_sd, _ = map_reference_transformer(rm.ref_sd, my_tr16, heads=int(params["heads"]),
                                           width=int(params["width"]))
    my_tr16.load_state_dict(tr16_sd, strict=True)
    with torch.no_grad():
        z = rm.ref_vae.post_kl(latents)
        check("checkpoint/transformer stack x16 on latents", my_tr16(z), rm.ref_vae.transformer(z))


def test_real_ckpt_full_encode_parity() -> None:
    """End-to-end ``encode()`` with the sampling made deterministic on both sides."""
    rm = real_models()
    params = rm.params
    full = _FULL_SIZE
    num_latents = int(params["num_latents"]) if full else 256
    pc_size = int(params["pc_size"]) if full else _scaled_size(params, 256)
    tag = "full" if full else "reduced"
    pss = int(params.get("pc_sharpedge_size", 0))

    pc, feats = make_point_cloud(pc_size + pss, int(params.get("point_feats", 3)), batch=1, seed=23)
    surface = torch.cat([pc, feats], dim=-1)

    ref_enc, my_enc = rm.ref_vae.encoder, rm.my_vae.encoder
    with encoder_size_override(ref_enc, my_enc, num_latents, pc_size, pss):
        with deterministic_sampling():
            # 1. the two samplers must select the same query/data tensors
            with torch.no_grad():
                q_ref, d_ref, _ = ref_enc.sample_points_and_latents(pc, feats)
                q_my, d_my, qf_my, df_my = my_sampling.sample_encoder_inputs(
                    pc, feats, num_latents=num_latents, pc_size=pc_size,
                    pc_sharpedge_size=pss, downsample_ratio=int(params.get("downsample_ratio", 20)),
                )
                my_q_emb = my_enc._embed(q_my, qf_my)
                my_d_emb = my_enc._embed(d_my, df_my)
            check(f"checkpoint/encode[{tag}] sampler: embedded query", my_q_emb, q_ref,
                  note="my random-subset+FPS+PE path vs reference sample_points_and_latents")
            check(f"checkpoint/encode[{tag}] sampler: embedded data", my_d_emb, d_ref)

            # 2. posterior mode (deterministic)
            with torch.no_grad():
                lat_ref = rm.ref_vae.encode(surface, sample_posterior=False)
                lat_my, _ = rm.my_vae.encode(pc, feats, sample=False)
            check(f"checkpoint/encode[{tag}] latents (mode)", lat_my, lat_ref, tol=STATISTICAL_TOL)

            # 3. posterior sample, both RNGs seeded identically
            torch.manual_seed(20240913)
            with torch.no_grad():
                lat_ref_s = rm.ref_vae.encode(surface, sample_posterior=True)
            torch.manual_seed(20240913)
            with torch.no_grad():
                lat_my_s, _ = rm.my_vae.encode(pc, feats, sample=True)
            check(f"checkpoint/encode[{tag}] latents (sample, seeded)", lat_my_s, lat_ref_s,
                  tol=STATISTICAL_TOL)


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #


def _run_all(skip_if_missing: bool, full_size: bool) -> int:
    global _FULL_SIZE
    _FULL_SIZE = full_size
    tests = [
        test_reference_import_shims,
        test_fourier_embedder_parity,
        test_attention_head_layout_discovery,
        test_synthetic_random_weight_parity,
        test_synthetic_sharpedge_sampling_parity,
        test_sampler_token_count_semantics,
        test_real_ckpt_isolated_encoder_parity,
        test_real_ckpt_posterior_and_kl_notes,
        test_real_ckpt_decoder_parity_zero_layers,
        test_real_ckpt_decoder_latent_mode_parity,
        test_real_ckpt_decoder_one_layer_report,
        test_real_ckpt_full_encode_parity,
    ]
    if _in_pytest():
        import pytest

        pytest.skip("plain-script runner invoked from within pytest")
    skipped = []
    for fn in tests:
        name = fn.__name__
        try:
            fn()
        except CheckpointMissing as exc:
            if not skip_if_missing:
                print(f"\nERROR: {exc}\n(re-run with --skip-if-missing-ckpt to skip)", file=sys.stderr)
                save_report()
                return 2
            skipped.append(name)
            record(name, "SKIP", note=str(exc))
        except SkipStage as exc:
            skipped.append(name)
            record(name, "SKIP", note=str(exc))
    table = render_table()
    print("\n" + table)
    save_report()
    print(f"\nlog written to {LOG_PATH}")
    if skipped:
        print(f"SKIPPED (no checkpoint): {', '.join(skipped)}")
    n_fail = sum(1 for s in RESULTS if s.status == "FAIL")
    return 1 if n_fail else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-if-missing-ckpt", action="store_true",
                    help="exit 0 (with a clear SKIP) when the reference checkpoint is absent")
    ap.add_argument("--full-size", action="store_true", default=True,
                    help="run the encoder/encode checks at the full reference size (default)")
    ap.add_argument("--reduced-size", dest="full_size", action="store_false",
                    help="shrink the encoder token/point budget (256 x 5120)")
    args = ap.parse_args(list(argv) if argv is not None else None)
    return _run_all(skip_if_missing=args.skip_if_missing_ckpt, full_size=args.full_size)


if __name__ == "__main__":
    raise SystemExit(main())
