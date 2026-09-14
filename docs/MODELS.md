# Model checkpoints

Everything the Seed3D-1.0 mini reproduction needs on disk, where it lives, what it
costs, and what its license allows.  All weights are downloaded from the
**hf-mirror.com** mirror (huggingface.co is not directly reachable from this
machine) into `checkpoints/` by `scripts/download_models.py`.

## Quick start

```bash
# makes HF_ENDPOINT / HF_HOME / SEED3D_PYTHON available to any script
source scripts/env_hf.sh

# download everything (RADIO is optional, --skip-optional drops it)
"$SEED3D_PYTHON" scripts/download_models.py --out "$SEED3D_CKPT_DIR" --verify

# subsets / other flags
"$SEED3D_PYTHON" scripts/download_models.py --models dinov2-base
"$SEED3D_PYTHON" scripts/download_models.py --skip-optional
"$SEED3D_PYTHON" scripts/download_models.py --dry-run      # plan + remote layout, no download
```

`scripts/env_hf.sh` exports:

| variable | value | why |
|---|---|---|
| `HF_ENDPOINT` | `https://hf-mirror.com` (unless already set) | huggingface.co is blocked; the mirror is not. Affects both `huggingface_hub` **and** `transformers`. |
| `HF_HOME` | `checkpoints/.hf_home` | keeps the hub cache inside the repo so the tree is self-contained |
| `HF_HUB_DISABLE_XET` | `1` | the mirror does not serve the Xet transfer path |
| `HF_HUB_DOWNLOAD_TIMEOUT` | `60` | fail a half-open connection faster than the 10-minute default |
| `SEED3D_CKPT_DIR` | `<repo>/checkpoints` | default `--out` for the downloader |
| `SEED3D_PYTHON` | `envs/seed3d/bin/python` | the prepared interpreter |

Re-running the downloader is always safe: it is idempotent and resumes partial
files (hf_hub keeps `<local_dir>/.cache/huggingface/download/*.incomplete`).

## What was downloaded

| model | role in this repo | exact local path | size (bytes) | license |
|---|---|---|---|---|
| **Hunyuan3D-2.1 shape VAE** | numerical-parity reference for `src/seed3d_mini/models/vae.py` (VecSet encoder/decoder, TSDF) | `checkpoints/hunyuan3d-vae-v2-1/model.fp16.ckpt` | 655,648,152 | Tencent Hunyuan 3D 2.1 Community License — **non-commercial-style, restrictive** (see below) |
| ↳ its architecture config | layer widths / latent count of the reference VAE | `checkpoints/hunyuan3d-vae-v2-1/config.yaml` | 409 | same |
| **DINOv2-small** | 2nd image encoder of the dual-encoder conditioner (`encoder_type: dinov2_pair`) | `checkpoints/dinov2-small/model.safetensors` (+ `config.json`, `preprocessor_config.json`) | 88,249,960 | Apache-2.0 |
| **DINOv2-base** | main semantic image encoder of the conditioner | `checkpoints/dinov2-base/model.safetensors` (+ `config.json`, `preprocessor_config.json`) | 346,345,912 | Apache-2.0 |
| **C-RADIOv3-B** *(optional)* | geometry-aware encoder for the `radio` conditioner variant; unused while unavailable | `checkpoints/C-RADIOv3-B/model.safetensors` (HF format) and `checkpoints/C-RADIOv3-B/c-radio_v3-b_half.pth.tar` (RADIO native), + remote code / configs | 393,038,784 + 546,632,054 | NVIDIA Open Model License |

Total on disk: **1.89 GiB** (2,030,138,032 bytes).
Machine-readable copy of this table: `checkpoints/download_manifest.json`
(per-file verification results included; `--verify` refreshes it).

Downloads are limited on purpose: the `hunyuan3d-dit-v2-1` (7.4 GB),
`hunyuan3d-paintpbr-v2-1` (~5.5 GB) and `hy3dpaint` texture/PBR files are
**skipped** — this reproduction owns only the shape VAE + image encoders.
The DINOv2 repos also ship a redundant `pytorch_model.bin`, which is skipped in
favour of `model.safetensors`.

### Repo-layout note

`snapshot_download` mirrors the upstream layout, so the VAE arrives at
`checkpoints/hunyuan3d-vae-v2-1/hunyuan3d-vae-v2-1/…`.  The downloader therefore
adds a **hardlink** (same inode, no extra disk space) at the canonical short
path, `checkpoints/hunyuan3d-vae-v2-1/model.fp16.ckpt`; both paths are valid and
identical.  The nested copy is the one hf_hub resumes into — do not delete it.

## Verification performed (2026-09-13)

Every artifact was re-opened after download, not just stat-ed:

| artifact | check | result |
|---|---|---|
| `hunyuan3d-vae-v2-1/config.yaml` | YAML parse | ✅ top-level keys `['target', 'params']` |
| `hunyuan3d-vae-v2-1/model.fp16.ckpt` | `torch.load(weights_only=True)` | ✅ **410 tensors**, 327,746,177 params, all `float16`, 625.1 MiB of tensor data |
| `dinov2-small/model.safetensors` | `safetensors.safe_open` | ✅ 223 tensors |
| `dinov2-small` end-to-end | `AutoModel.from_pretrained` + forward at 518×518 | ✅ `last_hidden_state = (1, 1370, 384)`, patch 14 |
| `dinov2-base/model.safetensors` | `safetensors.safe_open` | ✅ 223 tensors |
| `dinov2-base` end-to-end | `AutoModel.from_pretrained` + forward at 518×518 | ✅ `last_hidden_state = (1, 1370, 768)`, patch 14 |
| `C-RADIOv3-B/model.safetensors` | `safetensors.safe_open` | ✅ 175 tensors |
| `C-RADIOv3-B/*.pth.tar` | zip archive integrity | ✅ intact (327 entries); `weights_only=True` refuses it because the training checkpoint pickles an `argparse.Namespace` — expected, not corruption |

The `1370 = 1 + 1369 = 1 + 37²` token count is exactly the `text_len = 1370`
condition length that `src/seed3d_mini/models/conditioner.py` documents for a
518×518 input, so the local DINOv2 copy is directly usable by the conditioner
without any further conversion.  Both models were loaded with
`HF_HUB_OFFLINE=1`, i.e. purely from `checkpoints/`.

sha256 (for provenance / cache auditing):

```
5cbe97f25e6e7abd4bccc80ab07524ec0c86d24118486a9ba49bb5dfb070288a  hunyuan3d-vae-v2-1/model.fp16.ckpt
ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1  dinov2-small/model.safetensors
d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841  dinov2-base/model.safetensors
27881c65f10ab2a16d4b2bda2d6f52ddc8410b0980837fc96914e72a7d190b9b  C-RADIOv3-B/model.safetensors
530b0e827b1d409ca3b175466e11062885baedd823ec5dae3a88348e76d4c398  C-RADIOv3-B/c-radio_v3-b_half.pth.tar
```

Upstream revision pinned for the VAE: `tencent/Hunyuan3D-2.1@0b94677654c57bb9a6b6845cd7b704ccf551d327`.

## Hunyuan3D-2.1 VAE structure (from the downloaded checkpoint)

Full key/shape listing: **`docs/hunyuan_vae_keys.txt`** (410 rows).
`config.yaml` verbatim:

```yaml
target: hy3dshape.models.ShapeVAE
params:
  num_latents: 4096      # VecSet length M
  embed_dim: 64
  num_freqs: 8           # Fourier positional encoding bands
  include_pi: false
  heads: 16
  width: 1024
  num_encoder_layers: 8
  num_decoder_layers: 16
  qkv_bias: false
  qk_norm: true
  scale_factor: 1.0039506158752403
  geo_decoder_mlp_expand_ratio: 4
  geo_decoder_downsample_ratio: 1
  geo_decoder_ln_post: true
  point_feats: 4         # xyz + normal
  pc_size: 81920
  pc_sharpedge_size: 0
```

Parameter distribution by top-level prefix:

| prefix | tensors | params | meaning |
|---|---|---|---|
| `encoder` | 142 | 113,402,112 | cross-attention encoder over point clouds (input proj is `(1024, 55)` = 3+1 feats × (2·8 freqs+1)) |
| `transformer` | 240 | 201,494,528 | 16-layer self-attention decoder |
| `geo_decoder` | 24 | 12,651,777 | per-query coordinate/SDF MLP head |
| `pre_kl` / `post_kl` | 2 / 2 | 131,200 / 66,560 | latent projection to/from `embed_dim=64` |

The checkpoint stores raw module keys (no `model.`/`state_dict.` prefix after
unwrapping), and the upstream loader is
`hy3dshape/models/autoencoders/model.py :: ShapeVAE.from_pretrained(variant='fp16',
use_safetensors=False, subfolder='hunyuan3d-vae-v2-1')` →
`torch.load(...)['state_dict']` → `load_state_dict(strict=False)`.

## Encoder notes

* **DINOv2** — downloaded as HF-format `safetensors`, so
  `AutoModel.from_pretrained("checkpoints/dinov2-base")` works as-is.  Use the
  **local path**, not `facebook/dinov2-base`: offline/blob-cache behaviour then
  stops depending on the mirror.
* **C-RADIOv3-B** — **loads and runs** now that `timm` is installed in
  `envs/seed3d` (`uv pip install --python envs/seed3d/bin/python timm`, 1.0.29).
  Verified: 175 tensors load via
  `AutoModel.from_pretrained("checkpoints/C-RADIOv3-B", trust_remote_code=True)`
  and a 512² forward pass returns `RadioOutput(summary=[B,2304],
  features=[B,1024,768])`.  Two runtime facts the code depends on:
  * RADIO rejects input resolutions that are not a multiple of its resolution
    step (518² fails, 512² works), so `configs/dit_mini.yaml` uses
    `image_size: 512` while `conditioner.py` pads whatever it is given up to a
    patch multiple.  DINOv2 then emits 37x37 patches and RADIO 32x32 from the
    same image, which `PixelShuffleAlign` reconciles (bilinear in this
    non-integer case).
  * the RADIO wrapper's forward takes `x`, not `pixel_values`;
    `HFImageBackbone._call_backbone` resolves the keyword once from the
    signature, so both styles work.
  `encoder_type: dinov2_pair` (DINOv2-base + DINOv2-small) remains available for
  environments without `timm`; both paths were run successfully here
  (conditioned sequence: 1371 tokens for the DINOv2 pair, 1373 with RADIO).

## Licenses

* **Hunyuan3D-2.1 weights — Tencent Hunyuan 3D 2.1 Community License
  (restrictive; non-commercial-style, not an OSI open-source license).**
  The license is territory-limited (excludes the EU, UK and South Korea),
  requires a separate grant above 1M monthly active users, and — most relevant
  here — its §5(b) forbids using the works, outputs or results **to improve any
  other AI model**.  In this repository the VAE checkpoint is used **only for a
  numerical parity check** of our from-scratch re-implementation
  (`src/seed3d_mini/models/vae.py`): it is never trained, fine-tuned, distilled
  from, or redistributed, and its weights are not derived into the mini model.
  Keep it that way — do not train on checkpoints or outputs produced from these
  weights, and do not commit the `.ckpt` (655 MB) to git.
* **DINOv2-small / DINOv2-base** — Apache-2.0 (Meta); redistribution is allowed
  with attribution.
* **C-RADIOv3-B** — NVIDIA Open Model License (research/commercial use governed
  by that agreement); not needed for the default configuration.

## Troubleshooting

* **A download hangs with no output.**  The mirror serves large blobs through a
  CDN; occasionally a connection sticks in `SYN-SENT` and never times out (this
  actually happened to `dinov2-small`).  `scripts/download_models.py` therefore
  runs a byte-progress **stall watchdog** (`--stall-timeout`, default 150 s):
  it prints `!! STALL …` and exits with code **3**, keeping partial data.
  Just re-run the same command — it resumes.
* **Exit codes**: `0` all required models present · `1` a required model failed
  (see the summary) · `3` aborted by the watchdog, re-run · `130` Ctrl-C.
* **Optional failures never fail the run**: RADIO problems are reported in the
  summary table and the manifest but exit code stays 0.
* **`--verify`** re-opens configs (`yaml`/`json`), safetensors
  (`safe_open`) and `.ckpt`/`.bin` (`torch.load(weights_only=True)`); add
  `--no-deep-verify` to only stat weight files.  The VAE `.ckpt` takes ~2 s per
  open and the flat-view hardlink is detected and skipped instead of re-read.
