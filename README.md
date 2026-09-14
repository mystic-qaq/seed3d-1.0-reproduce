# Seed3D 1.0 — shape-generation reproduction (mini, trainable on 4×RTX3090)

A from-scratch reimplementation of the **shape/geometry stage** of
[Seed3D 1.0](https://arxiv.org/abs/2510.19944) (*From Images to
High-Fidelity Simulation-Ready 3D Assets*), plus a **mini training recipe**
(single-image → 3D, three categories: airplane / car / ship) that fits in a
~24 h budget on 4×RTX3090.

The architecture follows the report's Sec. 2.1: a **vector-set VAE**
(`Seed3D-VAE`) that encodes a surface point set into a latent token set `Z` and
decodes a **TSDF** field, and a **rectified-flow DiT** (`Seed3D-DiT`) that
generates `Z` conditioned on a reference image through a **dual (DINOv2 + RADIO)
encoder**. The same code also runs unmodified at the report's scale (4096
tokens, `L_k` decoder layers, FLUX-style backbone).

```
reference image ─► Seed3D-DiT (rectified flow, CFG, 40 steps)
                        │  latent token set Z ∈ R^{M×d}
                        ▼
                  Seed3D-VAE decoder ─► TSDF field ─► marching cubes ─► mesh
```

## What is in the box

| Path | What it is |
|---|---|
| `src/seed3d_mini/models/vae.py` | Seed3D-VAE: point-set encoder, TSDF decoder, KL warm-up, multi-scale tokens, optional eikonal loss |
| `src/seed3d_mini/models/dit.py` | Seed3D-DiT: FLUX-style double/single-stream backbone, velocity prediction |
| `src/seed3d_mini/models/conditioner.py` | Dual image encoder (DINOv2 ‖ RADIO, channel-wise fusion) + CFG dropout |
| `src/seed3d_mini/flow_matching.py` | Rectified flow: logit-normal timesteps, length-aware shift, Euler/Heun samplers |
| `src/seed3d_mini/mesh_ops.py` | Watertight remeshing, surface/sharp-edge sampling, TSDF queries, parity-vote inside test |
| `src/seed3d_mini/volume_decode.py` | Dense and coarse-to-fine (hierarchical) volume decoding + marching cubes |
| `src/seed3d_mini/render/rasterizer.py` | PyTorch z-buffer renderer for reference views (no Blender needed) |
| `scripts/prepare_data.py` | Data pipeline: download → normalise → remesh → sample → render → shards |
| `scripts/train_vae.py`, `scripts/train_dit.py` | torchrun/DDP training (AMP, grad-accum, cosine+warmup, resume, val) |
| `scripts/pack_latents.py` | Freezes the VAE and packs DiT training latents |
| `scripts/infer.py` | Image → mesh demo |
| `scripts/evaluate.py` | Chamfer-L1/L2, F-score@1%/2%, volumetric IoU, normal consistency |
| `scripts/run_mini.sh` | One-command 4-GPU mini training (data → VAE → latents → DiT) |
| `tests/test_vae_parity.py` | **Numerical parity** against the released Hunyuan3D-2.1 VAE |
| `tests/smoke_pipeline.py` | End-to-end smoke test of every stage (CPU or one GPU) |
| `docs/REPRODUCTION_NOTES.md` | Report → code mapping, ambiguities and how they were resolved, measurements |

## Quickstart

```bash
# 0. environment (torch 2.5.1+cu121, python 3.11)
envs/seed3d/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 1. verify the implementation without any data (fast, CPU or one GPU)
python tests/smoke_pipeline.py
python tests/test_vae_parity.py          # needs checkpoints/ (see docs/MODELS.md)

# 2. mini training on 4 GPUs (~12-20 h total, resumable)
bash scripts/run_mini.sh --gpus 0,1,2,3 --vae-steps 20000 --dit-steps 20000

# 3. sample a shape from an image
python scripts/infer.py --vae checkpoints/vae_mini/vae_final.pt \
    --dit checkpoints/dit_mini/dit_final.pt --image path/to/photo.png \
    --out artifacts/meshes/demo.glb --guidance 4.5 --steps 40

# 4. quantitative evaluation
python scripts/evaluate.py --vae checkpoints/vae_mini/vae_final.pt --level vae --n 20
```

## Verified behaviour

Run in this repository (RTX3090 / CPU, no cherry-picking):

| Check | Result |
|---|---|
| `tests/test_vae_parity.py` vs the released `tencent/Hunyuan3D-2.1` shape VAE (410 tensors, 327.7 M params) | **56 PASS / 0 FAIL — every asserted stage bit-identical (max abs diff 0.0)** in float32: encoder cross-attention, 8 self-attention blocks at full size (4096 latents × 81920 points), `ln_post`, `pre_kl`, posterior mean/logvar/std, the decoder on a 32³ grid, and the Fourier embedder. Also asserts the discovered weight-mapping rules (`c_kv` identity; `c_qkv` `[heads,3,d] → [3,heads,d]` permutation) |
| `tests/smoke_pipeline.py` (CPU and GPU) | **ALL PASS** — VAE fwd/bwd, multi-scale tokens, eikonal double-backward (GPU), DiT fwd/bwd, length-aware shift monotonicity, sampler determinism, dense-vs-hierarchical field agreement (2.4e-7), watertight extraction |
| `tests/test_mesh_ops.py` | **8 passed** — remeshing open shells yields genus-0 watertight solids with positive volume, TSDF signs/truncation, near-surface sign agreement, sampling invariants |
| Hierarchical volume decoding | matches dense decoding to `2.4e-7` on a 64³ grid while evaluating a fraction of the queries |
| Data pipeline on real assets | 100% of the ~820 ModelNet40 assets remesh to **watertight, positive-volume** solids (median 56 s/asset on CPU) |
| Data pipeline on real assets | **1005 assets** processed: 100% remesh to watertight, positive-volume solids |
| Training on 4xRTX3090 | `checkpoints/vae_mini/`, `checkpoints/dit_mini/` + `artifacts/eval/` (Chamfer/F-score/IoU tables) |

## Scale of the mini model vs the report

| | Report (Seed3D 1.0) | This mini reproduction |
|---|---|---|
| VAE | 327 M params (4096 tokens, width 1024, 16 latent-self-attn layers) | 21 M params (128–256 tokens, width 384, 2 query-self-attn layers) |
| DiT | 1.5 B params, 4096 tokens | 124 M params (`depth=8` double + 12 single blocks), 256 tokens |
| Conditioning | DINOv2 + RADIO @518² | DINOv2-base + C-RADIOv3-B @512² (1373 tokens) |
| Data | web-scale curated 3D corpus | ModelNet40, 3 classes x 250 assets (class-balanced; `ship` does not exist in this dataset — `guitar` is used, see `docs/DATA_PIPELINE.md`) |
| Training | PT → CT → SFT, large cluster | same three stages, 4×RTX3090, ~20 k steps each |
| Texture | Seed3D-MV + PBR + UV | not reproduced (out of scope) |

## Environment notes (this machine)

* Direct `huggingface.co` / `github.com` access is blocked; everything here goes
  through `https://hf-mirror.com` (`export HF_ENDPOINT=https://hf-mirror.com`,
  see `scripts/env_hf.sh`). The mirror rate-limits aggressive metadata probing, so
  the downloader fetches a file list once and then downloads serially with resume.
* Blender, `torch_cluster`, `pytorch3d`, `diso` and `omegaconf` are unavailable;
  none are required — FPS, ball queries, remeshing, rendering and extraction are
  implemented in numpy/PyTorch.
* `timm` is required only for RADIO's remote code; `encoder_type: dinov2_pair`
  works without it.

## Licensing / attribution

* This is an independent implementation; the Hunyuan3D-2.1 sources under
  `third_party/` are read-only reference material under the Tencent Hunyuan
  Community License and are **not** redistributed as part of the mini model.
  The parity test downloads their weights only to compare numbers.
* DINOv2 (Apache-2.0), C-RADIOv3-B (NVIDIA, see its model card), ModelNet40
  (see the dataset card) keep their own licenses.
