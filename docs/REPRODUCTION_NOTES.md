# Seed3D 1.0 shape-generation reproduction — design notes

This document records *what* was reproduced, *where the technical report is
ambiguous*, and *how each ambiguity was resolved*. It is the part of the
reproduction that is not code: every decision below is traceable to a line in the
report, to the reference implementation, or to a measurement taken here.

Primary source: **Seed3D 1.0: From Images to High-Fidelity Simulation-Ready 3D
Assets** (arXiv [2510.19944](https://arxiv.org/abs/2510.19944), `docs/seed3d.txt`
is the text extraction used throughout). The shape stage is Sec. 2.1
(Seed3D-VAE 2.1.1, Seed3D-DiT 2.1.2), data Sec. 3.1, training Sec. 4.1,
inference Sec. 6.

Reference implementation studied for the vector-set architecture:
**Hunyuan3D-2.1** (`third_party/hunyuan3d-2.1/`, Tencent Hunyuan Community
License — used here only for reading and for a numerical parity check, never
redistributed).

---

## 1. Architecture mapping, report -> code

| Report component | Statement in the report | Implementation |
|---|---|---|
| Seed3D-VAE encoder | "dual cross-attention encoder": `Z_0 = CrossAttn(PE(P), n_P)`, `Z_i = SelfAttn(Z_{i-1})`, `i = 1..L_e` (Eq. 1) | `VecSetEncoder` (`src/seed3d_mini/models/vae.py`): 1 cross-attention block whose queries are FPS-selected surface points, then `num_encoder_layers` self-attention blocks over the latent set. Inputs are `PE(p) ‖ n_p` (`FourierEmbedder` + normals). |
| Encoder input set | `P = P_u ∪ P_s`: *uniform* samples plus *salient edge* points | `sample_encoder_inputs` (`modules/point_sampling.py`) splits the point cloud into a random block and a sharp-edge block (`sample_edge_points` in `mesh_ops.py`, dihedral-angle based). |
| Seed3D-VAE decoder | `d̂(x) = MLP(CrossAttn(SelfAttn^{(j)}(PE(x)), Z))`, `j = 1..L_k` (Eq. 2) | `TSDFDecoder`: Fourier-embedded query points → `num_decoder_layers` self-attention layers → cross-attention to `Z` → MLP head with 1 output channel. |
| Supervision | "We adopt truncated signed distance functions (TSDFs) as the supervision signal …, effectively constraining the regression range while preserving fine details" | `MeshField.sdf()` produces a truncated SDF (`truncation=0.1`), the dataset stores near-surface + uniform queries with per-query weights. |
| VAE objective | `L_VAE = L_recon + λ_KL L_KL`, KL warm-up to `λ_KL = 1e-4` (Eq. 3) | `Seed3DVAE.forward`; `lambda_kl` is annealed from 0 to its target over `kl_warmup_steps` in `train_vae.py`. |
| Multi-scale VAE training | `M ∈ {256, 512, …, 4096}` sampled during training, "leveraging the vector set architecture's length-agnostic property" | `train_num_latents: [128, 192, 256]` in `configs/vae_mini.yaml`; a fresh `M` is drawn every step. The encoder resamples FPS queries to `M`, the decoder never sees `M` at all (no positional encoding on `Z`). |
| DiT conditioning | Dual encoder: **DINOv2 and RADIO**, "their feature representations are concatenated channel-wise" | `DualImageEncoder` (`models/conditioner.py`): both backbones are run, RADIO's patch grid is aligned to DINOv2's by `PixelShuffleAlign` (exact pixel-shuffle when the ratio is an integer, bilinear otherwise), features are concatenated on the channel axis and projected to `cond_dim`; summary tokens are concatenated on the sequence axis. |
| DiT backbone | "hybrid design of FLUX": double-stream blocks (modality-specific norms/QKV/MLPs, joint attention) then single-stream blocks | `DoubleStreamBlock` / `SingleStreamBlock` in `models/dit.py` — shape latents and image tokens are the two streams; the single-stream blocks operate on `[cond ‖ latent]` and the condition part is dropped before the head, exactly as the reference DiT does. |
| Diffusion scheduling | flow matching, velocity prediction, logit-normal timesteps, **length-aware timestep shift** | `flow_matching.py`: `t ~ sigmoid(N(0,1))`, then `t' = μt/(1+(μ-1)t)` with `μ = α·n_tokens + β`, `α = log((base+max)/base)/log(max_tokens/base_tokens)` (the SD3/FLUX resolution-shift rule with sequence length in place of resolution). |
| Inference | "deterministic sampling through the learned velocity fields" | `euler_sample` (default) and `heun_sample`, both with classifier-free guidance on the velocity. |
| Iso-surface | "reconstructs the mesh using dual marching cubes", coarse-prune + full-precision refinement, analytical gradients for vertex placement | `volume_decode.py`: `dense_volume_decode` and `hierarchical_volume_decode` (coarse pass → conservative active band → fine evaluation, far field saturated to the truncation value), extraction via marching cubes; the decoder is differentiable, so DMC-style analytic gradients are available through `torch.autograd.grad`. |
| Training schedule | PT on 256 latent tokens → CT on 4096 → SFT at lower LR | `stage_schedule` in `configs/dit_mini.yaml`: 128 tokens @2e-4 → 256 tokens @1.5e-4 → 256 tokens @6e-5. |

## 2. Deliberate deviations from Hunyuan3D-2.1 (and why)

The task description suggests reusing Hunyuan3D-2.1 as a scaffold. That was done
for the *architecture skeleton*, but several places had to change to follow the
report — these are the interesting differences.

1. **TSDF instead of binary occupancy.** Hunyuan3D-2.1's shape VAE is trained
   with `label_type: binary` occupancy and its `cross_attention_decoder` is a
   *single* block; the extracted surface sits at `mc_level = -1/512`. Seed3D
   supervises with a *TSDF*, i.e. a signed, bounded regression target. The
   consequence is that the zero level set is directly meaningful, `mc_level = 0`,
   and the field can be queried hierarchically. Implemented as a 1-channel
   regression head plus optional eikonal regularisation.
2. **Decoder self-attention placement.** Report Eq. 2 puts the `L_k`
   self-attention layers on the **query points**; Hunyuan3D-2.1's
   `ShapeVAE.decode()` applies its 16-layer transformer to the **latent tokens**
   (the Dora/3DShape2VecSet convention). We implement both:
   `decoder_self_attn: query|latent|none` (`models/vae.py`). All three are
   numerically identical at `L_k = 0`, which is how the parity test isolates the
   difference. The shipped mini config uses `query` (literal reading of the
   report) with `L_k = 2`; see §4 for the measured cost.
3. **Dual image encoder.** Hunyuan3D-2.1 conditions on DINOv2 alone
   (`SingleImageEncoder`); the report adds RADIO for geometry-aware features. The
   mini model implements the dual path and keeps `encoder_type: single` as a
   fallback, plus a `dinov2_pair` configuration (DINOv2-base + DINOv2-small) for
   environments where RADIO's remote code cannot be executed.
4. **Length-aware timestep shift.** Hunyuan3D-2.1 has the `logit_normal` option
   but no sequence-length-dependent shift; the report explicitly requires it, and
   it is implemented and unit-tested (`tests/smoke_pipeline.py` asserts that 4096
   tokens map to a higher mean noise level than 128 tokens).
5. **Multi-scale latent training.** Hunyuan3D-2.1 trains one token length per
   model (4096); the report's VAE randomly samples `M ∈ {256..4096}` per step so
   the *same* decoder serves all budgets. This is implemented and is what lets
   the mini DiT train short and sample long.
6. **Data pipeline without Blender.** The report renders with Blender Cycles
   (HDR/point lights, elevation in `[-30°, 70°]`); no Blender is available in this
   environment, so `src/seed3d_mini/render/rasterizer.py` implements a PyTorch
   z-buffer rasteriser with the same contract (RGB + normal + mask + camera
   origin, jittered directional light, identical elevation sampling).

## 3. What is *not* reproduced

Out of scope by design (the task asks for the **shape** stage, and the report's
texture stage is a separate model family):

* Seed3D-MV (multi-view RGB), Seed3D-PBR (albedo/metallic/roughness),
  Seed3D-UV (UV inpainting) — Sec. 2.2, not implemented.
* The cluster-scale training infrastructure: HSDP, multi-level activation
  checkpointing, kernel fusion, Ray Data, Kubernetes operators, fault tolerance
  (Sec. 3.2, 5). The mini version uses `torchrun` + DDP (`train_utils.py`), which
  is the honest single-node analogue.
* The full data pipeline (dedup by rendered-image features with FAISS,
  orientation canonisation, VLM quality filtering, Dual Marching Cubes
  remeshing). What *is* implemented: normalisation, watertight remeshing
  (voxelised SDF + parity-vote interior + marching cubes), uniform and
  sharp-edge sampling, reference-view rendering, TSDF query clouds. Dedup and
  orientation canonisation are skipped because ModelNet40 is already
  canonicalised per category and small.

## 4. Measurements taken in this environment

* **Parity with the reference VAE**: `tests/test_vae_parity.py` loads the real
  `tencent/Hunyuan3D-2.1` shape VAE (410 tensors, 327.7M params, fp16) and
  compares every stage against `src/seed3d_mini`. Result: **56 PASS / 0 FAIL,
  every asserted stage bit-identical (max |Δ| = 0.0)**
  (`artifacts/logs/vae_parity.txt`), including the encoder's cross-attention,
  8 self-attention blocks, `ln_post`, the posterior moments and the decoder on a
  32³ query grid. The weight-mapping rules discovered and asserted by the test:
  * `c_kv` is stored as `[2*width, width]` and must be read as
    `[heads, 2, head_dim]` per head — my split `kv[..., 0] / kv[..., 1]` layout is
    *identical*, no permutation needed;
  * `c_qkv` is stored as `[3*width, width]` with layout
    `[heads, 3, head_dim]` and must be permuted to `[3, heads, head_dim]`;
  * `q_norm/k_norm` operate on the per-head slice (`head_dim`), the Fourier
    embedder must use `include_pi=True` with `num_freqs=8` to reproduce the
    checkpoint's `input_proj` input width of 55.
* **Decoder self-attention placement**: with one layer on the query side (report
  Eq. 2) vs one layer on the latent side (reference), the decoded field differs by
  ~2.0 max-|Δ| (relative 0.52) — i.e. the two readings are *not* interchangeable,
  which is why the switch exists and why the mini config states its choice
  explicitly.
* **Hierarchical decoding**: the coarse-to-fine decoder reproduces the dense
  decoder to `2.4e-7` max-|Δ| on a 64³ grid while evaluating a fraction of the
  points (asserted in `tests/smoke_pipeline.py`).
* **Extraction sanity**: a unit-sphere field decodes to a watertight mesh
  (4728 vertices / 9452 faces in the smoke test).
* **Remeshing on real data**: ModelNet40 meshes are *open, non-manifold CAD
  shells* (measured interior fraction 0.8% via ray parity; `fill_holes` does not
  close them).  The flood-fill + morphological-cleanup strategy
  (`_flood_fill_outside` + `_clean_occupancy`) turns 100% of the ~820 assets into
  watertight, positive-volume solids, and the extracted surface stays within
  0.03 (mean) / 0.13 (p95) of the input samples in the canonical `[-1,1]³` frame.
  Two gotchas found by measurement and documented in the code:
  `binary_fill_holes` must *not* be part of the cleanup (it inflated the median
  volume from 0.20 to 0.42), and a half-voxel grid shift is required or
  axis-aligned faces land exactly on sample planes and marching cubes leaves open
  edges.
* **Sign-field cost**: an exact inside/outside test costs 60–110 queries/s on a
  100k-face mesh, which dominated the whole pipeline (~4 min/asset).  The fix is
  to give the near-surface queries their sign from the surface normal they were
  jittered along (`δ·n < 0`), and to run the exact test only on the far-field
  points that fall inside the truncation band: 4 min/asset → 5.6 s for the
  16k-query cloud, with 96% sign agreement against the exact reference inside the
  band (measured; see `tests/prof_asset.py`).
* **Eikonal term**: measured |∇d| ≈ 50–150 at initialisation, i.e. the raw loss is
  ~2.7e4 and swamps reconstruction, so the term is warmed up and restricted to
  the un-truncated part of the target (`eikonal_warmup_steps`).

## 4b. Open blocker: the mini VAE converges to the mean field

Status: **not solved**, root cause localised with a reproducible experiment. This is
the honest headline of the training part of this reproduction.

Symptom. Training looks healthy (weighted recon reaches 1.8e-4) but the decoded
field is input-independent: two different shapes decode to fields differing by
0.001 world units, and the zero latent decodes to the same thing
(`tests/probe_latent_usage.py` prints `DECODER IGNORES THE LATENT`). Meshes are
therefore garbage (Chamfer 0.74).

Reproducible evidence (all scripts are in `tests/`):

| experiment | result |
|---|---|
| `tests/overfit_one.py` (fit **one** asset) | plateaus at weighted MSE **1.8e-4**, i.e. exactly the best-constant MSE -> cannot overfit a single shape |
| decoder-only fit of one field (frozen random latent) | also plateaus at 1.8e-4 |
| plain unweighted MSE on the same data | trains (pred std grows 0.034 -> 0.093, loss falls 1.07 -> 0.15) but converges toward the same mean-field regime |
| fixed learning rate 1e-3 vs 3e-3, eikonal on vs off, far-field weight 0.25 -> 0.002 | plateau value unchanged in all four settings (only the step time changes) |

Root cause. The regression target is a *truncated* SDF, so its dynamic range is
tiny and its mean dominates the MSE:

| quantity | value |
|---|---|
| full target std | 0.0458 |
| in-band (|sdf| < 0.099) target std | **0.0135** |
| best constant predictor, weighted MSE | **1.814e-4** |
| measured model plateau, weighted MSE | **1.8e-4** |

The shape signal is ~1.3e-2 world units on an object of size 2, i.e. 0.7% of the
bounding box, while the surface itself is a thin sheet. Every configuration tested
finds the "output the mean field" solution because it is worth ~96% of the
achievable loss reduction. More steps, a larger model or a different optimiser do
not address this -- it is a *target definition* problem.

Update after the speed fix (`docs/TRAINING_RECIPE.md` §0a) and a 2000-step run at
0.6 s/step:

| step | recon (normalised) | in-band MAE (world) | MSE vs best-constant | shape dependence |
|---|---|---|---|---|
| 24 | 0.100 | 0.043 | worse | - |
| 1000 | 0.0148 | 0.0091 | 0.00254 vs 0.00210 | ~0 |
| 2000 | 0.0134 | 0.0091 | 0.00254 vs 0.00210 | **0.00003** |

So 20x more steps at 20x the speed **do not change the outcome**: the model sits at
the mean field with essentially zero shape dependence. This rules out the training
budget as the cause and confirms the target-definition diagnosis below.

### 4c. Decoder architecture is the blocker (localised, first fix tested)

The decisive experiment compares the decoder against a plain MLP on the **same**
Fourier features, fitting **one** asset's field:

| model | in-band correlation | band std (target 0.0135) |
|---|---|---|
| MLP(51 -> 256 -> 256 -> 1) on Fourier features | **+0.86** | 0.0110 |
| `TSDFDecoder` with a zero latent | -0.05 | 0.0024 |
| `TSDFDecoder`, no self-attention at all | +0.02 | 0.0022 |
| `TSDFDecoder`, query-side self-attention | +0.05 | 0.0017 |
| `TSDFDecoder` + query-skip (added here) | **+0.37** | 0.0065 |

Reading: a two-layer MLP can represent this field; the cross-attention decoder
cannot, and neither removing its self-attention nor moving it changes that. The
output path is simply too narrow for the coordinate function, so no amount of
latent information can reach the output. The `query_skip` connection (raw Fourier
features -> output head, `query_skip: true` by default) is a real improvement on
the single-asset probe (corr -0.06 -> +0.37) **but is not sufficient on the full
dataset** (1500 steps, corr -0.04, shape dependence 2.5e-5), so the remaining gap is
in the conditioning path (latent -> field), not just the coordinate path.

Conclusion: the mini model needs a decoder redesign (wider skip head + the latent
entering as a residual/FiLM modulation, or a coarse occupancy volume predicted
first), not more compute. Everything needed to reproduce this conclusion is in
`tests/probe_latent_usage.py`, `tests/overfit_one.py` and this section.

What to change next (in priority order):

1. **Widen the truncation band.** The reference implementation supervises occupancy
   over the whole volume and extracts at a tiny level (-1/512), i.e. it never asks a
   network to regress a 0.01-magnitude quantity. Using a *narrow band but
   normalised* target (e.g. band = 0.02 with the target rescaled to [-1, 1] inside
   it) would raise the in-band signal amplitude by ~5x for the same geometry.
2. **Supervise the level set, not the distance.** Train on occupancy inside a band
   with a per-point normalised target, and keep the SDF only for extraction.
3. **Predict from the latent before the field**: add a small parametric head that
   maps `Z` to a low-resolution occupancy volume and make the continuous field a
   residual on top of it, so the latent has a short, high-bandwidth path to the
   output.
4. Only after (1)-(3) is the budget question meaningful: the reference trains its
   VAE for far more steps than the ~400 used here.

## 5. Known limitations of the mini model

* ~700 assets (ModelNet40 airplane/car/ship, train split) is four orders of
  magnitude below the report's dataset; expect the mini DiT to learn the *category
  manifold* (plausible silhouettes within three classes) and nothing about
  in-the-wild images.
* Rotational augmentation is yaw-only: the report canonises orientation, and
  ModelNet40 is already axis-aligned.
* The reference-image renderer is a simple rasteriser (no HDR environment), so
  the image conditioning is easier and less photorealistic than the report's.
* No texture stage, so "simulation-ready" here means watertight, manifold,
  correctly-scaled geometry only.
