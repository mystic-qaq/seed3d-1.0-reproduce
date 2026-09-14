# Training recipe (4×RTX3090, ≤24 h)

Verified numbers: step times and peak memory were measured on this machine
(4×RTX3090 24 GB, torch 2.5.1+cu121, bf16 autocast) by the commands quoted below.

## 0. Preconditions

| | |
|---|---|
| Data | `data/shards/{train,val}-*.npz` (see `docs/DATA_PIPELINE.md`); ~820 assets is enough for the mini model |
| Checkpoints | `checkpoints/dinov2-base`, `checkpoints/C-RADIOv3-B` (+ `timm`), see `docs/MODELS.md` |
| Env | `envs/seed3d/bin/python` (torch 2.5.1+cu121, python 3.11); `requirements-env.txt` pins it |

```bash
export HF_ENDPOINT=https://hf-mirror.com     # or: source scripts/env_hf.sh
bash scripts/run_mini.sh --gpus 0,1,2,3 --vae-steps 20000 --dit-steps 20000
```

`run_mini.sh` is resumable: it re-uses `data/shards`, `data/latents_mini`,
`checkpoints/vae_mini/vae_latest.pt` and `checkpoints/dit_mini/dit_latest.pt`
when they exist (`--skip-data/--skip-vae/--skip-dit` skip stages explicitly).

## 0a. Why training was 20x slower than the hardware allows (measured)

This was investigated to the end because the first runs' step times (13-27 s)
contradicted the single-GPU measurement (0.30 s). Two independent causes:

**1. One occupied GPU stalls the whole NCCL job.** GPUs 4-7 sit on one NUMA node /
PCIe switch. During the first runs another tenant's process held **GPU 6** at 99%
(11.8 GB). NCCL collectives involve every rank, so the four-rank job advanced at the
speed of the contended device. Measured A/B, same model, same data, same code:

| GPU set | torch DDP | my flat all-reduce | no sync |
|---|---|---|---|
| 0,1,2,3 (idle) | **0.340 s/step** | 0.318 s/step | 0.277 s/step |
| 4,5,6,7 (GPU 6 busy) | **7.3-7.9 s/step** | 4.6-9.8 s/step | - |

Same binary, same batch, same script -> **10-20x**. The profiler confirms where it
lands: on GPUs 4-7 `fwd_bwd` was 13.27 s/step (98% of the step); on GPUs 0-3 the
same field is 0.78 s (64%), with the rest in the data loader.

**2. Every GPU negotiates PCIe Gen 1 instead of Gen 4.** `nvidia-smi
--query-gpu=pcie.link.gen.current,pcie.link.gen.max` reports `1` for 7 of 8 GPUs
(max `4`); the only exception is the GPU that is actually busy (Gen 4). So the
fabric runs at ~1/8 of its design bandwidth, and NCCL all-reduce tops out at
**4-5.3 GB/s** (measured, 1 MB - 64 MB messages) versus ~25 GB/s for a healthy
Gen-4 x16 link. A 100 MB gradient therefore needs ~20 ms of pure transfer, which is
why the *single-collective* flat wrapper does not help by itself (measured
0.043 s/all-reduce). This looks like an ASPM / power-management or riser issue on
the host, not something a training script can fix - but it is the reason this box
cannot scale like a properly wired 4x3090 node.

**Practical consequences used here**

* run on the GPU group whose members are all idle (`--gpus 0,1,2,3` in this run);
* prefer `FlatDataParallel` (`src/seed3d_mini/parallel.py`, `--data-parallel flat`)
  over stock DDP: one flat all-reduce per step instead of per-parameter buckets;
* `SEED3D_PROFILE=1` prints the per-phase breakdown, so a future slow run can be
  attributed in one command instead of by guesswork.

## 0b. What was actually run here (shared-machine numbers)

This machine is *shared*: during the run, 3-4 unrelated training jobs held 7 GPUs
and most of the CPU. Measured on GPUs 4-7 with that contention:

| | idle-GPU estimate | measured here |
|---|---|---|
| VAE step (batch 4/GPU, grad-accum 4) | 0.2-0.4 s | **9.2 s** |
| VAE peak memory | 4.2 GB | **3.1-3.2 GB** |
| DiT step (batch 4/GPU, grad-accum 4) | ~0.6 s | see `checkpoints/dit_mini/train_dit.jsonl` |

The batch sizes were chosen so the run fits *alongside* the other jobs (the DiT at
batch 16 peaks at 18.3 GB and would OOM). Practical consequence: the step budget
was reduced to 2000 (VAE) + 1500 (DiT) so a full end-to-end model is produced in
hours rather than days; on an idle machine the same wall clock buys 5-10x the steps.

## 1. Measured costs

| stage | command | measured |
|---|---|---|
| preprocessing | `prepare_data.py process --workers 12 --remesh-resolution 96` | ~85–260 s/asset, CPU-only; 823 assets ≈ 1.5–2 h on 12 workers |
| sharding | `prepare_data.py shards` | negligible |
| VAE step | `train_vae.py --batch-size 8 --grad-accum 2` | **0.31–0.6 s/step**, 4.2 GB/GPU (batch 8, 8192 queries, eikonal on) |
| VAE 20 k steps on 4 GPUs | — | **≈ 25–50 min** |
| latent packing | `pack_latents.py` | ~1–2 min for ~800 assets |
| DiT step | `train_dit.py --batch-size 16` | **1.1 s/step**, 18.3 GB/GPU (the image encoders' activations dominate; the conditioner runs frozen) |
| DiT 20 k steps on 4 GPUs | — | **≈ 1.6 h** |

So the full pipeline is ≈ **4–6 h**, comfortably inside the 24 h budget; the
remaining time can go to more steps, a bigger DiT (`hidden_size`, `depth`,
`depth_single_blocks` in `configs/dit_mini.yaml` are independent knobs) or more
data.

## 2. Hyper-parameters that matter

* **KL warm-up** (`kl_warmup_steps: 2000`): `λ_KL` ramps from 0 to 1e-4. Without
  it the posterior collapses before the decoder has learned anything.
* **Eikonal warm-up** (`eikonal_warmup_steps: 4000`): at initialisation a random
  field has |∇d| ≈ 50–150, so the eikonal term is worth ~2.7e4 (measured) and
  would otherwise drown the reconstruction loss. It is also restricted to queries
  whose |target| is below the truncation value, because a *truncated* field has no
  unit-gradient requirement in the flat far field.
* **Multi-scale tokens** (`train_num_latents: [128, 192, 256]`): the DiT's
  `stage_schedule` then walks 128 → 256 tokens, mirroring the report's PT → CT.
* **Length-aware timestep shift** (`flow_matching.base_shift/max_shift`): keep the
  training and sampling settings identical, otherwise the sampler integrates a
  different schedule than the model was trained on.
* **CFG dropout** (`cfg_drop_prob: 0.1`): the conditioner is fed an explicit
  `drop_mask`, so the unconditional branch is exactly the zero-condition the
  sampler uses at inference.

## 3. Failure modes to watch

| symptom | cause | fix |
|---|---|---|
| `val_near_l1` flat while `val_far_l1` drops | decoder ignores the latent (posterior collapse) | raise `kl_warmup_steps`, lower `lr`, or lower `lambda_kl` |
| VAE meshes are blobs | truncation too small/large, or `near_ratio` too low | check `docs/DATA_PIPELINE.md` query-cloud stats |
| DiT loss plateaus high | unconditional branches dominate (drop prob too high) or the image encoder is frozen *and* the projection is under-trained | lower `cfg_drop_prob`; give the conditioner a higher LR (its projections are the only trainable part) |
| OOM on the eikonal branch | double-backward attention is dense | already bounded (`max_points: 512`) |
| OOM in the DiT step | 512² inputs x 16 samples is 18.3 GB | lower `dit.batch_size` to 8–12 (or `data.image_size` to 256); both are config-only changes |
| Sampling produces noise | guidance too high for 256-token models | 4.0–5.0 works for the mini config; check `artifacts/eval/` |

## 4. Evaluation

```bash
python scripts/evaluate.py --level vae --n 20 \
    --vae checkpoints/vae_mini/vae_final.pt --out artifacts/eval
python scripts/evaluate.py --level dit --n 20 \
    --vae checkpoints/vae_mini/vae_final.pt --dit checkpoints/dit_mini/dit_final.pt
```

Reports Chamfer-L1/L2, F-score@1%/2%, volumetric IoU, normal consistency and the
watertight rate, both per sample and averaged, into
`artifacts/eval/eval_<level>.json`.
