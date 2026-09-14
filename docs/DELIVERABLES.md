# Deliverables index

What to look at, in the order a reviewer would.

## 1. Proof that the report was read and understood

| File | Why it matters |
|---|---|
| `docs/REPRODUCTION_NOTES.md` | report section -> code map, the **two places the report is ambiguous** (decoder self-attention placement in Eq. 2; which reference checkpoint convention to follow) and how each was resolved with a measurement |
| `docs/hunyuan_vae_keys.txt` | the released Hunyuan3D-2.1 VAE's full parameter listing + config, i.e. the evidence base for the comparison |
| `third_party/hunyuan3d-2.1/` | the reference sources studied (read-only) |

## 2. Proof the code is correct

| Command | Result |
|---|---|
| `envs/seed3d/bin/python tests/test_vae_parity.py` | **56 PASS / 0 FAIL**, every asserted stage **bit-identical** (max abs diff 0.0 in float32) against the real Hunyuan3D-2.1 shape VAE — encoder (4096 latents x 81920 points), 8 self-attention blocks, posterior moments, decoder on a 32^3 grid, Fourier embedder. Report in `artifacts/logs/vae_parity.txt` |
| `envs/seed3d/bin/python tests/test_mesh_ops.py` | 8 passed: remeshing open shells -> genus-0 watertight solids, TSDF signs/truncation, sampling invariants |
| `envs/seed3d/bin/python tests/smoke_pipeline.py` | ALL PASS (CPU and GPU): VAE/DiT fwd+bwd, multi-scale tokens, eikonal double-backward, length-aware shift monotonicity, sampler determinism, dense-vs-hierarchical field agreement (2.4e-7), watertight extraction |
| `envs/seed3d/bin/python tests/test_integration.py` | end-to-end through the CLIs: train_vae -> pack_latents -> train_dit -> infer -> evaluate, on synthetic shards |

## 3. Proof the data pipeline works on real data

`docs/DATA_PIPELINE.md` + `data/processed/**/*.json` (per-asset records):

* 1005 real ModelNet40 assets downloaded through `hf-mirror.com` and processed;
* 100% of them remesh to **watertight, positive-volume** solids;
* class-balanced shards: 3 classes x 250 assets (`data/shards/`, 750 samples);
* two failure modes found by measurement and fixed (documented in
  `docs/REPRODUCTION_NOTES.md` §4): `binary_fill_holes` inflates open shells, and a
  half-voxel grid shift is required for marching cubes to close axis-aligned faces.

## 4. Proof it trains

* `scripts/run_mini.sh` - one command, 4 GPUs, resumable;
* `checkpoints/vae_mini/train_vae.jsonl`, `checkpoints/dit_mini/train_dit.jsonl` -
  loss curves (reconstruction, KL, eikonal, timestep statistics, step time);
* `artifacts/eval/eval_vae.json`, `artifacts/eval/eval_dit.json` - Chamfer-L1/L2,
  F-score@1%/2%, volumetric IoU, normal consistency, watertight rate;
* `artifacts/meshes/` - sampled shapes;
* `tests/diagnose_latents.py` - latent-space health (dead dimensions, posterior
  drift), the thing that most often silently caps generation quality.

## 5. Proof the mini run is a *scaled* version, not a different model

`README.md` "Scale of the mini model vs the report" table: same VAE+DiT recipe,
same three-stage schedule (PT/CT/SFT), same losses; only width/depth/token count
and the dataset size are reduced. `docs/TRAINING_RECIPE.md` documents the measured
costs and the knobs to scale back up.
