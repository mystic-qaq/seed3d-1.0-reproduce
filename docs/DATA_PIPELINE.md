# Data pipeline

Implements Sec. 3.1 of the Seed3D 1.0 report at single-machine scale.  The
report's pipeline is: *format standardisation → geometric deduplication →
orientation canonisation → aesthetic/VLM quality filtering → multi-view
rendering → remeshing*, executed on Ray Data over a Kubernetes cluster.  Here the
same stages run as a resumable local pipeline with per-asset caching, so an
interrupted run continues where it stopped.

## Source data

| | |
|---|---|
| Dataset | [ModelNet40](https://huggingface.co/datasets/naderalfares/ModelNet40) (12,311 CAD meshes, 40 categories) via `HF_ENDPOINT=https://hf-mirror.com` |
| Categories used | `airplane`, `car`, `guitar` — **not** the task's suggested "ship": the mirror's copy of ModelNet40 ships 40 categories and `ship` is not among them (the full list is `airplane, bathtub, bed, bench, bookshelf, bottle, bowl, car, chair, cone, cup, curtain, desk, door, dresser, flower_pot, glass_box, guitar, keyboard, lamp, laptop, mantel, monitor, night_stand, person, piano, plant, radio, range_hood, sink, sofa, stairs, stool, table, tent, toilet, tv_stand, vase, wardrobe, xbox`). `guitar` keeps the "three visually distinct rigid classes" property the mini scope asked for; swapping in a ship category is a change of `--categories` plus one more source repo |
| Split | official `train` + `test` splits: airplane 726, car 297, guitar 255 |
| Class balancing | the raw splits are skewed 2.8:1, and an unbalanced mini model just learns the majority class, so sharding caps every category at `--per-category N` (smallest class size) |
| Why | watertight-ish CAD geometry, canonical per-category orientation, tiny on disk, and a real 3D-native distribution rather than renders |

The downloader fetches the repo's file list once (the mirror's recursive tree API
is unreliable) and then downloads serially/with modest concurrency, honouring
429 back-off and resuming from partial files.

## Stage by stage

| Stage | Report | Implementation | Notes |
|---|---|---|---|
| Format standardisation | OBJ/FBX/GLTF/PLY → GLB | `mesh_ops.load_mesh` | merges scenes, removes degenerate faces/unreferenced vertices |
| Canonical frame | orientation canonisation | `mesh_ops.normalize_mesh` | bbox-centred, largest extent = 2 → everything lives in `[-1, 1]³` |
| Deduplication | rendered-feature FAISS dedup | *skipped* | ModelNet40 has no near-duplicate mass problem at this scale; the hook is documented rather than faked |
| Quality filter | aesthetic score + VLM | *reduced* | assets that fail remeshing/watertightness are dropped and logged in `data/processed/<cat>/<id>.json` |
| Remeshing | voxelise → SDF floodfill → Dual Marching Cubes | `mesh_ops.remesh_watertight` | exact point-to-triangle distances with KD-tree pruning; interior by multi-ray parity voting (works on non-watertight input); marching-cubes extraction |
| Surface sampling | `P = P_u ∪ P_s` | `sample_surface_points`, `sample_edge_points` | area-weighted uniform samples + dihedral-angle sharp-edge points |
| Multi-view rendering | Blender Cycles, elevation ∈ [-30°, 70°] | `render/rasterizer.py` | PyTorch z-buffer, same elevation distribution, jittered key light, emits RGB/normal/mask |
| TSDF supervision | TSDF regression target | `MeshField.build_query_cloud` | 70% near-surface (thin shell) + 30% uniform volume, per-query weights (near-surface up-weighted 4x) |
| Packing | HDFS bundles of SDF samples / latents | `prepare_data.py shards`, `pack_latents.py` | `.npz` shards of 256 samples |

## Shard format

`data/shards/{split}-NNNNN.npz` — arrays stacked over samples:

| key | shape | meaning |
|---|---|---|
| `surface` | `[N, 8192, 3]` | uniform surface samples (encoder keys/values) |
| `surface_normal` | `[N, 8192, 3]` | outward normals at those samples |
| `edge` | `[N, 1024, 3]` | sharp-edge samples |
| `edge_normal` | `[N, 1024, 3]` | normals at the edge samples |
| `query` | `[N, 16384, 3]` | decoder supervision coordinates |
| `sdf` | `[N, 16384, 1]` | truncated signed distance (|d| ≤ 0.1) |
| `weight` | `[N, 16384, 1]` | loss weight (1.0 near-surface, 0.25 far field) |
| `image` | `[N, 3, 256, 256]` uint8 | reference view rendered from the canonical frame |
| `image_normal` | `[N, 3, 256, 256]` uint8 | world-space normal map of the same view |
| `category`, `sample_id` | `[N]` | provenance |

`data/latents_mini/{split}-NNNNN.npz` (produced by `scripts/pack_latents.py`)
holds `latent [N, M, d]`, `mu`, `logvar`, plus the same `image`, `category`,
`sample_id`, and `latent_stats.json` with the global latent std (the analogue of
Hunyuan3D's `z_scale_factor = 1/std`).

## Cost on this machine

Measured per asset (32-core CPU, no GPU needed for preprocessing):

| step | time |
|---|---|
| download | ~0.5 s/asset (mirror-latency bound) |
| remesh @128³ + sample + render @256² | see `data/processed/**/*.json` (`seconds` field), typically 3–10 s |
| sharding | negligible |

So ~820 assets ≈ 1–2 h of preprocessing on 16–32 workers — one-off, and cached
per asset, which is why the training config consumes shards rather than meshes.
