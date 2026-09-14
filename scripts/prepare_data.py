"""Stage-1 data preparation: raw assets -> training-ready SDF/latent shards.

This is the practical, single-machine version of the report's
"3.1 Data preprocessing" pipeline.  Every stage is explicit and cached, so a
failed run resumes instead of restarting:

    download -> normalise -> remesh to watertight -> sample surface/edge points
    -> render reference views -> sample (T)SDF query cloud -> pack shards

The output of this script is what ``train_vae.py`` consumes; ``train_dit.py``
then packs the VAE latents (``pack_latents.py``).

Usage
-----
    python scripts/prepare_data.py download --categories airplane,car,ship
    python scripts/prepare_data.py process  --categories airplane,car,ship --limit 200
    python scripts/prepare_data.py shards   --categories airplane,car,ship
    python scripts/prepare_data.py stats
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import threading
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seed3d_mini.mesh_ops import (  # noqa: E402
    MeshField,
    RemeshConfig,
    is_watertight,
    load_mesh,
    mesh_stats,
    normalize_mesh,
    remesh_watertight,
    sample_edge_points,
    sample_surface_points,
)

# ModelNet40 (the dataset used here) has no "ship" class, so the three visually
# distinct classes this mini reproduction trains on are airplane / car / guitar.
# See docs/DATA_PIPELINE.md for the category-availability table.
DEFAULT_CATEGORIES = ("airplane", "car", "guitar")
MODELNET_REPO = "naderalfares/ModelNet40"


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
@dataclass
class PrepConfig:
    categories: Tuple[str, ...] = DEFAULT_CATEGORIES
    raw_dir: str = str(ROOT / "data" / "raw" / "modelnet40")
    work_dir: str = str(ROOT / "data" / "processed")
    shard_dir: str = str(ROOT / "data" / "shards")
    shard_size: int = 256
    num_surface: int = 8192
    num_edge: int = 1024
    num_queries: int = 16384
    near_ratio: float = 0.7
    truncation: float = 0.1
    remesh_resolution: int = 128
    remesh_band: float = 1.5
    render_resolution: int = 256
    num_views: int = 1
    val_fraction: float = 0.05
    seed: int = 0
    workers: int = 8
    render_device: str = "cpu"
    #: cap samples per category when packing shards (0 = no cap).  The raw splits
    #: are skewed (airplane 726 / car 297 / guitar 255) and an unbalanced mini model
    #: just learns the majority class, so sharding balances the classes.
    per_category: int = 0


# --------------------------------------------------------------------------- #
# stage 1: download
# --------------------------------------------------------------------------- #
def _modelnet_candidates(categories: Sequence[str], splits: Sequence[str], max_index: int) -> List[str]:
    """Candidate remote paths.

    ModelNet40 files are named ``{category}_{index:04d}.off``, so the catalogue can
    be probed directly.  That matters here: the mirror's recursive file listing of
    this 12k-file repo routinely times out, while per-file downloads are fast.
    """
    return [
        f"ModelNet40/{cat}/{split}/{cat}_{idx:04d}.off"
        for cat in categories
        for split in splits
        for idx in range(0, max_index + 1)
    ]


def _exists(repo: str, path: str, timeout: float = 20.0) -> bool:
    """HEAD-like existence probe against the mirror's resolve endpoint."""
    import urllib.error
    import urllib.request

    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
    url = f"{endpoint}/datasets/{repo}/resolve/main/{path}"
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except urllib.error.HTTPError as exc:
        if exc.code in (404, 403):
            return False
        return False
    except Exception:
        return False


def _discover_files(repo: str, categories: Sequence[str], splits: Sequence[str], max_index: int, workers: int) -> List[str]:
    cands = _modelnet_candidates(categories, splits, max_index)
    print(f"[download] probing {len(cands)} candidate paths ...", flush=True)
    found: List[str] = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for path, ok in zip(cands, ex.map(lambda p: _exists(repo, p), cands)):
            if ok:
                found.append(path)
    return found


def cmd_download(args, cfg: PrepConfig) -> int:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from huggingface_hub import hf_hub_download

    if args.file_list and Path(args.file_list).exists():
        blob = json.loads(Path(args.file_list).read_text())
        wanted = blob["files"] if isinstance(blob, dict) else blob
        print(f"[download] {len(wanted)} paths from {args.file_list}")
    else:
        wanted = sorted(
            set(_discover_files(MODELNET_REPO, cfg.categories, args.splits.split(","), args.max_index, cfg.workers))
        )
        if args.list_only:
            out_path = Path(args.file_list or (ROOT / "data" / "raw" / "file_list.json"))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({"repo": MODELNET_REPO, "files": wanted}, indent=0))
            print(f"[download] discovered {len(wanted)} files -> {out_path}")
            return 0
    if not wanted:
        print("[download] nothing found; try --list-only with a larger --max-index")
        return 1
    if args.limit:
        # keep it deterministic: interleave categories instead of truncating
        by_cat: Dict[str, List[str]] = {}
        for f in wanted:
            by_cat.setdefault(f.split("/")[1], []).append(f)
        interleaved: List[str] = []
        i = 0
        while len(interleaved) < args.limit:
            added = False
            for cat in sorted(by_cat):
                if i < len(by_cat[cat]) and len(interleaved) < args.limit:
                    interleaved.append(by_cat[cat][i])
                    added = True
            if not added:
                break
            i += 1
        wanted = interleaved

    print(f"[download] {len(wanted)} files for categories={list(cfg.categories)}", flush=True)
    raw_root = Path(cfg.raw_dir)
    manifest_path = raw_root / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    lock = threading.Lock()
    counters = {"ok": 0, "cached": 0, "failed": 0}
    t0 = time.time()

    def fetch(remote: str) -> bool:
        local = raw_root / remote
        if local.exists() and local.stat().st_size > 0:
            with lock:
                manifest[remote] = str(local)
                counters["cached"] += 1
            return True
        local.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(args.retries):
            try:
                path = hf_hub_download(MODELNET_REPO, remote, repo_type="dataset", local_dir=str(raw_root))
                with lock:
                    manifest[remote] = str(path)
                    counters["ok"] += 1
                return True
            except Exception as exc:  # noqa: BLE001
                # hf-mirror answers 429 / RateLimit with a Retry-After, so back off
                if attempt == args.retries - 1:
                    print(f"  ! failed {remote}: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
                    with lock:
                        counters["failed"] += 1
                    return False
                time.sleep(2.0 * (2 ** attempt) + float(np.random.uniform(0, 1)))
        return False

    done = 0
    # modest concurrency: the mirror rate-limits global request rate, but pure
    # serial downloading wastes most of the wall clock on latency
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(fetch, r) for r in wanted]
        for fut in cf.as_completed(futs):
            fut.result()
            done += 1
            if done % 25 == 0 or done == len(wanted):
                rate = done / max(time.time() - t0, 1e-6)
                print(
                    f"  {done}/{len(wanted)}  ({rate:.1f} files/s)  ok={counters['ok']} "
                    f"cached={counters['cached']} failed={counters['failed']}",
                    flush=True,
                )
    ok = counters["ok"] + counters["cached"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=0, sort_keys=True))
    print(f"[download] done: {ok}/{len(wanted)} files -> {raw_root}")
    return 0


# --------------------------------------------------------------------------- #
# stage 2: process one asset
# --------------------------------------------------------------------------- #
def process_one(
    rel: str,
    cfg: PrepConfig,
    render_fn=None,
    force: bool = False,
) -> Dict:
    """Everything expensive for a single asset; returns a metadata record."""
    raw_path = Path(cfg.raw_dir) / rel
    category = rel.split("/")[1]
    stem = Path(rel).stem
    out_dir = Path(cfg.work_dir) / category
    out_path = out_dir / f"{stem}.npz"
    meta_path = out_dir / f"{stem}.json"
    if out_path.exists() and meta_path.exists() and not force:
        try:
            return json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            pass

    rng = np.random.default_rng(abs(hash((rel, cfg.seed))) % (2**32))
    t0 = time.time()
    record: Dict = {"id": f"{category}/{stem}", "category": category, "source": rel, "ok": False}

    try:
        mesh = normalize_mesh(load_mesh(str(raw_path)))
        stats_in = mesh_stats(mesh)
        record["input_stats"] = stats_in
        if is_watertight(mesh):
            watertight = mesh
            record["remeshed"] = False
        else:
            watertight = remesh_watertight(
                mesh,
                RemeshConfig(
                    resolution=cfg.remesh_resolution,
                    band=cfg.remesh_band,
                    min_component_faces=32,
                ),
            )
            record["remeshed"] = True
        record["output_stats"] = mesh_stats(watertight)
        if not is_watertight(watertight):
            record["error"] = "remesh_not_watertight"
            meta_path.write_text(json.dumps(record, indent=1))
            return record

        # --- surface + sharp-edge samples (encoder input) ------------------ #
        surf, surf_n = sample_surface_points(watertight, cfg.num_surface, rng=rng)
        if cfg.num_edge > 0:
            edge, edge_n = sample_edge_points(watertight, cfg.num_edge, rng=rng)
        else:
            edge = np.zeros((0, 3), np.float32)
            edge_n = np.zeros((0, 3), np.float32)

        # --- (T)SDF supervision cloud ------------------------------------- #
        field = MeshField(watertight, truncation=cfg.truncation)
        q_pts, q_sdf, q_w = field.build_query_cloud(
            cfg.num_queries, rng=rng, near_ratio=cfg.near_ratio, jitter=1.0 / cfg.remesh_resolution
        )

        arrays = {
            "surface": surf.astype(np.float32),
            "surface_normal": surf_n.astype(np.float32),
            "edge": edge.astype(np.float32),
            "edge_normal": edge_n.astype(np.float32),
            "query": q_pts.astype(np.float32),
            "sdf": q_sdf.astype(np.float32),
            "weight": q_w.astype(np.float32),
        }
        if render_fn is not None:
            images = render_fn(watertight, cfg, rng)
            arrays.update({k: v for k, v in images.items()})
            record["views"] = images["rgb"].shape[0]

        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **arrays)
        record["ok"] = True
        record["n_surface"] = int(surf.shape[0])
        record["n_edge"] = int(edge.shape[0])
        record["n_query"] = int(q_pts.shape[0])
        record["npz"] = str(out_path)
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc(limit=3)
    record["seconds"] = round(time.time() - t0, 2)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(record, indent=1))
    return record


def _make_render_fn(cfg: PrepConfig):
    if cfg.num_views <= 0:
        return None
    from seed3d_mini.render.rasterizer import render_mesh_views

    def render_fn(mesh, cfg_, rng):
        seed = int(rng.integers(1 << 31))
        out = render_mesh_views(
            mesh,
            n_views=cfg_.num_views,
            resolution=cfg_.render_resolution,
            seed=seed,
            device=cfg_.render_device,
        )
        return {"image": (out["rgb"] * 255).astype(np.uint8), "image_normal": (out["normal"] * 127 + 128).astype(np.uint8)}

    return render_fn


def cmd_process(args, cfg: PrepConfig) -> int:
    raw_root = Path(cfg.raw_dir)
    rels = sorted(
        str(p.relative_to(raw_root))
        for p in raw_root.rglob("*.off")
        if len(p.parts) >= 3
    )
    rels = [r for r in rels if len(r.split("/")) > 1 and r.split("/")[1] in cfg.categories]
    if args.limit:
        by_cat: Dict[str, List[str]] = {}
        for r in rels:
            by_cat.setdefault(r.split("/")[1], []).append(r)
        picked: List[str] = []
        i = 0
        while len(picked) < args.limit:
            added = False
            for cat in sorted(by_cat):
                if i < len(by_cat[cat]) and len(picked) < args.limit:
                    picked.append(by_cat[cat][i])
                    added = True
            if not added:
                break
            i += 1
        rels = picked
    print(f"[process] {len(rels)} assets, {args.workers} workers")
    render_fn = _make_render_fn(cfg)
    results: List[Dict] = []
    if args.workers <= 1:
        for i, rel in enumerate(rels):
            results.append(process_one(rel, cfg, render_fn, force=args.force))
            if (i + 1) % 10 == 0:
                _report(results)
    else:
        with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_one, rel, cfg, None, args.force): rel for rel in rels}
            done = 0
            for fut in cf.as_completed(futs):
                results.append(fut.result())
                done += 1
                if done % 25 == 0:
                    _report(results)
    _report(results)
    return 0


def _report(results: Sequence[Dict]) -> None:
    ok = sum(1 for r in results if r.get("ok"))
    print(f"  processed {len(results)}: ok={ok} failed={len(results) - ok}", flush=True)
    if results and len(results) % 100 == 0:
        times = [r.get("seconds", 0) for r in results if r.get("ok")]
        if times:
            print(f"    mean {np.mean(times):.2f}s/asset, total {np.sum(times) / 60:.1f} min", flush=True)


# --------------------------------------------------------------------------- #
# stage 3: pack shards
# --------------------------------------------------------------------------- #
def cmd_shards(args, cfg: PrepConfig) -> int:
    work = Path(cfg.work_dir)
    shard_root = Path(cfg.shard_dir)
    shard_root.mkdir(parents=True, exist_ok=True)
    records: List[Dict] = []
    for meta in sorted(work.rglob("*.json")):
        rec = json.loads(meta.read_text())
        if rec.get("ok"):
            records.append(rec)
    if cfg.per_category:
        by_cat: Dict[str, List[Dict]] = {}
        for r in records:
            by_cat.setdefault(r["category"], []).append(r)
        cap_rng = np.random.default_rng(cfg.seed)
        capped: List[Dict] = []
        for cat, rs in sorted(by_cat.items()):
            rs = sorted(rs, key=lambda r: r["id"])
            if len(rs) > cfg.per_category:
                pick = np.sort(cap_rng.permutation(len(rs))[: cfg.per_category])
                rs = [rs[i] for i in pick]
            capped.extend(rs)
        print(
            "[shards] class balancing (per_category=%d): " % cfg.per_category
            + ", ".join(f"{c} {len(v)} -> {min(len(v), cfg.per_category)}" for c, v in sorted(by_cat.items())),
            flush=True,
        )
        records = capped
    if not records:
        print("[shards] no processed assets found; run `process` first")
        return 1
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(records))
    n_val = max(1, int(round(len(records) * cfg.val_fraction)))
    val_ids = {records[i]["id"] for i in order[:n_val]}

    for split in ("train", "val"):
        items = [r for r in records if (r["id"] in val_ids) == (split == "val")]
        items.sort(key=lambda r: r["id"])
        for s in range(0, len(items), cfg.shard_size):
            chunk = items[s: s + cfg.shard_size]
            with np.load(chunk[0]["npz"]) as f0:
                keys = list(f0.keys())
            data: Dict[str, List[np.ndarray]] = {k: [] for k in keys}
            data["category"] = []
            data["id"] = []
            used = []
            for rec in chunk:
                try:
                    with np.load(rec["npz"]) as f:
                        for k in keys:
                            data[k].append(f[k])
                    data["category"].append(rec["category"])
                    data["id"].append(rec["id"])
                    used.append(rec)
                except Exception as exc:  # noqa: BLE001
                    print(f"  ! skip {rec['id']}: {exc}")
            if not used:
                continue
            out = shard_root / f"{split}-{s // cfg.shard_size:05d}.npz"
            np.savez(
                out,
                **{k: np.stack(v) for k, v in data.items() if k not in ("category", "id")},
                category=np.array(data["category"]),
                sample_id=np.array(data["id"]),
            )
            print(f"[shards] {out} : {len(used)} samples, keys={[k for k in keys]}")

    (shard_root / "manifest.json").write_text(
        json.dumps(
            {
                "categories": list(cfg.categories),
                "n_assets": len(records),
                "n_train": len(records) - len(val_ids),
                "n_val": len(val_ids),
                "config": {k: v for k, v in asdict(cfg).items()},
            },
            indent=1,
        )
    )
    print(f"[shards] done -> {shard_root}")
    return 0


def cmd_stats(args, cfg: PrepConfig) -> int:
    shard_root = Path(cfg.shard_dir)
    train = sorted(shard_root.glob("train-*.npz"))
    val = sorted(shard_root.glob("val-*.npz"))
    if not train:
        print("[stats] no shards yet")
        return 1
    n = 0
    cats: Dict[str, int] = {}
    for p in train + val:
        with np.load(p) as f:
            n += f["surface"].shape[0]
            for c in f["category"]:
                cats[str(c)] = cats.get(str(c), 0) + 1
    print(f"[stats] shards: {len(train)} train / {len(val)} val, {n} samples")
    for c, k in sorted(cats.items()):
        print(f"   {c:12s} {k}")
    return 0


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["download", "process", "shards", "stats", "all"])
    ap.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES))
    ap.add_argument("--splits", default="train", help="comma separated: train,test")
    ap.add_argument("--split", default="train", choices=["train", "test", "all"], help="(kept for compatibility)")
    ap.add_argument("--max-index", type=int, default=1000, help="highest filename index to probe")
    ap.add_argument("--list-only", action="store_true", help="only build the file list")
    ap.add_argument("--file-list", default=None, help="cache file for the discovered paths")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--render-resolution", type=int, default=256)
    ap.add_argument("--remesh-resolution", type=int, default=128)
    ap.add_argument("--num-views", type=int, default=1)
    ap.add_argument("--num-queries", type=int, default=16384)
    ap.add_argument("--render-device", default="cpu")
    ap.add_argument(
        "--per-category", type=int, default=0,
        help="cap samples per category when packing shards (0 = no cap)"
    )
    args = ap.parse_args()

    cfg = PrepConfig(
        per_category=args.per_category,
        categories=tuple(c.strip() for c in args.categories.split(",") if c.strip()),
        workers=args.workers,
        render_resolution=args.render_resolution,
        remesh_resolution=args.remesh_resolution,
        num_views=args.num_views,
        num_queries=args.num_queries,
        render_device=args.render_device,
    )
    stages = {
        "download": cmd_download,
        "process": cmd_process,
        "shards": cmd_shards,
        "stats": cmd_stats,
    }
    if args.stage == "all":
        for name in ("download", "process", "shards", "stats"):
            rc = stages[name](args, cfg)
            if rc:
                return rc
        return 0
    return stages[args.stage](args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
