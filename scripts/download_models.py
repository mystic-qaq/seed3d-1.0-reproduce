#!/usr/bin/env python
"""Resumable downloader for every checkpoint used by the Seed3D-1.0 mini reproduction.

What it fetches (into ``checkpoints/`` by default)
--------------------------------------------------
``hunyuan3d-vae-v2-1``  tencent/Hunyuan3D-2.1 :: ``hunyuan3d-vae-v2-1/``
                        shape VAE (VecSet) -- the parity reference for our
                        ``src/seed3d_mini/models/vae.py``.  Textures / paint
                        pipeline (``hy3dpaint``, ``hunyuan3d-paintpbr-v2-1``)
                        are explicitly skipped: several GB we do not need.
``dinov2-small``        facebook/dinov2-small      image encoder #2 of the
``dinov2-base``         facebook/dinov2-base       dual-encoder conditioner.
``C-RADIOv3-B``         nvidia/C-RADIOv3-B         *optional* geometry-aware
                        encoder; failure is reported, not fatal.

Design notes
------------
* ``HF_ENDPOINT`` defaults to https://hf-mirror.com because huggingface.co is
  not directly reachable from this host; an existing env var wins.
* ``snapshot_download(..., local_dir=...)`` is used so artifacts land at stable,
  human-readable paths instead of opaque ``models--org--name`` cache blobs.
  hf_hub keeps ``<local_dir>/.cache/huggingface/download/*.incomplete`` files,
  so re-running the script resumes partial downloads instead of restarting.
  Every model is additionally wrapped in a bounded retry loop.
* ``max_workers`` is capped at 4 on purpose: the mirror throttles aggressive
  parallelism and fewer, larger streams are more stable than many small ones.
* A byte-progress **stall watchdog** (``--stall-timeout``, default 150s) aborts
  a download whose CDN connection hangs without ever timing out -- an observed
  failure mode here, where a socket sat in ``SYN-SENT`` for minutes.  The abort
  uses exit code 3 and keeps partial data, so simply re-running resumes.
* ``--verify`` re-opens every artifact (YAML parse, ``safetensors.safe_open``,
  ``torch.load(weights_only=True)``) so a truncated download is caught here
  rather than in the middle of a training run.

Examples
--------
    python scripts/download_models.py                       # everything
    python scripts/download_models.py --skip-optional       # no RADIO
    python scripts/download_models.py --models dinov2-base  # just one
    python scripts/download_models.py --dry-run             # plan + sizes
    python scripts/download_models.py --verify              # download + check

Exit codes
----------
    0   every requested (non-optional) model is present
    1   a required model failed -- see the summary table
    3   aborted by the stall watchdog; re-run to resume
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "checkpoints"
DEFAULT_ENDPOINT = "https://hf-mirror.com"
DEFAULT_MAX_WORKERS = 4
DEFAULT_RETRIES = 5
DEFAULT_STALL_TIMEOUT = 150.0     # seconds without a single new byte -> abort
EXIT_STALLED = 3                  # distinct code: "hung CDN socket, please re-run"
MANIFEST_NAME = "download_manifest.json"


@dataclass
class ModelSpec:
    """One snapshot we want on local disk."""

    name: str                       # key used by --models / directory name
    repo_id: str                    # HuggingFace repo id
    description: str
    license: str
    optional: bool = False
    # allow_patterns: candidate pattern groups, tried in order; the first group
    # whose *every* pattern matches at least one file in the repo wins (so a
    # group is only chosen when its weights really exist, never config-only).
    # A group of None means "whole repo".
    pattern_groups: List[Optional[List[str]]] = field(default_factory=lambda: [None])
    required_files: Tuple[str, ...] = ()   # must exist after download
    # For repo subfolders: expose the files directly under checkpoints/<name>/
    # (via hardlinks) so consumers do not have to know the upstream layout.
    flatten_prefix: Optional[str] = None

    @property
    def local_dir_name(self) -> str:
        return self.name


# The VAE lives in a subfolder of a repo that also ships the paint pipeline.
# Preference order: fp16 safetensors > fp32 safetensors > fp16 ckpt > fp32 ckpt;
# every group also demands config.yaml so we never end up with weights but no
# architecture description.  Verified against the live repo layout: as of the
# current revision only ``model.fp16.ckpt`` exists, so group 3 is selected.
HUNYUAN_VAE_PATTERNS: List[List[str]] = [
    ["hunyuan3d-vae-v2-1/config.yaml", "hunyuan3d-vae-v2-1/model.fp16.safetensors"],
    ["hunyuan3d-vae-v2-1/config.yaml", "hunyuan3d-vae-v2-1/model.safetensors"],
    ["hunyuan3d-vae-v2-1/config.yaml", "hunyuan3d-vae-v2-1/model.fp16.ckpt"],
    ["hunyuan3d-vae-v2-1/config.yaml", "hunyuan3d-vae-v2-1/model.ckpt"],
    ["hunyuan3d-vae-v2-1/config.yaml", "hunyuan3d-vae-v2-1/*"],  # last resort
]

MODELS: Dict[str, ModelSpec] = {
    "hunyuan3d-vae-v2-1": ModelSpec(
        name="hunyuan3d-vae-v2-1",
        repo_id="tencent/Hunyuan3D-2.1",
        description="Hunyuan3D-2.1 shape VAE (VecSet encoder/decoder), fp16 ckpt",
        license="Tencent Hunyuan3D 2.1 Community License -- NON-COMMERCIAL, "
                "used here only as a numerical-parity reference",
        optional=False,
        pattern_groups=[list(g) for g in HUNYUAN_VAE_PATTERNS],
        required_files=("config.yaml",),
        flatten_prefix="hunyuan3d-vae-v2-1",
    ),
    "dinov2-small": ModelSpec(
        name="dinov2-small",
        repo_id="facebook/dinov2-small",
        description="DINOv2 ViT-S/14 -- 2nd image encoder (dinov2_pair mode)",
        license="Apache-2.0",
        optional=False,
        pattern_groups=[[
            "config.json", "preprocessor_config.json", "model.safetensors",
        ]],
        required_files=("config.json", "model.safetensors"),
    ),
    "dinov2-base": ModelSpec(
        name="dinov2-base",
        repo_id="facebook/dinov2-base",
        description="DINOv2 ViT-B/14 -- main semantic image encoder",
        license="Apache-2.0",
        optional=False,
        pattern_groups=[[
            "config.json", "preprocessor_config.json", "model.safetensors",
        ]],
        required_files=("config.json", "model.safetensors"),
    ),
    "C-RADIOv3-B": ModelSpec(
        name="C-RADIOv3-B",
        repo_id="nvidia/C-RADIOv3-B",
        description="C-RADIOv3-B -- OPTIONAL geometry-aware encoder "
                    "(falls back to dinov2_pair when absent)",
        license="NVIDIA RADIO source-code / model license -- see repo README "
                "(research use; check before redistribution)",
        optional=True,
        # Whole repo: config + remote code + both weight formats
        # (model.safetensors for transformers, *.pth.tar for the RADIO loader).
        pattern_groups=[None],
        required_files=("config.json",),
    ),
}

# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def log(msg: str) -> None:
    print(f"[download_models] {msg}", flush=True)


def human_bytes(n: int) -> str:
    f = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if f < 1024.0 or unit == "TiB":
            return f"{f:,.1f} {unit}" if unit != "B" else f"{int(f)} B"
        f /= 1024.0
    return f"{f:.1f} TiB"


def dir_size(path: Path) -> Tuple[int, int]:
    """(unique bytes, number of files) below *path*, ignoring hf_hub metadata.

    Files are deduplicated by inode: the flat view created by
    :func:`ensure_flat_view` hardlinks the authoritative copies, and counting
    them twice would overstate the disk footprint on every re-run.
    """
    total = 0
    count = 0
    seen: set = set()
    if not path.exists():
        return 0, 0
    for root, dirnames, filenames in os.walk(path):
        # hf_hub bookkeeping (.cache/huggingface/download/*.incomplete, locks)
        dirnames[:] = [d for d in dirnames if d != ".cache"]
        for fn in filenames:
            fp = Path(root) / fn
            try:
                st = fp.stat()
            except OSError:
                continue
            count += 1
            # st_nlink == 1 cannot possibly be a duplicate inode here.
            if st.st_nlink > 1:
                key = (st.st_dev, st.st_ino)
                if key in seen:
                    continue
                seen.add(key)
            total += st.st_size
    return total, count


def bytes_incl_cache(path: Path) -> int:
    """Every byte under *path*, including hf_hub's ``.incomplete`` part files.

    hf_hub streams a download into ``<local_dir>/.cache/huggingface/download/
    *.incomplete`` and only renames it into place when the file is complete, so
    a watchdog has to watch the cache directory to see progress at all.
    """
    total = 0
    if not path.exists():
        return 0
    for root, _dirnames, filenames in os.walk(path):
        for fn in filenames:
            try:
                total += (Path(root) / fn).stat().st_size
            except OSError:
                continue
    return total


class StallWatchdog:
    """Abort a download that stops making progress.

    Large files here are served by the mirror through a CDN: occasionally a
    connection to a CDN edge hangs in ``SYN-SENT`` forever.  hf_hub has no
    timeout for that case, so without a watchdog the script just sits there
    (observed: 0 bytes written over 5+ minutes).  We exit the *process* rather
    than raise, because the stalled socket lives in a worker thread that cannot
    be interrupted; hf_hub has already persisted the partial file, so the next
    invocation resumes instead of restarting.
    """

    def __init__(self, watch_dir: Path, stall_timeout: float, label: str) -> None:
        self.watch_dir = watch_dir
        self.stall_timeout = stall_timeout
        self.label = label
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.peak = 0

    def __enter__(self) -> "StallWatchdog":
        if self.stall_timeout > 0:
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()

    def _run(self) -> None:
        last = bytes_incl_cache(self.watch_dir)
        last_change = time.monotonic()
        # Give the etag/HEAD handshake time before judging progress.
        while not self._stop.wait(min(10.0, max(2.0, self.stall_timeout / 6.0))):
            now = bytes_incl_cache(self.watch_dir)
            self.peak = max(self.peak, now)
            if now != last:
                last, last_change = now, time.monotonic()
                continue
            stalled_for = time.monotonic() - last_change
            if stalled_for >= self.stall_timeout:
                log(f"  !! STALL: no byte written for {stalled_for:.0f}s while "
                    f"downloading {self.label} ({human_bytes(now)} so far).")
                log("  !! the CDN connection is hung; exiting so the caller can "
                    "resume (partial data is kept on disk).")
                # Flush *before* the hard exit, else the diagnostic is lost.
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(EXIT_STALLED)


def load_hf():
    """Import huggingface_hub lazily so ``--help`` works without the env."""
    try:
        from huggingface_hub import HfApi, snapshot_download  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover
        sys.exit(f"huggingface_hub is required but not importable: {exc}")
    return HfApi, snapshot_download


def build_allow_patterns(
    api, spec: ModelSpec, out_dir: Path, max_workers: int
) -> Optional[List[str]]:
    """Pick the first candidate pattern group whose files all exist upstream.

    Hunyuan3D-2.1 ships ``model.fp16.ckpt`` today, but a mirror or a future
    revision may ship ``model.safetensors`` / fp32 instead; this resolves the
    layout at runtime instead of hard-coding it, and -- because every pattern in
    a group must match -- never selects a config-only group by accident.
    """
    import fnmatch

    try:
        repo_files = api.list_repo_files(spec.repo_id)
    except Exception as exc:  # offline-ish / transient: fall back to broad group
        log(f"  ! could not list {spec.repo_id} ({type(exc).__name__}: {exc})")
        log("  ! falling back to the whole repo")
        return None

    log(f"  remote file count for {spec.repo_id}: {len(repo_files)}")
    for group in spec.pattern_groups:
        if group is None:
            log("  pattern: whole repo")
            return None
        matched = [f for f in repo_files
                   if any(fnmatch.fnmatch(f, p) for p in group)]
        missing = [p for p in group
                   if not any(fnmatch.fnmatch(f, p) for f in repo_files)]
        if not missing:
            log(f"  pattern hit: {group}")
            log(f"  will fetch {len(matched)} file(s), e.g. {sorted(matched)[:4]}")
            return list(group)
        log(f"  pattern miss ({missing[:2]} absent) -> trying next candidate")
    log(f"  ! no pattern group matched {spec.repo_id}; downloading whole repo")
    return None


def download_one(
    spec: ModelSpec,
    out_root: Path,
    *,
    allow_patterns: Optional[Sequence[str]],
    max_workers: int,
    retries: int,
    dry_run: bool,
    api,
    snapshot_download,
    stall_timeout: float = DEFAULT_STALL_TIMEOUT,
) -> Dict[str, object]:
    """Download a single spec with bounded retries. Never raises."""
    target = out_root / spec.local_dir_name
    target.mkdir(parents=True, exist_ok=True)
    record: Dict[str, object] = {
        "name": spec.name,
        "repo_id": spec.repo_id,
        "description": spec.description,
        "license": spec.license,
        "optional": spec.optional,
        "local_path": str(target),
        "allow_patterns": list(allow_patterns) if allow_patterns else None,
        "status": "pending",
        "attempts": 0,
        "error": None,
        "bytes": 0,
        "files": 0,
    }

    if dry_run:
        record["status"] = "planned"
        return record

    last_err: Optional[BaseException] = None
    for attempt in range(1, retries + 1):
        record["attempts"] = attempt
        try:
            if attempt == 1:
                log(f"downloading {spec.repo_id} -> {target}")
            else:
                log(f"retry {attempt}/{retries} for {spec.repo_id}")
            with StallWatchdog(target, stall_timeout, spec.name):
                snapshot_download(
                    repo_id=spec.repo_id,
                    local_dir=str(target),
                    allow_patterns=list(allow_patterns) if allow_patterns else None,
                    max_workers=max_workers,
                    # local_dir_dedup is intentionally left off: we want plain
                    # files in checkpoints/<name>/ that any tool can read.
                )
            last_err = None
            break
        except KeyboardInterrupt:
            record["status"] = "interrupted"
            record["error"] = "KeyboardInterrupt"
            raise
        except Exception as exc:  # noqa: BLE001 - report, retry, keep going
            last_err = exc
            log(f"  ! attempt {attempt}/{retries} failed: {type(exc).__name__}: {exc}")
            if attempt < retries:
                # Partial files stay on disk; hf_hub resumes from them.
                time.sleep(min(30, 2 ** attempt))

    nbytes, nfiles = dir_size(target)
    record["bytes"], record["files"] = nbytes, nfiles

    if last_err is not None:
        record["status"] = "failed"
        record["error"] = f"{type(last_err).__name__}: {last_err}"
        return record

    missing = [f for f in spec.required_files
               if not any(target.rglob(Path(f).name))]
    if missing and not any(target.rglob("*")):
        record["status"] = "empty"
        record["error"] = "snapshot_download returned but the directory is empty"
        return record
    if missing:
        record["status"] = "partial"
        record["error"] = f"expected file(s) not found: {missing}"
        return record

    record["status"] = "ok"
    return record


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #


def ensure_flat_view(spec: ModelSpec, target: Path) -> List[str]:
    """Expose files from an upstream subfolder directly under ``checkpoints/<name>/``.

    ``snapshot_download`` mirrors the repo layout, so the VAE would sit at
    ``checkpoints/hunyuan3d-vae-v2-1/hunyuan3d-vae-v2-1/model.fp16.ckpt``.  We
    keep that authoritative copy (it is what hf_hub resumes into) and add a
    *hardlink* at the top level, which costs no extra disk space and keeps the
    canonical path short: ``checkpoints/hunyuan3d-vae-v2-1/model.fp16.ckpt``.
    """
    created: List[str] = []
    if not spec.flatten_prefix:
        return created
    src_dir = target / spec.flatten_prefix
    if not src_dir.is_dir():
        return created
    for src in sorted(p for p in src_dir.rglob("*") if p.is_file()):
        dst = target / src.relative_to(src_dir)
        if dst.exists():
            # Already materialised by an earlier run: still report it, so the
            # manifest always documents the canonical short paths.
            created.append(str(dst.relative_to(target)))
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)                      # same inode: no extra space
        except OSError:
            try:
                dst.symlink_to(os.path.relpath(src, dst.parent))
            except OSError:
                import shutil

                shutil.copy2(src, dst)
        created.append(str(dst.relative_to(target)))
    if created:
        log(f"  flat view: {', '.join(created)}")
    return created


def verify_artifact(path: Path) -> Tuple[bool, str]:
    """Open one artifact the way a consumer would. Returns (ok, message)."""
    name = path.name.lower()
    try:
        if name.endswith((".yaml", ".yml")):
            import yaml  # type: ignore

            with path.open("r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh)
            top = list(cfg)[:6] if isinstance(cfg, dict) else type(cfg).__name__
            return True, f"yaml parsed, top-level keys: {top}"
        if name.endswith(".json"):
            with path.open("r", encoding="utf-8") as fh:
                json.load(fh)
            return True, "json parsed"
        if name.endswith(".safetensors"):
            from safetensors import safe_open

            with safe_open(str(path), framework="pt", device="cpu") as fh:
                keys = list(fh.keys())
                n = len(keys)
                sample = keys[0] if keys else "-"
                shape = tuple(fh.get_slice(sample).get_shape()) if keys else ()
            return True, f"{n} tensors, first={sample}{list(shape)}"
        if name.endswith(".ckpt") or name.endswith(".bin") or name.endswith(".pth.tar"):
            import torch

            try:
                obj = torch.load(str(path), map_location="cpu", weights_only=True)
            except Exception as exc:  # noqa: BLE001
                # Full training checkpoints (e.g. RADIO's *.pth.tar) embed
                # argparse.Namespace / other globals and *cannot* be unpickled
                # in weights_only mode.  That is not corruption: fall back to a
                # structural check of the zip archive instead of executing
                # arbitrary pickle code just to "verify" a download.
                if "Weights only load failed" in str(exc) or "WeightsUnpickler" in str(exc):
                    import zipfile

                    with zipfile.ZipFile(path) as zf:
                        bad = zf.testzip()
                        names = zf.namelist()
                    if bad is not None:
                        return False, f"corrupt zip entry: {bad}"
                    return True, (f"archive intact ({len(names)} entries); "
                                  "weights_only load refused (pickled non-tensor "
                                  "globals) -- expected for a training checkpoint")
                raise
            if isinstance(obj, dict):
                # Lightning / hy3d checkpoints nest the weights under 'state_dict'
                for key in ("state_dict", "model", "module"):
                    if key in obj and isinstance(obj[key], dict):
                        obj = obj[key]
                        break
                keys = list(obj.keys())
                return True, f"torch.load ok, {len(keys)} keys, first={keys[0] if keys else '-'}"
            return True, f"torch.load ok, type={type(obj).__name__}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    return True, "no check performed for this extension"


def run_verify(records: Sequence[Dict[str, object]], deep: bool = True) -> List[Dict[str, object]]:
    log("verifying artifacts ...")
    results: List[Dict[str, object]] = []
    for rec in records:
        if rec["status"] not in {"ok", "partial"}:
            continue
        root = Path(str(rec["local_path"]))
        files = sorted(p for p in root.rglob("*") if p.is_file()
                       and ".cache" not in p.parts)
        weight_files = [p for p in files
                        if p.suffix in {".safetensors", ".ckpt", ".bin"}
                        or p.name.endswith(".pth.tar")]
        conf_files = [p for p in files if p.suffix in {".yaml", ".yml", ".json"}]
        checks: List[Tuple[str, bool, str]] = []
        # The flat-view hardlinks point at the same inodes as the authoritative
        # copies; verifying them twice would just re-read 650MB for nothing.
        seen_inodes: Dict[Tuple[int, int], str] = {}

        def _check(p: Path) -> None:
            rel = str(p.relative_to(root))
            try:
                st = p.stat()
                key = (st.st_dev, st.st_ino)
            except OSError:
                key = None  # type: ignore[assignment]
            if key is not None and key in seen_inodes:
                checks.append((rel, True, f"hardlink of {seen_inodes[key]} (already checked)"))
                return
            if key is not None:
                seen_inodes[key] = rel
            checks.append((rel, *verify_artifact(p)))

        for p in conf_files:
            _check(p)
        # Only deep-open weights on request: loading 650MB+ costs real time.
        for p in weight_files:
            if deep:
                _check(p)
            else:
                checks.append((str(p.relative_to(root)), True,
                               "exists (skipped deep check)"))
        rec["verify"] = [
            {"file": f, "ok": ok, "detail": msg} for f, ok, msg in checks
        ]
        # a config-only failure still counts as a broken artifact
        rec["verified"] = all(ok for _, ok, _ in checks) if checks else False
        for f, ok, msg in checks:
            log(f"  [{'ok ' if ok else 'FAIL'}] {rec['name']}/{f}: {msg}")
        results.append(rec)
    return results


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def print_summary(records: Sequence[Dict[str, object]], out_root: Path) -> None:
    name_w = max([len(str(r["name"])) for r in records] + [12])
    print()
    print("=" * 100)
    print("DOWNLOAD SUMMARY")
    print("=" * 100)
    header = (f"{'model':<{name_w}}  {'status':<9} {'size':>12}  {'files':>6}  "
              f"{'local path'}")
    print(header)
    print("-" * 100)
    total = 0
    for rec in records:
        size = int(rec.get("bytes") or 0)
        total += size
        print(f"{str(rec['name']):<{name_w}}  {str(rec['status']):<9} "
              f"{human_bytes(size):>12}  {int(rec.get('files') or 0):>6}  "
              f"{rec['local_path']}")
        err = rec.get("error")
        if err:
            print(f"{'':<{name_w}}  -> {err}")
    print("-" * 100)
    print(f"{'TOTAL':<{name_w}}  {'':<9} {human_bytes(total):>12}")
    print("=" * 100)
    print(f"checkpoints root: {out_root}")
    print()


def write_manifest(records: Sequence[Dict[str, object]], out_root: Path) -> Path:
    """Write/merge the machine-readable manifest of local checkpoints.

    Runs are often split per model (e.g. fetch the VAE first, RADIO later), so
    entries for models *not* part of this run are preserved instead of being
    clobbered -- downstream tooling can rely on one complete manifest.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / MANIFEST_NAME

    merged: Dict[str, Dict[str, object]] = {}
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fh:
                old = json.load(fh)
            for rec in old.get("models", []):
                if isinstance(rec, dict) and rec.get("name"):
                    merged[str(rec["name"])] = rec
        except Exception as exc:  # noqa: BLE001 - a corrupt manifest must not be fatal
            log(f"  ! existing manifest unreadable ({exc}); rewriting from scratch")

    for rec in records:
        merged[str(rec["name"])] = dict(rec)

    ordered = sorted(merged.values(), key=lambda r: str(r.get("name")))
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        "checkpoints_root": str(out_root),
        "total_bytes": sum(int(r.get("bytes") or 0) for r in ordered),
        "models": ordered,
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download (resumably) the checkpoints used by the "
                    "Seed3D-1.0 mini reproduction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--models", default="all",
        help="comma-separated subset of: " + ", ".join(MODELS) + " (default: all)",
    )
    p.add_argument("--out", default=str(DEFAULT_OUT),
                   help=f"output directory (default: {DEFAULT_OUT})")
    p.add_argument("--skip-optional", action="store_true",
                   help="skip optional models (currently: C-RADIOv3-B)")
    p.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT),
                   help=f"HF endpoint (default: {DEFAULT_ENDPOINT})")
    p.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
                   help=f"parallel download workers (default: {DEFAULT_MAX_WORKERS})")
    p.add_argument("--retries", type=int, default=DEFAULT_RETRIES,
                   help=f"attempts per model (default: {DEFAULT_RETRIES})")
    p.add_argument("--stall-timeout", type=float, default=DEFAULT_STALL_TIMEOUT,
                   help="abort (exit code 3) if a download writes no bytes for "
                        f"this many seconds; 0 disables (default: {DEFAULT_STALL_TIMEOUT:.0f})")
    p.add_argument("--verify", action="store_true",
                   help="re-open every downloaded artifact after downloading")
    p.add_argument("--no-deep-verify", action="store_true",
                   help="with --verify: only stat weight files instead of loading them")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve remote layout and print the plan without downloading")
    return p.parse_args(argv)


def resolve_models(arg: str, skip_optional: bool) -> List[ModelSpec]:
    if arg.strip().lower() in {"all", "*", ""}:
        names = list(MODELS)
    else:
        names = [n.strip() for n in arg.split(",") if n.strip()]
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        sys.exit(f"unknown model(s): {unknown}\nvalid names: {', '.join(MODELS)}")
    specs = [MODELS[n] for n in names]
    if skip_optional:
        specs = [s for s in specs if not s.optional]
    return specs


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    # Mirror first: everything below (including transformers at import time)
    # must see it, and an explicit env var still wins.
    os.environ["HF_ENDPOINT"] = args.endpoint
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    # 60s socket read timeout: catches a half-open connection faster than the
    # 10-minute default, on top of the byte-progress watchdog below.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

    out_root = Path(args.out).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    specs = resolve_models(args.models, args.skip_optional)
    if not specs:
        log("nothing to do (all selected models are optional and --skip-optional was given)")
        return 0

    HfApi, snapshot_download = load_hf()

    log(f"endpoint      : {os.environ['HF_ENDPOINT']}")
    log(f"output root   : {out_root}")
    log(f"HF_HOME       : {os.environ.get('HF_HOME', '(default ~/.cache/huggingface)')}")
    log(f"models        : {', '.join(s.name for s in specs)}")
    log(f"max_workers   : {args.max_workers}   retries/model: {args.retries}")
    log(f"stall timeout : {args.stall_timeout:.0f}s without progress -> exit "
        f"{EXIT_STALLED} (0 = disabled)")

    api = HfApi(endpoint=os.environ["HF_ENDPOINT"])

    records: List[Dict[str, object]] = []
    exit_code = 0
    for spec in specs:
        log("-" * 78)
        log(f"model: {spec.name}  ({spec.repo_id})")
        log(f"  purpose: {spec.description}")
        if spec.optional:
            log("  OPTIONAL: failure is reported but does not fail the run")
        try:
            patterns = build_allow_patterns(api, spec, out_root, args.max_workers)
        except Exception as exc:  # noqa: BLE001
            log(f"  ! planning failed ({exc}); falling back to whole repo")
            patterns = None
        try:
            rec = download_one(
                spec, out_root,
                allow_patterns=patterns,
                max_workers=args.max_workers,
                retries=args.retries,
                dry_run=args.dry_run,
                api=api,
                snapshot_download=snapshot_download,
                stall_timeout=args.stall_timeout,
            )
        except KeyboardInterrupt:
            log("interrupted by user; partial files are kept and will resume")
            records.append({
                "name": spec.name, "repo_id": spec.repo_id, "status": "interrupted",
                "local_path": str(out_root / spec.local_dir_name), "bytes": 0,
                "files": 0, "optional": spec.optional,
                "error": "KeyboardInterrupt",
            })
            print_summary(records, out_root)
            return 130
        records.append(rec)
        if rec["status"] == "ok":
            rec["flat_view"] = ensure_flat_view(spec, out_root / spec.local_dir_name)
            log(f"  -> ok: {human_bytes(int(rec['bytes']))} in {rec['files']} files")
        else:
            log(f"  -> {rec['status']}: {rec.get('error')}")
            if not spec.optional:
                exit_code = 1

    if args.verify and not args.dry_run:
        run_verify(records, deep=not args.no_deep_verify)
        for rec in records:
            if rec.get("verify") and not rec.get("verified") and not rec["optional"]:
                exit_code = 1

    print_summary(records, out_root)
    if not args.dry_run:
        manifest = write_manifest(records, out_root)
        log(f"manifest written: {manifest}")

    # Optional models never fail the process; the summary says what happened.
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
