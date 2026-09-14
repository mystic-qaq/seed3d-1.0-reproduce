"""End-to-end script-level integration test (synthetic data, no downloads).

Exercises exactly the code paths the real run uses, on tiny synthetic shards:

    prepare->shards   (shard packing)
    train_vae.py      (2 steps, saves a checkpoint)
    pack_latents.py   (frozen-VAE latent packing)
    train_dit.py      (2 steps, saves a checkpoint)
    infer.py          (image -> latent -> TSDF -> mesh)
    evaluate.py       (metrics JSON)

It is deliberately a *script* (not a unit test) because it shells out to the
CLIs: the point is to catch interface drift between the stages -- config keys,
file layouts, tensor shapes -- which unit tests on the modules cannot see.

    python tests/test_integration.py [--gpu 4]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / "envs" / "seed3d" / "bin" / "python"


def run(cmd: list[str], env: dict, log: Path) -> None:
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    with log.open("w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
    if proc.returncode != 0:
        print(log.read_text()[-3000:])
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(str(c) for c in cmd)}")


def make_shards(dir_: Path, n: int = 12) -> None:
    rng = np.random.default_rng(0)
    surf = rng.uniform(-0.8, 0.8, (n, 512, 3)).astype(np.float32)
    nrm = rng.normal(size=(n, 512, 3)).astype(np.float32)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    edge = rng.uniform(-0.8, 0.8, (n, 64, 3)).astype(np.float32)
    enorm = rng.normal(size=(n, 64, 3)).astype(np.float32)
    enorm /= np.linalg.norm(enorm, axis=-1, keepdims=True)
    q = rng.uniform(-1, 1, (n, 512, 3)).astype(np.float32)
    sdf = (np.linalg.norm(q, axis=-1, keepdims=True) - 0.5).clip(-0.1, 0.1).astype(np.float32)
    w = np.ones((n, 512, 1), np.float32)
    img = (rng.random((n, 3, 128, 128)) * 255).astype(np.uint8)
    dir_.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        np.savez(
            dir_ / f"{split}-00000.npz",
            surface=surf, surface_normal=nrm, edge=edge, edge_normal=enorm,
            query=q, sdf=sdf, weight=w, image=img,
            category=np.array(["airplane"] * n), sample_id=np.array([f"s{i}" for i in range(n)]),
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="4")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="seed3d_integration_"))
    env = {**__import__("os").environ, "CUDA_VISIBLE_DEVICES": args.gpu, "PYTHONUNBUFFERED": "1"}
    logs = ROOT / "artifacts" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    try:
        shard_dir = tmp / "shards"
        make_shards(shard_dir)
        print(f"workdir: {tmp}")

        # ---- configs (tiny versions of the shipped ones) ------------------ #
        vae_cfg = yaml.safe_load((ROOT / "configs" / "vae_mini.yaml").read_text())
        vae_cfg["data"]["shard_dir"] = str(shard_dir)
        vae_cfg["data"].update({"pc_size": 256, "pc_sharpedge_size": 64, "num_queries": 256})
        vae_cfg["model"].update(
            {
                "width": 128, "heads": 4, "num_encoder_layers": 2, "num_decoder_layers": 1,
                "pc_size": 256, "pc_sharpedge_size": 64, "num_latents": 64,
                "train_num_latents": [32, 64],
            }
        )
        vae_cfg["training"].update(
            {
                "steps": 3, "batch_size": 2, "num_workers": 0, "log_every": 1, "save_every": 2,
                "warmup_steps": 1, "kl_warmup_steps": 2, "eikonal_warmup_steps": 2,
                "val_every": 2, "output_dir": str(tmp / "vae"),
            }
        )
        vae_yaml = tmp / "vae.yaml"
        vae_yaml.write_text(yaml.safe_dump(vae_cfg))

        dit_cfg = yaml.safe_load((ROOT / "configs" / "dit_mini.yaml").read_text())
        dit_cfg["dit"].update(
            {"hidden_size": 128, "num_heads": 4, "depth": 2, "depth_single_blocks": 2,
             "num_latents": 64, "in_channels": 8}
        )
        dit_cfg["conditioner"].update(
            {"main_version": str(ROOT / "checkpoints" / "dinov2-small"),
             "additional_version": str(ROOT / "checkpoints" / "dinov2-small"),
             "trust_remote_code": False}
        )
        dit_cfg["data"].update({"latent_shard_dir": str(tmp / "latents"), "image_size": 128, "max_latents": 64})
        dit_cfg["flow_matching"]["base_tokens"] = 32
        dit_cfg["training"].update(
            {"steps": 2, "batch_size": 2, "num_workers": 0, "log_every": 1, "save_every": 1,
             "warmup_steps": 1, "output_dir": str(tmp / "dit")}
        )
        dit_yaml = tmp / "dit.yaml"
        dit_yaml.write_text(yaml.safe_dump(dit_cfg))

        # ---- stage 1: VAE ------------------------------------------------- #
        run([str(PY), "scripts/train_vae.py", "--config", str(vae_yaml)], env, logs / "it_vae.log")
        assert (tmp / "vae" / "vae_final.pt").exists()

        # ---- stage 2: latents --------------------------------------------- #
        run(
            [str(PY), "scripts/pack_latents.py", "--vae", str(tmp / "vae" / "vae_final.pt"),
             "--config", str(vae_yaml), "--shard-dir", str(shard_dir), "--out", str(tmp / "latents"),
             "--batch-size", "2", "--num-latents", "64"],
            env, logs / "it_pack.log",
        )
        assert (tmp / "latents" / "train-00000.npz").exists()

        # ---- stage 3: DiT -------------------------------------------------- #
        run([str(PY), "scripts/train_dit.py", "--config", str(dit_yaml)], env, logs / "it_dit.log")
        assert (tmp / "dit" / "dit_final.pt").exists()

        # ---- stage 4: inference ------------------------------------------- #
        run(
            [str(PY), "scripts/infer.py", "--vae-config", str(vae_yaml), "--vae", str(tmp / "vae" / "vae_final.pt"),
             "--dit-config", str(dit_yaml), "--dit", str(tmp / "dit" / "dit_final.pt"),
             "--image", "auto", "--mesh-from-shard", str(shard_dir / "val-00000.npz"), "--index", "0",
             "--steps", "3", "--guidance", "2.0", "--resolution", "48", "--coarse", "16",
             "--out", str(tmp / "mesh" / "infer.glb")],
            env, logs / "it_infer.log",
        )
        meta = json.loads((tmp / "mesh" / "infer.json").read_text())
        assert meta["vertices"] > 0 and meta["faces"] > 0, meta

        # ---- stage 5: evaluation ------------------------------------------ #
        run(
            [str(PY), "scripts/evaluate.py", "--level", "vae", "--vae-config", str(vae_yaml),
             "--vae", str(tmp / "vae" / "vae_final.pt"), "--shard", str(shard_dir / "val-00000.npz"),
             "--n", "1", "--resolution", "48", "--coarse", "16", "--out", str(tmp / "eval")],
            env, logs / "it_eval.log",
        )
        summary = json.loads((tmp / "eval" / "eval_vae.json").read_text())["summary"]
        assert summary["n"] == 1, summary

        print("\nINTEGRATION OK: train_vae -> pack_latents -> train_dit -> infer -> evaluate")
        print(json.dumps({"infer": meta, "eval": {k: summary[k] for k in ("chamfer_l1", "fscore_1pct", "watertight_rate")}}, indent=1))
        return 0
    finally:
        if args.keep:
            print(f"kept {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
