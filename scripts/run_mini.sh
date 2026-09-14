#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-shot mini training run: VAE -> latents -> DiT, on N RTX3090s.
#
#   4 GPUs, ~24h budget:
#       bash scripts/run_mini.sh --gpus 0,1,2,3 --vae-steps 20000 --dit-steps 20000
#
# Stages are resumable: re-running picks up the latest checkpoint of each stage.
# Override anything via environment variables (see defaults below).
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPUS="${GPUS:-0,1,2,3}"
VAE_STEPS="${VAE_STEPS:-20000}"
DIT_STEPS="${DIT_STEPS:-20000}"
VAE_BS="${VAE_BS:-8}"
DIT_BS="${DIT_BS:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PY="${PY:-$ROOT/envs/seed3d/bin/python}"
LOGDIR="${LOGDIR:-$ROOT/artifacts/logs}"
DATA_SHARDS="${DATA_SHARDS:-$ROOT/data/shards}"
LATENT_SHARDS="${LATENT_SHARDS:-$ROOT/data/latents_mini}"
VAE_DIR="${VAE_DIR:-$ROOT/checkpoints/vae_mini}"
DIT_DIR="${DIT_DIR:-$ROOT/checkpoints/dit_mini}"

# arguments: --gpus 0,1,2,3 --vae-steps N --dit-steps N --skip-data --skip-vae ...
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2;;
    --vae-steps) VAE_STEPS="$2"; shift 2;;
    --dit-steps) DIT_STEPS="$2"; shift 2;;
    --vae-batch-size) VAE_BS="$2"; shift 2;;
    --dit-batch-size) DIT_BS="$2"; shift 2;;
    --skip-data) SKIP_DATA=1; shift;;
    --skip-vae) SKIP_VAE=1; shift;;
    --skip-dit) SKIP_DIT=1; shift;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done

NPROC="$(awk -F, '{print NF}' <<<"$GPUS")"
export CUDA_VISIBLE_DEVICES="$GPUS"
mkdir -p "$LOGDIR" "$VAE_DIR" "$DIT_DIR"

echo "=== mini Seed3D run ==="
echo "GPUs          : $GPUS ($NPROC processes)"
echo "VAE steps     : $VAE_STEPS (batch $VAE_BS/gpu)"
echo "DiT steps     : $DIT_STEPS (batch $DIT_BS/gpu)"
echo "python        : $PY"
"$PY" - <<'EOF'
import torch
print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()} | devices {torch.cuda.device_count()}")
EOF

# ---------------------------------------------------------------- 1. data ---
if [[ -z "${SKIP_DATA:-}" ]]; then
  if ! ls "$DATA_SHARDS"/train-*.npz >/dev/null 2>&1; then
    echo "--- stage 1/4: prepare data"
    "$PY" scripts/prepare_data.py download --categories airplane,car,ship --splits train --workers 8 \
        2>&1 | tee "$LOGDIR/prepare_download.log"
    "$PY" scripts/prepare_data.py process --categories airplane,car,ship --workers "$NUM_WORKERS" \
        2>&1 | tee "$LOGDIR/prepare_process.log"
    "$PY" scripts/prepare_data.py shards --categories airplane,car,ship 2>&1 | tee "$LOGDIR/prepare_shards.log"
    "$PY" scripts/prepare_data.py stats
  else
    echo "--- stage 1/4: shards already present, skipping data prep"
  fi
fi

# ----------------------------------------------------------------- 2. VAE ---
if [[ -z "${SKIP_VAE:-}" ]]; then
  echo "--- stage 2/4: train Seed3D-VAE"
  RESUME=""
  if [[ -f "$VAE_DIR/vae_latest.pt" ]]; then RESUME="--resume $VAE_DIR/vae_latest.pt"; fi
  "$PY" -m torch.distributed.run --nproc_per_node="$NPROC" --master_port=29511 \
      scripts/train_vae.py --config configs/vae_mini.yaml \
      --steps "$VAE_STEPS" --batch-size "$VAE_BS" $RESUME \
      2>&1 | tee "$LOGDIR/train_vae.log"
fi

# -------------------------------------------------------------- 3. latents ---
if [[ -z "${SKIP_DIT:-}" ]]; then
  if ! ls "$LATENT_SHARDS"/train-*.npz >/dev/null 2>&1; then
    echo "--- stage 3/4: pack VAE latents"
    "$PY" scripts/pack_latents.py --vae "$VAE_DIR/vae_final.pt" --config configs/vae_mini.yaml \
        --out "$LATENT_SHARDS" --batch-size 16 --device cuda \
        2>&1 | tee "$LOGDIR/pack_latents.log"
  else
    echo "--- stage 3/4: latents already packed"
  fi

  # ------------------------------------------------------------- 4. DiT ----
  echo "--- stage 4/4: train Seed3D-DiT"
  RESUME=""
  if [[ -f "$DIT_DIR/dit_latest.pt" ]]; then RESUME="--resume $DIT_DIR/dit_latest.pt"; fi
  "$PY" -m torch.distributed.run --nproc_per_node="$NPROC" --master_port=29512 \
      scripts/train_dit.py --config configs/dit_mini.yaml \
      --steps "$DIT_STEPS" --batch-size "$DIT_BS" $RESUME \
      2>&1 | tee "$LOGDIR/train_dit.log"
fi

echo "=== done: checkpoints in $VAE_DIR and $DIT_DIR ==="
ls -la "$VAE_DIR" "$DIT_DIR" || true
