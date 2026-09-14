#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Seed3D-1.0 reproduction -- HuggingFace / model-cache environment
#
# Source this file from any script that touches HuggingFace:
#
#     source "$(dirname "$0")/env_hf.sh"
#
# Rationale: huggingface.co is NOT directly reachable from this machine, while
# the mirror https://hf-mirror.com works.  Setting HF_ENDPOINT makes both
# huggingface_hub (snapshot_download / hf_hub_download) and transformers
# (AutoModel.from_pretrained) resolve through the mirror.
# ---------------------------------------------------------------------------

# --- mirror endpoint -------------------------------------------------------
# Respect a pre-existing override, otherwise default to the working mirror.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# --- repository root -------------------------------------------------------
# Works both when sourced (`source scripts/env_hf.sh`) and when executed.
if [ -n "${BASH_SOURCE[0]:-}" ]; then
  _SEED3D_ENV_HF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
  _SEED3D_ENV_HF_DIR="$(pwd)"
fi
export SEED3D_ROOT="${SEED3D_ROOT:-$(cd "${_SEED3D_ENV_HF_DIR}/.." && pwd)}"
unset _SEED3D_ENV_HF_DIR

# --- caches -----------------------------------------------------------------
# Single HF_HOME keeps the hub cache (snapshots/downloads) and any
# transformers cache in one place, inside the repo, so the working tree is
# self-contained and portable.
export HF_HOME="${HF_HOME:-${SEED3D_ROOT}/checkpoints/.hf_home}"
# `local_dir` downloads performed by scripts/download_models.py land here:
export SEED3D_CKPT_DIR="${SEED3D_CKPT_DIR:-${SEED3D_ROOT}/checkpoints}"

# --- offline / robustness knobs --------------------------------------------
# The mirror is reached over plain HTTPS; disable the optional Xet transfer
# path, which is not served by the mirror.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
# Keep the per-file HTTP timeout generous: >1GB checkpoints on the mirror can
# be slow, and hf_hub retries on top of this.
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"

# --- python interpreter -----------------------------------------------------
if [ -x "${SEED3D_ROOT}/envs/seed3d/bin/python" ]; then
  export SEED3D_PYTHON="${SEED3D_PYTHON:-${SEED3D_ROOT}/envs/seed3d/bin/python}"
else
  export SEED3D_PYTHON="${SEED3D_PYTHON:-python3}"
fi

mkdir -p "${HF_HOME}" "${SEED3D_CKPT_DIR}" 2>/dev/null || true

# NOTE: deliberately no `set -e` / no output -- this file is meant to be
# sourced, so it must not change the caller's shell options.
