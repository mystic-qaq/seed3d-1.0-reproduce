"""pytest configuration for the ``tests/`` directory.

Two jobs:

* make ``import seed3d_mini`` work when pytest is invoked from anywhere
  (``src/`` is not installed into the environment),
* expose the ``--skip-if-missing-ckpt`` switch used by
  :mod:`tests.test_vae_parity`, and dump the parity table at the end of the
  session so ``artifacts/logs/vae_parity.txt`` is always up to date.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
for _p in (str(SRC_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def pytest_addoption(parser):
    group = parser.getgroup("seed3d")
    group.addoption(
        "--skip-if-missing-ckpt",
        action="store_true",
        default=False,
        help="skip (instead of failing) the parity tests that need the "
        "Hunyuan3D-2.1 VAE checkpoint under checkpoints/",
    )
    group.addoption(
        "--parity-reduced-size",
        action="store_true",
        default=False,
        help="run the encoder parity checks at a reduced size "
        "(256 latents x 5120 points) instead of the full reference size",
    )


def pytest_configure(config):
    """Bridge the CLI switches to the test module (which is also a plain script)."""
    if config.getoption("--skip-if-missing-ckpt"):
        os.environ["SEED3D_PARITY_SKIP_IF_MISSING"] = "1"
    if config.getoption("--parity-reduced-size"):
        os.environ["SEED3D_PARITY_REDUCED_SIZE"] = "1"


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    """Write/refresh the parity report at the end of the session."""
    try:
        import importlib

        mod = importlib.import_module("test_vae_parity")
    except Exception:  # pragma: no cover - best effort only
        try:
            mod = importlib.import_module("tests.test_vae_parity")
        except Exception:
            return
    try:
        mod.save_report()
    except Exception:
        pass
