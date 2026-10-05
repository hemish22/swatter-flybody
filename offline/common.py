"""Shared config and the local-vs-DGX execution guard.

Standing rule for this repo: heavy work runs on the DGX, not on a laptop.
The expensive steps are the ones that touch the full MaleCNS tables (1.05 GB of
edges, 311.8M synapses) or sweep thousands of LIF trials. Those need the RAM,
the A100, and the wall-clock budget of the machine in the compute plan, and
running them on a laptop is slow enough that people start skipping them.

So the heavy entry points call `require_remote_execution()` before doing any
work. Local runs are still fine for everything cheap: generating stimuli,
linting, checking the manifest, unit tests. That split is what the Makefile
encodes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Set by scripts/dgx.sh when it launches a job on the remote machine.
REMOTE_ENV = "SWATTER_REMOTE"
# Escape hatch for the rare local run that is genuinely cheap, e.g. re-running
# extract_subgraph --report-only against data that is already local.
LOCAL_ENV = "SWATTER_ALLOW_LOCAL_HEAVY"

DGX_HOST_ENV = "SWATTER_DGX_HOST"
DEFAULT_DGX_HOST = "h4hgpu"
REMOTE_DIR_ENV = "SWATTER_REMOTE_DIR"
DEFAULT_REMOTE_DIR = "~/swatter-flybody"


def data_dir(explicit: Path | None = None) -> Path:
    """Where the big tables live. Override with SWATTER_DATA on the DGX."""
    if explicit is not None:
        return explicit
    return Path(os.environ.get("SWATTER_DATA", "data")).expanduser()


def dgx_host() -> str:
    return os.environ.get(DGX_HOST_ENV, DEFAULT_DGX_HOST)


def remote_dir() -> str:
    return os.environ.get(REMOTE_DIR_ENV, DEFAULT_REMOTE_DIR)


def on_remote() -> bool:
    return os.environ.get(REMOTE_ENV) == "1"


def local_allowed() -> bool:
    return os.environ.get(LOCAL_ENV) == "1"


def require_remote_execution(script: str, cost: str) -> None:
    """Refuse to start an expensive step on a laptop.

    `cost` should say what the step actually spends, so the message is a real
    decision aid rather than a policy statement.
    """
    if on_remote() or local_allowed():
        return

    host = dgx_host()
    print(
        f"\n{script}: refusing to run this on your laptop.\n"
        f"\n"
        f"  cost: {cost}\n"
        f"\n"
        f"  run it on the DGX instead:\n"
        f"      ./scripts/dgx.sh run {script.removesuffix('.py')}\n"
        f"\n"
        f"  the DGX host defaults to {host!r}; override with {DGX_HOST_ENV}=your-host.\n"
        f"\n"
        f"  if this step is genuinely cheap and you want it anyway:\n"
        f"      {LOCAL_ENV}=1 .venv/bin/python {script}\n",
        file=sys.stderr,
    )
    raise SystemExit(2)


def banner(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}", flush=True)
