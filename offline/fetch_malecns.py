"""Fetch the MaleCNS v1.0 connectome tables SWATTER needs.

Access route (verified Week 0, 2026-10-03): the flat connectome lives in a public
Google Cloud Storage bucket that serves anonymously over HTTPS, so no neuPrint
token is required for the offline build.

    https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/

neuPrint (`male-cns:v1.0`, dataset UUID 4b2087c0fbe046bfaf0d60bc970e3e5d) does
exist and needs a token, but it is only used as a fallback for the optic-lobe
hex column assignments.

Tables and why SWATTER needs them:

  body-annotations   211,577 bodies / 166,700 with a superclass. Gives `type`
                     (LPLC2, LC4, DNp01, ...), `instance`, `somaSide`.
  body-neurotransmitters
                     predicted NT per body -> the sign of each synapse. Critical:
                     glutamate is *inhibitory* in insects.
  connectome-weights 151,856,684 aggregate edges, weights are synapse counts and
                     sum to 311,833,243 synapses. Sorted by weight descending,
                     which makes thresholded streaming cheap.

Usage:
    python offline/fetch_malecns.py            # everything
    python offline/fetch_malecns.py --group core   # ~1.1 GB, the three tables
    python offline/fetch_malecns.py --group small  # ~57 MB, annotations + NT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import data_dir as default_data_dir, require_remote_execution  # noqa: E402

BUCKET = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"

# The hex column assignment spreadsheet is not in GCS; it ships with the
# MaleCNS supplement repo. Pinned to the commit fly.ai uses.
OPTIC_COLUMNS_URL = (
    "https://raw.githubusercontent.com/flyconnectome/2025malecns/"
    "67767d2233657983993ff6c2be48e836a935863c/supplemental_data/"
    "optic-column-type-assignments-v1.0.xlsx"
)

ANNOTATIONS = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS = "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"

FEATHER = {
    ANNOTATIONS,
    NEUROTRANSMITTERS,
    WEIGHTS,
}

GROUPS: dict[str, list[str]] = {
    "small": [ANNOTATIONS, NEUROTRANSMITTERS],
    "core": [ANNOTATIONS, NEUROTRANSMITTERS, WEIGHTS],
}

# Not downloaded by default: both are per-synapse (rather than per-edge) and are
# an order of magnitude larger than anything the offline build needs.
AVOID = """syn-partners (6.8 GB) is connectome-weights exploded one row per synapse;
body-stats (778 MB) is at supervoxel granularity and will not join to
bodyId. Use connectome-weights instead unless you need synapse coordinates."""


def url_for(name: str) -> str:
    if name == "optic-columns.xlsx":
        return OPTIC_COLUMNS_URL
    return f"{BUCKET}/{name}"


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def remote_size(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        length = response.headers.get("Content-Length")
    return int(length) if length is not None else None


def download(name: str, raw: Path) -> Path:
    target = raw / name
    url = url_for(name)

    expected = remote_size(url)
    if target.exists() and expected is not None and target.stat().st_size == expected:
        print(f"  have {name} ({expected:,} B)")
        return target
    if target.exists():
        print(f"  stale {name} ({target.stat().st_size:,} B != {expected:,} B) refetching")
        target.unlink()

    partial = target.with_name(target.name + ".part")
    print(f"  get  {name} ({expected:,} B)" if expected else f"  get  {name}")
    last = [0]

    def report(blocks: int, block_size: int, total: int) -> None:
        done = min(blocks * block_size, total)
        if total and (done - last[0]) > total // 40:
            last[0] = done
            print(f"      {done / 1e6:,.0f} / {total / 1e6:,.0f} MB", flush=True)

    try:
        urllib.request.urlretrieve(url, partial, report)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=None, type=Path, help="project data dir (default: $SWATTER_DATA or data)")
    parser.add_argument("--group", default="core", choices=[*GROUPS, "all", "small-only"])
    parser.add_argument("--columns", action="store_true", help="also fetch the optic hex column xlsx")
    parser.add_argument("--verify", action="store_true", help="sha256 existing files instead of downloading")
    args = parser.parse_args()

    require_remote_execution(
        "fetch_malecns.py",
        "~1.1 GB of downloads for --group core "
        "(annotations 14 MB + neurotransmitters 43 MB + connectome weights 1051 MB)",
    )

    names = GROUPS["core"] if args.group == "all" else GROUPS[args.group]
    if args.columns:
        names = [*names, "optic-columns.xlsx"]
    if args.group == "small-only":
        names = GROUPS["small"]

    raw = default_data_dir(args.data) / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    print(f"raw -> {raw}")
    for name in names:
        try:
            download(name, raw)
        except Exception as error:  # noqa: BLE001 - report and continue to next file
            print(f"  FAIL {name}: {error}", file=sys.stderr)
            return 1

    manifest = {
        "source": BUCKET,
        "files": {},
    }
    for path in sorted(raw.glob("*")):
        if path.suffix == ".part" or not path.is_file():
            continue
        entry: dict[str, object] = {"bytes": path.stat().st_size}
        if args.verify:
            entry["sha256"] = sha256(path)
        manifest["files"][path.name] = entry
    (raw / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    total = sum(int(v["bytes"]) for v in manifest["files"].values())
    print(f"done: {len(manifest['files'])} files, {total / 1e9:.2f} GB")
    print(f"note: {AVOID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
