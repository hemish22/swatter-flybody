"""Fixtures that pin the game's geometry (web/game/world.ts) to the Python reference.

Writes web/game/fixtures.json: for 16 body directions in the horizontal plane (azimuth 0 ahead,
+90 the fly's right) the eye-plane coordinates `heading.plane_position` gives for each eye. The
TypeScript world model derives the same numbers from a 3D swatter position, so a drift in either
side's convention shows up as a failing test, not as a mirrored fly.

    .venv/bin/python offline/game_fixtures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from heading import plane_position  # noqa: E402


def main() -> int:
    az = [float(a) for a in np.arange(0, 360, 22.5)]
    rows = [{"azimuth_deg": a, "R": plane_position(a, "R")[0].tolist(), "L": plane_position(a, "L")[0].tolist()} for a in az]
    out = Path("web/game/fixtures.json")
    out.write_text(json.dumps({"horizontal_plane_positions": rows}, indent=1))
    print(f"wrote {out} ({len(rows)} directions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
