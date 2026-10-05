"""The fly's eye as the optic lobe sees it: 721 hexagonal columns with real directions.

`loom.EyeGeometry` is a 1-D azimuth strip, good enough for stimulus bookkeeping
and nothing more (its docstring says to swap it out when the optic lobe lands).
This is the swap. A looming disc has to be rendered on the same lattice the
optic lobe's photoreceptors sit on, in two dimensions, or the radial-motion
signal that LPLC2 reads does not exist in the input.

Lattice: the flyvis hexagon of extent 15 (721 columns), flyvis's own (u, v)
coordinates and ordering, so column i here is photoreceptor i of the network.
Neighbouring columns are 5.8 degrees apart (the inter-ommatidial angle flyvis
renders with, `HexEye.omm_width_rad`), which puts the outermost columns 87
degrees off the optical axis: a 174 degree field per eye.

Each column samples the scene through a Gaussian acceptance function. The plan
asks for a width of about one inter-column spacing; `ACCEPTANCE_FWHM_DEG` is
that, as a full width at half maximum. Rendering is a Gaussian-weighted
supersample, so a disc smaller than the acceptance angle is dimmer rather than
simply missing, which is what a real eye does and what the 4.8 degree start of
the r/v = 20 ms loom needs.

Everything here is cheap (a trial is ~10M multiply-adds) and runs locally.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Inter-ommatidial angle flyvis renders with (flyvis/datasets/rendering/eye.py,
# HexEye.omm_width_rad = radians(5.8)).
OMMATIDIAL_ANGLE_DEG = 5.8
EXTENT = 15  # hexagon radius in columns; 3*15*16 + 1 = 721
N_COLUMNS = 3 * EXTENT * (EXTENT + 1) + 1

# Plan: "Gaussian acceptance angle of about one inter-column spacing".
ACCEPTANCE_FWHM_DEG = OMMATIDIAL_ANGLE_DEG

# Gaussian supersample stencil: hex rings of spacing STENCIL_STEP * sigma out to
# STENCIL_RINGS rings (2.7 sigma at 3 rings). 37 points keeps the coverage
# ripple from a sharp disc edge well under 1% of the contrast.
STENCIL_RINGS = 3
STENCIL_STEP = 0.9
# Disc edges are softened by this fraction of sigma so the 37-point sum is smooth
# in disc size instead of stepping as points cross the edge.
EDGE_SOFTNESS = 0.25


def flyvis_columns(extent: int = EXTENT) -> np.ndarray:
    """(u, v) of every column in flyvis's order, (n, 2) int.

    Same loop as `flyvis.datasets.rendering.BoxEye._receptor_centers`, so this
    module needs no flyvis import and `tests/test_eye.py` can pin the order
    against the real network where flyvis is installed.
    """
    cols = []
    for u in range(-extent, extent + 1):
        for v in range(max(-extent, -extent - u), min(extent, extent - u) + 1):
            cols.append((u, v))
    return np.asarray(cols, dtype=np.int64)


def column_plane_deg(uv: np.ndarray, spacing_deg: float = OMMATIDIAL_ANGLE_DEG) -> np.ndarray:
    """Position of each column in the tangent plane, degrees, nearest neighbours `spacing_deg` apart.

    flyvis's own embedding (`hex_utils.hex_to_pixel`, mode "default":
    x = 3/2 v, y = -sqrt(3) (u + v/2)), rescaled so neighbours are
    `spacing_deg` apart (they are sqrt(3) apart in flyvis's units). The turn of
    the lattice is NOT free: the T4/T5 preferred directions flyvis reports
    (T4a 180, T4b 0, T4c 90, T4d 270 degrees) are angles in this plane, and the
    radial-motion template over T4/T5 has to point its inputs along them.
    `tests/test_eye.py` and `optic_gate.py --calibrate` check that a bar drifting
    along a preferred direction in this plane drives that cell type hardest.
    """
    u = uv[:, 0].astype(float)
    v = uv[:, 1].astype(float)
    x = 1.5 * v
    y = -math.sqrt(3.0) * (u + v / 2.0)
    return np.stack([x, y], axis=1) * (spacing_deg / math.sqrt(3.0))


def directions_from_plane(plane_deg: np.ndarray) -> np.ndarray:
    """Unit vectors in the eye frame (z = optical axis), azimuthal-equidistant.

    The radial distance on the plane is the angle from the optical axis, which
    is how a hemispherical array of ommatidia is laid out.
    """
    rho = np.hypot(plane_deg[:, 0], plane_deg[:, 1])
    phi = np.arctan2(plane_deg[:, 1], plane_deg[:, 0])
    r = np.radians(rho)
    return np.stack([np.sin(r) * np.cos(phi), np.sin(r) * np.sin(phi), np.cos(r)], axis=1)


@dataclass
class Eye:
    """One compound eye: column directions in the eye frame."""

    uv: np.ndarray  # (n, 2) flyvis (u, v)
    directions: np.ndarray  # (n, 3) unit vectors, z = optical axis
    acceptance_fwhm_deg: float = ACCEPTANCE_FWHM_DEG

    @classmethod
    def flyvis(cls, acceptance_fwhm_deg: float = ACCEPTANCE_FWHM_DEG) -> "Eye":
        uv = flyvis_columns()
        return cls(uv, directions_from_plane(column_plane_deg(uv)), acceptance_fwhm_deg)

    @property
    def n_columns(self) -> int:
        return int(self.uv.shape[0])

    @property
    def sigma_deg(self) -> float:
        return self.acceptance_fwhm_deg / (2.0 * math.sqrt(2.0 * math.log(2.0)))

    def _stencil(self) -> tuple[np.ndarray, np.ndarray]:
        """Tangent-plane offsets (K, 2) in radians and their normalised Gaussian weights."""
        sigma = math.radians(self.sigma_deg)
        pts = []
        for a in range(-STENCIL_RINGS, STENCIL_RINGS + 1):
            for b in range(max(-STENCIL_RINGS, -STENCIL_RINGS - a), min(STENCIL_RINGS, STENCIL_RINGS - a) + 1):
                pts.append((math.sqrt(3.0) * (b + a / 2.0), 1.5 * a))
        offs = np.asarray(pts) * (STENCIL_STEP * sigma / math.sqrt(3.0))
        w = np.exp(-0.5 * (offs**2).sum(axis=1) / sigma**2)
        return offs, w / w.sum()

    def sample_directions(self) -> tuple[np.ndarray, np.ndarray]:
        """(n, K, 3) supersample directions per column and (K,) weights."""
        offs, w = self._stencil()
        d = self.directions
        # Tangent basis per column; the pole column is handled by picking a
        # reference axis that is not parallel to it.
        ref = np.where(np.abs(d[:, [2]]) > 0.9, np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 1.0]]))
        e1 = np.cross(ref, d)
        e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
        e2 = np.cross(d, e1)
        s = d[:, None, :] + offs[None, :, 0:1] * e1[:, None, :] + offs[None, :, 1:2] * e2[:, None, :]
        s /= np.linalg.norm(s, axis=2, keepdims=True)
        return s, w

    def coverage(self, centre: np.ndarray, radius_deg: np.ndarray) -> np.ndarray:
        """Fraction of each column's acceptance function covered by a disc.

        centre: (T, 3) unit vectors, radius_deg: (T,). Returns (T, n) in [0, 1].
        """
        s, w = self.sample_directions()
        # (T, n, K) angle from each supersample to each disc centre
        cosang = np.einsum("nkc,tc->tnk", s, centre)
        ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
        soft = max(EDGE_SOFTNESS * self.sigma_deg, 1e-6)
        inside = 1.0 / (1.0 + np.exp(-np.clip((radius_deg[:, None, None] - ang) / soft, -60.0, 60.0)))
        return inside @ w


def unit_vector(ecc_deg: np.ndarray | float, phi_deg: np.ndarray | float) -> np.ndarray:
    """Direction at eccentricity `ecc_deg` from the optical axis, position angle `phi_deg`."""
    ecc = np.radians(np.atleast_1d(np.asarray(ecc_deg, dtype=float)))
    phi = np.radians(np.atleast_1d(np.asarray(phi_deg, dtype=float)))
    ecc, phi = np.broadcast_arrays(ecc, phi)
    return np.stack([np.sin(ecc) * np.cos(phi), np.sin(ecc) * np.sin(phi), np.cos(ecc)], axis=-1)


# Where in the eye frame the lab looms sit by default. 40 degrees off the
# optical axis is well inside the 87 degree field, away from the pole, and where
# a loom's whole expanding disc stays on one eye through the 28 degree finish
# of the r/v = 20 ms trial.
LAB_ECCENTRICITY_DEG = 40.0


def render_loom(loom, eye: Eye, ecc_deg: float = LAB_ECCENTRICITY_DEG, translate_arc_deg: float = 40.0) -> np.ndarray:
    """(n_steps, n_columns) luminance of a `loom.Loom` on this eye.

    The loom's `azimuth_deg` is the position angle of the disc around the
    optical axis, at eccentricity `ecc_deg`. Size, growth and the control
    conditions come from `Loom` itself, so the controls are still generated by
    the same code path as the stimulus (see loom.py on why that matters); this
    function only decides where the disc lands on the lattice.

    `Loom.elevation_deg` is ignored: its default of 90 degrees ("straight up")
    is a pole where azimuth means nothing, and an eye-frame position angle needs
    no elevation.
    """
    theta = loom.angular_size_deg()
    radius = theta / 2.0
    n = theta.size
    phi = np.full(n, float(loom.azimuth_deg))
    if loom.kind == "translating":
        # loom.py sweeps +-40 degrees of arc (`translate_arc_deg`, the default);
        # at this eccentricity that is a position-angle change of arc / sin(ecc).
        # The speed diagnostic in optic_gate.py varies it.
        sweep = np.linspace(-1.0, 1.0, n) * translate_arc_deg / math.sin(math.radians(ecc_deg))
        phi = phi + sweep
    centre = unit_vector(ecc_deg, phi)
    covered = eye.coverage(centre, radius)

    fade = np.ones(n)
    if loom.kind == "dimming":
        t = loom.time_ms()
        fade = np.clip(1.0 - (t - t[0]) / max(t[-1] - t[0], 1e-6), 0.0, 1.0)
    level = loom.foreground + (1.0 - loom.foreground) * (1.0 - loom.contrast)
    return loom.background + (level - loom.background) * covered * fade[:, None]


def edge_movie(eye: Eye, phi_deg: float, speed_deg_s: float, dt_s: float, t_s: float,
               polarity: str = "on", width_deg: float = 20.0) -> np.ndarray:
    """(T, n) luminance of a bar sweeping the tangent plane in direction `phi_deg`.

    Calibration stimulus, defined directly on the plane `column_plane_deg`
    lays out: grey (0.5) with a bar of `width_deg` that is brighter ("on", 1.0)
    or darker ("off", 0.0), moving along (cos phi, sin phi). Uses the plane
    positions, not the sphere, so the direction it moves in is exactly the
    angle flyvis quotes for a preferred direction.
    """
    pos = column_plane_deg(eye.uv)
    n = np.array([math.cos(math.radians(phi_deg)), math.sin(math.radians(phi_deg))])
    along = pos @ n
    steps = int(round(t_s / dt_s))
    reach = 90.0 + width_deg
    centre = -reach + speed_deg_s * dt_s * np.arange(steps)
    inside = (np.abs(along[None, :] - centre[:, None]) < width_deg / 2.0).astype(float)
    level = 1.0 if polarity == "on" else 0.0
    return 0.5 + (level - 0.5) * inside
