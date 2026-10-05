# Browser engine: contract, export format, and reference

Status: contract, exporter and a Python reference engine exist and agree on 50
fixed stimuli. The Rust/WASM and WebGPU engine does not exist yet (no Rust or Node
toolchain on the DGX host).

## Contract

The browser runs the LIF circuit only. The optic lobe is not shipped
(`docs/week1_gate.md`): the game gives the engine one **frame** every 5 ms and gets
spikes back.

    frame = (rate, xR, yR, xL, yL)

`rate` is the swatter's angular growth rate in deg/ms; `(xR, yR)` and `(xL, yL)` are its
position on each eye's plane in degrees (`heading.plane_position` is the geometry, with
the stated assumptions about eye axes). LPLC2 neuron i is driven by

    relu(rate) * exp(-|rf_i - pos_eye(i)|^2 / (2 * 15^2)) * input_gain      mV/ms

linearly interpolated between frames. The engine runs 25 steps of 0.2 ms per frame,
so it is one frame (5 ms) behind the game, and `finish()` runs the last sample.

Outputs: spikes of 495 neurons per step. The game derives, with fixed rules and no
fitted constants: first giant-fiber spike, first parallel-DN spike, mode
(`takeoff.classify`, W = 6.87 ms) and heading side (`heading.asymmetry`, 20 ms
window). Because criterion 3 fails, the mode is always short in practice
(`docs/lif_fit.md`), and the game should present it that way.

## Files (`offline/export.py` writes `web/brain/`)

- `brain.bin` (252 KiB): CSR by presynaptic neuron, 42,396 edges, signed mV per spike
  with `weight_scale` baked in, plus the driven-neuron table. Little-endian,
  4-byte aligned, offsets in the manifest.
- `manifest.json`: array specs, LIF constants and the three fitted parameters, drive
  spec, neuron roles/types/sides, provenance (escape-graph hash, repo commit). Marked
  PROVISIONAL.
- `parity.json` (224 KiB): 50 stimuli as frames with the Python outcome. 42 expanding
  (7 r/v x 6 lateral azimuths), 4 controls, 4 edge cases (ahead, behind, r/v 7, 120).

Weights are float32, not the plan's float16: the file is small, and float16 would add a
rounding that parity would then have to excuse.

## Reference and parity

`offline/engine_ref.py` is the spec: event-driven (a spike scatters its CSR row into
a delay ring), streaming (frame in, 25 steps out), built from `web/brain/` only.
`tests/test_engine_ref.py` runs it on all 50 parity stimuli and compares with the
dense torch reference (`lif.py` + `angular_drive`): mode, heading side and the
first-spike times agree to within one step. Measured: **all 46 trials with a GF spike
agree exactly** (0.0 ms difference), and a 2% change to all weights moves 35 of 50 trials
outside one step, so the test is sensitive. The test skips when `web/brain` is absent.

The Rust/WASM engine's test is the same file's logic against the same `parity.json`;
leaderboard verification compares outcomes (mode, heading side), not raw floats, as the
plan's determinism note says.

## Not done

- The Rust/WASM and WebGPU engine itself, and its speed measurement ("faster than real
  time on a mid-range laptop"). Size of the work: 495 neurons, 42k edges, 2,000 steps
  per 400 ms trial, so well under 100M edge-operations per trial.
- Parameters are provisional (Brian2 check, criterion 3 decision).
- Nothing in the browser has run yet.
