# Browser engine: contract, export format, and reference

Status: contract, exporter, a Python reference engine and the Rust/WASM engine
(`engine/`, 31 KiB, no dependencies) exist and all agree on 50 fixed stimuli. No
browser has run it yet.

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

## The Rust/WASM engine

`engine/src/lib.rs`: C ABI (`alloc`, `init`, `reset`, `frame`, `finish`, spike readout), no
wasm-bindgen and no crates, so any host can load it. Arithmetic mirrors the numpy
reference step for step. Build and install:

    cd engine && cargo build --release --target wasm32-unknown-unknown
    cp target/wasm32-unknown-unknown/release/swatter_engine.wasm ../web/brain/engine.wasm

`offline/wasm_host.py` is the host glue (copy arrays in, call `init`, feed frames),
run under wasmtime; the browser glue is the same calls. `tests/test_engine_wasm.py`
checks, on `web/brain/parity.json`:

- all 50 stimuli: mode, heading side and first-spike times match the dense torch
  reference to within one step;
- one trial's whole spike train is **identical** to the numpy engine's;
- a second run on the same instance reproduces the first (reset is complete).

Plus four native `cargo test`s on a two-neuron network.

**Speed** (wasmtime on one server core, not a laptop browser): a 400 ms trial takes a
median 21 ms (p95 29 ms), 19x faster than real time, and the worst single 5 ms frame costs
0.29 ms against the 5 ms budget. A mid-range laptop browser is not measured; at several
times slower it would still have a wide margin. **WebGPU is therefore not needed**: the
circuit is 495 neurons and 42k edges, and the plan's WebGPU fast path would add
cross-GPU float differences for no speed the game uses. The plan's "WASM CPU fallback"
is the engine.

## Not done

- Nothing in a browser has run: the TypeScript glue and the in-browser parity run
  against `parity.json` are still to do (needs Node or a browser; neither is installed on
  this host). The leaderboard's Node re-sim has the same dependency.
- Parameters are provisional (Brian2 check, criterion 3 decision).
- The engine runs the LIF circuit only. Frame generation from the game's world (swatter
  position and size to `(rate, x, y)` per eye) is the game's job and is not written.
