"""Run the Rust engine (`engine/`, built to `web/brain/engine.wasm`) from Python, via wasmtime.

This is the same glue the browser needs, written once in Python so the WASM build
can be tested against `parity.json` and timed without a browser or Node: copy the
arrays of `brain.bin` into memory from `alloc`, call `init`, then `frame` per 5 ms.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np
import wasmtime

ARRAYS = ("csr_indptr", "csr_post", "csr_weight_mv", "drive_neuron", "drive_rf_deg", "drive_eye")


class WasmEngine:
    def __init__(self, brain_dir: Path):
        m = json.loads((brain_dir / "manifest.json").read_text())
        self.manifest = m
        raw = (brain_dir / m["binary"]).read_bytes()
        eng = wasmtime.Engine()
        store = wasmtime.Store(eng)
        module = wasmtime.Module.from_file(eng, str(brain_dir / "engine.wasm"))
        inst = wasmtime.Instance(store, module, [])
        self.store, self.x = store, inst.exports(store)
        self.mem = self.x["memory"]
        self._call = lambda name, *a: self.x[name](store, *a)

        def put(data: bytes) -> int:
            ptr = self._call("alloc", len(data))
            ctypes.memmove(self._addr(ptr), data, len(data))
            return ptr

        ptrs = {}
        for name in ARRAYS:
            s = m["arrays"][name]
            nbytes = int(np.prod(s["shape"])) * np.dtype(s["dtype"]).itemsize
            ptrs[name] = put(raw[s["offset"] : s["offset"] + nbytes])
        p = m["lif"]
        params = np.array([p["v0"], p["v_rst"], p["v_th"], p["t_mbr"], p["tau"], p["dt_ms"], p["n_delay"], p["n_refractory"],
                           p["input_gain"], m["drive"]["rf_sigma_deg"], 0.0, 0.0], dtype="<f8")
        ptrs["params"] = put(params.tobytes())
        self.dt = p["dt_ms"]
        self._call("init", m["neurons"], m["edges"], m["arrays"]["drive_neuron"]["shape"][0],
                   ptrs["csr_indptr"], ptrs["csr_post"], ptrs["csr_weight_mv"], ptrs["drive_neuron"],
                   ptrs["drive_rf_deg"], ptrs["drive_eye"], ptrs["params"])

    def _addr(self, ptr: int) -> int:
        return self._base() + ptr

    def _base(self) -> int:
        return ctypes.cast(self.mem.data_ptr(self.store), ctypes.c_void_p).value

    def run(self, frames: np.ndarray) -> list[tuple[int, int]]:
        self._call("reset")
        for f in frames:
            self._call("frame", *(float(v) for v in f))
        self._call("finish")
        n = self._call("spike_count")
        if n == 0:
            return []
        ptr = self._call("spikes_ptr")
        buf = (ctypes.c_uint32 * (2 * n)).from_address(self._base() + ptr)
        a = np.frombuffer(buf, dtype="<u4").reshape(n, 2)
        return [(int(t), int(j)) for t, j in a]
