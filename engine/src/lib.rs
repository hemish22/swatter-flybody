//! The LIF circuit that runs in the browser. Spec: `offline/engine_ref.py` (numpy) and
//! `docs/engine.md`; both are tested against the same 50 parity stimuli.
//!
//! Plain C ABI, no wasm-bindgen and no dependencies, so the module is ~tens of KiB and
//! loads from any host (browser, Node, wasmtime). The host copies the arrays of
//! `web/brain/brain.bin` into memory obtained from `alloc`, calls `init`, then feeds
//! one 5 ms frame at a time to `frame` and reads spikes back.
//!
//! Single-threaded global state: one trial at a time per module instance. Create
//! another instance for another trial.
//!
//! Arithmetic mirrors the numpy reference step for step (f32 state, f64 drive then
//! cast), so spike times agree with it to the step on the parity set.

use std::cell::RefCell;

const SUBSTEPS: usize = 25;

/// Parameter block, f64 in this order (see `Params::from_slice`).
const N_PARAMS: usize = 12;

struct Params {
    v0: f32,
    v_rst: f32,
    v_th: f32,
    t_mbr: f64,
    tau: f64,
    dt: f64,
    n_delay: usize,
    n_refractory: i32,
    input_gain: f32,
    sigma: f64,
}

impl Params {
    fn from_slice(p: &[f64]) -> Params {
        Params {
            v0: p[0] as f32,
            v_rst: p[1] as f32,
            v_th: p[2] as f32,
            t_mbr: p[3],
            tau: p[4],
            dt: p[5],
            n_delay: p[6] as usize,
            n_refractory: p[7] as i32,
            input_gain: p[8] as f32,
            sigma: p[9],
        }
    }
}

struct Engine {
    p: Params,
    n: usize,
    indptr: Vec<i32>,
    post: Vec<u16>,
    w: Vec<f32>,
    drive_neuron: Vec<u16>,
    drive_rf: Vec<f32>, // (m, 2)
    drive_eye: Vec<u8>,
    v: Vec<f32>,
    g: Vec<f32>,
    refr: Vec<i32>,
    ring: Vec<f32>, // n_delay * n
    x: Vec<f32>,
    prev_drive: Option<Vec<f32>>,
    step: u32,
    spikes: Vec<u32>, // (step, neuron) pairs, flat
}

thread_local! {
    static ENGINE: RefCell<Option<Engine>> = const { RefCell::new(None) };
}

impl Engine {
    fn drive(&self, frame: &[f64; 5]) -> Vec<f32> {
        let rate = frame[0].max(0.0);
        let pos = [[frame[1], frame[2]], [frame[3], frame[4]]];
        let two_s2 = 2.0 * self.p.sigma * self.p.sigma;
        (0..self.drive_neuron.len())
            .map(|i| {
                let c = pos[self.drive_eye[i] as usize];
                let dx = self.drive_rf[2 * i] as f64 - c[0];
                let dy = self.drive_rf[2 * i + 1] as f64 - c[1];
                (rate * (-(dx * dx + dy * dy) / two_s2).exp()) as f32
            })
            .collect()
    }

    fn substep(&mut self, drive: &[f32]) {
        let n = self.n;
        let slot = (self.step as usize) % self.p.n_delay;
        for v in self.x.iter_mut() {
            *v = 0.0;
        }
        for (i, &nu) in self.drive_neuron.iter().enumerate() {
            self.x[nu as usize] = drive[i] * self.p.input_gain;
        }
        // constants are formed in f64 and then cast, as the numpy reference does
        let dt = self.p.dt as f32;
        let k_v = (self.p.dt / self.p.t_mbr) as f32;
        let k_g = (self.p.dt / self.p.tau) as f32;
        let base = slot * n;
        for j in 0..n {
            let arriving = self.ring[base + j];
            self.ring[base + j] = 0.0;
            self.g[j] += arriving + self.x[j] * dt;
        }
        let mut fired: Vec<usize> = Vec::new();
        for j in 0..n {
            let active = self.refr[j] <= 0;
            if active {
                let dv = (self.p.v0 - self.v[j] + self.g[j]) * k_v;
                self.v[j] += dv;
            } else {
                self.v[j] = self.p.v_rst;
            }
            self.g[j] -= self.g[j] * k_g;
            if self.v[j] > self.p.v_th && active {
                self.v[j] = self.p.v_rst;
                self.refr[j] = self.p.n_refractory;
                fired.push(j);
            } else {
                self.refr[j] -= 1;
            }
        }
        for &j in &fired {
            let (lo, hi) = (self.indptr[j] as usize, self.indptr[j + 1] as usize);
            for e in lo..hi {
                // arrives n_delay steps from now, which is this slot's next visit
                self.ring[base + self.post[e] as usize] += self.w[e];
            }
            self.spikes.push(self.step);
            self.spikes.push(j as u32);
        }
        self.step += 1;
    }
}

fn with<R>(f: impl FnOnce(&mut Engine) -> R) -> R {
    ENGINE.with(|e| f(e.borrow_mut().as_mut().expect("engine not initialised: call init first")))
}

/// Memory for the host to fill. Never freed by the engine; call `init` once per instance.
#[no_mangle]
pub extern "C" fn alloc(size: u32) -> *mut u8 {
    let mut buf = Vec::<u8>::with_capacity(size.max(8) as usize);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

/// Copy the arrays in (all little-endian, as `brain.bin` stores them) and build the state.
///
/// # Safety
/// Every pointer must come from `alloc` with at least the size its array needs.
#[no_mangle]
pub unsafe extern "C" fn init(
    n: u32,
    edges: u32,
    n_driven: u32,
    p_indptr: *const i32,
    p_post: *const u16,
    p_weight: *const f32,
    p_dneuron: *const u16,
    p_drf: *const f32,
    p_deye: *const u8,
    p_params: *const f64,
) {
    let (n, edges, m) = (n as usize, edges as usize, n_driven as usize);
    let p = Params::from_slice(std::slice::from_raw_parts(p_params, N_PARAMS));
    let n_delay = p.n_delay;
    let v0 = p.v0;
    let eng = Engine {
        indptr: std::slice::from_raw_parts(p_indptr, n + 1).to_vec(),
        post: std::slice::from_raw_parts(p_post, edges).to_vec(),
        w: std::slice::from_raw_parts(p_weight, edges).to_vec(),
        drive_neuron: std::slice::from_raw_parts(p_dneuron, m).to_vec(),
        drive_rf: std::slice::from_raw_parts(p_drf, 2 * m).to_vec(),
        drive_eye: std::slice::from_raw_parts(p_deye, m).to_vec(),
        v: vec![v0; n],
        g: vec![0.0; n],
        refr: vec![0; n],
        ring: vec![0.0; n_delay * n],
        x: vec![0.0; n],
        prev_drive: None,
        step: 0,
        spikes: Vec::new(),
        n,
        p,
    };
    ENGINE.with(|e| *e.borrow_mut() = Some(eng));
}

/// Start a new trial: neurons at rest, no spikes, no pending frame.
#[no_mangle]
pub extern "C" fn reset() {
    with(|e| {
        let v0 = e.p.v0;
        e.v.iter_mut().for_each(|x| *x = v0);
        e.g.iter_mut().for_each(|x| *x = 0.0);
        e.refr.iter_mut().for_each(|x| *x = 0);
        e.ring.iter_mut().for_each(|x| *x = 0.0);
        e.prev_drive = None;
        e.step = 0;
        e.spikes.clear();
    });
}

/// Take the next 5 ms frame. The first call after `reset` only latches it; each later one
/// runs the 25 steps between the previous frame and this one.
#[no_mangle]
pub extern "C" fn frame(rate: f64, xr: f64, yr: f64, xl: f64, yl: f64) {
    with(|e| {
        let cur = e.drive(&[rate, xr, yr, xl, yl]);
        if let Some(prev) = e.prev_drive.take() {
            let mut mix = vec![0f32; cur.len()];
            for i in 0..SUBSTEPS {
                let f = (i as f64 / SUBSTEPS as f64) as f32;
                for k in 0..cur.len() {
                    mix[k] = prev[k] * (1.0 - f) + cur[k] * f;
                }
                e.substep(&mix);
            }
        }
        e.prev_drive = Some(cur);
    });
}

/// The trial's last sample, at the final frame's own drive.
#[no_mangle]
pub extern "C" fn finish() {
    with(|e| {
        if let Some(prev) = e.prev_drive.clone() {
            e.substep(&prev);
        }
    });
}

#[no_mangle]
pub extern "C" fn step_count() -> u32 {
    with(|e| e.step)
}

/// Number of recorded spikes; `spikes_ptr` points at that many (step, neuron) u32 pairs.
#[no_mangle]
pub extern "C" fn spike_count() -> u32 {
    with(|e| (e.spikes.len() / 2) as u32)
}

#[no_mangle]
pub extern "C" fn spikes_ptr() -> *const u32 {
    with(|e| e.spikes.as_ptr())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two neurons, 0 -> 1 with a weight big enough to fire 1 on one spike of 0.
    fn tiny() {
        let indptr = [0i32, 1, 1];
        let post = [1u16];
        let w = [60.0f32];
        let dneuron = [0u16];
        let drf = [0.0f32, 0.0];
        let deye = [0u8];
        let params = [-52.0, -52.0, -45.0, 20.0, 5.0, 0.2, 9.0, 11.0, 1000.0, 15.0, 0.0, 0.0];
        unsafe {
            init(2, 1, 1, indptr.as_ptr(), post.as_ptr(), w.as_ptr(), dneuron.as_ptr(), drf.as_ptr(), deye.as_ptr(), params.as_ptr());
        }
    }

    #[test]
    fn a_driven_neuron_fires_its_target_one_delay_later() {
        tiny();
        reset();
        for _ in 0..4 {
            frame(1.0, 0.0, 0.0, 99.0, 99.0);
        }
        finish();
        let s: Vec<(u32, u32)> = with(|e| e.spikes.chunks(2).map(|c| (c[0], c[1])).collect());
        let t0 = s.iter().find(|x| x.1 == 0).expect("neuron 0 fires").0;
        let t1 = s.iter().find(|x| x.1 == 1).expect("neuron 1 fires").0;
        assert!(t1 > t0, "target fires after its source: {s:?}");
    }

    #[test]
    fn no_growth_means_no_spikes() {
        tiny();
        reset();
        for _ in 0..4 {
            frame(0.0, 0.0, 0.0, 0.0, 0.0);
        }
        finish();
        assert_eq!(spike_count(), 0);
    }

    #[test]
    fn a_disc_far_from_the_receptive_field_drives_nothing() {
        tiny();
        reset();
        for _ in 0..4 {
            frame(1.0, 90.0, 0.0, 99.0, 99.0);
        }
        finish();
        assert_eq!(spike_count(), 0);
    }

    #[test]
    fn reset_restores_the_start_state() {
        tiny();
        reset();
        for _ in 0..4 {
            frame(1.0, 0.0, 0.0, 99.0, 99.0);
        }
        finish();
        let first = with(|e| e.spikes.clone());
        reset();
        for _ in 0..4 {
            frame(1.0, 0.0, 0.0, 99.0, 99.0);
        }
        finish();
        assert_eq!(with(|e| e.spikes.clone()), first);
    }
}
