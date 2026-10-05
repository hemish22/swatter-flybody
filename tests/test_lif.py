"""Tests for the LIF simulator. Cheap, CPU, no data needed.

Each one pins behaviour the rest of the pipeline depends on: the delay and
refractory period (short-mode vs long-mode takeoff is decided at millisecond
scale), the sign convention (inverting it inverts the circuit), determinism (the
leaderboard re-sim), and batch independence.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "offline"))

from lif import LifNetwork, LifParams, upsample_drive  # noqa: E402

P = LifParams()
STEPS_PER_MS = int(round(1 / P.dt))


def steps(ms: float) -> int:
    return int(round(ms / P.dt))


def two_neurons(count: float, sign: float = 1.0) -> LifNetwork:
    """neuron 0 -> neuron 1 with `count` synapses."""
    return LifNetwork.from_edges(2, [0], [1], [count], np.array([sign, 1.0]), P)


def constant_drive(mv_per_ms: float, ms: float, neurons=(0,), n=2) -> np.ndarray:
    d = np.zeros((1, steps(ms), n), dtype=np.float32)
    d[:, :, list(neurons)] = mv_per_ms
    return d


class TestSingleNeuron(unittest.TestCase):
    def test_no_drive_no_spikes_and_rest_potential(self):
        out = LifNetwork(np.zeros((1, 1)), P).run(np.zeros((1, steps(100), 1)), record_v=np.array([0]))
        self.assertFalse(out["spikes"].any())
        np.testing.assert_allclose(out["v"], P.v0)

    def test_threshold_in_drive(self):
        """Steady conductance is drive * tau; it must beat v_th - v0 = 7 mV to fire."""
        below = (P.v_th - P.v0) / P.tau * 0.95
        above = (P.v_th - P.v0) / P.tau * 1.3
        net = LifNetwork(np.zeros((1, 1)), P)
        self.assertFalse(net.run(np.full((1, steps(300), 1), below, dtype=np.float32))["spikes"].any())
        self.assertTrue(net.run(np.full((1, steps(300), 1), above, dtype=np.float32))["spikes"].any())

    def test_rate_rises_with_drive(self):
        net = LifNetwork(np.zeros((1, 1)), P)
        counts = [net.run(np.full((1, steps(500), 1), d, dtype=np.float32))["spikes"].sum() for d in (1.6, 2.2, 3.5)]
        self.assertLess(counts[0], counts[1])
        self.assertLess(counts[1], counts[2])

    def test_refractory_period_is_respected(self):
        net = LifNetwork(np.zeros((1, 1)), P)
        t = np.flatnonzero(net.run(np.full((1, steps(300), 1), 50.0, dtype=np.float32))["spikes"][0, :, 0])
        self.assertGreater(len(t), 5)
        self.assertGreaterEqual(np.diff(t).min() * P.dt, P.t_rfc)


class TestSynapses(unittest.TestCase):
    def first_spike_ms(self, out, neuron):
        t = np.flatnonzero(out["spikes"][0, :, neuron])
        return None if t.size == 0 else t[0] * P.dt

    def test_spike_arrives_after_exactly_the_delay(self):
        """Catches a delay off by a step, or a spike that arrives the step it is emitted.

        Measured on the membrane, not the spike: one spike adds count * 0.275 mV
        to the conductance, which decays over 5 ms, so a single input moves the
        target by only ~count/70 mV and a 40-synapse pair does not fire it.
        """
        net = two_neurons(count=40)
        pulse = np.zeros((1, steps(40), 2), dtype=np.float32)
        pulse[0, 10, 0] = 400.0  # one big kick to neuron 0
        out = net.run(pulse, record_v=np.array([1]))
        sent = int(np.flatnonzero(out["spikes"][0, :, 0])[0])
        moved = int(np.flatnonzero(out["v"][0, :, 0] > P.v0 + 1e-6)[0])
        self.assertEqual(moved - sent, P.n_delay)
        self.assertAlmostEqual((moved - sent) * P.dt, P.t_dly, places=6)

    def test_a_strong_enough_pair_fires_its_target(self):
        """One spike's peak PSP is 0.043 mV per synapse, so ~162 synapses are needed to cross 7 mV."""
        net = two_neurons(count=200)
        pulse = np.zeros((1, steps(40), 2), dtype=np.float32)
        pulse[0, 10, 0] = 400.0
        out = net.run(pulse)
        self.assertTrue(out["spikes"][0, :, 1].any())
        self.assertGreater(np.flatnonzero(out["spikes"][0, :, 1])[0], np.flatnonzero(out["spikes"][0, :, 0])[0])

    def test_excitatory_weight_is_count_times_w_syn(self):
        net = two_neurons(count=10)
        self.assertAlmostEqual(float(net.w[1, 0]), 10 * P.w_syn, places=5)

    def test_inhibitory_presynaptic_neuron_hyperpolarises(self):
        """Sign comes from the PRESYNAPTIC neuron's transmitter; wrong sign inverts the circuit."""
        net = two_neurons(count=40, sign=-1.0)
        pulse = np.zeros((1, steps(40), 2), dtype=np.float32)
        pulse[0, 10, 0] = 400.0
        out = net.run(pulse, record_v=np.array([1]))
        self.assertLess(float(out["v"].min()), P.v0 - 1.0)
        self.assertIsNone(self.first_spike_ms(out, 1))

    def test_weight_scale_scales_every_weight(self):
        a = two_neurons(count=10).w
        b = LifNetwork.from_edges(2, [0], [1], [10], np.array([1.0, 1.0]), LifParams(weight_scale=2.0)).w
        np.testing.assert_allclose(b.numpy(), 2 * a.numpy())


class TestDeterminismAndBatching(unittest.TestCase):
    def test_same_input_same_output(self):
        net = two_neurons(count=40)
        d = constant_drive(3.0, 100)
        np.testing.assert_array_equal(net.run(d)["spikes"], net.run(d)["spikes"])

    def test_trials_in_a_batch_do_not_interact(self):
        net = two_neurons(count=40)
        d = np.concatenate([constant_drive(3.0, 100), constant_drive(0.0, 100)])
        both = net.run(d)["spikes"]
        np.testing.assert_array_equal(both[0], net.run(d[:1])["spikes"][0])
        np.testing.assert_array_equal(both[1], net.run(d[1:])["spikes"][0])
        self.assertFalse(both[1].any())


class TestUpsample(unittest.TestCase):
    def test_linear_between_samples_and_exact_at_them(self):
        d = np.array([[[0.0], [10.0], [0.0]]], dtype=np.float32)  # 5 ms grid
        u = upsample_drive(d, 5.0, 0.5, 10.0)
        self.assertAlmostEqual(float(u[0, 0, 0]), 0.0)
        self.assertAlmostEqual(float(u[0, 10, 0]), 10.0, places=4)
        self.assertAlmostEqual(float(u[0, 5, 0]), 5.0, places=4)
        self.assertAlmostEqual(float(u[0, 15, 0]), 5.0, places=4)


if __name__ == "__main__":
    unittest.main()
