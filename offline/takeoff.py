"""Takeoff mode from descending-neuron spike times: the plan's mode rule (docs/build-plan.md).

    GF spikes first                              -> short
    parallel DNs cross threshold first (t_p)     -> wing raise starts
        GF spikes within the raise (<= t_p + W)  -> short
        raise completes without a GF spike       -> long
    nothing fires                                -> no escape

W is the published short-mode takeoff bound, 6.87 ms (von Reyn et al. 2014; the
plan's number). "Cross threshold" is the first spike of any parallel DN
(DNp04, DNp103, DNp02, DNp11, both sides). Nothing here is fitted: the rule and
W are fixed by the plan, so criterion 3 is a test of the circuit, not a tuning
target.
"""

from __future__ import annotations

SHORT_WINDOW_MS = 6.87


def classify(gf_t_ms: float | None, parallel_t_ms: float | None, window_ms: float = SHORT_WINDOW_MS) -> str:
    """'short', 'long' or 'none' from the two first-spike times (None = never spiked)."""
    if gf_t_ms is None and parallel_t_ms is None:
        return "none"
    if parallel_t_ms is None:
        return "short"  # GF spiked and nothing raised the wings first
    if gf_t_ms is None:
        return "long"
    return "short" if gf_t_ms <= parallel_t_ms + window_ms else "long"
