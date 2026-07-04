"""Systemic-risk metrics for Paper P6 (phase-transition analysis).

Given batch statistics across an adversary-share grid, these locate the
concentration threshold at which stabilisation breaks down and quantify the
degradation curve — the empirical objects of RQ-F7.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PhaseTransition:
    threshold_share: float          # share at which stabilisation < 50% of its 5%-share value
    found: bool
    stabilisation_curve: list[tuple[float, float]]   # (share, stabilisation_effectiveness)
    cascade_curve: list[tuple[float, float]]         # (share, cascade_frequency)


def locate_phase_transition(share_grid: list[float],
                            stabilisation: list[float],
                            cascade_freq: list[float]) -> PhaseTransition:
    """Threshold = first share where stabilisation drops below half its value at
    the smallest share (RQ-F7 operational definition)."""
    curve = list(zip(share_grid, stabilisation))
    cc = list(zip(share_grid, cascade_freq))
    if not stabilisation:
        return PhaseTransition(0.0, False, curve, cc)
    base = stabilisation[0]
    threshold, found = share_grid[-1], False
    for share, stab in curve:
        if base > 0 and stab < 0.5 * base:
            threshold, found = share, True
            break
    return PhaseTransition(threshold, found, curve, cc)
