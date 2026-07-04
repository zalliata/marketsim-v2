"""Scenario dataclass, sweep axes, YAML loading."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SweepAxis:
    """A parameter to vary across a batch of batches.

    kind:
    - 'tc_bps'          : market transaction cost (P3 cost sweep)
    - 'adversary_share' : scale adversary capital/order mass (P5/P6 share sweep)
    - 'breaker'         : intervention arm label (P3 circuit-breaker comparison)
    values are produced from start:stop:step or an explicit list.
    """
    kind: str
    values: list[float] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)


@dataclass
class Scenario:
    id: str
    name: str
    description: str
    agent_ids: list[str]
    paper: str = ""                      # e.g. "P1/RQ-F1", "P3/RQ-F4"
    n_ticks: int = 39
    tc_bps: float = 10.0
    crisis_enabled: bool = True
    agent_overrides: dict = field(default_factory=dict)   # agent_id -> config dict
    controller_config: dict = field(default_factory=dict) # merged into S1 if present
    sweep: Optional[SweepAxis] = None
    labels: dict = field(default_factory=dict)            # e.g. {"regime": "manipulated"}


def frange(start: float, stop: float, step: float) -> list[float]:
    out, v = [], start
    n = 0
    while v <= stop + 1e-9:
        out.append(round(v, 6))
        n += 1
        v = start + n * step
    return out
