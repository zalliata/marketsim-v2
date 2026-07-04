"""Sentiment process with the P0-calibrated crisis injection.

Sentiment is a scalar in [-1, 1] per tick, shared by all sentiment-enabled
agents. Calibration (P0): during the SVB run, 60% of crisis-window social-media
messages were negative; v1 injected this as an exogenous shock at crisis onset.
Here the process is an AR(1) with a crisis-onset level shift and seeded noise,
optionally perturbed further by the scenario controller.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .anchors import CRISIS_TICK


@dataclass
class SentimentProcess:
    seed: int = 0
    persistence: float = 0.92        # AR(1) coefficient
    noise_std: float = 0.05
    pre_crisis_mean: float = 0.0
    crisis_mean: float = -0.6        # 60% negative message share -> level -0.6
    crisis_tick: int = CRISIS_TICK
    crisis_enabled: bool = True
    _level: float = 0.0
    _rng: random.Random = field(init=False, repr=False)
    _extra_shocks: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed ^ 0x5E17)
        self._level = self.pre_crisis_mean

    def schedule_shock(self, tick: int, delta: float) -> None:
        """Scenario controller: add `delta` to sentiment at `tick`."""
        self._extra_shocks[tick] = self._extra_shocks.get(tick, 0.0) + delta

    def step(self, tick: int) -> float:
        mean = (self.crisis_mean if (self.crisis_enabled and tick >= self.crisis_tick)
                else self.pre_crisis_mean)
        self._level = (self.persistence * self._level
                       + (1 - self.persistence) * mean
                       + self._rng.gauss(0.0, self.noise_std)
                       + self._extra_shocks.pop(tick, 0.0))
        self._level = max(-1.0, min(1.0, self._level))
        return self._level
