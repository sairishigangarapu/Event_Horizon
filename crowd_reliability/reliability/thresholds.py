"""Scenario-dependent, explainable freshness thresholds (P2).

The P1 reliability layer used fixed global age boundaries. In practice the
rate at which evidence goes stale differs by *what* is being observed and by
*where* it is observed:

* **exit_open** is a binary event (door opens/closes) and must be treated as
  fresh only for a few seconds; after that it is almost certainly stale.
* **inflow / outflow** are rate quantities that react to upstream changes
  within the timescale of a few pedestrian steps.
* **speed** is a kinematic quantity that changes on the pedestrian reaction
  timescale.
* **queue_length** builds and sheds more slowly than speed but faster than
  bulk density.
* **density** is the slowest-varying macroscopic quantity (mass conservation
  smooths it), so it stays defensible for the longest.

Defaults below encode that ordering. They are **scenario dependent**: callers
may pass a ``scenario_type`` ("evacuation", "normal_flow", etc.) and a
``zone_type`` ("exit", "corridor", "space") to scale the boundaries per the
table in :data:`SCENARIO_MULTIPLIERS`. Every multiplier defaults to ``1.0``
(neutral) so the behaviour is always explainable and reproducible.

The simple :class:`~crowd_reliability.reliability.freshness.FixedExpiryFreshnessBaseline`
(P1) is retained as a single-number baseline for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

# Seconds after which an observation of a variable is "recent" vs "outdated".
# (recently_verified threshold, potentially_outdated threshold)
# Ordering is explainable by the physical timescale of each variable:
# event-driven binary states go stale fastest; bulk density (mass conservation)
# is the slowest to lose reliability and is the most tolerant.
DEFAULT_THRESHOLDS_SEC: Dict[str, Tuple[float, float]] = {
    "exit_open": (5.0, 15.0),       # event-driven binary: door opens/closes fast
    "inflow": (6.0, 25.0),          # rate, reacts within steps of upstream change
    "outflow": (6.0, 25.0),         # rate, reacts within steps of upstream change
    "speed": (10.0, 35.0),          # kinematic, pedestrian reaction timescale
    "queue_length": (15.0, 40.0),   # builds/sheds slower than speed
    "density": (20.0, 45.0),        # slowest (mass conservation); most tolerant
}

# Scenario / zone-type multipliers applied to the base thresholds. Default 1.0.
# A value > 1 widens the "fresh" window (e.g. high-confidence evacuation
# observations), < 1 tightens it (e.g. static camera with occlusion).
SCENARIO_MULTIPLIERS: Dict[str, float] = {
    "normal_flow": 1.0,
    "gradual_congestion": 1.0,
    "bottleneck_exit_closure": 1.2,
    "counterflow": 1.2,
    "sudden_stop": 0.8,
    "evacuation": 0.9,
    "dataset": 1.0,
}

ZONE_TYPE_MULTIPLIERS: Dict[str, float] = {
    "exit": 0.8,        # exits change abruptly, require fresher evidence
    "corridor": 1.0,
    "space": 1.1,       # open spaces evolve slowly
    "unknown": 1.0,
}

VARIABLES = list(DEFAULT_THRESHOLDS_SEC.keys())


@dataclass(frozen=True)
class FreshnessThresholds:
    """Explainable, scenario/zone dependent thresholds for the six channels."""

    variable_overrides: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    scenario_multipliers: Dict[str, float] = field(default_factory=lambda: dict(SCENARIO_MULTIPLIERS))
    zone_type_multipliers: Dict[str, float] = field(default_factory=lambda: dict(ZONE_TYPE_MULTIPLIERS))

    def base(self, variable: str) -> Tuple[float, float]:
        if variable in self.variable_overrides:
            return self.variable_overrides[variable]
        if variable not in DEFAULT_THRESHOLDS_SEC:
            # Unknown variable: fall back to the density (slowest) policy so we
            # never silently over-trust an unfamiliar observation.
            return DEFAULT_THRESHOLDS_SEC["density"]
        return DEFAULT_THRESHOLDS_SEC[variable]

    def for_variable(
        self,
        variable: str,
        zone_type: Optional[str] = None,
        scenario_type: Optional[str] = None,
    ) -> Tuple[float, float]:
        """Return (recent_sec, outdated_sec) thresholds for a variable/context."""
        recent, outdated = self.base(variable)
        m = self.zone_type_multipliers.get(zone_type or "unknown", 1.0)
        m = m * self.scenario_multipliers.get(scenario_type or "unknown", 1.0)
        return recent * m, outdated * m

    def is_recent(self, age_sec: float, variable: str,
                  zone_type: Optional[str] = None, scenario_type: Optional[str] = None) -> bool:
        recent, _ = self.for_variable(variable, zone_type, scenario_type)
        return age_sec <= recent

    def is_outdated(self, age_sec: float, variable: str,
                    zone_type: Optional[str] = None, scenario_type: Optional[str] = None) -> bool:
        _, outdated = self.for_variable(variable, zone_type, scenario_type)
        return age_sec > outdated

    def defaults(self) -> Dict[str, Tuple[float, float]]:
        """Clear default table for reporting / documentation."""
        return dict(DEFAULT_THRESHOLDS_SEC)


DEFAULT_FRESHNESS_THRESHOLDS = FreshnessThresholds()