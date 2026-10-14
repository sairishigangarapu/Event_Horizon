"""
comparison.py – Target-aligned, unit-aware paired differences between Stream A and B.

DISCLAIMER: Synthetic scaffold only. NOT for real-world use.

Contract:
  - Aligns forecasts on (scenario_id, zone_id, variable, unit, target_at).
  - Respects issued_at / received_at <= evaluation_time (caller must filter).
  - Missing Stream A channels are reported as "unavailable".
  - Outputs BOTH signed_diff AND absolute_diff per variable per pair.
  - A merged "trust probability" is NEVER published as the sole result.
  - Unit mismatches in otherwise matching records are skipped and flagged.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .config import STREAM_A_ID, STREAM_B_ID, VAR_NAMES, VAR_UNITS
from .inference import ForecastRecord


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ChannelDifference:
    """
    Signed and absolute difference for one channel pair.

    signed_diff   = stream_b_value - stream_a_value  (positive → B higher)
    absolute_diff = |stream_b_value - stream_a_value|
    """
    scenario_id:    str
    zone_id:        str
    variable:       str
    unit:           str
    target_at:      datetime
    horizon_sec:    int
    stream_b_value: float
    stream_a_value: float
    signed_diff:    float
    absolute_diff:  float

    def to_dict(self) -> Dict:
        return {
            "scenario_id":    self.scenario_id,
            "zone_id":        self.zone_id,
            "variable":       self.variable,
            "unit":           self.unit,
            "target_at":      self.target_at.isoformat(),
            "horizon_sec":    self.horizon_sec,
            "stream_b_value": self.stream_b_value,
            "stream_a_value": self.stream_a_value,
            "signed_diff":    self.signed_diff,
            "absolute_diff":  self.absolute_diff,
        }


@dataclass
class ComparisonReport:
    """
    Full comparison output for one evaluation call.

    Fields:
        paired:       List of matched channel-wise differences.
        unavailable:  Dict of {(zone_id, variable): "unavailable"} for
                      Stream A channels with no matching record.
        unit_skipped: List of (scenario_id, zone_id, variable) tuples where
                      a unit mismatch prevented pairing.
    """
    paired:       List[ChannelDifference]
    unavailable:  Dict[Tuple[str, str], str]
    unit_skipped: List[Tuple[str, str, str]]

    def to_dict(self) -> Dict:
        return {
            "paired":       [p.to_dict() for p in self.paired],
            "unavailable":  {f"{k[0]}::{k[1]}": v for k, v in self.unavailable.items()},
            "unit_skipped": [list(t) for t in self.unit_skipped],
        }


# ---------------------------------------------------------------------------
# Alignment key type
# ---------------------------------------------------------------------------

_AlignKey = Tuple[str, str, str, datetime]  # (scenario_id, zone_id, variable, target_at)


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------

class ForecastComparator:
    """
    Compares Stream B (architecture) and Stream A (central) forecasts.

    Usage:
        comparator = ForecastComparator()
        report = comparator.compare(stream_b_records, stream_a_records,
                                    evaluation_time=now)
    """

    def compare(
        self,
        stream_b:        List[ForecastRecord],
        stream_a:        List[ForecastRecord],
        evaluation_time: Optional[datetime] = None,
    ) -> ComparisonReport:
        """
        Aligns and computes per-channel differences.

        Args:
            stream_b:        Architecture forecasts (required).
            stream_a:        Central / fixture forecasts (may be partial).
            evaluation_time: If provided, only records with
                             issued_at <= evaluation_time and
                             received_at <= evaluation_time are used.

        Returns:
            ComparisonReport with paired differences, unavailable labels,
            and unit-mismatch skips.
        """
        # Filter by evaluation_time (causal constraint)
        b_filtered = self._filter_causal(stream_b, evaluation_time)
        a_filtered = self._filter_causal(stream_a, evaluation_time)

        # Build lookup for Stream A: key → ForecastRecord
        a_lookup: Dict[_AlignKey, ForecastRecord] = {}
        for rec in a_filtered:
            key = (rec.scenario_id, rec.zone_id, rec.variable, rec.target_at)
            # Last-write wins for same key (deterministic)
            a_lookup[key] = rec

        paired:       List[ChannelDifference] = []
        unavailable:  Dict[Tuple[str, str], str] = {}
        unit_skipped: List[Tuple[str, str, str]] = []

        for b_rec in b_filtered:
            key = (b_rec.scenario_id, b_rec.zone_id, b_rec.variable, b_rec.target_at)

            if key not in a_lookup:
                # Stream A did not provide this channel
                ua_key = (b_rec.zone_id, b_rec.variable)
                unavailable[ua_key] = "unavailable"
                continue

            a_rec = a_lookup[key]

            # Unit mismatch guard
            if b_rec.unit != a_rec.unit:
                unit_skipped.append((
                    b_rec.scenario_id, b_rec.zone_id, b_rec.variable
                ))
                continue

            signed   = round(b_rec.value - a_rec.value, 6)
            absolute = round(abs(signed), 6)

            paired.append(ChannelDifference(
                scenario_id    = b_rec.scenario_id,
                zone_id        = b_rec.zone_id,
                variable       = b_rec.variable,
                unit           = b_rec.unit,
                target_at      = b_rec.target_at,
                horizon_sec    = b_rec.horizon_sec,
                stream_b_value = b_rec.value,
                stream_a_value = a_rec.value,
                signed_diff    = signed,
                absolute_diff  = absolute,
            ))

        return ComparisonReport(
            paired       = paired,
            unavailable  = unavailable,
            unit_skipped = unit_skipped,
        )

    @staticmethod
    def _filter_causal(
        records:         List[ForecastRecord],
        evaluation_time: Optional[datetime],
    ) -> List[ForecastRecord]:
        """
        Filters records to those causally available at evaluation_time.

        Both issued_at and received_at must be <= evaluation_time.
        """
        if evaluation_time is None:
            return records

        # Strip timezone if needed for comparison
        def _naive(dt: datetime) -> datetime:
            from datetime import timezone
            if dt.tzinfo is not None:
                return dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt

        et = _naive(evaluation_time)
        return [
            r for r in records
            if _naive(r.issued_at) <= et and _naive(r.received_at) <= et
        ]


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def summarise_comparison(report: ComparisonReport) -> Dict:
    """
    Returns per-variable summary statistics from a ComparisonReport.

    Outputs mean and std of signed_diff and absolute_diff per variable.
    Never collapses to a single trust probability.
    """
    from collections import defaultdict
    import numpy as np

    signed_by_var:   Dict[str, List[float]] = defaultdict(list)
    absolute_by_var: Dict[str, List[float]] = defaultdict(list)

    for diff in report.paired:
        signed_by_var[diff.variable].append(diff.signed_diff)
        absolute_by_var[diff.variable].append(diff.absolute_diff)

    summary: Dict = {}
    for var in VAR_NAMES:
        if var in signed_by_var:
            s = np.array(signed_by_var[var])
            a = np.array(absolute_by_var[var])
            summary[var] = {
                "n_pairs":          len(s),
                "mean_signed_diff": round(float(s.mean()), 5),
                "std_signed_diff":  round(float(s.std()),  5),
                "mean_abs_diff":    round(float(a.mean()), 5),
                "std_abs_diff":     round(float(a.std()),  5),
            }
        else:
            summary[var] = {"status": "unavailable"}

    summary["_unit_skipped_count"] = len(report.unit_skipped)
    summary["_unavailable_pairs"]  = [
        f"{k[0]}::{k[1]}" for k in report.unavailable
    ]
    return summary
