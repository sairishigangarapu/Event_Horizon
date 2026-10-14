"""
Deterministic current-state tree vector inference.

infer_tree_vector(snapshot) -> values[Z,6], masks[Z,6], channel_metadata

This is the published, frozen inference contract for Persons 3 and 4.
It must exist and be stable BEFORE any tree optimization: fixed channel order
(TREE_CHANNELS), zones sorted by zone_id, no RNG at inference time.
"""
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from crowd_reliability.ml.features import FEATURE_SCHEMA_VERSION, build_feature_row
from crowd_reliability.ml.tree_bank import CHANNEL_METADATA, TREE_CHANNELS, DecisionTreeBank


@dataclass
class TreeSnapshot:
    """
    Observed-only input snapshot for tree inference.

    observations: ObservationRecord ORM objects or plain dicts with keys
        report_id, zone_id, variable, value, observed_at, received_at.
    zone_exit_flags: {zone_id: bool} — True for zones where the
        probability_exit_open channel is applicable.
    """
    scenario_id: str
    as_of_time: datetime
    observations: List[Any]
    zone_ids: List[str]
    zone_exit_flags: Dict[str, bool] = field(default_factory=dict)


@dataclass
class TreeVectorResult:
    """D[z,t] for all zones at one evaluation time. All six channels preserved."""
    scenario_id: str
    evaluation_time: datetime
    zone_ids: List[str]
    values: np.ndarray  # [Z, 6] float
    masks: np.ndarray   # [Z, 6] int
    channel_metadata: List[Dict[str, Any]]
    tree_version: str
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    scaler_version: Optional[str] = None

    def to_zone_records(self) -> List[Dict[str, Any]]:
        """Per-zone export dicts: names, units, values, masks, versions, ids."""
        units = [m["unit"] for m in self.channel_metadata]
        records = []
        for i, z_id in enumerate(self.zone_ids):
            records.append({
                "scenario_id": self.scenario_id,
                "zone_id": z_id,
                "evaluation_time": self.evaluation_time.isoformat(),
                "tree_version": self.tree_version,
                "feature_schema_version": self.feature_schema_version,
                "scaler_version": self.scaler_version,
                "channel_names": list(TREE_CHANNELS),
                "channel_units": units,
                "channel_values": [float(v) for v in self.values[i]],
                "channel_masks": [int(m) for m in self.masks[i]],
            })
        return records

    def to_json(self, indent: int = 2) -> str:
        payload = {
            "scenario_id": self.scenario_id,
            "evaluation_time": self.evaluation_time.isoformat(),
            "tree_version": self.tree_version,
            "feature_schema_version": self.feature_schema_version,
            "scaler_version": self.scaler_version,
            "channel_metadata": self.channel_metadata,
            "zone_ids": self.zone_ids,
            "values": self.values.tolist(),
            "masks": self.masks.astype(int).tolist(),
            "zones": self.to_zone_records(),
        }
        return json.dumps(payload, indent=indent)

    @classmethod
    def from_json(cls, payload: str) -> "TreeVectorResult":
        """Roundtrip deserialization. All six values and masks survive."""
        data = json.loads(payload) if isinstance(payload, str) else payload
        return cls(
            scenario_id=data["scenario_id"],
            evaluation_time=datetime.fromisoformat(data["evaluation_time"]),
            zone_ids=list(data["zone_ids"]),
            values=np.array(data["values"], dtype=np.float64),
            masks=np.array(data["masks"], dtype=np.int64),
            channel_metadata=data["channel_metadata"],
            tree_version=data["tree_version"],
            feature_schema_version=data.get("feature_schema_version", FEATURE_SCHEMA_VERSION),
            scaler_version=data.get("scaler_version"),
        )


def infer_tree_vector(snapshot: TreeSnapshot, bank: DecisionTreeBank) -> TreeVectorResult:
    """
    Deterministic inference over a snapshot of observations.

    - Zones are evaluated in sorted zone_id order.
    - Features are built with the fixed OBSERVED_ONLY_FEATURES order.
    - probability_exit_open is applicability-masked (0) outside exit zones.
    - No randomness: identical snapshot + identical bank => identical output.
    """
    zone_ids = sorted(snapshot.zone_ids)
    values = np.zeros((len(zone_ids), len(TREE_CHANNELS)), dtype=np.float64)
    masks = np.zeros((len(zone_ids), len(TREE_CHANNELS)), dtype=np.int64)

    for i, z_id in enumerate(zone_ids):
        feature_row = build_feature_row(
            z_id, snapshot.as_of_time, snapshot.observations,
            scenario_id=snapshot.scenario_id,
        )
        has_exit = snapshot.zone_exit_flags.get(z_id, False)
        d_vals, d_masks = bank.predict_zone(feature_row, has_exit=has_exit)
        values[i] = d_vals
        masks[i] = d_masks

    scaler_version = None
    if bank.scaler is not None:
        scaler_version = getattr(bank.scaler, "version", None)

    return TreeVectorResult(
        scenario_id=snapshot.scenario_id,
        evaluation_time=snapshot.as_of_time,
        zone_ids=zone_ids,
        values=values,
        masks=masks,
        channel_metadata=bank.channel_metadata(),
        tree_version=bank.version,
        scaler_version=scaler_version,
    )
