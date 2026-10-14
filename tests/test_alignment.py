from datetime import datetime, timezone, timedelta
import pytest
from crowd_reliability.models.schema import ForecastRecord, ObservationRecord
from crowd_reliability.services.comparator import ForecastComparator

def test_forecast_alignment_and_differences():
    comparator = ForecastComparator()
    t_base = datetime(2026, 10, 14, 12, 0, 0, tzinfo=timezone.utc)
    target_t = t_base + timedelta(seconds=60)

    arch_f = [
        ForecastRecord(
            forecast_id="f1", stream_id="stream_b_architecture",
            scenario_id="s1", zone_id="concourse", variable="density",
            unit="p/m2", issued_at=t_base, received_at=t_base,
            target_at=target_t, value=3.2, model_or_provider_version="v1"
        )
    ]
    cent_f = [
        ForecastRecord(
            forecast_id="f2", stream_id="stream_a_central",
            scenario_id="s1", zone_id="concourse", variable="density",
            unit="p/m2", issued_at=t_base, received_at=t_base,
            target_at=target_t, value=2.0, model_or_provider_version="v1"
        )
    ]

    diffs = comparator.align_and_compare(arch_f, cent_f)
    assert len(diffs) == 1
    d = diffs[0]
    assert d.zone_id == "concourse"
    assert d.variable == "density"
    assert d.target_at == target_t
    assert d.signed_diff == pytest.approx(1.2)
    assert d.abs_diff == pytest.approx(1.2)

def test_information_assessment_stale_and_conflicts():
    comparator = ForecastComparator()
    t_now = datetime(2026, 10, 14, 12, 5, 0, tzinfo=timezone.utc)

    # Stale observation: observed 70 seconds ago
    obs_stale = ObservationRecord(
        report_id="r_stale", scenario_id="s1", zone_id="concourse",
        source_id="cam_1", variable="density", value=1.5, unit="p/m2",
        observed_at=t_now - timedelta(seconds=70),
        published_at=t_now - timedelta(seconds=69),
        received_at=t_now - timedelta(seconds=68)
    )

    assessment = comparator.assess_information([obs_stale], t_now, "concourse", "density")
    assert assessment.per_variable_flags["stale"] is True
    assert assessment.observation_age_seconds >= 65.0

    # Conflicting observations
    obs_conf1 = ObservationRecord(
        report_id="r1", scenario_id="s1", zone_id="concourse",
        source_id="cam_1", variable="density", value=4.0, unit="p/m2",
        observed_at=t_now - timedelta(seconds=5),
        published_at=t_now - timedelta(seconds=4),
        received_at=t_now - timedelta(seconds=3)
    )
    obs_conf2 = ObservationRecord(
        report_id="r2", scenario_id="s1", zone_id="concourse",
        source_id="warden_patrol", variable="density", value=0.5, unit="p/m2",
        observed_at=t_now - timedelta(seconds=6),
        published_at=t_now - timedelta(seconds=5),
        received_at=t_now - timedelta(seconds=4)
    )

    assessment_conf = comparator.assess_information([obs_conf1, obs_conf2], t_now, "concourse", "density")
    assert assessment_conf.per_variable_flags["conflict"] is True
    assert len(assessment_conf.conflict_ids) > 0
