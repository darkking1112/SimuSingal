import pytest

from simusignal.simulation import Scenario, simulate


def test_horizon_includes_exact_delivery_and_causal_order():
    result = simulate(Scenario(messages=1, transit_s=.125, service_s=.25, duration_s=.5))
    assert result["summary"]["received"] == 1
    assert result["summary"]["mean_latency_s"] == .5
    assert [event["stage"] for event in result["events"]] == ["sent", "queued", "processing", "forwarded", "received"]


def test_pending_and_not_started_not_counted_as_delivery():
    result = simulate(Scenario(messages=3, interval_s=1, transit_s=.125, duration_s=.125))
    assert result["summary"]["sent"] == 1
    assert result["summary"]["pending"] == 1
    assert result["summary"]["not_started"] == 2
    assert result["summary"]["received"] == 0
    assert result["summary"]["mean_latency_s"] is None


def test_queue_serialization_and_repeatability():
    config = Scenario(messages=5, interval_s=.125, service_s=.25)
    a, b = simulate(config), simulate(config)
    assert a["events"] == b["events"]
    processing = [event["time_s"] for event in a["events"] if event["stage"] == "processing"]
    assert all(right - left >= .25 - 1e-12 for left, right in zip(processing, processing[1:]))
    assert a["summary"]["received"] + a["summary"]["pending"] == a["summary"]["sent"]


@pytest.mark.parametrize("config", [Scenario(messages=0), Scenario(messages=1.5), Scenario(service_s=0), Scenario(duration_s=float("nan"))])
def test_invalid_scenario(config):
    with pytest.raises(ValueError):
        simulate(config)
