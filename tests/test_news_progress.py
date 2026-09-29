import json

import pytest

from benchmarks.news_progress import DEFAULT_PROTOCOL, assess, verify_files


def protocol():
    result = json.loads(DEFAULT_PROTOCOL.read_text())
    result.update(
        fit_sessions=1,
        validation_sessions=1,
        assessment_sessions=1,
        boundary_embargo_sessions=1,
    )
    return result


def receipts(dates, hour="17:00:00"):
    return [
        {"ticker": t, "observed_at": f"{d}T{hour}Z"}
        for d in dates
        for t in protocol()["symbols"]
    ]


def test_holdout_period_cannot_count_toward_experiment():
    report = assess([], receipts(["2026-09-29"]), protocol(), "2026-09-29T23:00:00Z")
    assert report["session_monitoring"] == []
    assert not report["observation_coverage_ready"]
    invalid = protocol()
    invalid["eligible_origin_start"] = "2026-09-29"
    with pytest.raises(ValueError, match="protected"):
        assess([], [], invalid, "2026-09-29T23:00:00Z")


def test_empty_successful_polls_are_monitored_but_not_news_coverage():
    report = assess([], receipts(["2027-01-04"]), protocol(), "2027-01-04T22:00:00Z")
    assert report["session_monitoring"][0]["qualifies"]
    assert report["phases"]["fit"]["sessions"] == 1
    assert not report["phases"]["fit"]["coverage_gate_passed"]
    assert not report["training_ready"]


def test_late_stale_and_missing_polls_do_not_qualify():
    for observations in (
        receipts(["2027-01-04"], "21:01:00"),
        receipts(["2027-01-04"], "13:00:00"),
        receipts(["2027-01-04"])[:-1],
    ):
        report = assess([], observations, protocol(), "2027-01-04T23:00:00Z")
        assert not report["session_monitoring"][0]["qualifies"]
        assert report["phases"]["fit"]["sessions"] == 0


def test_phases_have_exchange_session_embargo():
    observations = receipts(
        ["2027-01-04", "2027-01-05", "2027-01-06", "2027-01-07", "2027-01-08"]
    )
    report = assess([], observations, protocol(), "2027-01-08T22:00:00Z")
    assert [
        report["phases"][name]["first_date"]
        for name in ("fit", "validation", "assessment")
    ] == ["2027-01-04", "2027-01-06", "2027-01-08"]


def test_progress_artifact_corruption_rejected(tmp_path):
    (tmp_path / "report.json").write_text("{}")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"files": {"report.json": "invalid"}})
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_files(tmp_path)
