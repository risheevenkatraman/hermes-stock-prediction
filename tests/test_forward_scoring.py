import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from backend import forecast_archive
from backend.evaluation import EvaluationProtocol
from benchmarks.forward import (
    load_outcomes,
    run,
    score_records,
    select_archives,
    verify_archive,
)


@pytest.fixture
def archive(monkeypatch, tmp_path):
    class Clock:
        @staticmethod
        def now(tz):
            return datetime(2026, 9, 28, 12, 0, tzinfo=UTC)

    monkeypatch.setattr(forecast_archive, "datetime", Clock)
    protocol = EvaluationProtocol.model_validate_json(
        Path("benchmarks/protocols/price_v1.json").read_bytes()
    )
    prices = pd.DataFrame({"date": ["2026-09-25"], "close": [100.0]})
    payload = {
        "ticker": "SPY",
        "published_at": "2026-09-28T12:00:00+00:00",
        "model_version": "v1",
        "model_name": protocol.candidate,
        "protocol_sha256": hashlib.sha256(
            protocol.model_dump_json().encode()
        ).hexdigest(),
        "market_data": {"as_of": "2026-09-25"},
        "data_hash": hashlib.sha256(prices.to_csv(index=False).encode()).hexdigest(),
        "status": "shadow_only_not_customer_published",
        "forecasts": [
            {
                "horizon": h,
                "predicted_return": 0.02,
                "training_label_end": "2026-09-25",
                "baselines": {
                    "zero_return": 0.0,
                    "historical_mean": 0.01,
                    "previous_horizon": -0.01,
                    "always_up_direction": 1,
                },
            }
            for h in range(1, 6)
        ],
    }
    directory = forecast_archive.archive_forecast(
        payload,
        prices,
        root=tmp_path / "archives",
        protocol=protocol.model_dump(mode="json"),
    )
    record = json.loads((directory / "record.json").read_bytes())
    # Verification treats serialized models as opaque bytes, never executable pickle.
    for h in range(1, 6):
        name = f"model-{h}.joblib"
        (directory / name).write_bytes(b"never deserialize me")
        record["files"][name] = hashlib.sha256(b"never deserialize me").hexdigest()
    (directory / "record.json").write_text(json.dumps(record))
    return directory


def update_record(directory, **changes):
    record = json.loads((directory / "record.json").read_bytes())
    record.update(changes)
    (directory / "record.json").write_text(json.dumps(record))


def test_verify_and_reject_modified_bytes(archive):
    assert verify_archive(archive)["ticker"] == "SPY"
    (archive / "prices.csv").write_text("changed")
    with pytest.raises(ValueError, match="Hash mismatch"):
        verify_archive(archive)


def test_reject_changed_forecast_and_timing(archive):
    record = verify_archive(archive)
    record["forecasts"][0]["predicted_return"] = 0.99
    update_record(archive, forecasts=record["forecasts"])
    with pytest.raises(ValueError, match="Forecasts differ"):
        verify_archive(archive)


def test_reject_eligibility_flag_and_path_escape(archive):
    update_record(archive, eligible_for_prospective_scoring=False)
    with pytest.raises(ValueError, match="eligibility"):
        verify_archive(archive)
    update_record(archive, eligible_for_prospective_scoring=True)
    record = verify_archive(archive)
    record["files"]["../outside"] = "bad"
    update_record(archive, files=record["files"])
    with pytest.raises(ValueError, match="escapes"):
        verify_archive(archive)


def test_earliest_record_chosen_before_outcomes(archive):
    import shutil

    later = archive.parent / "later"
    shutil.copytree(archive, later)
    update_record(later, recorded_at="2026-09-28T12:30:00+00:00")
    selected, audit = select_archives(
        archive.parent.parent, pd.Timestamp("2026-09-28T13:00Z")
    )
    assert [p for p, _ in selected] == [archive]
    assert sorted(r["status"] for r in audit) == [
        "selected",
        "superseded_by_earlier_record",
    ]
    # Corruption must stop evaluation, not quietly substitute the later forecast.
    (archive / "prices.csv").write_text("corrupt")
    with pytest.raises(ValueError, match="Invalid archive"):
        select_archives(archive.parent.parent, pd.Timestamp("2026-09-28T13:00Z"))


def test_model_versions_are_not_pooled(archive):
    import copy

    first = verify_archive(archive)
    second = copy.deepcopy(first)
    second["model_version"] = "v2"
    second["forecasts"][0]["predicted_return"] = -0.02
    _, _, metrics = score_records(
        [(archive, first), (archive, second)],
        {"SPY": {"2026-09-25": 100, "2026-09-28": 102}},
        pd.Timestamp("2026-09-28T20:00Z"),
    )
    assert set(metrics) == {"v1", "v2"}
    for version in metrics.values():
        assert (
            version[first["protocol_sha256"]]["pooled"]["1"]["price_gbt_v1"][
                "observations"
            ]
            == 1
        )


def test_legacy_schema_is_reported_without_inventing_baselines(archive):
    record = verify_archive(archive)
    del record["protocol_sha256"]
    snapshot = json.loads((archive / "snapshot.json").read_bytes())
    del snapshot["protocol_sha256"]
    content = json.dumps(snapshot).encode()
    (archive / "snapshot.json").write_bytes(content)
    record["files"]["snapshot.json"] = hashlib.sha256(content).hexdigest()
    (archive / "record.json").write_text(json.dumps(record))
    selected, audit = select_archives(
        archive.parent.parent, pd.Timestamp("2026-09-28T13:00Z")
    )
    assert selected == []
    assert audit[0]["status"] == "unsupported_legacy_schema"


def test_wrong_target_date_is_rejected(archive):
    record = verify_archive(archive)
    record["forecasts"][0]["target_date"] = "2026-09-29"
    update_record(archive, forecasts=record["forecasts"])
    with pytest.raises(ValueError, match="Target date"):
        verify_archive(archive)


def test_maturity_adjusted_basis_and_paired_baselines(archive):
    record = verify_archive(archive)
    # Split-adjusted origin is 50 in the outcome snapshot, archived price was 100.
    outcomes = {"SPY": {"2026-09-25": 50.0, "2026-09-28": 51.0, "2026-09-29": 52.0}}
    rows, predictions, metrics = score_records(
        [(archive, record)], outcomes, pd.Timestamp("2026-09-28T20:00Z")
    )
    assert rows[0]["actual_return"] == pytest.approx(0.02)
    assert [r["status"] for r in rows] == ["scored"] + ["pending_session_close"] * 4
    assert len(predictions) == 5
    scores = metrics["v1"][record["protocol_sha256"]]["pooled"]["1"]
    assert scores["price_gbt_v1"]["return_mae"] == pytest.approx(0)
    assert scores["zero_return"]["return_mae"] == pytest.approx(0.02)
    assert scores["always_up"]["return_mae"] is None
    assert all(v["observations"] == 1 for v in scores.values())
    rows, _, _ = score_records(
        [(archive, record)], {}, pd.Timestamp("2026-09-28T20:00Z")
    )
    assert rows[0]["status"] == "pending_outcome_data"


def test_no_scoring_before_close_or_recording(archive):
    selected, _ = select_archives(
        archive.parent.parent, pd.Timestamp("2026-09-28T11:59Z")
    )
    assert not selected
    rows, predictions, metrics = score_records(
        [(archive, verify_archive(archive))],
        {"SPY": {"2026-09-25": 50, "2026-09-28": 51}},
        pd.Timestamp("2026-09-28T19:59Z"),
    )
    assert all(r["status"] == "pending_session_close" for r in rows)
    assert predictions == [] and metrics == {}


def test_outcome_metadata_and_validation(tmp_path):
    metadata = {
        "source": "test provider",
        "price_basis": "split_and_dividend_adjusted_close",
        "retrieved_at": "2026-09-28T21:00Z",
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    path = tmp_path / "SPY.csv"
    path.write_text("date,adjusted_close\n2026-09-25,50\n2026-09-28,51\n")
    outcomes, _ = load_outcomes(tmp_path, pd.Timestamp("2026-09-28T22:00Z"))
    assert outcomes["SPY"]["2026-09-28"] == 51
    with pytest.raises(ValueError, match="after as-of"):
        load_outcomes(tmp_path, pd.Timestamp("2026-09-28T20:00Z"))
    path.write_text("date,adjusted_close\n2026-09-29,52\n")
    with pytest.raises(ValueError, match="unfinished"):
        load_outcomes(tmp_path, pd.Timestamp("2026-09-28T22:00Z"))
    path.write_text("date,adjusted_close\n2026-09-25,50\n2026-09-25,51\n")
    with pytest.raises(ValueError, match="Invalid outcome"):
        load_outcomes(tmp_path, pd.Timestamp("2026-09-28T22:00Z"))


def test_audit_writes_reproducible_evidence_without_outcomes(
    archive, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        pd.Timestamp, "now", lambda **kw: pd.Timestamp("2026-09-28T13:00Z")
    )
    directory = run(archive.parent.parent, tmp_path / "scores")
    report = json.loads((directory / "report.json").read_bytes())
    assert report["mode"] == "archive_audit"
    assert report["selected_records"] == 1
    assert report["predictions"] == []
    assert len(report["observations"]) == 5
    hashes = json.loads((directory / "manifest.json").read_bytes())
    for name, expected in hashes.items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
