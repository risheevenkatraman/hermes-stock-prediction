import json
from datetime import UTC, datetime

import pytest

from benchmarks import news_assemble, news_collect


def event(title, observed, version, disposition="candidate"):
    return {
        "identity": "a",
        "version_sha256": version,
        "snapshot": "fixture",
        "raw_feed_index": 0,
        "quality_flags": [],
        "disposition": disposition,
        "article": {
            "ticker": "AAPL",
            "title": title,
            "published_at": "2026-09-28T12:00:00Z",
            "available_at": observed,
        },
    }


def test_revisions_and_reversion_are_selected_only_after_observation():
    rows = [
        event("original", "2026-09-28T13:00:00Z", "v1"),
        event("revised", "2026-09-28T15:00:00Z", "v2"),
        event("original", "2026-09-28T17:00:00Z", "v1"),
    ]
    events, versions = news_assemble.assemble_events(rows + [rows[0]])
    assert len(events) == 3
    assert len(versions) == 2
    assert versions[0]["first_observed_at"] == "2026-09-28T13:00:00Z"
    assert versions[0]["observations"] == 2
    assert news_assemble.at_cutoff(events, "2026-09-28T12:59:00Z") == []
    assert (
        news_assemble.at_cutoff(events, "2026-09-28T14:00:00Z")[0]["version_sha256"]
        == "v1"
    )
    assert (
        news_assemble.at_cutoff(events, "2026-09-28T16:00:00Z")[0]["version_sha256"]
        == "v2"
    )
    assert (
        news_assemble.at_cutoff(events, "2026-09-28T18:00:00Z")[0]["version_sha256"]
        == "v1"
    )


def test_review_revision_does_not_resurrect_old_candidate():
    events = [
        event("original", "2026-09-28T13:00:00Z", "v1"),
        event("revised", "2026-09-28T15:00:00Z", "v2", "review"),
    ]
    assert news_assemble.at_cutoff(events, "2026-09-28T16:00:00Z") == []


def test_verified_collection_and_corruption(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "test-key-not-in-response")
    now = datetime.now(UTC)
    payload = {
        "feed": [
            {
                "title": "Apple earnings",
                "summary": "Apple earnings update",
                "url": "https://example.com/apple",
                "source": "fixture",
                "time_published": now.replace(hour=0, minute=0, second=0).strftime(
                    "%Y%m%dT%H%M%S"
                ),
                "ticker_sentiment": [
                    {
                        "ticker": "AAPL",
                        "relevance_score": "0.9",
                        "ticker_sentiment_score": "0.2",
                    }
                ],
            }
        ]
    }
    monkeypatch.setattr(
        news_collect, "fetch_news_payload", lambda *a, **k: (payload, now.isoformat())
    )
    run = news_collect.collect(tmp_path / "inputs", ["AAPL"], 24, 60)
    rows, reason = news_assemble.verified_run(run)
    assert len(rows) == 1 and reason is None
    assembled = news_assemble.build(tmp_path / "inputs", tmp_path / "outputs")
    assert json.loads((assembled / "report.json").read_text())["unique_versions"] == 1
    (run / "AAPL-records.json").write_text("[]")
    with pytest.raises(ValueError, match="hash mismatch"):
        news_assemble.verified_run(run)


def test_interrupted_snapshot_is_reported(tmp_path):
    (tmp_path / "inputs" / "interrupted").mkdir(parents=True)
    output = news_assemble.build(tmp_path / "inputs", tmp_path / "outputs")
    report = json.loads((output / "report.json").read_text())
    assert report["excluded_snapshots"][0]["reason"] == "missing_manifest"
    assert report["training_ready"] is False
