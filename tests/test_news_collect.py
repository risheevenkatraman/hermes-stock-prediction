import hashlib
import json

from benchmarks import news_collect


def payload(title="Apple announces earnings", published="20260928T180000"):
    return {
        "feed": [
            {
                "title": title,
                "summary": "Apple reports quarterly results.",
                "url": "https://example.com/news",
                "source": "fixture",
                "time_published": published,
                "ticker_sentiment": [
                    {
                        "ticker": "AAPL",
                        "ticker_sentiment_score": "0.5",
                        "relevance_score": "0.9",
                    }
                ],
            }
        ]
    }


def prepare(data, observed="2026-09-29T12:00:00Z"):
    return news_collect.prepare(
        data, "AAPL", observed, "2026-09-28T00:00:00Z", "2026-09-29T00:00:00Z"
    )


def test_half_open_boundary_and_precision_flags():
    records, stats = prepare(payload(published="20260929T000000"))
    assert "outside_requested_window" in records[0]["quality_flags"]
    assert "midnight_timestamp_precision_unknown" in records[0]["quality_flags"]
    assert stats["candidate_records"] == 0


def test_version_hash_tracks_content_not_receipt_time():
    first, _ = prepare(payload())
    second, _ = prepare(payload(), "2026-09-30T12:00:00Z")
    revised, _ = prepare(payload("Apple changes guidance"))
    assert first[0]["version_sha256"] == second[0]["version_sha256"]
    assert first[0]["version_sha256"] != revised[0]["version_sha256"]
    assert first[0]["identity"] == revised[0]["identity"]
    assert first[0]["article"]["available_at"] != second[0]["article"]["available_at"]


def test_duplicates_future_year_and_filtered_rows():
    data = payload("Apple outlook for 2030")
    data["feed"] *= 2
    data["feed"].append({"bad": "row"})
    records, stats = prepare(data)
    assert "future_year_in_title_review_revision" in records[0]["quality_flags"]
    assert "duplicate_url_in_response" in records[1]["quality_flags"]
    assert stats["raw_feed_count"] == 3
    assert stats["rows_not_normalized"] == 1


def test_partial_run_saves_payload_redacts_key_and_stops(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "fixture-secret-key")
    calls = []

    def fetch(ticker, **kwargs):
        calls.append(ticker)
        return {"Information": "error fixture-secret-key"}, "2026-09-29T12:00:00Z"

    monkeypatch.setattr(news_collect, "fetch_news_payload", fetch)
    run = news_collect.collect(tmp_path, ["AAPL", "MSFT"], 24, 60)
    assert calls == ["AAPL"]
    assert "fixture-secret-key" not in (run / "AAPL-response.json").read_text()
    assert "[REDACTED]" in (run / "AAPL-response.json").read_text()
    assert not json.loads((run / "report.json").read_text())["collection_complete"]
    manifest = json.loads((run / "manifest.json").read_text())
    assert all(
        hashlib.sha256((run / name).read_bytes()).hexdigest() == value
        for name, value in manifest["files"].items()
    )


def test_capped_response_is_retained_but_incomplete(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "fixture-secret-key")
    data = payload()
    data["feed"] *= 1000
    monkeypatch.setattr(
        news_collect,
        "fetch_news_payload",
        lambda *a, **k: (data, "2026-09-29T12:00:00Z"),
    )
    run = news_collect.collect(tmp_path, ["AAPL"], 24, 60)
    report = json.loads((run / "report.json").read_text())
    assert not report["collection_complete"]
    assert report["requests"][0]["response_at_cap"]
    assert (run / "AAPL-response.json").exists()
