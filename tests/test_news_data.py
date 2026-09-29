import hashlib
from dataclasses import replace

import pytest

from backend.news import NewsArticle, NewsStore
from benchmarks.news_data import audit, read_archive


def article():
    return NewsArticle(
        ticker="AAPL",
        title="Company reports earnings",
        summary="Example",
        url="https://example.com/story",
        source="fixture",
        published_at="2025-01-06T18:00:00Z",
        available_at="2025-01-06T18:05:00Z",
        sentiment=0.5,
        relevance=0.9,
    )


def test_newly_downloaded_history_is_not_historical_coverage():
    news = replace(article(), available_at="2026-09-29T12:00:00Z")
    row = audit([news], "2025-01-06T00:00:00Z", "2025-01-10T00:00:00Z")["tickers"][0]
    assert row["published_in_development"] == 1
    assert row["published_in_development_observed_after_window"] == 1
    assert row["news_covered_sessions"] == 0


def test_duplicate_url_keeps_earliest_observed_version():
    later = replace(article(), available_at="2026-09-29T12:00:00Z")
    report = audit([later, article()], "2025-01-06T00:00:00Z", "2025-01-10T00:00:00Z")
    assert report["duplicate_urls_removed"] == 1
    assert report["tickers"][0]["news_covered_sessions"] == 3


def test_archive_audit_is_read_only(tmp_path):
    path = tmp_path / "news.sqlite3"
    assert read_archive(path) == []
    assert not path.exists()
    NewsStore(path).add([article()])
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    assert read_archive(path) == [article()]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_holdout_range_rejected():
    with pytest.raises(ValueError, match="holdout"):
        audit([], "2026-09-25T00:00:00Z", "2026-09-28T00:00:00Z")


def test_partial_collection_reports_safe_error(tmp_path, monkeypatch):
    import json
    import sys

    from backend.news import NewsProviderError
    from benchmarks import news_data

    calls = []

    def fetch(symbol, **kwargs):
        calls.append(symbol)
        raise NewsProviderError("https://example.com/?apikey=secret")

    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "fixture")
    monkeypatch.setattr(news_data, "fetch_news", fetch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "news_data",
            "--db",
            str(tmp_path / "missing.sqlite3"),
            "--output-root",
            str(tmp_path / "output"),
            "--tickers",
            "MSFT",
            "NVDA",
            "--collect-from",
            "2026-09-01T00:00:00Z",
            "--collect-to",
            "2026-09-02T00:00:00Z",
        ],
    )
    with pytest.raises(SystemExit) as error:
        news_data.main()
    assert error.value.code == 1
    assert calls == ["MSFT"]
    report_path = next((tmp_path / "output").glob("*/report.json"))
    assert "secret" not in report_path.read_text()
    report = json.loads(report_path.read_text())
    assert report["collection_complete"] is False
    assert report["requested_tickers"] == ["MSFT", "NVDA"]
