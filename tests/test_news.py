from dataclasses import replace
import json

import exchange_calendars as xcals
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from backend.main import app
from backend.news import (
    NewsArticle,
    NewsStore,
    NewsProviderError,
    normalize_alpha_vantage,
)
from backend.news_features import build_news_features, session_closes
from backend.direction_model import (
    predict_direction,
    _select_policy,
    _predict_probability,
)


def article(**overrides):
    values = dict(
        ticker="AAPL",
        title="Company raises earnings outlook",
        summary="Profit exceeds expectations",
        url="https://example.com/news/1",
        source="test",
        published_at="2025-01-06T18:00:00Z",
        available_at="2025-01-06T18:01:00Z",
        sentiment=0.7,
        relevance=0.9,
    )
    return NewsArticle(**(values | overrides))


def prices(rows=220):
    dates = xcals.get_calendar("XNYS").sessions_in_range("2024-01-02", "2025-12-31")[
        :rows
    ]
    rng = np.random.default_rng(3)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(dates))))
    return pd.DataFrame(
        dict(
            date=dates,
            close=close,
            high=close + 1,
            low=close - 1,
            volume=rng.integers(1000, 10000, len(dates)),
        )
    )


def test_provider_uses_ticker_specific_sentiment_and_observation_time():
    payload = {
        "feed": [
            {
                "title": "Mixed story",
                "summary": "",
                "url": "https://example.com/a",
                "time_published": "20250106T180000",
                "overall_sentiment_score": "0.9",
                "ticker_sentiment": [
                    {
                        "ticker": "AAPL",
                        "ticker_sentiment_score": "-0.7",
                        "relevance_score": "0.8",
                    },
                    {
                        "ticker": "MSFT",
                        "ticker_sentiment_score": "0.8",
                        "relevance_score": "0.8",
                    },
                ],
            }
        ]
    }
    result = normalize_alpha_vantage(payload, "AAPL", "2025-02-01T00:00:00Z")
    assert len(result) == 1
    assert result[0].sentiment == -0.7
    assert result[0].available_at == "2025-02-01T00:00:00+00:00"
    with pytest.raises(NewsProviderError):
        normalize_alpha_vantage(
            {"Information": "rate limited"}, "AAPL", "2025-02-01T00:00:00Z"
        )


def test_archive_is_immutable_and_deduplicates_tracking_urls(tmp_path):
    store = NewsStore(tmp_path / "news.db")
    first = article()
    assert store.add([first]) == 1
    assert (
        store.add([replace(first, url=first.url + "?utm_source=x", sentiment=-0.9)])
        == 0
    )
    assert store.articles("AAPL") == [first]
    assert store.articles("MSFT") == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"available_at": "2025-01-06T17:00:00Z"},
        {"sentiment": np.nan},
        {"published_at": "2025-01-06"},
        {"relevance": 2},
        {"url": "javascript:alert(1)"},
    ],
)
def test_invalid_article_is_rejected(overrides):
    with pytest.raises(ValueError):
        article(**overrides)


def test_after_close_news_moves_to_next_session_and_deduplicates():
    frame = pd.DataFrame({"date": ["2025-01-06", "2025-01-07", "2025-01-08"]})
    a = article(
        published_at="2025-01-06T21:01:00Z", available_at="2025-01-06T21:02:00Z"
    )
    features = build_news_features(
        frame, [a, replace(a, url="https://example.com/copy")], "AAPL"
    )
    assert features.news_count.tolist() == [0, 1, 1]
    assert features.news_sentiment.iloc[1] == pytest.approx(0.7)


def test_backfilled_news_does_not_leak_into_history():
    frame = pd.DataFrame({"date": ["2025-01-06", "2025-01-07"]})
    features = build_news_features(
        frame, [article(available_at="2025-02-01T00:00:00Z")], "AAPL"
    )
    assert features.news_count.sum() == 0


def test_calendar_uses_early_closes_and_dst():
    frame = pd.DataFrame({"date": ["2025-07-03", "2025-11-28"]})
    closes = session_closes(frame)
    assert closes[0] == pd.Timestamp("2025-07-03T17:00:00Z")
    assert closes[1] == pd.Timestamp("2025-11-28T18:00:00Z")


def test_direction_returns_explicit_cold_start_and_holdout_baseline():
    result = predict_direction(prices(), [], "AAPL")
    assert result["news_weight"] == 0
    assert result["news_status"] == "insufficient_point_in_time_history"
    assert result["news_up_probability"] is None
    assert 0 <= result["up_probability"] <= 1
    assert result["evaluation"]["observations"] > 30
    assert result["evaluation"]["excess_accuracy"] == pytest.approx(
        result["evaluation"]["accuracy"] - result["evaluation"]["always_up_accuracy"]
    )


def test_policy_keeps_always_up_on_ties():
    assert _select_policy(np.ones(10), np.ones(10) * 0.9, np.ones(10) * 0.8, True) == (
        0,
        0,
    )


def test_policy_can_select_news_when_it_predicts_down_days():
    labels = np.array([0, 1] * 10)
    weight, threshold = _select_policy(
        labels, np.full(20, 0.6), labels.astype(float), True
    )
    assert weight > 0
    assert threshold > 0


def test_evaluation_prices_do_not_change_selected_policy():
    original = prices()
    first = predict_direction(original, [], "AAPL")
    changed = original.copy()
    changed.loc[195:, ["close", "high", "low"]] *= 1.2
    second = predict_direction(changed, [], "AAPL")
    assert first["decision_threshold"] == second["decision_threshold"]
    assert first["news_weight"] == second["news_weight"]


def test_news_classifier_learns_reaction_from_text():
    from backend.model import FEATURE_COLUMNS
    from backend.news_features import NEWS_COLUMNS

    frame = pd.DataFrame(0.0, index=range(120), columns=FEATURE_COLUMNS + NEWS_COLUMNS)
    frame["article_text"] = [
        "earnings beat raised guidance",
        "earnings miss cut guidance",
    ] * 60
    labels = pd.Series([1, 0] * 60)
    prediction = _predict_probability(frame, labels, frame.iloc[:2], with_news=True)
    assert prediction[0] > 0.5 > prediction[1]


def test_news_endpoints_import_and_refresh_without_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("NEWS_DB_PATH", str(tmp_path / "news.db"))
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    client = TestClient(app)
    record = article().__dict__
    response = client.post("/news/import", json={"articles": [record]})
    assert response.status_code == 200
    assert response.json()["inserted"] == 1
    assert client.get("/news/AAPL").json()["article_count"] == 1
    assert client.post("/news/refresh/AAPL").status_code == 503
    bad = dict(record, published_at="bad")
    assert client.post("/news/import", json={"articles": [bad]}).status_code == 422


def test_news_import_without_availability_does_not_backdate(tmp_path, monkeypatch):
    monkeypatch.setenv("NEWS_DB_PATH", str(tmp_path / "news.db"))
    record = article().__dict__.copy()
    record.pop("available_at")
    client = TestClient(app)
    assert client.post("/news/import", json={"articles": [record]}).status_code == 200
    imported = client.get("/news/AAPL").json()["articles"][0]
    assert pd.Timestamp(imported["available_at"]) > pd.Timestamp("2025-02-01T00:00:00Z")


def test_news_training_path_with_dated_history():
    frame = prices()
    closes = session_closes(frame)
    articles = [
        article(
            title=f"Earnings update {i}",
            url=f"https://example.com/{i}",
            published_at=(close - pd.Timedelta(hours=2)).isoformat(),
            available_at=(close - pd.Timedelta(hours=1)).isoformat(),
            sentiment=0.5 if i % 2 else -0.5,
        )
        for i, close in enumerate(closes)
    ]
    result = predict_direction(frame, articles, "AAPL")
    assert result["news_status"] == "trained"
    assert result["news_training_days"] >= 30
    assert 0 <= result["news_up_probability"] <= 1
    assert result["news_features"]["news_count"] > 0


def test_news_relevance_and_company_filter():
    frame = pd.DataFrame({"date": ["2025-01-06"]})
    features = build_news_features(
        frame, [article(ticker="MSFT"), article(relevance=0.1)], "AAPL"
    )
    assert features.news_count.sum() == 0


def test_content_query_parameters_do_not_collapse_distinct_stories():
    first = article(url="https://example.com/article?id=1")
    second = article(url="https://example.com/article?id=2")
    assert first.identity != second.identity


def test_live_direction_endpoint_and_hybrid_integration(tmp_path, monkeypatch):
    from backend import main

    monkeypatch.setenv("NEWS_DB_PATH", str(tmp_path / "news.db"))
    monkeypatch.setattr(main, "fetch_daily_prices", lambda _: prices())
    client = TestClient(app)
    response = client.get("/predict/direction/live/AAPL")
    assert response.status_code == 200
    assert response.json()["news_weight"] == 0
    response = client.get("/predict/live/AAPL?include_news=true")
    assert response.status_code == 200
    assert "hybrid_predicted_return" in response.json()
    assert response.json()["direction_analysis"]["status"] == "experimental"


def test_import_is_atomic_for_invalid_article(tmp_path, monkeypatch):
    monkeypatch.setenv("NEWS_DB_PATH", str(tmp_path / "news.db"))
    first = article().__dict__
    second = dict(first, available_at="2020-01-01T00:00:00Z")
    client = TestClient(app)
    assert (
        client.post("/news/import", json={"articles": [first, second]}).status_code
        == 422
    )
    assert client.get("/news/AAPL").json()["article_count"] == 0


def test_provider_fetch_normalizes_response_and_caps_history(monkeypatch):
    from io import BytesIO
    from backend import news

    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "test-key")
    response = {
        "feed": [
            {
                "title": "Company update",
                "time_published": "20250106T180000",
                "url": "https://example.com/story",
                "ticker_sentiment": [
                    {
                        "ticker": "AAPL",
                        "ticker_sentiment_score": "0.3",
                        "relevance_score": "0.9",
                    }
                ],
            }
        ]
    }
    requests = []

    def open_response(request, timeout):
        requests.append(request.full_url)
        assert timeout == 30
        return BytesIO(json.dumps(response).encode())

    monkeypatch.setattr(news, "urlopen", open_response)
    records = news.fetch_news("AAPL", time_from="2025-01-01T00:00:00Z")
    assert records[0].sentiment == 0.3
    assert "time_from=20250101T0000" in requests[0]
    response["feed"] *= 1000
    with pytest.raises(NewsProviderError, match="narrower"):
        news.fetch_news("AAPL")


def test_provider_error_does_not_expose_api_key(monkeypatch):
    from urllib.error import URLError
    from backend import news

    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "test-key")

    def fail(request, timeout):
        raise URLError("https://provider.invalid/?apikey=test-key")

    monkeypatch.setattr(news, "urlopen", fail)
    response = TestClient(app).post("/news/refresh/AAPL")
    assert response.status_code == 503
    assert "test-key" not in response.text


def test_news_features_preserve_past_when_future_articles_arrive():
    frame = pd.DataFrame({"date": ["2025-01-06", "2025-01-07"]})
    before = build_news_features(frame, [article()], "AAPL")
    future = article(
        title="Future story",
        url="https://example.com/future",
        published_at="2025-01-08T18:00:00Z",
        available_at="2025-01-08T18:01:00Z",
    )
    after = build_news_features(frame, [article(), future], "AAPL")
    pd.testing.assert_frame_equal(before, after)
