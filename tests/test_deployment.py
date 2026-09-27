import pytest
from fastapi.testclient import TestClient

from backend import main


@pytest.mark.parametrize(
    "method,path",
    [
        ("POST", "/news/import"),
        ("POST", "/news/refresh/SPY"),
        ("POST", "/trader-pipeline/refresh"),
        ("GET", "/backtest/live/SPY"),
        ("GET", "/backtest/live?tickers=SPY&period=1y"),
        ("GET", "/predict/direction/live/SPY?refresh_news_first=true"),
        ("GET", "/recommendations/live?refresh=true"),
        ("GET", "/predict/live/AAPL"),
        ("GET", "/predict/direction/live/AAPL"),
        ("POST", "/predict"),
        ("GET", "/prices/live/AAPL"),
    ],
)
def test_production_rejects_anonymous_admin_operations(
    monkeypatch, tmp_path, method, path
):
    monkeypatch.setenv("HERMES_DEPLOYMENT", "production")
    monkeypatch.setenv("NEWS_DB_PATH", str(tmp_path / "news.sqlite3"))
    monkeypatch.setenv("HERMES_ADMIN_TOKEN", "test-secret")
    client = TestClient(main.app)
    for headers in ({}, {"X-Hermes-Admin-Token": "wrong"}):
        response = client.request(method, path, headers=headers)
        assert response.status_code == 403
        assert "administrator" in response.json()["detail"]


def test_admin_token_and_public_reads(monkeypatch):
    monkeypatch.setenv("HERMES_DEPLOYMENT", "production")
    monkeypatch.setenv("HERMES_ADMIN_TOKEN", "test-secret")
    monkeypatch.setattr(main.pipeline, "refresh", lambda: [])
    client = TestClient(main.app)
    assert client.get("/health").status_code == 200
    assert client.get("/config.js").status_code == 404
    assert (
        client.post(
            "/trader-pipeline/refresh", headers={"X-Hermes-Admin-Token": "test-secret"}
        ).status_code
        == 200
    )
    monkeypatch.delenv("HERMES_ADMIN_TOKEN")
    assert client.post("/trader-pipeline/refresh").status_code == 403


def test_cors_configuration_replaces_local_origins(monkeypatch):
    monkeypatch.setenv("HERMES_DEPLOYMENT", "production")
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    assert main.cors_origins() == []
    monkeypatch.setenv(
        "CORS_ORIGINS", "https://owner.github.io, https://app.example.com ,"
    )
    assert main.cors_origins() == ["https://owner.github.io", "https://app.example.com"]
