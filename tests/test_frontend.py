"""Browser interaction tests. API fixtures exist only here, never in the app."""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from threading import Thread
from urllib.parse import urlsplit

import pytest

playwright = pytest.importorskip("playwright.sync_api")
ROOT = Path(__file__).resolve().parents[1]


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def frontend_server():
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(QuietHandler, directory=str(ROOT))
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.fixture
def browser_page(frontend_server):
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(
            channel="msedge" if sys.platform == "win32" else None, headless=True
        )
        context = browser.new_context(
            viewport={"width": 1440, "height": 1100}, accept_downloads=True
        )
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        overrides = {}
        requests = []

        def route_handler(route):
            parsed = urlsplit(route.request.url)
            path = parsed.path
            if path in {"/", "/index.html", "/app.js", "/styles.css"}:
                route.continue_()
                return
            requests.append((route.request.method, path, parsed.query))
            if path in overrides:
                code, payload = overrides[path]
            else:
                code, payload = response_for(path)
            route.fulfill(
                status=code, content_type="application/json", body=json.dumps(payload)
            )

        page.route("**/*", route_handler)
        yield page, frontend_server, overrides, requests, errors
        assert not errors
        context.close()
        browser.close()


def response_for(path):
    symbol = path.split("/")[-1]
    close = 202.0 if symbol == "AAPL" else 102.0
    metadata = {
        "latest_price": close,
        "as_of": "2026-09-11",
        "source": "Test market provider",
    }
    if path == "/health":
        return 200, {"status": "ok", "service": "hermes-prediction-api"}
    if path.startswith("/prices/live/"):
        return 200, {
            "ticker": symbol,
            "market_data": metadata,
            "prices": [
                {"date": "2026-09-10", "close": close - 2},
                {"date": "2026-09-11", "close": close},
            ],
        }
    if path.startswith("/predict/direction/live/"):
        return 200, {
            "ticker": symbol,
            "policy": "always_up",
            "direction": "up",
            "up_probability": 0.51,
            "evaluation": {
                "accuracy": 0.5,
                "always_up_accuracy": 0.54,
                "observations": 100,
            },
            "news_status": "insufficient_point_in_time_history",
            "news_training_days": 0,
            "minimum_news_training_days": 30,
            "articles_after_cutoff": 0,
            "evaluation_start": "2026-01-02",
            "evaluation_end": "2026-09-10",
            "as_of": "2026-09-11T20:00:00Z",
        }
    if path.startswith("/predict/live/"):
        return 200, {
            "ticker": symbol,
            "expected_price": close + 1,
            "predicted_return": 0.01,
            "market_data": metadata,
            "five_day": {"predicted_return": -0.02},
            "training_rows": 400,
            "validation": {
                "mae": 0.015,
                "directional_accuracy": 0.51,
                "deep_weight": 0.3,
                "evaluation_rows": 40,
            },
        }
    if path.startswith("/news/refresh/"):
        return 200, {"inserted": 2, "received": 3}
    if path == "/news/import":
        return 200, {"inserted": 1, "duplicates": 0}
    if path.startswith("/news/"):
        return 200, {"provider_configured": False, "articles": [], "article_count": 0}
    if path == "/trader-pipeline/status":
        return 200, {"configured_sources": [], "records": 0, "provider_errors": {}}
    if path == "/trader-pipeline/refresh":
        return 200, {
            "status": {"configured_sources": [], "records": 0, "provider_errors": {}}
        }
    if path == "/backtest/live":
        metrics = {
            "directional_accuracy": 0.51,
            "mae": 0.012,
            "observations": 100,
            "brier_score": None,
        }
        strategies = {
            key: dict(metrics)
            for key in [
                "model",
                "deep_model",
                "hybrid_model",
                "previous_day",
                "buy_and_hold",
                "five_day_model",
                "direction_classifier",
            ]
        }
        strategies["direction_classifier"].update(mae=0.999, brier_score=0.27)
        return 200, {
            "period": "2y",
            "results": {"SPY": {"market_data": metadata, "strategies": strategies}},
            "errors": {"BAD": "No history available"},
        }
    return 404, {"detail": "Not found"}


def test_dashboard_real_fields_and_saved_tickers(browser_page):
    page, url, _, requests, _ = browser_page
    page.goto(url)
    playwright.expect(page.locator("#forecastPrice")).to_have_text("103.00")
    playwright.expect(page.locator("#closeValue")).to_have_text("102.00")
    playwright.expect(page.locator("#forecastMae")).to_have_text("1.50 pp")
    playwright.expect(page.locator("#accuracyDifference")).to_have_text("-4.00 pp")
    playwright.expect(page.locator("#directionConclusion")).to_contain_text(
        "underperformed"
    )
    playwright.expect(page.locator("#scanNews")).to_be_disabled()
    playwright.expect(page.locator("#newsStatus")).to_contain_text("No saved news")
    assert not page.get_by_text("Strong buy", exact=True).count()
    page.locator("#saveTicker").click()
    page.reload()
    playwright.expect(page.locator("#savedCount")).to_have_text("1")
    page.get_by_role("button", name="Remove SPY from saved tickers").click()
    playwright.expect(page.locator("#savedCount")).to_have_text("0")
    page.locator('[data-ticker="AAPL"]').click()
    playwright.expect(page.locator("#closeValue")).to_have_text("202.00")
    playwright.expect(page.locator("#forecastPrice")).to_have_text("203.00")
    page.locator('[data-range="3M"]').click()
    playwright.expect(page.locator('[data-range="3M"]')).to_have_attribute(
        "aria-pressed", "true"
    )
    page.locator("#chartScrubber").fill("0")
    page.locator("#chartScrubber").dispatch_event("input")
    playwright.expect(page.locator("#chartReadout")).to_contain_text("200.00")
    assert any(
        query == "range=3M"
        for _, path, query in requests
        if path == "/prices/live/AAPL"
    )


def test_panel_failure_does_not_hide_price_forecast(browser_page):
    page, url, overrides, _, _ = browser_page
    overrides["/predict/direction/live/SPY"] = (
        422,
        {"detail": "Not enough news-direction history"},
    )
    page.goto(url)
    playwright.expect(page.locator("#forecastContent")).to_be_visible()
    playwright.expect(page.locator("#directionStatus")).to_contain_text("Not enough")
    overrides["/prices/live/AAPL"] = (502, {"detail": "Provider unavailable"})
    overrides["/predict/live/AAPL"] = (502, {"detail": "Provider unavailable"})
    page.locator('[data-ticker="AAPL"]').click()
    playwright.expect(page.locator("#chartStatus")).to_contain_text(
        "Provider unavailable"
    )
    playwright.expect(page.locator("#forecastContent")).to_be_hidden()
    playwright.expect(page.locator("#closeValue")).to_have_text("\u2014")


def test_news_text_import_and_backtest_export(browser_page):
    page, url, overrides, requests, _ = browser_page
    overrides["/news/SPY"] = (
        200,
        {
            "provider_configured": True,
            "article_count": 1,
            "articles": [
                {
                    "title": "<img src=x onerror=alert(1)>",
                    "summary": "Actual provider summary",
                    "url": "javascript:alert(1)",
                    "source": "Test source",
                    "sentiment": -0.5,
                    "published_at": "2026-09-11T12:00:00Z",
                }
            ],
        },
    )
    page.goto(url)
    playwright.expect(page.locator("#newsArticles")).to_contain_text("<img src=x")
    assert page.locator("#newsArticles img").count() == 0
    assert page.locator("#newsArticles a").count() == 0
    page.locator("#scanNews").click()
    playwright.expect(page.locator("#toast")).to_contain_text("2 new articles")
    page.locator("#newsImport").set_input_files(
        {
            "name": "news.json",
            "mimeType": "application/json",
            "buffer": b'{"articles":[{"title":"Import test"}]}',
        }
    )
    playwright.expect(page.locator("#toast")).to_contain_text("1 articles imported")
    page.locator("#runBacktest").click()
    playwright.expect(page.locator("#backtestStatus")).to_contain_text("1 failed")
    playwright.expect(page.locator("#backtestResults")).to_contain_text(
        "Neural return model"
    )
    playwright.expect(page.locator("#backtestResults")).to_contain_text(
        "Always-up baseline"
    )
    playwright.expect(page.locator("#backtestResults")).to_contain_text(
        "No history available"
    )
    assert "99.90 pp" not in page.locator("#backtestResults").inner_text()
    with page.expect_download() as downloaded:
        page.locator("#exportBacktest").click()
    assert json.loads(Path(downloaded.value.path()).read_text())["results"]["SPY"]
    assert ("POST", "/news/import", "") in requests


def test_stale_response_is_ignored(browser_page):
    page, url, _, _, _ = browser_page
    page.add_init_script(
        """
      const original = window.fetch;
      window.fetch = async (...args) => {
        const result = await original(...args);
        if (String(args[0]).includes('/predict/live/SPY')) {
          await new Promise(resolve => setTimeout(resolve, 500));
        }
        return result;
      };
    """
    )
    page.goto(url)
    page.locator('[data-ticker="AAPL"]').click()
    playwright.expect(page.locator("#forecastPrice")).to_have_text("203.00")
    page.wait_for_timeout(700)
    playwright.expect(page.locator("#forecastPrice")).to_have_text("203.00")


def test_mobile_layout_settings_and_account_preview(browser_page):
    page, url, _, _, _ = browser_page
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(url)
    playwright.expect(page.locator("#forecastContent")).to_be_visible()
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    page.locator("#openAccount").click()
    playwright.expect(page.locator("#accountDialog")).to_contain_text(
        "not available yet"
    )
    page.locator('[data-close="accountDialog"]').click()
    page.locator("#openSettings").click()
    page.locator("#apiUrl").fill(url)
    page.get_by_role("button", name="Save and reconnect").click()
    playwright.expect(page.locator("#connectionText")).to_have_text("API connected")
    page.screenshot(
        path=str(ROOT / "tests/browser_artifacts/mobile.png"), full_page=True
    )
    page.set_viewport_size({"width": 1440, "height": 1100})
    page.screenshot(
        path=str(ROOT / "tests/browser_artifacts/desktop.png"), full_page=True
    )
