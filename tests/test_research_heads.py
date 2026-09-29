import json
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from backend.compact_model import compact_groups
from backend.context_features import ContextSpec
from backend.research_heads import NEWS_NUMERIC, ResearchHead, long_term_signal
from benchmarks.long_term import evaluate
from benchmarks.news_model import evaluate_news


def features(n=60):
    rng = np.random.default_rng(42)
    result = pd.DataFrame(rng.normal(size=(n, 11)), columns=compact_groups()["sector"])
    for name in NEWS_NUMERIC:
        result[name] = 0.0
    result["article_text"] = ""
    return result


def test_empty_news_and_single_class_fit():
    x = features()
    head = ResearchHead("news_only").fit(x, pd.Series(np.ones(len(x))))
    assert np.all(head.predict(x) == 1)
    model = ResearchHead("price_sector_plus_news", False).fit(
        x, pd.Series(np.linspace(-0.1, 0.1, len(x)))
    )
    assert np.isfinite(model.predict(x)).all()


def test_future_text_never_enters_training_vocabulary():
    x = features()
    x["article_text"] = "earnings growth"
    model = ResearchHead("price_sector_plus_news").fit(
        x, pd.Series(np.tile([-0.01, 0.01], 30))
    )
    future = x.iloc[:1].copy()
    future["article_text"] = "unseenfuturetoken"
    model.predict(future)
    vectorizer = model.model.named_steps["features"].named_transformers_["text"]
    assert "unseenfuturetoken" not in vectorizer.vocabulary_


def test_qualitative_signal_has_uncertainty_and_no_trade_eligibility():
    assert long_term_signal(0.5, 126, "2026-09-11")["signal"] == "uncertain"
    assert long_term_signal(0.7, 252, "2026-09-11")["signal"] == "positive"
    assert long_term_signal(0.2, 126, "2026-09-11")["signal"] == "negative_or_flat"
    assert not long_term_signal(0.9, 252, "2026-09-11")["recommendation_eligible"]
    with pytest.raises(ValueError):
        long_term_signal(float("nan"), 252, "2026-09-11")


def fixture_prices(n=900):
    dates = xcals.get_calendar("XNYS").sessions_in_range("2020-01-02", "2024-12-31")[:n]
    rng = np.random.default_rng(12)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "close": close,
            "high": close + 1,
            "low": close - 1,
            "volume": rng.integers(100, 1000, n),
        }
    )


def spec():
    return ContextSpec.model_validate_json(
        Path("benchmarks/protocols/sector_rolling_context.json").read_bytes()
    )


def test_long_labels_are_purged_and_future_price_cannot_change_forecast():
    prices = fixture_prices()
    protocol = {
        "development_start": prices.date.iloc[600],
        "development_end": prices.date.iloc[-1],
        "protected_start": "2026-09-28",
        "minimum_fit_rows": 40,
        "horizons": [126, 252],
        "step_sessions": 100,
    }
    first = evaluate(prices, "AAPL", {"SPY": prices, "XLK": prices}, spec(), protocol)
    changed = prices.copy()
    changed.loc[601:, ["close", "high", "low"]] *= 1.2
    second = evaluate(
        changed, "AAPL", {"SPY": changed, "XLK": changed}, spec(), protocol
    )
    assert [r["probability"] for r in first] == [r["probability"] for r in second]
    assert all(r["fit_label_end"] <= r["origin_date"] for r in first)
    with pytest.raises(ValueError, match="Protected"):
        evaluate(
            prices,
            "AAPL",
            {"SPY": prices, "XLK": prices},
            spec(),
            {**protocol, "protected_start": prices.date.iloc[-1]},
        )


def test_news_comparison_runs_all_variants_and_rejects_immature_labels():
    prices = fixture_prices(400)
    protocol = json.loads(
        Path("benchmarks/protocols/news_prospective_v1.json").read_text()
    )
    protocol.update(
        eligible_origin_start="2020-01-01",
        fit_sessions=40,
        validation_sessions=10,
        assessment_sessions=10,
        minimum_news_fraction_per_ticker_per_phase=0,
    )
    phases = {
        "fit": prices.date.iloc[260:300].tolist(),
        "validation": prices.date.iloc[306:316].tolist(),
        "assessment": prices.date.iloc[322:332].tolist(),
    }
    out = evaluate_news(
        prices, "AAPL", {"SPY": prices, "XLK": prices}, spec(), [], phases, protocol
    )
    assert len(out) == 2 * 2 * 10 * 7
    assert out.groupby(["phase", "horizon", "model"]).size().eq(10).all()
    assert (out.fit_label_end < out.origin_date).all()
    late = {**phases, "assessment": prices.date.iloc[-10:].tolist()}
    with pytest.raises(ValueError, match="matured"):
        evaluate_news(
            prices, "AAPL", {"SPY": prices, "XLK": prices}, spec(), [], late, protocol
        )


def test_paired_comparison_preserves_matched_ticker_dates():
    from benchmarks.news_train import paired_primary

    symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA"]
    rows = []
    for day in range(63):
        for ticker in symbols:
            for model, p in (
                ("price_sector_plus_news", 0.9),
                ("price_sector_only", 0.5),
                ("historical_class_prior", 0.6),
            ):
                rows.append(
                    {
                        "phase": "assessment",
                        "horizon": 1,
                        "origin_date": f"day{day:03d}",
                        "ticker": ticker,
                        "model": model,
                        "probability": p,
                        "actual_return": 0.01,
                    }
                )
    result = paired_primary(pd.DataFrame(rows), symbols)
    assert result["research_gate_passed"]
    assert not result["production_promotion"]
    with pytest.raises(ValueError, match="matched"):
        paired_primary(pd.DataFrame(rows[:-1]), symbols)


def test_long_inference_refuses_short_history():
    from backend.long_term import predict_long_term

    prices = fixture_prices(400)
    with pytest.raises(ValueError, match="756"):
        predict_long_term(prices, "AAPL", {"SPY": prices, "XLK": prices}, spec())
