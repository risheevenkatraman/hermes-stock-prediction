"""FastAPI entry point for Hermes price and trader-flow predictions."""

from __future__ import annotations

import asyncio
import os
import secrets
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .backtest import serialize_metrics, walk_forward_backtest
from .data import MarketDataError, fetch_daily_prices, latest_market_metadata
from .direction_model import predict_direction
from .model import InsufficientHistoryError, forecast
from .news import NewsArticle, NewsProviderError, NewsStore, fetch_news, ticker_symbol
from .news_features import session_closes
from .trader_pipeline import build_recommendations, pipeline
from .storage import initialize
from .workspace import router as workspace_router


async def _scheduled_trader_refresh() -> None:
    interval_minutes = float(os.getenv("TRADER_REFRESH_INTERVAL_MINUTES", "0"))
    if interval_minutes <= 0:
        return
    while True:
        await asyncio.to_thread(pipeline.refresh)
        await asyncio.sleep(interval_minutes * 60)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if os.getenv("HERMES_DEPLOYMENT", "local") != "production":
        initialize()
    refresh_task = asyncio.create_task(_scheduled_trader_refresh())
    try:
        yield
    finally:
        refresh_task.cancel()
        with suppress(asyncio.CancelledError):
            await refresh_task


app = FastAPI(title="Hermes Research API", version="0.3.0", lifespan=lifespan)
app.include_router(workspace_router)


def require_admin(request: Request) -> None:
    """Local development stays open; deployed administrative work requires a token."""
    if os.getenv("HERMES_DEPLOYMENT", "local") != "production":
        return
    token = os.getenv("HERMES_ADMIN_TOKEN", "")
    supplied = request.headers.get("X-Hermes-Admin-Token", "")
    if not token or not secrets.compare_digest(supplied.encode(), token.encode()):
        raise HTTPException(
            status_code=403,
            detail="This operation requires server administrator access. "
            "Use an authenticated admin API request; browser accounts are not available yet.",
        )


def cors_origins() -> list[str]:
    if "CORS_ORIGINS" in os.environ:
        return [
            origin.strip()
            for origin in os.environ["CORS_ORIGINS"].split(",")
            if origin.strip()
        ]
    if os.getenv("HERMES_DEPLOYMENT", "local") == "production":
        return []
    return [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "null",
    ]


app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["*"],
)


class PriceRow(BaseModel):
    date: str | None = None
    close: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    volume: float = Field(ge=0)


class PredictionRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=10, pattern=r"^[A-Za-z0-9.-]+$")
    prices: Annotated[list[PriceRow], Field(min_length=60, max_length=10000)]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "hermes-prediction-api"}


@app.get("/trader-pipeline/status")
def trader_pipeline_status() -> dict[str, object]:
    """Report configured sources and the last normalized refresh."""
    return pipeline.status()


@app.post("/trader-pipeline/refresh", dependencies=[Depends(require_admin)])
def refresh_trader_pipeline() -> dict[str, object]:
    """Fetch and normalize the latest records from configured providers."""
    records = pipeline.refresh()
    return {
        "refreshed_at": pipeline.last_refresh,
        "records": len(records),
        "status": pipeline.status(),
        "disclaimer": "Trader disclosures can be delayed or incomplete.",
    }


@app.get("/recommendations/live", dependencies=[Depends(require_admin)])
def recommendations_live(
    request: Request,
    refresh: bool = Query(default=True),
    include_price_model: bool = Query(default=True),
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, object]:
    """Return current recency-weighted trader recommendations."""
    if refresh or not pipeline.records:
        require_admin(request)
        pipeline.refresh()
    if not pipeline.records:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "No trader records are available.",
                "configure": [
                    "QUIVER_QUANT_API_URL and QUIVER_QUANT_API_KEY",
                    "STOCKCIRCLE_API_URL and STOCKCIRCLE_API_KEY",
                    "TRADINGVIEW_API_URL and TRADINGVIEW_API_KEY",
                ],
                "provider_errors": pipeline.provider_errors,
            },
        )
    recommendations = build_recommendations(
        pipeline.records,
        include_price_model=include_price_model,
        limit=limit,
    )
    return {
        "updated_at": pipeline.last_refresh,
        "recommendations": [
            {
                "ticker": item.ticker,
                "action": item.action,
                "score": item.score,
                "trader_signal": item.trader_signal,
                "trader_return": item.trader_return,
                "trader_count": item.trader_count,
                "sources": item.sources,
                "recent_trades": item.recent_trades,
                "price_forecast": item.price_forecast,
            }
            for item in recommendations
        ],
        "disclaimer": "Educational estimate, not financial advice. Disclosed trades may be delayed.",
    }


@app.post("/predict", dependencies=[Depends(require_admin)])
def predict(request: PredictionRequest) -> dict:
    try:
        result = forecast(pd.DataFrame([row.model_dump() for row in request.prices]))
    except InsufficientHistoryError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return {
        "ticker": request.ticker.upper(),
        "horizon": "next trading day",
        "direction": result.direction,
        "predicted_return": result.predicted_return,
        "expected_price": result.expected_price,
        "confidence": result.confidence,
        "training_rows": result.training_rows,
        "validation": result.metrics,
        "features": result.feature_snapshot,
        "five_day": {
            "predicted_return": result.five_day_return,
            "direction": result.five_day_direction,
        },
        "direction_probability": result.direction_probability,
        "deep_learning": {
            "predicted_return": result.deep_predicted_return,
            "validation_mae": result.metrics["deep_mae"],
            "validation_directional_accuracy": result.metrics[
                "deep_directional_accuracy"
            ],
        },
        "hybrid_predicted_return": result.hybrid_predicted_return,
        "disclaimer": "Educational estimate, not financial advice.",
    }


@app.get("/predict/live/{ticker}", dependencies=[Depends(require_admin)])
def predict_live(ticker: str, include_news: bool = False) -> dict:
    """Ingest current daily history and run the price-only prediction."""
    try:
        prices = fetch_daily_prices(ticker)
        if include_news:
            prices = prices.loc[
                session_closes(prices) <= pd.Timestamp.now(tz="UTC")
            ].reset_index(drop=True)
        result = forecast(prices)
    except (MarketDataError, InsufficientHistoryError, ValueError) as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    direction_analysis = None
    direction_error = None
    if include_news:
        try:
            direction_analysis = predict_direction(
                prices, NewsStore().articles(ticker), ticker
            )
        except ValueError as error:
            direction_error = str(error)
    metadata = latest_market_metadata(prices)
    return {
        **(
            {
                "direction_analysis": direction_analysis,
                "direction_error": direction_error,
            }
            if include_news
            else {}
        ),
        "ticker": ticker.upper(),
        "horizon": "next trading day",
        "direction": result.direction,
        "predicted_return": result.predicted_return,
        "expected_price": result.expected_price,
        "confidence": result.confidence,
        "training_rows": result.training_rows,
        "validation": result.metrics,
        "features": result.feature_snapshot,
        "five_day": {
            "predicted_return": result.five_day_return,
            "direction": result.five_day_direction,
        },
        "direction_probability": result.direction_probability,
        "deep_learning": {
            "predicted_return": result.deep_predicted_return,
            "validation_mae": result.metrics["deep_mae"],
            "validation_directional_accuracy": result.metrics[
                "deep_directional_accuracy"
            ],
        },
        "hybrid_predicted_return": result.hybrid_predicted_return,
        "market_data": metadata,
        "disclaimer": "Educational estimate, not financial advice.",
    }


@app.get("/prices/live/{ticker}", dependencies=[Depends(require_admin)])
def prices_live(
    ticker: str,
    range: str = Query(default="1D", pattern=r"^(1D|1W|1M|3M)$"),
) -> dict:
    """Return the recent daily bars used to render the trend chart."""
    provider_period = "1y"
    visible_rows = {"1D": 2, "1W": 5, "1M": 22, "3M": 66}[range]
    try:
        prices = fetch_daily_prices(ticker, period=provider_period)
    except (MarketDataError, ValueError) as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    rows = prices.tail(visible_rows)
    return {
        "ticker": ticker.upper(),
        "range": range,
        "market_data": latest_market_metadata(prices),
        "prices": [
            {
                "date": str(row["date"]),
                "close": round(float(row["close"]), 2),
            }
            for _, row in rows.iterrows()
        ],
    }


@app.get("/backtest/live/{ticker}", dependencies=[Depends(require_admin)])
def backtest_live(ticker: str) -> dict:
    """Evaluate the model and benchmarks against the latest daily history."""
    try:
        prices = fetch_daily_prices(ticker, period="5y")
        metrics = walk_forward_backtest(prices)
    except (MarketDataError, ValueError) as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    return {
        "ticker": ticker.upper(),
        "market_data": latest_market_metadata(prices),
        "strategies": serialize_metrics(metrics),
        "disclaimer": "Historical backtests are not guarantees of future performance.",
    }


@app.get("/backtest/live", dependencies=[Depends(require_admin)])
def backtest_multiple(
    tickers: str = Query(
        default="SPY,QQQ,AAPL,MSFT,NVDA,AMZN,TSLA",
        description="Comma-separated ticker symbols.",
    ),
    period: str = Query(default="5y", pattern=r"^(1y|2y|5y|10y|max)$"),
) -> dict:
    """Run the same walk-forward evaluation across several live symbols."""
    symbols = [
        symbol.strip().upper() for symbol in tickers.split(",") if symbol.strip()
    ]
    if not symbols or len(symbols) > 20:
        raise HTTPException(
            status_code=422,
            detail="Provide between 1 and 20 comma-separated tickers.",
        )

    results: dict[str, dict] = {}
    errors: dict[str, str] = {}
    for symbol in dict.fromkeys(symbols):
        try:
            prices = fetch_daily_prices(symbol, period=period)
            metrics = walk_forward_backtest(prices)
            results[symbol] = {
                "market_data": latest_market_metadata(prices),
                "strategies": serialize_metrics(metrics),
            }
        except (MarketDataError, ValueError) as error:
            errors[symbol] = str(error)

    if not results:
        raise HTTPException(
            status_code=502,
            detail={"message": "No ticker could be backtested.", "errors": errors},
        )

    summary: dict[str, dict[str, float | int]] = {}
    strategy_keys = tuple(next(iter(results.values()))["strategies"])
    for strategy in strategy_keys:
        values = [result["strategies"][strategy] for result in results.values()]
        summary[strategy] = {
            "tickers_evaluated": len(values),
            "average_mae": round(
                sum(float(value["mae"]) for value in values) / len(values),
                6,
            ),
            "average_directional_accuracy": round(
                sum(float(value["directional_accuracy"]) for value in values)
                / len(values),
                4,
            ),
            "average_cumulative_return": round(
                sum(float(value["cumulative_return"]) for value in values)
                / len(values),
                4,
            ),
            "average_max_drawdown": round(
                sum(float(value["max_drawdown"]) for value in values) / len(values),
                4,
            ),
            "average_annualized_volatility": round(
                sum(float(value["annualized_volatility"]) for value in values)
                / len(values),
                4,
            ),
            "average_brier_score": (
                round(
                    sum(
                        float(value["brier_score"])
                        for value in values
                        if value["brier_score"] is not None
                    )
                    / sum(value["brier_score"] is not None for value in values),
                    6,
                )
                if any(value["brier_score"] is not None for value in values)
                else None
            ),
        }

    return {
        "period": period,
        "results": results,
        "errors": errors,
        "summary": summary,
        "disclaimer": "Historical backtests are not guarantees of future performance.",
    }


class NewsInput(BaseModel):
    ticker: str = Field(min_length=1, max_length=10)
    title: str = Field(min_length=1, max_length=2000)
    summary: str = Field(default="", max_length=20000)
    url: str = Field(max_length=4000)
    source: str = Field(max_length=200)
    published_at: str
    available_at: str | None = None
    sentiment: float = Field(ge=-1, le=1, allow_inf_nan=False)
    relevance: float = Field(ge=0, le=1, allow_inf_nan=False)
    sentiment_model: str = Field(default="imported", max_length=200)


class NewsImportRequest(BaseModel):
    articles: Annotated[list[NewsInput], Field(min_length=1, max_length=5000)]


@app.post("/news/import", dependencies=[Depends(require_admin)])
def import_news(request: NewsImportRequest) -> dict:
    """Import scored articles. Omitted availability means first observed now."""
    now = datetime.now(timezone.utc)
    try:
        articles = []
        for row in request.articles:
            data = row.model_dump()
            data["available_at"] = data["available_at"] or now.isoformat()
            article = NewsArticle(**data)
            if pd.Timestamp(article.available_at) > now:
                raise ValueError("Article availability cannot be in the future.")
            articles.append(article)
        inserted = NewsStore().add(articles)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {
        "received": len(articles),
        "inserted": inserted,
        "duplicates": len(articles) - inserted,
    }


@app.post("/news/refresh/{ticker}", dependencies=[Depends(require_admin)])
def refresh_news(
    ticker: str, time_from: str | None = None, time_to: str | None = None
) -> dict:
    try:
        articles = fetch_news(ticker, time_from=time_from, time_to=time_to)
        store = NewsStore()
        inserted = store.add(articles)
        return {"inserted": inserted, "received": len(articles), **store.status(ticker)}
    except NewsProviderError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/news/{ticker}")
def company_news(ticker: str, limit: int = Query(default=20, ge=1, le=100)) -> dict:
    try:
        store = NewsStore()
        articles = store.articles(ticker)
        return {
            **store.status(ticker),
            "articles": [asdict(a) for a in articles[-limit:][::-1]],
            "article_count": len(articles),
        }
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/predict/direction/live/{ticker}", dependencies=[Depends(require_admin)])
def live_direction(
    ticker: str, request: Request, refresh_news_first: bool = False
) -> dict:
    """Experimental direction head; news refresh is explicit and never implicit."""
    if refresh_news_first:
        require_admin(request)
    try:
        symbol = ticker_symbol(ticker)
        store = NewsStore()
        if refresh_news_first:
            store.add(fetch_news(symbol))
        prices = fetch_daily_prices(symbol)
        # In-progress provider daily bars must not become labeled training bars.
        closes = session_closes(prices)
        prices = prices.loc[closes <= pd.Timestamp.now(tz="UTC")].reset_index(drop=True)
        return predict_direction(prices, store.articles(symbol), symbol)
    except NewsProviderError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except MarketDataError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
