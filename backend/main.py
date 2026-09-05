"""FastAPI entry point for Hermes price trend predictions."""

from __future__ import annotations

from typing import Annotated

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .backtest import serialize_metrics, walk_forward_backtest
from .data import MarketDataError, fetch_daily_prices, latest_market_metadata
from .model import InsufficientHistoryError, forecast

app = FastAPI(title="Hermes Prediction API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "null"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
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
    prices: Annotated[list[PriceRow], Field(min_length=60)]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "hermes-prediction-api"}


@app.post("/predict")
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
        "disclaimer": "Educational estimate, not financial advice.",
    }


@app.get("/predict/live/{ticker}")
def predict_live(ticker: str) -> dict:
    """Ingest current daily history and run the price-only prediction."""
    try:
        prices = fetch_daily_prices(ticker)
        result = forecast(prices)
    except (MarketDataError, InsufficientHistoryError, ValueError) as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    metadata = latest_market_metadata(prices)
    return {
        "ticker": ticker.upper(),
        "horizon": "next trading day",
        "direction": result.direction,
        "predicted_return": result.predicted_return,
        "expected_price": result.expected_price,
        "confidence": result.confidence,
        "training_rows": result.training_rows,
        "validation": result.metrics,
        "features": result.feature_snapshot,
        "market_data": metadata,
        "disclaimer": "Educational estimate, not financial advice.",
    }


@app.get("/backtest/live/{ticker}")
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


@app.get("/backtest/live")
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
    strategy_keys = ("model", "previous_day", "buy_and_hold")
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
        }

    return {
        "period": period,
        "results": results,
        "errors": errors,
        "summary": summary,
        "disclaimer": "Historical backtests are not guarantees of future performance.",
    }
