"""FastAPI entry point for Hermes price trend predictions."""

from __future__ import annotations

from typing import Annotated

import pandas as pd
from fastapi import FastAPI, HTTPException
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
