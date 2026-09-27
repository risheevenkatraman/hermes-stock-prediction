"""Customer APIs: user-owned settings and precomputed research only."""
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from .auth import current_user
from .news import ticker_symbol
from .storage import read_record, research, save_record, workspaces

router = APIRouter(prefix="/v1")


class Profile(BaseModel):
    budget: float = Field(default=0, ge=0, le=1_000_000_000, allow_inf_nan=False)
    currency: Literal["USD"] = "USD"
    goal: str = Field(default="", max_length=200)
    horizon: Literal["short", "long"] = "long"
    risk: Literal["conservative", "balanced", "growth"] = "balanced"


class Workspace(BaseModel):
    profile: Profile = Field(default_factory=Profile)
    watchlist: list[str] = Field(default_factory=lambda: ["AAPL", "MSFT", "NVDA", "AMZN"], max_length=50)

    @field_validator("watchlist")
    @classmethod
    def normalize_watchlist(cls, values):
        return list(dict.fromkeys(ticker_symbol(value) for value in values))


@router.get("/workspace", response_model=Workspace)
def get_workspace(user: str = Depends(current_user)):
    record = read_record(workspaces, user)
    return record["payload"] if record else Workspace()


@router.put("/workspace", response_model=Workspace)
def put_workspace(value: Workspace, user: str = Depends(current_user)):
    save_record(workspaces, user, value.model_dump())
    return value


@router.get("/research/{ticker}")
def get_research(ticker: str):
    try:
        symbol = ticker_symbol(ticker)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    record = read_record(research, symbol)
    if not record:
        raise HTTPException(404, "Research has not been published for this ticker yet.")
    payload = dict(record["payload"])
    expiry = datetime.fromisoformat(payload["expires_at"])
    payload["stale"] = datetime.now(timezone.utc) > expiry
    return payload
