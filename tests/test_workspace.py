from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from backend import auth, main, storage


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'workspace.sqlite3').as_posix()}")
    monkeypatch.setenv("HERMES_DEPLOYMENT", "local")
    with TestClient(main.app) as client:
        yield client
    main.app.dependency_overrides.clear()


def test_workspace_requires_authentication(client):
    assert client.get("/v1/workspace").status_code == 401
    assert client.put("/v1/workspace", json={}).status_code == 401


def test_workspace_is_owned_by_verified_user_and_persists(client):
    main.app.dependency_overrides[auth.current_user] = lambda: "alice"
    value = {"profile": {"budget": 2500, "goal": "Save for a house", "horizon": "long", "risk": "balanced"}, "watchlist": ["aapl", "MSFT", "AAPL"]}
    response = client.put("/v1/workspace", json=value)
    assert response.status_code == 200
    assert response.json()["watchlist"] == ["AAPL", "MSFT"]
    assert client.get("/v1/workspace").json()["profile"]["budget"] == 2500
    main.app.dependency_overrides[auth.current_user] = lambda: "bob"
    assert client.get("/v1/workspace").json()["profile"]["budget"] == 0
    client.put("/v1/workspace", json={"profile": {"budget": 10}, "watchlist": []})
    main.app.dependency_overrides[auth.current_user] = lambda: "alice"
    assert client.get("/v1/workspace").json()["profile"]["budget"] == 2500


@pytest.mark.parametrize("payload", [
    {"profile": {"goal": "x" * 201}}, {"profile": {"budget": -1}},
    {"profile": {"risk": "unknown"}}, {"watchlist": ["../secret"]},
    {"watchlist": ["AAPL"] * 51},
])
def test_invalid_profiles_and_watchlists_are_rejected(client, payload):
    main.app.dependency_overrides[auth.current_user] = lambda: "alice"
    assert client.put("/v1/workspace", json=payload).status_code == 422


def test_research_reads_snapshots_without_training(client, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A customer request must never fetch market data or train")
    monkeypatch.setattr(main, "forecast", forbidden)
    monkeypatch.setattr(main, "fetch_daily_prices", forbidden)
    assert client.get("/v1/research/AAPL").status_code == 404
    expiry = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    storage.save_record(storage.research, "AAPL", {"ticker": "AAPL", "expires_at": expiry, "forecasts": []})
    response = client.get("/v1/research/aapl")
    assert response.status_code == 200
    assert response.json()["stale"] is True
    assert response.json()["ticker"] == "AAPL"


@pytest.fixture
def signing_key(monkeypatch):
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    monkeypatch.setenv("COGNITO_USER_POOL_ID", "us-east-1_example")
    monkeypatch.setenv("COGNITO_CLIENT_ID", "hermes-client")
    monkeypatch.setattr(auth, "key_client", lambda _: SimpleNamespace(
        get_signing_key_from_jwt=lambda _: SimpleNamespace(key=private.public_key())
    ))
    return private


def token(key, **overrides):
    claims = {"sub": "alice", "iss": "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_example", "exp": datetime.now(timezone.utc) + timedelta(minutes=10), "token_use": "access", "client_id": "hermes-client"}
    claims.update(overrides)
    return jwt.encode(claims, key, algorithm="RS256")


def test_valid_cognito_access_token(client, signing_key):
    response = client.get("/v1/workspace", headers={"Authorization": f"Bearer {token(signing_key)}"})
    assert response.status_code == 200


@pytest.mark.parametrize("overrides", [
    {"client_id": "another-client"}, {"token_use": "id"},
    {"iss": "https://untrusted.example.com"}, {"sub": ""},
    {"exp": datetime.now(timezone.utc) - timedelta(minutes=1)},
])
def test_invalid_cognito_claims_are_rejected(client, signing_key, overrides):
    response = client.get("/v1/workspace", headers={"Authorization": f"Bearer {token(signing_key, **overrides)}"})
    assert response.status_code == 401


def test_invalid_signature_is_rejected(client, signing_key):
    other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    assert client.get("/v1/workspace", headers={"Authorization": f"Bearer {token(other_key)}"}).status_code == 401
