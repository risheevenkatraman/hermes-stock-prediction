"""Verify Cognito access tokens; there is no development authentication bypass."""

from functools import lru_cache
import os

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

bearer = HTTPBearer(auto_error=False)


@lru_cache(maxsize=4)
def key_client(issuer: str) -> jwt.PyJWKClient:
    return jwt.PyJWKClient(f"{issuer}/.well-known/jwks.json", timeout=5)


def current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> str:
    if credentials is None:
        raise HTTPException(401, "Sign in to access your workspace.")
    pool = os.getenv("COGNITO_USER_POOL_ID", "")
    client_id = os.getenv("COGNITO_CLIENT_ID", "")
    if not pool or not client_id:
        raise HTTPException(503, "Account services are not configured yet.")
    region = pool.split("_")[0]
    issuer = f"https://cognito-idp.{region}.amazonaws.com/{pool}"
    try:
        key = key_client(issuer).get_signing_key_from_jwt(credentials.credentials)
        claims = jwt.decode(
            credentials.credentials,
            key.key,
            algorithms=["RS256"],
            issuer=issuer,
            options={
                "verify_aud": False,
                "require": ["exp", "iss", "sub", "token_use", "client_id"],
            },
        )
        if (
            claims["token_use"] != "access"
            or claims["client_id"] != client_id
            or not claims["sub"]
        ):
            raise jwt.InvalidTokenError("Wrong token purpose or client")
        return claims["sub"]
    except (jwt.PyJWTError, ValueError) as error:
        raise HTTPException(
            401, "Your session could not be verified. Please sign in again."
        ) from error
