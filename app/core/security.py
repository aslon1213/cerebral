import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from pwdlib import PasswordHash

from app.core.config import settings

_password_hash = PasswordHash.recommended()

ACCESS_TOKEN_TYPE = "access"
REFRESH_TOKEN_TYPE = "refresh"


def hash_password(password: str) -> str:
    return _password_hash.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    return _password_hash.verify(password, hashed_password)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _encode(payload: dict[str, Any]) -> str:
    return jwt.encode(payload, settings.jwt.secret, algorithm=settings.jwt.algorithm)


def decode_token(token: str) -> dict[str, Any]:
    """Decode and validate a JWT. Raises jwt.PyJWTError on any problem."""
    return jwt.decode(token, settings.jwt.secret, algorithms=[settings.jwt.algorithm])


def create_access_token(subject: str | uuid.UUID) -> str:
    now = _now()
    payload = {
        "sub": str(subject),
        "type": ACCESS_TOKEN_TYPE,
        "iat": now,
        "exp": now + timedelta(minutes=settings.jwt.access_token_expire_minutes),
    }
    return _encode(payload)


def create_refresh_token(subject: str | uuid.UUID) -> tuple[str, str, datetime]:
    """Return (token, jti, expires_at). The jti is persisted so the token can
    be looked up, rotated, and revoked."""
    now = _now()
    jti = uuid.uuid4().hex
    expires_at = now + timedelta(days=settings.jwt.refresh_token_expire_days)
    payload = {
        "sub": str(subject),
        "type": REFRESH_TOKEN_TYPE,
        "jti": jti,
        "iat": now,
        "exp": expires_at,
    }
    return _encode(payload), jti, expires_at
