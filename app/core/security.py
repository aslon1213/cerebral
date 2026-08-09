import hashlib
import hmac
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from pwdlib import PasswordHash

from app.core.config import settings

_password_hash = PasswordHash.recommended()

ACCESS_TOKEN_TYPE = "access"
REFRESH_TOKEN_TYPE = "refresh"

# API keys look like: cbrl_<prefix>_<secret>
#   cbrl    brand marker, so a leaked key is greppable in logs and scanners can
#           be taught to recognise it
#   prefix  12 hex chars, stored in the clear and uniquely indexed -- the handle
#           the key is looked up by, and the only part ever shown again
#   secret  256 bits from secrets.token_urlsafe, never stored
API_KEY_BRAND = "cbrl"
API_KEY_PREFIX_BYTES = 6
API_KEY_SECRET_BYTES = 32


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


def hash_api_key(key: str) -> str:
    """The digest stored for an API key.

    SHA-256, deliberately, where a password would use argon2. The two are not
    the same problem: a password is low-entropy and human-chosen, so the hash
    has to be slow enough to make guessing it expensive. An API key is 256 bits
    from a CSPRNG -- there is no dictionary to walk, and brute force is already
    impossible. Paying argon2's ~50-100ms on every request would instead be a
    tax on the ingest path, which an observer bot hits once per agent event.
    """
    return hashlib.sha256(key.encode()).hexdigest()


def generate_api_key() -> tuple[str, str, str]:
    """A fresh key as ``(full_key, prefix, key_hash)``.

    ``full_key`` is returned to the caller once and never stored; only the
    prefix and the digest are persisted.
    """
    prefix = secrets.token_hex(API_KEY_PREFIX_BYTES)
    secret = secrets.token_urlsafe(API_KEY_SECRET_BYTES)
    full_key = f"{API_KEY_BRAND}_{prefix}_{secret}"
    return full_key, prefix, hash_api_key(full_key)


def parse_api_key(key: str) -> str | None:
    """The prefix of a presented key, or None if it is not shaped like one.

    Lets the lookup be a single indexed read instead of hashing the candidate
    against every key in the table.
    """
    parts = key.split("_", 2)
    if len(parts) != 3:
        return None
    brand, prefix, secret = parts
    if brand != API_KEY_BRAND or not prefix or not secret:
        return None
    if len(prefix) != API_KEY_PREFIX_BYTES * 2:
        return None
    return prefix


def verify_api_key(presented: str, key_hash: str) -> bool:
    """Constant-time comparison, so timing cannot leak the stored digest."""
    return hmac.compare_digest(hash_api_key(presented), key_hash)
