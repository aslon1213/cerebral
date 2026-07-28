import uuid

import jwt
from fastapi import APIRouter, HTTPException, status

from app.core.deps import CurrentUser, TokenRepoDep, UserRepoDep
from app.core.response import Response, error_responses, ok
from app.core.security import (
    REFRESH_TOKEN_TYPE,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.repo.tokens import LogoutRequest, RefreshRequest, TokenResponse
from app.repo.users import LoginRequest, RegisterRequest, User, UserResponse

router = APIRouter(prefix="/auth", tags=["auth"])


async def _issue_token_pair(tokens: TokenRepoDep, user_id: uuid.UUID) -> TokenResponse:
    access_token = create_access_token(user_id)
    refresh_token, jti, expires_at = create_refresh_token(user_id)
    await tokens.create(jti=jti, user_id=user_id, expires_at=expires_at)
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
    responses=error_responses(409),
)
async def register(
    body: RegisterRequest, users: UserRepoDep
) -> Response[UserResponse]:
    if await users.get_by_name(body.name) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already taken",
        )
    user = User(name=body.name, hashed_password=hash_password(body.password))
    created = await users.create(user)
    return ok(UserResponse.model_validate(created))


@router.post("/login", responses=error_responses(401, 403))
async def login(
    body: LoginRequest, users: UserRepoDep, tokens: TokenRepoDep
) -> Response[TokenResponse]:
    user = await users.get_by_name(body.name)
    if user is None or not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user",
        )
    assert user.id is not None
    return ok(await _issue_token_pair(tokens, user.id))


@router.post("/refresh", responses=error_responses(401))
async def refresh(
    body: RefreshRequest, users: UserRepoDep, tokens: TokenRepoDep
) -> Response[TokenResponse]:
    invalid = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid refresh token",
    )
    try:
        payload = decode_token(body.refresh_token)
    except jwt.PyJWTError:
        raise invalid

    if payload.get("type") != REFRESH_TOKEN_TYPE:
        raise invalid

    jti = payload.get("jti")
    sub = payload.get("sub")
    if not jti or sub is None:
        raise invalid

    stored = await tokens.get_by_jti(jti)
    if stored is None or stored.revoked:
        raise invalid

    try:
        user_id = uuid.UUID(str(sub))
    except ValueError:
        raise invalid

    # Rotate: revoke the presented refresh token before issuing a new pair.
    await tokens.revoke(jti)

    user = await users.get_by_id(user_id)
    if user is None or not user.is_active:
        raise invalid

    assert user.id is not None
    return ok(await _issue_token_pair(tokens, user.id))


@router.post("/logout")
async def logout(body: LogoutRequest, tokens: TokenRepoDep) -> Response[None]:
    # Idempotent: an invalid/expired token simply has nothing to revoke.
    try:
        payload = decode_token(body.refresh_token)
    except jwt.PyJWTError:
        return ok(None)
    jti = payload.get("jti")
    if jti:
        await tokens.revoke(jti)
    return ok(None)


@router.get("/me", responses=error_responses(401, 403))
async def me(current_user: CurrentUser) -> Response[UserResponse]:
    return ok(UserResponse.model_validate(current_user))
