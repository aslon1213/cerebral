"""Managing the credentials a bot authenticates with.

Managed by a person, over a JWT: issuing and revoking a credential is not
something the credential itself may do, or a leaked key could mint replacements
and outlive its own revocation. The one exception is ``/verify``, which a bot
calls with its own key to check the credential works and see what it may do.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from app.core.deps import (
    AgentStoreDep,
    ApiKeyAuth,
    ApiKeyStoreDep,
    CurrentUser,
)
from app.core.response import Response, error_responses, ok
from app.repo.api_keys import (
    ApiKeyCreate,
    ApiKeyCreated,
    ApiKeyIdentity,
    ApiKeyResponse,
)
from app.repo.base import Page

router = APIRouter(
    prefix="/api-keys", tags=["api-keys"], responses=error_responses(401)
)


@router.post("", status_code=status.HTTP_201_CREATED, responses=error_responses(404))
async def create_api_key(
    body: ApiKeyCreate,
    current_user: CurrentUser,
    keys: ApiKeyStoreDep,
    agents: AgentStoreDep,
) -> Response[ApiKeyCreated]:
    """Issue an API key.

    The plaintext key is in this response and nowhere else: only its SHA-256
    digest is stored, so it cannot be shown again. A caller that loses it has
    to issue another.
    """
    if body.agent_id is not None:
        # Binding a key to an agent lets executions started with it inherit
        # that agent, so the agent has to be one the caller owns.
        owner_id = await agents.get_owner(body.agent_id)
        if owner_id is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Agent not found")
        if owner_id != current_user.id:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                "Only the user who created this agent can issue keys for it",
            )

    key, api_key = await keys.create(
        user_id=current_user.id,
        name=body.name,
        agent_id=body.agent_id,
        scopes=body.scopes,
        expires_at=body.expires_at,
    )
    return ok(
        ApiKeyCreated(key=key, api_key=ApiKeyResponse.model_validate(api_key)),
    )


@router.get("")
async def list_api_keys(
    current_user: CurrentUser,
    keys: ApiKeyStoreDep,
    agent_id: uuid.UUID | None = None,
    include_revoked: bool = False,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Response[Page[ApiKeyResponse]]:
    """Revoked keys are hidden by default but never deleted, so an audit of
    what a key did still resolves after it is turned off."""
    items, total = await keys.list(
        current_user.id,
        agent_id=agent_id,
        include_revoked=include_revoked,
        limit=limit,
        offset=offset,
    )
    return ok(
        Page[ApiKeyResponse].model_validate(
            {"items": items, "total": total, "limit": limit, "offset": offset}
        )
    )


# Declared before /{key_id} so the literal path is matched first.
@router.get("/verify", responses=error_responses(403))
async def verify_api_key(
    principal: ApiKeyAuth, keys: ApiKeyStoreDep
) -> Response[ApiKeyIdentity]:
    """What this API key is — for a bot to call at startup.

    Authenticated by the key itself rather than a JWT: it answers "does my
    credential work, and what may I do with it", which is the question a bot
    has before it starts sending events. Reaching this handler at all means the
    key is valid, unrevoked and unexpired.
    """
    assert principal.key_id is not None  # guaranteed by ApiKeyAuth
    api_key = await keys.get(principal.key_id, principal.user_id)
    assert api_key is not None  # it authenticated a moment ago
    return ok(
        ApiKeyIdentity(
            key_id=api_key.id,
            name=api_key.name,
            prefix=api_key.prefix,
            user_id=api_key.created_by,
            agent_id=api_key.agent_id,
            scopes=sorted(principal.scopes),
            expires_at=api_key.expires_at,
        )
    )


@router.get("/{key_id}", responses=error_responses(404))
async def get_api_key(
    key_id: uuid.UUID, current_user: CurrentUser, keys: ApiKeyStoreDep
) -> Response[ApiKeyResponse]:
    api_key = await keys.get(key_id, current_user.id)
    if api_key is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "API key not found")
    return ok(ApiKeyResponse.model_validate(api_key))


@router.delete("/{key_id}", responses=error_responses(404))
async def revoke_api_key(
    key_id: uuid.UUID, current_user: CurrentUser, keys: ApiKeyStoreDep
) -> Response[ApiKeyResponse]:
    """Turn a key off. Idempotent, and keeps the row: revoking twice reports
    the first revocation rather than moving the timestamp."""
    api_key = await keys.revoke(key_id, current_user.id)
    if api_key is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "API key not found")
    return ok(ApiKeyResponse.model_validate(api_key))
