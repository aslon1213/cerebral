"""Agents — the identity an execution runs as.

Managed by a person over a JWT. A bot does not create its own identity: the key
it authenticates with is already bound to one, so letting it mint agents would
let a leaked credential invent provenance for the history it writes.
"""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from app.core.deps import AgentAccess, AgentStoreDep, CurrentUser
from app.core.response import Response, error_responses, ok
from app.repo.agent import (
    Agent,
    AgentCreate,
    AgentResponse,
    AgentSort,
    AgentUpdate,
)
from app.repo.base import Page, SortOrder

router = APIRouter(
    prefix="/agents", tags=["agents"], responses=error_responses(401, 403)
)


@router.post("", status_code=status.HTTP_201_CREATED, responses=error_responses(409))
async def create_agent(
    body: AgentCreate, current_user: CurrentUser, agents: AgentStoreDep
) -> Response[AgentResponse]:
    """Names are unique per owner, so 409 means you already have this one."""
    agent = Agent(**body.model_dump(), created_by=current_user.id)
    created = await agents.create(agent)
    return ok(AgentResponse.model_validate(created))


@router.get("")
async def list_agents(
    current_user: CurrentUser,
    agents: AgentStoreDep,
    q: Annotated[str | None, Query(max_length=100)] = None,
    is_active: bool | None = None,
    sort_by: AgentSort = AgentSort.NAME,
    order: SortOrder = SortOrder.ASC,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Response[Page[AgentResponse]]:
    items, total = await agents.list(
        current_user.id,
        q=q,
        is_active=is_active,
        sort_by=sort_by,
        order=order,
        limit=limit,
        offset=offset,
    )
    return ok(
        Page[AgentResponse].model_validate(
            {"items": items, "total": total, "limit": limit, "offset": offset}
        )
    )


# Declared before /{agent_id} so the literal path is matched first.
@router.get("/by-name/{name}", responses=error_responses(404))
async def get_agent_by_name(
    name: str, current_user: CurrentUser, agents: AgentStoreDep
) -> Response[AgentResponse]:
    """Resolve an agent the way a bot configured with a name would.

    Saves the bot from carrying a uuid in its config when a name it chose is
    the thing it actually knows.
    """
    agent = await agents.get_by_name(name, current_user.id)
    if agent is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Agent not found")
    return ok(AgentResponse.model_validate(agent))


@router.get("/{agent_id}", responses=error_responses(404))
async def get_agent(
    agent_id: AgentAccess, current_user: CurrentUser, agents: AgentStoreDep
) -> Response[AgentResponse]:
    agent = await agents.get(agent_id, current_user.id)
    if agent is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Agent not found")
    return ok(AgentResponse.model_validate(agent))


@router.patch("/{agent_id}", responses=error_responses(404, 409))
async def update_agent(
    agent_id: AgentAccess,
    body: AgentUpdate,
    current_user: CurrentUser,
    agents: AgentStoreDep,
) -> Response[AgentResponse]:
    changes = body.model_dump(exclude_unset=True)
    agent = await agents.update(agent_id, current_user.id, changes)
    if agent is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Agent not found")
    return ok(AgentResponse.model_validate(agent))


@router.delete("/{agent_id}", responses=error_responses(404, 409))
async def delete_agent(
    agent_id: AgentAccess, current_user: CurrentUser, agents: AgentStoreDep
) -> Response[None]:
    """Only for agents that never ran.

    Executions and API keys reference an agent under RESTRICT, so once it has
    history it answers 409. Set ``is_active = false`` to retire one instead --
    that keeps it out of new runs without breaking the old ones.
    """
    if not await agents.delete(agent_id, current_user.id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Agent not found")
    return ok(None)
