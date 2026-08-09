"""What an agent blocks on, and how a person unblocks it.

The asymmetry here is the whole design. An agent **asks** with its API key; a
person **answers** with a session token, and no API key can reach the answering
routes at all. That is why ``OBSERVER_BOT_SCOPES`` carries no scope that would
allow it: a bot that could approve its own request for approval is not asking
for permission, it is narrating.

Three verbs rather than one ``respond`` with a status field, because
``ck_execution_interventions_status_matches_kind`` already says approvals
resolve approved or rejected while questions resolve answered. Separate routes
make the illegal pairing something a caller cannot even express.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from app.core.deps import (
    CurrentUser,
    ExecutionAccess,
    ExecutionsRead,
    ExecutionsWrite,
    InterventionAccess,
    InterventionStoreDep,
)
from app.core.response import Response, error_responses, ok
from app.repo.base import Page
from app.repo.execution import (
    InterventionAnswer,
    InterventionDecision,
    InterventionOpen,
    InterventionResponse,
    InterventionStatus,
)

# Hung off an execution: the agent asking, and the log of what it asked.
execution_router = APIRouter(
    prefix="/executions",
    tags=["interventions"],
    responses=error_responses(401, 403, 404),
)

# The human side. A person's queue across every run they own.
router = APIRouter(
    prefix="/interventions",
    tags=["interventions"],
    responses=error_responses(401, 403, 404),
)


@execution_router.post(
    "/{execution_id}/interventions",
    status_code=status.HTTP_201_CREATED,
    responses=error_responses(409),
)
async def open_intervention(
    execution_id: ExecutionAccess,
    body: InterventionOpen,
    principal: ExecutionsWrite,
    interventions: InterventionStoreDep,
) -> Response[InterventionResponse]:
    """Ask for an approval or an answer, and park the run until it comes.

    Several may be open at once — a real agent batches its tool approvals into
    a single turn — so this never replaces an existing request. The run stays in
    ``waiting_*`` until the last one is resolved.

    Answers 409 for a run that is not under way: there is nothing to interrupt
    in a run that has not started, or has already stopped.
    """
    intervention = await interventions.open(execution_id, principal.user_id, body)
    if intervention is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Execution not found")
    return ok(InterventionResponse.model_validate(intervention))


@execution_router.get("/{execution_id}/interventions")
async def list_execution_interventions(
    execution_id: ExecutionAccess,
    principal: ExecutionsRead,
    interventions: InterventionStoreDep,
    intervention_status: Annotated[
        InterventionStatus | None, Query(alias="status")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Response[Page[InterventionResponse]]:
    """Everything this run has ever been blocked on, oldest first.

    Readable with either credential: an agent that restarts needs to find out
    whether the thing it was waiting for has been answered yet.
    """
    items, total = await interventions.list_for_execution(
        execution_id,
        principal.user_id,
        status=intervention_status,
        limit=limit,
        offset=offset,
    )
    return ok(
        Page[InterventionResponse].model_validate(
            {"items": items, "total": total, "limit": limit, "offset": offset}
        )
    )


@router.get("")
async def list_pending_interventions(
    current_user: CurrentUser,
    interventions: InterventionStoreDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Response[Page[InterventionResponse]]:
    """The inbox — everything waiting on you, across every run, oldest first.

    A person's queue, so a session token and nothing else. Oldest first because
    the agent that has been blocked longest is the one losing the most time.
    """
    items, total = await interventions.list_pending(
        current_user.id, limit=limit, offset=offset
    )
    return ok(
        Page[InterventionResponse].model_validate(
            {"items": items, "total": total, "limit": limit, "offset": offset}
        )
    )


@router.get("/{intervention_id}")
async def get_intervention(
    intervention_id: InterventionAccess,
    current_user: CurrentUser,
    interventions: InterventionStoreDep,
) -> Response[InterventionResponse]:
    intervention = await interventions.get(intervention_id, current_user.id)
    if intervention is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Intervention not found")
    return ok(InterventionResponse.model_validate(intervention))


@router.post("/{intervention_id}/approve", responses=error_responses(409))
async def approve_intervention(
    intervention_id: InterventionAccess,
    body: InterventionDecision,
    current_user: CurrentUser,
    interventions: InterventionStoreDep,
) -> Response[InterventionResponse]:
    """Let the agent go ahead. Approvals and QA reviews only."""
    return await _resolve(
        interventions,
        intervention_id,
        current_user.id,
        InterventionStatus.APPROVED,
        response=body.response,
        reasoning=body.reasoning,
    )


@router.post("/{intervention_id}/reject", responses=error_responses(409))
async def reject_intervention(
    intervention_id: InterventionAccess,
    body: InterventionDecision,
    current_user: CurrentUser,
    interventions: InterventionStoreDep,
) -> Response[InterventionResponse]:
    """Refuse it. The agent carries on, told no."""
    return await _resolve(
        interventions,
        intervention_id,
        current_user.id,
        InterventionStatus.REJECTED,
        response=body.response,
        reasoning=body.reasoning,
    )


@router.post("/{intervention_id}/answer", responses=error_responses(409))
async def answer_intervention(
    intervention_id: InterventionAccess,
    body: InterventionAnswer,
    current_user: CurrentUser,
    interventions: InterventionStoreDep,
) -> Response[InterventionResponse]:
    """Answer a question. Questions only — an approval is approved or rejected."""
    return await _resolve(
        interventions,
        intervention_id,
        current_user.id,
        InterventionStatus.ANSWERED,
        response=body.response,
        reasoning=body.reasoning,
    )


async def _resolve(
    interventions: InterventionStoreDep,
    intervention_id: uuid.UUID,
    user_id: uuid.UUID,
    resolution: InterventionStatus,
    *,
    response: dict[str, object] | None,
    reasoning: str | None,
) -> Response[InterventionResponse]:
    """Apply one resolution.

    Answering something already answered, or reaching for the wrong verb, both
    raise from the store and leave here as a 409 — the route does not catch
    them.
    """
    intervention = await interventions.resolve(
        intervention_id,
        user_id,
        status=resolution,
        resolved_by_user_id=user_id,
        response=response,  # pyright: ignore[reportArgumentType]
        reasoning=reasoning,
    )
    if intervention is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Intervention not found")
    return ok(InterventionResponse.model_validate(intervention))
