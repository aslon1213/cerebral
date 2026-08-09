"""The transcript — what an agent did, in the order it did it.

This is the bot's hot path. Everything here is shaped by two facts about the
channel it arrives over: it is unreliable, so an append must be safe to retry;
and it is concurrent with the reader, so the transcript is read by cursor, never
by offset.

There is no PUT, PATCH or DELETE, deliberately. A row here is a record of
something that happened, and a correction is a later event saying so. Rewriting
the transcript would defeat the reason for keeping one — and the whole design
exists so that months later a line of code can be traced back to the reasoning
that produced it.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status
from fastapi import Response as FastAPIResponse

from app.core.deps import (
    EventStoreDep,
    ExecutionAccess,
    ExecutionsRead,
    ExecutionsWrite,
)
from app.core.response import Response, error_responses, ok
from app.repo.execution import (
    ActorType,
    EventPage,
    ExecutionEventCreate,
    ExecutionEventDetail,
    ExecutionEventResponse,
    ExecutionEventType,
)

# Same prefix as the executions router: these hang off one execution, and
# splitting the module keeps the transcript endpoints together without nesting
# the routes.
router = APIRouter(
    prefix="/executions", tags=["events"], responses=error_responses(401, 403, 404)
)


@router.post(
    "/{execution_id}/events",
    status_code=status.HTTP_201_CREATED,
    responses=error_responses(400, 409, 422),
)
async def append_event(
    execution_id: ExecutionAccess,
    body: ExecutionEventCreate,
    response: FastAPIResponse,
    principal: ExecutionsWrite,
    events: EventStoreDep,
) -> Response[ExecutionEventDetail]:
    """Append one event, with any code changes it produced.

    **Retrying is safe.** Send a ``client_event_id`` and the append becomes
    idempotent: 201 when it created the event, 200 with the event the first
    attempt wrote when it was a replay. A bot whose response was lost can send
    the same thing again without doubling the transcript.

    Code changes ride along rather than being posted separately, so an event and
    the edits it explains are never half-written. They are pointers into git —
    blob ids and paths, never file content — and their order in the list is the
    order they happened in.
    """
    result = await events.append(execution_id, principal.user_id, body)
    if result is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Execution not found")

    event, changes, created = result
    response.status_code = (
        status.HTTP_201_CREATED if created else status.HTTP_200_OK
    )
    return ok(ExecutionEventDetail.from_row(event, changes))


@router.get("/{execution_id}/events")
async def list_events(
    execution_id: ExecutionAccess,
    principal: ExecutionsRead,
    events: EventStoreDep,
    after_seq: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    event_type: ExecutionEventType | None = None,
    actor_type: ActorType | None = None,
) -> Response[EventPage]:
    """The transcript from ``after_seq`` onwards, oldest first.

    Cursored on ``seq``, not offset. The log is appended to while it is being
    read, and an offset shifts under every arriving event — page two would skip
    or repeat whatever landed in between. Pass the ``next_after_seq`` from one
    page as the ``after_seq`` of the next; start at 0 for the beginning.

    Polling the tail is the same call: ask again with the cursor you already
    have, and an empty page means you are caught up.
    """
    items, has_more = await events.list_after(
        execution_id,
        principal.user_id,
        after_seq=after_seq,
        limit=limit,
        event_type=event_type,
        actor_type=actor_type,
    )
    return ok(
        EventPage(
            items=[ExecutionEventResponse.model_validate(item) for item in items],
            limit=limit,
            next_after_seq=items[-1].seq if items else None,
            has_more=has_more,
        )
    )


@router.get("/{execution_id}/events/{event_id}")
async def get_event(
    execution_id: ExecutionAccess,
    event_id: uuid.UUID,
    principal: ExecutionsRead,
    events: EventStoreDep,
) -> Response[ExecutionEventDetail]:
    """One event with its code changes expanded.

    The changes carry both blob ids, so a client renders the diff from its own
    checkout: ``git diff <before_blob> <after_blob>``. Nothing here holds file
    content — git already has it.
    """
    found = await events.get(execution_id, principal.user_id, event_id)
    if found is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Event not found")
    event, changes = found
    return ok(ExecutionEventDetail.from_row(event, changes))
