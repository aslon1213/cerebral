"""Executions — one run of an agent against a task.

Written by a bot with an API key, read by the person who owns the run with
either credential. Issuing keys and creating agents stay on the JWT side, so a
leaked key can record what an agent did but cannot invent a new agent to blame.

The status changes are separate verbs rather than a writable ``status`` field.
The check constraints on ``executions`` are a state machine, and a free-form
PATCH turns "I completed this twice" into whichever constraint fired first,
reported as a 500. A verb per legal step answers 409 and names both states.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from app.core.deps import (
    ExecutionAccess,
    ExecutionsRead,
    ExecutionsWrite,
    ExecutionsWriteAny,
    ExecutionStoreDep,
    GitRepoStoreDep,
    Principal,
    ProjectRepoDep,
    TaskRepoDep,
    ensure_project_owner,
    ensure_task_owner,
)
from app.core.response import Response, error_responses, ok
from app.repo.base import Page, SortOrder
from app.repo.execution import (
    Execution,
    ExecutionComplete,
    ExecutionCreate,
    ExecutionDetail,
    ExecutionFail,
    ExecutionResponse,
    ExecutionSort,
    ExecutionStatus,
    ExecutionUsage,
    ExecutorType,
)

router = APIRouter(
    prefix="/executions", tags=["executions"], responses=error_responses(401, 403)
)


@router.post("", status_code=status.HTTP_201_CREATED, responses=error_responses(400, 404))
async def create_execution(
    body: ExecutionCreate,
    principal: ExecutionsWrite,
    executions: ExecutionStoreDep,
    projects: ProjectRepoDep,
    tasks: TaskRepoDep,
    repos: GitRepoStoreDep,
) -> Response[ExecutionDetail]:
    """Start a run. One call: the execution, its attempt number and its repos.

    ``executor_agent_id`` is taken from the API key rather than the body — a bot
    must not be able to record work under an agent it was not issued for. The
    ref its commits live on is derived from the new execution's id for the same
    reason.

    The routes are flat, so ``project_id`` and ``task_id`` both arrive in the
    body and both are checked: the project has to be the caller's, and the task
    has to actually be in it.
    """
    await ensure_project_owner(projects, body.project_id, principal.user_id)
    await ensure_task_owner(tasks, body.task_id, principal.user_id)

    task = await tasks.get(body.task_id, principal.user_id)
    if task is None or task.project_id != body.project_id:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "That task does not belong to the project given",
        )

    executor_user_id, executor_agent_id = _executor_for(principal, body.executor_type)
    await _ensure_repos_owned(repos, body, principal.user_id)

    execution = Execution(
        task_id=body.task_id,
        created_by=principal.user_id,
        executor_type=body.executor_type,
        executor_user_id=executor_user_id,
        executor_agent_id=executor_agent_id,
        model=body.model,
        provider=body.provider,
        additional_context=body.additional_context,
    )
    created = await executions.create(execution, body.repos)
    links = await executions.repos_of(created.id)
    return ok(ExecutionDetail.from_row(created, links))


@router.get("")
async def list_executions(
    principal: ExecutionsRead,
    executions: ExecutionStoreDep,
    task_id: uuid.UUID | None = None,
    project_id: uuid.UUID | None = None,
    # Aliased: a parameter called `status` would shadow fastapi.status here.
    execution_status: Annotated[
        ExecutionStatus | None, Query(alias="status")
    ] = None,
    agent_id: uuid.UUID | None = None,
    repo_id: uuid.UUID | None = None,
    sort_by: ExecutionSort = ExecutionSort.CREATED_AT,
    order: SortOrder = SortOrder.DESC,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Response[Page[ExecutionResponse]]:
    """Runs the caller owns, newest first.

    Repo links are not inlined here: a project's worth of runs would otherwise
    fan out into a query per row. Fetch one execution to see its git state.
    """
    items, total = await executions.list(
        principal.user_id,
        task_id=task_id,
        project_id=project_id,
        status=execution_status,
        agent_id=agent_id,
        repo_id=repo_id,
        sort_by=sort_by,
        order=order,
        limit=limit,
        offset=offset,
    )
    return ok(
        Page[ExecutionResponse].model_validate(
            {"items": items, "total": total, "limit": limit, "offset": offset}
        )
    )


@router.get("/{execution_id}", responses=error_responses(404))
async def get_execution(
    execution_id: ExecutionAccess,
    principal: ExecutionsRead,
    executions: ExecutionStoreDep,
) -> Response[ExecutionDetail]:
    """One run, with the ref, base, head and landed state of each repo it works in."""
    execution = await executions.get(execution_id, principal.user_id)
    if execution is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Execution not found")
    links = await executions.repos_of(execution_id)
    return ok(ExecutionDetail.from_row(execution, links))


# --------------------------------------------------------------------------- #
# Status changes — one route per legal step
# --------------------------------------------------------------------------- #
@router.post("/{execution_id}/start", responses=error_responses(404, 409))
async def start_execution(
    execution_id: ExecutionAccess,
    principal: ExecutionsWrite,
    executions: ExecutionStoreDep,
) -> Response[ExecutionResponse]:
    """pending → running. Stamps ``started_at``."""
    return await _transition(
        executions, execution_id, principal, ExecutionStatus.RUNNING
    )


@router.post("/{execution_id}/complete", responses=error_responses(404, 409))
async def complete_execution(
    execution_id: ExecutionAccess,
    body: ExecutionComplete,
    principal: ExecutionsWrite,
    executions: ExecutionStoreDep,
) -> Response[ExecutionResponse]:
    """running → succeeded.

    Not reachable from ``waiting_approval`` or ``waiting_input``: a run blocked
    on a human has not finished its work, and completing it there would leave
    the intervention pending for good. Resolve it first, which returns the run
    to running.
    """
    return await _transition(
        executions,
        execution_id,
        principal,
        ExecutionStatus.SUCCEEDED,
        result=body.result,
    )


@router.post("/{execution_id}/fail", responses=error_responses(404, 409))
async def fail_execution(
    execution_id: ExecutionAccess,
    body: ExecutionFail,
    principal: ExecutionsWrite,
    executions: ExecutionStoreDep,
) -> Response[ExecutionResponse]:
    """→ failed, with the reason. The error is required, not inferred."""
    return await _transition(
        executions,
        execution_id,
        principal,
        ExecutionStatus.FAILED,
        error=body.error,
    )


@router.post("/{execution_id}/cancel", responses=error_responses(404, 409))
async def cancel_execution(
    execution_id: ExecutionAccess,
    principal: ExecutionsWrite,
    executions: ExecutionStoreDep,
) -> Response[ExecutionResponse]:
    """→ cancelled. Legal from pending too, for a run abandoned before it began."""
    return await _transition(
        executions, execution_id, principal, ExecutionStatus.CANCELLED
    )


@router.post("/{execution_id}/usage", responses=error_responses(404))
async def add_execution_usage(
    execution_id: ExecutionAccess,
    body: ExecutionUsage,
    principal: ExecutionsWrite,
    executions: ExecutionStoreDep,
) -> Response[ExecutionResponse]:
    """Add a turn's tokens and cost to the run's totals. No state change.

    The numbers are deltas, added in SQL, so a bot reporting as it goes never
    has to know the running total — and two overlapping reports cannot lose one
    another the way a read-modify-write would.
    """
    execution = await executions.add_usage(execution_id, principal.user_id, body)
    if execution is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Execution not found")
    return ok(ExecutionResponse.model_validate(execution))


@router.delete("/{execution_id}", responses=error_responses(404, 409))
async def delete_execution(
    execution_id: ExecutionAccess,
    principal: ExecutionsWriteAny,
    executions: ExecutionStoreDep,
    purge: bool = False,
) -> Response[None]:
    """Delete a run. Refuses while it has history, unless ``purge=true``.

    Without purge this only removes a run that recorded nothing — everything
    under an execution references it under RESTRICT precisely so an audit trail
    cannot evaporate as a side effect of a cleanup. ``purge=true`` says the
    caller means it, and drops the code changes, interventions and events too.
    """
    if not await executions.delete(execution_id, principal.user_id, purge=purge):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Execution not found")
    return ok(None)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _executor_for(
    principal: Principal, executor_type: ExecutorType
) -> tuple[uuid.UUID | None, uuid.UUID | None]:
    """Who this run is attributed to, as (user_id, agent_id).

    ck_executions_executor_matches_type demands exactly one of the two, matching
    the type. Both come from the credential, never from the body.
    """
    if executor_type is ExecutorType.AI_AGENT:
        if principal.agent_id is None:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "This API key is not bound to an agent, so it cannot start an "
                "ai_agent run. Issue a key with an agent_id, or send "
                'executor_type "human".',
            )
        return None, principal.agent_id
    return principal.user_id, None


async def _ensure_repos_owned(
    repos: GitRepoStoreDep, body: ExecutionCreate, user_id: uuid.UUID
) -> None:
    """Refuse to attach a repo the caller does not own.

    The foreign key only says the repo exists; ownership is ours to check, and
    without it one user's run could anchor code changes onto another's repo.
    """
    owners = await repos.owners_of([attachment.repo_id for attachment in body.repos])
    for attachment in body.repos:
        owner_id = owners.get(attachment.repo_id)
        if owner_id is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"Repo {attachment.repo_id} not found"
            )
        if owner_id != user_id:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                "Only the user who connected this repo can run against it",
            )


async def _transition(
    executions: ExecutionStoreDep,
    execution_id: uuid.UUID,
    principal: Principal,
    target: ExecutionStatus,
    **changes: object,
) -> Response[ExecutionResponse]:
    """Apply one status change, or 404 if the run is gone.

    An illegal step raises InvalidTransitionError from the store and leaves
    here as a 409 — the route deliberately does not catch it.
    """
    execution = await executions.transition(
        execution_id,
        principal.user_id,
        target,
        **changes,  # pyright: ignore[reportArgumentType]
    )
    if execution is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Execution not found")
    return ok(ExecutionResponse.model_validate(execution))
