"""Git state, and the history it exists to answer.

Two halves that meet in the middle.

Under ``/executions`` is the git state of a run: which repos it works in, where
each one started, where its cerebral ref has got to, and where the work landed
when it was done. Written by the agent as it goes.

Under ``/repos`` is the payoff. Someone opens a file months later and asks why a
line is there; ``/repos/{id}/history?path=`` answers it by joining the change
back to the event that produced it, and therefore to the agent's reasoning. It
returns coordinates, never content — blob ids and commit shas the client renders
the diff from against its own checkout, because git already holds the bytes.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from app.core.deps import (
    CodeHistoryStoreDep,
    ExecutionAccess,
    ExecutionsRead,
    ExecutionsWrite,
    ExecutionStoreDep,
    GitRepoReadAccess,
    GitRepoStoreDep,
    ReposRead,
)
from app.core.response import Response, error_responses, ok
from app.repo.base import Page
from app.repo.execution import (
    CodeChangeContext,
    CodeChangeHistoryEntry,
    CodeChangeResponse,
    ExecutionRepoAttach,
    ExecutionRepoHead,
    ExecutionRepoLand,
    ExecutionRepoResponse,
)

execution_router = APIRouter(
    prefix="/executions", tags=["code"], responses=error_responses(401, 403, 404)
)
repo_router = APIRouter(
    prefix="/repos", tags=["code"], responses=error_responses(401, 403, 404)
)


@execution_router.post(
    "/{execution_id}/repos",
    status_code=status.HTTP_201_CREATED,
    responses=error_responses(409),
)
async def attach_repo(
    execution_id: ExecutionAccess,
    body: ExecutionRepoAttach,
    principal: ExecutionsWrite,
    executions: ExecutionStoreDep,
    repos: GitRepoStoreDep,
) -> Response[ExecutionRepoResponse]:
    """Add a repo to a run already under way.

    An agent does not always know every repo it will touch when it starts — it
    may follow an import into a sibling checkout halfway through. The run's ref
    is the same in every repo, so the commits stay grouped by execution.

    409 if it is already attached: a second ref and a second base commit for one
    repo in one run is not a shape the git model has.
    """
    owner_id = (await repos.owners_of([body.repo_id])).get(body.repo_id)
    if owner_id is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Repo not found")
    if owner_id != principal.user_id:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Only the user who connected this repo can run against it",
        )

    link = await executions.attach_repo(execution_id, principal.user_id, body)
    if link is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Execution not found")
    return ok(ExecutionRepoResponse.model_validate(link))


@execution_router.get("/{execution_id}/repos")
async def list_execution_repos(
    execution_id: ExecutionAccess,
    principal: ExecutionsRead,
    executions: ExecutionStoreDep,
) -> Response[list[ExecutionRepoResponse]]:
    """The git state of this run, one entry per repo.

    Unpaginated: this is a sub-resource of one execution, bounded by how many
    repos an agent can hold open at once, and the same list is inlined in the
    execution itself.
    """
    links = await executions.repos_of(execution_id)
    return ok([ExecutionRepoResponse.model_validate(link) for link in links])


@execution_router.post("/{execution_id}/repos/{repo_id}/head")
async def set_repo_head(
    execution_id: ExecutionAccess,
    repo_id: uuid.UUID,
    body: ExecutionRepoHead,
    principal: ExecutionsWrite,
    executions: ExecutionStoreDep,
) -> Response[ExecutionRepoResponse]:
    """Advance the tip of the run's cerebral ref, as the agent commits.

    These commits live under ``refs/cerebral/executions/<id>``, outside
    ``refs/heads/*``, so nothing an agent does is visible to normal git tooling
    until the run lands.
    """
    link = await executions.set_head(
        execution_id, principal.user_id, repo_id, body.head_commit_sha
    )
    if link is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Repo not attached to this run")
    return ok(ExecutionRepoResponse.model_validate(link))


@execution_router.post("/{execution_id}/repos/{repo_id}/land")
async def land_repo(
    execution_id: ExecutionAccess,
    repo_id: uuid.UUID,
    body: ExecutionRepoLand,
    principal: ExecutionsWrite,
    executions: ExecutionStoreDep,
) -> Response[ExecutionRepoResponse]:
    """Record that the run's work reached the default namespace.

    The run's cerebral range becomes a commit on the default branch, so every
    change this run recorded in this repo is stamped with it in the same
    transaction. A repo link that said "landed" while its changes still read
    NULL would break file history for exactly the runs that finished properly.

    Which commit produced any individual change is a different question, already
    answered by the cerebral commit on that change's event.
    """
    link = await executions.land(
        execution_id,
        principal.user_id,
        repo_id,
        landed_branch=body.landed_branch,
        landed_commit_shas=body.landed_commit_shas,
        merge_commit_sha=body.merge_commit_sha,
    )
    if link is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Repo not attached to this run")
    return ok(ExecutionRepoResponse.model_validate(link))


@execution_router.get("/{execution_id}/changes")
async def list_execution_changes(
    execution_id: ExecutionAccess,
    principal: ExecutionsRead,
    changes: CodeHistoryStoreDep,
    repo_id: uuid.UUID | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Response[Page[CodeChangeResponse]]:
    """Everything this run touched, grouped by repo and in the order it happened."""
    items, total = await changes.list_for_execution(
        execution_id, principal.user_id, repo_id=repo_id, limit=limit, offset=offset
    )
    return ok(
        Page[CodeChangeResponse].model_validate(
            {"items": items, "total": total, "limit": limit, "offset": offset}
        )
    )


@repo_router.get("/{repo_id}/history")
async def repo_history(
    repo_id: GitRepoReadAccess,
    principal: ReposRead,
    changes: CodeHistoryStoreDep,
    path: Annotated[str | None, Query(max_length=1024)] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Response[Page[CodeChangeHistoryEntry]]:
    """Why is this line here?

    Every agent change to ``path``, across every run against this repo, newest
    first — each one next to the reasoning of the event that produced it. Omit
    ``path`` for the whole repo.

    Coordinates, not content: each entry carries both blob ids and both commits,
    and the client renders the diff from its own checkout with
    ``git diff <before_blob> <after_blob>``.
    """
    rows, total = await changes.file_history(
        repo_id, principal.user_id, path=path, limit=limit, offset=offset
    )
    return ok(
        Page[CodeChangeHistoryEntry](
            items=[CodeChangeHistoryEntry.from_row(*row) for row in rows],
            total=total,
            limit=limit,
            offset=offset,
        )
    )


@repo_router.get("/{repo_id}/history/{change_id}")
async def repo_history_entry(
    repo_id: GitRepoReadAccess,
    change_id: uuid.UUID,
    principal: ReposRead,
    changes: CodeHistoryStoreDep,
) -> Response[CodeChangeContext]:
    """One change with everything around it: its event, and the run it came from."""
    row = await changes.get_with_context(repo_id, change_id, principal.user_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Change not found")
    return ok(CodeChangeContext.from_row(*row))
