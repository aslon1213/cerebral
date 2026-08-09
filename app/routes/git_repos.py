"""Connecting the git repositories executions run against."""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status
from fastapi import Response as FastAPIResponse

from app.core.deps import CurrentUser, GitRepoAccess, GitRepoStoreDep
from app.core.response import Response, error_responses, ok
from app.repo.base import Page, SortOrder
from app.repo.git_repo import (
    GitRepoConnected,
    GitRepoCreate,
    GitRepoResponse,
    GitRepoSort,
    GitRepoUpdate,
)

router = APIRouter(prefix="/repos", tags=["repos"], responses=error_responses(401, 403))


@router.post("")
async def connect_repo(
    body: GitRepoCreate,
    response: FastAPIResponse,
    current_user: CurrentUser,
    repos: GitRepoStoreDep,
) -> Response[GitRepoConnected]:
    """Register a repo, or return the one already registered under this name.

    Idempotent by name, so a bot can call it unconditionally at startup rather
    than doing a lookup-then-create dance that races with itself. 201 when it
    created the row, 200 when it already existed — and ``created`` in the body
    says which, so a client need not read the status code to branch.

    Reconnecting refreshes ``local_path``, ``default_branch`` and (when sent)
    ``remote_url``: the path is per-machine and goes stale as soon as the same
    repo is cloned somewhere else.
    """
    repo, created = await repos.connect(current_user.id, body)
    response.status_code = (
        status.HTTP_201_CREATED if created else status.HTTP_200_OK
    )
    return ok(
        GitRepoConnected(
            repo=GitRepoResponse.model_validate(repo), created=created
        )
    )


@router.get("")
async def list_repos(
    current_user: CurrentUser,
    repos: GitRepoStoreDep,
    q: Annotated[str | None, Query(max_length=200)] = None,
    name: Annotated[str | None, Query(max_length=200)] = None,
    local_path: Annotated[str | None, Query(max_length=1024)] = None,
    remote_url: Annotated[str | None, Query(max_length=1024)] = None,
    sort_by: GitRepoSort = GitRepoSort.NAME,
    order: SortOrder = SortOrder.ASC,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Response[Page[GitRepoResponse]]:
    """``local_path`` is an exact match, and is how a bot resolves the working
    directory it was pointed at to a repo id. ``q`` is the fuzzy search."""
    items, total = await repos.list(
        current_user.id,
        q=q,
        name=name,
        local_path=local_path,
        remote_url=remote_url,
        sort_by=sort_by,
        order=order,
        limit=limit,
        offset=offset,
    )
    return ok(
        Page[GitRepoResponse].model_validate(
            {"items": items, "total": total, "limit": limit, "offset": offset}
        )
    )


# Declared before /{repo_id} so the literal path is matched first.
@router.get("/by-name/{name}", responses=error_responses(404))
async def get_repo_by_name(
    name: str, current_user: CurrentUser, repos: GitRepoStoreDep
) -> Response[GitRepoResponse]:
    repo = await repos.get_by_name(name, current_user.id)
    if repo is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Repo not found")
    return ok(GitRepoResponse.model_validate(repo))


@router.get("/{repo_id}", responses=error_responses(404))
async def get_repo(
    repo_id: GitRepoAccess, current_user: CurrentUser, repos: GitRepoStoreDep
) -> Response[GitRepoResponse]:
    repo = await repos.get(repo_id, current_user.id)
    if repo is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Repo not found")
    return ok(GitRepoResponse.model_validate(repo))


@router.patch("/{repo_id}", responses=error_responses(404, 409))
async def update_repo(
    repo_id: GitRepoAccess,
    body: GitRepoUpdate,
    current_user: CurrentUser,
    repos: GitRepoStoreDep,
) -> Response[GitRepoResponse]:
    changes = body.model_dump(exclude_unset=True)
    repo = await repos.update(repo_id, current_user.id, changes)
    if repo is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Repo not found")
    return ok(GitRepoResponse.model_validate(repo))


@router.delete("/{repo_id}", responses=error_responses(404, 409))
async def delete_repo(
    repo_id: GitRepoAccess, current_user: CurrentUser, repos: GitRepoStoreDep
) -> Response[None]:
    """Only for repos no execution has touched.

    execution_repos and code_changes reference a repo under RESTRICT: once an
    agent has changed a file here, those changes are anchored to this row and
    removing it would orphan them, so it answers 409.
    """
    if not await repos.delete(repo_id, current_user.id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Repo not found")
    return ok(None)
