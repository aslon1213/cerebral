import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from app.core.deps import (
    CurrentUser,
    LabelAccess,
    ProjectAccess,
    ProjectRepoDep,
    TaskRepoDep,
)
from app.core.response import Response, error_responses, ok
from app.repo.base import Page, SortOrder
from app.repo.priority import PriorityType
from app.repo.project import (
    Project,
    ProjectCreate,
    ProjectResponse,
    ProjectSort,
    ProjectUpdate,
)
from app.repo.task import TaskResponse, TaskSort, TaskStatus

# Every route here needs a bearer token, and answers 403 for a project owned by
# somebody else; the rest is documented per route. 400 is the unknown-label
# case, which any route that accepts label ids can hit.
router = APIRouter(
    prefix="/projects", tags=["projects"], responses=error_responses(401, 403)
)


@router.post("", status_code=status.HTTP_201_CREATED, responses=error_responses(400))
async def create_project(
    body: ProjectCreate, current_user: CurrentUser, projects: ProjectRepoDep
) -> Response[ProjectResponse]:
    project = Project(
        **body.model_dump(exclude={"label_ids"}), created_by=current_user.id
    )
    created = await projects.create(project, body.label_ids)
    return ok(ProjectResponse.model_validate(created))


@router.get("")
async def list_projects(
    current_user: CurrentUser,
    projects: ProjectRepoDep,
    q: Annotated[str | None, Query(max_length=100)] = None,
    priority: PriorityType | None = None,
    label_id: uuid.UUID | None = None,
    sort_by: ProjectSort = ProjectSort.CREATED_AT,
    order: SortOrder = SortOrder.DESC,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Response[Page[ProjectResponse]]:
    items, total = await projects.list(
        current_user.id,
        q=q,
        priority=priority,
        label_id=label_id,
        sort_by=sort_by,
        order=order,
        limit=limit,
        offset=offset,
    )
    return ok(
        Page[ProjectResponse].model_validate(
            {"items": items, "total": total, "limit": limit, "offset": offset}
        )
    )


@router.get("/{project_id}", responses=error_responses(404))
async def get_project(
    project_id: ProjectAccess, current_user: CurrentUser, projects: ProjectRepoDep
) -> Response[ProjectResponse]:
    project = await projects.get(project_id, current_user.id)
    if project is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")
    return ok(ProjectResponse.model_validate(project))


@router.patch("/{project_id}", responses=error_responses(400, 404))
async def update_project(
    project_id: ProjectAccess,
    body: ProjectUpdate,
    current_user: CurrentUser,
    projects: ProjectRepoDep,
) -> Response[ProjectResponse]:
    project = await projects.update(
        project_id,
        current_user.id,
        body.model_dump(exclude_unset=True, exclude={"label_ids"}),
        body.label_ids,
    )
    if project is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")
    return ok(ProjectResponse.model_validate(project))


@router.delete("/{project_id}", responses=error_responses(404))
async def delete_project(
    project_id: ProjectAccess, current_user: CurrentUser, projects: ProjectRepoDep
) -> Response[None]:
    """Deleting a project also deletes the tasks inside it."""
    if not await projects.delete(project_id, current_user.id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")
    return ok(None)


@router.put("/{project_id}/labels/{label_id}", responses=error_responses(404))
async def attach_label(
    project_id: ProjectAccess,
    label_id: LabelAccess,
    current_user: CurrentUser,
    projects: ProjectRepoDep,
) -> Response[ProjectResponse]:
    """Idempotent: attaching a label twice leaves a single link row."""
    project = await projects.add_label(project_id, current_user.id, label_id)
    if project is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")
    return ok(ProjectResponse.model_validate(project))


@router.delete("/{project_id}/labels/{label_id}", responses=error_responses(404))
async def detach_label(
    project_id: ProjectAccess,
    label_id: LabelAccess,
    current_user: CurrentUser,
    projects: ProjectRepoDep,
) -> Response[ProjectResponse]:
    project = await projects.remove_label(project_id, current_user.id, label_id)
    if project is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")
    return ok(ProjectResponse.model_validate(project))


@router.get("/{project_id}/tasks", responses=error_responses(404))
async def list_project_tasks(
    project_id: ProjectAccess,
    current_user: CurrentUser,
    tasks: TaskRepoDep,
    q: Annotated[str | None, Query(max_length=100)] = None,
    priority: PriorityType | None = None,
    task_status: Annotated[TaskStatus | None, Query(alias="status")] = None,
    label_id: uuid.UUID | None = None,
    due_before: datetime | None = None,
    due_after: datetime | None = None,
    sort_by: TaskSort = TaskSort.CREATED_AT,
    order: SortOrder = SortOrder.DESC,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Response[Page[TaskResponse]]:
    items, total = await tasks.list(
        current_user.id,
        project_id=project_id,
        q=q,
        priority=priority,
        status=task_status,
        label_id=label_id,
        due_before=due_before,
        due_after=due_after,
        sort_by=sort_by,
        order=order,
        limit=limit,
        offset=offset,
    )
    return ok(
        Page[TaskResponse].model_validate(
            {"items": items, "total": total, "limit": limit, "offset": offset}
        )
    )
