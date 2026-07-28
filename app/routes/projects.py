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

router = APIRouter(prefix="/projects", tags=["projects"])


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    body: ProjectCreate, current_user: CurrentUser, projects: ProjectRepoDep
) -> Project:
    project = Project(
        **body.model_dump(exclude={"label_ids"}), created_by=current_user.id
    )
    return await projects.create(project, body.label_ids)


@router.get("", response_model=Page[ProjectResponse])
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
) -> Page[ProjectResponse]:
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
    return Page[ProjectResponse].model_validate(
        {"items": items, "total": total, "limit": limit, "offset": offset}
    )


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: ProjectAccess, current_user: CurrentUser, projects: ProjectRepoDep
) -> Project:
    project = await projects.get(project_id, current_user.id)
    if project is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")
    return project


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: ProjectAccess,
    body: ProjectUpdate,
    current_user: CurrentUser,
    projects: ProjectRepoDep,
) -> Project:
    project = await projects.update(
        project_id,
        current_user.id,
        body.model_dump(exclude_unset=True, exclude={"label_ids"}),
        body.label_ids,
    )
    if project is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")
    return project


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: ProjectAccess, current_user: CurrentUser, projects: ProjectRepoDep
) -> None:
    """Deleting a project also deletes the tasks inside it."""
    if not await projects.delete(project_id, current_user.id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")


@router.put("/{project_id}/labels/{label_id}", response_model=ProjectResponse)
async def attach_label(
    project_id: ProjectAccess,
    label_id: LabelAccess,
    current_user: CurrentUser,
    projects: ProjectRepoDep,
) -> Project:
    """Idempotent: attaching a label twice leaves a single link row."""
    project = await projects.add_label(project_id, current_user.id, label_id)
    if project is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")
    return project


@router.delete("/{project_id}/labels/{label_id}", response_model=ProjectResponse)
async def detach_label(
    project_id: ProjectAccess,
    label_id: LabelAccess,
    current_user: CurrentUser,
    projects: ProjectRepoDep,
) -> Project:
    project = await projects.remove_label(project_id, current_user.id, label_id)
    if project is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")
    return project


@router.get("/{project_id}/tasks", response_model=Page[TaskResponse])
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
) -> Page[TaskResponse]:
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
    return Page[TaskResponse].model_validate(
        {"items": items, "total": total, "limit": limit, "offset": offset}
    )
