import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from app.core.deps import (
    CurrentUser,
    LabelAccess,
    ProjectRepoDep,
    TaskAccess,
    TaskRepoDep,
    ensure_project_owner,
)
from app.repo.base import Page, SortOrder
from app.repo.priority import PriorityType
from app.repo.task import (
    Task,
    TaskCreate,
    TaskResponse,
    TaskSort,
    TaskStatus,
    TaskUpdate,
)

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.post("", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
async def create_task(
    body: TaskCreate,
    current_user: CurrentUser,
    tasks: TaskRepoDep,
    projects: ProjectRepoDep,
) -> Task:
    """Tasks can only be created inside a project the caller created."""
    await ensure_project_owner(projects, body.project_id, current_user.id)
    task = Task(**body.model_dump(exclude={"label_ids"}), created_by=current_user.id)
    return await tasks.create(task, body.label_ids)


@router.get("", response_model=Page[TaskResponse])
async def list_tasks(
    current_user: CurrentUser,
    tasks: TaskRepoDep,
    project_id: uuid.UUID | None = None,
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


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: TaskAccess, current_user: CurrentUser, tasks: TaskRepoDep
) -> Task:
    task = await tasks.get(task_id, current_user.id)
    if task is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Task not found")
    return task


@router.patch("/{task_id}", response_model=TaskResponse)
async def update_task(
    task_id: TaskAccess,
    body: TaskUpdate,
    current_user: CurrentUser,
    tasks: TaskRepoDep,
    projects: ProjectRepoDep,
) -> Task:
    changes = body.model_dump(exclude_unset=True, exclude={"label_ids"})
    # Moving a task is only allowed into another project the caller created.
    if "project_id" in changes:
        await ensure_project_owner(projects, changes["project_id"], current_user.id)

    task = await tasks.update(task_id, current_user.id, changes, body.label_ids)
    if task is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Task not found")
    return task


@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(
    task_id: TaskAccess, current_user: CurrentUser, tasks: TaskRepoDep
) -> None:
    if not await tasks.delete(task_id, current_user.id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Task not found")


@router.put("/{task_id}/labels/{label_id}", response_model=TaskResponse)
async def attach_label(
    task_id: TaskAccess,
    label_id: LabelAccess,
    current_user: CurrentUser,
    tasks: TaskRepoDep,
) -> Task:
    """Idempotent: attaching a label twice leaves a single link row."""
    task = await tasks.add_label(task_id, current_user.id, label_id)
    if task is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Task not found")
    return task


@router.delete("/{task_id}/labels/{label_id}", response_model=TaskResponse)
async def detach_label(
    task_id: TaskAccess,
    label_id: LabelAccess,
    current_user: CurrentUser,
    tasks: TaskRepoDep,
) -> Task:
    task = await tasks.remove_label(task_id, current_user.id, label_id)
    if task is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Task not found")
    return task
