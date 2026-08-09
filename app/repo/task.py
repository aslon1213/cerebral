import uuid
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField
from sqlalchemy import Case, String, or_
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.orm import selectinload
from sqlmodel import DateTime, Field, Relationship, col, func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.repo.priority import PriorityType, priority_rank

from .base import SortOrder, TimestampMixin
from .labels import Label, LabelResponse, TaskLabel, resolve_owned_labels

if TYPE_CHECKING:
    from .project import Project


class TaskStatus(StrEnum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    CANCELLED = "cancelled"


class Task(TimestampMixin, table=True):
    __tablename__ = "tasks"  # pyright: ignore[reportAssignmentType]
    id: uuid.UUID = Field(primary_key=True, default_factory=uuid.uuid4)
    project_id: uuid.UUID = Field(
        foreign_key="projects.id", index=True, ondelete="CASCADE"
    )
    name: str = Field(max_length=200, index=True)
    description: str | None = Field(default=None)
    priority: PriorityType = Field(default=PriorityType.LOW, sa_type=String(16))  # pyright: ignore[reportArgumentType]
    status: TaskStatus = Field(default=TaskStatus.TODO, sa_type=String(16))  # pyright: ignore[reportArgumentType]
    due_date: datetime | None = Field(default=None, sa_type=DateTime(timezone=True))  # pyright: ignore[reportArgumentType]
    labels: list[Label] = Relationship(back_populates="tasks", link_model=TaskLabel)
    project: "Project" = Relationship(back_populates="tasks")
    created_by: uuid.UUID = Field(
        foreign_key="users.id", index=True, ondelete="CASCADE"
    )
    assigned_to: str = Field(default="creator", nullable=True)


class TaskCreate(BaseModel):
    project_id: uuid.UUID
    name: str = PydanticField(min_length=1, max_length=200)
    description: str | None = None
    priority: PriorityType = PriorityType.LOW
    status: TaskStatus = TaskStatus.TODO
    due_date: datetime | None = None
    label_ids: list[uuid.UUID] = PydanticField(default_factory=list)


class TaskUpdate(BaseModel):
    """Every field is optional; only the ones sent are applied.

    Setting ``project_id`` moves the task, and is only allowed when the target
    project is also owned by the caller. ``label_ids`` replaces the whole label
    set; omit it to leave labels alone.
    """

    project_id: uuid.UUID | None = None
    name: str | None = PydanticField(default=None, min_length=1, max_length=200)
    description: str | None = None
    priority: PriorityType | None = None
    status: TaskStatus | None = None
    due_date: datetime | None = None
    label_ids: list[uuid.UUID] | None = None


class TaskResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    project_id: uuid.UUID
    created_by: uuid.UUID
    name: str
    description: str | None
    priority: PriorityType
    status: TaskStatus
    due_date: datetime | None
    labels: list[LabelResponse]
    created_at: datetime
    updated_at: datetime


class TaskSort(StrEnum):
    NAME = "name"
    PRIORITY = "priority"
    STATUS = "status"
    DUE_DATE = "due_date"
    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"


class TaskRepo:
    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    @staticmethod
    def _sort_column(sort_by: TaskSort):  # pyright: ignore[reportUnknownParameterType]
        return {
            TaskSort.NAME: col(Task.name),
            TaskSort.PRIORITY: priority_rank(col(Task.priority)),
            TaskSort.STATUS: col(Task.status),
            TaskSort.DUE_DATE: col(Task.due_date),
            TaskSort.CREATED_AT: col(Task.created_at),
            TaskSort.UPDATED_AT: col(Task.updated_at),
        }[sort_by]

    @staticmethod
    async def _load(
        session: AsyncSession, task_id: uuid.UUID, owner_id: uuid.UUID
    ) -> Task | None:
        """Fetch one owned task with its labels eager loaded."""
        statement = (
            select(Task)
            .where(Task.id == task_id, Task.created_by == owner_id)
            .options(selectinload(Task.labels))  # type: ignore[arg-type]
        )
        return (await session.exec(statement)).first()

    # Note: label id parameters are typed Sequence rather than list because the
    # `list` method below shadows the builtin inside this class body.
    async def create(
        self, task: Task, label_ids: Sequence[uuid.UUID] | None = None
    ) -> Task:
        # Read the ids up front: commit expires the instance, and reading an
        # expired attribute would emit a blocking lazy load.
        task_id, owner_id = task.id, task.created_by
        async with AsyncSession(self.engine) as session:
            if label_ids:
                task.labels = await resolve_owned_labels(session, label_ids, owner_id)
            session.add(task)
            await session.commit()
            created = await self._load(session, task_id, owner_id)
            assert created is not None
            return created

    async def get(self, task_id: uuid.UUID, owner_id: uuid.UUID) -> Task | None:
        async with AsyncSession(self.engine) as session:
            return await self._load(session, task_id, owner_id)

    async def get_owner(self, task_id: uuid.UUID) -> uuid.UUID | None:
        """Owner of a task regardless of who is asking, for access checks."""
        async with AsyncSession(self.engine) as session:
            statement = select(Task.created_by).where(Task.id == task_id)
            return (await session.exec(statement)).first()

    async def list(
        self,
        owner_id: uuid.UUID,
        *,
        project_id: uuid.UUID | None = None,
        q: str | None = None,
        priority: PriorityType | None = None,
        status: TaskStatus | None = None,
        label_id: uuid.UUID | None = None,
        due_before: datetime | None = None,
        due_after: datetime | None = None,
        sort_by: TaskSort = TaskSort.CREATED_AT,
        order: SortOrder = SortOrder.DESC,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Task], int]:
        conditions: list[Any] = [Task.created_by == owner_id]
        if project_id is not None:
            conditions.append(Task.project_id == project_id)
        if q:
            pattern = f"%{q}%"
            conditions.append(
                or_(col(Task.name).ilike(pattern), col(Task.description).ilike(pattern))
            )
        if priority is not None:
            conditions.append(Task.priority == priority)
        if status is not None:
            conditions.append(Task.status == status)
        if due_before is not None:
            conditions.append(col(Task.due_date) <= due_before)
        if due_after is not None:
            conditions.append(col(Task.due_date) >= due_after)
        if label_id is not None:
            # A subquery rather than a join: it cannot duplicate rows, so the
            # query stays free of DISTINCT (which Postgres will not combine
            # with an ORDER BY over the priority CASE expression).
            conditions.append(
                col(Task.id).in_(
                    select(TaskLabel.task_id).where(TaskLabel.label_id == label_id)
                )
            )

        column = self._sort_column(sort_by)
        ordering = column.desc() if order is SortOrder.DESC else column.asc()

        async with AsyncSession(self.engine) as session:
            total = (
                await session.exec(
                    select(func.count()).select_from(Task).where(*conditions)
                )
            ).one()
            statement = (
                select(Task)
                .options(selectinload(Task.labels))  # type: ignore[arg-type]
                .where(*conditions)
                .order_by(ordering.nulls_last(), col(Task.id))
                .offset(offset)
                .limit(limit)
            )
            items = list((await session.exec(statement)).all())
            return items, total

    async def update(
        self,
        task_id: uuid.UUID,
        owner_id: uuid.UUID,
        changes: dict[str, Any],
        label_ids: Sequence[uuid.UUID] | None = None,
    ) -> Task | None:
        async with AsyncSession(self.engine) as session:
            task = await self._load(session, task_id, owner_id)
            if task is None:
                return None

            for key, value in changes.items():
                setattr(task, key, value)

            if label_ids is not None:
                task.labels = await resolve_owned_labels(session, label_ids, owner_id)

            session.add(task)
            await session.commit()
            return await self._load(session, task_id, owner_id)

    async def delete(self, task_id: uuid.UUID, owner_id: uuid.UUID) -> bool:
        async with AsyncSession(self.engine) as session:
            statement = (
                select(Task)
                .where(Task.id == task_id, Task.created_by == owner_id)
                .options(selectinload(Task.labels))  # type: ignore[arg-type]
            )
            task = (await session.exec(statement)).first()
            if task is None:
                return False
            await session.delete(task)
            await session.commit()
            return True

    async def add_label(
        self, task_id: uuid.UUID, owner_id: uuid.UUID, label_id: uuid.UUID
    ) -> Task | None:
        async with AsyncSession(self.engine) as session:
            task = await self._load(session, task_id, owner_id)
            if task is None:
                return None

            labels = await resolve_owned_labels(session, [label_id], owner_id)
            label = labels[0]
            if all(existing.id != label.id for existing in task.labels):
                task.labels.append(label)
                session.add(task)
                await session.commit()
            return await self._load(session, task_id, owner_id)

    async def remove_label(
        self, task_id: uuid.UUID, owner_id: uuid.UUID, label_id: uuid.UUID
    ) -> Task | None:
        async with AsyncSession(self.engine) as session:
            task = await self._load(session, task_id, owner_id)
            if task is None:
                return None

            remaining = [label for label in task.labels if label.id != label_id]
            if len(remaining) != len(task.labels):
                task.labels = remaining
                session.add(task)
                await session.commit()
            return await self._load(session, task_id, owner_id)
