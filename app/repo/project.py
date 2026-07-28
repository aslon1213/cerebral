import uuid
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator
from pydantic import Field as PydanticField
from sqlalchemy import String, or_
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.orm import selectinload
from sqlmodel import CheckConstraint, DateTime, Field, Relationship, col, func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.repo.priority import PriorityType, priority_rank

from .base import InvalidDateRangeError, SortOrder, TimestampMixin
from .labels import Label, LabelResponse, ProjectLabel, resolve_owned_labels
from .task import Task


class Project(TimestampMixin, table=True):
    __tablename__ = "projects"
    __table_args__ = (
        CheckConstraint(
            "target_date IS NULL"
            " OR started_date IS NULL"
            " OR target_date >= started_date",
            name="target_after_start",
        ),
    )
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    created_by: uuid.UUID = Field(
        foreign_key="users.id", index=True, ondelete="CASCADE"
    )
    name: str = Field(max_length=200, index=True)
    description: str | None = Field(default=None)
    priority: PriorityType = Field(default=PriorityType.LOW, sa_type=String(16))
    started_date: datetime | None = Field(
        default=None,
        sa_type=DateTime(timezone=True),
    )  #### This is set by the user
    target_date: datetime | None = Field(
        default=None,
        sa_type=DateTime(timezone=True),
    )
    goals: list[str] = Field(default_factory=list, sa_type=JSONB, nullable=False)
    labels: list[Label] = Relationship(
        back_populates="projects", link_model=ProjectLabel
    )
    tasks: list[Task] = Relationship(back_populates="project", cascade_delete=True)


class ProjectCreate(BaseModel):
    name: str = PydanticField(min_length=1, max_length=200)
    description: str | None = None
    priority: PriorityType = PriorityType.LOW
    started_date: datetime | None = None
    target_date: datetime | None = None
    goals: list[str] = PydanticField(default_factory=list)
    label_ids: list[uuid.UUID] = PydanticField(default_factory=list)

    @model_validator(mode="after")
    def _check_dates(self) -> "ProjectCreate":
        if (
            self.started_date is not None
            and self.target_date is not None
            and self.target_date < self.started_date
        ):
            raise ValueError("target_date must not be earlier than started_date")
        return self


class ProjectUpdate(BaseModel):
    """Every field is optional; only the ones sent are applied.

    ``label_ids`` replaces the whole label set; omit it to leave labels alone.
    """

    name: str | None = PydanticField(default=None, min_length=1, max_length=200)
    description: str | None = None
    priority: PriorityType | None = None
    started_date: datetime | None = None
    target_date: datetime | None = None
    goals: list[str] | None = None
    label_ids: list[uuid.UUID] | None = None


class ProjectResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    created_by: uuid.UUID
    name: str
    description: str | None
    priority: PriorityType
    started_date: datetime | None
    target_date: datetime | None
    goals: list[str]
    labels: list[LabelResponse]
    created_at: datetime
    updated_at: datetime


class ProjectSort(StrEnum):
    NAME = "name"
    PRIORITY = "priority"
    STARTED_DATE = "started_date"
    TARGET_DATE = "target_date"
    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"


class ProjectRepo:
    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    @staticmethod
    def _sort_column(sort_by: ProjectSort):
        return {
            ProjectSort.NAME: col(Project.name),
            ProjectSort.PRIORITY: priority_rank(col(Project.priority)),
            ProjectSort.STARTED_DATE: col(Project.started_date),
            ProjectSort.TARGET_DATE: col(Project.target_date),
            ProjectSort.CREATED_AT: col(Project.created_at),
            ProjectSort.UPDATED_AT: col(Project.updated_at),
        }[sort_by]

    @staticmethod
    async def _load(
        session: AsyncSession, project_id: uuid.UUID, owner_id: uuid.UUID
    ) -> Project | None:
        """Fetch one owned project with its labels eager loaded."""
        statement = (
            select(Project)
            .where(Project.id == project_id, Project.created_by == owner_id)
            .options(selectinload(Project.labels))  # type: ignore[arg-type]
        )
        return (await session.exec(statement)).first()

    @staticmethod
    def _validate_dates(project: Project) -> None:
        if (
            project.started_date is not None
            and project.target_date is not None
            and project.target_date < project.started_date
        ):
            raise InvalidDateRangeError()

    # Note: label id parameters are typed Sequence rather than list because the
    # `list` method below shadows the builtin inside this class body.
    async def create(
        self, project: Project, label_ids: Sequence[uuid.UUID] | None = None
    ) -> Project:
        # Read the ids up front: commit expires the instance, and reading an
        # expired attribute would emit a blocking lazy load.
        project_id, owner_id = project.id, project.created_by
        async with AsyncSession(self.engine) as session:
            if label_ids:
                project.labels = await resolve_owned_labels(
                    session, label_ids, owner_id
                )
            session.add(project)
            await session.commit()
            created = await self._load(session, project_id, owner_id)
            assert created is not None
            return created

    async def get(self, project_id: uuid.UUID, owner_id: uuid.UUID) -> Project | None:
        async with AsyncSession(self.engine) as session:
            return await self._load(session, project_id, owner_id)

    async def get_owner(self, project_id: uuid.UUID) -> uuid.UUID | None:
        """Owner of a project regardless of who is asking, for access checks."""
        async with AsyncSession(self.engine) as session:
            statement = select(Project.created_by).where(Project.id == project_id)
            return (await session.exec(statement)).first()

    async def list(
        self,
        owner_id: uuid.UUID,
        *,
        q: str | None = None,
        priority: PriorityType | None = None,
        label_id: uuid.UUID | None = None,
        sort_by: ProjectSort = ProjectSort.CREATED_AT,
        order: SortOrder = SortOrder.DESC,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Project], int]:
        conditions: list[Any] = [Project.created_by == owner_id]
        if q:
            pattern = f"%{q}%"
            conditions.append(
                or_(
                    col(Project.name).ilike(pattern),
                    col(Project.description).ilike(pattern),
                )
            )
        if priority is not None:
            conditions.append(Project.priority == priority)
        if label_id is not None:
            # A subquery rather than a join: it cannot duplicate rows, so the
            # query stays free of DISTINCT (which Postgres will not combine
            # with an ORDER BY over the priority CASE expression).
            conditions.append(
                col(Project.id).in_(
                    select(ProjectLabel.project_id).where(
                        ProjectLabel.label_id == label_id
                    )
                )
            )

        column = self._sort_column(sort_by)
        ordering = column.desc() if order is SortOrder.DESC else column.asc()

        async with AsyncSession(self.engine) as session:
            total = (
                await session.exec(
                    select(func.count()).select_from(Project).where(*conditions)
                )
            ).one()
            statement = (
                select(Project)
                .options(selectinload(Project.labels))  # type: ignore[arg-type]
                .where(*conditions)
                .order_by(ordering.nulls_last(), col(Project.id))
                .offset(offset)
                .limit(limit)
            )
            items = list((await session.exec(statement)).all())
            return items, total

    async def update(
        self,
        project_id: uuid.UUID,
        owner_id: uuid.UUID,
        changes: dict[str, Any],
        label_ids: Sequence[uuid.UUID] | None = None,
    ) -> Project | None:
        async with AsyncSession(self.engine) as session:
            project = await self._load(session, project_id, owner_id)
            if project is None:
                return None

            for key, value in changes.items():
                setattr(project, key, value)
            self._validate_dates(project)

            if label_ids is not None:
                project.labels = await resolve_owned_labels(
                    session, label_ids, owner_id
                )

            session.add(project)
            await session.commit()
            return await self._load(session, project_id, owner_id)

    async def delete(self, project_id: uuid.UUID, owner_id: uuid.UUID) -> bool:
        async with AsyncSession(self.engine) as session:
            # Tasks are eager loaded so the ORM cascade can delete them without
            # a lazy load, and labels so the link rows are cleared.
            statement = (
                select(Project)
                .where(Project.id == project_id, Project.created_by == owner_id)
                .options(
                    selectinload(Project.labels),  # type: ignore[arg-type]
                    selectinload(Project.tasks).selectinload(Task.labels),  # type: ignore[arg-type]
                )
            )
            project = (await session.exec(statement)).first()
            if project is None:
                return False
            await session.delete(project)
            await session.commit()
            return True

    async def add_label(
        self, project_id: uuid.UUID, owner_id: uuid.UUID, label_id: uuid.UUID
    ) -> Project | None:
        async with AsyncSession(self.engine) as session:
            project = await self._load(session, project_id, owner_id)
            if project is None:
                return None

            labels = await resolve_owned_labels(session, [label_id], owner_id)
            label = labels[0]
            if all(existing.id != label.id for existing in project.labels):
                project.labels.append(label)
                session.add(project)
                await session.commit()
            return await self._load(session, project_id, owner_id)

    async def remove_label(
        self, project_id: uuid.UUID, owner_id: uuid.UUID, label_id: uuid.UUID
    ) -> Project | None:
        async with AsyncSession(self.engine) as session:
            project = await self._load(session, project_id, owner_id)
            if project is None:
                return None

            remaining = [label for label in project.labels if label.id != label_id]
            if len(remaining) != len(project.labels):
                project.labels = remaining
                session.add(project)
                await session.commit()
            return await self._load(session, project_id, owner_id)
