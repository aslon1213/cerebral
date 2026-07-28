import uuid
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField
from sqlalchemy import UniqueConstraint, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.orm import selectinload
from sqlmodel import Field, Relationship, SQLModel, col, func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from .base import DuplicateLabelNameError, SortOrder, TimestampMixin, UnknownLabelsError

if TYPE_CHECKING:
    from .project import Project
    from .task import Task


class ProjectLabel(SQLModel, table=True):
    """Association table. Composite PK prevents duplicate label assignments."""

    __tablename__ = "project_label"

    project_id: uuid.UUID = Field(
        foreign_key="projects.id", primary_key=True, ondelete="CASCADE"
    )
    label_id: uuid.UUID = Field(
        foreign_key="labels.id", primary_key=True, ondelete="CASCADE"
    )


class TaskLabel(SQLModel, table=True):
    __tablename__ = "task_label"
    task_id: uuid.UUID = Field(
        foreign_key="tasks.id",
        primary_key=True,
        ondelete="CASCADE",
    )
    label_id: uuid.UUID = Field(
        foreign_key="labels.id",
        primary_key=True,
        ondelete="CASCADE",
    )


class Label(TimestampMixin, table=True):
    __tablename__ = "labels"
    # Labels are per-user, so names only have to be unique within one owner.
    __table_args__ = (UniqueConstraint("created_by", "name"),)

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str = Field(max_length=64, index=True)
    description: str | None = Field(default=None, max_length=500)
    created_by: uuid.UUID = Field(
        foreign_key="users.id", index=True, ondelete="RESTRICT"
    )

    projects: list["Project"] = Relationship(
        back_populates="labels", link_model=ProjectLabel
    )

    tasks: list["Task"] = Relationship(back_populates="labels", link_model=TaskLabel)


class LabelCreate(BaseModel):
    name: str = PydanticField(min_length=1, max_length=64)
    description: str | None = PydanticField(default=None, max_length=500)


class LabelUpdate(BaseModel):
    """Every field is optional; only the ones sent are applied."""

    name: str | None = PydanticField(default=None, min_length=1, max_length=64)
    description: str | None = PydanticField(default=None, max_length=500)


class LabelResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str | None
    created_by: uuid.UUID
    created_at: datetime
    updated_at: datetime


class LabelSort(StrEnum):
    NAME = "name"
    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"


async def resolve_owned_labels(
    session: AsyncSession,
    label_ids: Sequence[uuid.UUID],
    owner_id: uuid.UUID,
) -> list[Label]:
    """Load labels owned by ``owner_id``, inside the caller's session.

    Shared by ProjectRepo and TaskRepo so a label set can be attached in the
    same transaction that writes the row.

    Raises UnknownLabelsError when an id is missing or owned by someone else,
    which keeps one user from attaching another user's labels.
    """
    wanted = list(dict.fromkeys(label_ids))
    if not wanted:
        return []

    statement = select(Label).where(
        col(Label.id).in_(wanted), Label.created_by == owner_id
    )
    found = list((await session.exec(statement)).all())
    if len(found) != len(wanted):
        missing = set(wanted) - {label.id for label in found}
        raise UnknownLabelsError(missing)
    return found


class LabelRepo:
    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    @staticmethod
    def _sort_column(sort_by: LabelSort):
        return {
            LabelSort.NAME: col(Label.name),
            LabelSort.CREATED_AT: col(Label.created_at),
            LabelSort.UPDATED_AT: col(Label.updated_at),
        }[sort_by]

    async def create(self, label: Label) -> Label:
        name = label.name
        async with AsyncSession(self.engine) as session:
            session.add(label)
            try:
                await session.commit()
            except IntegrityError as exc:
                await session.rollback()
                raise DuplicateLabelNameError(name) from exc
            await session.refresh(label)
            return label

    async def get(self, label_id: uuid.UUID, owner_id: uuid.UUID) -> Label | None:
        async with AsyncSession(self.engine) as session:
            statement = select(Label).where(
                Label.id == label_id, Label.created_by == owner_id
            )
            return (await session.exec(statement)).first()

    async def get_owner(self, label_id: uuid.UUID) -> uuid.UUID | None:
        """Owner of a label regardless of who is asking, for access checks."""
        async with AsyncSession(self.engine) as session:
            statement = select(Label.created_by).where(Label.id == label_id)
            return (await session.exec(statement)).first()

    async def get_by_name(self, name: str, owner_id: uuid.UUID) -> Label | None:
        async with AsyncSession(self.engine) as session:
            statement = select(Label).where(
                Label.name == name, Label.created_by == owner_id
            )
            return (await session.exec(statement)).first()

    async def list(
        self,
        owner_id: uuid.UUID,
        *,
        q: str | None = None,
        sort_by: LabelSort = LabelSort.NAME,
        order: SortOrder = SortOrder.ASC,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Label], int]:
        conditions: list[Any] = [Label.created_by == owner_id]
        if q:
            pattern = f"%{q}%"
            conditions.append(
                or_(
                    col(Label.name).ilike(pattern),
                    col(Label.description).ilike(pattern),
                )
            )

        column = self._sort_column(sort_by)
        ordering = column.desc() if order is SortOrder.DESC else column.asc()

        async with AsyncSession(self.engine) as session:
            total = (
                await session.exec(
                    select(func.count()).select_from(Label).where(*conditions)
                )
            ).one()
            statement = (
                select(Label)
                .where(*conditions)
                .order_by(ordering, col(Label.id))
                .offset(offset)
                .limit(limit)
            )
            items = list((await session.exec(statement)).all())
            return items, total

    async def update(
        self, label_id: uuid.UUID, owner_id: uuid.UUID, changes: dict[str, Any]
    ) -> Label | None:
        async with AsyncSession(self.engine) as session:
            statement = select(Label).where(
                Label.id == label_id, Label.created_by == owner_id
            )
            label = (await session.exec(statement)).first()
            if label is None:
                return None

            for key, value in changes.items():
                setattr(label, key, value)
            # Kept for the error path below: a rollback expires the instance.
            name = label.name

            session.add(label)
            try:
                await session.commit()
            except IntegrityError as exc:
                await session.rollback()
                raise DuplicateLabelNameError(name) from exc
            await session.refresh(label)
            return label

    async def delete(self, label_id: uuid.UUID, owner_id: uuid.UUID) -> bool:
        async with AsyncSession(self.engine) as session:
            # The link collections are eager loaded so SQLAlchemy can clear the
            # association rows without emitting a lazy load on a closed session.
            statement = (
                select(Label)
                .where(Label.id == label_id, Label.created_by == owner_id)
                .options(
                    selectinload(Label.projects),  # type: ignore[arg-type]
                    selectinload(Label.tasks),  # type: ignore[arg-type]
                )
            )
            label = (await session.exec(statement)).first()
            if label is None:
                return False
            await session.delete(label)
            await session.commit()
            return True
