import uuid
from collections.abc import Iterable
from datetime import datetime
from enum import StrEnum
from typing import Generic, TypeVar

from pydantic import BaseModel
from sqlmodel import DateTime, Field, SQLModel, func

from .utils import utcnow

SQLModel.metadata.naming_convention = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

T = TypeVar("T")


class TimestampMixin(SQLModel):
    created_at: datetime = Field(
        default_factory=utcnow,
        sa_type=DateTime(timezone=True),
        sa_column_kwargs={"server_default": func.now()},
        nullable=False,
    )
    updated_at: datetime = Field(
        default_factory=utcnow,
        sa_type=DateTime(timezone=True),
        sa_column_kwargs={"server_default": func.now(), "onupdate": func.now()},
        nullable=False,
    )


class SortOrder(StrEnum):
    ASC = "asc"
    DESC = "desc"


class Page(BaseModel, Generic[T]):
    """Envelope returned by every list endpoint."""

    items: list[T]
    total: int
    limit: int
    offset: int


class RepoError(Exception):
    """Base class for domain errors raised by a repository.

    Repos never raise HTTP errors; the handlers in app.main translate these
    into responses.
    """


class UnknownLabelsError(RepoError):
    """Referenced labels do not exist or belong to another user."""

    def __init__(self, label_ids: Iterable[uuid.UUID]) -> None:
        self.label_ids = list(label_ids)
        joined = ", ".join(str(label_id) for label_id in self.label_ids)
        super().__init__(f"Unknown or inaccessible labels: {joined}")


class DuplicateLabelNameError(RepoError):
    """The user already owns a label with this name."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"A label named {name!r} already exists")


class InvalidDateRangeError(RepoError):
    """A partial update would leave target_date before started_date."""

    def __init__(self) -> None:
        super().__init__("target_date must not be earlier than started_date")
