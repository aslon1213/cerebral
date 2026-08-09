import uuid
from collections.abc import Iterable
from datetime import datetime
from enum import StrEnum
from typing import Generic, TypeVar

import sqlalchemy as sa
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


class CreatedAtMixin(SQLModel):
    """Creation timestamp only, for append-only tables.

    Deliberately has no ``updated_at``: a row in the execution history is a
    record of something that happened, so there is nothing to update. Tables
    that carry mutable state use :class:`TimestampMixin` instead.
    """

    created_at: datetime = Field(
        default_factory=utcnow,
        sa_type=DateTime(timezone=True),  # pyright: ignore[reportArgumentType]
        sa_column_kwargs={"server_default": func.now()},
        nullable=False,
    )


class TimestampMixin(CreatedAtMixin):
    updated_at: datetime = Field(
        default_factory=utcnow,
        sa_type=DateTime(timezone=True),  # pyright: ignore[reportArgumentType]
        sa_column_kwargs={"server_default": func.now(), "onupdate": func.now()},
        nullable=False,
    )


def enum_check(column: str, enum_cls: type[StrEnum], name: str) -> sa.CheckConstraint:
    """Constrain a ``String`` column to the values of ``enum_cls``.

    Enum columns are stored as plain strings rather than native Postgres enums:
    ``ALTER TYPE ... ADD VALUE`` is awkward to run inside a migration, and these
    vocabularies (event types especially) grow. Deriving the constraint from the
    enum keeps the two from drifting -- adding a member is a migration, not a
    silently-diverging literal list.
    """
    values = ", ".join(f"'{member.value}'" for member in enum_cls)
    return sa.CheckConstraint(f"{column} IN ({values})", name=name)


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


class DuplicateAgentNameError(RepoError):
    """The user already owns an agent with this name."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"An agent named {name!r} already exists")


class DuplicateGitRepoNameError(RepoError):
    """The user already connected a repo under this name."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"A repo named {name!r} is already connected")


class ResourceInUseError(RepoError):
    """Something still references this row, so it cannot be deleted.

    Raised where the schema uses RESTRICT deliberately -- an agent or repo named
    by an execution is part of that execution's audit trail, and deleting it
    would leave history that no longer explains itself.
    """

    def __init__(self, what: str, used_by: str) -> None:
        self.what = what
        self.used_by = used_by
        super().__init__(
            f"This {what} cannot be deleted while {used_by} still reference it"
        )


class InvalidTransitionError(RepoError):
    """The execution is not in a state this status change is legal from.

    Raised before the write, so a client mistake comes back as a 409 naming both
    states rather than as whichever check constraint happened to fire first.
    """

    def __init__(self, current: StrEnum, target: StrEnum) -> None:
        self.current = current
        self.target = target
        super().__init__(
            f"An execution that is {current.value} cannot become {target.value}"
        )


class ExecutionHistoryExistsError(RepoError):
    """Deleting this execution would erase history that explains a codebase."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message
            or "This execution has recorded history. Delete it with purge=true "
            "to drop its events and code changes as well."
        )


class InterventionAlreadyResolvedError(RepoError):
    """Somebody already answered this one.

    Two reviewers opening the same inbox is the normal case, not a rare one, so
    the loser needs to be told what the answer was rather than silently
    overwriting it.
    """

    def __init__(self, status: StrEnum) -> None:
        self.status = status
        super().__init__(
            f"This request was already {status.value} and cannot be answered again"
        )


class InterventionKindMismatchError(RepoError):
    """An approval was answered, or a question approved.

    ``ck_execution_interventions_status_matches_kind`` refuses the pairing; this
    catches it first so the message can say which verb the caller wanted.
    """

    def __init__(self, kind: StrEnum, status: StrEnum) -> None:
        self.kind = kind
        self.status = status
        super().__init__(
            f"A {kind.value} request cannot be {status.value}. Approvals are "
            "approved or rejected; questions are answered."
        )


class RepoNotAttachedError(RepoError):
    """A code change names a repo the execution is not working in.

    The composite foreign key into ``execution_repos`` would refuse this anyway;
    caught first so the message can name the repo instead of the constraint.
    """

    def __init__(self, repo_ids: Iterable[uuid.UUID]) -> None:
        self.repo_ids = list(repo_ids)
        joined = ", ".join(str(repo_id) for repo_id in self.repo_ids)
        super().__init__(
            f"This execution is not working in these repos, so it cannot record "
            f"changes to them: {joined}. Attach the repo to the execution first."
        )


class UnknownParentEventError(RepoError):
    """The event this one claims to answer is not part of this execution."""

    def __init__(self, event_id: uuid.UUID) -> None:
        self.event_id = event_id
        super().__init__(
            f"No event {event_id} in this execution to be the parent of this one"
        )


class InvalidCodeChangeError(RepoError):
    """A code change the schema refuses to store.

    The rules it broke are the ones git itself implies: a created file has no
    before blob, a deleted file has no after blob, a rename has to say what it
    was called before, and no file is touched twice in one event.
    """

    def __init__(self) -> None:
        super().__init__(
            "One of these code changes is not consistent. A created file has no "
            "before_blob, a deleted file has no after_blob, a modified or "
            "renamed file needs both, and a rename must carry previous_path."
        )


class RepoAlreadyAttachedError(RepoError):
    """This repo is already part of the execution.

    Attaching it again would need a second ref and a second base commit for the
    same repo in the same run, which is not a shape the git model has.
    """

    def __init__(self, repo_id: uuid.UUID) -> None:
        self.repo_id = repo_id
        super().__init__(f"This execution is already working in repo {repo_id}")
