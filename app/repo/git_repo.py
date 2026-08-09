"""The git repository an execution works in.

Lives at the top level rather than under ``app.repo.execution`` because a repo
outlives any single execution, and because ``app.repo`` already means
"repository pattern" -- a module named ``_repo.py`` holding a table called
``Repo`` inside it was asking for confusion.
"""

import uuid
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import Any

import sqlalchemy as sa
from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import Field, String, col, func, or_, select
from sqlmodel.ext.asyncio.session import AsyncSession

from .base import (
    DuplicateGitRepoNameError,
    ResourceInUseError,
    SortOrder,
    TimestampMixin,
)

# Agents never commit to the default namespace directly. Everything they do
# lands under this namespace first, one ref per execution, and is squashed into
# the default branch only when the execution finishes. Keeping it out of
# refs/heads/* means these commits are invisible to normal git tooling (branch
# listings, checkouts, `git log` on the default branch) until they are landed.
CEREBRAL_REF_NAMESPACE = "refs/cerebral"


def cerebral_ref(execution_id: uuid.UUID) -> str:
    """The ref an execution's in-progress commits live on."""
    return f"{CEREBRAL_REF_NAMESPACE}/executions/{execution_id}"


class GitRepo(TimestampMixin, table=True):
    __tablename__ = "git_repos"  # pyright: ignore[reportAssignmentType]
    __table_args__ = (
        # Scoped per owner: two users may each have a repo called "api".
        sa.UniqueConstraint("created_by", "name", name="uq_git_repos_created_by_name"),
        # The lookup a bot does at startup, knowing only its working directory.
        sa.Index("ix_git_repos_created_by_local_path", "created_by", "local_path"),
    )

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    created_by: uuid.UUID = Field(
        foreign_key="users.id", index=True, ondelete="RESTRICT"
    )
    name: str = Field(sa_type=String(200))  # pyright: ignore[reportArgumentType]
    # Absolute path on the machine the agent runs on. Inherently per-machine:
    # the same repo checked out on a laptop and a build box has two paths and
    # only one row, so treat this as "where it was last seen", not identity.
    # remote_url is the stable identifier when there is one.
    local_path: str = Field(sa_type=String(1024))  # pyright: ignore[reportArgumentType]
    default_branch: str = Field(default="main", sa_type=String(255))  # pyright: ignore[reportArgumentType]
    # refs/cerebral/* is not covered by a default clone's refspec, so a fresh
    # clone will not have the history. Record where the shadow refs are pushed
    # so the client can add `+refs/cerebral/*:refs/cerebral/*` when fetching.
    remote_url: str | None = Field(default=None, sa_type=String(1024))  # pyright: ignore[reportArgumentType]


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #
class GitRepoCreate(BaseModel):
    name: str = PydanticField(min_length=1, max_length=200)
    local_path: str = PydanticField(min_length=1, max_length=1024)
    default_branch: str = PydanticField(default="main", max_length=255)
    remote_url: str | None = PydanticField(default=None, max_length=1024)


class GitRepoUpdate(BaseModel):
    """Every field is optional; only the ones sent are applied."""

    name: str | None = PydanticField(default=None, min_length=1, max_length=200)
    local_path: str | None = PydanticField(default=None, min_length=1, max_length=1024)
    default_branch: str | None = PydanticField(default=None, max_length=255)
    remote_url: str | None = PydanticField(default=None, max_length=1024)


class GitRepoResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    created_by: uuid.UUID
    name: str
    local_path: str
    default_branch: str
    remote_url: str | None
    created_at: datetime
    updated_at: datetime


class GitRepoConnected(BaseModel):
    """The result of connecting a repo, and whether that created it.

    A bot cannot tell 200 from 201 without inspecting the status code, and the
    branch it wants to take ("did I just register this, or was it already
    known?") is easier to read off the body.
    """

    repo: GitRepoResponse
    created: bool


class GitRepoSort(StrEnum):
    NAME = "name"
    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #
class GitRepoStore:
    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    @staticmethod
    def _sort_column(sort_by: GitRepoSort):  # pyright: ignore[reportUnknownParameterType]
        return {
            GitRepoSort.NAME: col(GitRepo.name),
            GitRepoSort.CREATED_AT: col(GitRepo.created_at),
            GitRepoSort.UPDATED_AT: col(GitRepo.updated_at),
        }[sort_by]

    async def create(self, repo: GitRepo) -> GitRepo:
        name = repo.name
        async with AsyncSession(self.engine) as session:
            session.add(repo)
            try:
                await session.commit()
            except IntegrityError as exc:
                await session.rollback()
                raise DuplicateGitRepoNameError(name) from exc
            await session.refresh(repo)
            return repo

    async def connect(
        self, owner_id: uuid.UUID, body: GitRepoCreate
    ) -> tuple[GitRepo, bool]:
        """Register a repo, or return the one already registered under this name.

        Idempotent by ``(owner, name)`` so a bot can call it unconditionally at
        startup instead of doing a lookup-then-create dance that races with
        itself. When the repo already exists, the connection details it sent are
        refreshed: local_path in particular is per-machine and goes stale the
        moment the same repo is cloned somewhere else.

        Returns ``(repo, created)``.
        """
        async with AsyncSession(self.engine) as session:
            statement = select(GitRepo).where(
                GitRepo.created_by == owner_id, GitRepo.name == body.name
            )
            existing = (await session.exec(statement)).first()

            if existing is not None:
                existing.local_path = body.local_path
                existing.default_branch = body.default_branch
                if body.remote_url is not None:
                    existing.remote_url = body.remote_url
                session.add(existing)
                await session.commit()
                await session.refresh(existing)
                return existing, False

            repo = GitRepo(created_by=owner_id, **body.model_dump())
            session.add(repo)
            try:
                await session.commit()
            except IntegrityError as exc:
                # Another request registered the same name between the SELECT
                # and the INSERT. The unique constraint is what makes that safe;
                # re-read rather than failing, since the caller's intent -- "make
                # sure this repo is connected" -- is now satisfied.
                await session.rollback()
                again = (await session.exec(statement)).first()
                if again is None:
                    raise DuplicateGitRepoNameError(body.name) from exc
                return again, False
            await session.refresh(repo)
            return repo, True

    async def get(self, repo_id: uuid.UUID, owner_id: uuid.UUID) -> GitRepo | None:
        async with AsyncSession(self.engine) as session:
            statement = select(GitRepo).where(
                GitRepo.id == repo_id, GitRepo.created_by == owner_id
            )
            return (await session.exec(statement)).first()

    async def get_owner(self, repo_id: uuid.UUID) -> uuid.UUID | None:
        """Owner of a repo regardless of who is asking, for access checks."""
        async with AsyncSession(self.engine) as session:
            statement = select(GitRepo.created_by).where(GitRepo.id == repo_id)
            return (await session.exec(statement)).first()

    async def get_by_name(self, name: str, owner_id: uuid.UUID) -> GitRepo | None:
        async with AsyncSession(self.engine) as session:
            statement = select(GitRepo).where(
                GitRepo.name == name, GitRepo.created_by == owner_id
            )
            return (await session.exec(statement)).first()

    async def owners_of(
        self, repo_ids: Sequence[uuid.UUID]
    ) -> dict[uuid.UUID, uuid.UUID]:
        """Owner of each repo that exists, keyed by repo id.

        One query for a whole list, so a route attaching several repos to an
        execution can tell "no such repo" (absent from the mapping) from
        "someone else's repo" (present, different owner) without a lookup each.
        """
        wanted = list(dict.fromkeys(repo_ids))
        if not wanted:
            return {}
        async with AsyncSession(self.engine) as session:
            statement = select(GitRepo.id, GitRepo.created_by).where(
                col(GitRepo.id).in_(wanted)
            )
            return {
                repo_id: owner_id
                for repo_id, owner_id in (await session.exec(statement)).all()
            }

    async def list(
        self,
        owner_id: uuid.UUID,
        *,
        q: str | None = None,
        name: str | None = None,
        local_path: str | None = None,
        remote_url: str | None = None,
        sort_by: GitRepoSort = GitRepoSort.NAME,
        order: SortOrder = SortOrder.ASC,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[GitRepo], int]:
        """``local_path`` is the exact-match filter a bot uses at startup: it
        knows the directory it was pointed at and needs the repo id."""
        conditions: list[Any] = [GitRepo.created_by == owner_id]
        if q:
            pattern = f"%{q}%"
            conditions.append(
                or_(
                    col(GitRepo.name).ilike(pattern),
                    col(GitRepo.local_path).ilike(pattern),
                    col(GitRepo.remote_url).ilike(pattern),
                )
            )
        if name is not None:
            conditions.append(GitRepo.name == name)
        if local_path is not None:
            conditions.append(GitRepo.local_path == local_path)
        if remote_url is not None:
            conditions.append(GitRepo.remote_url == remote_url)

        column = self._sort_column(sort_by)
        ordering = column.desc() if order is SortOrder.DESC else column.asc()

        async with AsyncSession(self.engine) as session:
            total = (
                await session.exec(
                    select(func.count()).select_from(GitRepo).where(*conditions)
                )
            ).one()
            statement = (
                select(GitRepo)
                .where(*conditions)
                .order_by(ordering, col(GitRepo.id))
                .offset(offset)
                .limit(limit)
            )
            items = list((await session.exec(statement)).all())
            return items, total

    async def update(
        self, repo_id: uuid.UUID, owner_id: uuid.UUID, changes: dict[str, Any]
    ) -> GitRepo | None:
        async with AsyncSession(self.engine) as session:
            statement = select(GitRepo).where(
                GitRepo.id == repo_id, GitRepo.created_by == owner_id
            )
            repo = (await session.exec(statement)).first()
            if repo is None:
                return None

            for key, value in changes.items():
                setattr(repo, key, value)
            # Read before commit: a rollback expires the instance.
            name = repo.name

            session.add(repo)
            try:
                await session.commit()
            except IntegrityError as exc:
                await session.rollback()
                raise DuplicateGitRepoNameError(name) from exc
            await session.refresh(repo)
            return repo

    async def delete(self, repo_id: uuid.UUID, owner_id: uuid.UUID) -> bool:
        """Delete a repo no execution has touched.

        execution_repos and code_changes both RESTRICT on repo_id: once an agent
        has changed a file here, this row is what those changes are anchored to
        and removing it would orphan them.
        """
        async with AsyncSession(self.engine) as session:
            statement = select(GitRepo).where(
                GitRepo.id == repo_id, GitRepo.created_by == owner_id
            )
            repo = (await session.exec(statement)).first()
            if repo is None:
                return False
            await session.delete(repo)
            try:
                await session.commit()
            except IntegrityError as exc:
                await session.rollback()
                raise ResourceInUseError("repo", "executions or code changes") from exc
            return True
