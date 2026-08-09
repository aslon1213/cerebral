"""API keys, the credential the observer bot authenticates with.

A bot is not a person: it has no password to type, no refresh cycle to run, and
it lives on a machine whose disk may be read by someone else. So it gets a
long-lived bearer credential that is scoped to what it actually needs, revocable
on its own without disturbing the user's sessions, and stored only as a digest.

The key is shown exactly once, when it is created. Everything afterwards refers
to it by ``prefix``.
"""

import uuid
from datetime import datetime
from enum import StrEnum

import sqlalchemy as sa
from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import DateTime, Field, String, col, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.security import generate_api_key, parse_api_key, verify_api_key
from app.repo.base import RepoError, TimestampMixin
from app.repo.utils import utcnow

# How stale last_used_at may get before a request pays for a write. Every
# authenticated call would otherwise UPDATE the row, turning a read-only ingest
# request into a write and putting every bot event behind a row lock.
LAST_USED_RESOLUTION = 60.0


class ApiKeyScope(StrEnum):
    """What a key is allowed to do.

    Deliberately coarse. Scopes are painful to add later -- every key issued
    before the new scope existed lacks it -- so the axes worth separating are
    separated now: reading versus writing, and executions versus repos.
    """

    EXECUTIONS_READ = "executions:read"
    EXECUTIONS_WRITE = "executions:write"
    REPOS_READ = "repos:read"
    REPOS_WRITE = "repos:write"


# What an observer bot needs: record what the agent did, and resolve the repo
# it is running in. Notably absent is any ability to answer an intervention --
# that is a human decision and no bot credential should be able to make it.
OBSERVER_BOT_SCOPES = [
    ApiKeyScope.EXECUTIONS_READ,
    ApiKeyScope.EXECUTIONS_WRITE,
    ApiKeyScope.REPOS_READ,
]


class ApiKey(TimestampMixin, table=True):
    __tablename__ = "api_keys"  # pyright: ignore[reportAssignmentType]
    __table_args__ = (
        sa.Index(
            "ix_api_keys_created_by_active",
            "created_by",
            "created_at",
            postgresql_where=sa.text("revoked_at IS NULL"),
        ),
        sa.CheckConstraint(
            "expires_at IS NULL OR expires_at > created_at",
            name="expires_after_created",
        ),
    )

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    created_by: uuid.UUID = Field(
        foreign_key="users.id", index=True, ondelete="RESTRICT"
    )
    # The agent this key acts as. Executions started with the key inherit it as
    # executor_agent_id, so the transcript records which agent produced an
    # event without the bot having to assert it per request -- and without it
    # being able to lie about it.
    agent_id: uuid.UUID | None = Field(
        default=None, foreign_key="agents.id", index=True, ondelete="RESTRICT"
    )
    name: str = Field(sa_type=String(200))  # pyright: ignore[reportArgumentType]
    # Public half. Unique because it is the lookup handle.
    prefix: str = Field(sa_type=String(32), unique=True, index=True)  # pyright: ignore[reportArgumentType]
    key_hash: str = Field(sa_type=String(64))  # pyright: ignore[reportArgumentType]
    scopes: list[str] = Field(
        default_factory=list,
        sa_column=sa.Column(JSONB, nullable=False, server_default="[]"),
    )
    last_used_at: datetime | None = Field(
        default=None,
        sa_type=DateTime(timezone=True),  # pyright: ignore[reportArgumentType]
    )
    expires_at: datetime | None = Field(
        default=None,
        sa_type=DateTime(timezone=True),  # pyright: ignore[reportArgumentType]
    )
    # Soft revocation: the row stays so an audit of what this key did still
    # resolves after it is turned off.
    revoked_at: datetime | None = Field(
        default=None,
        sa_type=DateTime(timezone=True),  # pyright: ignore[reportArgumentType]
    )

    def is_usable(self, *, now: datetime | None = None) -> bool:
        now = now or utcnow()
        if self.revoked_at is not None:
            return False
        return self.expires_at is None or self.expires_at > now


# --------------------------------------------------------------------------- #
# Domain errors
# --------------------------------------------------------------------------- #
class UnknownAgentError(RepoError):
    """The referenced agent does not exist or belongs to another user."""

    def __init__(self, agent_id: uuid.UUID) -> None:
        self.agent_id = agent_id
        super().__init__(f"Unknown or inaccessible agent: {agent_id}")


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #
class ApiKeyCreate(BaseModel):
    name: str = PydanticField(min_length=1, max_length=200)
    agent_id: uuid.UUID | None = None
    scopes: list[ApiKeyScope] = PydanticField(
        default_factory=lambda: [*OBSERVER_BOT_SCOPES]
    )
    expires_at: datetime | None = None


class ApiKeyResponse(BaseModel):
    """An API key as it can safely be shown again: no secret, ever."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    prefix: str
    agent_id: uuid.UUID | None
    scopes: list[ApiKeyScope]
    last_used_at: datetime | None
    expires_at: datetime | None
    revoked_at: datetime | None
    created_at: datetime


class ApiKeyCreated(BaseModel):
    """The one response that carries the secret.

    ``key`` is not recoverable afterwards -- only its digest is stored -- so a
    caller that loses it has to issue a new key.
    """

    key: str = PydanticField(
        description="The full API key. Shown once; store it now. "
        "Send it as the X-API-Key header."
    )
    api_key: ApiKeyResponse


class ApiKeyIdentity(BaseModel):
    """What a key says about itself, for a bot to check at startup."""

    key_id: uuid.UUID
    name: str
    prefix: str
    user_id: uuid.UUID
    agent_id: uuid.UUID | None
    scopes: list[ApiKeyScope]
    expires_at: datetime | None


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #
class ApiKeyStore:
    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    async def create(
        self,
        *,
        user_id: uuid.UUID,
        name: str,
        agent_id: uuid.UUID | None,
        scopes: list[ApiKeyScope],
        expires_at: datetime | None = None,
    ) -> tuple[str, ApiKey]:
        """Issue a key. Returns the plaintext once, alongside the stored row."""
        full_key, prefix, key_hash = generate_api_key()
        api_key = ApiKey(
            created_by=user_id,
            agent_id=agent_id,
            name=name,
            prefix=prefix,
            key_hash=key_hash,
            scopes=[scope.value for scope in scopes],
            expires_at=expires_at,
        )
        async with AsyncSession(self.engine) as session:
            session.add(api_key)
            await session.commit()
            await session.refresh(api_key)
            return full_key, api_key

    async def authenticate(self, presented: str) -> ApiKey | None:
        """Resolve a presented key, or None if it is not a usable credential.

        One indexed read on the prefix, then a constant-time digest comparison.
        Revocation and expiry are checked by the caller so it can tell the two
        apart in the response; this returns the row for any key whose secret is
        correct.
        """
        prefix = parse_api_key(presented)
        if prefix is None:
            return None
        async with AsyncSession(self.engine) as session:
            statement = select(ApiKey).where(ApiKey.prefix == prefix)
            api_key = (await session.exec(statement)).first()
        if api_key is None:
            return None
        if not verify_api_key(presented, api_key.key_hash):
            return None
        return api_key

    async def touch(self, key_id: uuid.UUID) -> None:
        """Record that a key was used, at most once per LAST_USED_RESOLUTION.

        Guarded in the WHERE clause rather than by reading first: the row is
        only locked when the write actually happens, so a burst of ingest
        requests does not serialise on it.
        """
        now = utcnow()
        async with AsyncSession(self.engine) as session:
            connection = await session.connection()
            await connection.execute(
                sa.text(
                    "UPDATE api_keys SET last_used_at = :now"
                    " WHERE id = :key_id"
                    "   AND (last_used_at IS NULL"
                    "        OR last_used_at < :now - make_interval(secs => :window))"
                ).bindparams(now=now, key_id=key_id, window=LAST_USED_RESOLUTION)
            )
            await session.commit()

    async def get(self, key_id: uuid.UUID, user_id: uuid.UUID) -> ApiKey | None:
        async with AsyncSession(self.engine) as session:
            statement = select(ApiKey).where(
                ApiKey.id == key_id, ApiKey.created_by == user_id
            )
            return (await session.exec(statement)).first()

    async def list(
        self,
        user_id: uuid.UUID,
        *,
        include_revoked: bool = False,
        agent_id: uuid.UUID | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[ApiKey], int]:
        conditions = [ApiKey.created_by == user_id]
        if not include_revoked:
            conditions.append(col(ApiKey.revoked_at).is_(None))  # pyright: ignore[reportArgumentType]
        if agent_id is not None:
            conditions.append(ApiKey.agent_id == agent_id)

        async with AsyncSession(self.engine) as session:
            total = (
                await session.exec(
                    select(sa.func.count()).select_from(ApiKey).where(*conditions)
                )
            ).one()
            statement = (
                select(ApiKey)
                .where(*conditions)
                .order_by(col(ApiKey.created_at).desc(), col(ApiKey.id))
                .offset(offset)
                .limit(limit)
            )
            items = list((await session.exec(statement)).all())
            return items, total

    async def revoke(self, key_id: uuid.UUID, user_id: uuid.UUID) -> ApiKey | None:
        """Turn a key off. Idempotent: revoking twice keeps the first time."""
        async with AsyncSession(self.engine) as session:
            statement = select(ApiKey).where(
                ApiKey.id == key_id, ApiKey.created_by == user_id
            )
            api_key = (await session.exec(statement)).first()
            if api_key is None:
                return None
            if api_key.revoked_at is None:
                api_key.revoked_at = utcnow()
                session.add(api_key)
                await session.commit()
                await session.refresh(api_key)
            return api_key
