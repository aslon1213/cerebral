"""The agent identity an execution runs as.

A generic record, deliberately: "claude-code on my laptop" and "the nightly
refactor bot" are both agents, and an execution only needs to name which one
produced it. Capabilities, tool allow-lists and prompts are not modelled here --
add columns when there is something that reads them, rather than widening the
JSONB on executions.
"""

import uuid
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
    DuplicateAgentNameError,
    ResourceInUseError,
    SortOrder,
    TimestampMixin,
)


class Agent(TimestampMixin, table=True):
    __tablename__ = "agents"  # pyright: ignore[reportAssignmentType]
    __table_args__ = (
        sa.UniqueConstraint("created_by", "name", name="uq_agents_created_by_name"),
    )

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    created_by: uuid.UUID = Field(
        foreign_key="users.id", index=True, ondelete="RESTRICT"
    )
    name: str = Field(sa_type=String(200))  # pyright: ignore[reportArgumentType]
    description: str | None = Field(default=None)
    # Default model for runs by this agent; the model actually used is recorded
    # per execution, since it can be overridden per run.
    default_model: str | None = Field(default=None, sa_type=String(128))  # pyright: ignore[reportArgumentType]
    is_active: bool = Field(default=True)


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #
class AgentCreate(BaseModel):
    name: str = PydanticField(min_length=1, max_length=200)
    description: str | None = None
    default_model: str | None = PydanticField(default=None, max_length=128)
    is_active: bool = True


class AgentUpdate(BaseModel):
    """Every field is optional; only the ones sent are applied."""

    name: str | None = PydanticField(default=None, min_length=1, max_length=200)
    description: str | None = None
    default_model: str | None = PydanticField(default=None, max_length=128)
    is_active: bool | None = None


class AgentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    created_by: uuid.UUID
    name: str
    description: str | None
    default_model: str | None
    is_active: bool
    created_at: datetime
    updated_at: datetime


class AgentSort(StrEnum):
    NAME = "name"
    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #
class AgentStore:
    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    @staticmethod
    def _sort_column(sort_by: AgentSort):  # pyright: ignore[reportUnknownParameterType]
        return {
            AgentSort.NAME: col(Agent.name),
            AgentSort.CREATED_AT: col(Agent.created_at),
            AgentSort.UPDATED_AT: col(Agent.updated_at),
        }[sort_by]

    async def create(self, agent: Agent) -> Agent:
        name = agent.name
        async with AsyncSession(self.engine) as session:
            session.add(agent)
            try:
                await session.commit()
            except IntegrityError as exc:
                await session.rollback()
                raise DuplicateAgentNameError(name) from exc
            await session.refresh(agent)
            return agent

    async def get(self, agent_id: uuid.UUID, owner_id: uuid.UUID) -> Agent | None:
        async with AsyncSession(self.engine) as session:
            statement = select(Agent).where(
                Agent.id == agent_id, Agent.created_by == owner_id
            )
            return (await session.exec(statement)).first()

    async def get_owner(self, agent_id: uuid.UUID) -> uuid.UUID | None:
        """Owner of an agent regardless of who is asking, for access checks."""
        async with AsyncSession(self.engine) as session:
            statement = select(Agent.created_by).where(Agent.id == agent_id)
            return (await session.exec(statement)).first()

    async def get_by_name(self, name: str, owner_id: uuid.UUID) -> Agent | None:
        """Resolve an agent the way a bot configured with a name would."""
        async with AsyncSession(self.engine) as session:
            statement = select(Agent).where(
                Agent.name == name, Agent.created_by == owner_id
            )
            return (await session.exec(statement)).first()

    async def list(
        self,
        owner_id: uuid.UUID,
        *,
        q: str | None = None,
        is_active: bool | None = None,
        sort_by: AgentSort = AgentSort.NAME,
        order: SortOrder = SortOrder.ASC,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Agent], int]:
        conditions: list[Any] = [Agent.created_by == owner_id]
        if q:
            pattern = f"%{q}%"
            conditions.append(
                or_(
                    col(Agent.name).ilike(pattern),
                    col(Agent.description).ilike(pattern),
                )
            )
        if is_active is not None:
            conditions.append(Agent.is_active == is_active)

        column = self._sort_column(sort_by)
        ordering = column.desc() if order is SortOrder.DESC else column.asc()

        async with AsyncSession(self.engine) as session:
            total = (
                await session.exec(
                    select(func.count()).select_from(Agent).where(*conditions)
                )
            ).one()
            statement = (
                select(Agent)
                .where(*conditions)
                .order_by(ordering, col(Agent.id))
                .offset(offset)
                .limit(limit)
            )
            items = list((await session.exec(statement)).all())
            return items, total

    async def update(
        self, agent_id: uuid.UUID, owner_id: uuid.UUID, changes: dict[str, Any]
    ) -> Agent | None:
        async with AsyncSession(self.engine) as session:
            statement = select(Agent).where(
                Agent.id == agent_id, Agent.created_by == owner_id
            )
            agent = (await session.exec(statement)).first()
            if agent is None:
                return None

            for key, value in changes.items():
                setattr(agent, key, value)
            # Read before commit: a rollback expires the instance.
            name = agent.name

            session.add(agent)
            try:
                await session.commit()
            except IntegrityError as exc:
                await session.rollback()
                raise DuplicateAgentNameError(name) from exc
            await session.refresh(agent)
            return agent

    async def delete(self, agent_id: uuid.UUID, owner_id: uuid.UUID) -> bool:
        """Delete an agent nothing refers to.

        Executions and API keys both RESTRICT on agent_id, so an agent that has
        ever run cannot be removed -- its executions would be left pointing at
        nothing, and the history would stop explaining itself. Deactivate it
        instead (``is_active = false``) to keep it out of new runs.
        """
        async with AsyncSession(self.engine) as session:
            statement = select(Agent).where(
                Agent.id == agent_id, Agent.created_by == owner_id
            )
            agent = (await session.exec(statement)).first()
            if agent is None:
                return False
            await session.delete(agent)
            try:
                await session.commit()
            except IntegrityError as exc:
                await session.rollback()
                raise ResourceInUseError("agent", "executions or API keys") from exc
            return True
