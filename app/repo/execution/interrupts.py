import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

import sqlalchemy as sa
from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import DateTime, Field, col, func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.repo.base import (
    InterventionAlreadyResolvedError,
    InterventionKindMismatchError,
    InvalidTransitionError,
    TimestampMixin,
    enum_check,
)
from app.repo.types import StrEnumType
from app.repo.utils import utcnow

from .execution import Execution, ExecutionStatus
from .history import (
    ActorType,
    ExecutionEvent,
    ExecutionEventPayload,
    ExecutionEventType,
    allocate_event_seq,
)


class InterventionKind(StrEnum):
    APPROVAL = "approval"
    QA_REVIEW = "qa_review"
    INPUT_REQUIRED = "input_required"


class InterventionStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    # A question is answered, not approved. Without this, resolving an
    # input_required means writing "approved" into a row where it means nothing.
    ANSWERED = "answered"
    EXPIRED = "expired"


# Which waiting_* an execution takes while a request of this kind is open. Both
# review kinds are approve/reject flows, so both park the run in the same place.
WAITING_FOR_KIND: dict[InterventionKind, ExecutionStatus] = {
    InterventionKind.APPROVAL: ExecutionStatus.WAITING_APPROVAL,
    InterventionKind.QA_REVIEW: ExecutionStatus.WAITING_APPROVAL,
    InterventionKind.INPUT_REQUIRED: ExecutionStatus.WAITING_INPUT,
}

# How each kind may be resolved, mirroring
# ck_execution_interventions_status_matches_kind. Duplicated here on purpose:
# the constraint can only refuse the write, and refusing it as a 500 naming a
# constraint is no use to somebody who clicked the wrong button.
RESOLUTIONS_FOR_KIND: dict[InterventionKind, frozenset[InterventionStatus]] = {
    InterventionKind.APPROVAL: frozenset(
        {InterventionStatus.APPROVED, InterventionStatus.REJECTED}
    ),
    InterventionKind.QA_REVIEW: frozenset(
        {InterventionStatus.APPROVED, InterventionStatus.REJECTED}
    ),
    InterventionKind.INPUT_REQUIRED: frozenset({InterventionStatus.ANSWERED}),
}

# An execution can only start or stop waiting from one of these. A run that has
# already finished keeps whatever it finished as, even if somebody answers a
# question it left behind.
BLOCKABLE_STATUSES = (
    ExecutionStatus.RUNNING,
    ExecutionStatus.WAITING_APPROVAL,
    ExecutionStatus.WAITING_INPUT,
)


class ExecutionIntervention(TimestampMixin, table=True):
    """A question or approval the agent is blocked on.

    ``executions.status`` carries WAITING_APPROVAL / WAITING_INPUT as a
    denormalisation so list endpoints do not need this join. This table is the
    authoritative one: write both in the same transaction, and if they ever
    disagree, the intervention row is right. An execution may have more than
    one open at a time, so the status clears only when the last one resolves.
    """

    __tablename__ = "execution_interventions"  # pyright: ignore[reportAssignmentType]
    __table_args__ = (
        # Several interventions may be pending at once: a real agent batches
        # tool approvals into a single turn, and a unique index here would
        # reject the second one. So the execution's waiting_* status means "at
        # least one intervention is open", not "exactly one" -- it clears when
        # the last of them resolves.
        sa.Index(
            "ix_execution_interventions_execution_pending",
            "execution_id",
            "created_at",
            postgresql_where=sa.text("status = 'pending'"),
        ),
        # The inbox query: everything waiting on a human, oldest first.
        sa.Index(
            "ix_execution_interventions_pending",
            "created_at",
            postgresql_where=sa.text("status = 'pending'"),
        ),
        # An intervention's event must belong to the same execution.
        sa.ForeignKeyConstraint(
            ["event_id", "execution_id"],
            ["execution_events.id", "execution_events.execution_id"],
            name="fk_execution_interventions_event_execution",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["resolution_event_id", "execution_id"],
            ["execution_events.id", "execution_events.execution_id"],
            name="fk_execution_interventions_resolution_event_execution",
            ondelete="RESTRICT",
        ),
        enum_check("kind", InterventionKind, "kind"),
        enum_check("status", InterventionStatus, "status"),
        # Resolution fields move together with the status.
        sa.CheckConstraint(
            "(status = 'pending') = (resolved_at IS NULL)",
            name="resolved_at_matches_status",
        ),
        # Nothing can expire without a deadline to expire against.
        sa.CheckConstraint(
            "status <> 'expired' OR expires_at IS NOT NULL",
            name="expired_has_deadline",
        ),
        # An approval resolves approved/rejected; a question resolves answered.
        sa.CheckConstraint(
            "(kind = 'input_required' AND status IN ('pending', 'answered', 'expired'))"
            " OR (kind IN ('approval', 'qa_review')"
            "     AND status IN ('pending', 'approved', 'rejected', 'expired'))",
            name="status_matches_kind",
        ),
    )

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    execution_id: uuid.UUID = Field(
        foreign_key="executions.id", index=True, ondelete="RESTRICT"
    )
    # The event that raised this, so the UI can render the question inline in
    # the transcript instead of as a detached side-channel.
    event_id: uuid.UUID | None = Field(default=None, index=True)
    # The event appended when a human answered, so the answer is part of the
    # history too rather than only living in `response` below.
    resolution_event_id: uuid.UUID | None = Field(default=None)

    kind: InterventionKind = Field(sa_type=StrEnumType(InterventionKind, 16))  # pyright: ignore[reportArgumentType]
    status: InterventionStatus = Field(
        default=InterventionStatus.PENDING,
        sa_type=StrEnumType(InterventionStatus, 16),  # pyright: ignore[reportArgumentType]
    )
    request: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=sa.Column(JSONB, nullable=False, server_default="{}"),
    )
    response: dict[str, Any] | None = Field(
        default=None, sa_column=sa.Column(JSONB(none_as_null=True))
    )
    expires_at: datetime | None = Field(default=None, sa_type=DateTime(timezone=True))  # pyright: ignore[reportArgumentType]
    resolved_by_user_id: uuid.UUID | None = Field(
        default=None, foreign_key="users.id", ondelete="RESTRICT"
    )
    resolved_at: datetime | None = Field(default=None, sa_type=DateTime(timezone=True))  # pyright: ignore[reportArgumentType]


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #
class InterventionOpen(BaseModel):
    """What the agent is blocked on."""

    kind: InterventionKind
    # The question, or what needs approving. Free-form: an approval for a shell
    # command and a question about which database to use share no shape.
    request: dict[str, Any] = PydanticField(default_factory=dict)
    # Why the agent is asking. Goes on the intervention_requested event, so the
    # request reads inline in the transcript rather than as a detached prompt.
    reasoning: str | None = None
    expires_at: datetime | None = None


class InterventionDecision(BaseModel):
    """Approving or rejecting. Both may carry a note back to the agent."""

    response: dict[str, Any] | None = None
    reasoning: str | None = None


class InterventionAnswer(BaseModel):
    """Answering a question. The answer is the point, so it is required."""

    response: dict[str, Any]
    reasoning: str | None = None


class InterventionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    execution_id: uuid.UUID
    event_id: uuid.UUID | None
    resolution_event_id: uuid.UUID | None
    kind: InterventionKind
    status: InterventionStatus
    request: dict[str, Any]
    response: dict[str, Any] | None
    expires_at: datetime | None
    resolved_by_user_id: uuid.UUID | None
    resolved_at: datetime | None
    created_at: datetime
    updated_at: datetime


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #
class InterventionStore:
    """Opening and resolving the things an agent blocks on.

    Every write here spans three tables -- the intervention, the event that
    records it, and the execution's denormalised status -- and each one is a
    single transaction. A run parked in ``waiting_approval`` with no pending
    approval to explain it is stuck forever, and an approval nobody can see in
    the transcript is a decision with no provenance.
    """

    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    async def open(
        self, execution_id: uuid.UUID, owner_id: uuid.UUID, body: InterventionOpen
    ) -> ExecutionIntervention | None:
        """Block a run on a human, and say so in its transcript.

        Several may be open at once: a real agent batches its tool approvals
        into one turn, and rejecting the second is not a reason to lose the
        first. The run's status stays ``waiting_*`` until the last resolves.
        """
        async with AsyncSession(self.engine) as session:
            execution = await self._lock_execution(session, execution_id, owner_id)
            if execution is None:
                return None
            if execution.status not in BLOCKABLE_STATUSES:
                # Nothing to interrupt: the agent has not started, or has
                # already stopped.
                raise InvalidTransitionError(
                    execution.status, WAITING_FOR_KIND[body.kind]
                )

            event = await self._append_event(
                session,
                execution_id,
                event_type=ExecutionEventType.INTERVENTION_REQUESTED,
                actor_type=ActorType.AGENT,
                actor_user_id=None,
                payload=ExecutionEventPayload(
                    reasoning=body.reasoning, data=body.request
                ),
            )

            intervention = ExecutionIntervention(
                execution_id=execution_id,
                event_id=event.id,
                kind=body.kind,
                request=body.request,
                expires_at=body.expires_at,
            )
            session.add(intervention)
            await session.flush()

            await self._sync_execution_status(session, execution)
            await session.commit()
            await session.refresh(intervention)
            return intervention

    async def resolve(
        self,
        intervention_id: uuid.UUID,
        owner_id: uuid.UUID,
        *,
        status: InterventionStatus,
        resolved_by_user_id: uuid.UUID,
        response: dict[str, Any] | None = None,
        reasoning: str | None = None,
    ) -> ExecutionIntervention | None:
        """Answer one request, and let the agent carry on if it was the last.

        The answer becomes an ``intervention_resolved`` event attributed to
        whoever gave it, so the transcript records not just what the agent
        decided but what a person told it to do.
        """
        async with AsyncSession(self.engine) as session:
            # The execution is locked before the intervention, the same order
            # `open` takes them in. Two orders would eventually deadlock.
            execution_id = (
                await session.exec(
                    select(ExecutionIntervention.execution_id).where(
                        ExecutionIntervention.id == intervention_id
                    )
                )
            ).first()
            if execution_id is None:
                return None
            execution = await self._lock_execution(session, execution_id, owner_id)
            if execution is None:
                return None

            intervention = (
                await session.exec(
                    select(ExecutionIntervention)
                    .where(ExecutionIntervention.id == intervention_id)
                    .with_for_update()
                )
            ).first()
            if intervention is None:
                return None

            if intervention.status is not InterventionStatus.PENDING:
                raise InterventionAlreadyResolvedError(intervention.status)
            if status not in RESOLUTIONS_FOR_KIND[intervention.kind]:
                raise InterventionKindMismatchError(intervention.kind, status)

            event = await self._append_event(
                session,
                execution_id,
                event_type=ExecutionEventType.INTERVENTION_RESOLVED,
                # A person answered, so the event says so and carries their id.
                actor_type=ActorType.USER,
                actor_user_id=resolved_by_user_id,
                payload=ExecutionEventPayload(
                    reasoning=reasoning,
                    data={"status": status.value, "response": response},
                ),
            )

            intervention.status = status
            intervention.response = response
            intervention.resolution_event_id = event.id
            intervention.resolved_by_user_id = resolved_by_user_id
            intervention.resolved_at = utcnow()
            session.add(intervention)
            await session.flush()

            await self._sync_execution_status(session, execution)
            await session.commit()
            await session.refresh(intervention)
            return intervention

    async def get(
        self, intervention_id: uuid.UUID, owner_id: uuid.UUID
    ) -> ExecutionIntervention | None:
        async with AsyncSession(self.engine) as session:
            statement = (
                select(ExecutionIntervention)
                .join(
                    Execution,
                    col(Execution.id) == col(ExecutionIntervention.execution_id),
                )
                .where(
                    ExecutionIntervention.id == intervention_id,
                    Execution.created_by == owner_id,
                )
            )
            return (await session.exec(statement)).first()

    async def get_owner(self, intervention_id: uuid.UUID) -> uuid.UUID | None:
        """Owner of the run this belongs to, for access checks."""
        async with AsyncSession(self.engine) as session:
            statement = (
                select(Execution.created_by)
                .join(
                    ExecutionIntervention,
                    col(ExecutionIntervention.execution_id) == col(Execution.id),
                )
                .where(ExecutionIntervention.id == intervention_id)
            )
            return (await session.exec(statement)).first()

    async def list_pending(
        self, owner_id: uuid.UUID, *, limit: int = 50, offset: int = 0
    ) -> tuple[list[ExecutionIntervention], int]:
        """The inbox: everything waiting on this person, oldest first.

        Oldest first because it is a queue -- the agent that has been blocked
        longest is the one losing the most time.
        """
        return await self._list(
            [ExecutionIntervention.status == InterventionStatus.PENDING],
            owner_id,
            ascending=True,
            limit=limit,
            offset=offset,
        )

    async def list_for_execution(
        self,
        execution_id: uuid.UUID,
        owner_id: uuid.UUID,
        *,
        status: InterventionStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[ExecutionIntervention], int]:
        conditions: list[Any] = [ExecutionIntervention.execution_id == execution_id]
        if status is not None:
            conditions.append(ExecutionIntervention.status == status)
        return await self._list(
            conditions, owner_id, ascending=True, limit=limit, offset=offset
        )

    # ----------------------------------------------------------------------- #
    # Internals
    # ----------------------------------------------------------------------- #
    async def _list(
        self,
        conditions: list[Any],
        owner_id: uuid.UUID,
        *,
        ascending: bool,
        limit: int,
        offset: int,
    ) -> tuple[list[ExecutionIntervention], int]:
        joined: list[Any] = [*conditions, Execution.created_by == owner_id]
        created = col(ExecutionIntervention.created_at)
        ordering = created.asc() if ascending else created.desc()

        async with AsyncSession(self.engine) as session:
            total = (
                await session.exec(
                    select(func.count())
                    .select_from(ExecutionIntervention)
                    .join(
                        Execution,
                        col(Execution.id) == col(ExecutionIntervention.execution_id),
                    )
                    .where(*joined)
                )
            ).one()
            statement = (
                select(ExecutionIntervention)
                .join(
                    Execution,
                    col(Execution.id) == col(ExecutionIntervention.execution_id),
                )
                .where(*joined)
                .order_by(ordering, col(ExecutionIntervention.id))
                .offset(offset)
                .limit(limit)
            )
            items = list((await session.exec(statement)).all())
            return items, total

    @staticmethod
    async def _lock_execution(
        session: AsyncSession, execution_id: uuid.UUID, owner_id: uuid.UUID
    ) -> Execution | None:
        statement = (
            select(Execution)
            .where(Execution.id == execution_id, Execution.created_by == owner_id)
            .with_for_update()
        )
        return (await session.exec(statement)).first()

    @staticmethod
    async def _append_event(
        session: AsyncSession,
        execution_id: uuid.UUID,
        *,
        event_type: ExecutionEventType,
        actor_type: ActorType,
        actor_user_id: uuid.UUID | None,
        payload: ExecutionEventPayload,
    ) -> ExecutionEvent:
        """Add one event inside the caller's transaction.

        Not EventStore.append: that opens its own transaction, and the whole
        point here is that the event and the intervention land together. There
        is no client_event_id either -- these events are the server's own, so
        there is nothing for a bot to replay.
        """
        event = ExecutionEvent(
            execution_id=execution_id,
            seq=await allocate_event_seq(session, execution_id),
            event_type=event_type,
            actor_type=actor_type,
            actor_user_id=actor_user_id,
            payload=payload,
        )
        session.add(event)
        await session.flush()
        return event

    @staticmethod
    async def _sync_execution_status(
        session: AsyncSession, execution: Execution
    ) -> None:
        """Point the run's status at whatever is still pending, if anything.

        One rule for both opening and resolving: the status follows the oldest
        request still open -- the one a person will answer next -- and goes back
        to running when there are none. A finished run is left alone; answering
        a question it left behind must not bring it back to life.
        """
        if execution.status not in BLOCKABLE_STATUSES:
            return

        oldest = (
            await session.exec(
                select(ExecutionIntervention)
                .where(
                    ExecutionIntervention.execution_id == execution.id,
                    ExecutionIntervention.status == InterventionStatus.PENDING,
                )
                .order_by(
                    col(ExecutionIntervention.created_at),
                    col(ExecutionIntervention.id),
                )
                .limit(1)
            )
        ).first()

        target = (
            WAITING_FOR_KIND[oldest.kind] if oldest is not None else ExecutionStatus.RUNNING
        )
        if execution.status is not target:
            execution.status = target
            session.add(execution)
