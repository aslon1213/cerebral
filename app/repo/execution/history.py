import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

import sqlalchemy as sa
from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import Field, String, col, func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.repo.base import (
    CreatedAtMixin,
    InvalidCodeChangeError,
    RepoNotAttachedError,
    UnknownParentEventError,
    enum_check,
)
from app.repo.types import PydanticJSONB, StrEnumType

from .codebase import CodeChange, CodeChangeCreate, CodeChangeResponse
from .execution import (
    OID,
    Execution,
    ExecutionRepoLink,
    ExecutionResponse,
    GitSha,
)


class ExecutionEventType(StrEnum):
    CHAT_MESSAGE = "chat_message"
    REASONING = "reasoning"
    DECISION = "decision"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    CODE_CHANGE = "code_change"
    STATUS_CHANGE = "status_change"
    INTERVENTION_REQUESTED = "intervention_requested"
    INTERVENTION_RESOLVED = "intervention_resolved"
    MEMORY_LOADED = "memory_loaded"
    MEMORY_SAVED = "memory_saved"


class ActorType(StrEnum):
    """Who produced an event.

    Needed because a chat_message flows in both directions. Without it the
    agent's output and the user's reply are indistinguishable rows, and the
    transcript can only be rendered by inspecting the payload.
    """

    AGENT = "agent"
    USER = "user"
    SYSTEM = "system"


class ExecutionEventPayload(BaseModel):
    """The body of an event, stored as JSONB via :class:`PydanticJSONB`.

    ``data`` stays untyped because its shape depends on ``event_type``. The
    natural next step is a discriminated union keyed on the event type and
    validated at the API boundary -- that needs no column change, only a
    different model passed to the decorator.
    """

    # Optional: a tool_result or status_change has no reasoning behind it.
    reasoning: str | None = None
    data: dict[str, Any] = PydanticField(default_factory=dict)
    # Denormalised, ordered view of the changes this event produced. The
    # authoritative link is code_changes.event_id -- this list lives inside a
    # JSON document, so the database cannot enforce it as a foreign key. The
    # API writes both in one transaction; treat a mismatch as a bug and read
    # from code_changes when correctness matters.
    code_changes: list[uuid.UUID] = PydanticField(default_factory=list)


class ExecutionEvent(CreatedAtMixin, table=True):
    """One append-only entry in an execution's history.

    Rows are never updated or deleted; a correction is a new event. Hence
    CreatedAtMixin rather than TimestampMixin -- there is no updated_at to
    tempt anyone into writing an UPDATE.
    """

    __tablename__ = "execution_events"  # pyright: ignore[reportAssignmentType]
    __table_args__ = (
        # Also the index the transcript is read through, and the pagination
        # cursor: ORDER BY seq is a total order, unlike created_at which ties.
        sa.UniqueConstraint("execution_id", "seq", name="uq_execution_events_seq"),
        # Idempotency. Agents append over an unreliable channel, so retries and
        # at-least-once delivery will replay an append. The API upserts on this
        # key, making a replay a no-op instead of a duplicate transcript entry.
        sa.UniqueConstraint(
            "execution_id",
            "client_event_id",
            name="uq_execution_events_client_event_id",
        ),
        # Lets execution_interventions carry a composite FK, so an intervention
        # cannot point at an event belonging to a different execution.
        sa.UniqueConstraint(
            "id", "execution_id", name="uq_execution_events_id_execution"
        ),
        enum_check("event_type", ExecutionEventType, "event_type"),
        enum_check("actor_type", ActorType, "actor_type"),
        sa.CheckConstraint(
            "(actor_type = 'user') = (actor_user_id IS NOT NULL)",
            name="user_actor_has_user_id",
        ),
        sa.CheckConstraint("seq > 0", name="seq_positive"),
    )

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    # RESTRICT rather than CASCADE: code_changes restricts on event_id, so a
    # cascade arriving from executions would be blocked part-way and fail the
    # whole delete. Dropping history is explicit, innermost table first.
    execution_id: uuid.UUID = Field(
        foreign_key="executions.id", index=True, ondelete="RESTRICT"
    )
    # Allocated from executions.last_event_seq; see the note on that column.
    seq: int
    client_event_id: str | None = Field(default=None, sa_type=String(128))  # pyright: ignore[reportArgumentType]

    event_type: ExecutionEventType = Field(sa_type=StrEnumType(ExecutionEventType, 32))  # pyright: ignore[reportArgumentType]
    actor_type: ActorType = Field(
        default=ActorType.AGENT,
        sa_type=StrEnumType(ActorType, 16),  # pyright: ignore[reportArgumentType]
    )  # pyright: ignore[reportArgumentType]
    # Only set for actor_type='user'; the agent's identity is on the execution,
    # so it is not repeated on every row.
    actor_user_id: uuid.UUID | None = Field(
        default=None, foreign_key="users.id", ondelete="RESTRICT"
    )
    # Correlates a tool_result back to its tool_call, and any event back to the
    # one that caused it. Without it the pairing is guesswork over ordering.
    parent_event_id: uuid.UUID | None = Field(
        default=None, foreign_key="execution_events.id", index=True, ondelete="RESTRICT"
    )

    payload: ExecutionEventPayload = Field(
        sa_column=sa.Column(
            PydanticJSONB(ExecutionEventPayload), nullable=False, server_default="{}"
        ),
    )
    # The commit this event produced on refs/cerebral/*, if it touched code.
    cerebral_commit_sha: str | None = Field(default=None, sa_type=OID)  # pyright: ignore[reportArgumentType]


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #
class ExecutionEventPayloadIn(BaseModel):
    """The half of an event payload a client writes.

    ``code_changes`` is deliberately missing compared to
    :class:`ExecutionEventPayload`: that list is the server's index of the rows
    it just inserted. A client-supplied one could name changes that do not
    exist, or belong to somebody else's event.
    """

    reasoning: str | None = None
    data: dict[str, Any] = PydanticField(default_factory=dict)


class ExecutionEventCreate(BaseModel):
    """One entry appended to the transcript, with any code it changed.

    ``client_event_id`` is what makes the append safe to retry. An agent posts
    over an unreliable channel, so a delivered append whose response was lost
    will be sent again; keyed on this, the second one is absorbed and answers
    with the event the first one created.
    """

    client_event_id: str | None = PydanticField(
        default=None, min_length=1, max_length=128
    )
    event_type: ExecutionEventType
    actor_type: ActorType = ActorType.AGENT
    # Correlates a tool_result back to its tool_call. Must name an event of this
    # same execution.
    parent_event_id: uuid.UUID | None = None
    cerebral_commit_sha: GitSha | None = None
    payload: ExecutionEventPayloadIn = PydanticField(
        default_factory=ExecutionEventPayloadIn
    )
    code_changes: list[CodeChangeCreate] = PydanticField(default_factory=list)


class ExecutionEventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    execution_id: uuid.UUID
    seq: int
    client_event_id: str | None
    event_type: ExecutionEventType
    actor_type: ActorType
    actor_user_id: uuid.UUID | None
    parent_event_id: uuid.UUID | None
    payload: ExecutionEventPayload
    cerebral_commit_sha: str | None
    created_at: datetime


class ExecutionEventDetail(ExecutionEventResponse):
    """An event with the changes it produced expanded.

    ``payload.code_changes`` holds the same ids in the same order; this is the
    authoritative version, read straight from ``code_changes``.
    """

    code_changes: list[CodeChangeResponse]

    @classmethod
    def from_row(
        cls, event: ExecutionEvent, changes: "list[CodeChange]"
    ) -> "ExecutionEventDetail":
        return cls(
            **ExecutionEventResponse.model_validate(event).model_dump(),
            code_changes=[
                CodeChangeResponse.model_validate(change) for change in changes
            ],
        )


class EventPage(BaseModel):
    """A page of transcript, cursored on ``seq``.

    Not :class:`app.repo.base.Page`, which is offset based. This is an
    append-only log that a client reads while an agent is still writing to it,
    and an offset shifts under every concurrent append -- page two would skip
    or repeat whatever arrived in between. ``seq`` is a stable total order, so
    "everything after 42" means the same thing however much has landed since.
    """

    items: list[ExecutionEventResponse]
    limit: int
    # Feed back as ``after_seq`` to get the next page. None when the page is
    # empty, i.e. the client has caught up.
    next_after_seq: int | None
    has_more: bool


class CodeChangeHistoryEntry(BaseModel):
    """One change to a file, next to the reasoning that produced it.

    This is the answer to "why is this line here?" -- a code change joined back
    to the event that caused it, and so to what the agent was thinking.

    Pointers, never content. ``change.before_blob`` and ``change.after_blob``
    are what a client renders the diff from against its own checkout, and the
    two commits are both here: the cerebral-side one on the event, which is the
    commit that produced the change, and ``change.landed_commit_sha``, where the
    run's work ended up on the default namespace.
    """

    change: CodeChangeResponse
    event: ExecutionEventResponse
    # Which run, in enough detail to say "the nightly bot, on its second try",
    # without a second request per row.
    attempt: int
    executor_agent_id: uuid.UUID | None

    @classmethod
    def from_row(
        cls,
        change: CodeChange,
        event: ExecutionEvent,
        execution: Execution,
    ) -> "CodeChangeHistoryEntry":
        return cls(
            change=CodeChangeResponse.model_validate(change),
            event=ExecutionEventResponse.model_validate(event),
            attempt=execution.attempt,
            executor_agent_id=execution.executor_agent_id,
        )


class CodeChangeContext(BaseModel):
    """One change with everything around it: its event, and its whole run."""

    change: CodeChangeResponse
    event: ExecutionEventResponse
    execution: ExecutionResponse

    @classmethod
    def from_row(
        cls, change: CodeChange, event: ExecutionEvent, execution: Execution
    ) -> "CodeChangeContext":
        return cls(
            change=CodeChangeResponse.model_validate(change),
            event=ExecutionEventResponse.model_validate(event),
            execution=ExecutionResponse.model_validate(execution),
        )


async def allocate_event_seq(session: AsyncSession, execution_id: uuid.UUID) -> int:
    """Reserve the next ``seq`` for an execution's event log.

    Must run in the same transaction as the INSERT that uses the result. The
    UPDATE takes a row lock on the execution, so concurrent appends queue behind
    each other and every caller gets a distinct number -- unlike
    ``SELECT max(seq) + 1``, where two readers see the same value and the second
    INSERT dies on uq_execution_events_seq.
    """
    # Run on the session's connection: session.exec() is typed for SELECTs, and
    # this is DML with a RETURNING clause. It still joins the session's
    # transaction, which is what keeps the row lock held until commit.
    connection = await session.connection()
    result = await connection.execute(
        sa.text(
            "UPDATE executions SET last_event_seq = last_event_seq + 1"
            " WHERE id = :execution_id RETURNING last_event_seq"
        ).bindparams(execution_id=execution_id)
    )
    return result.scalar_one()


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #
class EventStore:
    """The ingest path: appending to an execution's transcript, and reading it.

    There is no update and no delete. A row here is a record of something that
    happened, and a correction is a later event that says so -- rewriting the
    transcript would defeat the reason for keeping one.
    """

    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    async def append(
        self,
        execution_id: uuid.UUID,
        owner_id: uuid.UUID,
        body: ExecutionEventCreate,
    ) -> "tuple[ExecutionEvent, list[CodeChange], bool] | None":
        """Append one event and the changes it produced. Idempotent.

        Returns ``(event, changes, created)``, or None when there is no such
        execution for this owner. ``created`` is False when this was a replay,
        and the event returned is the one the first attempt wrote.

        One transaction throughout: an event whose code changes did not land
        would claim reasoning for edits nobody can find, and changes without
        their event would be edits with no reason at all.
        """
        async with AsyncSession(self.engine) as session:
            owned = (
                await session.exec(
                    select(Execution.id).where(
                        Execution.id == execution_id,
                        Execution.created_by == owner_id,
                    )
                )
            ).first()
            if owned is None:
                return None

            # Look for the replay before allocating anything. The retry that
            # actually happens in production -- response lost, same append sent
            # again -- costs one indexed read here and leaves seq untouched.
            if body.client_event_id is not None:
                existing = await self._by_client_event_id(
                    session, execution_id, body.client_event_id
                )
                if existing is not None:
                    return existing, await self._changes_of(session, existing.id), False

            await self._check_parent(session, execution_id, body)
            await self._check_repos_attached(session, execution_id, body)

            seq = await allocate_event_seq(session, execution_id)
            event_id = uuid.uuid4()
            connection = await session.connection()
            inserted = (
                await connection.execute(
                    pg_insert(ExecutionEvent)
                    .values(
                        id=event_id,
                        execution_id=execution_id,
                        seq=seq,
                        client_event_id=body.client_event_id,
                        event_type=body.event_type,
                        actor_type=body.actor_type,
                        # Taken from the credential, never the body: the only
                        # user an execution's events can be attributed to is the
                        # one the run belongs to.
                        actor_user_id=(
                            owner_id if body.actor_type is ActorType.USER else None
                        ),
                        parent_event_id=body.parent_event_id,
                        cerebral_commit_sha=body.cerebral_commit_sha,
                        payload=ExecutionEventPayload(
                            reasoning=body.payload.reasoning,
                            data=body.payload.data,
                        ),
                    )
                    # The idempotency key. DO NOTHING rather than DO UPDATE: a
                    # replay must not be able to rewrite what the first append
                    # recorded. NULL client_event_ids never conflict, so an
                    # unkeyed append always inserts.
                    .on_conflict_do_nothing(
                        constraint="uq_execution_events_client_event_id"
                    )
                    .returning(col(ExecutionEvent.id))
                )
            ).first()

            if inserted is None:
                # Another writer got this same client_event_id in between the
                # check above and here. Roll back rather than commit: that
                # returns the seq we allocated as well, so a racing replay does
                # not leave a hole in the numbering.
                await session.rollback()
                existing = await self._by_client_event_id(
                    session, execution_id, body.client_event_id
                )
                # ON CONFLICT waits on an uncommitted duplicate and only skips
                # the insert once that writer commits, so the row is there.
                assert existing is not None
                return existing, await self._changes_of(session, existing.id), False

            changes = [
                CodeChange(
                    execution_id=execution_id,
                    event_id=event_id,
                    repo_id=change.repo_id,
                    # Positional: the order they were sent is the order they
                    # happened, and uq_code_changes_event_seq keeps it unique.
                    seq=index,
                    change_type=change.change_type,
                    path=change.path,
                    previous_path=change.previous_path,
                    language=change.language,
                    before_blob=change.before_blob,
                    after_blob=change.after_blob,
                    lines_added=change.lines_added,
                    lines_deleted=change.lines_deleted,
                    diff=change.diff,
                    applied_at=change.applied_at,
                    reverts_change_id=change.reverts_change_id,
                )
                for index, change in enumerate(body.code_changes)
            ]
            if changes:
                session.add_all(changes)
                try:
                    await session.flush()
                except IntegrityError as exc:
                    # blobs_match_type, rename_has_previous_path, a duplicate
                    # path, a reverts_change_id pointing nowhere. The database
                    # is the authority on all of them; this only turns its
                    # refusal into something the caller can read.
                    await session.rollback()
                    raise InvalidCodeChangeError() from exc

                # Backfill the denormalised list now that the ids exist. The
                # whole payload is reassigned rather than edited: PydanticJSONB
                # replaces its value and would not notice a nested mutation.
                await connection.execute(
                    sa.update(ExecutionEvent)
                    .where(col(ExecutionEvent.id) == event_id)
                    .values(
                        payload=ExecutionEventPayload(
                            reasoning=body.payload.reasoning,
                            data=body.payload.data,
                            code_changes=[change.id for change in changes],
                        )
                    )
                )

            await session.commit()

            event = await session.get(ExecutionEvent, event_id)
            assert event is not None  # committed a moment ago
            return event, await self._changes_of(session, event_id), True

    async def list_after(
        self,
        execution_id: uuid.UUID,
        owner_id: uuid.UUID,
        *,
        after_seq: int = 0,
        limit: int = 100,
        event_type: ExecutionEventType | None = None,
        actor_type: ActorType | None = None,
    ) -> tuple[list[ExecutionEvent], bool]:
        """The transcript after ``after_seq``, oldest first.

        Returns ``(events, has_more)``. Ownership is folded into the query
        rather than checked first, so reading a run belonging to someone else
        comes back empty instead of costing a second round trip.
        """
        conditions: list[Any] = [
            ExecutionEvent.execution_id == execution_id,
            col(ExecutionEvent.seq) > after_seq,
            Execution.created_by == owner_id,
        ]
        if event_type is not None:
            conditions.append(ExecutionEvent.event_type == event_type)
        if actor_type is not None:
            conditions.append(ExecutionEvent.actor_type == actor_type)

        async with AsyncSession(self.engine) as session:
            statement = (
                select(ExecutionEvent)
                .join(Execution, col(Execution.id) == col(ExecutionEvent.execution_id))
                .where(*conditions)
                .order_by(col(ExecutionEvent.seq))
                # One more than asked for, to answer has_more without a second
                # count over a table that is being appended to as we read.
                .limit(limit + 1)
            )
            found = list((await session.exec(statement)).all())

        has_more = len(found) > limit
        return found[:limit], has_more

    async def get(
        self, execution_id: uuid.UUID, owner_id: uuid.UUID, event_id: uuid.UUID
    ) -> "tuple[ExecutionEvent, list[CodeChange]] | None":
        async with AsyncSession(self.engine) as session:
            statement = (
                select(ExecutionEvent)
                .join(Execution, col(Execution.id) == col(ExecutionEvent.execution_id))
                .where(
                    ExecutionEvent.id == event_id,
                    ExecutionEvent.execution_id == execution_id,
                    Execution.created_by == owner_id,
                )
            )
            event = (await session.exec(statement)).first()
            if event is None:
                return None
            return event, await self._changes_of(session, event_id)

    # ----------------------------------------------------------------------- #
    # Internals
    # ----------------------------------------------------------------------- #
    @staticmethod
    async def _by_client_event_id(
        session: AsyncSession, execution_id: uuid.UUID, client_event_id: str | None
    ) -> ExecutionEvent | None:
        statement = select(ExecutionEvent).where(
            ExecutionEvent.execution_id == execution_id,
            ExecutionEvent.client_event_id == client_event_id,
        )
        return (await session.exec(statement)).first()

    @staticmethod
    async def _changes_of(
        session: AsyncSession, event_id: uuid.UUID
    ) -> "list[CodeChange]":
        statement = (
            select(CodeChange)
            .where(CodeChange.event_id == event_id)
            .order_by(col(CodeChange.seq), col(CodeChange.id))
        )
        return list((await session.exec(statement)).all())

    @staticmethod
    async def _check_parent(
        session: AsyncSession, execution_id: uuid.UUID, body: ExecutionEventCreate
    ) -> None:
        """A parent must be an event of this same execution.

        The column's foreign key only says the event exists somewhere, so
        without this a bot could hang a tool_result off another run's tool_call
        and the transcript would render a reply to a question nobody asked.
        """
        if body.parent_event_id is None:
            return
        parent = (
            await session.exec(
                select(ExecutionEvent.id).where(
                    ExecutionEvent.id == body.parent_event_id,
                    ExecutionEvent.execution_id == execution_id,
                )
            )
        ).first()
        if parent is None:
            raise UnknownParentEventError(body.parent_event_id)

    @staticmethod
    async def _check_repos_attached(
        session: AsyncSession, execution_id: uuid.UUID, body: ExecutionEventCreate
    ) -> None:
        """Every change has to name a repo the run is actually working in."""
        wanted = {change.repo_id for change in body.code_changes}
        if not wanted:
            return
        attached = set(
            (
                await session.exec(
                    select(ExecutionRepoLink.repo_id).where(
                        ExecutionRepoLink.execution_id == execution_id,
                        col(ExecutionRepoLink.repo_id).in_(wanted),
                    )
                )
            ).all()
        )
        missing = wanted - attached
        if missing:
            raise RepoNotAttachedError(sorted(missing, key=str))


class CodeHistoryStore:
    """Reading what an agent did to a codebase, and why.

    Lives here rather than beside ``CodeChange`` in ``codebase.py`` because
    every query it answers joins code changes to the events that explain them,
    and codebase.py cannot import this module -- EventStore writes code changes,
    so the dependency already runs the other way.
    """

    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    async def list_for_execution(
        self,
        execution_id: uuid.UUID,
        owner_id: uuid.UUID,
        *,
        repo_id: uuid.UUID | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[CodeChange], int]:
        """Everything one run touched, grouped by repo and in the order it
        happened."""
        conditions: list[Any] = [
            CodeChange.execution_id == execution_id,
            Execution.created_by == owner_id,
        ]
        if repo_id is not None:
            conditions.append(CodeChange.repo_id == repo_id)

        async with AsyncSession(self.engine) as session:
            total = (
                await session.exec(
                    select(func.count())
                    .select_from(CodeChange)
                    .join(
                        Execution, col(Execution.id) == col(CodeChange.execution_id)
                    )
                    .where(*conditions)
                )
            ).one()
            statement = (
                select(CodeChange)
                .join(Execution, col(Execution.id) == col(CodeChange.execution_id))
                .where(*conditions)
                .order_by(
                    col(CodeChange.repo_id),
                    col(CodeChange.created_at),
                    col(CodeChange.seq),
                    col(CodeChange.id),
                )
                .offset(offset)
                .limit(limit)
            )
            return list((await session.exec(statement)).all()), total

    async def file_history(
        self,
        repo_id: uuid.UUID,
        owner_id: uuid.UUID,
        *,
        path: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[tuple[CodeChange, ExecutionEvent, Execution]], int]:
        """Every agent change to a file, across every run, newest first.

        The endpoint the whole design exists for. Ordered and filtered to match
        ix_code_changes_repo_path_created, which is (repo_id, path, created_at
        DESC) -- the same shape as the question being asked.

        A rename is visible as its own row carrying ``previous_path``, but this
        does not walk back through one: asking for the new path returns the
        history since the rename. Following it further is a git question, and
        the client has the repo.
        """
        conditions: list[Any] = [
            CodeChange.repo_id == repo_id,
            Execution.created_by == owner_id,
        ]
        if path is not None:
            conditions.append(CodeChange.path == path)

        async with AsyncSession(self.engine) as session:
            total = (
                await session.exec(
                    select(func.count())
                    .select_from(CodeChange)
                    .join(
                        Execution, col(Execution.id) == col(CodeChange.execution_id)
                    )
                    .where(*conditions)
                )
            ).one()
            statement = (
                select(CodeChange, ExecutionEvent, Execution)
                .join(ExecutionEvent, col(ExecutionEvent.id) == col(CodeChange.event_id))
                .join(Execution, col(Execution.id) == col(CodeChange.execution_id))
                .where(*conditions)
                .order_by(
                    col(CodeChange.created_at).desc(), col(CodeChange.id).desc()
                )
                .offset(offset)
                .limit(limit)
            )
            rows = [tuple(row) for row in (await session.exec(statement)).all()]
            return rows, total  # pyright: ignore[reportReturnType]

    async def get_with_context(
        self, repo_id: uuid.UUID, change_id: uuid.UUID, owner_id: uuid.UUID
    ) -> tuple[CodeChange, ExecutionEvent, Execution] | None:
        """One change, its event and the run it belongs to."""
        async with AsyncSession(self.engine) as session:
            statement = (
                select(CodeChange, ExecutionEvent, Execution)
                .join(ExecutionEvent, col(ExecutionEvent.id) == col(CodeChange.event_id))
                .join(Execution, col(Execution.id) == col(CodeChange.execution_id))
                .where(
                    CodeChange.id == change_id,
                    CodeChange.repo_id == repo_id,
                    Execution.created_by == owner_id,
                )
            )
            row = (await session.exec(statement)).first()
            return tuple(row) if row is not None else None  # pyright: ignore[reportReturnType]
