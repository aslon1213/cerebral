import uuid
from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Any

import sqlalchemy as sa
from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator
from pydantic import Field as PydanticField
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import DateTime, Field, String, col, func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.repo.base import (
    ExecutionHistoryExistsError,
    InvalidTransitionError,
    RepoAlreadyAttachedError,
    SortOrder,
    TimestampMixin,
    enum_check,
)
from app.repo.git_repo import cerebral_ref
from app.repo.task import Task
from app.repo.types import PydanticJSONB, StrEnumType
from app.repo.utils import utcnow

# git object ids are stored as String, not CHAR: `character(n)` is blank-padded
# by Postgres, so a 40-hex SHA-1 would come back as 40 chars plus 24 spaces and
# fail every equality check in Python. 64 leaves room for SHA-256 repos.
OID = String(64)


class ExecutorType(StrEnum):
    HUMAN = "human"
    AI_AGENT = "ai_agent"


class ExecutionStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    WAITING_INPUT = "waiting_input"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = (
    ExecutionStatus.SUCCEEDED,
    ExecutionStatus.FAILED,
    ExecutionStatus.CANCELLED,
)

# The state machine, in Python, ahead of the check constraints.
#
# The constraints below already make most illegal states unstorable, but they
# reject them as a 500 naming a constraint. Deciding here means a client that
# completes a run twice gets a 409 that says so. Notably:
#   - waiting_* cannot succeed. A run blocked on a question has not finished its
#     work, and completing it would strand the intervention as pending forever.
#     It has to come back to running first, which is what resolving one does.
#   - pending can be cancelled: a run may be abandoned before the agent ever
#     picks it up. ck_executions_finished_after_started then forces a started_at
#     it never really had, so `transition` stamps both timestamps together.
#   - terminal is terminal. A rerun is a new attempt, not a resurrection.
LEGAL_TRANSITIONS: dict[ExecutionStatus, frozenset[ExecutionStatus]] = {
    ExecutionStatus.PENDING: frozenset(
        {ExecutionStatus.RUNNING, ExecutionStatus.CANCELLED}
    ),
    ExecutionStatus.RUNNING: frozenset(
        {
            ExecutionStatus.WAITING_APPROVAL,
            ExecutionStatus.WAITING_INPUT,
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        }
    ),
    # waiting_* to the other waiting_* is legal because several interventions
    # may be open at once: resolving the approval an agent was blocked on can
    # leave it blocked on a question that was asked later. No route drives this
    # -- InterventionStore keeps the status in step with what is still pending.
    ExecutionStatus.WAITING_APPROVAL: frozenset(
        {
            ExecutionStatus.RUNNING,
            ExecutionStatus.WAITING_INPUT,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        }
    ),
    ExecutionStatus.WAITING_INPUT: frozenset(
        {
            ExecutionStatus.RUNNING,
            ExecutionStatus.WAITING_APPROVAL,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        }
    ),
    ExecutionStatus.SUCCEEDED: frozenset(),
    ExecutionStatus.FAILED: frozenset(),
    ExecutionStatus.CANCELLED: frozenset(),
}


class ExecutionError(BaseModel):
    """Structured failure detail, stored in ``executions.error``.

    A bare string cannot answer "should the caller retry this?" without parsing
    prose, which is the one thing the API layer actually needs to know.
    """

    code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] = PydanticField(default_factory=dict)


class Execution(TimestampMixin, table=True):
    """One run of an agent (or a human) against a task.

    Holds the mutable state of the run. The immutable record of what happened
    lives in ``execution_events``; per-repo git state lives in
    ``execution_repos``.
    """

    __tablename__ = "executions"  # pyright: ignore[reportAssignmentType]
    __table_args__ = (
        sa.Index("ix_executions_task_created", "task_id", "created_at"),
        sa.Index("ix_executions_created_by_created", "created_by", "created_at"),
        sa.UniqueConstraint("task_id", "attempt", name="uq_executions_task_attempt"),
        enum_check("executor_type", ExecutorType, "executor_type"),
        enum_check("status", ExecutionStatus, "status"),
        sa.CheckConstraint(
            "(executor_type = 'human'    AND executor_user_id  IS NOT NULL AND executor_agent_id IS NULL)"
            " OR "
            "(executor_type = 'ai_agent' AND executor_agent_id IS NOT NULL AND executor_user_id  IS NULL)",
            name="executor_matches_type",
        ),
        sa.CheckConstraint(
            "finished_at IS NULL"
            " OR (started_at IS NOT NULL AND finished_at >= started_at)",
            name="finished_after_started",
        ),
        sa.CheckConstraint(
            "status NOT IN ('succeeded', 'failed', 'cancelled')"
            " OR finished_at IS NOT NULL",
            name="terminal_has_finished_at",
        ),
        # An error is only meaningful on a failed run, and a failed run must say
        # why it failed.
        sa.CheckConstraint(
            "(status = 'failed') = (error IS NOT NULL)",
            name="error_matches_status",
        ),
        sa.CheckConstraint(
            "input_tokens >= 0 AND output_tokens >= 0 AND cache_read_tokens >= 0"
            " AND (cost_usd IS NULL OR cost_usd >= 0)",
            name="usage_non_negative",
        ),
        sa.CheckConstraint("last_event_seq >= 0", name="last_seq_non_negative"),
    )

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    # RESTRICT, not CASCADE: an execution is an audit record. Deleting the task
    # it belongs to must not silently erase what an agent did to a codebase --
    # the history has to be dropped explicitly first. Note this makes
    # TaskRepo.delete fail for tasks that have executions.
    task_id: uuid.UUID = Field(foreign_key="tasks.id", ondelete="RESTRICT")
    # Denormalised owner. Every other list endpoint filters by owner, and
    # without this column each one needs executions -> tasks -> projects.
    created_by: uuid.UUID = Field(foreign_key="users.id", ondelete="RESTRICT")
    attempt: int = Field(default=1)

    executor_type: ExecutorType = Field(
        default=ExecutorType.HUMAN,
        sa_type=StrEnumType(ExecutorType, 16),  # pyright: ignore[reportArgumentType]
    )
    executor_user_id: uuid.UUID | None = Field(
        default=None, foreign_key="users.id", ondelete="RESTRICT"
    )
    executor_agent_id: uuid.UUID | None = Field(
        default=None, foreign_key="agents.id", ondelete="RESTRICT"
    )

    status: ExecutionStatus = Field(
        default=ExecutionStatus.PENDING,
        sa_type=StrEnumType(ExecutionStatus, 24),  # pyright: ignore[reportArgumentType]
    )  # pyright: ignore[reportArgumentType]
    additional_context: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=sa.Column(JSONB, nullable=False, server_default="{}"),
    )
    # none_as_null: otherwise a None result is stored as the JSON scalar
    # 'null', which is not SQL NULL, and "no result yet" becomes unqueryable.
    result: dict[str, Any] | None = Field(
        default=None, sa_column=sa.Column(JSONB(none_as_null=True))
    )
    error: ExecutionError | None = Field(
        default=None, sa_column=sa.Column(PydanticJSONB(ExecutionError), nullable=True)
    )

    # Allocator for execution_events.seq. Appending an event does
    #   UPDATE executions SET last_event_seq = last_event_seq + 1
    #    WHERE id = :id RETURNING last_event_seq
    # in the same transaction as the INSERT. The row lock serialises concurrent
    # appends to one execution, so two writers cannot pick the same seq -- which
    # is what a naive `SELECT max(seq) + 1` would do under the unique index.
    last_event_seq: int = Field(
        default=0,
        sa_column=sa.Column(sa.Integer, nullable=False, server_default="0"),
    )

    # Run metadata: what produced this, and what it cost.
    model: str | None = Field(default=None, sa_type=String(128))  # pyright: ignore[reportArgumentType]
    provider: str | None = Field(default=None, sa_type=String(64))  # pyright: ignore[reportArgumentType]
    input_tokens: int = Field(default=0)
    output_tokens: int = Field(default=0)
    cache_read_tokens: int = Field(default=0)
    cost_usd: Decimal | None = Field(
        default=None, sa_column=sa.Column(sa.Numeric(14, 6), nullable=True)
    )

    started_at: datetime | None = Field(default=None, sa_type=DateTime(timezone=True))  # pyright: ignore[reportArgumentType]
    finished_at: datetime | None = Field(default=None, sa_type=DateTime(timezone=True))  # pyright: ignore[reportArgumentType]


class ExecutionRepoLink(TimestampMixin, table=True):
    """The git state of one execution in one repo.

    An execution can span several repos, and every git coordinate an execution
    has -- the ref it works on, where it started, where it is now, where it
    landed -- is *per repo*. A ``list[uuid]`` column on ``executions`` could
    hold the repo ids but nowhere to put any of that, so the list is this table.
    Zero rows means the execution touched no code.

    Lifecycle: the agent commits onto ``ref_name`` (under refs/cerebral) as it
    works, advancing ``head_commit_sha``. When the run finishes, that range is
    squashed into one or more commits on the default namespace, recorded in
    ``landed_commit_shas``, with ``merge_commit_sha`` naming the merge commit if
    one was created.
    """

    __tablename__ = "execution_repos"  # pyright: ignore[reportAssignmentType]
    __table_args__ = (
        sa.CheckConstraint(
            "(landed_at IS NULL) = (jsonb_array_length(landed_commit_shas) = 0)",
            name="landed_consistent",
        ),
        sa.CheckConstraint(
            "merge_commit_sha IS NULL OR landed_at IS NOT NULL",
            name="merge_implies_landed",
        ),
    )

    execution_id: uuid.UUID = Field(
        foreign_key="executions.id", primary_key=True, ondelete="RESTRICT"
    )
    repo_id: uuid.UUID = Field(
        foreign_key="git_repos.id", primary_key=True, ondelete="RESTRICT"
    )

    # refs/cerebral/executions/<execution_id>; see git_repo.cerebral_ref().
    # Stored rather than derived so the ref layout can change without
    # orphaning the commits of runs that used the old one.
    ref_name: str = Field(sa_type=String(512))  # pyright: ignore[reportArgumentType]
    # The commit the run started from. Without it there is no diff range for
    # the execution, and no way to tell whether the branch moved underneath it.
    base_commit_sha: str = Field(sa_type=OID)  # pyright: ignore[reportArgumentType]
    # Tip of ref_name, advanced as the agent commits.
    head_commit_sha: str | None = Field(default=None, sa_type=OID)  # pyright: ignore[reportArgumentType]

    # Set once the run's work is squashed onto the default namespace.
    landed_branch: str | None = Field(default=None, sa_type=String(255))  # pyright: ignore[reportArgumentType]
    landed_commit_shas: list[str] = Field(
        default_factory=list,
        sa_column=sa.Column(JSONB, nullable=False, server_default="[]"),
    )
    merge_commit_sha: str | None = Field(default=None, sa_type=OID)  # pyright: ignore[reportArgumentType]
    landed_at: datetime | None = Field(default=None, sa_type=DateTime(timezone=True))  # pyright: ignore[reportArgumentType]


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #
# A git object id is 40 hex for SHA-1 and 64 for SHA-256; the column is
# String(64) either way. Length is all that is checked -- validating the alphabet
# here would reject a repo format we have not met yet for no gain. The lower
# bound leaves room for the abbreviated ids people paste from `git log`.
GitSha = Annotated[str, StringConstraints(min_length=7, max_length=64)]


class ExecutionRepoAttach(BaseModel):
    """A repo the run works in, and the commit it starts from."""

    repo_id: uuid.UUID
    base_commit_sha: GitSha


class ExecutionRepoHead(BaseModel):
    """The new tip of the run's cerebral ref in one repo."""

    head_commit_sha: GitSha


class ExecutionRepoLand(BaseModel):
    """Where the run's work ended up once it left the cerebral namespace.

    ``landed_commit_shas`` cannot be empty: ck_execution_repos_landed_consistent
    ties it to ``landed_at``, so a landing with nothing to point at is a landing
    that did not happen.
    """

    landed_branch: str = PydanticField(min_length=1, max_length=255)
    landed_commit_shas: list[GitSha] = PydanticField(min_length=1)
    merge_commit_sha: GitSha | None = None


class ExecutionCreate(BaseModel):
    """Everything needed to start a run, in one call.

    ``project_id`` is carried alongside ``task_id`` because the routes are flat:
    the API checks the task really belongs to that project rather than taking
    the client's word for the pairing.

    Absent on purpose: ``executor_agent_id``, which is read off the API key, and
    ``ref_name``, which the server derives from the execution id. A bot must not
    be able to claim an agent it was not issued for, or point its commits at a
    ref belonging to another run.
    """

    project_id: uuid.UUID
    task_id: uuid.UUID
    executor_type: ExecutorType = ExecutorType.AI_AGENT
    model: str | None = PydanticField(default=None, max_length=128)
    provider: str | None = PydanticField(default=None, max_length=64)
    additional_context: dict[str, Any] = PydanticField(default_factory=dict)
    repos: list[ExecutionRepoAttach] = PydanticField(default_factory=list)

    @model_validator(mode="after")
    def _repos_are_distinct(self) -> "ExecutionCreate":
        # execution_repos is keyed on (execution_id, repo_id), so a repeated
        # repo would come back as a primary key violation from deep inside the
        # insert. Rejected here it is a 422 naming the field.
        seen = {attachment.repo_id for attachment in self.repos}
        if len(seen) != len(self.repos):
            raise ValueError("repos must not name the same repo twice")
        return self


class ExecutionComplete(BaseModel):
    result: dict[str, Any] | None = None


class ExecutionFail(BaseModel):
    """``/fail`` carries the error rather than inferring one.

    ck_executions_error_matches_status makes a failed run without an error
    unstorable, and a failure nobody can act on is not worth recording.
    """

    error: ExecutionError


class ExecutionUsage(BaseModel):
    """Usage to add to the running totals, not the totals themselves.

    A bot reports what a turn cost as it goes and does not track the sum, so
    these accumulate. Every field defaults to zero: a caller that only knows the
    cost sends only the cost.
    """

    input_tokens: int = PydanticField(default=0, ge=0)
    output_tokens: int = PydanticField(default=0, ge=0)
    cache_read_tokens: int = PydanticField(default=0, ge=0)
    cost_usd: Decimal | None = PydanticField(default=None, ge=0)


class ExecutionRepoResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    repo_id: uuid.UUID
    ref_name: str
    base_commit_sha: str
    head_commit_sha: str | None
    landed_branch: str | None
    landed_commit_shas: list[str]
    merge_commit_sha: str | None
    landed_at: datetime | None
    created_at: datetime
    updated_at: datetime


class ExecutionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    task_id: uuid.UUID
    created_by: uuid.UUID
    attempt: int
    executor_type: ExecutorType
    executor_user_id: uuid.UUID | None
    executor_agent_id: uuid.UUID | None
    status: ExecutionStatus
    model: str | None
    provider: str | None
    additional_context: dict[str, Any]
    result: dict[str, Any] | None
    error: ExecutionError | None
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cost_usd: Decimal | None
    # The high-water mark of the event log. A client tailing the transcript can
    # tell from this alone whether it is behind, without fetching any events.
    last_event_seq: int
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    updated_at: datetime


class ExecutionDetail(ExecutionResponse):
    """One execution with its per-repo git state inlined.

    Only the single-execution read carries this: a list of runs across a project
    would otherwise fan out into a repo query per row.
    """

    repos: list[ExecutionRepoResponse]

    @classmethod
    def from_row(
        cls, execution: Execution, links: Sequence[ExecutionRepoLink]
    ) -> "ExecutionDetail":
        return cls(
            **ExecutionResponse.model_validate(execution).model_dump(),
            repos=[ExecutionRepoResponse.model_validate(link) for link in links],
        )


class ExecutionSort(StrEnum):
    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"
    STARTED_AT = "started_at"
    FINISHED_AT = "finished_at"
    ATTEMPT = "attempt"
    STATUS = "status"


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #
class ExecutionStore:
    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    @staticmethod
    def _sort_column(sort_by: ExecutionSort):  # pyright: ignore[reportUnknownParameterType]
        return {
            ExecutionSort.CREATED_AT: col(Execution.created_at),
            ExecutionSort.UPDATED_AT: col(Execution.updated_at),
            ExecutionSort.STARTED_AT: col(Execution.started_at),
            ExecutionSort.FINISHED_AT: col(Execution.finished_at),
            ExecutionSort.ATTEMPT: col(Execution.attempt),
            ExecutionSort.STATUS: col(Execution.status),
        }[sort_by]

    async def create(
        self, execution: Execution, repos: Sequence[ExecutionRepoAttach] = ()
    ) -> Execution:
        """Start a run: allocate its attempt number and attach its repos.

        One transaction, because an execution whose repo links did not make it
        is a run that cannot record a single code change -- code_changes carries
        a composite foreign key into execution_repos.
        """
        execution_id, task_id = execution.id, execution.task_id
        async with AsyncSession(self.engine) as session:
            connection = await session.connection()
            # Lock the task for the length of the transaction. `attempt` is
            # allocated as max + 1 and uq_executions_task_attempt rejects a tie,
            # so two runs starting on one task at the same moment have to queue
            # -- the same reason allocate_event_seq locks the execution row.
            await connection.execute(
                sa.text("SELECT id FROM tasks WHERE id = :task_id FOR UPDATE")
                .bindparams(task_id=task_id)
            )
            taken = (
                await session.exec(
                    select(func.max(Execution.attempt)).where(
                        Execution.task_id == task_id
                    )
                )
            ).one()
            execution.attempt = (taken or 0) + 1

            session.add(execution)
            session.add_all(
                [
                    ExecutionRepoLink(
                        execution_id=execution_id,
                        repo_id=attachment.repo_id,
                        # Derived, never taken from the client: the ref namespace
                        # is what keeps one run's commits out of another's.
                        ref_name=cerebral_ref(execution_id),
                        base_commit_sha=attachment.base_commit_sha,
                    )
                    for attachment in repos
                ]
            )
            await session.commit()
            await session.refresh(execution)
            return execution

    async def get(
        self, execution_id: uuid.UUID, owner_id: uuid.UUID
    ) -> Execution | None:
        async with AsyncSession(self.engine) as session:
            statement = select(Execution).where(
                Execution.id == execution_id, Execution.created_by == owner_id
            )
            return (await session.exec(statement)).first()

    async def get_owner(self, execution_id: uuid.UUID) -> uuid.UUID | None:
        """Owner of a run regardless of who is asking, for access checks."""
        async with AsyncSession(self.engine) as session:
            statement = select(Execution.created_by).where(
                Execution.id == execution_id
            )
            return (await session.exec(statement)).first()

    async def repos_of(self, execution_id: uuid.UUID) -> list[ExecutionRepoLink]:
        async with AsyncSession(self.engine) as session:
            statement = (
                select(ExecutionRepoLink)
                .where(ExecutionRepoLink.execution_id == execution_id)
                .order_by(col(ExecutionRepoLink.created_at), col(ExecutionRepoLink.repo_id))
            )
            return list((await session.exec(statement)).all())

    async def list(
        self,
        owner_id: uuid.UUID,
        *,
        task_id: uuid.UUID | None = None,
        project_id: uuid.UUID | None = None,
        status: ExecutionStatus | None = None,
        agent_id: uuid.UUID | None = None,
        repo_id: uuid.UUID | None = None,
        sort_by: ExecutionSort = ExecutionSort.CREATED_AT,
        order: SortOrder = SortOrder.DESC,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Execution], int]:
        conditions: list[Any] = [Execution.created_by == owner_id]
        if task_id is not None:
            conditions.append(Execution.task_id == task_id)
        if project_id is not None:
            # Executions carry no project_id: a run belongs to a task, and the
            # task says which project that is. Subqueries rather than joins so
            # the row count cannot be inflated by the filter.
            conditions.append(
                col(Execution.task_id).in_(
                    select(Task.id).where(Task.project_id == project_id)
                )
            )
        if status is not None:
            conditions.append(Execution.status == status)
        if agent_id is not None:
            conditions.append(Execution.executor_agent_id == agent_id)
        if repo_id is not None:
            conditions.append(
                col(Execution.id).in_(
                    select(ExecutionRepoLink.execution_id).where(
                        ExecutionRepoLink.repo_id == repo_id
                    )
                )
            )

        column = self._sort_column(sort_by)
        ordering = column.desc() if order is SortOrder.DESC else column.asc()

        async with AsyncSession(self.engine) as session:
            total = (
                await session.exec(
                    select(func.count()).select_from(Execution).where(*conditions)
                )
            ).one()
            statement = (
                select(Execution)
                .where(*conditions)
                # started_at and finished_at are NULL for a run still queued;
                # nulls_last keeps those at the end rather than at whichever end
                # Postgres defaults to for the direction.
                .order_by(ordering.nulls_last(), col(Execution.id))
                .offset(offset)
                .limit(limit)
            )
            items = list((await session.exec(statement)).all())
            return items, total

    async def transition(
        self,
        execution_id: uuid.UUID,
        owner_id: uuid.UUID,
        target: ExecutionStatus,
        *,
        error: ExecutionError | None = None,
        result: dict[str, Any] | None = None,
    ) -> Execution | None:
        """Move a run to ``target``, or raise if that is not a legal step.

        Raises :class:`InvalidTransitionError` rather than letting the check
        constraints decide, so an illegal step is a 409 that names both states.
        Returns None when there is no such run for this owner.
        """
        async with AsyncSession(self.engine) as session:
            statement = (
                select(Execution)
                .where(Execution.id == execution_id, Execution.created_by == owner_id)
                # Locked for the transaction: two callers completing the same
                # run would otherwise both read `running` and both pass the
                # check below, and the second would move a finished run.
                .with_for_update()
            )
            execution = (await session.exec(statement)).first()
            if execution is None:
                return None

            current = execution.status
            if target not in LEGAL_TRANSITIONS[current]:
                raise InvalidTransitionError(current, target)

            moment = utcnow()
            if execution.started_at is None and (
                target is ExecutionStatus.RUNNING or target in TERMINAL_STATUSES
            ):
                # A run cancelled before it ever started still needs a
                # started_at: ck_executions_finished_after_started refuses a
                # finished_at without one.
                execution.started_at = moment
            if target in TERMINAL_STATUSES:
                execution.finished_at = moment
            if target is ExecutionStatus.FAILED:
                execution.error = error
            if result is not None:
                execution.result = result
            execution.status = target

            session.add(execution)
            await session.commit()
            await session.refresh(execution)
            return execution

    async def add_usage(
        self,
        execution_id: uuid.UUID,
        owner_id: uuid.UUID,
        usage: ExecutionUsage,
    ) -> Execution | None:
        """Add a turn's usage to the run's totals.

        Incremented in SQL rather than read-modify-written: usage arrives from
        the ingest path, where two reports for one run overlap routinely and a
        Python-side sum would drop one of them.

        Allowed in any status, including a terminal one -- the final token count
        often arrives after the agent has already reported that it finished.
        """
        assignments = [
            "input_tokens = input_tokens + :input_tokens",
            "output_tokens = output_tokens + :output_tokens",
            "cache_read_tokens = cache_read_tokens + :cache_read_tokens",
            "updated_at = now()",
        ]
        params: dict[str, Any] = {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read_tokens": usage.cache_read_tokens,
            "execution_id": execution_id,
            "owner_id": owner_id,
        }
        if usage.cost_usd is not None:
            # NULL means "no cost reported yet", so the first report starts the
            # sum at zero instead of leaving NULL + x = NULL.
            assignments.append("cost_usd = coalesce(cost_usd, 0) + :cost_usd")
            params["cost_usd"] = usage.cost_usd

        async with AsyncSession(self.engine) as session:
            connection = await session.connection()
            updated = (
                await connection.execute(
                    sa.text(
                        f"UPDATE executions SET {', '.join(assignments)}"
                        " WHERE id = :execution_id AND created_by = :owner_id"
                        " RETURNING id"
                    ).bindparams(**params)
                )
            ).first()
            if updated is None:
                return None
            await session.commit()

        return await self.get(execution_id, owner_id)

    async def attach_repo(
        self,
        execution_id: uuid.UUID,
        owner_id: uuid.UUID,
        attachment: ExecutionRepoAttach,
    ) -> ExecutionRepoLink | None:
        """Add a repo to a run already under way.

        An agent does not always know every repo it will touch when it starts --
        it may follow an import into a sibling checkout halfway through.
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

            link = ExecutionRepoLink(
                execution_id=execution_id,
                repo_id=attachment.repo_id,
                ref_name=cerebral_ref(execution_id),
                base_commit_sha=attachment.base_commit_sha,
            )
            session.add(link)
            try:
                await session.commit()
            except IntegrityError as exc:
                await session.rollback()
                raise RepoAlreadyAttachedError(attachment.repo_id) from exc
            await session.refresh(link)
            return link

    async def set_head(
        self,
        execution_id: uuid.UUID,
        owner_id: uuid.UUID,
        repo_id: uuid.UUID,
        head_commit_sha: str,
    ) -> ExecutionRepoLink | None:
        """Advance the tip of the run's cerebral ref in one repo."""
        async with AsyncSession(self.engine) as session:
            link = await self._owned_link(session, execution_id, owner_id, repo_id)
            if link is None:
                return None
            link.head_commit_sha = head_commit_sha
            session.add(link)
            await session.commit()
            await session.refresh(link)
            return link

    async def land(
        self,
        execution_id: uuid.UUID,
        owner_id: uuid.UUID,
        repo_id: uuid.UUID,
        *,
        landed_branch: str,
        landed_commit_shas: Sequence[str],
        merge_commit_sha: str | None = None,
    ) -> ExecutionRepoLink | None:
        """Record that the run's work was squashed onto the default namespace.

        While the run is under way its commits live on ``ref_name``, one per
        event; landing merges that range into the default namespace. Every
        change this repo recorded therefore lands in the same commit, which is
        why one sha can be stamped across all of them.

        Stamped in the same transaction as the link: a repo link that says
        "landed" while its changes still read NULL would break the file-history
        query for exactly the runs that finished properly.
        """
        async with AsyncSession(self.engine) as session:
            link = await self._owned_link(session, execution_id, owner_id, repo_id)
            if link is None:
                return None

            link.landed_branch = landed_branch
            link.landed_commit_shas = list(landed_commit_shas)
            link.merge_commit_sha = merge_commit_sha
            link.landed_at = utcnow()
            session.add(link)
            await session.flush()

            connection = await session.connection()
            await connection.execute(
                sa.text(
                    "UPDATE code_changes SET landed_commit_sha = :sha"
                    " WHERE execution_id = :execution_id AND repo_id = :repo_id"
                ).bindparams(
                    # The commit the run's cerebral range became on the default
                    # namespace. Which commit produced any individual change is
                    # a separate question, and it is already answered: that is
                    # the cerebral commit on the change's event.
                    sha=landed_commit_shas[0],
                    execution_id=execution_id,
                    repo_id=repo_id,
                )
            )
            await session.commit()
            await session.refresh(link)
            return link

    async def delete(
        self, execution_id: uuid.UUID, owner_id: uuid.UUID, *, purge: bool = False
    ) -> bool:
        """Delete a run. Without ``purge`` it refuses to erase any history.

        Everything under an execution references it under RESTRICT, so the
        deletes have to run innermost first. Events and code changes both
        reference themselves as well -- an event's parent, a change's revert --
        and RESTRICT is checked per row rather than per statement, so a single
        DELETE could try to remove a parent before its child. Hence the loops:
        each pass removes whatever is now a leaf.
        """
        async with AsyncSession(self.engine) as session:
            # Ownership checked with a SELECT rather than folded into the
            # DELETE, so someone else's run reads as missing instead of as a
            # silent success.
            owned = (
                await session.exec(
                    select(Execution.id).where(
                        Execution.id == execution_id,
                        Execution.created_by == owner_id,
                    )
                )
            ).first()
            if owned is None:
                return False

            connection = await session.connection()
            try:
                if purge:
                    await self._delete_leaves(
                        connection,
                        "code_changes",
                        execution_id,
                        self_reference="reverts_change_id",
                    )
                    await connection.execute(
                        sa.text(
                            "DELETE FROM execution_interventions"
                            " WHERE execution_id = :execution_id"
                        ).bindparams(execution_id=execution_id)
                    )
                    await self._delete_leaves(
                        connection,
                        "execution_events",
                        execution_id,
                        self_reference="parent_event_id",
                    )
                for statement in (
                    "DELETE FROM execution_repos WHERE execution_id = :execution_id",
                    "DELETE FROM executions WHERE id = :execution_id",
                ):
                    await connection.execute(
                        sa.text(statement).bindparams(execution_id=execution_id)
                    )
                await session.commit()
            except IntegrityError as exc:
                await session.rollback()
                if purge:
                    # Something outside this run still points into it -- a later
                    # execution that reverted one of its changes, say. RESTRICT
                    # is doing its job; the answer is not to delete harder.
                    raise ExecutionHistoryExistsError(
                        "Part of this execution's history is referenced by "
                        "another execution and cannot be deleted."
                    ) from exc
                raise ExecutionHistoryExistsError() from exc
            return True

    # ----------------------------------------------------------------------- #
    # Internals
    # ----------------------------------------------------------------------- #
    @staticmethod
    async def _owned_link(
        session: AsyncSession,
        execution_id: uuid.UUID,
        owner_id: uuid.UUID,
        repo_id: uuid.UUID,
    ) -> ExecutionRepoLink | None:
        """The repo link of an execution this owner has, or None."""
        statement = (
            select(ExecutionRepoLink)
            .join(Execution, col(Execution.id) == col(ExecutionRepoLink.execution_id))
            .where(
                ExecutionRepoLink.execution_id == execution_id,
                ExecutionRepoLink.repo_id == repo_id,
                Execution.created_by == owner_id,
            )
        )
        return (await session.exec(statement)).first()

    @staticmethod
    async def _delete_leaves(
        connection: Any, table: str, execution_id: uuid.UUID, *, self_reference: str
    ) -> None:
        """Empty ``table`` for one execution, leaves of the self-reference first.

        Bounded rather than `while True`: a row referenced from outside this
        execution can never become a leaf, and the loop must not spin on it.
        Whatever survives is left to RESTRICT to refuse, which is where the
        caller turns it into a 409.
        """
        for _ in range(64):
            deleted = await connection.execute(
                sa.text(
                    f"DELETE FROM {table} AS t"
                    "  WHERE t.execution_id = :execution_id"
                    f"   AND NOT EXISTS (SELECT 1 FROM {table} AS child"
                    f"                    WHERE child.{self_reference} = t.id)"
                ).bindparams(execution_id=execution_id)
            )
            if deleted.rowcount == 0:
                return
