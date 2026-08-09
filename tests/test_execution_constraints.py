"""What the database must refuse to store.

A CheckConstraint that compiles is not a CheckConstraint that works -- the
original ``ChangeType`` constraints compiled fine while comparing 'created'
against an enum whose only labels were 'CREATED', which would have rejected
every row. Each test here writes a bad row and asserts Postgres rejects it.
"""

import uuid
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel.ext.asyncio.session import AsyncSession

from app.repo.execution import (
    ChangeType,
    CodeChange,
    Execution,
    ExecutionEvent,
    ExecutionEventPayload,
    ExecutionEventType,
    ExecutionIntervention,
    ExecutionRepoLink,
    ExecutionStatus,
    ExecutorType,
    InterventionKind,
    InterventionStatus,
)
from app.repo.execution.history import ActorType
from app.repo.git_repo import GitRepo, cerebral_ref

from .conftest import BlobFactory, Seed
from app.repo.utils import utcnow

from .test_execution_lifecycle import append, start_run

# asyncpg surfaces check violations as DBAPIError and key violations as
# IntegrityError; both are acceptable proof that the write was refused.
REJECTED = (IntegrityError, DBAPIError)


class TestExecutionConstraints:
    async def test_status_outside_the_enum_is_rejected(
        self, engine: AsyncEngine, seed: Seed
    ):
        """The String + CHECK pairing has to actually constrain the column."""
        execution_id = await start_run(engine, seed, repos=[])
        with pytest.raises(REJECTED):
            async with AsyncSession(engine) as session:
                connection = await session.connection()
                await connection.execute(
                    sa.text(
                        "UPDATE executions SET status = 'bogus' WHERE id = :id"
                    ).bindparams(id=execution_id)
                )
                await session.commit()

    async def test_change_type_accepts_lowercase_values(
        self, engine: AsyncEngine, seed: Seed, blob: BlobFactory
    ):
        """Regression: a native PG enum stores member *names*, so the labels
        would have been 'CREATED' and this constraint would reject everything."""
        execution_id = await start_run(engine, seed, repos=[seed.api_repo_id])
        async with AsyncSession(engine) as session:
            event = await append(
                session,
                execution_id,
                event_type=ExecutionEventType.CODE_CHANGE,
                payload=ExecutionEventPayload(),
            )
            session.add(
                CodeChange(
                    execution_id=execution_id,
                    event_id=event.id,
                    repo_id=seed.api_repo_id,
                    change_type=ChangeType.CREATED,
                    path="app/x.py",
                    after_blob=blob(),
                )
            )
            await session.commit()  # must not raise

    async def test_ai_run_cannot_have_a_human_executor(
        self, engine: AsyncEngine, seed: Seed
    ):
        with pytest.raises(REJECTED):
            async with AsyncSession(engine) as session:
                session.add(
                    Execution(
                        task_id=seed.task_id,
                        created_by=seed.user_id,
                        executor_type=ExecutorType.AI_AGENT,
                        executor_user_id=seed.user_id,  # wrong column for the type
                    )
                )
                await session.commit()

    async def test_human_run_cannot_have_an_agent_executor(
        self, engine: AsyncEngine, seed: Seed
    ):
        with pytest.raises(REJECTED):
            async with AsyncSession(engine) as session:
                session.add(
                    Execution(
                        task_id=seed.task_id,
                        created_by=seed.user_id,
                        executor_type=ExecutorType.HUMAN,
                        executor_agent_id=seed.agent_id,
                    )
                )
                await session.commit()

    async def test_terminal_status_requires_finished_at(
        self, engine: AsyncEngine, seed: Seed
    ):
        execution_id = await start_run(engine, seed, repos=[])
        with pytest.raises(REJECTED):
            async with AsyncSession(engine) as session:
                execution = await session.get(Execution, execution_id)
                assert execution is not None
                execution.status = ExecutionStatus.SUCCEEDED  # no finished_at
                session.add(execution)
                await session.commit()

    async def test_finished_at_cannot_precede_started_at(
        self, engine: AsyncEngine, seed: Seed
    ):
        execution_id = await start_run(engine, seed, repos=[])
        with pytest.raises(REJECTED):
            async with AsyncSession(engine) as session:
                connection = await session.connection()
                await connection.execute(
                    sa.text(
                        "UPDATE executions SET status = 'cancelled',"
                        " finished_at = started_at - interval '1 hour'"
                        " WHERE id = :id"
                    ).bindparams(id=execution_id)
                )
                await session.commit()

    async def test_failed_run_must_say_why(self, engine: AsyncEngine, seed: Seed):
        execution_id = await start_run(engine, seed, repos=[])
        with pytest.raises(REJECTED):
            async with AsyncSession(engine) as session:
                execution = await session.get(Execution, execution_id)
                assert execution is not None
                execution.status = ExecutionStatus.FAILED
                execution.finished_at = utcnow()  # but error stays NULL
                session.add(execution)
                await session.commit()

    async def test_error_on_a_succeeded_run_is_rejected(
        self, engine: AsyncEngine, seed: Seed
    ):
        execution_id = await start_run(engine, seed, repos=[])
        with pytest.raises(REJECTED):
            async with AsyncSession(engine) as session:
                connection = await session.connection()
                await connection.execute(
                    sa.text(
                        "UPDATE executions"
                        " SET status = 'succeeded', finished_at = now(),"
                        '     error = \'{"code":"x","message":"y"}\'::jsonb'
                        " WHERE id = :id"
                    ).bindparams(id=execution_id)
                )
                await session.commit()

    async def test_negative_token_counts_are_rejected(
        self, engine: AsyncEngine, seed: Seed
    ):
        execution_id = await start_run(engine, seed, repos=[])
        with pytest.raises(REJECTED):
            async with AsyncSession(engine) as session:
                connection = await session.connection()
                await connection.execute(
                    sa.text(
                        "UPDATE executions SET output_tokens = -1 WHERE id = :id"
                    ).bindparams(id=execution_id)
                )
                await session.commit()

    async def test_two_runs_cannot_share_an_attempt_number(
        self, engine: AsyncEngine, seed: Seed
    ):
        await start_run(engine, seed, repos=[])
        with pytest.raises(REJECTED):
            async with AsyncSession(engine) as session:
                session.add(
                    Execution(
                        task_id=seed.task_id,
                        created_by=seed.user_id,
                        attempt=1,  # already taken
                        executor_type=ExecutorType.AI_AGENT,
                        executor_agent_id=seed.agent_id,
                    )
                )
                await session.commit()


class TestEventConstraints:
    async def test_duplicate_seq_is_rejected(self, engine: AsyncEngine, seed: Seed):
        execution_id = await start_run(engine, seed, repos=[])
        async with AsyncSession(engine) as session:
            await append(
                session,
                execution_id,
                event_type=ExecutionEventType.REASONING,
                payload=ExecutionEventPayload(),
            )
            await session.commit()

        with pytest.raises(REJECTED):
            async with AsyncSession(engine) as session:
                session.add(
                    ExecutionEvent(
                        execution_id=execution_id,
                        seq=1,  # collides
                        event_type=ExecutionEventType.REASONING,
                        payload=ExecutionEventPayload(),
                    )
                )
                await session.commit()

    async def test_replayed_client_event_id_is_rejected(
        self, engine: AsyncEngine, seed: Seed
    ):
        execution_id = await start_run(engine, seed, repos=[])
        async with AsyncSession(engine) as session:
            await append(
                session,
                execution_id,
                client_event_id="evt-1",
                event_type=ExecutionEventType.CHAT_MESSAGE,
                payload=ExecutionEventPayload(),
            )
            await session.commit()

        with pytest.raises(REJECTED):
            async with AsyncSession(engine) as session:
                await append(
                    session,
                    execution_id,
                    client_event_id="evt-1",  # same key, different seq
                    event_type=ExecutionEventType.CHAT_MESSAGE,
                    payload=ExecutionEventPayload(),
                )
                await session.commit()

    async def test_same_client_event_id_in_another_execution_is_fine(
        self, engine: AsyncEngine, seed: Seed
    ):
        """Idempotency is scoped per execution; agents restart their counters."""
        first = await start_run(engine, seed, repos=[])
        async with AsyncSession(engine) as session:
            execution = Execution(
                task_id=seed.task_id,
                created_by=seed.user_id,
                attempt=2,
                executor_type=ExecutorType.AI_AGENT,
                executor_agent_id=seed.agent_id,
            )
            second = execution.id
            session.add(execution)
            await session.commit()

        async with AsyncSession(engine) as session:
            for execution_id in (first, second):
                await append(
                    session,
                    execution_id,
                    client_event_id="evt-1",
                    event_type=ExecutionEventType.CHAT_MESSAGE,
                    payload=ExecutionEventPayload(),
                )
            await session.commit()  # must not raise

    async def test_user_event_without_a_user_id_is_rejected(
        self, engine: AsyncEngine, seed: Seed
    ):
        execution_id = await start_run(engine, seed, repos=[])
        with pytest.raises(REJECTED):
            async with AsyncSession(engine) as session:
                await append(
                    session,
                    execution_id,
                    event_type=ExecutionEventType.CHAT_MESSAGE,
                    actor_type=ActorType.USER,  # but no actor_user_id
                    payload=ExecutionEventPayload(),
                )
                await session.commit()

    async def test_agent_event_carrying_a_user_id_is_rejected(
        self, engine: AsyncEngine, seed: Seed
    ):
        execution_id = await start_run(engine, seed, repos=[])
        with pytest.raises(REJECTED):
            async with AsyncSession(engine) as session:
                await append(
                    session,
                    execution_id,
                    event_type=ExecutionEventType.CHAT_MESSAGE,
                    actor_type=ActorType.AGENT,
                    actor_user_id=seed.user_id,
                    payload=ExecutionEventPayload(),
                )
                await session.commit()

    async def test_seq_must_be_positive(self, engine: AsyncEngine, seed: Seed):
        execution_id = await start_run(engine, seed, repos=[])
        with pytest.raises(REJECTED):
            async with AsyncSession(engine) as session:
                session.add(
                    ExecutionEvent(
                        execution_id=execution_id,
                        seq=0,
                        event_type=ExecutionEventType.REASONING,
                        payload=ExecutionEventPayload(),
                    )
                )
                await session.commit()


class TestCodeChangeConstraints:
    async def test_change_in_an_unattached_repo_is_rejected(
        self, engine: AsyncEngine, seed: Seed, blob: BlobFactory
    ):
        """The composite FK: a change may only name a repo the execution
        actually works in. docs_repo is never attached to any execution."""
        execution_id = await start_run(engine, seed, repos=[seed.api_repo_id])
        with pytest.raises(REJECTED):
            async with AsyncSession(engine) as session:
                event = await append(
                    session,
                    execution_id,
                    event_type=ExecutionEventType.CODE_CHANGE,
                    payload=ExecutionEventPayload(),
                )
                session.add(
                    CodeChange(
                        execution_id=execution_id,
                        event_id=event.id,
                        repo_id=seed.docs_repo_id,
                        change_type=ChangeType.CREATED,
                        path="README.md",
                        after_blob=blob(),
                    )
                )
                await session.commit()

    async def test_created_change_cannot_have_a_before_blob(
        self, engine: AsyncEngine, seed: Seed, blob: BlobFactory
    ):
        execution_id = await start_run(engine, seed, repos=[seed.api_repo_id])
        with pytest.raises(REJECTED):
            async with AsyncSession(engine) as session:
                event = await append(
                    session,
                    execution_id,
                    event_type=ExecutionEventType.CODE_CHANGE,
                    payload=ExecutionEventPayload(),
                )
                session.add(
                    CodeChange(
                        execution_id=execution_id,
                        event_id=event.id,
                        repo_id=seed.api_repo_id,
                        change_type=ChangeType.CREATED,
                        path="app/x.py",
                        before_blob=blob(),  # a new file has no before
                        after_blob=blob(),
                    )
                )
                await session.commit()

    async def test_deleted_change_cannot_have_an_after_blob(
        self, engine: AsyncEngine, seed: Seed, blob: BlobFactory
    ):
        execution_id = await start_run(engine, seed, repos=[seed.api_repo_id])
        with pytest.raises(REJECTED):
            async with AsyncSession(engine) as session:
                event = await append(
                    session,
                    execution_id,
                    event_type=ExecutionEventType.CODE_CHANGE,
                    payload=ExecutionEventPayload(),
                )
                session.add(
                    CodeChange(
                        execution_id=execution_id,
                        event_id=event.id,
                        repo_id=seed.api_repo_id,
                        change_type=ChangeType.DELETED,
                        path="app/x.py",
                        before_blob=blob(),
                        after_blob=blob(),
                    )
                )
                await session.commit()

    async def test_rename_requires_a_previous_path(
        self, engine: AsyncEngine, seed: Seed, blob: BlobFactory
    ):
        execution_id = await start_run(engine, seed, repos=[seed.api_repo_id])
        with pytest.raises(REJECTED):
            async with AsyncSession(engine) as session:
                event = await append(
                    session,
                    execution_id,
                    event_type=ExecutionEventType.CODE_CHANGE,
                    payload=ExecutionEventPayload(),
                )
                session.add(
                    CodeChange(
                        execution_id=execution_id,
                        event_id=event.id,
                        repo_id=seed.api_repo_id,
                        change_type=ChangeType.RENAMED,
                        path="app/new.py",  # previous_path missing
                        before_blob=blob(),
                        after_blob=blob(),
                    )
                )
                await session.commit()

    async def test_non_rename_cannot_carry_a_previous_path(
        self, engine: AsyncEngine, seed: Seed, blob: BlobFactory
    ):
        execution_id = await start_run(engine, seed, repos=[seed.api_repo_id])
        with pytest.raises(REJECTED):
            async with AsyncSession(engine) as session:
                event = await append(
                    session,
                    execution_id,
                    event_type=ExecutionEventType.CODE_CHANGE,
                    payload=ExecutionEventPayload(),
                )
                session.add(
                    CodeChange(
                        execution_id=execution_id,
                        event_id=event.id,
                        repo_id=seed.api_repo_id,
                        change_type=ChangeType.MODIFIED,
                        path="app/x.py",
                        previous_path="app/old.py",
                        before_blob=blob(),
                        after_blob=blob(),
                    )
                )
                await session.commit()

    async def test_duplicate_seq_within_one_event_is_rejected(
        self, engine: AsyncEngine, seed: Seed, blob: BlobFactory
    ):
        execution_id = await start_run(engine, seed, repos=[seed.api_repo_id])
        with pytest.raises(REJECTED):
            async with AsyncSession(engine) as session:
                event = await append(
                    session,
                    execution_id,
                    event_type=ExecutionEventType.CODE_CHANGE,
                    payload=ExecutionEventPayload(),
                )
                session.add_all(
                    [
                        CodeChange(
                            execution_id=execution_id,
                            event_id=event.id,
                            repo_id=seed.api_repo_id,
                            seq=0,
                            change_type=ChangeType.CREATED,
                            path=f"app/{name}.py",
                            after_blob=blob(),
                        )
                        for name in ("a", "b")
                    ]
                )
                await session.commit()

    async def test_a_change_cannot_revert_itself(
        self, engine: AsyncEngine, seed: Seed, blob: BlobFactory
    ):
        execution_id = await start_run(engine, seed, repos=[seed.api_repo_id])
        with pytest.raises(REJECTED):
            async with AsyncSession(engine) as session:
                event = await append(
                    session,
                    execution_id,
                    event_type=ExecutionEventType.CODE_CHANGE,
                    payload=ExecutionEventPayload(),
                )
                change = CodeChange(
                    execution_id=execution_id,
                    event_id=event.id,
                    repo_id=seed.api_repo_id,
                    change_type=ChangeType.CREATED,
                    path="app/x.py",
                    after_blob=blob(),
                )
                change.reverts_change_id = change.id
                session.add(change)
                await session.commit()


class TestInterventionConstraints:
    async def _open(
        self, engine: AsyncEngine, execution_id: uuid.UUID, **kwargs: Any
    ) -> uuid.UUID:
        async with AsyncSession(engine) as session:
            intervention = ExecutionIntervention(
                execution_id=execution_id,
                kind=InterventionKind.APPROVAL,
                request={"question": "apply the patch?"},
                **kwargs,
            )
            intervention_id = intervention.id
            session.add(intervention)
            await session.commit()
        return intervention_id

    async def test_a_new_one_may_open_once_the_first_resolves(
        self, engine: AsyncEngine, seed: Seed
    ):
        execution_id = await start_run(engine, seed, repos=[])
        first = await self._open(engine, execution_id)
        async with AsyncSession(engine) as session:
            intervention = await session.get(ExecutionIntervention, first)
            assert intervention is not None
            intervention.status = InterventionStatus.APPROVED
            intervention.resolved_at = utcnow()
            intervention.resolved_by_user_id = seed.user_id
            session.add(intervention)
            await session.commit()
        await self._open(engine, execution_id)  # must not raise

    async def test_a_question_cannot_be_approved(self, engine: AsyncEngine, seed: Seed):
        """approved/rejected belong to approvals; a question is answered."""
        execution_id = await start_run(engine, seed, repos=[])
        with pytest.raises(REJECTED):
            async with AsyncSession(engine) as session:
                session.add(
                    ExecutionIntervention(
                        execution_id=execution_id,
                        kind=InterventionKind.INPUT_REQUIRED,
                        status=InterventionStatus.APPROVED,
                        resolved_at=utcnow(),
                    )
                )
                await session.commit()

    async def test_a_question_can_be_answered(self, engine: AsyncEngine, seed: Seed):
        execution_id = await start_run(engine, seed, repos=[])
        async with AsyncSession(engine) as session:
            session.add(
                ExecutionIntervention(
                    execution_id=execution_id,
                    kind=InterventionKind.INPUT_REQUIRED,
                    status=InterventionStatus.ANSWERED,
                    response={"answer": "use postgres"},
                    resolved_at=utcnow(),
                    resolved_by_user_id=seed.user_id,
                )
            )
            await session.commit()  # must not raise

    async def test_an_approval_cannot_be_answered(
        self, engine: AsyncEngine, seed: Seed
    ):
        execution_id = await start_run(engine, seed, repos=[])
        with pytest.raises(REJECTED):
            async with AsyncSession(engine) as session:
                session.add(
                    ExecutionIntervention(
                        execution_id=execution_id,
                        kind=InterventionKind.APPROVAL,
                        status=InterventionStatus.ANSWERED,
                        resolved_at=utcnow(),
                    )
                )
                await session.commit()

    async def test_resolved_status_requires_a_resolved_at(
        self, engine: AsyncEngine, seed: Seed
    ):
        execution_id = await start_run(engine, seed, repos=[])
        with pytest.raises(REJECTED):
            async with AsyncSession(engine) as session:
                session.add(
                    ExecutionIntervention(
                        execution_id=execution_id,
                        kind=InterventionKind.APPROVAL,
                        status=InterventionStatus.REJECTED,  # resolved_at NULL
                    )
                )
                await session.commit()

    async def test_pending_status_forbids_a_resolved_at(
        self, engine: AsyncEngine, seed: Seed
    ):
        execution_id = await start_run(engine, seed, repos=[])
        with pytest.raises(REJECTED):
            async with AsyncSession(engine) as session:
                session.add(
                    ExecutionIntervention(
                        execution_id=execution_id,
                        kind=InterventionKind.APPROVAL,
                        status=InterventionStatus.PENDING,
                        resolved_at=utcnow(),
                    )
                )
                await session.commit()

    async def test_expiring_requires_a_deadline(self, engine: AsyncEngine, seed: Seed):
        execution_id = await start_run(engine, seed, repos=[])
        with pytest.raises(REJECTED):
            async with AsyncSession(engine) as session:
                session.add(
                    ExecutionIntervention(
                        execution_id=execution_id,
                        kind=InterventionKind.APPROVAL,
                        status=InterventionStatus.EXPIRED,
                        resolved_at=utcnow(),  # but expires_at is NULL
                    )
                )
                await session.commit()

    async def test_cannot_point_at_another_executions_event(
        self, engine: AsyncEngine, seed: Seed
    ):
        """The composite FK stops an intervention borrowing a foreign event."""
        first = await start_run(engine, seed, repos=[])
        async with AsyncSession(engine) as session:
            event = await append(
                session,
                first,
                event_type=ExecutionEventType.INTERVENTION_REQUESTED,
                payload=ExecutionEventPayload(),
            )
            foreign_event_id = event.id
            await session.commit()

        async with AsyncSession(engine) as session:
            other = Execution(
                task_id=seed.task_id,
                created_by=seed.user_id,
                attempt=2,
                executor_type=ExecutorType.AI_AGENT,
                executor_agent_id=seed.agent_id,
            )
            other_id = other.id
            session.add(other)
            await session.commit()

        with pytest.raises(REJECTED):
            async with AsyncSession(engine) as session:
                session.add(
                    ExecutionIntervention(
                        execution_id=other_id,
                        event_id=foreign_event_id,
                        kind=InterventionKind.APPROVAL,
                    )
                )
                await session.commit()

    async def test_intervention_and_its_event_link_both_ways(
        self, engine: AsyncEngine, seed: Seed
    ):
        execution_id = await start_run(engine, seed, repos=[])
        async with AsyncSession(engine) as session:
            event = await append(
                session,
                execution_id,
                event_type=ExecutionEventType.INTERVENTION_REQUESTED,
                payload=ExecutionEventPayload(data={"question": "apply?"}),
            )
            event_id = event.id
            session.add(
                ExecutionIntervention(
                    execution_id=execution_id,
                    event_id=event_id,
                    kind=InterventionKind.APPROVAL,
                    request={"question": "apply?"},
                )
            )
            await session.commit()

        async with AsyncSession(engine) as session:
            connection = await session.connection()
            row = (
                await connection.execute(
                    sa.text(
                        "SELECT iv.status, ev.seq, ev.event_type"
                        " FROM execution_interventions iv"
                        " JOIN execution_events ev ON ev.id = iv.event_id"
                        " WHERE iv.execution_id = :ex"
                    ).bindparams(ex=execution_id)
                )
            ).one()
        assert row.status == InterventionStatus.PENDING
        assert row.event_type == ExecutionEventType.INTERVENTION_REQUESTED


class TestRepoLinkConstraints:
    async def test_landing_needs_both_a_time_and_a_commit(
        self, engine: AsyncEngine, seed: Seed
    ):
        execution_id = await start_run(engine, seed, repos=[seed.api_repo_id])
        with pytest.raises(REJECTED):
            async with AsyncSession(engine) as session:
                link = await session.get(
                    ExecutionRepoLink, (execution_id, seed.api_repo_id)
                )
                assert link is not None
                link.landed_at = utcnow()  # but landed_commit_shas stays empty
                session.add(link)
                await session.commit()

    async def test_merge_commit_without_landing_is_rejected(
        self, engine: AsyncEngine, seed: Seed, blob: BlobFactory
    ):
        execution_id = await start_run(engine, seed, repos=[seed.api_repo_id])
        with pytest.raises(REJECTED):
            async with AsyncSession(engine) as session:
                link = await session.get(
                    ExecutionRepoLink, (execution_id, seed.api_repo_id)
                )
                assert link is not None
                link.merge_commit_sha = blob()
                session.add(link)
                await session.commit()

    async def test_a_repo_attaches_to_an_execution_only_once(
        self, engine: AsyncEngine, seed: Seed, blob: BlobFactory
    ):
        execution_id = await start_run(engine, seed, repos=[seed.api_repo_id])
        with pytest.raises(REJECTED):
            async with AsyncSession(engine) as session:
                session.add(
                    ExecutionRepoLink(
                        execution_id=execution_id,
                        repo_id=seed.api_repo_id,
                        ref_name=cerebral_ref(execution_id),
                        base_commit_sha=blob(),
                    )
                )
                await session.commit()


class TestHistoryIsProtected:
    """RESTRICT everywhere: an audit trail must not evaporate as a side effect
    of deleting something upstream."""

    async def test_execution_with_history_cannot_be_deleted(
        self, engine: AsyncEngine, seed: Seed
    ):
        execution_id = await start_run(engine, seed, repos=[])
        async with AsyncSession(engine) as session:
            await append(
                session,
                execution_id,
                event_type=ExecutionEventType.REASONING,
                payload=ExecutionEventPayload(),
            )
            await session.commit()

        with pytest.raises(REJECTED):
            async with AsyncSession(engine) as session:
                connection = await session.connection()
                await connection.execute(
                    sa.text("DELETE FROM executions WHERE id = :id").bindparams(
                        id=execution_id
                    )
                )
                await session.commit()

    async def test_task_with_executions_cannot_be_deleted(
        self, engine: AsyncEngine, seed: Seed
    ):
        await start_run(engine, seed, repos=[])
        with pytest.raises(REJECTED):
            async with AsyncSession(engine) as session:
                connection = await session.connection()
                await connection.execute(
                    sa.text("DELETE FROM tasks WHERE id = :id").bindparams(
                        id=seed.task_id
                    )
                )
                await session.commit()

    async def test_user_with_executions_cannot_be_deleted(
        self, engine: AsyncEngine, seed: Seed
    ):
        await start_run(engine, seed, repos=[])
        with pytest.raises(REJECTED):
            async with AsyncSession(engine) as session:
                connection = await session.connection()
                await connection.execute(
                    sa.text("DELETE FROM users WHERE id = :id").bindparams(
                        id=seed.user_id
                    )
                )
                await session.commit()

    async def test_repo_in_use_cannot_be_deleted(self, engine: AsyncEngine, seed: Seed):
        await start_run(engine, seed, repos=[seed.api_repo_id])
        with pytest.raises(REJECTED):
            async with AsyncSession(engine) as session:
                connection = await session.connection()
                await connection.execute(
                    sa.text("DELETE FROM git_repos WHERE id = :id").bindparams(
                        id=seed.api_repo_id
                    )
                )
                await session.commit()

    async def test_history_can_still_be_dropped_deliberately(
        self, engine: AsyncEngine, seed: Seed, blob: BlobFactory
    ):
        """RESTRICT is not a dead end -- it forces the order, innermost first."""
        execution_id = await start_run(engine, seed, repos=[seed.api_repo_id])
        async with AsyncSession(engine) as session:
            event = await append(
                session,
                execution_id,
                event_type=ExecutionEventType.CODE_CHANGE,
                payload=ExecutionEventPayload(),
            )
            session.add(
                CodeChange(
                    execution_id=execution_id,
                    event_id=event.id,
                    repo_id=seed.api_repo_id,
                    change_type=ChangeType.CREATED,
                    path="app/x.py",
                    after_blob=blob(),
                )
            )
            await session.commit()

        async with AsyncSession(engine) as session:
            connection = await session.connection()
            for statement in (
                "DELETE FROM code_changes WHERE execution_id = :id",
                "DELETE FROM execution_events WHERE execution_id = :id",
                "DELETE FROM execution_repos WHERE execution_id = :id",
                "DELETE FROM executions WHERE id = :id",
            ):
                await connection.execute(sa.text(statement).bindparams(id=execution_id))
            await session.commit()

        async with AsyncSession(engine) as session:
            connection = await session.connection()
            remaining = (
                await connection.execute(sa.text("SELECT count(*) FROM executions"))
            ).scalar_one()
        assert remaining == 0


class TestGitRepoConstraints:
    async def test_two_users_may_each_own_a_repo_named_api(
        self, engine: AsyncEngine, seed: Seed
    ):
        """The old model had a globally unique name, so one user's 'api'
        blocked everyone else's."""
        async with AsyncSession(engine) as session:
            session.add(
                GitRepo(
                    created_by=seed.other_user_id,
                    name="api",
                    local_path="/elsewhere/api",
                )
            )
            await session.commit()  # must not raise

    async def test_one_user_cannot_own_two_repos_named_api(
        self, engine: AsyncEngine, seed: Seed
    ):
        with pytest.raises(REJECTED):
            async with AsyncSession(engine) as session:
                session.add(
                    GitRepo(
                        created_by=seed.user_id,
                        name="api",
                        local_path="/duplicate/api",
                    )
                )
                await session.commit()
