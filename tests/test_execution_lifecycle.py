"""The happy path: a run that spans two repos, logs events, and lands its work."""

import uuid
from collections.abc import Sequence
from decimal import Decimal
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.repo.execution import (
    ChangeType,
    CodeChange,
    CodeDiffApplied,
    Execution,
    ExecutionEvent,
    ExecutionEventPayload,
    ExecutionEventType,
    ExecutionRepoLink,
    ExecutionStatus,
    ExecutorType,
)
from app.repo.execution.history import ActorType, allocate_event_seq
from app.repo.git_repo import CEREBRAL_REF_NAMESPACE, cerebral_ref
from app.repo.utils import utcnow

from .conftest import BlobFactory, Seed


async def start_run(
    engine: AsyncEngine,
    seed: Seed,
    *,
    repos: Sequence[uuid.UUID],
    base: str = "0" * 40,
) -> uuid.UUID:
    """Create a running execution attached to `repos`, returning its id."""
    execution = Execution(
        task_id=seed.task_id,
        created_by=seed.user_id,
        executor_type=ExecutorType.AI_AGENT,
        executor_agent_id=seed.agent_id,
        status=ExecutionStatus.RUNNING,
        model="claude-opus-5",
        provider="anthropic",
        started_at=utcnow(),
    )
    execution_id = execution.id
    async with AsyncSession(engine) as session:
        session.add(execution)
        await session.commit()
    async with AsyncSession(engine) as session:
        session.add_all(
            [
                ExecutionRepoLink(
                    execution_id=execution_id,
                    repo_id=repo_id,
                    ref_name=cerebral_ref(execution_id),
                    base_commit_sha=base,
                )
                for repo_id in repos
            ]
        )
        await session.commit()
    return execution_id


async def append(
    session: AsyncSession, execution_id: uuid.UUID, **kwargs: Any
) -> ExecutionEvent:
    """Append an event the way the API is meant to: allocate, then insert."""
    seq = await allocate_event_seq(session, execution_id)
    event = ExecutionEvent(execution_id=execution_id, seq=seq, **kwargs)
    session.add(event)
    await session.flush()
    return event


class TestExecutionSetup:
    async def test_execution_spans_multiple_repos(
        self, engine: AsyncEngine, seed: Seed
    ):
        execution_id = await start_run(
            engine, seed, repos=[seed.api_repo_id, seed.web_repo_id]
        )
        async with AsyncSession(engine) as session:
            links = (
                await session.exec(
                    select(ExecutionRepoLink).where(
                        ExecutionRepoLink.execution_id == execution_id
                    )
                )
            ).all()
        assert {link.repo_id for link in links} == {seed.api_repo_id, seed.web_repo_id}

    async def test_execution_may_touch_no_repo_at_all(
        self, engine: AsyncEngine, seed: Seed
    ):
        """repo attachment is optional -- a run can be pure conversation."""
        execution_id = await start_run(engine, seed, repos=[])
        async with AsyncSession(engine) as session:
            links = (
                await session.exec(
                    select(ExecutionRepoLink).where(
                        ExecutionRepoLink.execution_id == execution_id
                    )
                )
            ).all()
        assert links == []

    async def test_agent_works_under_the_cerebral_namespace(
        self, engine: AsyncEngine, seed: Seed
    ):
        execution_id = await start_run(engine, seed, repos=[seed.api_repo_id])
        async with AsyncSession(engine) as session:
            link = await session.get(
                ExecutionRepoLink, (execution_id, seed.api_repo_id)
            )
        assert link is not None
        assert link.ref_name.startswith(CEREBRAL_REF_NAMESPACE)
        # Not a branch: invisible to `git branch` and to the default fetch.
        assert not link.ref_name.startswith("refs/heads/")
        assert link.head_commit_sha is None
        assert link.landed_at is None


class TestEventLog:
    async def test_seq_is_allocated_densely_from_one(
        self, engine: AsyncEngine, seed: Seed
    ):
        execution_id = await start_run(engine, seed, repos=[seed.api_repo_id])
        async with AsyncSession(engine) as session:
            events = [
                await append(
                    session,
                    execution_id,
                    event_type=ExecutionEventType.REASONING,
                    payload=ExecutionEventPayload(reasoning=f"step {n}"),
                )
                for n in range(4)
            ]
            seqs = [event.seq for event in events]
            await session.commit()
        assert seqs == [1, 2, 3, 4]

    async def test_concurrent_appends_do_not_collide(
        self, engine: AsyncEngine, seed: Seed
    ):
        """Two sessions appending at once must get different seq values.

        This is the case `SELECT max(seq) + 1` gets wrong: both readers see the
        same maximum and the second INSERT dies on uq_execution_events_seq.
        """
        execution_id = await start_run(engine, seed, repos=[seed.api_repo_id])
        async with AsyncSession(engine) as first, AsyncSession(engine) as second:
            a = await allocate_event_seq(first, execution_id)
            await first.commit()
            b = await allocate_event_seq(second, execution_id)
            await second.commit()
        assert (a, b) == (1, 2)

    async def test_payload_round_trips_as_a_typed_model(
        self, engine: AsyncEngine, seed: Seed
    ):
        """The payload is a Pydantic model in Python and JSONB in Postgres."""
        execution_id = await start_run(engine, seed, repos=[seed.api_repo_id])
        payload = ExecutionEventPayload(
            reasoning="the router never verified the signature",
            data={"text": "add jwt auth", "tool": "Edit"},
        )
        async with AsyncSession(engine) as session:
            event = await append(
                session,
                execution_id,
                event_type=ExecutionEventType.CHAT_MESSAGE,
                payload=payload,
            )
            event_id = event.id
            await session.commit()

        async with AsyncSession(engine) as session:
            stored = await session.get(ExecutionEvent, event_id)
        assert stored is not None
        assert isinstance(stored.payload, ExecutionEventPayload)
        assert stored.payload.reasoning == "the router never verified the signature"
        assert stored.payload.data["tool"] == "Edit"

    async def test_payload_is_queryable_as_json_in_postgres(
        self, engine: AsyncEngine, seed: Seed
    ):
        """Storing a model must not make the column opaque to SQL."""
        execution_id = await start_run(engine, seed, repos=[seed.api_repo_id])
        async with AsyncSession(engine) as session:
            await append(
                session,
                execution_id,
                event_type=ExecutionEventType.DECISION,
                payload=ExecutionEventPayload(reasoning="chose argon2 over bcrypt"),
            )
            await session.commit()

        async with AsyncSession(engine) as session:
            connection = await session.connection()
            found = (
                await connection.execute(
                    sa.text(
                        "SELECT payload->>'reasoning' FROM execution_events"
                        " WHERE payload->>'reasoning' ILIKE :pattern"
                    ).bindparams(pattern="%argon2%")
                )
            ).scalar_one()
        assert found == "chose argon2 over bcrypt"

    async def test_reasoning_is_optional(self, engine: AsyncEngine, seed: Seed):
        """A tool_result has no reasoning behind it; the column must allow that."""
        execution_id = await start_run(engine, seed, repos=[seed.api_repo_id])
        async with AsyncSession(engine) as session:
            event = await append(
                session,
                execution_id,
                event_type=ExecutionEventType.TOOL_RESULT,
                payload=ExecutionEventPayload(data={"bytes": 940}),
            )
            event_id = event.id
            await session.commit()
        async with AsyncSession(engine) as session:
            stored = await session.get(ExecutionEvent, event_id)
        assert stored is not None and stored.payload.reasoning is None

    async def test_tool_result_links_back_to_its_call(
        self, engine: AsyncEngine, seed: Seed
    ):
        execution_id = await start_run(engine, seed, repos=[seed.api_repo_id])
        async with AsyncSession(engine) as session:
            call = await append(
                session,
                execution_id,
                event_type=ExecutionEventType.TOOL_CALL,
                payload=ExecutionEventPayload(data={"tool": "Read"}),
            )
            result = await append(
                session,
                execution_id,
                event_type=ExecutionEventType.TOOL_RESULT,
                parent_event_id=call.id,
                payload=ExecutionEventPayload(data={"bytes": 940}),
            )
            call_id, result_parent = call.id, result.parent_event_id
            await session.commit()
        assert result_parent == call_id

    async def test_user_and_agent_messages_are_distinguishable(
        self, engine: AsyncEngine, seed: Seed
    ):
        """Both directions are chat_message; only actor_type tells them apart."""
        execution_id = await start_run(engine, seed, repos=[seed.api_repo_id])
        async with AsyncSession(engine) as session:
            await append(
                session,
                execution_id,
                event_type=ExecutionEventType.CHAT_MESSAGE,
                actor_type=ActorType.USER,
                actor_user_id=seed.user_id,
                payload=ExecutionEventPayload(data={"text": "add jwt auth"}),
            )
            await append(
                session,
                execution_id,
                event_type=ExecutionEventType.CHAT_MESSAGE,
                actor_type=ActorType.AGENT,
                payload=ExecutionEventPayload(data={"text": "on it"}),
            )
            await session.commit()

        async with AsyncSession(engine) as session:
            events = (
                await session.exec(
                    select(ExecutionEvent)
                    .where(ExecutionEvent.execution_id == execution_id)
                    .order_by(sa.column("seq"))
                )
            ).all()
        assert [event.actor_type for event in events] == [
            ActorType.USER,
            ActorType.AGENT,
        ]
        assert all(isinstance(event.actor_type, ActorType) for event in events)
        assert all(isinstance(event.event_type, ExecutionEventType) for event in events)
        assert events[0].actor_user_id == seed.user_id
        assert events[1].actor_user_id is None

    async def test_replayed_append_is_absorbed_by_the_idempotency_key(
        self, engine: AsyncEngine, seed: Seed
    ):
        """An agent retrying a delivered append must not double the transcript."""
        execution_id = await start_run(engine, seed, repos=[seed.api_repo_id])
        async with AsyncSession(engine) as session:
            await append(
                session,
                execution_id,
                client_event_id="evt-1",
                event_type=ExecutionEventType.CHAT_MESSAGE,
                payload=ExecutionEventPayload(data={"text": "hello"}),
            )
            await session.commit()

        # The API's upsert: ON CONFLICT DO NOTHING against the idempotency key.
        async with AsyncSession(engine) as session:
            connection = await session.connection()
            await connection.execute(
                sa.text(
                    "INSERT INTO execution_events"
                    " (id, created_at, execution_id, seq, client_event_id,"
                    "  event_type, actor_type, payload)"
                    " VALUES (gen_random_uuid(), now(), :ex, 99, 'evt-1',"
                    "  'chat_message', 'agent', '{}'::jsonb)"
                    " ON CONFLICT (execution_id, client_event_id) DO NOTHING"
                ).bindparams(ex=execution_id)
            )
            await session.commit()

        async with AsyncSession(engine) as session:
            count = (
                await session.exec(
                    select(sa.func.count())
                    .select_from(ExecutionEvent)
                    .where(ExecutionEvent.execution_id == execution_id)
                )
            ).one()
        assert count == 1


class TestCodeChanges:
    async def test_change_records_git_coordinates_and_its_reasoning(
        self, engine: AsyncEngine, seed: Seed, blob: BlobFactory
    ):
        execution_id = await start_run(engine, seed, repos=[seed.api_repo_id])
        before, after = blob(), blob()
        async with AsyncSession(engine) as session:
            event = await append(
                session,
                execution_id,
                event_type=ExecutionEventType.CODE_CHANGE,
                cerebral_commit_sha=blob(),
                payload=ExecutionEventPayload(reasoning="jwt verify was missing"),
            )
            change = CodeChange(
                execution_id=execution_id,
                event_id=event.id,
                repo_id=seed.api_repo_id,
                seq=0,
                change_type=ChangeType.MODIFIED,
                path="app/auth.py",
                language="python",
                before_blob=before,
                after_blob=after,
                lines_added=12,
                lines_deleted=3,
                applied_at=utcnow(),
            )
            session.add(change)
            await session.flush()
            # The event's denormalised list is written in the same transaction.
            event.payload = ExecutionEventPayload(
                reasoning="jwt verify was missing", code_changes=[change.id]
            )
            session.add(event)
            change_id, event_id = change.id, event.id
            await session.commit()

        async with AsyncSession(engine) as session:
            stored_event = await session.get(ExecutionEvent, event_id)
            stored_change = await session.get(CodeChange, change_id)
        assert stored_change is not None
        assert (stored_change.before_blob, stored_change.after_blob) == (before, after)
        assert stored_event is not None
        # UUIDs survive the JSON round-trip as UUIDs, not strings.
        assert stored_event.payload.code_changes == [change_id]
        assert isinstance(stored_event.payload.code_changes[0], uuid.UUID)

    async def test_blob_sha_is_not_blank_padded(
        self, engine: AsyncEngine, seed: Seed, blob: BlobFactory
    ):
        """CHAR(64) would pad a 40-hex SHA-1 with 24 spaces and break equality."""
        execution_id = await start_run(engine, seed, repos=[seed.api_repo_id])
        sha = blob()
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
                    path="app/new.py",
                    after_blob=sha,
                )
            )
            await session.commit()

        async with AsyncSession(engine) as session:
            found = (
                await session.exec(
                    select(CodeChange).where(CodeChange.after_blob == sha)
                )
            ).one()
        assert found.after_blob == sha
        assert found.after_blob is not None and len(found.after_blob) == 40

    async def test_diff_is_optional(
        self, engine: AsyncEngine, seed: Seed, blob: BlobFactory
    ):
        """Git is the source of truth; the stored hunks are a rendering cache."""
        execution_id = await start_run(engine, seed, repos=[seed.api_repo_id])
        async with AsyncSession(engine) as session:
            event = await append(
                session,
                execution_id,
                event_type=ExecutionEventType.CODE_CHANGE,
                payload=ExecutionEventPayload(),
            )
            cached = CodeChange(
                execution_id=execution_id,
                event_id=event.id,
                repo_id=seed.api_repo_id,
                seq=0,
                change_type=ChangeType.CREATED,
                path="app/cached.py",
                after_blob=blob(),
                diff=[
                    CodeDiffApplied(old="", new="verify()", line_start=10, line_end=10)
                ],
            )
            omitted = CodeChange(
                execution_id=execution_id,
                event_id=event.id,
                repo_id=seed.api_repo_id,
                seq=1,
                change_type=ChangeType.CREATED,
                path="app/huge.py",
                after_blob=blob(),
            )
            session.add_all([cached, omitted])
            ids = (cached.id, omitted.id)
            await session.commit()

        async with AsyncSession(engine) as session:
            with_diff = await session.get(CodeChange, ids[0])
            without = await session.get(CodeChange, ids[1])
        assert with_diff is not None and with_diff.diff is not None
        assert with_diff.diff[0].new == "verify()"
        # Reads back as the model, not a raw dict.
        assert isinstance(with_diff.diff[0], CodeDiffApplied)
        assert without is not None and without.diff is None

    async def test_several_unordered_changes_may_share_one_event(
        self, engine: AsyncEngine, seed: Seed, blob: BlobFactory
    ):
        """seq is optional: an agent batching parallel tool calls into one event
        can leave it NULL, and NULLs do not collide under the unique index."""
        execution_id = await start_run(engine, seed, repos=[seed.api_repo_id])
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
                        change_type=ChangeType.CREATED,
                        path=f"app/mod_{n}.py",
                        after_blob=blob(),
                    )
                    for n in range(3)
                ]
            )
            await session.commit()

        async with AsyncSession(engine) as session:
            count = (
                await session.exec(select(sa.func.count()).select_from(CodeChange))
            ).one()
        assert count == 3

    async def test_a_revert_is_a_new_change_not_an_edit(
        self, engine: AsyncEngine, seed: Seed, blob: BlobFactory
    ):
        execution_id = await start_run(engine, seed, repos=[seed.api_repo_id])
        async with AsyncSession(engine) as session:
            event = await append(
                session,
                execution_id,
                event_type=ExecutionEventType.CODE_CHANGE,
                payload=ExecutionEventPayload(),
            )
            original = CodeChange(
                execution_id=execution_id,
                event_id=event.id,
                repo_id=seed.api_repo_id,
                seq=0,
                change_type=ChangeType.CREATED,
                path="app/oops.py",
                after_blob=blob(),
            )
            session.add(original)
            await session.flush()
            session.add(
                CodeChange(
                    execution_id=execution_id,
                    event_id=event.id,
                    repo_id=seed.api_repo_id,
                    seq=1,
                    change_type=ChangeType.DELETED,
                    path="app/oops.py",
                    before_blob=original.after_blob,
                    reverts_change_id=original.id,
                )
            )
            original_id = original.id
            await session.commit()

        async with AsyncSession(engine) as session:
            rows = (
                await session.exec(
                    select(CodeChange)
                    .where(CodeChange.path == "app/oops.py")
                    .order_by(sa.column("seq"))
                )
            ).all()
        # Two rows, and the original is untouched.
        assert [row.change_type for row in rows] == [
            ChangeType.CREATED,
            ChangeType.DELETED,
        ]
        assert rows[0].change_type is ChangeType.CREATED
        assert rows[0].reverts_change_id is None
        assert rows[1].reverts_change_id == original_id


class TestFileHistoryQuery:
    """The query the whole design exists to serve: open a file, see what the
    agent did to it and why."""

    async def test_history_of_a_file_carries_the_reasoning(
        self, engine: AsyncEngine, seed: Seed, blob: BlobFactory
    ):
        execution_id = await start_run(engine, seed, repos=[seed.api_repo_id])
        reasons = ["extracted the token parser", "jwt verify was missing"]
        async with AsyncSession(engine) as session:
            for reason in reasons:
                event = await append(
                    session,
                    execution_id,
                    event_type=ExecutionEventType.CODE_CHANGE,
                    cerebral_commit_sha=blob(),
                    payload=ExecutionEventPayload(reasoning=reason),
                )
                session.add(
                    CodeChange(
                        execution_id=execution_id,
                        event_id=event.id,
                        repo_id=seed.api_repo_id,
                        change_type=ChangeType.MODIFIED,
                        path="app/auth.py",
                        before_blob=blob(),
                        after_blob=blob(),
                    )
                )
                await session.flush()
            await session.commit()

        async with AsyncSession(engine) as session:
            connection = await session.connection()
            rows = (
                await connection.execute(
                    sa.text("""
                        SELECT cc.path,
                               cc.change_type,
                               cc.before_blob,
                               cc.after_blob,
                               ev.payload->>'reasoning' AS why,
                               ev.cerebral_commit_sha,
                               ev.seq
                        FROM code_changes cc
                        JOIN execution_events ev ON ev.id = cc.event_id
                        WHERE cc.repo_id = :repo AND cc.path = :path
                        ORDER BY ev.seq
                    """).bindparams(repo=seed.api_repo_id, path="app/auth.py")
                )
            ).all()

        assert [row.why for row in rows] == reasons
        # Every row carries the git coordinates needed to render the diff
        # locally, so the payload never has to store the file content.
        assert all(row.before_blob and row.after_blob for row in rows)
        assert all(row.cerebral_commit_sha for row in rows)

    async def test_same_path_in_two_repos_stays_separate(
        self, engine: AsyncEngine, seed: Seed, blob: BlobFactory
    ):
        """An unscoped index on path alone would mix these together."""
        execution_id = await start_run(
            engine, seed, repos=[seed.api_repo_id, seed.web_repo_id]
        )
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
                        repo_id=repo_id,
                        seq=n,
                        change_type=ChangeType.MODIFIED,
                        path="src/main.py",
                        before_blob=blob(),
                        after_blob=blob(),
                    )
                    for n, repo_id in enumerate((seed.api_repo_id, seed.web_repo_id))
                ]
            )
            await session.commit()

        async with AsyncSession(engine) as session:
            api_rows = (
                await session.exec(
                    select(CodeChange).where(
                        CodeChange.repo_id == seed.api_repo_id,
                        CodeChange.path == "src/main.py",
                    )
                )
            ).all()
        assert len(api_rows) == 1
        assert api_rows[0].repo_id == seed.api_repo_id


class TestLanding:
    async def test_run_squashes_from_cerebral_onto_the_default_branch(
        self, engine: AsyncEngine, seed: Seed, blob: BlobFactory
    ):
        execution_id = await start_run(engine, seed, repos=[seed.api_repo_id])
        head, merge, landed = blob(), blob(), blob()

        async with AsyncSession(engine) as session:
            link = await session.get(
                ExecutionRepoLink, (execution_id, seed.api_repo_id)
            )
            assert link is not None
            link.head_commit_sha = head  # tip of refs/cerebral/...
            link.landed_branch = "main"
            link.landed_commit_shas = [landed]
            link.merge_commit_sha = merge
            link.landed_at = utcnow()
            session.add(link)

            execution = await session.get(Execution, execution_id)
            assert execution is not None
            execution.status = ExecutionStatus.SUCCEEDED
            execution.finished_at = utcnow()
            execution.input_tokens = 12_400
            execution.output_tokens = 3_100
            execution.cache_read_tokens = 9_000
            execution.cost_usd = Decimal("0.184500")
            session.add(execution)
            await session.commit()

        async with AsyncSession(engine) as session:
            link = await session.get(
                ExecutionRepoLink, (execution_id, seed.api_repo_id)
            )
            execution = await session.get(Execution, execution_id)
        assert link is not None and execution is not None
        assert link.landed_commit_shas == [landed]
        assert link.merge_commit_sha == merge
        assert link.base_commit_sha != link.head_commit_sha
        # `is`, not `==`: proves the column round-trips to the enum member
        # rather than the bare string a plain String column returns.
        assert execution.status is ExecutionStatus.SUCCEEDED
        assert execution.cost_usd == Decimal("0.184500")

    async def test_landing_marks_the_changes_with_their_default_branch_commit(
        self, engine: AsyncEngine, seed: Seed, blob: BlobFactory
    ):
        execution_id = await start_run(engine, seed, repos=[seed.api_repo_id])
        landed = blob()
        async with AsyncSession(engine) as session:
            event = await append(
                session,
                execution_id,
                event_type=ExecutionEventType.CODE_CHANGE,
                cerebral_commit_sha=blob(),
                payload=ExecutionEventPayload(),
            )
            session.add(
                CodeChange(
                    execution_id=execution_id,
                    event_id=event.id,
                    repo_id=seed.api_repo_id,
                    change_type=ChangeType.CREATED,
                    path="app/jwt.py",
                    after_blob=blob(),
                )
            )
            await session.commit()

        async with AsyncSession(engine) as session:
            connection = await session.connection()
            await connection.execute(
                sa.text(
                    "UPDATE code_changes SET landed_commit_sha = :sha"
                    " WHERE execution_id = :ex"
                ).bindparams(sha=landed, ex=execution_id)
            )
            await session.commit()

        async with AsyncSession(engine) as session:
            change = (await session.exec(select(CodeChange))).one()
        # Both coordinates survive: the cerebral-side commit through the event,
        # the default-namespace commit on the change itself.
        assert change.landed_commit_sha == landed
        async with AsyncSession(engine) as session:
            event = await session.get(ExecutionEvent, change.event_id)
        assert event is not None and event.cerebral_commit_sha is not None
        assert event.cerebral_commit_sha != change.landed_commit_sha


class TestExecutionAccounting:
    async def test_usage_and_cost_are_recorded(self, engine: AsyncEngine, seed: Seed):
        execution_id = await start_run(engine, seed, repos=[])
        async with AsyncSession(engine) as session:
            execution = await session.get(Execution, execution_id)
            assert execution is not None
            assert (execution.input_tokens, execution.output_tokens) == (0, 0)
            assert execution.cost_usd is None
            assert execution.model == "claude-opus-5"
            assert execution.provider == "anthropic"

    async def test_failed_run_carries_structured_error(
        self, engine: AsyncEngine, seed: Seed
    ):
        from app.repo.execution import ExecutionError

        execution_id = await start_run(engine, seed, repos=[])
        async with AsyncSession(engine) as session:
            execution = await session.get(Execution, execution_id)
            assert execution is not None
            execution.status = ExecutionStatus.FAILED
            execution.finished_at = utcnow()
            execution.error = ExecutionError(
                code="tool_timeout",
                message="Read timed out after 30s",
                retryable=True,
                details={"tool": "Read", "seconds": 30},
            )
            session.add(execution)
            await session.commit()

        async with AsyncSession(engine) as session:
            execution = await session.get(Execution, execution_id)
        assert execution is not None and execution.error is not None
        # The API can branch on this without parsing prose.
        assert execution.error.retryable is True
        assert execution.error.code == "tool_timeout"
        assert execution.error.details["seconds"] == 30


class TestOwnership:
    async def test_executions_are_listable_by_owner_without_joins(
        self, engine: AsyncEngine, seed: Seed
    ):
        await start_run(engine, seed, repos=[])
        async with AsyncSession(engine) as session:
            mine = (
                await session.exec(
                    select(Execution).where(Execution.created_by == seed.user_id)
                )
            ).all()
            theirs = (
                await session.exec(
                    select(Execution).where(Execution.created_by == seed.other_user_id)
                )
            ).all()
        assert len(mine) == 1 and theirs == []


class TestAppendOnly:
    async def test_history_tables_have_no_updated_at(self) -> None:
        """An audit row is a fact. There is nothing to update, so the column
        that would invite an UPDATE is deliberately absent."""
        assert "updated_at" not in ExecutionEvent.model_fields
        assert "updated_at" not in CodeChange.model_fields
        assert "created_at" in ExecutionEvent.model_fields
        # Executions do carry mutable state, so they keep theirs.
        assert "updated_at" in Execution.model_fields
