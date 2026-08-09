"""Event ingest — the bot's hot path.

The two properties that matter here are the ones the channel forces. An append
must be safe to retry, because a bot whose response was lost will send the same
thing again and the transcript must not double. And an event and the code
changes it explains must land together or not at all, because reasoning
attached to edits nobody can find is worse than no record at all.
"""

import asyncio
import uuid
from typing import Any

import httpx
import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.security import create_access_token
from app.repo.api_keys import ApiKeyScope, ApiKeyStore
from app.repo.execution import (
    ActorType,
    CodeChange,
    EventStore,
    Execution,
    ExecutionEvent,
    ExecutionEventCreate,
    ExecutionEventType,
)

from .conftest import API, BlobFactory, Seed
from .test_executions import BASE, running_run


def change(seed: Seed, blob: BlobFactory, **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "repo_id": str(seed.api_repo_id),
        "path": "app/auth.py",
        "change_type": "modified",
        "before_blob": blob(),
        "after_blob": blob(),
        "lines_added": 12,
        "lines_deleted": 3,
    }
    body.update(overrides)
    return body


def event(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "event_type": "reasoning",
        "actor_type": "agent",
        "payload": {"reasoning": "the router never verified the signature"},
    }
    body.update(overrides)
    return body


async def with_repo(bot_client: httpx.AsyncClient, seed: Seed) -> str:
    """A running execution attached to the api repo."""
    execution = await running_run(
        bot_client,
        seed,
        repos=[{"repo_id": str(seed.api_repo_id), "base_commit_sha": BASE}],
    )
    return execution["id"]


async def post(
    bot_client: httpx.AsyncClient, execution_id: str, **body: Any
) -> httpx.Response:
    return await bot_client.post(
        f"{API}/executions/{execution_id}/events", json=event(**body)
    )


class TestAppend:
    async def test_an_event_lands_with_its_seq(
        self, bot_client: httpx.AsyncClient, seed: Seed
    ):
        execution_id = await with_repo(bot_client, seed)
        response = await post(bot_client, execution_id)
        assert response.status_code == 201
        appended = response.json()["data"]
        assert appended["seq"] == 1
        assert appended["execution_id"] == execution_id
        assert appended["payload"]["reasoning"].startswith("the router")
        assert appended["code_changes"] == []

    async def test_seq_is_allocated_densely_from_one(
        self, bot_client: httpx.AsyncClient, seed: Seed
    ):
        execution_id = await with_repo(bot_client, seed)
        seqs = [
            (await post(bot_client, execution_id)).json()["data"]["seq"]
            for _ in range(4)
        ]
        assert seqs == [1, 2, 3, 4]

    async def test_the_execution_tracks_the_high_water_mark(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        """A client tailing the log can tell it is behind without fetching."""
        execution_id = await with_repo(bot_client, seed)
        for _ in range(3):
            await post(bot_client, execution_id)
        execution = (
            await user_client.get(f"{API}/executions/{execution_id}")
        ).json()["data"]
        assert execution["last_event_seq"] == 3

    async def test_a_user_message_is_attributed_to_the_owner(
        self, bot_client: httpx.AsyncClient, seed: Seed
    ):
        """Both directions are chat_message; only actor_type tells them apart,
        and the user id comes from the credential rather than the body."""
        execution_id = await with_repo(bot_client, seed)
        response = await post(
            bot_client,
            execution_id,
            event_type="chat_message",
            actor_type="user",
            payload={"data": {"text": "add jwt auth"}},
        )
        assert response.status_code == 201
        assert response.json()["data"]["actor_user_id"] == str(seed.user_id)

    async def test_an_agent_message_carries_no_user(
        self, bot_client: httpx.AsyncClient, seed: Seed
    ):
        execution_id = await with_repo(bot_client, seed)
        response = await post(
            bot_client, execution_id, event_type="chat_message", actor_type="agent"
        )
        assert response.json()["data"]["actor_user_id"] is None

    async def test_a_tool_result_links_back_to_its_call(
        self, bot_client: httpx.AsyncClient, seed: Seed
    ):
        execution_id = await with_repo(bot_client, seed)
        call = (
            await post(bot_client, execution_id, event_type="tool_call")
        ).json()["data"]
        result = await post(
            bot_client,
            execution_id,
            event_type="tool_result",
            parent_event_id=call["id"],
        )
        assert result.status_code == 201
        assert result.json()["data"]["parent_event_id"] == call["id"]

    async def test_a_parent_from_another_run_is_refused(
        self, bot_client: httpx.AsyncClient, seed: Seed
    ):
        """Otherwise the transcript renders a reply to a question nobody asked."""
        first = await with_repo(bot_client, seed)
        second = await with_repo(bot_client, seed)
        foreign = (await post(bot_client, first, event_type="tool_call")).json()["data"]

        response = await post(
            bot_client, second, event_type="tool_result", parent_event_id=foreign["id"]
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "unknown_parent_event"

    async def test_appending_to_an_unknown_run_is_not_found(
        self, bot_client: httpx.AsyncClient
    ):
        response = await post(bot_client, str(uuid.uuid4()))
        assert response.status_code == 404

    async def test_a_person_cannot_write_to_the_transcript(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        """Every event has to be attributable to a credential issued for an agent."""
        execution_id = await with_repo(bot_client, seed)
        response = await user_client.post(
            f"{API}/executions/{execution_id}/events", json=event()
        )
        assert response.status_code == 401

    async def test_a_read_only_key_cannot_append(
        self,
        bot_client: httpx.AsyncClient,
        client: httpx.AsyncClient,
        engine: AsyncEngine,
        seed: Seed,
    ):
        execution_id = await with_repo(bot_client, seed)
        key, _ = await ApiKeyStore(engine).create(
            user_id=seed.user_id,
            name="read only",
            agent_id=seed.agent_id,
            scopes=[ApiKeyScope.EXECUTIONS_READ],
        )
        response = await client.post(
            f"{API}/executions/{execution_id}/events",
            json=event(),
            headers={"X-API-Key": key},
        )
        assert response.status_code == 403

    @pytest.mark.parametrize("method", ["put", "patch", "delete"])
    async def test_the_transcript_cannot_be_rewritten(
        self, bot_client: httpx.AsyncClient, seed: Seed, method: str
    ):
        """A correction is a new event, never an edit of an old one."""
        execution_id = await with_repo(bot_client, seed)
        appended = (await post(bot_client, execution_id)).json()["data"]
        for path in (
            f"{API}/executions/{execution_id}/events",
            f"{API}/executions/{execution_id}/events/{appended['id']}",
        ):
            response = await getattr(bot_client, method)(path)
            assert response.status_code == 405, path


class TestIdempotency:
    async def test_a_replay_returns_the_original_event(
        self, bot_client: httpx.AsyncClient, seed: Seed
    ):
        """201 the first time, 200 and the same event the second."""
        execution_id = await with_repo(bot_client, seed)
        body = event(client_event_id="bot-run7-0042")

        first = await bot_client.post(
            f"{API}/executions/{execution_id}/events", json=body
        )
        second = await bot_client.post(
            f"{API}/executions/{execution_id}/events", json=body
        )

        assert (first.status_code, second.status_code) == (201, 200)
        assert first.json()["data"]["id"] == second.json()["data"]["id"]
        assert first.json()["data"]["seq"] == second.json()["data"]["seq"]

    async def test_a_replay_does_not_double_the_transcript(
        self, bot_client: httpx.AsyncClient, engine: AsyncEngine, seed: Seed
    ):
        execution_id = await with_repo(bot_client, seed)
        body = event(client_event_id="bot-run7-0042")
        for _ in range(3):
            await bot_client.post(f"{API}/executions/{execution_id}/events", json=body)

        async with AsyncSession(engine) as session:
            count = (
                await session.exec(select(sa.func.count()).select_from(ExecutionEvent))
            ).one()
        assert count == 1

    async def test_a_replay_does_not_burn_a_seq(
        self, bot_client: httpx.AsyncClient, seed: Seed
    ):
        """The retry that actually happens costs one read, not a gap in the log."""
        execution_id = await with_repo(bot_client, seed)
        body = event(client_event_id="evt-1")
        await bot_client.post(f"{API}/executions/{execution_id}/events", json=body)
        await bot_client.post(f"{API}/executions/{execution_id}/events", json=body)

        following = await post(bot_client, execution_id, client_event_id="evt-2")
        assert following.json()["data"]["seq"] == 2

    async def test_a_replay_does_not_duplicate_the_code_changes(
        self, bot_client: httpx.AsyncClient, engine: AsyncEngine, seed: Seed,
        blob: BlobFactory,
    ):
        execution_id = await with_repo(bot_client, seed)
        body = event(
            client_event_id="evt-1",
            event_type="code_change",
            code_changes=[change(seed, blob)],
        )
        first = await bot_client.post(
            f"{API}/executions/{execution_id}/events", json=body
        )
        second = await bot_client.post(
            f"{API}/executions/{execution_id}/events", json=body
        )

        assert second.status_code == 200
        # The replay answers with the rows the first attempt wrote, not new ones.
        assert [c["id"] for c in second.json()["data"]["code_changes"]] == [
            c["id"] for c in first.json()["data"]["code_changes"]
        ]
        async with AsyncSession(engine) as session:
            count = (
                await session.exec(select(sa.func.count()).select_from(CodeChange))
            ).one()
        assert count == 1

    async def test_two_appends_racing_on_one_key_produce_one_event(
        self, bot_client: httpx.AsyncClient, engine: AsyncEngine, seed: Seed
    ):
        """The replay check can lose its race, so ON CONFLICT is the real
        guarantee — the check in front of it is only there to save a seq.

        Both callers allocate a seq under the execution's row lock, so the
        second reaches its insert with the first already committed and is
        absorbed. Whichever way the interleaving falls, one event exists.

        Driven at the store rather than over HTTP: the invariant lives here,
        and two concurrent requests would drag the whole dependency chain --
        key authentication, its last_used write -- into the race for nothing.
        """
        execution_id = uuid.UUID(await with_repo(bot_client, seed))
        store = EventStore(engine)
        body = ExecutionEventCreate(
            client_event_id="evt-1", event_type=ExecutionEventType.REASONING
        )

        # return_exceptions so both coroutines are always awaited. Bare gather
        # abandons the sibling the moment one raises, and an abandoned request
        # keeps its transaction open into the next test, where it races the
        # TRUNCATE that is supposed to be giving that test a clean database.
        results = await asyncio.gather(
            store.append(execution_id, seed.user_id, body),
            store.append(execution_id, seed.user_id, body),
            return_exceptions=True,
        )
        assert all(not isinstance(result, BaseException) for result in results), results

        appended = [result for result in results if result is not None]
        assert len(appended) == 2
        # Exactly one of them created it; both answer with the same event.
        assert sorted(created for *_, created in appended) == [False, True]  # pyright: ignore[reportGeneralTypeIssues]
        assert len({event.id for event, _, _ in appended}) == 1  # pyright: ignore[reportGeneralTypeIssues]

        async with AsyncSession(engine) as session:
            count = (
                await session.exec(select(sa.func.count()).select_from(ExecutionEvent))
            ).one()
            execution = await session.get(Execution, execution_id)
        assert count == 1
        # The loser rolled back, so it returned the seq it had taken.
        assert execution is not None and execution.last_event_seq == 1

    async def test_an_unkeyed_append_is_never_deduplicated(
        self, bot_client: httpx.AsyncClient, seed: Seed
    ):
        """Two identical events with no client_event_id are two real events —
        an agent may genuinely say the same thing twice."""
        execution_id = await with_repo(bot_client, seed)
        first = await post(bot_client, execution_id)
        second = await post(bot_client, execution_id)
        assert (first.status_code, second.status_code) == (201, 201)
        assert first.json()["data"]["id"] != second.json()["data"]["id"]

    async def test_the_key_is_scoped_to_one_execution(
        self, bot_client: httpx.AsyncClient, seed: Seed
    ):
        """Agents restart their counters, so the same id in another run is a
        different event."""
        first = await with_repo(bot_client, seed)
        second = await with_repo(bot_client, seed)
        a = await post(bot_client, first, client_event_id="evt-1")
        b = await post(bot_client, second, client_event_id="evt-1")
        assert (a.status_code, b.status_code) == (201, 201)
        assert a.json()["data"]["id"] != b.json()["data"]["id"]


class TestCodeChanges:
    async def test_changes_ride_along_with_their_event(
        self, bot_client: httpx.AsyncClient, seed: Seed, blob: BlobFactory
    ):
        execution_id = await with_repo(bot_client, seed)
        response = await post(
            bot_client,
            execution_id,
            event_type="code_change",
            cerebral_commit_sha=blob(),
            payload={"reasoning": "jwt verify was missing"},
            code_changes=[
                change(seed, blob, path="app/auth.py"),
                change(seed, blob, path="app/jwt.py", change_type="created",
                       before_blob=None),
            ],
        )
        assert response.status_code == 201
        appended = response.json()["data"]
        assert [c["path"] for c in appended["code_changes"]] == [
            "app/auth.py",
            "app/jwt.py",
        ]
        # Positional: the order sent is the order recorded.
        assert [c["seq"] for c in appended["code_changes"]] == [0, 1]
        # Pointers into git, never content.
        assert all(c["after_blob"] for c in appended["code_changes"])

    async def test_the_payload_index_matches_the_rows(
        self, bot_client: httpx.AsyncClient, seed: Seed, blob: BlobFactory
    ):
        """payload.code_changes is denormalised, so it has to be backfilled in
        the same transaction that wrote the rows."""
        execution_id = await with_repo(bot_client, seed)
        appended = (
            await post(
                bot_client,
                execution_id,
                event_type="code_change",
                code_changes=[
                    change(seed, blob, path="a.py"),
                    change(seed, blob, path="b.py"),
                ],
            )
        ).json()["data"]
        assert appended["payload"]["code_changes"] == [
            c["id"] for c in appended["code_changes"]
        ]

    async def test_a_change_in_an_unattached_repo_is_refused(
        self, bot_client: httpx.AsyncClient, seed: Seed, blob: BlobFactory
    ):
        """docs_repo is never attached to any run."""
        execution_id = await with_repo(bot_client, seed)
        response = await post(
            bot_client,
            execution_id,
            event_type="code_change",
            code_changes=[change(seed, blob, repo_id=str(seed.docs_repo_id))],
        )
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "repo_not_attached"

    @pytest.mark.parametrize(
        ("label", "overrides"),
        [
            ("created with a before blob", {"change_type": "created"}),
            ("deleted with an after blob", {"change_type": "deleted"}),
            ("rename with no previous_path", {"change_type": "renamed"}),
        ],
    )
    async def test_a_change_git_could_not_produce_is_refused(
        self, bot_client: httpx.AsyncClient, seed: Seed, blob: BlobFactory,
        label: str, overrides: dict[str, Any],
    ):
        """A created file has no before, a deleted file has no after, and a
        rename has to say what it was called. The database is the authority;
        this only checks its refusal arrives as something a caller can read."""
        execution_id = await with_repo(bot_client, seed)
        response = await post(
            bot_client,
            execution_id,
            event_type="code_change",
            code_changes=[change(seed, blob, **overrides)],
        )
        assert response.status_code == 422, label
        assert response.json()["error"]["code"] == "invalid_code_change"

    async def test_a_rejected_change_takes_its_event_with_it(
        self, bot_client: httpx.AsyncClient, engine: AsyncEngine, seed: Seed,
        blob: BlobFactory,
    ):
        """The whole point of one transaction: no orphan event claiming to
        explain edits that were never written, and no seq burned either."""
        execution_id = await with_repo(bot_client, seed)
        response = await post(
            bot_client,
            execution_id,
            event_type="code_change",
            code_changes=[change(seed, blob, change_type="created")],
        )
        assert response.status_code == 422

        async with AsyncSession(engine) as session:
            events = (
                await session.exec(select(sa.func.count()).select_from(ExecutionEvent))
            ).one()
            changes = (
                await session.exec(select(sa.func.count()).select_from(CodeChange))
            ).one()
            execution = await session.get(Execution, uuid.UUID(execution_id))
        assert (events, changes) == (0, 0)
        assert execution is not None and execution.last_event_seq == 0

        # And the next real append still starts at 1.
        following = await post(bot_client, execution_id)
        assert following.json()["data"]["seq"] == 1

    async def test_a_change_may_cache_its_hunks(
        self, bot_client: httpx.AsyncClient, seed: Seed, blob: BlobFactory
    ):
        """Optional: git is the source of truth, this is a rendering cache."""
        execution_id = await with_repo(bot_client, seed)
        appended = (
            await post(
                bot_client,
                execution_id,
                event_type="code_change",
                code_changes=[
                    change(
                        seed,
                        blob,
                        diff=[
                            {
                                "old": "",
                                "new": "verify()",
                                "line_start": 10,
                                "line_end": 10,
                            }
                        ],
                    )
                ],
            )
        ).json()["data"]
        assert appended["code_changes"][0]["diff"][0]["new"] == "verify()"

    async def test_a_change_landed_nowhere_yet(
        self, bot_client: httpx.AsyncClient, seed: Seed, blob: BlobFactory
    ):
        execution_id = await with_repo(bot_client, seed)
        appended = (
            await post(
                bot_client,
                execution_id,
                event_type="code_change",
                code_changes=[change(seed, blob)],
            )
        ).json()["data"]
        assert appended["code_changes"][0]["landed_commit_sha"] is None


class TestReadingTheTranscript:
    async def test_the_cursor_survives_a_concurrent_append(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        """This is why the cursor is seq and not an offset: the log is written
        while it is being read, and an offset would skip or repeat."""
        execution_id = await with_repo(bot_client, seed)
        for _ in range(4):
            await post(bot_client, execution_id)

        first = (
            await user_client.get(f"{API}/executions/{execution_id}/events?limit=2")
        ).json()["data"]
        assert [item["seq"] for item in first["items"]] == [1, 2]
        assert first["has_more"] is True
        assert first["next_after_seq"] == 2

        # An agent appends between the two reads.
        await post(bot_client, execution_id)

        second = (
            await user_client.get(
                f"{API}/executions/{execution_id}/events"
                f"?limit=2&after_seq={first['next_after_seq']}"
            )
        ).json()["data"]
        assert [item["seq"] for item in second["items"]] == [3, 4]

        third = (
            await user_client.get(
                f"{API}/executions/{execution_id}/events"
                f"?limit=2&after_seq={second['next_after_seq']}"
            )
        ).json()["data"]
        # The event that arrived mid-read shows up once, in order, at the end.
        assert [item["seq"] for item in third["items"]] == [5]
        assert third["has_more"] is False

    async def test_catching_up_returns_an_empty_page(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        """Polling the tail is the same call with the cursor you already have."""
        execution_id = await with_repo(bot_client, seed)
        await post(bot_client, execution_id)
        page = (
            await user_client.get(
                f"{API}/executions/{execution_id}/events?after_seq=1"
            )
        ).json()["data"]
        assert page["items"] == []
        assert page["has_more"] is False
        assert page["next_after_seq"] is None

    async def test_the_transcript_reads_oldest_first(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        execution_id = await with_repo(bot_client, seed)
        for _ in range(5):
            await post(bot_client, execution_id)
        page = (
            await user_client.get(f"{API}/executions/{execution_id}/events")
        ).json()["data"]
        assert [item["seq"] for item in page["items"]] == [1, 2, 3, 4, 5]

    async def test_filtering_by_event_and_actor(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        execution_id = await with_repo(bot_client, seed)
        await post(bot_client, execution_id, event_type="reasoning")
        await post(
            bot_client,
            execution_id,
            event_type="chat_message",
            actor_type="user",
            payload={"data": {"text": "add jwt auth"}},
        )
        await post(bot_client, execution_id, event_type="chat_message")

        async def seqs(query: str) -> list[int]:
            page = (
                await user_client.get(
                    f"{API}/executions/{execution_id}/events?{query}"
                )
            ).json()["data"]
            return [item["seq"] for item in page["items"]]

        assert await seqs("event_type=chat_message") == [2, 3]
        assert await seqs("actor_type=user") == [2]
        assert await seqs("event_type=chat_message&actor_type=agent") == [3]
        assert await seqs("event_type=decision") == []

    async def test_one_event_expands_its_changes(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient,
        seed: Seed, blob: BlobFactory,
    ):
        execution_id = await with_repo(bot_client, seed)
        appended = (
            await post(
                bot_client,
                execution_id,
                event_type="code_change",
                payload={"reasoning": "jwt verify was missing"},
                code_changes=[change(seed, blob)],
            )
        ).json()["data"]

        response = await user_client.get(
            f"{API}/executions/{execution_id}/events/{appended['id']}"
        )
        assert response.status_code == 200
        fetched = response.json()["data"]
        assert fetched["payload"]["reasoning"] == "jwt verify was missing"
        [only] = fetched["code_changes"]
        assert only["path"] == "app/auth.py"
        assert only["lines_added"] == 12

    async def test_an_event_of_another_run_is_not_found_here(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        first = await with_repo(bot_client, seed)
        second = await with_repo(bot_client, seed)
        appended = (await post(bot_client, first)).json()["data"]

        response = await user_client.get(
            f"{API}/executions/{second}/events/{appended['id']}"
        )
        assert response.status_code == 404

    async def test_unknown_event_is_not_found(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        execution_id = await with_repo(bot_client, seed)
        response = await user_client.get(
            f"{API}/executions/{execution_id}/events/{uuid.uuid4()}"
        )
        assert response.status_code == 404

    async def test_the_bot_can_read_back_what_it_wrote(
        self, bot_client: httpx.AsyncClient, seed: Seed
    ):
        execution_id = await with_repo(bot_client, seed)
        await post(bot_client, execution_id)
        response = await bot_client.get(f"{API}/executions/{execution_id}/events")
        assert response.status_code == 200
        assert len(response.json()["data"]["items"]) == 1

    async def test_another_users_transcript_is_forbidden(
        self, bot_client: httpx.AsyncClient, client: httpx.AsyncClient, seed: Seed
    ):
        execution_id = await with_repo(bot_client, seed)
        await post(bot_client, execution_id)
        other = {"Authorization": f"Bearer {create_access_token(seed.other_user_id)}"}
        response = await client.get(
            f"{API}/executions/{execution_id}/events", headers=other
        )
        assert response.status_code == 403

    async def test_an_anonymous_caller_gets_nothing(
        self, bot_client: httpx.AsyncClient, client: httpx.AsyncClient, seed: Seed
    ):
        execution_id = await with_repo(bot_client, seed)
        response = await client.get(f"{API}/executions/{execution_id}/events")
        assert response.status_code == 401


class TestRoundTrip:
    async def test_the_types_survive_postgres(
        self, bot_client: httpx.AsyncClient, engine: AsyncEngine, seed: Seed,
        blob: BlobFactory,
    ):
        """`is`, not `==`: proves the columns come back as enum members rather
        than the bare strings a plain String column would hand over."""
        execution_id = await with_repo(bot_client, seed)
        await post(
            bot_client,
            execution_id,
            event_type="code_change",
            actor_type="agent",
            code_changes=[change(seed, blob)],
        )
        async with AsyncSession(engine) as session:
            stored = (await session.exec(select(ExecutionEvent))).one()
            stored_change = (await session.exec(select(CodeChange))).one()

        assert stored.event_type is ExecutionEventType.CODE_CHANGE
        assert stored.actor_type is ActorType.AGENT
        # The payload is a typed model in Python and JSONB in Postgres.
        assert stored.payload.code_changes == [stored_change.id]
        assert isinstance(stored.payload.code_changes[0], uuid.UUID)

    async def test_the_reasoning_is_still_queryable_as_json(
        self, bot_client: httpx.AsyncClient, engine: AsyncEngine, seed: Seed
    ):
        """Storing a model must not make the column opaque to SQL — the file
        history query reads through payload->>'reasoning'."""
        execution_id = await with_repo(bot_client, seed)
        await post(
            bot_client,
            execution_id,
            payload={"reasoning": "chose argon2 over bcrypt"},
        )
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
