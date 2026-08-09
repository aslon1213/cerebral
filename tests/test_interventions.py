"""Interventions — the agent asks, a person answers.

The asymmetry is the point and most of these tests are about it: an API key can
open a request and read one, and cannot resolve one under any circumstances. A
bot that could approve its own request for approval is not asking permission.

The other half is the denormalised ``executions.status``. It exists so list
endpoints need no join, which means every write here has to keep it honest —
including when several requests are open at once and only some are answered.
"""

import uuid
from typing import Any

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.security import create_access_token
from app.repo.api_keys import ApiKeyScope, ApiKeyStore
from app.repo.execution import (
    ActorType,
    ExecutionEvent,
    ExecutionEventType,
    ExecutionStatus,
    InterventionKind,
    InterventionStatus,
)

from .conftest import API, Seed
from .test_executions import running_run, start_run


async def ask(
    bot_client: httpx.AsyncClient,
    execution_id: str,
    kind: str = "approval",
    **overrides: Any,
) -> dict[str, Any]:
    """The agent raises a request and blocks."""
    body: dict[str, Any] = {
        "kind": kind,
        "request": {"question": "apply the patch?"},
        "reasoning": "this rewrites the auth middleware",
    }
    body.update(overrides)
    response = await bot_client.post(
        f"{API}/executions/{execution_id}/interventions", json=body
    )
    assert response.status_code == 201, response.text
    return response.json()["data"]


async def status_of(user_client: httpx.AsyncClient, execution_id: str) -> str:
    execution = (
        await user_client.get(f"{API}/executions/{execution_id}")
    ).json()["data"]
    return execution["status"]


class TestAsking:
    async def test_an_approval_parks_the_run(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        execution = await running_run(bot_client, seed)
        intervention = await ask(bot_client, execution["id"])

        assert intervention["status"] == InterventionStatus.PENDING
        assert intervention["kind"] == InterventionKind.APPROVAL
        assert intervention["resolved_at"] is None
        assert await status_of(user_client, execution["id"]) == (
            ExecutionStatus.WAITING_APPROVAL
        )

    async def test_a_question_parks_it_differently(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        execution = await running_run(bot_client, seed)
        await ask(bot_client, execution["id"], kind="input_required")
        assert await status_of(user_client, execution["id"]) == (
            ExecutionStatus.WAITING_INPUT
        )

    async def test_a_qa_review_waits_for_approval(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        """Both review kinds are approve/reject flows, so both park the same."""
        execution = await running_run(bot_client, seed)
        await ask(bot_client, execution["id"], kind="qa_review")
        assert await status_of(user_client, execution["id"]) == (
            ExecutionStatus.WAITING_APPROVAL
        )

    async def test_asking_shows_up_in_the_transcript(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        """The request renders inline in the history rather than as a detached
        side channel, so the intervention and its event link both ways."""
        execution = await running_run(bot_client, seed)
        intervention = await ask(bot_client, execution["id"])

        page = (
            await user_client.get(f"{API}/executions/{execution['id']}/events")
        ).json()["data"]
        [event] = page["items"]
        assert event["event_type"] == ExecutionEventType.INTERVENTION_REQUESTED
        assert event["id"] == intervention["event_id"]
        assert event["payload"]["reasoning"] == "this rewrites the auth middleware"
        assert event["payload"]["data"] == {"question": "apply the patch?"}

    async def test_several_may_be_open_at_once(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        """A real agent batches its tool approvals into one turn, so the second
        must not displace the first."""
        execution = await running_run(bot_client, seed)
        opened = [await ask(bot_client, execution["id"]) for _ in range(3)]

        assert len({item["id"] for item in opened}) == 3
        listed = (
            await user_client.get(
                f"{API}/executions/{execution['id']}/interventions?status=pending"
            )
        ).json()["data"]
        assert listed["total"] == 3

    @pytest.mark.parametrize("verb", ["complete", "cancel"])
    async def test_a_finished_run_cannot_be_interrupted(
        self, bot_client: httpx.AsyncClient, seed: Seed, verb: str
    ):
        execution = await running_run(bot_client, seed)
        await bot_client.post(f"{API}/executions/{execution['id']}/{verb}", json={})

        response = await bot_client.post(
            f"{API}/executions/{execution['id']}/interventions",
            json={"kind": "approval"},
        )
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "invalid_transition"

    async def test_a_run_that_never_started_cannot_be_interrupted(
        self, bot_client: httpx.AsyncClient, seed: Seed
    ):
        """Nothing to interrupt: the agent has not picked it up yet."""
        execution = await start_run(bot_client, seed)
        response = await bot_client.post(
            f"{API}/executions/{execution['id']}/interventions",
            json={"kind": "approval"},
        )
        assert response.status_code == 409

    async def test_a_person_cannot_raise_one(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        execution = await running_run(bot_client, seed)
        response = await user_client.post(
            f"{API}/executions/{execution['id']}/interventions",
            json={"kind": "approval"},
        )
        assert response.status_code == 401

    async def test_asking_on_an_unknown_run_is_not_found(
        self, bot_client: httpx.AsyncClient
    ):
        response = await bot_client.post(
            f"{API}/executions/{uuid.uuid4()}/interventions",
            json={"kind": "approval"},
        )
        assert response.status_code == 404


class TestTheInbox:
    async def test_it_spans_every_run(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        first = await running_run(bot_client, seed)
        second = await running_run(bot_client, seed)
        await ask(bot_client, first["id"])
        await ask(bot_client, second["id"], kind="input_required")

        inbox = (await user_client.get(f"{API}/interventions")).json()["data"]
        assert inbox["total"] == 2
        assert {item["execution_id"] for item in inbox["items"]} == {
            first["id"],
            second["id"],
        }

    async def test_it_is_oldest_first(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        """It is a queue: the agent blocked longest is losing the most time."""
        execution = await running_run(bot_client, seed)
        opened = [
            await ask(bot_client, execution["id"], request={"n": n}) for n in range(3)
        ]

        inbox = (await user_client.get(f"{API}/interventions")).json()["data"]
        assert [item["id"] for item in inbox["items"]] == [
            item["id"] for item in opened
        ]

    async def test_resolved_ones_leave_the_inbox(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        execution = await running_run(bot_client, seed)
        intervention = await ask(bot_client, execution["id"])
        await user_client.post(
            f"{API}/interventions/{intervention['id']}/approve", json={}
        )

        inbox = (await user_client.get(f"{API}/interventions")).json()["data"]
        assert inbox["total"] == 0

    async def test_it_is_scoped_to_the_owner(
        self, bot_client: httpx.AsyncClient, client: httpx.AsyncClient, seed: Seed
    ):
        execution = await running_run(bot_client, seed)
        await ask(bot_client, execution["id"])

        other = {"Authorization": f"Bearer {create_access_token(seed.other_user_id)}"}
        inbox = (await client.get(f"{API}/interventions", headers=other)).json()
        assert inbox["data"]["total"] == 0

    async def test_a_bot_cannot_read_the_inbox(
        self, bot_client: httpx.AsyncClient, seed: Seed
    ):
        """A human's queue takes a session token and nothing else."""
        execution = await running_run(bot_client, seed)
        await ask(bot_client, execution["id"])
        response = await bot_client.get(f"{API}/interventions")
        assert response.status_code == 401

    async def test_an_agent_can_still_see_what_it_asked(
        self, bot_client: httpx.AsyncClient, seed: Seed
    ):
        """An agent that restarts needs to find out whether it was answered."""
        execution = await running_run(bot_client, seed)
        await ask(bot_client, execution["id"])
        response = await bot_client.get(
            f"{API}/executions/{execution['id']}/interventions"
        )
        assert response.status_code == 200
        assert response.json()["data"]["total"] == 1


class TestResolving:
    async def test_approving_releases_the_run(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        execution = await running_run(bot_client, seed)
        intervention = await ask(bot_client, execution["id"])

        response = await user_client.post(
            f"{API}/interventions/{intervention['id']}/approve",
            json={"response": {"note": "looks right"}},
        )
        assert response.status_code == 200
        resolved = response.json()["data"]
        assert resolved["status"] == InterventionStatus.APPROVED
        assert resolved["resolved_by_user_id"] == str(seed.user_id)
        assert resolved["resolved_at"] is not None
        assert resolved["response"] == {"note": "looks right"}
        assert await status_of(user_client, execution["id"]) == ExecutionStatus.RUNNING

    async def test_rejecting_also_releases_the_run(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        """The agent carries on, told no — it is not blocked any more."""
        execution = await running_run(bot_client, seed)
        intervention = await ask(bot_client, execution["id"])

        response = await user_client.post(
            f"{API}/interventions/{intervention['id']}/reject", json={}
        )
        assert response.json()["data"]["status"] == InterventionStatus.REJECTED
        assert await status_of(user_client, execution["id"]) == ExecutionStatus.RUNNING

    async def test_answering_a_question(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        execution = await running_run(bot_client, seed)
        intervention = await ask(
            bot_client, execution["id"], kind="input_required"
        )

        response = await user_client.post(
            f"{API}/interventions/{intervention['id']}/answer",
            json={"response": {"answer": "use postgres"}},
        )
        assert response.status_code == 200
        resolved = response.json()["data"]
        assert resolved["status"] == InterventionStatus.ANSWERED
        assert resolved["response"] == {"answer": "use postgres"}

    async def test_an_answer_without_an_answer_is_rejected(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        execution = await running_run(bot_client, seed)
        intervention = await ask(bot_client, execution["id"], kind="input_required")
        response = await user_client.post(
            f"{API}/interventions/{intervention['id']}/answer", json={}
        )
        assert response.status_code == 422

    async def test_the_answer_joins_the_transcript(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        """So the history records not only what the agent decided but what a
        person told it to do."""
        execution = await running_run(bot_client, seed)
        intervention = await ask(bot_client, execution["id"])
        resolved = (
            await user_client.post(
                f"{API}/interventions/{intervention['id']}/approve",
                json={"reasoning": "the diff is small enough"},
            )
        ).json()["data"]

        page = (
            await user_client.get(
                f"{API}/executions/{execution['id']}/events?event_type=intervention_resolved"
            )
        ).json()["data"]
        [event] = page["items"]
        assert event["id"] == resolved["resolution_event_id"]
        assert event["actor_type"] == ActorType.USER
        assert event["actor_user_id"] == str(seed.user_id)
        assert event["payload"]["reasoning"] == "the diff is small enough"
        assert event["payload"]["data"]["status"] == InterventionStatus.APPROVED

    async def test_a_question_cannot_be_approved(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        execution = await running_run(bot_client, seed)
        intervention = await ask(bot_client, execution["id"], kind="input_required")

        response = await user_client.post(
            f"{API}/interventions/{intervention['id']}/approve", json={}
        )
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "intervention_kind_mismatch"

    @pytest.mark.parametrize("kind", ["approval", "qa_review"])
    async def test_an_approval_cannot_be_answered(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient,
        seed: Seed, kind: str,
    ):
        execution = await running_run(bot_client, seed)
        intervention = await ask(bot_client, execution["id"], kind=kind)

        response = await user_client.post(
            f"{API}/interventions/{intervention['id']}/answer",
            json={"response": {"answer": "yes"}},
        )
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "intervention_kind_mismatch"

    async def test_answering_twice_conflicts(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        """Two reviewers with the same inbox open is the normal case."""
        execution = await running_run(bot_client, seed)
        intervention = await ask(bot_client, execution["id"])

        first = await user_client.post(
            f"{API}/interventions/{intervention['id']}/approve", json={}
        )
        second = await user_client.post(
            f"{API}/interventions/{intervention['id']}/reject", json={}
        )
        assert first.status_code == 200
        assert second.status_code == 409
        assert second.json()["error"]["code"] == "intervention_already_resolved"
        # The first answer stands.
        assert "approved" in second.json()["error"]["message"]

    async def test_the_run_waits_until_the_last_one_is_answered(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        """waiting_* means "at least one open", not "exactly one"."""
        execution = await running_run(bot_client, seed)
        opened = [await ask(bot_client, execution["id"]) for _ in range(3)]

        for intervention in opened[:-1]:
            await user_client.post(
                f"{API}/interventions/{intervention['id']}/approve", json={}
            )
            assert await status_of(user_client, execution["id"]) == (
                ExecutionStatus.WAITING_APPROVAL
            )

        await user_client.post(
            f"{API}/interventions/{opened[-1]['id']}/approve", json={}
        )
        assert await status_of(user_client, execution["id"]) == ExecutionStatus.RUNNING

    async def test_the_status_follows_whatever_is_still_pending(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        """Answering the approval a run was blocked on can leave it blocked on
        a question asked later, so the status has to move rather than clear."""
        execution = await running_run(bot_client, seed)
        approval = await ask(bot_client, execution["id"], kind="approval")
        question = await ask(bot_client, execution["id"], kind="input_required")
        assert await status_of(user_client, execution["id"]) == (
            ExecutionStatus.WAITING_APPROVAL
        )

        await user_client.post(f"{API}/interventions/{approval['id']}/approve", json={})
        assert await status_of(user_client, execution["id"]) == (
            ExecutionStatus.WAITING_INPUT
        )

        await user_client.post(
            f"{API}/interventions/{question['id']}/answer",
            json={"response": {"answer": "postgres"}},
        )
        assert await status_of(user_client, execution["id"]) == ExecutionStatus.RUNNING

    async def test_answering_does_not_resurrect_a_finished_run(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        """A run cancelled while blocked stays cancelled, however late the
        answer arrives."""
        execution = await running_run(bot_client, seed)
        intervention = await ask(bot_client, execution["id"])
        await bot_client.post(f"{API}/executions/{execution['id']}/cancel")

        response = await user_client.post(
            f"{API}/interventions/{intervention['id']}/approve", json={}
        )
        assert response.status_code == 200
        assert await status_of(user_client, execution["id"]) == (
            ExecutionStatus.CANCELLED
        )

    async def test_resolving_is_one_transaction(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient,
        engine: AsyncEngine, seed: Seed,
    ):
        """A rejected resolution must leave nothing behind — no orphan event
        claiming a person answered something they did not."""
        execution = await running_run(bot_client, seed)
        intervention = await ask(bot_client, execution["id"], kind="input_required")

        before = await self._event_count(engine)
        response = await user_client.post(
            f"{API}/interventions/{intervention['id']}/approve", json={}
        )
        assert response.status_code == 409
        assert await self._event_count(engine) == before

    @staticmethod
    async def _event_count(engine: AsyncEngine) -> int:
        async with AsyncSession(engine) as session:
            events = (await session.exec(select(ExecutionEvent))).all()
        return len(events)


class TestOnlyPeopleAnswer:
    """The rule the whole design hangs on: a bot must never resolve its own
    request. OBSERVER_BOT_SCOPES omits any scope that would allow it, and these
    routes take no API key at all."""

    @pytest.mark.parametrize("verb", ["approve", "reject", "answer"])
    async def test_an_api_key_cannot_resolve_one(
        self, bot_client: httpx.AsyncClient, seed: Seed, verb: str
    ):
        execution = await running_run(bot_client, seed)
        intervention = await ask(bot_client, execution["id"])

        response = await bot_client.post(
            f"{API}/interventions/{intervention['id']}/{verb}",
            json={"response": {"answer": "yes"}},
        )
        assert response.status_code == 401
        # And it really is still pending.
        assert (
            await bot_client.get(
                f"{API}/executions/{execution['id']}/interventions?status=pending"
            )
        ).json()["data"]["total"] == 1

    async def test_not_even_a_key_with_every_scope(
        self,
        bot_client: httpx.AsyncClient,
        client: httpx.AsyncClient,
        engine: AsyncEngine,
        seed: Seed,
    ):
        """Scopes are not the mechanism here — the route simply takes no key."""
        execution = await running_run(bot_client, seed)
        intervention = await ask(bot_client, execution["id"])

        key, _ = await ApiKeyStore(engine).create(
            user_id=seed.user_id,
            name="over-privileged",
            agent_id=seed.agent_id,
            scopes=list(ApiKeyScope),
        )
        response = await client.post(
            f"{API}/interventions/{intervention['id']}/approve",
            json={},
            headers={"X-API-Key": key},
        )
        assert response.status_code == 401

    async def test_another_user_cannot_answer_it(
        self, bot_client: httpx.AsyncClient, client: httpx.AsyncClient, seed: Seed
    ):
        execution = await running_run(bot_client, seed)
        intervention = await ask(bot_client, execution["id"])

        other = {"Authorization": f"Bearer {create_access_token(seed.other_user_id)}"}
        response = await client.post(
            f"{API}/interventions/{intervention['id']}/approve",
            json={},
            headers=other,
        )
        assert response.status_code == 403

    async def test_an_anonymous_caller_cannot(
        self, bot_client: httpx.AsyncClient, client: httpx.AsyncClient, seed: Seed
    ):
        execution = await running_run(bot_client, seed)
        intervention = await ask(bot_client, execution["id"])
        response = await client.post(
            f"{API}/interventions/{intervention['id']}/approve", json={}
        )
        assert response.status_code == 401

    async def test_answering_an_unknown_one_is_not_found(
        self, user_client: httpx.AsyncClient
    ):
        response = await user_client.post(
            f"{API}/interventions/{uuid.uuid4()}/approve", json={}
        )
        assert response.status_code == 404
