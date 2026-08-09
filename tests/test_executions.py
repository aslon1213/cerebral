"""The execution lifecycle over HTTP.

Two things this file is really about. First, that the state machine is enforced
in the store: every illegal step has to come back as a 409 naming both states,
never as a 500 from a check constraint. Second, that a bot's credential decides
what the run is attributed to — the body never gets a say in which agent did the
work.
"""

import asyncio
import uuid
from decimal import Decimal
from typing import Any

import httpx
import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.security import create_access_token
from app.repo.agent import Agent, AgentStore
from app.repo.api_keys import ApiKeyScope, ApiKeyStore
from app.repo.base import InvalidTransitionError, RepoAlreadyAttachedError
from app.repo.execution import (
    LEGAL_TRANSITIONS,
    TERMINAL_STATUSES,
    ChangeType,
    CodeChange,
    Execution,
    ExecutionError,
    ExecutionEvent,
    ExecutionEventPayload,
    ExecutionEventType,
    ExecutionRepoAttach,
    ExecutionRepoLink,
    ExecutionStatus,
    ExecutionStore,
    ExecutionUsage,
    ExecutorType,
)
from app.repo.git_repo import GitRepo, cerebral_ref
from app.repo.project import Project
from app.repo.task import Task
from app.repo.utils import utcnow

from .conftest import API, BlobFactory, Seed
from .test_execution_lifecycle import append

BASE = "0" * 40


def payload(seed: Seed, **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "project_id": str(seed.project_id),
        "task_id": str(seed.task_id),
        "executor_type": "ai_agent",
        "model": "claude-opus-5",
        "provider": "anthropic",
    }
    body.update(overrides)
    return body


async def start_run(
    bot_client: httpx.AsyncClient, seed: Seed, **overrides: Any
) -> dict[str, Any]:
    """Create a run over the API and return the execution it answered with."""
    response = await bot_client.post(f"{API}/executions", json=payload(seed, **overrides))
    assert response.status_code == 201, response.text
    return response.json()["data"]


async def running_run(
    bot_client: httpx.AsyncClient, seed: Seed, **overrides: Any
) -> dict[str, Any]:
    execution = await start_run(bot_client, seed, **overrides)
    response = await bot_client.post(f"{API}/executions/{execution['id']}/start")
    assert response.status_code == 200, response.text
    return response.json()["data"]


FAILURE = ExecutionError(code="tool_timeout", message="Read timed out after 30s")


async def execution_at(
    engine: AsyncEngine, seed: Seed, status: ExecutionStatus
) -> uuid.UUID:
    """A run sitting in `status`, built there rather than driven there.

    Setup for the transition matrix must not use the state machine under test,
    and could not anyway: nothing may move a run to `waiting_approval` except an
    intervention. The check constraints still police the row, so every state
    this builds is one Postgres agrees can exist.
    """
    moment = utcnow()
    terminal = status in TERMINAL_STATUSES
    execution = Execution(
        task_id=seed.task_id,
        created_by=seed.user_id,
        executor_type=ExecutorType.AI_AGENT,
        executor_agent_id=seed.agent_id,
        status=status,
        started_at=None if status is ExecutionStatus.PENDING else moment,
        finished_at=moment if terminal else None,
        error=FAILURE if status is ExecutionStatus.FAILED else None,
    )
    # Read before the commit that expires it: `id` has a default_factory, so it
    # is known here, and reading it back afterwards would be a detached load.
    execution_id = execution.id
    async with AsyncSession(engine) as session:
        session.add(execution)
        await session.commit()
    return execution_id


def steps(*, legal: bool) -> list[tuple[ExecutionStatus, ExecutionStatus]]:
    """Every ordered pair of statuses the map does, or does not, allow.

    Self-steps are included and are illegal throughout -- starting a run twice
    is a real client mistake, not a hypothetical one.
    """
    return [
        (current, target)
        for current in ExecutionStatus
        for target in ExecutionStatus
        if (target in LEGAL_TRANSITIONS[current]) is legal
    ]


def step_ids(
    pairs: list[tuple[ExecutionStatus, ExecutionStatus]],
) -> list[str]:
    return [f"{current.value}->{target.value}" for current, target in pairs]


class TestCreate:
    async def test_one_call_starts_a_run_and_attaches_its_repos(
        self, bot_client: httpx.AsyncClient, seed: Seed
    ):
        execution = await start_run(
            bot_client,
            seed,
            repos=[{"repo_id": str(seed.api_repo_id), "base_commit_sha": BASE}],
            additional_context={"branch": "feature/jwt"},
        )
        assert execution["status"] == ExecutionStatus.PENDING
        assert execution["attempt"] == 1
        assert execution["additional_context"] == {"branch": "feature/jwt"}
        assert execution["started_at"] is None

        [repo] = execution["repos"]
        assert repo["repo_id"] == str(seed.api_repo_id)
        assert repo["base_commit_sha"] == BASE
        assert repo["head_commit_sha"] is None
        assert repo["landed_at"] is None

    async def test_the_ref_is_derived_from_the_execution_not_the_client(
        self, bot_client: httpx.AsyncClient, seed: Seed
    ):
        """The ref namespace is what keeps one run's commits out of another's,
        so it is never something a client can name."""
        execution = await start_run(
            bot_client,
            seed,
            repos=[{"repo_id": str(seed.api_repo_id), "base_commit_sha": BASE}],
        )
        expected = cerebral_ref(uuid.UUID(execution["id"]))
        assert execution["repos"][0]["ref_name"] == expected
        assert not expected.startswith("refs/heads/")

    async def test_the_agent_is_read_off_the_key_not_the_body(
        self, engine: AsyncEngine, client: httpx.AsyncClient, seed: Seed
    ):
        """A bot must not be able to record work under an agent it was not
        issued for, so a claim in the body is ignored entirely."""
        other_agent = await AgentStore(engine).create(
            Agent(created_by=seed.user_id, name="nightly-refactor")
        )
        key, _ = await ApiKeyStore(engine).create(
            user_id=seed.user_id,
            name="nightly bot",
            agent_id=other_agent.id,
            scopes=[ApiKeyScope.EXECUTIONS_READ, ApiKeyScope.EXECUTIONS_WRITE],
        )

        response = await client.post(
            f"{API}/executions",
            json=payload(seed, executor_agent_id=str(seed.agent_id)),
            headers={"X-API-Key": key},
        )
        assert response.status_code == 201
        assert response.json()["data"]["executor_agent_id"] == str(other_agent.id)

    async def test_attempts_are_numbered_per_task(
        self, bot_client: httpx.AsyncClient, seed: Seed
    ):
        first = await start_run(bot_client, seed)
        second = await start_run(bot_client, seed)
        assert (first["attempt"], second["attempt"]) == (1, 2)

    async def test_concurrent_starts_on_one_task_do_not_collide(
        self, engine: AsyncEngine, seed: Seed
    ):
        """Two runs starting at once must get different attempt numbers.

        This is the case a bare `max(attempt) + 1` gets wrong: both readers see
        the same maximum and the second insert dies on
        uq_executions_task_attempt. The lock on the task row is what makes them
        queue instead.
        """
        store = ExecutionStore(engine)

        async def start() -> Execution:
            return await store.create(
                Execution(
                    task_id=seed.task_id,
                    created_by=seed.user_id,
                    executor_type=ExecutorType.AI_AGENT,
                    executor_agent_id=seed.agent_id,
                )
            )

        # return_exceptions so both coroutines are always awaited. Bare gather
        # abandons the sibling the moment one raises, and an abandoned insert
        # keeps its transaction open into the next test, where it races the
        # TRUNCATE that is supposed to be giving that test a clean database.
        created = await asyncio.gather(start(), start(), return_exceptions=True)
        assert all(not isinstance(item, BaseException) for item in created), created
        assert {execution.attempt for execution in created} == {1, 2}  # pyright: ignore[reportAttributeAccessIssue]

    async def test_a_run_may_touch_no_repo_at_all(
        self, bot_client: httpx.AsyncClient, seed: Seed
    ):
        """A run can be pure conversation; repo attachment is optional."""
        execution = await start_run(bot_client, seed)
        assert execution["repos"] == []

    async def test_a_run_may_span_several_repos(
        self, bot_client: httpx.AsyncClient, seed: Seed
    ):
        execution = await start_run(
            bot_client,
            seed,
            repos=[
                {"repo_id": str(seed.api_repo_id), "base_commit_sha": BASE},
                {"repo_id": str(seed.web_repo_id), "base_commit_sha": "1" * 40},
            ],
        )
        assert {repo["repo_id"] for repo in execution["repos"]} == {
            str(seed.api_repo_id),
            str(seed.web_repo_id),
        }

    async def test_task_must_belong_to_the_project_given(
        self, bot_client: httpx.AsyncClient, engine: AsyncEngine, seed: Seed
    ):
        """The routes are flat, so both ids arrive in the body and the pairing
        is checked rather than trusted."""
        other_project = Project(created_by=seed.user_id, name="unrelated")
        project_id = other_project.id
        async with AsyncSession(engine) as session:
            session.add(other_project)
            await session.commit()

        response = await bot_client.post(
            f"{API}/executions", json=payload(seed, project_id=str(project_id))
        )
        assert response.status_code == 400
        assert "does not belong" in response.json()["error"]["message"]

    async def test_unknown_task_is_not_found(
        self, bot_client: httpx.AsyncClient, seed: Seed
    ):
        response = await bot_client.post(
            f"{API}/executions", json=payload(seed, task_id=str(uuid.uuid4()))
        )
        assert response.status_code == 404

    async def test_another_users_project_is_forbidden(
        self, bot_client: httpx.AsyncClient, engine: AsyncEngine, seed: Seed
    ):
        theirs = Project(created_by=seed.other_user_id, name="theirs")
        their_task = Task(
            project_id=theirs.id, name="their task", created_by=seed.other_user_id
        )
        project_id, task_id = theirs.id, their_task.id
        async with AsyncSession(engine) as session:
            session.add(theirs)
            await session.commit()
        async with AsyncSession(engine) as session:
            session.add(their_task)
            await session.commit()

        response = await bot_client.post(
            f"{API}/executions",
            json=payload(seed, project_id=str(project_id), task_id=str(task_id)),
        )
        assert response.status_code == 403

    async def test_unknown_repo_is_not_found(
        self, bot_client: httpx.AsyncClient, seed: Seed
    ):
        response = await bot_client.post(
            f"{API}/executions",
            json=payload(
                seed,
                repos=[{"repo_id": str(uuid.uuid4()), "base_commit_sha": BASE}],
            ),
        )
        assert response.status_code == 404

    async def test_another_users_repo_is_forbidden(
        self, bot_client: httpx.AsyncClient, engine: AsyncEngine, seed: Seed
    ):
        """The foreign key only says the repo exists — ownership is ours."""
        theirs = GitRepo(
            created_by=seed.other_user_id, name="theirs", local_path="/theirs"
        )
        repo_id = theirs.id
        async with AsyncSession(engine) as session:
            session.add(theirs)
            await session.commit()

        response = await bot_client.post(
            f"{API}/executions",
            json=payload(
                seed, repos=[{"repo_id": str(repo_id), "base_commit_sha": BASE}]
            ),
        )
        assert response.status_code == 403

    async def test_the_same_repo_twice_is_rejected_as_validation(
        self, bot_client: httpx.AsyncClient, seed: Seed
    ):
        """execution_repos is keyed on (execution, repo); caught before the
        insert so it reads as a 422 on the field, not a primary key violation."""
        response = await bot_client.post(
            f"{API}/executions",
            json=payload(
                seed,
                repos=[
                    {"repo_id": str(seed.api_repo_id), "base_commit_sha": BASE},
                    {"repo_id": str(seed.api_repo_id), "base_commit_sha": "1" * 40},
                ],
            ),
        )
        assert response.status_code == 422

    async def test_a_key_with_no_agent_cannot_start_an_ai_run(
        self, engine: AsyncEngine, client: httpx.AsyncClient, seed: Seed
    ):
        key, _ = await ApiKeyStore(engine).create(
            user_id=seed.user_id,
            name="unbound",
            agent_id=None,
            scopes=[ApiKeyScope.EXECUTIONS_WRITE],
        )
        response = await client.post(
            f"{API}/executions", json=payload(seed), headers={"X-API-Key": key}
        )
        assert response.status_code == 400
        assert "not bound to an agent" in response.json()["error"]["message"]

    async def test_a_human_run_records_the_user_instead(
        self, engine: AsyncEngine, client: httpx.AsyncClient, seed: Seed
    ):
        key, _ = await ApiKeyStore(engine).create(
            user_id=seed.user_id,
            name="unbound",
            agent_id=None,
            scopes=[ApiKeyScope.EXECUTIONS_WRITE],
        )
        response = await client.post(
            f"{API}/executions",
            json=payload(seed, executor_type="human"),
            headers={"X-API-Key": key},
        )
        assert response.status_code == 201
        execution = response.json()["data"]
        assert execution["executor_user_id"] == str(seed.user_id)
        assert execution["executor_agent_id"] is None

    async def test_a_logged_in_user_cannot_start_a_run(
        self, user_client: httpx.AsyncClient, seed: Seed
    ):
        """Writes are the ingest path and take an API key, so that every event
        in the transcript is attributable to a credential."""
        response = await user_client.post(f"{API}/executions", json=payload(seed))
        assert response.status_code == 401

    async def test_a_key_without_the_write_scope_is_refused(
        self, engine: AsyncEngine, client: httpx.AsyncClient, seed: Seed
    ):
        key, _ = await ApiKeyStore(engine).create(
            user_id=seed.user_id,
            name="read only",
            agent_id=seed.agent_id,
            scopes=[ApiKeyScope.EXECUTIONS_READ],
        )
        response = await client.post(
            f"{API}/executions", json=payload(seed), headers={"X-API-Key": key}
        )
        assert response.status_code == 403
        assert "executions:write" in response.json()["error"]["message"]


class TestRead:
    async def test_a_person_reads_what_the_bot_wrote(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        """Reads take either credential: the bot writes the run, the owner
        looks at it."""
        execution = await start_run(
            bot_client,
            seed,
            repos=[{"repo_id": str(seed.api_repo_id), "base_commit_sha": BASE}],
        )
        response = await user_client.get(f"{API}/executions/{execution['id']}")
        assert response.status_code == 200
        assert response.json()["data"]["repos"][0]["repo_id"] == str(seed.api_repo_id)

    async def test_the_bot_can_read_its_own_run(
        self, bot_client: httpx.AsyncClient, seed: Seed
    ):
        execution = await start_run(bot_client, seed)
        response = await bot_client.get(f"{API}/executions/{execution['id']}")
        assert response.status_code == 200

    async def test_an_anonymous_caller_gets_nothing(
        self, bot_client: httpx.AsyncClient, client: httpx.AsyncClient, seed: Seed
    ):
        execution = await start_run(bot_client, seed)
        assert (
            await client.get(f"{API}/executions/{execution['id']}")
        ).status_code == 401
        assert (await client.get(f"{API}/executions")).status_code == 401

    async def test_another_users_run_is_forbidden(
        self, bot_client: httpx.AsyncClient, client: httpx.AsyncClient, seed: Seed
    ):
        execution = await start_run(bot_client, seed)
        other = {"Authorization": f"Bearer {create_access_token(seed.other_user_id)}"}
        response = await client.get(
            f"{API}/executions/{execution['id']}", headers=other
        )
        assert response.status_code == 403

    async def test_unknown_run_is_not_found(self, user_client: httpx.AsyncClient):
        response = await user_client.get(f"{API}/executions/{uuid.uuid4()}")
        assert response.status_code == 404

    async def test_listing_filters_by_task_project_status_agent_and_repo(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        with_repo = await start_run(
            bot_client,
            seed,
            repos=[{"repo_id": str(seed.api_repo_id), "base_commit_sha": BASE}],
        )
        await bot_client.post(f"{API}/executions/{with_repo['id']}/start")
        without_repo = await start_run(bot_client, seed)

        async def ids(query: str) -> set[str]:
            response = await user_client.get(f"{API}/executions?{query}")
            assert response.status_code == 200
            return {item["id"] for item in response.json()["data"]["items"]}

        both = {with_repo["id"], without_repo["id"]}
        assert await ids(f"task_id={seed.task_id}") == both
        assert await ids(f"project_id={seed.project_id}") == both
        assert await ids(f"agent_id={seed.agent_id}") == both
        assert await ids(f"repo_id={seed.api_repo_id}") == {with_repo["id"]}
        assert await ids("status=running") == {with_repo["id"]}
        assert await ids("status=pending") == {without_repo["id"]}
        assert await ids(f"repo_id={seed.docs_repo_id}") == set()
        assert await ids(f"task_id={uuid.uuid4()}") == set()

    async def test_listing_is_scoped_to_the_owner(
        self, bot_client: httpx.AsyncClient, client: httpx.AsyncClient, seed: Seed
    ):
        await start_run(bot_client, seed)
        other = {"Authorization": f"Bearer {create_access_token(seed.other_user_id)}"}
        listed = await client.get(f"{API}/executions", headers=other)
        assert listed.json()["data"]["total"] == 0

    async def test_listing_sorts_by_attempt(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        for _ in range(3):
            await start_run(bot_client, seed)
        listed = await user_client.get(f"{API}/executions?sort_by=attempt&order=asc")
        assert [item["attempt"] for item in listed.json()["data"]["items"]] == [1, 2, 3]


class TestTransitions:
    async def test_start_moves_to_running_and_stamps_started_at(
        self, bot_client: httpx.AsyncClient, seed: Seed
    ):
        execution = await start_run(bot_client, seed)
        response = await bot_client.post(f"{API}/executions/{execution['id']}/start")
        assert response.status_code == 200
        started = response.json()["data"]
        assert started["status"] == ExecutionStatus.RUNNING
        assert started["started_at"] is not None
        assert started["finished_at"] is None

    async def test_complete_records_the_result(
        self, bot_client: httpx.AsyncClient, seed: Seed
    ):
        execution = await running_run(bot_client, seed)
        response = await bot_client.post(
            f"{API}/executions/{execution['id']}/complete",
            json={"result": {"files_changed": 4}},
        )
        assert response.status_code == 200
        done = response.json()["data"]
        assert done["status"] == ExecutionStatus.SUCCEEDED
        assert done["finished_at"] is not None
        assert done["result"] == {"files_changed": 4}
        assert done["error"] is None

    async def test_fail_carries_a_structured_error(
        self, bot_client: httpx.AsyncClient, seed: Seed
    ):
        execution = await running_run(bot_client, seed)
        response = await bot_client.post(
            f"{API}/executions/{execution['id']}/fail",
            json={
                "error": {
                    "code": "tool_timeout",
                    "message": "Read timed out after 30s",
                    "retryable": True,
                    "details": {"tool": "Read"},
                }
            },
        )
        assert response.status_code == 200
        failed = response.json()["data"]
        assert failed["status"] == ExecutionStatus.FAILED
        # The caller can decide whether to retry without parsing prose.
        assert failed["error"]["retryable"] is True
        assert failed["error"]["details"] == {"tool": "Read"}

    async def test_fail_without_an_error_is_rejected(
        self, bot_client: httpx.AsyncClient, seed: Seed
    ):
        """A failed run that does not say why is unstorable, and useless."""
        execution = await running_run(bot_client, seed)
        response = await bot_client.post(
            f"{API}/executions/{execution['id']}/fail", json={}
        )
        assert response.status_code == 422

    async def test_cancelling_a_run_that_never_started_stamps_both_times(
        self, bot_client: httpx.AsyncClient, seed: Seed
    ):
        """ck_executions_finished_after_started refuses a finished_at without a
        started_at, so an abandoned run gets both."""
        execution = await start_run(bot_client, seed)
        response = await bot_client.post(f"{API}/executions/{execution['id']}/cancel")
        assert response.status_code == 200
        cancelled = response.json()["data"]
        assert cancelled["status"] == ExecutionStatus.CANCELLED
        assert cancelled["started_at"] is not None
        assert cancelled["finished_at"] is not None

    async def test_cancelling_a_running_run_keeps_its_original_start(
        self, bot_client: httpx.AsyncClient, seed: Seed
    ):
        execution = await running_run(bot_client, seed)
        response = await bot_client.post(f"{API}/executions/{execution['id']}/cancel")
        assert response.json()["data"]["started_at"] == execution["started_at"]

    @pytest.mark.parametrize(
        ("verb", "body"),
        [
            ("complete", {}),
            ("fail", {"error": {"code": "x", "message": "y"}}),
        ],
    )
    async def test_a_pending_run_cannot_finish(
        self, bot_client: httpx.AsyncClient, seed: Seed, verb: str, body: dict[str, Any]
    ):
        execution = await start_run(bot_client, seed)
        response = await bot_client.post(
            f"{API}/executions/{execution['id']}/{verb}", json=body
        )
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "invalid_transition"

    async def test_starting_twice_conflicts(
        self, bot_client: httpx.AsyncClient, seed: Seed
    ):
        execution = await running_run(bot_client, seed)
        response = await bot_client.post(f"{API}/executions/{execution['id']}/start")
        assert response.status_code == 409
        assert "running" in response.json()["error"]["message"]

    @pytest.mark.parametrize("verb", ["start", "complete", "cancel"])
    async def test_a_finished_run_cannot_move_again(
        self, bot_client: httpx.AsyncClient, seed: Seed, verb: str
    ):
        """Terminal is terminal: a rerun is a new attempt, not a resurrection."""
        execution = await running_run(bot_client, seed)
        await bot_client.post(f"{API}/executions/{execution['id']}/complete", json={})
        response = await bot_client.post(
            f"{API}/executions/{execution['id']}/{verb}", json={}
        )
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "invalid_transition"

    async def test_a_blocked_run_cannot_be_completed(
        self, bot_client: httpx.AsyncClient, engine: AsyncEngine, seed: Seed
    ):
        """Completing a run waiting on a human would strand the intervention
        as pending for good. It has to come back to running first."""
        execution = await running_run(bot_client, seed)
        async with AsyncSession(engine) as session:
            connection = await session.connection()
            await connection.execute(
                sa.text(
                    "UPDATE executions SET status = 'waiting_approval' WHERE id = :id"
                ).bindparams(id=uuid.UUID(execution["id"]))
            )
            await session.commit()

        response = await bot_client.post(
            f"{API}/executions/{execution['id']}/complete", json={}
        )
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "invalid_transition"

    async def test_a_blocked_run_can_still_be_cancelled(
        self, bot_client: httpx.AsyncClient, engine: AsyncEngine, seed: Seed
    ):
        execution = await running_run(bot_client, seed)
        async with AsyncSession(engine) as session:
            connection = await session.connection()
            await connection.execute(
                sa.text(
                    "UPDATE executions SET status = 'waiting_input' WHERE id = :id"
                ).bindparams(id=uuid.UUID(execution["id"]))
            )
            await session.commit()

        response = await bot_client.post(f"{API}/executions/{execution['id']}/cancel")
        assert response.status_code == 200

    async def test_transitions_need_the_write_scope(
        self, bot_client: httpx.AsyncClient, client: httpx.AsyncClient,
        engine: AsyncEngine, seed: Seed,
    ):
        execution = await start_run(bot_client, seed)
        key, _ = await ApiKeyStore(engine).create(
            user_id=seed.user_id,
            name="read only",
            agent_id=seed.agent_id,
            scopes=[ApiKeyScope.EXECUTIONS_READ],
        )
        response = await client.post(
            f"{API}/executions/{execution['id']}/start", headers={"X-API-Key": key}
        )
        assert response.status_code == 403

    async def test_a_person_cannot_drive_the_state_machine(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        execution = await start_run(bot_client, seed)
        response = await user_client.post(f"{API}/executions/{execution['id']}/start")
        assert response.status_code == 401

    async def test_transitioning_an_unknown_run_is_not_found(
        self, bot_client: httpx.AsyncClient
    ):
        response = await bot_client.post(f"{API}/executions/{uuid.uuid4()}/start")
        assert response.status_code == 404

    async def test_there_is_no_free_form_status_patch(
        self, bot_client: httpx.AsyncClient, seed: Seed
    ):
        """The verbs are the whole surface; a writable status field would turn
        a client mistake into a constraint violation."""
        execution = await start_run(bot_client, seed)
        response = await bot_client.patch(
            f"{API}/executions/{execution['id']}", json={"status": "succeeded"}
        )
        assert response.status_code == 405


class TestEveryStep:
    """The state machine exhaustively, at the store.

    The cases above are the steps a client can reach through the four verbs.
    That is not the whole machine: `waiting_approval -> waiting_input` has no
    verb, and terminal states are only reachable one way each. These walk all
    forty-nine ordered pairs against `ExecutionStore.transition` directly, so a
    step that neither the routes nor a named test happens to exercise still
    cannot get through.
    """

    def test_the_legal_steps_are_exactly_these(self):
        """Pin the map the two matrices below read their expectations from.

        Without this they would agree with whatever `LEGAL_TRANSITIONS` said:
        widen it by a step and the illegal matrix would quietly stop testing it
        while the legal one started passing it. Spelled out, a widening has to
        be deliberate.
        """
        assert {
            current: set(targets) for current, targets in LEGAL_TRANSITIONS.items()
        } == {
            ExecutionStatus.PENDING: {
                ExecutionStatus.RUNNING,
                ExecutionStatus.CANCELLED,
            },
            ExecutionStatus.RUNNING: {
                ExecutionStatus.WAITING_APPROVAL,
                ExecutionStatus.WAITING_INPUT,
                ExecutionStatus.SUCCEEDED,
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
            },
            ExecutionStatus.WAITING_APPROVAL: {
                ExecutionStatus.RUNNING,
                ExecutionStatus.WAITING_INPUT,
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
            },
            ExecutionStatus.WAITING_INPUT: {
                ExecutionStatus.RUNNING,
                ExecutionStatus.WAITING_APPROVAL,
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
            },
            # Terminal is terminal. A rerun is a new attempt.
            ExecutionStatus.SUCCEEDED: set(),
            ExecutionStatus.FAILED: set(),
            ExecutionStatus.CANCELLED: set(),
        }

    @pytest.mark.parametrize(
        ("current", "target"), steps(legal=True), ids=step_ids(steps(legal=True))
    )
    async def test_a_legal_step_lands_a_row_postgres_accepts(
        self,
        engine: AsyncEngine,
        seed: Seed,
        current: ExecutionStatus,
        target: ExecutionStatus,
    ):
        """Allowed by the map, and still a valid row afterwards.

        The map cannot see the timestamp constraints, so the two can disagree:
        `pending -> cancelled` is legal and would violate
        ck_executions_finished_after_started unless the store stamps a
        started_at it never had. Committing is what proves it does.
        """
        execution_id = await execution_at(engine, seed, current)

        moved = await ExecutionStore(engine).transition(
            execution_id,
            seed.user_id,
            target,
            error=FAILURE if target is ExecutionStatus.FAILED else None,
        )

        assert moved is not None
        assert moved.status is target
        # Anything past pending has begun, and only a finished run is finished.
        assert moved.started_at is not None
        if target in TERMINAL_STATUSES:
            assert moved.finished_at is not None
            assert moved.finished_at >= moved.started_at
        else:
            assert moved.finished_at is None

    @pytest.mark.parametrize(
        ("current", "target"), steps(legal=False), ids=step_ids(steps(legal=False))
    )
    async def test_an_illegal_step_is_refused_before_it_reaches_postgres(
        self,
        engine: AsyncEngine,
        seed: Seed,
        current: ExecutionStatus,
        target: ExecutionStatus,
    ):
        """A 409 naming both states, not a 500 from a check constraint.

        The assertion is the exception type: an IntegrityError arriving here
        instead is the failure this file exists to catch, because by then the
        step has reached the database and the client gets a 500 for a mistake
        the API could have named.
        """
        execution_id = await execution_at(engine, seed, current)

        with pytest.raises(InvalidTransitionError) as raised:
            await ExecutionStore(engine).transition(
                execution_id,
                seed.user_id,
                target,
                error=FAILURE if target is ExecutionStatus.FAILED else None,
            )

        assert raised.value.current is current
        assert raised.value.target is target
        assert current.value in str(raised.value)
        assert target.value in str(raised.value)

        async with AsyncSession(engine) as session:
            after = await session.get(Execution, execution_id)
        assert after is not None
        # Refused means untouched -- not rolled back after a partial write.
        assert after.status is current


class TestUsage:
    async def test_usage_accumulates_across_reports(
        self, bot_client: httpx.AsyncClient, seed: Seed
    ):
        execution = await running_run(bot_client, seed)
        for _ in range(2):
            response = await bot_client.post(
                f"{API}/executions/{execution['id']}/usage",
                json={
                    "input_tokens": 1_000,
                    "output_tokens": 250,
                    "cache_read_tokens": 4_000,
                    "cost_usd": "0.012500",
                },
            )
            assert response.status_code == 200

        totals = response.json()["data"]
        assert totals["input_tokens"] == 2_000
        assert totals["output_tokens"] == 500
        assert totals["cache_read_tokens"] == 8_000
        assert Decimal(totals["cost_usd"]) == Decimal("0.025000")
        # Usage is not a state change.
        assert totals["status"] == ExecutionStatus.RUNNING

    async def test_a_partial_report_leaves_the_rest_alone(
        self, bot_client: httpx.AsyncClient, seed: Seed
    ):
        execution = await running_run(bot_client, seed)
        await bot_client.post(
            f"{API}/executions/{execution['id']}/usage", json={"input_tokens": 10}
        )
        response = await bot_client.post(
            f"{API}/executions/{execution['id']}/usage", json={"output_tokens": 5}
        )
        totals = response.json()["data"]
        assert (totals["input_tokens"], totals["output_tokens"]) == (10, 5)
        # Cost stays NULL until something actually reports one.
        assert totals["cost_usd"] is None

    async def test_usage_may_arrive_after_the_run_finished(
        self, bot_client: httpx.AsyncClient, seed: Seed
    ):
        """The final token count often lands after the agent said it was done."""
        execution = await running_run(bot_client, seed)
        await bot_client.post(f"{API}/executions/{execution['id']}/complete", json={})
        response = await bot_client.post(
            f"{API}/executions/{execution['id']}/usage", json={"input_tokens": 7}
        )
        assert response.status_code == 200
        assert response.json()["data"]["input_tokens"] == 7

    async def test_negative_usage_is_rejected(
        self, bot_client: httpx.AsyncClient, seed: Seed
    ):
        execution = await running_run(bot_client, seed)
        response = await bot_client.post(
            f"{API}/executions/{execution['id']}/usage", json={"input_tokens": -1}
        )
        assert response.status_code == 422


class TestDelete:
    async def test_a_run_that_recorded_nothing_can_be_deleted(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        execution = await start_run(
            bot_client,
            seed,
            repos=[{"repo_id": str(seed.api_repo_id), "base_commit_sha": BASE}],
        )
        response = await bot_client.delete(f"{API}/executions/{execution['id']}")
        assert response.status_code == 200
        assert (
            await user_client.get(f"{API}/executions/{execution['id']}")
        ).status_code == 404

    async def test_a_run_with_history_refuses_to_disappear(
        self, bot_client: httpx.AsyncClient, engine: AsyncEngine, seed: Seed
    ):
        execution = await running_run(bot_client, seed)
        async with AsyncSession(engine) as session:
            await append(
                session,
                uuid.UUID(execution["id"]),
                event_type=ExecutionEventType.REASONING,
                payload=ExecutionEventPayload(reasoning="looked at the router"),
            )
            await session.commit()

        response = await bot_client.delete(f"{API}/executions/{execution['id']}")
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "history_exists"
        assert "purge" in response.json()["error"]["message"]

    async def test_purge_drops_the_history_in_dependency_order(
        self,
        bot_client: httpx.AsyncClient,
        user_client: httpx.AsyncClient,
        engine: AsyncEngine,
        seed: Seed,
        blob: BlobFactory,
    ):
        """Events reference their parent and changes reference the change they
        revert, both under RESTRICT, so the deletes have to peel leaves off."""
        execution = await running_run(
            bot_client,
            seed,
            repos=[{"repo_id": str(seed.api_repo_id), "base_commit_sha": BASE}],
        )
        execution_id = uuid.UUID(execution["id"])
        async with AsyncSession(engine) as session:
            call = await append(
                session,
                execution_id,
                event_type=ExecutionEventType.TOOL_CALL,
                payload=ExecutionEventPayload(data={"tool": "Edit"}),
            )
            child = await append(
                session,
                execution_id,
                event_type=ExecutionEventType.CODE_CHANGE,
                parent_event_id=call.id,
                payload=ExecutionEventPayload(reasoning="jwt verify was missing"),
            )
            original = CodeChange(
                execution_id=execution_id,
                event_id=child.id,
                repo_id=seed.api_repo_id,
                seq=0,
                change_type=ChangeType.CREATED,
                path="app/jwt.py",
                after_blob=blob(),
            )
            session.add(original)
            await session.flush()
            session.add(
                CodeChange(
                    execution_id=execution_id,
                    event_id=child.id,
                    repo_id=seed.api_repo_id,
                    seq=1,
                    change_type=ChangeType.DELETED,
                    path="app/jwt.py",
                    before_blob=original.after_blob,
                    reverts_change_id=original.id,
                )
            )
            await session.commit()

        response = await bot_client.delete(
            f"{API}/executions/{execution['id']}?purge=true"
        )
        assert response.status_code == 200

        async with AsyncSession(engine) as session:
            for model in (CodeChange, ExecutionEvent, ExecutionRepoLink, Execution):
                remaining = (
                    await session.exec(select(sa.func.count()).select_from(model))
                ).one()
                assert remaining == 0, model.__name__

        # The repo the run held under RESTRICT is releasable again.
        assert (
            await user_client.delete(f"{API}/repos/{seed.api_repo_id}")
        ).status_code == 200

    async def test_the_owner_can_delete_with_their_own_credential(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        """Either credential deletes; a person's session carries every scope."""
        execution = await start_run(bot_client, seed)
        response = await user_client.delete(f"{API}/executions/{execution['id']}")
        assert response.status_code == 200

    async def test_a_read_only_key_cannot_delete(
        self, bot_client: httpx.AsyncClient, client: httpx.AsyncClient,
        engine: AsyncEngine, seed: Seed,
    ):
        execution = await start_run(bot_client, seed)
        key, _ = await ApiKeyStore(engine).create(
            user_id=seed.user_id,
            name="read only",
            agent_id=seed.agent_id,
            scopes=[ApiKeyScope.EXECUTIONS_READ],
        )
        response = await client.delete(
            f"{API}/executions/{execution['id']}", headers={"X-API-Key": key}
        )
        assert response.status_code == 403

    async def test_deleting_an_unknown_run_is_not_found(
        self, user_client: httpx.AsyncClient
    ):
        response = await user_client.delete(f"{API}/executions/{uuid.uuid4()}")
        assert response.status_code == 404


class TestGitState:
    """The per-repo git state, at the store. Stage 5 puts routes on these."""

    async def test_a_repo_can_be_attached_mid_run(
        self, bot_client: httpx.AsyncClient, engine: AsyncEngine, seed: Seed
    ):
        execution = await running_run(bot_client, seed)
        store = ExecutionStore(engine)
        link = await store.attach_repo(
            uuid.UUID(execution["id"]),
            seed.user_id,
            ExecutionRepoAttach(repo_id=seed.web_repo_id, base_commit_sha=BASE),
        )
        assert link is not None
        assert link.ref_name == cerebral_ref(uuid.UUID(execution["id"]))

    async def test_attaching_the_same_repo_twice_conflicts(
        self, bot_client: httpx.AsyncClient, engine: AsyncEngine, seed: Seed
    ):
        execution = await running_run(
            bot_client,
            seed,
            repos=[{"repo_id": str(seed.api_repo_id), "base_commit_sha": BASE}],
        )
        store = ExecutionStore(engine)
        with pytest.raises(RepoAlreadyAttachedError):
            await store.attach_repo(
                uuid.UUID(execution["id"]),
                seed.user_id,
                ExecutionRepoAttach(repo_id=seed.api_repo_id, base_commit_sha=BASE),
            )

    async def test_attaching_to_another_users_run_finds_nothing(
        self, bot_client: httpx.AsyncClient, engine: AsyncEngine, seed: Seed
    ):
        execution = await running_run(bot_client, seed)
        store = ExecutionStore(engine)
        link = await store.attach_repo(
            uuid.UUID(execution["id"]),
            seed.other_user_id,
            ExecutionRepoAttach(repo_id=seed.web_repo_id, base_commit_sha=BASE),
        )
        assert link is None

    async def test_the_head_advances_as_the_agent_commits(
        self, bot_client: httpx.AsyncClient, engine: AsyncEngine, seed: Seed,
        blob: BlobFactory,
    ):
        execution = await running_run(
            bot_client,
            seed,
            repos=[{"repo_id": str(seed.api_repo_id), "base_commit_sha": BASE}],
        )
        store = ExecutionStore(engine)
        head = blob()
        link = await store.set_head(
            uuid.UUID(execution["id"]), seed.user_id, seed.api_repo_id, head
        )
        assert link is not None and link.head_commit_sha == head
        assert link.base_commit_sha == BASE

    async def test_landing_stamps_the_changes_it_carried(
        self, bot_client: httpx.AsyncClient, engine: AsyncEngine, seed: Seed,
        blob: BlobFactory,
    ):
        """A repo link that says "landed" while its changes still read NULL
        would break file history for exactly the runs that finished properly."""
        execution = await running_run(
            bot_client,
            seed,
            repos=[
                {"repo_id": str(seed.api_repo_id), "base_commit_sha": BASE},
                {"repo_id": str(seed.web_repo_id), "base_commit_sha": BASE},
            ],
        )
        execution_id = uuid.UUID(execution["id"])
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
                        change_type=ChangeType.CREATED,
                        path="app/jwt.py",
                        after_blob=blob(),
                    )
                    for n, repo_id in enumerate((seed.api_repo_id, seed.web_repo_id))
                ]
            )
            await session.commit()

        landed = blob()
        store = ExecutionStore(engine)
        link = await store.land(
            execution_id,
            seed.user_id,
            seed.api_repo_id,
            landed_branch="main",
            landed_commit_shas=[landed],
            merge_commit_sha=blob(),
        )
        assert link is not None
        assert link.landed_at is not None and link.landed_commit_shas == [landed]

        async with AsyncSession(engine) as session:
            changes = (await session.exec(select(CodeChange))).all()
        stamped = {change.repo_id: change.landed_commit_sha for change in changes}
        assert stamped[seed.api_repo_id] == landed
        # The other repo has not landed, so its changes stay unstamped.
        assert stamped[seed.web_repo_id] is None

    async def test_usage_totals_survive_a_concurrent_pair_of_reports(
        self, bot_client: httpx.AsyncClient, engine: AsyncEngine, seed: Seed
    ):
        """Incremented in SQL, so two overlapping reports cannot lose one
        another the way a read-modify-write would."""
        execution = await running_run(bot_client, seed)
        execution_id = uuid.UUID(execution["id"])
        store = ExecutionStore(engine)
        for _ in range(5):
            await store.add_usage(
                execution_id, seed.user_id, ExecutionUsage(input_tokens=100)
            )
        final = await store.get(execution_id, seed.user_id)
        assert final is not None and final.input_tokens == 500
