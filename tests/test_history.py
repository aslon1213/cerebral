"""Git state, and the question the whole design exists to answer.

Somebody opens a file months later and asks why a line is there. The answer is a
join: the code change back to the event that produced it, and so to what the
agent was thinking at the time. These tests build that situation across two runs
and two repos and then ask the question.

Everything returned is a coordinate — blob ids, commit shas, paths. Never file
content: git already holds the bytes, and the client renders the diff locally.
"""

import uuid
from typing import Any

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.security import create_access_token
from app.repo.api_keys import ApiKeyScope, ApiKeyStore
from app.repo.execution import ChangeType, ExecutionStatus

from .conftest import API, BlobFactory, Seed
from .test_executions import BASE, running_run

MAIN = "main"


async def run_touching(
    bot_client: httpx.AsyncClient, seed: Seed, *repo_ids: uuid.UUID
) -> dict[str, Any]:
    return await running_run(
        bot_client,
        seed,
        repos=[
            {"repo_id": str(repo_id), "base_commit_sha": BASE} for repo_id in repo_ids
        ],
    )


async def record(
    bot_client: httpx.AsyncClient,
    execution_id: str,
    repo_id: uuid.UUID,
    blob: BlobFactory,
    *,
    path: str,
    reasoning: str,
    change_type: str = "modified",
) -> dict[str, Any]:
    """One code_change event: the agent edits a file and says why."""
    change: dict[str, Any] = {
        "repo_id": str(repo_id),
        "path": path,
        "change_type": change_type,
        "after_blob": blob(),
        "lines_added": 4,
        "lines_deleted": 1,
    }
    if change_type != "created":
        change["before_blob"] = blob()

    response = await bot_client.post(
        f"{API}/executions/{execution_id}/events",
        json={
            "event_type": "code_change",
            "actor_type": "agent",
            # The commit this change was made in, under refs/cerebral/*.
            "cerebral_commit_sha": blob(),
            "payload": {"reasoning": reasoning},
            "code_changes": [change],
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["data"]


class TestRepoState:
    async def test_a_repo_can_join_a_run_already_under_way(
        self, bot_client: httpx.AsyncClient, seed: Seed
    ):
        """An agent may follow an import into a sibling checkout halfway."""
        execution = await run_touching(bot_client, seed, seed.api_repo_id)
        response = await bot_client.post(
            f"{API}/executions/{execution['id']}/repos",
            json={"repo_id": str(seed.web_repo_id), "base_commit_sha": BASE},
        )
        assert response.status_code == 201
        link = response.json()["data"]
        assert link["repo_id"] == str(seed.web_repo_id)
        # Same run, same ref: the commits stay grouped by execution.
        assert link["ref_name"].endswith(execution["id"])
        assert not link["ref_name"].startswith("refs/heads/")

    async def test_attaching_the_same_repo_twice_conflicts(
        self, bot_client: httpx.AsyncClient, seed: Seed
    ):
        execution = await run_touching(bot_client, seed, seed.api_repo_id)
        response = await bot_client.post(
            f"{API}/executions/{execution['id']}/repos",
            json={"repo_id": str(seed.api_repo_id), "base_commit_sha": BASE},
        )
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "repo_already_attached"

    async def test_another_users_repo_cannot_be_attached(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        execution = await run_touching(bot_client, seed)
        other = create_access_token(seed.other_user_id)
        theirs = (
            await user_client.post(
                f"{API}/repos",
                json={"name": "theirs", "local_path": "/theirs"},
                headers={"Authorization": f"Bearer {other}"},
            )
        ).json()["data"]["repo"]

        response = await bot_client.post(
            f"{API}/executions/{execution['id']}/repos",
            json={"repo_id": theirs["id"], "base_commit_sha": BASE},
        )
        assert response.status_code == 403

    async def test_the_head_advances_as_the_agent_commits(
        self, bot_client: httpx.AsyncClient, seed: Seed, blob: BlobFactory
    ):
        execution = await run_touching(bot_client, seed, seed.api_repo_id)
        head = blob()
        response = await bot_client.post(
            f"{API}/executions/{execution['id']}/repos/{seed.api_repo_id}/head",
            json={"head_commit_sha": head},
        )
        assert response.status_code == 200
        link = response.json()["data"]
        assert link["head_commit_sha"] == head
        assert link["base_commit_sha"] == BASE

    async def test_git_state_is_listed_per_repo(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient,
        seed: Seed, blob: BlobFactory,
    ):
        execution = await run_touching(
            bot_client, seed, seed.api_repo_id, seed.web_repo_id
        )
        await bot_client.post(
            f"{API}/executions/{execution['id']}/repos/{seed.api_repo_id}/head",
            json={"head_commit_sha": blob()},
        )

        listed = (
            await user_client.get(f"{API}/executions/{execution['id']}/repos")
        ).json()["data"]
        by_repo = {link["repo_id"]: link for link in listed}
        assert set(by_repo) == {str(seed.api_repo_id), str(seed.web_repo_id)}
        assert by_repo[str(seed.api_repo_id)]["head_commit_sha"] is not None
        assert by_repo[str(seed.web_repo_id)]["head_commit_sha"] is None

    async def test_head_on_an_unattached_repo_is_not_found(
        self, bot_client: httpx.AsyncClient, seed: Seed, blob: BlobFactory
    ):
        execution = await run_touching(bot_client, seed, seed.api_repo_id)
        response = await bot_client.post(
            f"{API}/executions/{execution['id']}/repos/{seed.docs_repo_id}/head",
            json={"head_commit_sha": blob()},
        )
        assert response.status_code == 404

    async def test_a_person_cannot_move_the_git_state(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient,
        seed: Seed, blob: BlobFactory,
    ):
        execution = await run_touching(bot_client, seed, seed.api_repo_id)
        response = await user_client.post(
            f"{API}/executions/{execution['id']}/repos/{seed.api_repo_id}/head",
            json={"head_commit_sha": blob()},
        )
        assert response.status_code == 401


class TestLanding:
    async def test_landing_records_where_the_work_went(
        self, bot_client: httpx.AsyncClient, seed: Seed, blob: BlobFactory
    ):
        execution = await run_touching(bot_client, seed, seed.api_repo_id)
        landed, merge = blob(), blob()

        response = await bot_client.post(
            f"{API}/executions/{execution['id']}/repos/{seed.api_repo_id}/land",
            json={
                "landed_branch": MAIN,
                "landed_commit_shas": [landed],
                "merge_commit_sha": merge,
            },
        )
        assert response.status_code == 200
        link = response.json()["data"]
        assert link["landed_branch"] == MAIN
        assert link["landed_commit_shas"] == [landed]
        assert link["merge_commit_sha"] == merge
        assert link["landed_at"] is not None

    async def test_landing_stamps_the_changes_it_carried(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient,
        seed: Seed, blob: BlobFactory,
    ):
        """The run's cerebral range becomes one commit on the default branch, so
        every change of that repo landed in the same place."""
        execution = await run_touching(
            bot_client, seed, seed.api_repo_id, seed.web_repo_id
        )
        await record(
            bot_client, execution["id"], seed.api_repo_id, blob,
            path="app/auth.py", reasoning="jwt verify was missing",
        )
        await record(
            bot_client, execution["id"], seed.web_repo_id, blob,
            path="src/login.ts", reasoning="call the new endpoint",
        )

        landed = blob()
        await bot_client.post(
            f"{API}/executions/{execution['id']}/repos/{seed.api_repo_id}/land",
            json={"landed_branch": MAIN, "landed_commit_shas": [landed]},
        )

        changes = (
            await user_client.get(f"{API}/executions/{execution['id']}/changes")
        ).json()["data"]["items"]
        stamped = {change["repo_id"]: change["landed_commit_sha"] for change in changes}
        assert stamped[str(seed.api_repo_id)] == landed
        # The other repo has not landed, so its changes stay unstamped.
        assert stamped[str(seed.web_repo_id)] is None

    async def test_landing_needs_somewhere_to_have_landed(
        self, bot_client: httpx.AsyncClient, seed: Seed
    ):
        """An empty commit list is a landing that did not happen."""
        execution = await run_touching(bot_client, seed, seed.api_repo_id)
        response = await bot_client.post(
            f"{API}/executions/{execution['id']}/repos/{seed.api_repo_id}/land",
            json={"landed_branch": MAIN, "landed_commit_shas": []},
        )
        assert response.status_code == 422

    async def test_landing_an_unattached_repo_is_not_found(
        self, bot_client: httpx.AsyncClient, seed: Seed, blob: BlobFactory
    ):
        execution = await run_touching(bot_client, seed, seed.api_repo_id)
        response = await bot_client.post(
            f"{API}/executions/{execution['id']}/repos/{seed.docs_repo_id}/land",
            json={"landed_branch": MAIN, "landed_commit_shas": [blob()]},
        )
        assert response.status_code == 404


class TestWhatARunTouched:
    async def test_changes_are_grouped_by_repo(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient,
        seed: Seed, blob: BlobFactory,
    ):
        execution = await run_touching(
            bot_client, seed, seed.api_repo_id, seed.web_repo_id
        )
        for repo_id, path in (
            (seed.api_repo_id, "app/auth.py"),
            (seed.web_repo_id, "src/login.ts"),
            (seed.api_repo_id, "app/jwt.py"),
        ):
            await record(
                bot_client, execution["id"], repo_id, blob,
                path=path, reasoning=f"work on {path}",
            )

        page = (
            await user_client.get(f"{API}/executions/{execution['id']}/changes")
        ).json()["data"]
        assert page["total"] == 3
        # Grouped: each repo appears as one contiguous run, not interleaved,
        # even though the middle change went to the other repo.
        repos = [change["repo_id"] for change in page["items"]]
        blocks = [repo for index, repo in enumerate(repos) if index == 0 or repo != repos[index - 1]]
        assert len(blocks) == len(set(repos)) == 2

    async def test_changes_filter_by_repo(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient,
        seed: Seed, blob: BlobFactory,
    ):
        execution = await run_touching(
            bot_client, seed, seed.api_repo_id, seed.web_repo_id
        )
        await record(
            bot_client, execution["id"], seed.api_repo_id, blob,
            path="app/auth.py", reasoning="a",
        )
        await record(
            bot_client, execution["id"], seed.web_repo_id, blob,
            path="src/login.ts", reasoning="b",
        )

        page = (
            await user_client.get(
                f"{API}/executions/{execution['id']}/changes?repo_id={seed.api_repo_id}"
            )
        ).json()["data"]
        assert page["total"] == 1
        assert page["items"][0]["path"] == "app/auth.py"

    async def test_a_run_that_touched_nothing(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        execution = await run_touching(bot_client, seed)
        page = (
            await user_client.get(f"{API}/executions/{execution['id']}/changes")
        ).json()["data"]
        assert page["total"] == 0


class TestWhyIsThisLineHere:
    """The endpoint the whole design exists for."""

    async def test_file_history_carries_the_reasoning(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient,
        seed: Seed, blob: BlobFactory,
    ):
        execution = await run_touching(bot_client, seed, seed.api_repo_id)
        reasons = ["extracted the token parser", "jwt verify was missing"]
        for reason in reasons:
            await record(
                bot_client, execution["id"], seed.api_repo_id, blob,
                path="app/auth.py", reasoning=reason,
            )

        page = (
            await user_client.get(
                f"{API}/repos/{seed.api_repo_id}/history?path=app/auth.py"
            )
        ).json()["data"]

        assert page["total"] == 2
        # Newest first: the most recent answer to "why is this line here".
        assert [
            entry["event"]["payload"]["reasoning"] for entry in page["items"]
        ] == list(reversed(reasons))

        for entry in page["items"]:
            change, event = entry["change"], entry["event"]
            # Coordinates to render the diff from, never the content itself.
            assert change["before_blob"] and change["after_blob"]
            assert "content" not in change
            # The commit that produced it, under refs/cerebral/*.
            assert event["cerebral_commit_sha"]
            # And who did it.
            assert entry["executor_agent_id"] == str(seed.agent_id)
            assert entry["attempt"] == 1

    async def test_history_spans_executions(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient,
        seed: Seed, blob: BlobFactory,
    ):
        """The point of asking by repo and path rather than by run: the file was
        touched by two different runs, months apart."""
        first = await run_touching(bot_client, seed, seed.api_repo_id)
        await record(
            bot_client, first["id"], seed.api_repo_id, blob,
            path="app/auth.py", reasoning="first pass at jwt",
        )
        landed = blob()
        await bot_client.post(
            f"{API}/executions/{first['id']}/repos/{seed.api_repo_id}/land",
            json={"landed_branch": MAIN, "landed_commit_shas": [landed]},
        )
        await bot_client.post(f"{API}/executions/{first['id']}/complete", json={})

        second = await run_touching(bot_client, seed, seed.api_repo_id)
        await record(
            bot_client, second["id"], seed.api_repo_id, blob,
            path="app/auth.py", reasoning="rotate the signing key",
        )

        page = (
            await user_client.get(
                f"{API}/repos/{seed.api_repo_id}/history?path=app/auth.py"
            )
        ).json()["data"]

        assert page["total"] == 2
        assert {entry["change"]["execution_id"] for entry in page["items"]} == {
            first["id"],
            second["id"],
        }
        assert [entry["attempt"] for entry in page["items"]] == [2, 1]
        # The landed run carries both commits; the run still going has only the
        # cerebral one, because its work has not reached the default branch.
        by_attempt = {entry["attempt"]: entry for entry in page["items"]}
        assert by_attempt[1]["change"]["landed_commit_sha"] == landed
        assert by_attempt[2]["change"]["landed_commit_sha"] is None
        assert all(entry["event"]["cerebral_commit_sha"] for entry in page["items"])

    async def test_the_same_path_in_two_repos_stays_separate(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient,
        seed: Seed, blob: BlobFactory,
    ):
        """Two repos both have a src/main.py, and they are different files."""
        execution = await run_touching(
            bot_client, seed, seed.api_repo_id, seed.web_repo_id
        )
        await record(
            bot_client, execution["id"], seed.api_repo_id, blob,
            path="src/main.py", reasoning="api entrypoint",
        )
        await record(
            bot_client, execution["id"], seed.web_repo_id, blob,
            path="src/main.py", reasoning="web entrypoint",
        )

        for repo_id, expected in (
            (seed.api_repo_id, "api entrypoint"),
            (seed.web_repo_id, "web entrypoint"),
        ):
            page = (
                await user_client.get(
                    f"{API}/repos/{repo_id}/history?path=src/main.py"
                )
            ).json()["data"]
            assert page["total"] == 1
            assert page["items"][0]["event"]["payload"]["reasoning"] == expected

    async def test_history_without_a_path_covers_the_repo(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient,
        seed: Seed, blob: BlobFactory,
    ):
        execution = await run_touching(bot_client, seed, seed.api_repo_id)
        for path in ("app/auth.py", "app/jwt.py"):
            await record(
                bot_client, execution["id"], seed.api_repo_id, blob,
                path=path, reasoning=f"touch {path}",
            )

        page = (
            await user_client.get(f"{API}/repos/{seed.api_repo_id}/history")
        ).json()["data"]
        assert page["total"] == 2

    async def test_an_untouched_file_has_no_history(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient,
        seed: Seed, blob: BlobFactory,
    ):
        execution = await run_touching(bot_client, seed, seed.api_repo_id)
        await record(
            bot_client, execution["id"], seed.api_repo_id, blob,
            path="app/auth.py", reasoning="only this one",
        )
        page = (
            await user_client.get(
                f"{API}/repos/{seed.api_repo_id}/history?path=app/never.py"
            )
        ).json()["data"]
        assert page["total"] == 0
        assert page["items"] == []

    async def test_one_change_comes_with_its_whole_context(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient,
        seed: Seed, blob: BlobFactory,
    ):
        execution = await run_touching(bot_client, seed, seed.api_repo_id)
        appended = await record(
            bot_client, execution["id"], seed.api_repo_id, blob,
            path="app/auth.py", reasoning="jwt verify was missing",
            change_type="created",
        )
        change_id = appended["code_changes"][0]["id"]

        response = await user_client.get(
            f"{API}/repos/{seed.api_repo_id}/history/{change_id}"
        )
        assert response.status_code == 200
        context = response.json()["data"]

        assert context["change"]["id"] == change_id
        assert context["change"]["change_type"] == ChangeType.CREATED
        assert context["event"]["payload"]["reasoning"] == "jwt verify was missing"
        assert context["event"]["id"] == appended["id"]
        assert context["execution"]["id"] == execution["id"]
        assert context["execution"]["status"] == ExecutionStatus.RUNNING
        assert context["execution"]["executor_agent_id"] == str(seed.agent_id)

    async def test_a_change_from_another_repo_is_not_found_here(
        self, bot_client: httpx.AsyncClient, user_client: httpx.AsyncClient,
        seed: Seed, blob: BlobFactory,
    ):
        execution = await run_touching(
            bot_client, seed, seed.api_repo_id, seed.web_repo_id
        )
        appended = await record(
            bot_client, execution["id"], seed.api_repo_id, blob,
            path="app/auth.py", reasoning="x",
        )
        change_id = appended["code_changes"][0]["id"]

        response = await user_client.get(
            f"{API}/repos/{seed.web_repo_id}/history/{change_id}"
        )
        assert response.status_code == 404

    async def test_unknown_change_is_not_found(
        self, user_client: httpx.AsyncClient, seed: Seed
    ):
        response = await user_client.get(
            f"{API}/repos/{seed.api_repo_id}/history/{uuid.uuid4()}"
        )
        assert response.status_code == 404


class TestHistoryAccess:
    async def test_an_agent_may_read_what_earlier_runs_did(
        self, bot_client: httpx.AsyncClient, seed: Seed, blob: BlobFactory
    ):
        """Useful context before editing a file, and nothing a bot could not
        already reach through its own execution."""
        execution = await run_touching(bot_client, seed, seed.api_repo_id)
        await record(
            bot_client, execution["id"], seed.api_repo_id, blob,
            path="app/auth.py", reasoning="jwt verify was missing",
        )
        response = await bot_client.get(
            f"{API}/repos/{seed.api_repo_id}/history?path=app/auth.py"
        )
        assert response.status_code == 200
        assert response.json()["data"]["total"] == 1

    async def test_a_key_without_repos_read_is_refused(
        self, engine: AsyncEngine, client: httpx.AsyncClient, seed: Seed
    ):
        key, _ = await ApiKeyStore(engine).create(
            user_id=seed.user_id,
            name="executions only",
            agent_id=seed.agent_id,
            scopes=[ApiKeyScope.EXECUTIONS_READ, ApiKeyScope.EXECUTIONS_WRITE],
        )
        response = await client.get(
            f"{API}/repos/{seed.api_repo_id}/history", headers={"X-API-Key": key}
        )
        assert response.status_code == 403

    async def test_another_users_repo_history_is_forbidden(
        self, client: httpx.AsyncClient, seed: Seed
    ):
        other = {"Authorization": f"Bearer {create_access_token(seed.other_user_id)}"}
        response = await client.get(
            f"{API}/repos/{seed.api_repo_id}/history", headers=other
        )
        assert response.status_code == 403

    async def test_an_anonymous_caller_gets_nothing(
        self, client: httpx.AsyncClient, seed: Seed
    ):
        response = await client.get(f"{API}/repos/{seed.api_repo_id}/history")
        assert response.status_code == 401

    @pytest.mark.parametrize("method", ["put", "patch", "delete"])
    async def test_history_is_read_only(
        self, user_client: httpx.AsyncClient, seed: Seed, method: str
    ):
        response = await getattr(user_client, method)(
            f"{API}/repos/{seed.api_repo_id}/history"
        )
        assert response.status_code == 405
