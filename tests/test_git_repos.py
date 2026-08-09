"""Connecting git repos — the lookup a bot does at startup."""

import httpx
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.security import create_access_token
from app.repo.execution import Execution, ExecutionRepoLink, ExecutorType
from app.repo.git_repo import cerebral_ref

from .conftest import API, Seed


class TestConnect:
    async def test_connecting_a_new_repo_creates_it(
        self, user_client: httpx.AsyncClient
    ):
        response = await user_client.post(
            f"{API}/repos",
            json={
                "name": "billing",
                "local_path": "/src/billing",
                "default_branch": "trunk",
                "remote_url": "git@github.com:acme/billing.git",
            },
        )
        assert response.status_code == 201
        body = response.json()["data"]
        assert body["created"] is True
        assert body["repo"]["default_branch"] == "trunk"

    async def test_connecting_again_is_idempotent(
        self, user_client: httpx.AsyncClient
    ):
        """A bot calls this unconditionally at startup, so the second call has
        to return the same repo rather than conflict."""
        payload = {"name": "billing", "local_path": "/src/billing"}
        first = await user_client.post(f"{API}/repos", json=payload)
        second = await user_client.post(f"{API}/repos", json=payload)

        assert first.status_code == 201
        assert second.status_code == 200
        assert second.json()["data"]["created"] is False
        assert second.json()["data"]["repo"]["id"] == first.json()["data"]["repo"]["id"]

    async def test_reconnecting_refreshes_the_local_path(
        self, user_client: httpx.AsyncClient
    ):
        """The path is per-machine, so the same repo on a second machine must
        update it rather than being rejected as a duplicate."""
        await user_client.post(
            f"{API}/repos", json={"name": "billing", "local_path": "/laptop/billing"}
        )
        response = await user_client.post(
            f"{API}/repos",
            json={"name": "billing", "local_path": "/buildbox/src/billing"},
        )
        assert response.status_code == 200
        assert response.json()["data"]["repo"]["local_path"] == "/buildbox/src/billing"

    async def test_reconnect_without_a_remote_keeps_the_stored_one(
        self, user_client: httpx.AsyncClient
    ):
        await user_client.post(
            f"{API}/repos",
            json={
                "name": "billing",
                "local_path": "/src/billing",
                "remote_url": "git@github.com:acme/billing.git",
            },
        )
        response = await user_client.post(
            f"{API}/repos", json={"name": "billing", "local_path": "/src/billing"}
        )
        assert (
            response.json()["data"]["repo"]["remote_url"]
            == "git@github.com:acme/billing.git"
        )

    async def test_two_users_may_each_connect_a_repo_named_api(
        self, client: httpx.AsyncClient, seed: Seed
    ):
        """The seed owner already has one called "api"."""
        other = {"Authorization": f"Bearer {create_access_token(seed.other_user_id)}"}
        response = await client.post(
            f"{API}/repos",
            json={"name": "api", "local_path": "/elsewhere/api"},
            headers=other,
        )
        assert response.status_code == 201

    async def test_default_branch_defaults_to_main(
        self, user_client: httpx.AsyncClient
    ):
        response = await user_client.post(
            f"{API}/repos", json={"name": "fresh", "local_path": "/src/fresh"}
        )
        assert response.json()["data"]["repo"]["default_branch"] == "main"

    async def test_requires_authentication(self, client: httpx.AsyncClient):
        response = await client.post(
            f"{API}/repos", json={"name": "x", "local_path": "/x"}
        )
        assert response.status_code == 401


class TestLookup:
    async def test_resolve_by_local_path(
        self, user_client: httpx.AsyncClient, seed: Seed
    ):
        """The bot knows the directory it was pointed at, and needs the id."""
        response = await user_client.get(f"{API}/repos?local_path=/src/api")
        assert response.status_code == 200
        items = response.json()["data"]["items"]
        assert len(items) == 1
        assert items[0]["id"] == str(seed.api_repo_id)

    async def test_local_path_is_an_exact_match(self, user_client: httpx.AsyncClient):
        """A prefix must not resolve: /src/api and /src/api-v2 are two repos."""
        await user_client.post(
            f"{API}/repos", json={"name": "api-v2", "local_path": "/src/api-v2"}
        )
        found = (await user_client.get(f"{API}/repos?local_path=/src/api")).json()
        assert found["data"]["total"] == 1
        assert found["data"]["items"][0]["local_path"] == "/src/api"

    async def test_resolve_by_remote_url(self, user_client: httpx.AsyncClient):
        remote = "git@github.com:acme/billing.git"
        await user_client.post(
            f"{API}/repos",
            json={
                "name": "billing",
                "local_path": "/src/billing",
                "remote_url": remote,
            },
        )
        found = await user_client.get(f"{API}/repos", params={"remote_url": remote})
        assert found.json()["data"]["total"] == 1

    async def test_lookup_by_name(self, user_client: httpx.AsyncClient, seed: Seed):
        response = await user_client.get(f"{API}/repos/by-name/api")
        assert response.status_code == 200
        assert response.json()["data"]["id"] == str(seed.api_repo_id)

    async def test_search_spans_name_path_and_remote(
        self, user_client: httpx.AsyncClient
    ):
        await user_client.post(
            f"{API}/repos",
            json={
                "name": "billing",
                "local_path": "/work/ledger",
                "remote_url": "git@github.com:acme/ledger.git",
            },
        )
        for query in ("billing", "ledger", "acme"):
            found = (await user_client.get(f"{API}/repos?q={query}")).json()
            assert found["data"]["total"] >= 1, query

    async def test_list_is_scoped_to_the_owner(
        self, client: httpx.AsyncClient, seed: Seed
    ):
        other = {"Authorization": f"Bearer {create_access_token(seed.other_user_id)}"}
        listed = await client.get(f"{API}/repos", headers=other)
        assert listed.json()["data"]["total"] == 0

    async def test_another_users_repo_is_forbidden(
        self, client: httpx.AsyncClient, seed: Seed
    ):
        other = {"Authorization": f"Bearer {create_access_token(seed.other_user_id)}"}
        response = await client.get(f"{API}/repos/{seed.api_repo_id}", headers=other)
        assert response.status_code == 403


class TestUpdateAndDelete:
    async def test_partial_update(self, user_client: httpx.AsyncClient, seed: Seed):
        response = await user_client.patch(
            f"{API}/repos/{seed.api_repo_id}", json={"default_branch": "develop"}
        )
        assert response.status_code == 200
        repo = response.json()["data"]
        assert repo["default_branch"] == "develop"
        assert repo["name"] == "api"

    async def test_renaming_onto_an_existing_name_conflicts(
        self, user_client: httpx.AsyncClient, seed: Seed
    ):
        response = await user_client.patch(
            f"{API}/repos/{seed.api_repo_id}", json={"name": "web"}
        )
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "duplicate_repo_name"

    async def test_unused_repo_can_be_deleted(
        self, user_client: httpx.AsyncClient, seed: Seed
    ):
        assert (
            await user_client.delete(f"{API}/repos/{seed.docs_repo_id}")
        ).status_code == 200
        assert (
            await user_client.get(f"{API}/repos/{seed.docs_repo_id}")
        ).status_code == 404

    async def test_repo_used_by_an_execution_cannot_be_deleted(
        self, user_client: httpx.AsyncClient, seed: Seed, engine: AsyncEngine
    ):
        """RESTRICT: code changes are anchored to this row."""
        execution = Execution(
            task_id=seed.task_id,
            created_by=seed.user_id,
            executor_type=ExecutorType.AI_AGENT,
            executor_agent_id=seed.agent_id,
        )
        execution_id = execution.id
        async with AsyncSession(engine) as session:
            session.add(execution)
            await session.commit()
        async with AsyncSession(engine) as session:
            session.add(
                ExecutionRepoLink(
                    execution_id=execution_id,
                    repo_id=seed.api_repo_id,
                    ref_name=cerebral_ref(execution_id),
                    base_commit_sha="0" * 40,
                )
            )
            await session.commit()

        response = await user_client.delete(f"{API}/repos/{seed.api_repo_id}")
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "resource_in_use"
