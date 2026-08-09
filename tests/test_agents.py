"""Agents — the identity an execution runs as."""

import httpx

from app.core.security import create_access_token

from .conftest import API, Seed


class TestCreate:
    async def test_create_and_read_back(self, user_client: httpx.AsyncClient):
        response = await user_client.post(
            f"{API}/agents",
            json={
                "name": "nightly-refactor",
                "description": "runs the refactor pass overnight",
                "default_model": "claude-opus-5",
            },
        )
        assert response.status_code == 201
        agent = response.json()["data"]
        assert agent["name"] == "nightly-refactor"
        assert agent["default_model"] == "claude-opus-5"
        assert agent["is_active"] is True

        fetched = await user_client.get(f"{API}/agents/{agent['id']}")
        assert fetched.json()["data"] == agent

    async def test_duplicate_name_is_a_conflict(self, user_client: httpx.AsyncClient):
        await user_client.post(f"{API}/agents", json={"name": "dup"})
        response = await user_client.post(f"{API}/agents", json={"name": "dup"})
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "duplicate_agent_name"

    async def test_two_users_may_each_have_the_same_agent_name(
        self, client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        """Names are unique per owner, not globally."""
        await user_client.post(f"{API}/agents", json={"name": "claude-code"})
        other = {"Authorization": f"Bearer {create_access_token(seed.other_user_id)}"}
        response = await client.post(
            f"{API}/agents", json={"name": "claude-code"}, headers=other
        )
        assert response.status_code == 201

    async def test_empty_name_is_rejected(self, user_client: httpx.AsyncClient):
        response = await user_client.post(f"{API}/agents", json={"name": ""})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"

    async def test_requires_authentication(self, client: httpx.AsyncClient):
        assert (
            await client.post(f"{API}/agents", json={"name": "x"})
        ).status_code == 401


class TestListAndLookup:
    async def test_list_is_scoped_to_the_owner(
        self, client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        await user_client.post(f"{API}/agents", json={"name": "mine"})
        other = {"Authorization": f"Bearer {create_access_token(seed.other_user_id)}"}

        # The seed fixture already created one agent for the owner.
        mine = (await user_client.get(f"{API}/agents")).json()["data"]
        assert mine["total"] == 2
        theirs = (await client.get(f"{API}/agents", headers=other)).json()["data"]
        assert theirs["total"] == 0

    async def test_filter_by_active(self, user_client: httpx.AsyncClient):
        await user_client.post(
            f"{API}/agents", json={"name": "retired", "is_active": False}
        )
        active = (await user_client.get(f"{API}/agents?is_active=true")).json()
        inactive = (await user_client.get(f"{API}/agents?is_active=false")).json()
        assert {a["name"] for a in inactive["data"]["items"]} == {"retired"}
        assert "retired" not in {a["name"] for a in active["data"]["items"]}

    async def test_search_matches_name_and_description(
        self, user_client: httpx.AsyncClient
    ):
        await user_client.post(
            f"{API}/agents", json={"name": "sweeper", "description": "tidies imports"}
        )
        by_name = (await user_client.get(f"{API}/agents?q=sweep")).json()
        by_description = (await user_client.get(f"{API}/agents?q=imports")).json()
        assert by_name["data"]["total"] == 1
        assert by_description["data"]["total"] == 1

    async def test_lookup_by_name(self, user_client: httpx.AsyncClient, seed: Seed):
        """How a bot configured with a name rather than a uuid resolves it."""
        response = await user_client.get(f"{API}/agents/by-name/claude-code")
        assert response.status_code == 200
        assert response.json()["data"]["id"] == str(seed.agent_id)

    async def test_lookup_by_unknown_name_is_404(
        self, user_client: httpx.AsyncClient
    ):
        response = await user_client.get(f"{API}/agents/by-name/nope")
        assert response.status_code == 404

    async def test_another_users_agent_is_forbidden(
        self, client: httpx.AsyncClient, seed: Seed
    ):
        other = {"Authorization": f"Bearer {create_access_token(seed.other_user_id)}"}
        response = await client.get(f"{API}/agents/{seed.agent_id}", headers=other)
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "forbidden"


class TestUpdate:
    async def test_partial_update_leaves_other_fields_alone(
        self, user_client: httpx.AsyncClient, seed: Seed
    ):
        response = await user_client.patch(
            f"{API}/agents/{seed.agent_id}", json={"description": "now documented"}
        )
        assert response.status_code == 200
        agent = response.json()["data"]
        assert agent["description"] == "now documented"
        assert agent["name"] == "claude-code"
        assert agent["default_model"] == "claude-opus-5"

    async def test_renaming_onto_an_existing_name_conflicts(
        self, user_client: httpx.AsyncClient, seed: Seed
    ):
        await user_client.post(f"{API}/agents", json={"name": "taken"})
        response = await user_client.patch(
            f"{API}/agents/{seed.agent_id}", json={"name": "taken"}
        )
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "duplicate_agent_name"

    async def test_deactivating_keeps_the_agent(
        self, user_client: httpx.AsyncClient, seed: Seed
    ):
        """Retiring an agent is a flag, not a delete: its executions still
        need it to resolve."""
        response = await user_client.patch(
            f"{API}/agents/{seed.agent_id}", json={"is_active": False}
        )
        assert response.json()["data"]["is_active"] is False
        assert (
            await user_client.get(f"{API}/agents/{seed.agent_id}")
        ).status_code == 200


class TestDelete:
    async def test_unused_agent_can_be_deleted(self, user_client: httpx.AsyncClient):
        agent_id = (
            await user_client.post(f"{API}/agents", json={"name": "throwaway"})
        ).json()["data"]["id"]
        assert (await user_client.delete(f"{API}/agents/{agent_id}")).status_code == 200
        assert (await user_client.get(f"{API}/agents/{agent_id}")).status_code == 404

    async def test_agent_with_an_api_key_cannot_be_deleted(
        self, user_client: httpx.AsyncClient, seed: Seed
    ):
        """RESTRICT, so the credential never points at a missing agent."""
        await user_client.post(
            f"{API}/api-keys", json={"name": "bot", "agent_id": str(seed.agent_id)}
        )
        response = await user_client.delete(f"{API}/agents/{seed.agent_id}")
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "resource_in_use"
        # Still there, and the message says what to do instead.
        assert (
            await user_client.get(f"{API}/agents/{seed.agent_id}")
        ).status_code == 200
