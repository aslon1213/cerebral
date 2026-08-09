"""The credential the observer bot authenticates with.

Two halves: the key primitives (pure, no database), and the endpoints that issue
and check keys over the real app.
"""

import uuid

import httpx
import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.deps import Principal
from app.core.security import (
    API_KEY_BRAND,
    create_access_token,
    generate_api_key,
    hash_api_key,
    parse_api_key,
    verify_api_key,
)
from app.repo.api_keys import ApiKeyScope, ApiKeyStore
from app.repo.users import User

from .conftest import API, Seed


class TestKeyPrimitives:
    def test_generated_key_is_branded_and_parses(self):
        key, prefix, key_hash = generate_api_key()
        assert key.startswith(f"{API_KEY_BRAND}_")
        # The brand makes a leaked key greppable in logs and recognisable to
        # secret scanners.
        assert parse_api_key(key) == prefix
        assert verify_api_key(key, key_hash)

    def test_the_secret_is_not_recoverable_from_what_is_stored(self):
        key, prefix, key_hash = generate_api_key()
        # Only these two are persisted, and neither reveals the key.
        assert prefix in key
        assert key_hash not in key
        assert key.split("_", 2)[2] not in key_hash

    def test_keys_are_unique(self):
        keys = {generate_api_key()[0] for _ in range(200)}
        assert len(keys) == 200

    @pytest.mark.parametrize(
        "candidate",
        [
            "",
            "nonsense",
            "cbrl_onlytwo",
            "wrong_aabbccddeeff_secret",  # wrong brand
            "cbrl__secret",  # empty prefix
            "cbrl_aabbccddeeff_",  # empty secret
            "cbrl_short_secret",  # prefix wrong length
        ],
    )
    def test_malformed_keys_are_rejected_before_any_lookup(self, candidate: str):
        assert parse_api_key(candidate) is None

    def test_a_tampered_key_does_not_verify(self):
        key, _, key_hash = generate_api_key()
        assert not verify_api_key(key + "x", key_hash)
        assert not verify_api_key(key[:-1], key_hash)
        assert not verify_api_key(generate_api_key()[0], key_hash)

    def test_hash_is_stable_and_sha256_shaped(self):
        key, _, key_hash = generate_api_key()
        assert hash_api_key(key) == key_hash
        assert len(key_hash) == 64
        assert int(key_hash, 16) >= 0  # hex


class TestIssuing:
    async def test_key_is_returned_once_and_never_again(
        self, user_client: httpx.AsyncClient, seed: Seed
    ):
        response = await user_client.post(
            f"{API}/api-keys", json={"name": "observer bot"}
        )
        assert response.status_code == 201
        body = response.json()
        assert body["ok"] is True
        key = body["data"]["key"]
        prefix = body["data"]["api_key"]["prefix"]
        assert key.startswith(f"{API_KEY_BRAND}_{prefix}_")

        # Nothing that lists or fetches the key may ever include the secret.
        listed = (await user_client.get(f"{API}/api-keys")).json()
        assert listed["data"]["total"] == 1
        assert "key" not in listed["data"]["items"][0]
        assert key not in response.request.url.params.get("q", "") + str(listed)

    async def test_default_scopes_suit_an_observer_bot(
        self, user_client: httpx.AsyncClient, seed: Seed
    ):
        response = await user_client.post(f"{API}/api-keys", json={"name": "bot"})
        scopes = response.json()["data"]["api_key"]["scopes"]
        assert ApiKeyScope.EXECUTIONS_WRITE in scopes
        assert ApiKeyScope.EXECUTIONS_READ in scopes
        # A bot must never be able to answer its own approval requests.
        assert ApiKeyScope.REPOS_WRITE not in scopes

    async def test_key_can_be_bound_to_an_agent(
        self, user_client: httpx.AsyncClient, seed: Seed
    ):
        response = await user_client.post(
            f"{API}/api-keys",
            json={"name": "bot", "agent_id": str(seed.agent_id)},
        )
        assert response.status_code == 201
        assert response.json()["data"]["api_key"]["agent_id"] == str(seed.agent_id)

    async def test_cannot_bind_a_key_to_an_unknown_agent(
        self, user_client: httpx.AsyncClient
    ):
        response = await user_client.post(
            f"{API}/api-keys",
            json={
                "name": "bot",
                "agent_id": "00000000-0000-0000-0000-000000000000",
            },
        )
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"

    async def test_issuing_requires_a_logged_in_user(
        self, client: httpx.AsyncClient
    ):
        response = await client.post(f"{API}/api-keys", json={"name": "bot"})
        assert response.status_code == 401
        assert response.json()["ok"] is False

    async def test_a_key_cannot_issue_another_key(
        self, client: httpx.AsyncClient, user_client: httpx.AsyncClient
    ):
        """Otherwise a leaked key mints replacements and outlives revocation."""
        key = (
            await user_client.post(f"{API}/api-keys", json={"name": "bot"})
        ).json()["data"]["key"]

        # `client` is anonymous; presenting the key alone must not authenticate
        # a management route.
        response = await client.post(
            f"{API}/api-keys", json={"name": "sneaky"}, headers={"X-API-Key": key}
        )
        assert response.status_code == 401


class TestAuthenticating:
    async def _issue(self, user_client: httpx.AsyncClient, **body: object) -> str:
        response = await user_client.post(
            f"{API}/api-keys", json={"name": "observer bot", **body}
        )
        assert response.status_code == 201
        return response.json()["data"]["key"]

    async def test_verify_reports_what_the_key_may_do(
        self, client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        key = await self._issue(user_client, agent_id=str(seed.agent_id))
        response = await client.get(
            f"{API}/api-keys/verify", headers={"X-API-Key": key}
        )
        assert response.status_code == 200
        identity = response.json()["data"]
        assert identity["user_id"] == str(seed.user_id)
        assert identity["agent_id"] == str(seed.agent_id)
        assert identity["name"] == "observer bot"
        assert ApiKeyScope.EXECUTIONS_WRITE in identity["scopes"]

    async def test_missing_header_is_rejected(self, client: httpx.AsyncClient):
        response = await client.get(f"{API}/api-keys/verify")
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "unauthorized"
        assert response.headers["WWW-Authenticate"] == "X-API-Key"

    @pytest.mark.parametrize(
        "candidate",
        ["nonsense", "cbrl_aabbccddeeff_wrongsecret", "cbrl_bad_x"],
    )
    async def test_bad_keys_are_rejected(
        self, client: httpx.AsyncClient, candidate: str
    ):
        response = await client.get(
            f"{API}/api-keys/verify", headers={"X-API-Key": candidate}
        )
        assert response.status_code == 401

    async def test_unknown_and_malformed_keys_look_identical(
        self, client: httpx.AsyncClient
    ):
        """A distinguishable answer would confirm a prefix exists."""
        unknown = await client.get(
            f"{API}/api-keys/verify",
            headers={"X-API-Key": "cbrl_aabbccddeeff_notarealsecretvalue"},
        )
        malformed = await client.get(
            f"{API}/api-keys/verify", headers={"X-API-Key": "nonsense"}
        )
        assert unknown.status_code == malformed.status_code == 401
        assert unknown.json()["error"]["code"] == malformed.json()["error"]["code"]

    async def test_a_revoked_key_stops_working(
        self, client: httpx.AsyncClient, user_client: httpx.AsyncClient
    ):
        created = (
            await user_client.post(f"{API}/api-keys", json={"name": "bot"})
        ).json()["data"]
        key, key_id = created["key"], created["api_key"]["id"]

        assert (
            await client.get(f"{API}/api-keys/verify", headers={"X-API-Key": key})
        ).status_code == 200

        revoked = await user_client.delete(f"{API}/api-keys/{key_id}")
        assert revoked.status_code == 200
        assert revoked.json()["data"]["revoked_at"] is not None

        after = await client.get(
            f"{API}/api-keys/verify", headers={"X-API-Key": key}
        )
        assert after.status_code == 401
        assert "revoked" in after.json()["error"]["message"].lower()

    async def test_revocation_is_idempotent_and_keeps_the_first_time(
        self, user_client: httpx.AsyncClient
    ):
        key_id = (
            await user_client.post(f"{API}/api-keys", json={"name": "bot"})
        ).json()["data"]["api_key"]["id"]

        first = await user_client.delete(f"{API}/api-keys/{key_id}")
        second = await user_client.delete(f"{API}/api-keys/{key_id}")
        assert second.status_code == 200
        assert second.json()["data"]["revoked_at"] == first.json()["data"]["revoked_at"]

    async def test_an_expired_key_stops_working(
        self,
        client: httpx.AsyncClient,
        user_client: httpx.AsyncClient,
        engine,
        seed: Seed,
    ):
        created = (
            await user_client.post(f"{API}/api-keys", json={"name": "bot"})
        ).json()["data"]
        key, key_id = created["key"], uuid.UUID(created["api_key"]["id"])

        # Backdate the whole row rather than sleeping. Creation has to move too:
        # ck_api_keys_expires_after_created forbids expires_at <= created_at, so
        # a key cannot be born expired -- it has to have aged into it.
        # The id must be a real UUID, not the string it arrived as in JSON, or
        # asyncpg binds it as varchar and Postgres has no uuid = varchar operator.
        async with AsyncSession(engine) as session:
            connection = await session.connection()
            await connection.execute(
                sa.text(
                    "UPDATE api_keys"
                    "   SET created_at = now() - interval '2 hours',"
                    "       expires_at = now() - interval '1 hour'"
                    " WHERE id = :id"
                ).bindparams(id=key_id)
            )
            await session.commit()

        response = await client.get(
            f"{API}/api-keys/verify", headers={"X-API-Key": key}
        )
        assert response.status_code == 401
        assert "expired" in response.json()["error"]["message"].lower()

    async def test_last_used_is_recorded(
        self, client: httpx.AsyncClient, user_client: httpx.AsyncClient
    ):
        created = (
            await user_client.post(f"{API}/api-keys", json={"name": "bot"})
        ).json()["data"]
        assert created["api_key"]["last_used_at"] is None

        await client.get(f"{API}/api-keys/verify", headers={"X-API-Key": created["key"]})

        fetched = await user_client.get(f"{API}/api-keys/{created['api_key']['id']}")
        assert fetched.json()["data"]["last_used_at"] is not None


class TestScopes:
    async def test_scopes_round_trip_through_the_principal(
        self, engine: AsyncEngine, seed: Seed
    ):
        store = ApiKeyStore(engine)
        _, api_key = await store.create(
            user_id=seed.user_id,
            name="narrow",
            agent_id=None,
            scopes=[ApiKeyScope.EXECUTIONS_READ],
        )
        principal = Principal.for_api_key(api_key)
        assert principal.has_scope(ApiKeyScope.EXECUTIONS_READ)
        assert not principal.has_scope(ApiKeyScope.EXECUTIONS_WRITE)
        assert principal.kind == "api_key"
        assert principal.user_id == seed.user_id

    async def test_a_user_principal_is_not_scope_limited(self, seed: Seed):
        """Scopes narrow a credential left on a machine; a person's session
        already carries their full authority."""
        principal = Principal.for_user(
            User(id=seed.user_id, name="aslon", hashed_password="x")
        )
        assert all(principal.has_scope(scope) for scope in ApiKeyScope)
        assert principal.agent_id is None
        assert principal.key_id is None


class TestIsolation:
    async def test_keys_are_scoped_to_their_owner(
        self, client: httpx.AsyncClient, user_client: httpx.AsyncClient, seed: Seed
    ):
        key_id = (
            await user_client.post(f"{API}/api-keys", json={"name": "mine"})
        ).json()["data"]["api_key"]["id"]

        other_token = create_access_token(seed.other_user_id)
        headers = {"Authorization": f"Bearer {other_token}"}

        assert (
            await client.get(f"{API}/api-keys/{key_id}", headers=headers)
        ).status_code == 404
        assert (
            await client.delete(f"{API}/api-keys/{key_id}", headers=headers)
        ).status_code == 404
        listed = (await client.get(f"{API}/api-keys", headers=headers)).json()
        assert listed["data"]["total"] == 0

    async def test_revoked_keys_are_hidden_but_retained(
        self, user_client: httpx.AsyncClient
    ):
        """The row survives revocation so an audit of what the key did still
        resolves."""
        key_id = (
            await user_client.post(f"{API}/api-keys", json={"name": "bot"})
        ).json()["data"]["api_key"]["id"]
        await user_client.delete(f"{API}/api-keys/{key_id}")

        default = (await user_client.get(f"{API}/api-keys")).json()
        assert default["data"]["total"] == 0

        including = (
            await user_client.get(f"{API}/api-keys?include_revoked=true")
        ).json()
        assert including["data"]["total"] == 1
        assert including["data"]["items"][0]["id"] == key_id
