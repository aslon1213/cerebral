"""Shared fixtures for the database tests.

These run against a real Postgres, not a mock or SQLite. The whole point of the
execution schema is its constraints -- partial unique indexes, composite foreign
keys, JSONB checks, RESTRICT chains -- and none of those exist anywhere else. A
test that mocked the database would pass against a schema Postgres refuses.
"""

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx
import pytest
import pytest_asyncio
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

import app.repo  # noqa: F401  -- registers every table on SQLModel.metadata
from app.core.db import get_engine
from app.core.security import create_access_token
from app.main import app as fastapi_app
from app.repo.agent import Agent
from app.repo.api_keys import OBSERVER_BOT_SCOPES, ApiKeyStore
from app.repo.git_repo import GitRepo
from app.repo.project import Project
from app.repo.task import Task
from app.repo.users import User

API = "/api/v1"

TEST_DSN = "postgresql+asyncpg://postgres:postgres@localhost:55432/cerebral_test"
# Connect here to issue CREATE DATABASE, since you cannot create a database
# from inside a connection to it.
ADMIN_DSN = "postgresql+asyncpg://postgres:postgres@localhost:55432/postgres"


async def _ensure_database() -> None:
    """Create cerebral_test if this is the first run."""
    admin = create_async_engine(ADMIN_DSN, isolation_level="AUTOCOMMIT")
    try:
        async with admin.connect() as conn:
            exists = (
                await conn.execute(
                    sa.text("SELECT 1 FROM pg_database WHERE datname = 'cerebral_test'")
                )
            ).scalar()
            if not exists:
                await conn.exec_driver_sql("CREATE DATABASE cerebral_test")
    finally:
        await admin.dispose()


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def engine() -> AsyncIterator[AsyncEngine]:
    """One engine per test session, with the schema built up and torn down.

    The schema is created directly from the models rather than by running
    Alembic: these tests check that the models are coherent, so making them
    depend on the migration would test two things at once and fail confusingly
    when only one is broken.
    """
    await _ensure_database()
    engine = create_async_engine(TEST_DSN)

    async with engine.begin() as conn:
        # Start from nothing: a leftover table from an older shape of the
        # schema would otherwise survive and quietly break create_all.
        await conn.run_sync(SQLModel.metadata.drop_all)
        await conn.run_sync(SQLModel.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture(loop_scope="session", autouse=True)
async def _clean_tables(engine: AsyncEngine) -> AsyncIterator[None]:
    """Empty every table before each test.

    TRUNCATE ... CASCADE rather than DELETE: the schema is a web of RESTRICT
    foreign keys, so row-by-row deletion would need the exact reverse
    dependency order, and any new table would silently start leaking rows
    between tests.
    """
    tables = ", ".join(f'"{table.name}"' for table in SQLModel.metadata.sorted_tables)
    async with engine.begin() as conn:
        await conn.exec_driver_sql(f"TRUNCATE {tables} RESTART IDENTITY CASCADE")
    yield


@pytest_asyncio.fixture(loop_scope="session")
async def session(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    async with AsyncSession(engine) as session:
        yield session


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
@pytest_asyncio.fixture(loop_scope="session")
async def client(engine: AsyncEngine) -> AsyncIterator[httpx.AsyncClient]:
    """The real app, over ASGI, pointed at the test database.

    No network and no live server: ASGITransport calls the app in-process, so
    these exercise the actual routing, dependencies, validation and error
    handlers rather than calling handler functions directly.
    """
    fastapi_app.dependency_overrides[get_engine] = lambda: engine
    transport = httpx.ASGITransport(app=fastapi_app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        yield client
    fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture(loop_scope="session")
async def user_client(
    engine: AsyncEngine, seed: "Seed"
) -> AsyncIterator[httpx.AsyncClient]:
    """A client carrying a real access token for the seed user.

    A genuinely signed JWT rather than a stubbed dependency, so the auth path
    under test is the one that runs in production.

    Its own client instance, not ``client`` with a header bolted on: a test
    that wants both an authenticated and an anonymous caller needs the second
    one to actually be anonymous.
    """
    fastapi_app.dependency_overrides[get_engine] = lambda: engine
    token = create_access_token(seed.user_id)
    transport = httpx.ASGITransport(app=fastapi_app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
        headers={"Authorization": f"Bearer {token}"},
    ) as client:
        yield client
    fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture(loop_scope="session")
async def bot_key(engine: AsyncEngine, seed: "Seed") -> str:
    """A real API key bound to the seed agent, with an observer bot's scopes.

    Issued through the store rather than the endpoint so a test that only wants
    to send events does not have to authenticate as a person first.
    """
    store = ApiKeyStore(engine)
    key, _ = await store.create(
        user_id=seed.user_id,
        name="observer bot",
        agent_id=seed.agent_id,
        scopes=[*OBSERVER_BOT_SCOPES],
    )
    return key


@pytest_asyncio.fixture(loop_scope="session")
async def bot_client(
    engine: AsyncEngine, bot_key: str
) -> AsyncIterator[httpx.AsyncClient]:
    """A client authenticating the way the observer bot does: an API key.

    Its own instance, like ``user_client``: the point of several clients is that
    a test can hold a bot, a person and an anonymous caller at once.
    """
    fastapi_app.dependency_overrides[get_engine] = lambda: engine
    transport = httpx.ASGITransport(app=fastapi_app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
        headers={"X-API-Key": bot_key},
    ) as client:
        yield client
    fastapi_app.dependency_overrides.clear()


@dataclass(frozen=True)
class Seed:
    """Ids of the baseline rows every test starts from.

    Ids only, never ORM instances: ``commit()`` expires an instance, and reading
    an attribute off an expired object outside its session raises
    DetachedInstanceError.
    """

    user_id: uuid.UUID
    other_user_id: uuid.UUID
    project_id: uuid.UUID
    task_id: uuid.UUID
    agent_id: uuid.UUID
    api_repo_id: uuid.UUID
    web_repo_id: uuid.UUID
    docs_repo_id: uuid.UUID  # deliberately never attached to an execution


@pytest_asyncio.fixture(loop_scope="session")
async def seed(engine: AsyncEngine) -> Seed:
    """The mock-up data: one user, project and task, an agent, and three repos."""
    user = User(name="aslon", hashed_password="argon2:x")
    other = User(name="someone-else", hashed_password="argon2:y")
    async with AsyncSession(engine) as s:
        s.add_all([user, other])
        ids = (user.id, other.id)
        await s.commit()
    user_id, other_user_id = ids

    project = Project(created_by=user_id, name="cerebral")
    project_id = project.id
    async with AsyncSession(engine) as s:
        s.add(project)
        await s.commit()

    task = Task(project_id=project_id, name="add jwt auth", created_by=user_id)
    agent = Agent(created_by=user_id, name="claude-code", default_model="claude-opus-5")
    api = GitRepo(created_by=user_id, name="api", local_path="/src/api")
    web = GitRepo(created_by=user_id, name="web", local_path="/src/web")
    docs = GitRepo(created_by=user_id, name="docs", local_path="/src/docs")
    ids = (task.id, agent.id, api.id, web.id, docs.id)
    async with AsyncSession(engine) as s:
        s.add_all([task, agent, api, web, docs])
        await s.commit()
    task_id, agent_id, api_id, web_id, docs_id = ids

    return Seed(
        user_id=user_id,
        other_user_id=other_user_id,
        project_id=project_id,
        task_id=task_id,
        agent_id=agent_id,
        api_repo_id=api_id,
        web_repo_id=web_id,
        docs_repo_id=docs_id,
    )


@pytest.fixture
def blob() -> "BlobFactory":
    return BlobFactory()


class BlobFactory:
    """Deterministic, distinct 40-hex git object ids."""

    def __init__(self) -> None:
        self._n = 0

    def __call__(self) -> str:
        self._n += 1
        return f"{self._n:040x}"
