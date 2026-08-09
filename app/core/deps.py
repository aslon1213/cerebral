import uuid
from dataclasses import dataclass
from typing import Annotated, Literal

import jwt
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.db import get_engine
from app.core.security import ACCESS_TOKEN_TYPE, decode_token
from app.repo.agent import AgentStore
from app.repo.api_keys import ApiKey, ApiKeyScope, ApiKeyStore
from app.repo.execution import (
    CodeHistoryStore,
    EventStore,
    ExecutionStore,
    InterventionStore,
)
from app.repo.git_repo import GitRepoStore
from app.repo.labels import LabelRepo
from app.repo.project import ProjectRepo
from app.repo.task import TaskRepo
from app.repo.tokens import RefreshTokenRepo
from app.repo.users import User, UserRepo

EngineDep = Annotated[AsyncEngine, Depends(get_engine)]


def get_user_repo(engine: EngineDep) -> UserRepo:
    return UserRepo(engine)


def get_token_repo(engine: EngineDep) -> RefreshTokenRepo:
    return RefreshTokenRepo(engine)


def get_project_repo(engine: EngineDep) -> ProjectRepo:
    return ProjectRepo(engine)


def get_task_repo(engine: EngineDep) -> TaskRepo:
    return TaskRepo(engine)


def get_label_repo(engine: EngineDep) -> LabelRepo:
    return LabelRepo(engine)


def get_api_key_store(engine: EngineDep) -> ApiKeyStore:
    return ApiKeyStore(engine)


def get_agent_store(engine: EngineDep) -> AgentStore:
    return AgentStore(engine)


def get_git_repo_store(engine: EngineDep) -> GitRepoStore:
    return GitRepoStore(engine)


def get_execution_store(engine: EngineDep) -> ExecutionStore:
    return ExecutionStore(engine)


def get_event_store(engine: EngineDep) -> EventStore:
    return EventStore(engine)


def get_intervention_store(engine: EngineDep) -> InterventionStore:
    return InterventionStore(engine)


def get_code_history_store(engine: EngineDep) -> CodeHistoryStore:
    return CodeHistoryStore(engine)


UserRepoDep = Annotated[UserRepo, Depends(get_user_repo)]
TokenRepoDep = Annotated[RefreshTokenRepo, Depends(get_token_repo)]
ProjectRepoDep = Annotated[ProjectRepo, Depends(get_project_repo)]
TaskRepoDep = Annotated[TaskRepo, Depends(get_task_repo)]
LabelRepoDep = Annotated[LabelRepo, Depends(get_label_repo)]
ApiKeyStoreDep = Annotated[ApiKeyStore, Depends(get_api_key_store)]
AgentStoreDep = Annotated[AgentStore, Depends(get_agent_store)]
GitRepoStoreDep = Annotated[GitRepoStore, Depends(get_git_repo_store)]
ExecutionStoreDep = Annotated[ExecutionStore, Depends(get_execution_store)]
EventStoreDep = Annotated[EventStore, Depends(get_event_store)]
InterventionStoreDep = Annotated[InterventionStore, Depends(get_intervention_store)]
CodeHistoryStoreDep = Annotated[CodeHistoryStore, Depends(get_code_history_store)]

_bearer = HTTPBearer(auto_error=True)
# The same scheme without the automatic rejection, for the combined dependency
# below: there, a missing bearer token is not yet a failure because an API key
# may be carrying the request instead.
_optional_bearer = HTTPBearer(auto_error=False)


async def user_from_access_token(token: str, users: UserRepo) -> User:
    """Resolve an access token to its user, or raise 401.

    Shared by the bearer-only dependency and the combined one, so both accept
    exactly the same tokens.
    """
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_token(token)
    except jwt.PyJWTError:
        raise unauthorized

    if payload.get("type") != ACCESS_TOKEN_TYPE:
        raise unauthorized

    sub = payload.get("sub")
    if sub is None:
        raise unauthorized

    try:
        user_id = uuid.UUID(str(sub))
    except ValueError:
        raise unauthorized

    user = await users.get_by_id(user_id)
    if user is None or not user.is_active:
        raise unauthorized
    return user


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer)],
    users: UserRepoDep,
) -> User:
    return await user_from_access_token(credentials.credentials, users)


CurrentUser = Annotated[User, Depends(get_current_user)]


# --------------------------------------------------------------------------- #
# API keys — how the observer bot authenticates
# --------------------------------------------------------------------------- #
# auto_error=False so a missing header reaches our own handler and comes back as
# the Response envelope with an api-key-specific message, rather than FastAPI's
# generic 403.
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@dataclass(frozen=True)
class Principal:
    """Whoever is making this request, however they authenticated.

    Both a person with a JWT and a bot with an API key act *on behalf of* one
    user, so ``user_id`` is always present and every ownership check downstream
    reads the same field regardless of how the caller got here.
    """

    user_id: uuid.UUID
    kind: Literal["user", "api_key"]
    # Set only for an API key bound to an agent. Executions started by that key
    # inherit it as executor_agent_id, so a bot cannot claim to be an agent it
    # was not issued for.
    agent_id: uuid.UUID | None = None
    key_id: uuid.UUID | None = None
    scopes: frozenset[ApiKeyScope] = frozenset()

    @classmethod
    def for_user(cls, user: User) -> "Principal":
        # A person is not scope-limited: scopes exist to narrow what a
        # credential left on a machine can do, and a session token already
        # carries the user's full authority.
        return cls(user_id=user.id, kind="user", scopes=frozenset(ApiKeyScope))

    @classmethod
    def for_api_key(cls, api_key: ApiKey) -> "Principal":
        return cls(
            user_id=api_key.created_by,
            kind="api_key",
            agent_id=api_key.agent_id,
            key_id=api_key.id,
            scopes=frozenset(ApiKeyScope(scope) for scope in api_key.scopes),
        )

    def has_scope(self, scope: ApiKeyScope) -> bool:
        return scope in self.scopes


def _api_key_error(message: str) -> HTTPException:
    # 401 across the board, so the envelope's code is `unauthorized`; which of
    # the credential problems it was lives in the message. 403 is reserved for
    # a key that authenticated but lacks the scope.
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=message,
        headers={"WWW-Authenticate": "X-API-Key"},
    )


async def get_optional_api_key_principal(
    presented: Annotated[str | None, Security(_api_key_header)],
    keys: ApiKeyStoreDep,
) -> Principal | None:
    """Authenticate an X-API-Key header, or None if there was not one.

    Only an absent header returns None. A header that is present but wrong is
    an error either way -- a caller who sent a key meant to use it, and quietly
    falling through to another credential would hide a broken deployment.

    A malformed, unknown or wrong key is one indistinguishable answer: telling
    a caller which of the three it was would confirm that a prefix exists.
    Revoked and expired are reported separately, because reaching either means
    presenting a key whose secret already verified -- the holder is entitled to
    know why it stopped working.
    """
    if not presented:
        return None

    api_key = await keys.authenticate(presented)
    if api_key is None:
        raise _api_key_error("That API key is not valid.")

    if api_key.revoked_at is not None:
        raise _api_key_error("That API key has been revoked. Issue a new one.")
    if not api_key.is_usable():
        raise _api_key_error("That API key has expired. Issue a new one.")

    await keys.touch(api_key.id)
    return Principal.for_api_key(api_key)


async def get_api_key_principal(
    principal: Annotated[Principal | None, Depends(get_optional_api_key_principal)],
) -> Principal:
    """The caller's API key. For routes only a bot credential may reach."""
    if principal is None:
        raise _api_key_error("This endpoint needs an API key in the X-API-Key header.")
    return principal


async def get_principal(
    api_key_principal: Annotated[
        Principal | None, Depends(get_optional_api_key_principal)
    ],
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(_optional_bearer)
    ],
    users: UserRepoDep,
) -> Principal:
    """Either credential, for the routes a person and a bot both read.

    An execution is written by a bot and read by the person who owns it, so the
    read side cannot insist on one or the other. The key wins when both are
    present: a request carrying an API key is the bot's request.

    Both dependencies resolve ``get_optional_api_key_principal``, and FastAPI
    caches a dependency per request, so a route that asks for this *and* for
    ``ApiKeyAuth`` still authenticates the key once.
    """
    if api_key_principal is not None:
        return api_key_principal
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "This endpoint needs either an API key in the X-API-Key header "
                "or a bearer token."
            ),
            headers={"WWW-Authenticate": "Bearer"},
        )
    return Principal.for_user(
        await user_from_access_token(credentials.credentials, users)
    )


ApiKeyAuth = Annotated[Principal, Depends(get_api_key_principal)]
AnyAuth = Annotated[Principal, Depends(get_principal)]


def _assert_scopes(principal: Principal, required: tuple[ApiKeyScope, ...]) -> None:
    missing = [scope for scope in required if not principal.has_scope(scope)]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "This API key is not allowed to do that. Missing scope: "
                + ", ".join(sorted(scope.value for scope in missing))
            ),
        )


def require_scopes(*required: ApiKeyScope):
    """A dependency asserting the caller's API key carries every scope.

    Use for the routes a bot writes through, which take no other credential::

        principal: Annotated[Principal, Depends(
            require_scopes(ApiKeyScope.EXECUTIONS_WRITE)
        )]
    """

    async def dependency(principal: ApiKeyAuth) -> Principal:
        _assert_scopes(principal, required)
        return principal

    return dependency


def require_any_scopes(*required: ApiKeyScope):
    """``require_scopes`` for a route either credential may reach.

    A person's principal carries every scope, so this only ever narrows what a
    key may do -- which is the point of scopes: they limit a credential left on
    a machine, not the user behind it.
    """

    async def dependency(principal: AnyAuth) -> Principal:
        _assert_scopes(principal, required)
        return principal

    return dependency


# The credential each side of the execution API takes, resolved once here so
# every route module names the same dependency rather than building its own.
#
# Writes are the ingest path and are API-key only: every event in a transcript
# has to be attributable to a credential that was issued for an agent. Reads are
# the same data a person looks at afterwards, so they take either.
ExecutionsWrite = Annotated[
    Principal, Depends(require_scopes(ApiKeyScope.EXECUTIONS_WRITE))
]
ExecutionsRead = Annotated[
    Principal, Depends(require_any_scopes(ApiKeyScope.EXECUTIONS_READ))
]
ExecutionsWriteAny = Annotated[
    Principal, Depends(require_any_scopes(ApiKeyScope.EXECUTIONS_WRITE))
]
# Only the file-history reads take this. The rest of the /repos router is a
# person managing their connected repos, over a session token.
ReposRead = Annotated[
    Principal, Depends(require_any_scopes(ApiKeyScope.REPOS_READ))
]


# Ownership checks. Every project, task and label belongs to the user that
# created it; nobody else may read or write it. The checks below run before the
# route body so the repos are only ever asked for rows the caller owns.
async def ensure_project_owner(
    projects: ProjectRepo, project_id: uuid.UUID, user_id: uuid.UUID
) -> None:
    """Raise unless ``user_id`` created the project.

    Used both by the path dependency below and by the task routes, where the
    project is named in the request body.
    """
    owner_id = await projects.get_owner(project_id)
    if owner_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )
    if owner_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the user who created this project can access it",
        )


async def require_project_access(
    project_id: uuid.UUID, current_user: CurrentUser, projects: ProjectRepoDep
) -> uuid.UUID:
    await ensure_project_owner(projects, project_id, current_user.id)
    return project_id


async def ensure_task_owner(
    tasks: TaskRepo, task_id: uuid.UUID, user_id: uuid.UUID
) -> None:
    """Raise unless ``user_id`` created the task.

    Used both by the path dependency below and by the execution routes, where
    the task is named in the request body and the caller may be a bot rather
    than a logged-in user.
    """
    owner_id = await tasks.get_owner(task_id)
    if owner_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found",
        )
    if owner_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the user who created this task can access it",
        )


async def require_task_access(
    task_id: uuid.UUID, current_user: CurrentUser, tasks: TaskRepoDep
) -> uuid.UUID:
    await ensure_task_owner(tasks, task_id, current_user.id)
    return task_id


async def require_label_access(
    label_id: uuid.UUID, current_user: CurrentUser, labels: LabelRepoDep
) -> uuid.UUID:
    owner_id = await labels.get_owner(label_id)
    if owner_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Label not found",
        )
    if owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the user who created this label can access it",
        )
    return label_id


async def require_agent_access(
    agent_id: uuid.UUID, current_user: CurrentUser, agents: AgentStoreDep
) -> uuid.UUID:
    owner_id = await agents.get_owner(agent_id)
    if owner_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found",
        )
    if owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the user who created this agent can access it",
        )
    return agent_id


async def require_git_repo_access(
    repo_id: uuid.UUID, current_user: CurrentUser, repos: GitRepoStoreDep
) -> uuid.UUID:
    owner_id = await repos.get_owner(repo_id)
    if owner_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repo not found",
        )
    if owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the user who connected this repo can access it",
        )
    return repo_id


async def require_execution_access(
    execution_id: uuid.UUID, principal: AnyAuth, executions: ExecutionStoreDep
) -> uuid.UUID:
    """Ownership for an execution, whichever credential is carrying the request.

    Reads on an execution come from a person and writes come from a bot acting
    for that same person, so this resolves the caller through ``AnyAuth`` and
    compares against ``principal.user_id`` either way.
    """
    owner_id = await executions.get_owner(execution_id)
    if owner_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Execution not found",
        )
    if owner_id != principal.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the user this execution belongs to can access it",
        )
    return execution_id


async def require_git_repo_read_access(
    repo_id: uuid.UUID, principal: AnyAuth, repos: GitRepoStoreDep
) -> uuid.UUID:
    """Ownership for a repo, over either credential.

    Only the file-history reads use this; the rest of the /repos router is
    JWT-only management. A bot holding executions:read can already reach every
    code change through its execution, so refusing it the same rows addressed
    by repo would protect nothing.
    """
    owner_id = await repos.get_owner(repo_id)
    if owner_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repo not found",
        )
    if owner_id != principal.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the user who connected this repo can access it",
        )
    return repo_id


async def require_intervention_access(
    intervention_id: uuid.UUID,
    current_user: CurrentUser,
    interventions: InterventionStoreDep,
) -> uuid.UUID:
    """Ownership for an intervention, over a JWT only.

    Deliberately built on ``CurrentUser`` rather than ``AnyAuth``: answering one
    is a human decision, and no bot credential should reach a route that makes
    it. See OBSERVER_BOT_SCOPES, which omits any scope that would allow it.
    """
    owner_id = await interventions.get_owner(intervention_id)
    if owner_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Intervention not found",
        )
    if owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the user this execution belongs to can answer it",
        )
    return intervention_id


ProjectAccess = Annotated[uuid.UUID, Depends(require_project_access)]
TaskAccess = Annotated[uuid.UUID, Depends(require_task_access)]
LabelAccess = Annotated[uuid.UUID, Depends(require_label_access)]
AgentAccess = Annotated[uuid.UUID, Depends(require_agent_access)]
GitRepoAccess = Annotated[uuid.UUID, Depends(require_git_repo_access)]
GitRepoReadAccess = Annotated[uuid.UUID, Depends(require_git_repo_read_access)]
ExecutionAccess = Annotated[uuid.UUID, Depends(require_execution_access)]
InterventionAccess = Annotated[uuid.UUID, Depends(require_intervention_access)]
