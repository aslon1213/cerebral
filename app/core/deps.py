import uuid
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.db import get_engine
from app.core.security import ACCESS_TOKEN_TYPE, decode_token
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


UserRepoDep = Annotated[UserRepo, Depends(get_user_repo)]
TokenRepoDep = Annotated[RefreshTokenRepo, Depends(get_token_repo)]
ProjectRepoDep = Annotated[ProjectRepo, Depends(get_project_repo)]
TaskRepoDep = Annotated[TaskRepo, Depends(get_task_repo)]
LabelRepoDep = Annotated[LabelRepo, Depends(get_label_repo)]

_bearer = HTTPBearer(auto_error=True)


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer)],
    users: UserRepoDep,
) -> User:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_token(credentials.credentials)
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


CurrentUser = Annotated[User, Depends(get_current_user)]


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


async def require_task_access(
    task_id: uuid.UUID, current_user: CurrentUser, tasks: TaskRepoDep
) -> uuid.UUID:
    owner_id = await tasks.get_owner(task_id)
    if owner_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found",
        )
    if owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the user who created this task can access it",
        )
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


ProjectAccess = Annotated[uuid.UUID, Depends(require_project_access)]
TaskAccess = Annotated[uuid.UUID, Depends(require_task_access)]
LabelAccess = Annotated[uuid.UUID, Depends(require_label_access)]
