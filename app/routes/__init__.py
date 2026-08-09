from fastapi import APIRouter

from app.core.response import error_responses
from app.routes import (
    agents,
    api_keys,
    auth,
    code,
    events,
    executions,
    git_repos,
    interventions,
    labels,
    projects,
    tasks,
)

# Every endpoint can fail validation or hit a bug, so both are documented once
# here; routers add the statuses only they can return. Declaring 422 also keeps
# FastAPI from documenting its own HTTPValidationError, which this app never
# sends — errors always come back as the Response envelope.
api_router = APIRouter(prefix="/api/v1", responses=error_responses(422, 500))
api_router.include_router(auth.router)
api_router.include_router(api_keys.router)
api_router.include_router(agents.router)
api_router.include_router(git_repos.router)
api_router.include_router(executions.router)
api_router.include_router(events.router)
api_router.include_router(interventions.execution_router)
api_router.include_router(interventions.router)
api_router.include_router(code.execution_router)
api_router.include_router(code.repo_router)
api_router.include_router(projects.router)
api_router.include_router(tasks.router)
api_router.include_router(labels.router)
