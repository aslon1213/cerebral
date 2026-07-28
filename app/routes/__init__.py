from fastapi import APIRouter

from app.routes import auth, labels, projects, tasks

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(auth.router)
api_router.include_router(projects.router)
api_router.include_router(tasks.router)
api_router.include_router(labels.router)
