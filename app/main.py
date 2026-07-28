from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.db import init_db
from app.repo.base import (
    DuplicateLabelNameError,
    InvalidDateRangeError,
    UnknownLabelsError,
)
from app.routes import api_router

try:
    import sentry_sdk
except ModuleNotFoundError:  # sentry is optional / not always installed
    sentry_sdk = None

if sentry_sdk and settings.sentry_dsn and settings.ENVIRONMENT != "local":
    sentry_sdk.init(dsn=str(settings.sentry_dsn), enable_tracing=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(debug=True, lifespan=lifespan)
app.include_router(api_router)


# Repositories raise domain errors instead of HTTP ones; they are translated
# here so every route reports them the same way.
@app.exception_handler(UnknownLabelsError)
async def unknown_labels_handler(
    request: Request, exc: UnknownLabelsError
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)}
    )


@app.exception_handler(DuplicateLabelNameError)
async def duplicate_label_handler(
    request: Request, exc: DuplicateLabelNameError
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT, content={"detail": str(exc)}
    )


@app.exception_handler(InvalidDateRangeError)
async def invalid_date_range_handler(
    request: Request, exc: InvalidDateRangeError
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={"detail": str(exc)}
    )
