from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.errors import register_exception_handlers
from app.core.tracing import RequestIdMiddleware
from app.routes import api_router

try:
    import sentry_sdk
except ModuleNotFoundError:  # sentry is optional / not always installed
    sentry_sdk = None

if sentry_sdk and settings.sentry_dsn and settings.ENVIRONMENT != "local":
    sentry_sdk.init(dsn=str(settings.sentry_dsn), enable_tracing=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # await init_db()
    yield


# debug stays off: with it on, Starlette answers unhandled errors with the
# traceback itself, and app.core.errors never gets to replace it with something
# the caller can safely be shown.
app = FastAPI(debug=False, lifespan=lifespan)

# Outermost of the two, so the trace is set before anything else runs and the
# id is on the response whichever handler produced it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    allow_credentials=True,
)
app.add_middleware(RequestIdMiddleware)

register_exception_handlers(app)
app.include_router(api_router)
