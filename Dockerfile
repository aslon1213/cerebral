# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Railway's builder rejects cache mounts without an id, and only persists a
# cache whose id is prefixed with `s/<service id>-`. Railway passes the service
# id in as a build arg; locally the default just yields a stable literal id.
ARG RAILWAY_SERVICE_ID=local

# Dependency metadata only, so the (slow) dependency layer is cached and only
# rebuilt when the lockfile or a workspace member's manifest changes.
# libs/*/pyproject.toml and their READMEs are part of the uv workspace, so uv
# needs them present to resolve against the lock even before the sources land.
COPY pyproject.toml uv.lock ./
COPY libs/cerebral/pyproject.toml libs/cerebral/README.md ./libs/cerebral/
COPY libs/observer/pyproject.toml libs/observer/README.md ./libs/observer/

RUN --mount=type=cache,id=s/${RAILWAY_SERVICE_ID}-uv,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace

# Now the workspace sources, then install the workspace package itself
# (cerebral-lib). --no-editable bakes it into the venv as a real wheel, so the
# runtime image does not need libs/ on disk.
COPY libs/ ./libs/
RUN --mount=type=cache,id=s/${RAILWAY_SERVICE_ID}-uv,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# Runtime
FROM python:3.13-slim-bookworm AS runtime

# libpq5: psycopg (pure-python wheel) loads the system libpq at runtime.
# asyncpg needs nothing extra, but psycopg is a declared dependency and will
# fail to import without it.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libpq5 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv

# Application code. migrations/ and alembic.ini come along so the same image can
# run `alembic upgrade head` as a one-off job before rollout.
COPY --chown=app:app app/ ./app/
COPY --chown=app:app migrations/ ./migrations/
COPY --chown=app:app alembic.ini ./

USER app

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:'+os.environ['PORT']+'/openapi.json',timeout=4)"

# `fastapi run` binds :8000 unless told otherwise, so the port is passed
# explicitly — Railway injects its own PORT and routes to that, and the
# HEALTHCHECK above probes the same variable.
CMD ["sh", "-c", "exec fastapi run app/main.py --host 0.0.0.0 --port ${PORT}"]
