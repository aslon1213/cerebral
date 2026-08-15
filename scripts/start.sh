#!/bin/sh
# Container entrypoint: bring the database up to date, then serve the API.
#
# Migrations run in the foreground and `set -e` stops us here if they fail, so a
# release that cannot migrate never starts serving against a stale schema.
# alembic/env.py reads the DSN from CEREBRAL_PG__DSN via app.core.config, so
# nothing needs to be passed in here.
set -e

if [ "${SKIP_MIGRATIONS:-0}" = "1" ]; then
    echo "start.sh: SKIP_MIGRATIONS=1, leaving the schema alone"
else
    echo "start.sh: running alembic upgrade head"
    alembic upgrade head
    echo "start.sh: migrations up to date"
fi

# exec, so uvicorn replaces this shell as PID 1 and receives SIGTERM directly on
# shutdown instead of being killed after the grace period.
echo "start.sh: starting API on port ${PORT:-8080}"
exec fastapi run app/main.py --host 0.0.0.0 --port "${PORT:-8080}"
