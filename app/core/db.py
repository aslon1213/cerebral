from sqlmodel import SQLModel
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


from app.core.config import settings

engine: AsyncEngine = create_async_engine(
    str(settings.pg.dsn),
    echo=False,
    pool_pre_ping=True,
)


def get_engine() -> AsyncEngine:
    """FastAPI dependency that returns the shared async engine."""
    return engine


async def init_db() -> None:
    """Create tables for all registered models.

    Pragmatic for development; use Alembic migrations for production schema
    changes. Importing the repo modules registers their tables on the metadata.
    """
    from app.repo import labels, project, task, tokens, users  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
