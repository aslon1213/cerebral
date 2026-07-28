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
