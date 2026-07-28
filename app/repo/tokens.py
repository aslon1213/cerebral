import uuid
from datetime import datetime, timezone

from pydantic import BaseModel
from sqlalchemy import Column, DateTime
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import Field, SQLModel, col, select
from sqlmodel.ext.asyncio.session import AsyncSession


# --------------------------------------------------------------------------- #
# Table model
# --------------------------------------------------------------------------- #
class RefreshToken(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    jti: str = Field(index=True, unique=True)
    user_id: uuid.UUID = Field(index=True, foreign_key="users.id")
    expires_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    revoked: bool = Field(default=False)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #
class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str


# --------------------------------------------------------------------------- #
# Repository
# --------------------------------------------------------------------------- #
class RefreshTokenRepo:
    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    async def create(
        self, jti: str, user_id: uuid.UUID, expires_at: datetime
    ) -> RefreshToken:
        token = RefreshToken(jti=jti, user_id=user_id, expires_at=expires_at)
        async with AsyncSession(self.engine) as session:
            session.add(token)
            await session.commit()
            await session.refresh(token)
            return token

    async def get_by_jti(self, jti: str) -> RefreshToken | None:
        async with AsyncSession(self.engine) as session:
            statement = select(RefreshToken).where(RefreshToken.jti == jti)
            return (await session.exec(statement)).first()

    async def revoke(self, jti: str) -> None:
        async with AsyncSession(self.engine) as session:
            statement = select(RefreshToken).where(RefreshToken.jti == jti)
            token = (await session.exec(statement)).first()
            if token is not None and not token.revoked:
                token.revoked = True
                session.add(token)
                await session.commit()

    async def revoke_all_for_user(self, user_id: uuid.UUID) -> None:
        async with AsyncSession(self.engine) as session:
            statement = select(RefreshToken).where(
                RefreshToken.user_id == user_id,
                col(RefreshToken.revoked).is_(False),
            )
            tokens = (await session.exec(statement)).all()
            for token in tokens:
                token.revoked = True
                session.add(token)
            if tokens:
                await session.commit()
