from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict
from sqlalchemy import Column, DateTime
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import Field, SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession
import uuid


# --------------------------------------------------------------------------- #
# Table model
# --------------------------------------------------------------------------- #
class User(SQLModel, table=True):
    __tablename__ = "users"
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str = Field(index=True, unique=True)
    hashed_password: str
    secret_name: str | None = None
    age: int | None = None
    is_active: bool = Field(default=True)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #
class RegisterRequest(BaseModel):
    name: str
    password: str


class LoginRequest(BaseModel):
    name: str
    password: str


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    is_active: bool
    created_at: datetime


# --------------------------------------------------------------------------- #
# Repository
# --------------------------------------------------------------------------- #
class UserRepo:
    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    async def create(self, user: User) -> User:
        async with AsyncSession(self.engine) as session:
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user

    async def get_by_id(self, id: uuid.UUID) -> User | None:
        async with AsyncSession(self.engine) as session:
            statement = select(User).where(User.id == id)
            return (await session.exec(statement)).first()

    async def get_by_name(self, name: str) -> User | None:
        async with AsyncSession(self.engine) as session:
            statement = select(User).where(User.name == name)
            return (await session.exec(statement)).first()
