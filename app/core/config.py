from pydantic import AliasChoices, BaseModel, Field, ImportString, PostgresDsn, RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict

import os


class PostgreSqlDatabaseSettings(BaseSettings):
    dsn: PostgresDsn = "postgresql+asyncpg://admin:admin@localhost:5432/postgres"


class JwtSettings(BaseSettings):
    secret: str = "change-me-set-CEREBRAL_JWT__SECRET-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 30


class AppSettings(BaseSettings):
    port: int = Field(
        default=8080,
    )
    pg: PostgreSqlDatabaseSettings = PostgreSqlDatabaseSettings()
    jwt: JwtSettings = JwtSettings()
    redis_dsn: RedisDsn = Field(
        "redis://localhost:6379/1",
        # validation_alias=AliasChoices("service_redis_dsn", "redis_url"),
    )
    ENVIRONMENT: str = Field(
        "development",
        validation_alias=AliasChoices("CEREBRAL_ENVIRONMENT", "ENVIRONMENT"),
    )
    sentry_dsn: str = ""
    model_config = SettingsConfigDict(
        env_prefix="CEREBRAL_",
        env_file=os.getenv("ENV_FILE", ".env"),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        validate_default=True,
    )


settings = AppSettings()


if __name__ == "__main__":
    print(settings)
