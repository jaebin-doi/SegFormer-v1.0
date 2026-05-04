"""환경변수 기반 설정 — pydantic-settings 로 .env 자동 로드."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    database_url: str = Field(..., description="동기 SQLAlchemy URL (alembic 용)")
    database_url_async: str = Field(..., description="async SQLAlchemy URL (FastAPI 용)")
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_reload: bool = False
    cors_origins: str = "http://localhost:8000"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
