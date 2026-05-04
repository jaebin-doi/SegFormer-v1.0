"""SQLAlchemy 비동기 엔진 + 세션 팩토리.

`Base` 는 모든 ORM 모델의 공통 부모. alembic env.py 가 `Base.metadata` 를 읽어
autogenerate 마이그레이션을 만든다.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from src.api.config import get_settings


class Base(DeclarativeBase):
    pass


_settings = get_settings()
engine = create_async_engine(
    _settings.database_url_async,
    echo=False,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI Depends 용 세션 컨텍스트."""
    async with AsyncSessionLocal() as session:
        yield session
