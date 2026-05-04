"""Alembic 환경 — settings.database_url (sync, psycopg) 와 Base.metadata 주입.

비동기 엔진 (asyncpg) 은 FastAPI 만 사용. alembic 은 동기 psycopg 드라이버 사용.
"""
from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool

from alembic import context

# repo root 를 sys.path 에 추가 — `src.api.*` 임포트용
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.api.config import get_settings  # noqa: E402
from src.api.db import Base  # noqa: E402
from src.api import models as _orm_models  # noqa: E402,F401  (ORM 등록을 위한 import)

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# .env 의 동기 URL 을 alembic 에 주입 (alembic.ini 의 sqlalchemy.url 무시)
config.set_main_option("sqlalchemy.url", get_settings().database_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
