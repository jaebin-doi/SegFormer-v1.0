"""ORM 모델 — alembic 가 metadata 를 발견하려면 여기서 모두 import 해야 한다."""
from src.api.models.model import Model

__all__ = ["Model"]
