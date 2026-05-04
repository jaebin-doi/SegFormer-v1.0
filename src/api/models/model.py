"""모델 레지스트리 테이블 — m5_model_performance.html 가 표시하는 행.

스키마 컬럼은 m5 페이지의 mock generator (`generateModelData`) 가 만드는 형태와 1:1 대응.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Enum, Float, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.api.db import Base


class Backbone(str, enum.Enum):
    B0 = "B0"
    B2 = "B2"
    B3 = "B3"


class DeploymentState(str, enum.Enum):
    PRODUCTION = "production"
    STAGING = "staging"
    ARCHIVED = "archived"


class Model(Base):
    """학습 완료된 모델 한 버전."""

    __tablename__ = "models"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # 표시명 — "SegFormer-B2-v.1.0"
    version: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    backbone: Mapped[Backbone] = mapped_column(
        Enum(Backbone, name="backbone"), index=True
    )
    training_completed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True
    )

    # 학습 데이터 양
    training_sheets: Mapped[int] = mapped_column(Integer)  # 영상 매수
    # 픽셀 수 — 영상 1장당 2048×2048 ≈ 4.2M 이라 8000장 = 34B → BigInteger 필수
    training_pixels: Mapped[int] = mapped_column(BigInteger)

    # 평가 지표
    miou_overall: Mapped[float] = mapped_column(Float)
    miou_rice_paddy: Mapped[float] = mapped_column(Float)  # 논 (0)
    miou_dry_field: Mapped[float] = mapped_column(Float)  # 밭 (1)
    miou_orchard: Mapped[float] = mapped_column(Float)  # 과수 (2)
    miou_greenhouse: Mapped[float] = mapped_column(Float)  # 시설 (3)

    deployment_state: Mapped[DeploymentState] = mapped_column(
        Enum(DeploymentState, name="deployment_state"),
        default=DeploymentState.STAGING,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now()
    )
