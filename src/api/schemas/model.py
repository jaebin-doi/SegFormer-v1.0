"""m5_model_performance.html 가 기대하는 응답 스키마."""
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from src.api.models.model import Backbone, DeploymentState


class ModelOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    version: str
    backbone: Backbone
    training_completed_at: datetime
    training_sheets: int
    training_pixels: int
    miou_overall: float
    miou_rice_paddy: float
    miou_dry_field: float
    miou_orchard: float
    miou_greenhouse: float
    deployment_state: DeploymentState
    created_at: datetime


class ModelPerformancePage(BaseModel):
    """페이지네이션된 모델 목록 응답."""

    items: list[ModelOut]
    total: int
    page: int
    limit: int


class ModelStateUpdate(BaseModel):
    state: DeploymentState
