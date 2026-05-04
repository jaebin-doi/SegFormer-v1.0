"""모델 레지스트리 라우터 — m5_model_performance.html 백엔드."""
from __future__ import annotations

import csv
import io
import uuid
from datetime import datetime, timedelta
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.db import get_session
from src.api.models.model import Backbone, DeploymentState, Model
from src.api.schemas.model import ModelOut, ModelPerformancePage, ModelStateUpdate

router = APIRouter(prefix="/api/models", tags=["models"])

SortKey = Literal["latest", "miou", "data"]


def _apply_filters(
    stmt,
    *,
    date_start: datetime | None,
    date_end: datetime | None,
    backbone: Backbone | None,
    state: DeploymentState | None,
):
    if date_start is not None:
        stmt = stmt.where(Model.training_completed_at >= date_start)
    if date_end is not None:
        # 종료일은 그 날의 23:59:59 까지 포함
        stmt = stmt.where(Model.training_completed_at < date_end + timedelta(days=1))
    if backbone is not None:
        stmt = stmt.where(Model.backbone == backbone)
    if state is not None:
        stmt = stmt.where(Model.deployment_state == state)
    return stmt


def _apply_sort(stmt, sort: SortKey):
    if sort == "latest":
        return stmt.order_by(Model.training_completed_at.desc())
    if sort == "miou":
        return stmt.order_by(Model.miou_overall.desc())
    if sort == "data":
        return stmt.order_by(Model.training_sheets.desc())
    return stmt


@router.get("", response_model=list[ModelOut])
async def list_models(session: AsyncSession = Depends(get_session)) -> list[Model]:
    """드롭다운용 — 모든 모델 (필터/페이지네이션 없음)."""
    result = await session.execute(
        select(Model).order_by(Model.training_completed_at.desc())
    )
    return list(result.scalars().all())


@router.get("/performance", response_model=ModelPerformancePage)
async def models_performance(
    date_start: datetime | None = Query(None),
    date_end: datetime | None = Query(None),
    backbone: Backbone | None = Query(None),
    state: DeploymentState | None = Query(None),
    sort: SortKey = Query("latest"),
    page: int = Query(1, ge=1),
    limit: int = Query(25, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
) -> ModelPerformancePage:
    """m5 페이지 메인 테이블 — 필터/정렬/페이지네이션."""
    base = _apply_filters(
        select(Model),
        date_start=date_start,
        date_end=date_end,
        backbone=backbone,
        state=state,
    )

    total_stmt = _apply_filters(
        select(func.count(Model.id)),
        date_start=date_start,
        date_end=date_end,
        backbone=backbone,
        state=state,
    )
    total = (await session.execute(total_stmt)).scalar_one()

    stmt = _apply_sort(base, sort).offset((page - 1) * limit).limit(limit)
    items = list((await session.execute(stmt)).scalars().all())

    return ModelPerformancePage(items=items, total=total, page=page, limit=limit)


@router.get("/performance/export")
async def export_models_csv(
    date_start: datetime | None = Query(None),
    date_end: datetime | None = Query(None),
    backbone: Backbone | None = Query(None),
    state: DeploymentState | None = Query(None),
    sort: SortKey = Query("latest"),
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    """필터된 모델 목록을 CSV 로 다운로드."""
    stmt = _apply_sort(
        _apply_filters(
            select(Model),
            date_start=date_start,
            date_end=date_end,
            backbone=backbone,
            state=state,
        ),
        sort,
    )
    rows = list((await session.execute(stmt)).scalars().all())

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "version",
            "backbone",
            "training_completed_at",
            "training_sheets",
            "training_pixels",
            "miou_overall",
            "miou_rice_paddy",
            "miou_dry_field",
            "miou_orchard",
            "miou_greenhouse",
            "deployment_state",
        ]
    )
    for m in rows:
        writer.writerow(
            [
                m.version,
                m.backbone.value,
                m.training_completed_at.isoformat(),
                m.training_sheets,
                m.training_pixels,
                f"{m.miou_overall:.4f}",
                f"{m.miou_rice_paddy:.4f}",
                f"{m.miou_dry_field:.4f}",
                f"{m.miou_orchard:.4f}",
                f"{m.miou_greenhouse:.4f}",
                m.deployment_state.value,
            ]
        )

    csv_bytes = buf.getvalue().encode("utf-8-sig")  # 엑셀 한글 깨짐 방지 BOM
    filename = f"models_performance_{datetime.now():%Y%m%d_%H%M%S}.csv"
    return StreamingResponse(
        iter([csv_bytes]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.patch("/{model_id}/state", response_model=ModelOut)
async def update_model_state(
    model_id: uuid.UUID,
    payload: ModelStateUpdate,
    session: AsyncSession = Depends(get_session),
) -> Model:
    """배포 상태 변경 (production / staging / archived)."""
    obj = await session.get(Model, model_id)
    if obj is None:
        raise HTTPException(404, f"model {model_id} not found")
    obj.deployment_state = payload.state
    await session.commit()
    await session.refresh(obj)
    return obj
