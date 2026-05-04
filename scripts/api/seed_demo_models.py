"""데모 모델 23개 시드 — m5 페이지가 빈 화면이 되지 않도록 채움.

사용:
    python scripts/api/seed_demo_models.py        # 추가 (중복 version 은 skip)
    python scripts/api/seed_demo_models.py --reset  # 전체 삭제 후 재삽입

기존 m5_model_performance.html 의 mock generator 와 같은 분포 (백본별 base mIoU,
데이터량 효과, 클래스별 난이도) 를 사용하되, 결과물은 PostgreSQL 행으로 저장.
"""
from __future__ import annotations

import argparse
import asyncio
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import delete, select  # noqa: E402

from src.api.db import AsyncSessionLocal  # noqa: E402
from src.api.models.model import Backbone, DeploymentState, Model  # noqa: E402

BACKBONE_BASE_MIOU = {Backbone.B0: 0.74, Backbone.B2: 0.81, Backbone.B3: 0.85}
BACKBONE_WEIGHT = [Backbone.B0, Backbone.B2, Backbone.B2, Backbone.B3]
STATE_WEIGHT = (
    [DeploymentState.PRODUCTION] * 3
    + [DeploymentState.STAGING] * 5
    + [DeploymentState.ARCHIVED] * 12
)
# 클래스 난이도 (옵션 D mask 기반: 논 풍부 / 시설 어려움)
CLASS_DELTA = {
    "rice_paddy": +0.04,
    "dry_field": +0.02,
    "orchard": -0.04,
    "greenhouse": -0.10,
}


def _gen_models(count: int = 23, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    backbone_versions: dict[Backbone, int] = {b: 0 for b in Backbone}
    rows: list[dict] = []
    base_date = datetime.now(timezone.utc).replace(microsecond=0)

    for i in range(count):
        backbone = rng.choice(BACKBONE_WEIGHT)
        backbone_versions[backbone] += 1
        v = backbone_versions[backbone]
        version = f"SegFormer-{backbone.value}-v.1.{v - 1}"

        # 최근일수록 최신 (5일 간격, 약간의 jitter)
        completed_at = base_date - timedelta(days=i * 5 + rng.randint(0, 3))

        # 데이터량 — 최근일수록 많음 (선형 + jitter)
        sheets = max(50, int(8000 - i * 300 + rng.randint(-500, 500)))
        pixels = sheets * 2048 * 2048  # 대략 — 2048^2 타일 가정

        # mIoU = 백본 base + 데이터량 보너스 + 노이즈
        data_bonus = min(0.08, sheets / 100_000)
        noise = rng.uniform(-0.02, 0.02)
        miou_overall = round(BACKBONE_BASE_MIOU[backbone] + data_bonus + noise, 4)
        miou_overall = max(0.40, min(0.92, miou_overall))

        per_class = {
            cls: round(
                max(
                    0.30,
                    min(
                        0.97,
                        miou_overall + delta + rng.uniform(-0.03, 0.03),
                    ),
                ),
                4,
            )
            for cls, delta in CLASS_DELTA.items()
        }

        rows.append(
            dict(
                version=version,
                backbone=backbone,
                training_completed_at=completed_at,
                training_sheets=sheets,
                training_pixels=pixels,
                miou_overall=miou_overall,
                miou_rice_paddy=per_class["rice_paddy"],
                miou_dry_field=per_class["dry_field"],
                miou_orchard=per_class["orchard"],
                miou_greenhouse=per_class["greenhouse"],
                deployment_state=rng.choice(STATE_WEIGHT),
            )
        )
    return rows


async def _seed(reset: bool) -> None:
    async with AsyncSessionLocal() as session:
        if reset:
            await session.execute(delete(Model))
            await session.commit()
            print("[reset] models 테이블 비움")

        existing_versions = set(
            (await session.execute(select(Model.version))).scalars().all()
        )
        rows = _gen_models()
        new_rows = [r for r in rows if r["version"] not in existing_versions]
        if not new_rows:
            print(f"중복 — 추가할 행 없음 (기존 {len(existing_versions)}개)")
            return

        session.add_all([Model(**r) for r in new_rows])
        await session.commit()
        print(
            f"삽입 완료 — {len(new_rows)}개 추가 (기존 {len(existing_versions)}개, 합계 {len(existing_versions) + len(new_rows)}개)"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset", action="store_true", help="삽입 전에 models 테이블 비움"
    )
    args = parser.parse_args()
    asyncio.run(_seed(reset=args.reset))


if __name__ == "__main__":
    main()
