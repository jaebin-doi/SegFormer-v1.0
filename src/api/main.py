"""FastAPI 진입점 — `uvicorn src.api.main:app --reload --port 8000`.

정적 프론트 (`public/`) 를 루트에 마운트하여 단일 포트에서 UI + API 를 서빙한다.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.config import REPO_ROOT, get_settings
from src.api.routers import models as models_router

settings = get_settings()

app = FastAPI(
    title="DOI AI — SegFormer v1.0 백엔드",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(models_router.router)

# 정적 프론트엔드 서빙 (마지막에 마운트 — /api/* 라우트가 우선 매칭됨)
PUBLIC_DIR: Path = REPO_ROOT / "public"
if PUBLIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(PUBLIC_DIR), html=True), name="public")
