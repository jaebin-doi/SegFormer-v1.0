#!/usr/bin/env bash
# 농지 SegFormer 평가 — v2 (4-class)
#
# 사용:
#   bash scripts/training/eval_segformer_farmland.sh
#
# 환경변수:
#   PYTHON       — Python 인터프리터
#   CHECKPOINT   — 평가할 체크포인트 (.pth) — 기본: 최신 best_mIoU_*.pth
#   DATASET      — 데이터셋 이름 (기본: farmland-5cm-v2)
#   OUTPUT       — 평가 결과 JSON (기본: metrics/farmland_v2.json)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-C:/Users/admin/Miniconda3/python.exe}"
WORK_DIR="${WORK_DIR:-work_dirs/farmland_segformer_v2/}"
DATASET="${DATASET:-farmland-5cm-v2}"
OUTPUT="${OUTPUT:-metrics/farmland_v2.json}"

# CHECKPOINT 자동 탐지 (best_mIoU 우선, 없으면 가장 최근 iter)
if [ -z "${CHECKPOINT:-}" ]; then
    CHECKPOINT=$(ls -t "$WORK_DIR"/best_mIoU_*.pth 2>/dev/null | head -1 || true)
    if [ -z "$CHECKPOINT" ]; then
        CHECKPOINT=$(ls -t "$WORK_DIR"/iter_*.pth 2>/dev/null | head -1 || true)
    fi
fi

if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: 체크포인트 없음 — $WORK_DIR" >&2
    echo "  먼저 학습: bash scripts/training/train_segformer_farmland.sh" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"

echo "================================================================"
echo "  농지 SegFormer 평가 v2 — 4-class"
echo "================================================================"
echo "  CHECKPOINT: $CHECKPOINT"
echo "  DATASET:    $DATASET"
echo "  OUTPUT:     $OUTPUT"
echo ""

PYTHONIOENCODING=utf-8 "$PYTHON" src/evaluation/eval_segformer_farmland.py \
    --checkpoint "$CHECKPOINT" \
    --dataset "$DATASET" \
    --output "$OUTPUT"

echo ""
echo "================================================================"
echo "  ✅ 평가 완료"
echo "================================================================"
echo "  결과 JSON: $OUTPUT"
echo "  KPI 통과 여부: jq '.verdict.passed' $OUTPUT"
