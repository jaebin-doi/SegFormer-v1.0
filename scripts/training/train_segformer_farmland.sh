#!/usr/bin/env bash
# 농지 SegFormer 학습 — v2 (4-class 팜맵 자동 라벨)
#
# 전제: build_dataset.sh 로 data/dataset/farmland-5cm-v2/ 가 빌드되어 있어야 함.
#
# 사용:
#   bash scripts/training/train_segformer_farmland.sh
#
# 환경변수:
#   PYTHON      — Python 인터프리터 (워크스테이션 conda env)
#   DATASET_DIR — 학습 데이터셋 (기본: data/dataset/farmland-5cm-v2/)
#   WORK_DIR    — 학습 산출물 폴더 (기본: work_dirs/farmland_segformer_v2/)
#   CONFIG      — config 파일 (기본: src/training/configs/segformer_farmland.py)
#   AMP         — 1/0 bf16 AMP 활성 (기본 1)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-C:/Users/admin/Miniconda3/python.exe}"
DATASET_DIR="${DATASET_DIR:-data/dataset/farmland-5cm-v2/}"
WORK_DIR="${WORK_DIR:-work_dirs/farmland_segformer_v2/}"
CONFIG="${CONFIG:-src/training/configs/segformer_farmland.py}"
AMP="${AMP:-1}"

echo "================================================================"
echo "  농지 SegFormer 학습 v2 — 4-class (논/밭/과수/시설)"
echo "================================================================"
echo "  CONFIG:      $CONFIG"
echo "  DATASET:     $DATASET_DIR"
echo "  WORK_DIR:    $WORK_DIR"
echo "  AMP (bf16):  $AMP"
echo ""

# 데이터셋 사전 점검
if [ ! -d "$DATASET_DIR/images/train" ]; then
    echo "ERROR: 학습 데이터셋 없음 — $DATASET_DIR/images/train" >&2
    echo "  먼저 실행: bash scripts/training/build_dataset.sh" >&2
    exit 1
fi
if [ ! -d "$DATASET_DIR/masks/train" ]; then
    echo "ERROR: 학습 mask 없음 — $DATASET_DIR/masks/train" >&2
    exit 1
fi

# data_root 동적 주입 (config 의 DATA_ROOT 변수)
mkdir -p "$WORK_DIR"

ARGS=(
    "$PYTHON" -m mim train mmsegmentation
    "$CONFIG"
    --work-dir "$WORK_DIR"
    --cfg-options
    "data_root=$DATASET_DIR"
    "train_dataloader.dataset.dataset.data_root=$DATASET_DIR"
    "val_dataloader.dataset.data_root=$DATASET_DIR"
    "test_dataloader.dataset.data_root=$DATASET_DIR"
)

if [ "$AMP" = "1" ]; then
    ARGS+=(--amp)
fi

echo "[train] command:"
printf '  %q ' "${ARGS[@]}"
echo ""
echo ""

PYTHONIOENCODING=utf-8 "${ARGS[@]}"

echo ""
echo "================================================================"
echo "  ✅ 학습 완료"
echo "================================================================"
echo "  체크포인트: $WORK_DIR/best_mIoU_iter_*.pth"
echo "  로그:       $WORK_DIR/{날짜}.log"
echo ""
echo "  평가:"
echo "    bash scripts/training/eval_segformer_farmland.sh"
