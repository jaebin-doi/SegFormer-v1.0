#!/usr/bin/env bash
# 학습 데이터셋 빌드 — 정사영상 + mask GeoTIFF (옵션 D 산출물) → 타일링 + region split
#
# 산출물:
#   data/dataset/farmland-5cm-v2/
#     ├── images/{train,val,test}/*.png
#     ├── masks/{train,val,test}/*.png
#     └── split_manifest.json
#
# 사용:
#   bash scripts/training/build_dataset.sh
#
# 환경변수 오버라이드:
#   PYTHON                 — Python 인터프리터
#   GEOTIFF_DIR            — 원본 정사영상 폴더 (기본: img/.../경주/)
#   MASK_DIR               — 옵션 D mask 폴더 (기본: data/labels/팜맵_논밭과수시설_AI학습용/)
#   TILES_DIR              — 타일링 출력 (기본: data/tiles_masked/)
#   DATASET_DIR            — split 출력 (기본: data/dataset/farmland-5cm-v2/)
#   TILE_SIZE / OVERLAP    — 타일 크기 (기본 2048 / 512)
#   SKIP_RATIO             — mask ignore X% 이상 타일 skip (기본 0.8 = 라벨 < 20% skip)
#   TRAIN_RATIO/VAL_RATIO  — region split 비율 (기본 0.70 / 0.15)
#   SEED                   — split 재현용 (기본 42)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-C:/Users/admin/Miniconda3/python.exe}"
GEOTIFF_DIR="${GEOTIFF_DIR:-img/1. 농촌 학습(전, 답, 과 구분)/경주/}"
MASK_DIR="${MASK_DIR:-data/labels/팜맵_논밭과수시설_AI학습용/}"
TILES_DIR="${TILES_DIR:-data/tiles_masked/}"
DATASET_DIR="${DATASET_DIR:-data/dataset/farmland-5cm-v2/}"
TILE_SIZE="${TILE_SIZE:-2048}"
OVERLAP="${OVERLAP:-512}"
SKIP_RATIO="${SKIP_RATIO:-0.8}"
TRAIN_RATIO="${TRAIN_RATIO:-0.70}"
VAL_RATIO="${VAL_RATIO:-0.15}"
SEED="${SEED:-42}"

echo "================================================================"
echo "  학습 데이터셋 빌드 — v2 (4-class 팜맵 자동 라벨)"
echo "================================================================"
echo "  GEOTIFF_DIR: $GEOTIFF_DIR"
echo "  MASK_DIR:    $MASK_DIR"
echo "  TILES_DIR:   $TILES_DIR"
echo "  DATASET_DIR: $DATASET_DIR"
echo ""

# Pre-check
if [ ! -d "$GEOTIFF_DIR" ]; then
    echo "ERROR: GEOTIFF_DIR not found — $GEOTIFF_DIR" >&2
    exit 1
fi
if [ ! -d "$MASK_DIR" ]; then
    echo "ERROR: MASK_DIR not found — $MASK_DIR" >&2
    echo "  옵션 D 먼저 실행: OPTION=D bash scripts/labeling/run_four_options.sh" >&2
    exit 1
fi

mkdir -p "$TILES_DIR" "$DATASET_DIR"

echo ">>> [1/2] 타일링 (영상 + mask 동시, ${TILE_SIZE}x${TILE_SIZE} / overlap ${OVERLAP})"
PYTHONIOENCODING=utf-8 "$PYTHON" src/preprocessing/tiling.py \
    --input "$GEOTIFF_DIR" \
    --mask-input "$MASK_DIR" \
    --output "$TILES_DIR" \
    --tile-size "$TILE_SIZE" \
    --overlap "$OVERLAP" \
    --target-crs "EPSG:5186" \
    --skip-empty-mask-ratio "$SKIP_RATIO"
echo ""

# meta.json 파일들 수집 (split_dataset.py 의 --metas 인자 다중 입력)
META_FILES=()
for d in "$TILES_DIR"/*/; do
    if [ -f "$d/meta.json" ]; then
        META_FILES+=("$d/meta.json")
    fi
done

if [ "${#META_FILES[@]}" -lt 3 ]; then
    echo "WARN: meta.json 파일 ${#META_FILES[@]}개 — region 단위 정식 split 위해 3개 이상 필요." >&2
    echo "      (n=2 이하면 split_dataset.py 가 --allow-sanity 요구함)" >&2
fi

echo ">>> [2/2] Region 단위 split (train/val/test, seed=$SEED)"
PYTHONIOENCODING=utf-8 "$PYTHON" src/preprocessing/split_dataset.py \
    --images-root "$TILES_DIR/images" \
    --masks-root "$TILES_DIR/masks" \
    --metas "${META_FILES[@]}" \
    --output "$DATASET_DIR" \
    --train-ratio "$TRAIN_RATIO" \
    --val-ratio "$VAL_RATIO" \
    --seed "$SEED" \
    --force

echo ""
echo "================================================================"
echo "  ✅ 데이터셋 빌드 완료"
echo "================================================================"
echo "  $DATASET_DIR"
echo "    ├── images/{train,val,test}/*.png"
echo "    ├── masks/{train,val,test}/*.png"
echo "    └── split_manifest.json"
echo ""
echo "  학습 진입:"
echo "    bash scripts/training/train_segformer_farmland.sh"
