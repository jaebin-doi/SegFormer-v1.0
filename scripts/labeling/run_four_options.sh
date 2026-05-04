#!/usr/bin/env bash
# 4가지 라벨링 옵션 한 번에 실행 (또는 OPTION 환경변수로 1개만):
#   A: 팜맵 단독 2-class (논/밭, PNG only)
#         → data/labels/팜맵_논밭만/
#   B: 팜맵 단독 4-class (논/밭/과수/시설, PNG only)
#         → data/labels/팜맵_논밭과수시설/
#   C: 팜맵 4-class + 텍스트 라벨 (PNG only, 텍스트=주소+농지분류)
#         → data/labels/팜맵_논밭과수시설_지적정보/
#   D: AI 학습용 마스크 GeoTIFF (PNG ❌, 원본과 동일 size 의 mask raster)
#         → data/labels/팜맵_논밭과수시설_AI학습용/
#
# 사용:
#   bash scripts/labeling/run_four_options.sh           # 4 옵션 모두 실행
#   OPTION=A bash scripts/labeling/run_four_options.sh  # 옵션 A 만
#   OPTION=D bash scripts/labeling/run_four_options.sh  # 학습용 mask 만
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-C:/Users/admin/Miniconda3/python.exe}"
GEOTIFF_DIR="${GEOTIFF_DIR:-img/1. 농촌 학습(전, 답, 과 구분)/경주/}"
FARMMAP_DIR="${FARMMAP_DIR:-data/labels/farmmap/}"
MAX_SIZE="${MAX_SIZE:-4096}"
OPTION="${OPTION:-ALL}"

echo ">>> [의존성 확인]"
bash "$SCRIPT_DIR/install_deps.sh"
echo ""

run_preview() {
    local label="$1"; shift
    local out_dir="$1"; shift
    echo "================================================================"
    echo "  $label"
    echo "  → $out_dir (PNG only)"
    echo "================================================================"
    mkdir -p "$out_dir"
    PYTHONIOENCODING=utf-8 "$PYTHON" src/labeling/cadastral_preview.py \
        --geotiff-dir "$GEOTIFF_DIR" \
        --output-dir "$out_dir" \
        --source farmmap \
        --farmmap-dir "$FARMMAP_DIR" \
        --max-size "$MAX_SIZE" \
        "$@"
    echo ""
}

run_rasterize() {
    local label="$1"; shift
    local out_dir="$1"; shift
    echo "================================================================"
    echo "  $label"
    echo "  → $out_dir (학습용 mask GeoTIFF only)"
    echo "================================================================"
    mkdir -p "$out_dir"
    PYTHONIOENCODING=utf-8 "$PYTHON" src/labeling/cadastral_rasterize.py \
        --geotiff-dir "$GEOTIFF_DIR" \
        --output-dir "$out_dir" \
        --farmmap-dir "$FARMMAP_DIR" \
        --classes 4
    echo ""
}

if [ "$OPTION" = "A" ] || [ "$OPTION" = "ALL" ]; then
    run_preview "[옵션 A] 팜맵 단독 2-class (논/밭) — PNG only" \
        "data/labels/팜맵_논밭만/" \
        --farmmap-classes 2
fi

if [ "$OPTION" = "B" ] || [ "$OPTION" = "ALL" ]; then
    run_preview "[옵션 B] 팜맵 단독 4-class (논/밭/과수/시설) — PNG only" \
        "data/labels/팜맵_논밭과수시설/" \
        --farmmap-classes 4
fi

if [ "$OPTION" = "C" ] || [ "$OPTION" = "ALL" ]; then
    run_preview "[옵션 C] 팜맵 4-class + 텍스트 라벨 — PNG only" \
        "data/labels/팜맵_논밭과수시설_지적정보/" \
        --farmmap-classes 4 \
        --show-labels
fi

if [ "$OPTION" = "D" ] || [ "$OPTION" = "ALL" ]; then
    run_rasterize "[옵션 D] AI 학습용 마스크 GeoTIFF" \
        "data/labels/팜맵_논밭과수시설_AI학습용/"
fi

echo "================================================================"
echo "  ✅ 옵션 ${OPTION} 완료"
echo "================================================================"
case "$OPTION" in
    A) echo "  data/labels/팜맵_논밭만/  (PNG 6장)" ;;
    B) echo "  data/labels/팜맵_논밭과수시설/  (PNG 6장)" ;;
    C) echo "  data/labels/팜맵_논밭과수시설_지적정보/  (PNG 6장 + 텍스트)" ;;
    D) echo "  data/labels/팜맵_논밭과수시설_AI학습용/  (mask TIF 6장)" ;;
    ALL)
        echo "  A: data/labels/팜맵_논밭만/"
        echo "  B: data/labels/팜맵_논밭과수시설/"
        echo "  C: data/labels/팜맵_논밭과수시설_지적정보/"
        echo "  D: data/labels/팜맵_논밭과수시설_AI학습용/"
        ;;
esac
