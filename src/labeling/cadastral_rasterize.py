"""학습용 마스크 GeoTIFF 생성 — polygon → 픽셀별 클래스 인덱스 raster.

원본 정사영상 GeoTIFF 와 동일한 transform/CRS/size 의 단일 밴드 uint8 mask 를 출력.
``data/labels/팜맵_논밭과수시설_AI학습용/<원본_stem>.tif`` 형태로 저장되어 학습 시
정사영상 + 마스크 1:1 매핑으로 사용 가능.

픽셀 값 매핑:
    0   = rice_paddy (논)
    1   = dry_field  (밭)
    2   = orchard    (과수)
    3   = greenhouse (시설)
    255 = ignore (라벨 없음 — mmsegmentation LoadAnnotations 기본 ignore_index)

사용:
    python src/labeling/cadastral_rasterize.py \\
        --geotiff-dir "img/.../경주/" \\
        --farmmap-dir "data/labels/farmmap/" \\
        --output-dir "data/labels/팜맵_논밭과수시설_AI학습용/" \\
        --classes 4
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)


def rasterize_to_mask(
    tif_path: Path,
    geometries: list,
    class_codes: list[int],
    out_path: Path,
    ignore_value: int = 255,
) -> dict:
    """polygon list → 학습용 mask GeoTIFF 저장.

    Args:
        tif_path: 원본 정사영상 GeoTIFF (transform/CRS/size 기준).
        geometries: shapely Polygon/MultiPolygon 리스트 (원본 GeoTIFF CRS 좌표계).
        class_codes: 같은 길이의 클래스 인덱스 리스트 (0~N-1).
        out_path: 출력 mask GeoTIFF 경로.
        ignore_value: 라벨 없는 픽셀 값 (기본 255).

    Returns:
        ``{"shape": (H, W), "size_bytes": int, "class_pixel_counts": {class: count, ...}}``.
    """
    import rasterio
    from rasterio.features import rasterize

    if len(geometries) != len(class_codes):
        raise ValueError(
            f"geometries({len(geometries)}) and class_codes({len(class_codes)}) length mismatch"
        )

    with rasterio.open(tif_path) as src:
        transform = src.transform
        out_shape = (src.height, src.width)
        crs = src.crs

    shapes = [
        (geom, int(code)) for geom, code in zip(geometries, class_codes) if not geom.is_empty
    ]

    if shapes:
        mask = rasterize(
            shapes=shapes,
            out_shape=out_shape,
            transform=transform,
            fill=ignore_value,
            dtype="uint8",
            all_touched=False,
        )
    else:
        log.warning(
            "[%s] no valid polygons — writing all-ignore mask", tif_path.name
        )
        mask = np.full(out_shape, ignore_value, dtype=np.uint8)

    profile: dict = {
        "driver": "GTiff",
        "height": out_shape[0],
        "width": out_shape[1],
        "count": 1,
        "dtype": "uint8",
        "compress": "lzw",
        "tiled": True,
        "nodata": ignore_value,
    }
    if crs:
        profile["crs"] = crs
        profile["transform"] = transform

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask, 1)

    unique, counts = np.unique(mask, return_counts=True)
    class_pixel_counts = {int(v): int(c) for v, c in zip(unique, counts)}

    return {
        "shape": out_shape,
        "size_bytes": out_path.stat().st_size,
        "class_pixel_counts": class_pixel_counts,
    }


def main() -> None:
    """CLI 진입점 — 6 GeoTIFF 일괄 처리."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="팜맵 polygon → 학습용 mask GeoTIFF (원본 GeoTIFF 와 1:1 매핑)"
    )
    parser.add_argument(
        "--geotiff-dir", type=Path, required=True, help="원본 GeoTIFF 폴더 또는 단일 .tif"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="mask GeoTIFF 출력 폴더"
    )
    parser.add_argument(
        "--farmmap-dir",
        type=Path,
        required=True,
        help="팜맵 SHP 폴더 (FARM_*.shp 자동 발견)",
    )
    parser.add_argument(
        "--classes",
        type=int,
        choices=[2, 4],
        default=4,
        help="클래스 수 — 2(논/밭) 또는 4(논/밭/과수/시설). 기본 4.",
    )
    parser.add_argument("--ignore-value", type=int, default=255)
    args = parser.parse_args()

    if args.geotiff_dir.is_file():
        geotiffs = [args.geotiff_dir]
    else:
        geotiffs = sorted(
            p
            for p in args.geotiff_dir.iterdir()
            if p.is_file() and p.suffix.lower() in (".tif", ".tiff")
        )
    if not geotiffs:
        raise SystemExit(f"No GeoTIFF in {args.geotiff_dir}")

    # 팜맵 SHP 자동 발견
    from src.labeling.farmmap_loader import (
        FARMMAP_CLASS_MAP_2,
        FARMMAP_CLASS_MAP_4,
        discover_farmmap_shps,
        load_farmmap_polygons,
    )
    from src.labeling.cadastral_bootstrap import get_valid_region

    farmmap_shps = discover_farmmap_shps(args.farmmap_dir)
    class_map = FARMMAP_CLASS_MAP_4 if args.classes == 4 else FARMMAP_CLASS_MAP_2

    log.info(
        "Rasterizing %d GeoTIFF(s) → %s (classes=%d, ignore=%d)",
        len(geotiffs),
        args.output_dir,
        args.classes,
        args.ignore_value,
    )

    import rasterio  # noqa: E402

    for tif in geotiffs:
        with rasterio.open(tif) as src:
            bounds = src.bounds
            src_crs = str(src.crs) if src.crs else "EPSG:5186"

        valid_region = get_valid_region(tif, src_crs)

        feats, stats = load_farmmap_polygons(
            farmmap_shps,
            (bounds.left, bounds.bottom, bounds.right, bounds.top),
            src_crs,
            valid_region_dst_crs=valid_region,
            class_map=class_map,
        )
        geometries = [f["geometry"] for f in feats]
        class_codes = [f["properties"]["class_code"] for f in feats]

        log.info(
            "[%s] polygons=%d (class breakdown: %s)",
            tif.name,
            len(feats),
            stats.per_class,
        )

        out_path = args.output_dir / f"{tif.stem}.tif"
        result = rasterize_to_mask(
            tif, geometries, class_codes, out_path, ignore_value=args.ignore_value
        )
        log.info(
            "[%s] mask shape=%s size=%.1f MB pixels=%s",
            tif.name,
            result["shape"],
            result["size_bytes"] / 1024**2,
            result["class_pixel_counts"],
        )

    log.info("=" * 60)
    log.info("DONE — %d mask GeoTIFF(s) → %s", len(geotiffs), args.output_dir)


if __name__ == "__main__":
    main()
