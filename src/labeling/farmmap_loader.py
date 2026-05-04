"""팜맵(Farm Map) SHP → polygon 메모리 어댑터.

농림축산식품부 팜맵 SHP 파일에서 GeoTIFF 영역에 들어오는 농지 polygon 을 추출.
``cadastral_bootstrap.process_geotiff`` 와 동일한 ``[{"geometry","properties"}, ...]``
형태로 반환하므로 ``cadastral_preview`` 가 그대로 사용 가능.

검증 (2026-04-30):
    - 압축 해제 후 9 SHP (FARM_1 ~ FARM_9)
    - CRS: EPSG:5179 (Korea 2000 통합원점)
    - DBF 인코딩: cp949
    - INTPR_CD: 01=논 / 02=밭 / 03=과수 / 04=시설
    - 컬럼: FMAP_INNB, PNU_LNM_CD, LGL_EMD_NM, LNM, INTPR_CD/NM, AREA, ...
    - 경주 영역(EPSG:5186 약 (402296, 377864, 404021, 381304))은 FARM_2 + FARM_3 에 분포
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


# 팜맵 INTPR_CD → (class_code, class_name)
# v1.0 = 2-class (논/밭) — 학습 진입 시 정책
FARMMAP_CLASS_MAP_2: dict[str, tuple[int, str]] = {
    "01": (0, "rice_paddy"),
    "02": (1, "dry_field"),
}

# 4-class — 검수 시각화 시 정책 (시설/과수도 색상 표시)
FARMMAP_CLASS_MAP_4: dict[str, tuple[int, str]] = {
    "01": (0, "rice_paddy"),
    "02": (1, "dry_field"),
    "03": (2, "orchard"),
    "04": (3, "greenhouse"),
}

# 하위 호환 (기존 코드/테스트가 FARMMAP_CLASS_MAP 참조)
FARMMAP_CLASS_MAP = FARMMAP_CLASS_MAP_2

# 농지 외 INTPR_CD (참고 — 명시적 제외, 2-class 정책 시)
FARMMAP_EXCLUDED: dict[str, str] = {
    "03": "과수",
    "04": "시설",
}


@dataclass
class FarmmapStats:
    """팜맵 SHP 처리 통계."""

    shp_names: str
    polygons_total: int = 0
    polygons_kept: int = 0
    polygons_skipped_class: int = 0
    polygons_skipped_outside: int = 0
    per_class: dict[str, int] = field(default_factory=dict)


def load_farmmap_polygons(
    shp_paths: list[Path],
    geotiff_bounds_dst_crs: tuple[float, float, float, float],
    dst_crs: str,
    valid_region_dst_crs=None,
    encoding: str = "cp949",
    class_map: dict[str, tuple[int, str]] | None = None,
) -> tuple[list[dict], FarmmapStats]:
    """다중 팜맵 SHP → GeoTIFF 영역 안의 농지 polygon 추출.

    Args:
        shp_paths: 팜맵 SHP 파일 리스트 (예: FARM_2.shp, FARM_3.shp).
            여러 SHP 가 GeoTIFF 영역에 걸쳐 있어도 모두 합쳐서 처리.
        geotiff_bounds_dst_crs: GeoTIFF bounds (left, bottom, right, top) — dst_crs 좌표계.
        dst_crs: 출력 좌표계 문자열 (예: ``"EPSG:5186"``).
        valid_region_dst_crs: GeoTIFF alpha clip valid region (shapely (Multi)Polygon).
            None 이면 GeoTIFF bbox 와 intersection 만 적용.
        encoding: DBF 한글 인코딩 (기본 ``"cp949"``).

    Returns:
        ``(features, stats)``:
            features = ``[{"geometry": shapely, "properties": {...}}, ...]`` —
            cadastral_bootstrap.process_geotiff 와 동일 형태 (cadastral_preview 호환).
    """
    import pyogrio
    from pyproj import Transformer
    from shapely.geometry import box
    from shapely.ops import transform as shapely_transform

    if not shp_paths:
        raise ValueError("shp_paths must not be empty")

    if class_map is None:
        class_map = FARMMAP_CLASS_MAP_2

    stats = FarmmapStats(shp_names=",".join(p.name for p in shp_paths))

    # 첫 SHP 의 CRS 를 SHP 좌표계로 가정 (팜맵 9 SHP 모두 EPSG:5179 검증됨)
    first_info = pyogrio.read_info(shp_paths[0])
    src_crs_str = first_info["crs"] or "EPSG:5179"

    # GeoTIFF bounds (dst) → SHP 좌표계 로 변환해 pyogrio bbox filter 인자로 사용
    if str(src_crs_str).upper() != dst_crs.upper():
        bbox_to_src = Transformer.from_crs(
            dst_crs, src_crs_str, always_xy=True
        ).transform_bounds
        bbox_src = bbox_to_src(*geotiff_bounds_dst_crs)
        to_dst = Transformer.from_crs(src_crs_str, dst_crs, always_xy=True).transform

        def project(x: float, y: float, z: float | None = None) -> tuple[float, float]:
            return to_dst(x, y)

    else:
        bbox_src = geotiff_bounds_dst_crs
        project = None

    geotiff_box = box(*geotiff_bounds_dst_crs)
    clip_geom = (
        valid_region_dst_crs if valid_region_dst_crs is not None else geotiff_box
    )

    out_features: list[dict] = []

    for shp_path in shp_paths:
        log.info(
            "[farmmap %s] reading (bbox in %s = %s)",
            shp_path.name,
            src_crs_str,
            tuple(round(x, 1) for x in bbox_src),
        )
        gdf = pyogrio.read_dataframe(
            shp_path,
            bbox=bbox_src,
            encoding=encoding,
            columns=[
                "FMAP_INNB",
                "PNU_LNM_CD",
                "LGL_EMD_NM",
                "LNM",
                "INTPR_CD",
                "INTPR_NM",
                "AREA",
            ],
        )
        log.info(
            "[farmmap %s] bbox filter → %d polygon(s)", shp_path.name, len(gdf)
        )

        for _, row in gdf.iterrows():
            stats.polygons_total += 1
            cd = row.get("INTPR_CD")
            cls = class_map.get(cd)
            if cls is None:
                stats.polygons_skipped_class += 1
                continue
            class_code, class_name = cls

            geom_src = row.geometry
            if geom_src is None or geom_src.is_empty:
                stats.polygons_skipped_outside += 1
                continue
            geom_dst = shapely_transform(project, geom_src) if project else geom_src

            if not geom_dst.intersects(clip_geom):
                stats.polygons_skipped_outside += 1
                continue
            clipped = geom_dst.intersection(clip_geom)
            if clipped.is_empty:
                stats.polygons_skipped_outside += 1
                continue

            out_features.append(
                {
                    "geometry": clipped,
                    "properties": {
                        "fmap_innb": row.get("FMAP_INNB"),
                        "pnu": row.get("PNU_LNM_CD"),
                        "emd_nm": row.get("LGL_EMD_NM"),
                        "lnm": row.get("LNM"),
                        "intpr_cd": cd,
                        "intpr_nm": row.get("INTPR_NM"),
                        "area": float(row.get("AREA"))
                        if row.get("AREA") is not None
                        else None,
                        "class_code": class_code,
                        "class_name": class_name,
                        "source": "farmmap",
                    },
                }
            )
            stats.polygons_kept += 1
            stats.per_class[class_name] = stats.per_class.get(class_name, 0) + 1

    log.info(
        "[farmmap] total=%d kept=%d (skip class=%d, outside=%d) — %s",
        stats.polygons_total,
        stats.polygons_kept,
        stats.polygons_skipped_class,
        stats.polygons_skipped_outside,
        stats.per_class,
    )
    return out_features, stats


def discover_farmmap_shps(farmmap_dir: Path) -> list[Path]:
    """팜맵 디렉토리에서 모든 ``FARM_*.shp`` 자동 발견 (정렬된 리스트)."""
    if not farmmap_dir.is_dir():
        raise FileNotFoundError(f"farmmap dir not found: {farmmap_dir}")
    shps = sorted(farmmap_dir.glob("FARM_*.shp"))
    if not shps:
        raise FileNotFoundError(f"no FARM_*.shp in {farmmap_dir}")
    return shps
