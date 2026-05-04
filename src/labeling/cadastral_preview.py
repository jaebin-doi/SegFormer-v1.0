"""GeoTIFF + GeoPackage 라벨 → PNG 미리보기 시각화.

QGIS 가 없는 환경에서 자동 라벨링 결과를 빠르게 검수할 수 있도록, 각 GeoTIFF 위에
GeoPackage polygon 을 반투명 색상으로 오버레이한 PNG 1 장을 생성한다.

설계:
    - GeoTIFF 가 너무 커서 (예: 16844×24167) 원본 해상도 PNG 는 비현실적.
      ``--max-size`` 로 가로/세로 long edge 를 다운샘플 (기본 4096px).
    - 색상:
        rice_paddy (논) = 파랑 반투명 + 진한 파랑 외곽선
        dry_field  (밭) = 주황 반투명 + 진한 주황 외곽선
    - layer 이름 = GeoTIFF stem 매칭. 일치하는 layer 가 없으면 빈 이미지로 저장.

사용:
    python src/labeling/cadastral_preview.py \\
        --geotiff-dir "img/1. 농촌 학습(전, 답, 과 구분)/경주/" \\
        --gpkg        data/labels/farmland_v1.0_cadastral.gpkg \\
        --output-dir  data/labels/preview/
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


# 클래스별 RGBA 색상 맵.
# 라벨링 스타일은 src/labeling/render_overlay.py 패턴 참고:
#   fill = 옅은 색 (alpha 0.18 ≈ 46/255)
#   border = 같은 색의 100% solid (alpha 255), 두께 2px
#
# 출처별로 색상 분리 — VWorld 와 팜맵을 같은 PNG 에 겹쳐 그릴 때 시각적 구분.
#   VWorld(연속지적도) = 파랑(논) / 주황(밭) — fill+outline (현재 기본)
#   팜맵(Farm Map)    = 보라(논) / 빨강(밭) — outline only (fill alpha=0)
#                       → VWorld 와 일치 영역은 외곽선만, 불일치 영역은 두 polygon 따로 보임

CLASS_COLORS_VWORLD: dict[str, tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = {
    "rice_paddy": ((59, 130, 246, 46), (59, 130, 246, 255)),  # blue-500
    "dry_field": ((245, 158, 11, 46), (245, 158, 11, 255)),   # amber-500
}

# 팜맵 색상 — 대표님 지정 팔레트 (2026-04-30 갱신):
#   논(rice_paddy) = 녹색 #22C55E (green-500)
#   밭(dry_field)  = 주황 #F97316 (orange-500)
#   과수(orchard)   = 빨강 #EF4444 (red-500)
#   시설(greenhouse)= 파랑 #3B82F6 (blue-500)
CLASS_COLORS_FARMMAP: dict[str, tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = {
    # 비교 모드(both) — outline only (VWorld 와 겹쳐 그리기)
    "rice_paddy": ((34, 197, 94, 0), (34, 197, 94, 255)),       # green-500
    "dry_field": ((249, 115, 22, 0), (249, 115, 22, 255)),       # orange-500
    "orchard": ((239, 68, 68, 0), (239, 68, 68, 255)),           # red-500
    "greenhouse": ((59, 130, 246, 0), (59, 130, 246, 255)),      # blue-500
}

# 팜맵 단독(2-class) — fill 표시 (가시성 우선)
CLASS_COLORS_FARMMAP_SOLO_2: dict[str, tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = {
    "rice_paddy": ((34, 197, 94, 46), (34, 197, 94, 255)),       # green-500 (논)
    "dry_field": ((249, 115, 22, 46), (249, 115, 22, 255)),       # orange-500(밭)
}

# 팜맵 4-class — 논/밭/과수/시설
CLASS_COLORS_FARMMAP_4: dict[str, tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = {
    "rice_paddy": ((34, 197, 94, 46), (34, 197, 94, 255)),        # green-500  (논)
    "dry_field": ((249, 115, 22, 46), (249, 115, 22, 255)),        # orange-500 (밭)
    "orchard": ((239, 68, 68, 46), (239, 68, 68, 255)),            # red-500    (과수)
    "greenhouse": ((59, 130, 246, 46), (59, 130, 246, 255)),       # blue-500   (시설)
}

# 하위 호환 alias (기존 코드/테스트가 CLASS_COLORS 참조)
CLASS_COLORS = CLASS_COLORS_VWORLD

# 클래스 한글명 (텍스트 라벨용)
CLASS_NAMES_KR: dict[str, str] = {
    "rice_paddy": "논",
    "dry_field": "밭",
    "orchard": "과수",
    "greenhouse": "시설",
}


def world_to_pixel(
    x: float,
    y: float,
    transform_a: float,
    transform_e: float,
    transform_c: float,
    transform_f: float,
    scale: float,
) -> tuple[float, float]:
    """world(EPSG:5186) → 다운샘플된 PNG 픽셀 좌표.

    Args:
        x, y: world 좌표 (EPSG:5186 등).
        transform_a: pixel_width (positive).
        transform_e: pixel_height (negative for north-up).
        transform_c: origin_x (top-left).
        transform_f: origin_y (top-left).
        scale: 다운샘플 배수 (>= 1.0). 4096 max 면 큰 영상은 4~8 정도.

    Returns:
        (px, py) — 다운샘플된 PNG 의 (col, row) 좌표.
    """
    px_full = (x - transform_c) / transform_a
    py_full = (y - transform_f) / transform_e
    return px_full / scale, py_full / scale


def _read_geotiff_downsampled(
    tif_path: Path, max_size: int
) -> tuple[np.ndarray, dict]:
    """GeoTIFF 읽기 + 다운샘플.

    Returns:
        ``(rgb_array_HW3, meta)``:
            rgb_array: shape (H, W, 3) uint8.
            meta: dict with keys ``width, height, scale, transform_*, crs, src_width, src_height``.
    """
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(tif_path) as src:
        long_edge = max(src.width, src.height)
        scale = max(1.0, long_edge / max_size)
        out_w = max(1, int(src.width / scale))
        out_h = max(1, int(src.height / scale))

        # 첫 3 밴드만 읽기 (alpha 4번째 밴드는 무시)
        bands_to_read = min(3, src.count)
        data = src.read(
            indexes=list(range(1, bands_to_read + 1)),
            out_shape=(bands_to_read, out_h, out_w),
            resampling=Resampling.bilinear,
        )
        # (C, H, W) → (H, W, C). 1밴드면 그레이 → RGB 복제
        if data.shape[0] == 1:
            arr = np.repeat(data, 3, axis=0)
        elif data.shape[0] == 2:
            arr = np.concatenate([data, data[:1]], axis=0)
        else:
            arr = data[:3]
        rgb = np.moveaxis(arr, 0, -1).astype(np.uint8, copy=False)

        t = src.transform
        meta = {
            "width": out_w,
            "height": out_h,
            "scale": scale,
            "src_width": src.width,
            "src_height": src.height,
            "crs": str(src.crs) if src.crs else None,
            "transform_a": t.a,
            "transform_e": t.e,
            "transform_c": t.c,
            "transform_f": t.f,
        }
    return rgb, meta


def _draw_polygon_overlay(
    base_rgb: np.ndarray,
    geometries: list,
    class_names: list[str],
    meta: dict,
    color_map: dict | None = None,
    outline_width: int = 2,
    labels: list[str] | None = None,
    label_min_polygon_size_px: int = 60,
) -> "np.ndarray":  # uint8 (H, W, 3)
    """RGB base 위에 polygon 반투명 fill + outline 합성.

    Args:
        color_map: ``{class_name: ((fill_rgba), (outline_rgba))}`` — 미지정 시
            ``CLASS_COLORS_VWORLD`` (하위 호환).
    """
    from PIL import Image, ImageDraw
    from shapely.geometry import MultiPolygon, Polygon

    if color_map is None:
        color_map = CLASS_COLORS_VWORLD

    h, w = base_rgb.shape[:2]
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    scale = meta["scale"]
    a, e, c, f = (
        meta["transform_a"],
        meta["transform_e"],
        meta["transform_c"],
        meta["transform_f"],
    )

    def coords_to_pixels(coords) -> list[tuple[float, float]]:
        return [world_to_pixel(x, y, a, e, c, f, scale) for x, y in coords]

    def draw_one_polygon(poly: Polygon, fill_rgba, outline_rgba) -> None:
        if poly.is_empty:
            return
        ext = coords_to_pixels(poly.exterior.coords)
        if len(ext) < 3:
            return
        # fill alpha 가 0 이면 fill 안 그림 (외곽선만)
        if fill_rgba[3] > 0:
            draw.polygon(ext, fill=fill_rgba, outline=outline_rgba)
        # 외곽선 두께 보강 (PIL polygon outline 두께가 1px 고정)
        if outline_width >= 1:
            for i in range(len(ext) - 1):
                draw.line([ext[i], ext[i + 1]], fill=outline_rgba, width=outline_width)
        # interior holes
        for ring in poly.interiors:
            ring_pts = coords_to_pixels(ring.coords)
            if len(ring_pts) >= 3 and outline_width >= 1:
                for i in range(len(ring_pts) - 1):
                    draw.line(
                        [ring_pts[i], ring_pts[i + 1]],
                        fill=outline_rgba,
                        width=outline_width,
                    )

    # 텍스트 라벨용 폰트 — 한글 지원 (Windows 기본)
    label_font = None
    if labels is not None:
        try:
            from PIL import ImageFont

            label_font = ImageFont.truetype("malgun.ttf", 16)
        except OSError:
            try:
                label_font = ImageFont.truetype("gulim.ttc", 16)
            except OSError:
                from PIL import ImageFont

                label_font = ImageFont.load_default()

    label_iter = iter(labels) if labels is not None else None

    for geom, cls_name in zip(geometries, class_names, strict=True):
        colors = color_map.get(cls_name)
        if colors is None:
            if label_iter is not None:
                next(label_iter, None)
            continue
        fill_rgba, outline_rgba = colors
        if isinstance(geom, MultiPolygon):
            for sub in geom.geoms:
                if isinstance(sub, Polygon):
                    draw_one_polygon(sub, fill_rgba, outline_rgba)
        elif isinstance(geom, Polygon):
            draw_one_polygon(geom, fill_rgba, outline_rgba)

        # 텍스트 라벨 — polygon centroid 에 작게
        if label_iter is not None:
            text = next(label_iter, None)
            if text and not geom.is_empty:
                # polygon 의 외접 사각형이 충분히 큰지 (작은 폴리곤은 라벨 생략)
                minx, miny, maxx, maxy = geom.bounds
                px_minx, px_miny = world_to_pixel(minx, miny, a, e, c, f, scale)
                px_maxx, px_maxy = world_to_pixel(maxx, maxy, a, e, c, f, scale)
                box_w = abs(px_maxx - px_minx)
                box_h = abs(px_maxy - px_miny)
                if min(box_w, box_h) >= label_min_polygon_size_px:
                    cx, cy = geom.centroid.x, geom.centroid.y
                    px, py = world_to_pixel(cx, cy, a, e, c, f, scale)
                    # textbbox 로 실제 텍스트 영역 정확히 측정 (한글 폰트는 left/top
                    # offset 이 0 이 아닐 수 있으므로 4 좌표 모두 사용).
                    # anchor="lt" 명시로 ImageDraw 좌표 해석 일관 (PIL 버전 무관).
                    try:
                        bbox = draw.textbbox(
                            (0, 0), text, font=label_font, anchor="lt"
                        )
                    except (AttributeError, TypeError):
                        # 매우 오래된 PIL fallback
                        bbox = (0, 0, *draw.textsize(text, font=label_font))
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                    tx = int(px - text_w / 2)
                    ty = int(py - text_h / 2)
                    pad_x, pad_y = 4, 3
                    draw.rectangle(
                        [
                            tx - pad_x,
                            ty - pad_y,
                            tx + text_w + pad_x,
                            ty + text_h + pad_y,
                        ],
                        fill=(0, 0, 0, 160),
                    )
                    draw.text(
                        (tx, ty),
                        text,
                        fill=(255, 255, 255, 255),
                        font=label_font,
                        anchor="lt",
                    )

    base_pil = Image.fromarray(base_rgb).convert("RGBA")
    blended = Image.alpha_composite(base_pil, overlay).convert("RGB")
    return np.asarray(blended)


def _draw_legend(
    width: int,
    palettes_in: list[tuple[str, dict]] | None = None,
    height: int | None = None,
) -> "np.ndarray":
    """클래스 색상 범례 PNG 생성.

    Args:
        palettes_in: ``[(source_label, color_map), ...]`` — 각 행에 표시할 색상 팔레트.
            None 이면 ``[("VWorld", CLASS_COLORS_VWORLD)]``.
    """
    from PIL import Image, ImageDraw

    if palettes_in is None:
        palettes_in = [("VWorld", CLASS_COLORS_VWORLD)]

    try:
        from PIL import ImageFont

        font = ImageFont.truetype("malgun.ttf", 12)
    except OSError:
        from PIL import ImageFont

        font = ImageFont.load_default()

    line_h = 40
    h = height if height is not None else line_h * len(palettes_in) + 16

    legend = Image.new("RGB", (width, h), (240, 240, 240))
    draw = ImageDraw.Draw(legend)
    swatch_w, swatch_h = 50, 24
    left_margin = 20
    label_pad = 8
    item_gap = 130

    for row_idx, (src_label, palette) in enumerate(palettes_in):
        y = 8 + row_idx * line_h + (line_h - swatch_h) // 2
        x = left_margin
        draw.text((x, y + 4), f"{src_label}:", fill=(20, 20, 20), font=font)
        x += 80
        for cls_name, kr in CLASS_NAMES_KR.items():
            colors = palette.get(cls_name)
            if colors is None:
                continue
            fill, outline = colors
            if fill[3] > 0:
                draw.rectangle(
                    [x, y, x + swatch_w, y + swatch_h],
                    fill=fill[:3],
                    outline=outline[:3],
                    width=2,
                )
            else:
                draw.rectangle(
                    [x, y, x + swatch_w, y + swatch_h],
                    fill=None,
                    outline=outline[:3],
                    width=2,
                )
            draw.text((x + swatch_w + label_pad, y + 4), kr, fill=(20, 20, 20), font=font)
            x += swatch_w + item_gap

    return np.asarray(legend)


def _save_geotiff_overlay(
    rgb_hw3: np.ndarray, meta: dict, out_path: Path
) -> None:
    """다운샘플된 RGB 합성 결과를 GeoTIFF (좌표 메타 포함) 로 저장.

    PNG 와 동일한 합성 결과지만 좌표계(EPSG:5186 등) + 다운샘플 transform 보존.
    QGIS/ArcGIS 등 GIS 도구에서 그대로 raster 레이어로 사용 가능.
    """
    import rasterio
    from rasterio.transform import Affine

    h, w, _ = rgb_hw3.shape
    profile: dict = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 3,
        "dtype": "uint8",
        "compress": "lzw",
        "tiled": True,
    }
    if meta.get("crs"):
        ds_transform = Affine(
            meta["transform_a"] * meta["scale"],
            0,
            meta["transform_c"],
            0,
            meta["transform_e"] * meta["scale"],
            meta["transform_f"],
        )
        profile["crs"] = meta["crs"]
        profile["transform"] = ds_transform
    else:
        log.warning("[%s] no CRS — GeoTIFF will lack spatial reference", out_path.name)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(np.moveaxis(rgb_hw3, -1, 0))
    log.info(
        "[%s] wrote GeoTIFF (%.0f KB)",
        out_path.name,
        out_path.stat().st_size / 1024,
    )


def render_preview(
    tif_path: Path,
    sources: list[dict],
    output_path: Path,
    max_size: int,
    *,
    also_tif: bool = False,
) -> tuple[int, int]:
    """GeoTIFF 1개 + (다중) polygon source → PNG (옵션: GeoTIFF 도).

    Args:
        sources: 각 source 의 dict ``{
            "geometries": list[shapely],
            "class_names": list[str],
            "color_map": dict,
            "source_label": str,   # 범례에 표시 (예: "VWorld", "Farmmap 4-class")
            "labels": list[str] | None,  # 폴리곤별 텍스트 라벨 (옵션)
        }`` 리스트.
        output_path: PNG 출력 경로.
        max_size: 다운샘플 long edge 최대 px.
        also_tif: True 면 GeoTIFF 도 출력 (좌표 메타 포함).

    Returns:
        ``(polygon_count_total, png_size_kb)``.
    """
    from PIL import Image

    log.info("[%s] reading GeoTIFF...", tif_path.name)
    base_rgb, meta = _read_geotiff_downsampled(tif_path, max_size)
    log.info(
        "[%s] downsampled: %dx%d (scale=%.2f)",
        tif_path.name,
        meta["width"],
        meta["height"],
        meta["scale"],
    )

    polygon_total = sum(len(s["geometries"]) for s in sources)
    log.info(
        "[%s] %d polygon(s) total across %d source(s)",
        tif_path.name,
        polygon_total,
        len(sources),
    )

    blended = base_rgb
    for src in sources:
        if not src["geometries"]:
            continue
        blended = _draw_polygon_overlay(
            blended,
            src["geometries"],
            src["class_names"],
            meta,
            color_map=src["color_map"],
            labels=src.get("labels"),
        )

    # 범례 — 사용된 source 의 (label, palette) 페어
    palettes_in = [(s["source_label"], s["color_map"]) for s in sources]
    legend = _draw_legend(blended.shape[1], palettes_in=palettes_in)
    final = np.concatenate([blended, legend], axis=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(final).save(output_path, format="PNG", optimize=True)
    size_kb = output_path.stat().st_size / 1024
    log.info("[%s] wrote PNG (%.0f KB)", output_path.name, size_kb)

    if also_tif:
        tif_out = output_path.with_suffix(".tif")
        _save_geotiff_overlay(blended, meta, tif_out)

    return polygon_total, int(size_kb)


def _features_from_gpkg(
    tif_path: Path, gpkg_path: Path
) -> tuple[list, list[str]]:
    """GeoPackage layer (= GeoTIFF stem) 에서 polygon + class_name 추출."""
    import geopandas as gpd

    layer_name = tif_path.stem
    try:
        gdf = gpd.read_file(gpkg_path, layer=layer_name)
    except (ValueError, Exception) as e:  # noqa: BLE001
        log.warning(
            "[%s] layer '%s' not found in %s: %s",
            tif_path.name,
            layer_name,
            gpkg_path,
            e,
        )
        return [], []
    geometries = list(gdf.geometry)
    class_names = (
        list(gdf["class_name"]) if "class_name" in gdf.columns else ["other"] * len(gdf)
    )
    return geometries, class_names


def _features_from_vworld(tif_path: Path, client) -> tuple[list, list[str], object]:
    """VWorld API 를 직접 호출해 GeoTIFF 의 polygon + class_name 추출.

    GeoPackage 를 거치지 않는 통합 흐름. ``cadastral_bootstrap.process_geotiff`` 재사용.
    """
    # 같은 패키지의 다른 모듈을 직접 실행 환경에서도 import 하기 위해 sys.path 보강
    import sys

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.labeling.cadastral_bootstrap import process_geotiff  # noqa: E402

    feats, stats, _ = process_geotiff(tif_path, client)
    geometries = [f["geometry"] for f in feats]
    class_names = [f["properties"]["class_name"] for f in feats]
    return geometries, class_names, stats


def _features_from_farmmap(
    tif_path: Path,
    farmmap_shps: list[Path],
    *,
    classes: int = 2,
    with_labels: bool = False,
) -> tuple[list, list[str], list[str] | None, object]:
    """팜맵 SHP 다중 입력 → GeoTIFF 영역 안의 polygon + class_name + (옵션) 텍스트 라벨.

    Args:
        classes: 2 (논/밭) 또는 4 (논/밭/과수/시설).
        with_labels: True 면 폴리곤별 텍스트 라벨 (지번 + 농지분류) 반환.
    """
    import sys

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.labeling.cadastral_bootstrap import get_valid_region  # noqa: E402
    from src.labeling.farmmap_loader import (  # noqa: E402
        FARMMAP_CLASS_MAP_2,
        FARMMAP_CLASS_MAP_4,
        load_farmmap_polygons,
    )
    import rasterio  # noqa: E402

    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        src_crs = str(src.crs) if src.crs else "EPSG:5186"

    valid_region = get_valid_region(tif_path, src_crs)

    class_map = FARMMAP_CLASS_MAP_4 if classes == 4 else FARMMAP_CLASS_MAP_2
    feats, stats = load_farmmap_polygons(
        farmmap_shps,
        (bounds.left, bounds.bottom, bounds.right, bounds.top),
        src_crs,
        valid_region_dst_crs=valid_region,
        class_map=class_map,
    )
    geometries = [f["geometry"] for f in feats]
    class_names = [f["properties"]["class_name"] for f in feats]

    labels: list[str] | None = None
    if with_labels:
        labels = []
        for f in feats:
            p = f["properties"]
            emd = (p.get("emd_nm") or "").strip()
            lnm = (p.get("lnm") or "").strip()
            intpr = (p.get("intpr_nm") or "").strip()
            # 예: "보문동 922-9 | 논" 또는 "보문동 | 논" (지번 없을 때)
            addr = f"{emd} {lnm}".strip()
            labels.append(f"{addr} | {intpr}" if addr else intpr)

    return geometries, class_names, labels, stats


def main() -> None:
    """CLI 진입점.

    Source 모드 (``--source``):
      - ``vworld`` (기본): VWorld API 직접 호출. ``VWORLD_API_KEY`` env 필요.
      - ``farmmap``: 팜맵 SHP 입력 (``--farmmap-shp`` 또는 ``--farmmap-dir``).
      - ``both``: 둘 다 그리기 (검수용 비교 모드).

    추가 입력 모드 (호환):
      - ``--gpkg path.gpkg``: 기존 GeoPackage layer (vworld 결과). source=vworld 만.
    """
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="GeoTIFF → polygon overlay PNG/TIF (VWorld + 팜맵)"
    )
    parser.add_argument(
        "--geotiff-dir", type=Path, required=True, help="GeoTIFF 폴더 또는 단일 .tif"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="출력 디렉토리 (PNG/TIF)"
    )
    parser.add_argument(
        "--source",
        choices=["vworld", "farmmap", "both"],
        default="vworld",
        help="polygon source — 기본 vworld",
    )
    parser.add_argument(
        "--gpkg",
        type=Path,
        default=None,
        help="(옵셔널, source=vworld 한정) GeoPackage(.gpkg) 사용. 미지정 시 VWorld 직접 호출.",
    )
    parser.add_argument(
        "--farmmap-shp",
        type=Path,
        nargs="+",
        default=None,
        help="(source=farmmap/both) 팜맵 SHP 파일 1개 이상",
    )
    parser.add_argument(
        "--farmmap-dir",
        type=Path,
        default=None,
        help="(source=farmmap/both) 팜맵 SHP 폴더 (FARM_*.shp 자동 검색)",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=4096,
        help="다운샘플 후 long edge 최대 px (기본 4096)",
    )
    parser.add_argument(
        "--also-tif",
        action="store_true",
        help="PNG 외에 GeoTIFF (좌표 메타 포함) 도 함께 출력",
    )
    parser.add_argument(
        "--farmmap-classes",
        type=int,
        choices=[2, 4],
        default=2,
        help="팜맵 클래스 수 — 2(논/밭) 또는 4(논/밭/과수/시설). 기본 2.",
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="(팜맵) 폴리곤 위에 지번 + 농지분류 텍스트 라벨 표시",
    )
    args = parser.parse_args()

    if args.geotiff_dir.is_file():
        geotiffs = [args.geotiff_dir]
    elif args.geotiff_dir.is_dir():
        geotiffs = sorted(
            p
            for p in args.geotiff_dir.iterdir()
            if p.is_file() and p.suffix.lower() in (".tif", ".tiff")
        )
    else:
        raise SystemExit(f"Not found: {args.geotiff_dir}")
    if not geotiffs:
        raise SystemExit(f"No GeoTIFF in {args.geotiff_dir}")

    use_vworld = args.source in ("vworld", "both")
    use_farmmap = args.source in ("farmmap", "both")
    use_gpkg = args.gpkg is not None
    if use_gpkg and not args.gpkg.exists():
        raise SystemExit(f"GeoPackage not found: {args.gpkg}")
    if use_gpkg and not use_vworld:
        raise SystemExit("--gpkg can only be used with --source vworld or both")

    # VWorld client (필요 시)
    vworld_client = None
    if use_vworld and not use_gpkg:
        api_key = os.environ.get("VWORLD_API_KEY")
        if not api_key:
            raise SystemExit(
                "VWORLD_API_KEY env missing — see SECRETS.md, or pass --gpkg"
            )
        domain = os.environ.get("VWORLD_DOMAIN", "http://localhost:8080")
        import sys

        root = Path(__file__).resolve().parents[2]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from src.labeling.cadastral_bootstrap import VWorldClient  # noqa: E402

        vworld_client = VWorldClient(api_key=api_key, domain=domain)

    # 팜맵 SHP 리스트 정리 (필요 시)
    farmmap_shps: list[Path] = []
    if use_farmmap:
        if args.farmmap_shp:
            farmmap_shps = list(args.farmmap_shp)
        elif args.farmmap_dir:
            import sys

            root = Path(__file__).resolve().parents[2]
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            from src.labeling.farmmap_loader import discover_farmmap_shps  # noqa: E402

            farmmap_shps = discover_farmmap_shps(args.farmmap_dir)
        else:
            raise SystemExit(
                "source=farmmap/both → --farmmap-shp 또는 --farmmap-dir 중 하나 필수"
            )

    # 팜맵 색상 팔레트 결정 (단독 모드 = fill 표시, both = outline only)
    farmmap_color_map: dict
    farmmap_label: str
    if args.farmmap_classes == 4:
        farmmap_color_map = CLASS_COLORS_FARMMAP_4
        farmmap_label = "팜맵 4-class"
    elif use_vworld:
        farmmap_color_map = CLASS_COLORS_FARMMAP  # outline only (비교 모드)
        farmmap_label = "팜맵"
    else:
        farmmap_color_map = CLASS_COLORS_FARMMAP_SOLO_2  # fill 표시 (단독)
        farmmap_label = "팜맵"

    log.info(
        "Rendering %d preview(s) → %s "
        "(source=%s, farmmap_classes=%d, show_labels=%s, also_tif=%s)",
        len(geotiffs),
        args.output_dir,
        args.source,
        args.farmmap_classes,
        args.show_labels,
        args.also_tif,
    )

    total_polygons = 0
    total_size_kb = 0
    for tif in geotiffs:
        sources: list[dict] = []

        if use_vworld:
            if use_gpkg:
                gv, cv = _features_from_gpkg(tif, args.gpkg)
                stats_v = None
            else:
                gv, cv, stats_v = _features_from_vworld(tif, vworld_client)
            if stats_v is not None:
                log.info(
                    "[%s] VWorld total=%d kept=%d (skip jimok=%d, outside=%d) — %s",
                    tif.name,
                    stats_v.polygons_total,
                    stats_v.polygons_kept,
                    stats_v.polygons_skipped_jimok,
                    stats_v.polygons_skipped_outside,
                    stats_v.per_class,
                )
            sources.append(
                {
                    "geometries": gv,
                    "class_names": cv,
                    "color_map": CLASS_COLORS_VWORLD,
                    "source_label": "VWorld",
                    "labels": None,
                }
            )

        if use_farmmap:
            gf, cf, lf, stats_f = _features_from_farmmap(
                tif,
                farmmap_shps,
                classes=args.farmmap_classes,
                with_labels=args.show_labels,
            )
            log.info(
                "[%s] Farmmap(%d-class) total=%d kept=%d (skip class=%d, outside=%d) — %s",
                tif.name,
                args.farmmap_classes,
                stats_f.polygons_total,
                stats_f.polygons_kept,
                stats_f.polygons_skipped_class,
                stats_f.polygons_skipped_outside,
                stats_f.per_class,
            )
            sources.append(
                {
                    "geometries": gf,
                    "class_names": cf,
                    "color_map": farmmap_color_map,
                    "source_label": farmmap_label,
                    "labels": lf,
                }
            )

        out_png = args.output_dir / f"{tif.stem}.png"
        n, size_kb = render_preview(
            tif, sources, out_png, args.max_size, also_tif=args.also_tif
        )
        total_polygons += n
        total_size_kb += size_kb

    log.info("=" * 60)
    log.info(
        "DONE — %d preview(s), %d polygon(s) total across all sources, %.1f MB PNG",
        len(geotiffs),
        total_polygons,
        total_size_kb / 1024,
    )


if __name__ == "__main__":
    main()
