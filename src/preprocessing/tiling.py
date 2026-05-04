"""정사영상 (+ 옵션 mask) 타일링 스크립트.

영상만:
    python src/preprocessing/tiling.py \\
        --input data/raw \\
        --output data/tiles \\
        --tile-size 2048 --overlap 512 \\
        --target-crs EPSG:5186

영상 + mask 동시 (학습 데이터셋 빌드):
    python src/preprocessing/tiling.py \\
        --input "img/.../경주/" \\
        --mask-input "data/labels/팜맵_논밭과수시설_AI학습용/" \\
        --output data/tiles_masked \\
        --tile-size 2048 --overlap 512 \\
        --skip-empty-mask-ratio 0.8

  → output/images/{stem}/{stem}_r####_c####.png
    output/masks/{stem}/{stem}_r####_c####.png
    output/{stem}/meta.json

mask 가 ignore (255) 픽셀 비율 X 이상인 타일은 ``--skip-empty-mask-ratio`` 로 자동 skip.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.windows import Window

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


@dataclass
class TileMeta:
    """타일 하나에 대한 메타데이터."""

    source_file: str
    tile_index: int
    row: int
    col: int
    tile_size: int
    source_crs: str
    target_crs: str
    # 타일 좌하단~우상단 bbox (target_crs 기준)
    bbox_minx: float
    bbox_miny: float
    bbox_maxx: float
    bbox_maxy: float
    gsd_cm: float


def estimate_gsd_cm(src: rasterio.io.DatasetReader) -> float:
    """GSD(픽셀당 지표면 거리)를 cm 단위로 추정.

    투영 좌표계(미터) 기준. 경위도(EPSG:4326)면 대략적 변환.
    """
    if src.crs and src.crs.is_geographic:
        # WGS84인 경우 위도 기준 근사: 1도 ≈ 111km
        pixel_deg = abs(src.transform.a)
        return pixel_deg * 111_000 * 100
    return abs(src.transform.a) * 100  # m → cm


def reproject_to_target(src_path: Path, target_crs: str) -> tuple[np.ndarray, rasterio.transform.Affine, str]:
    """원본을 target_crs로 reproject해서 메모리에 로드."""
    with rasterio.open(src_path) as src:
        src_crs = str(src.crs)
        if src_crs == target_crs:
            return src.read(), src.transform, src_crs

        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        data = np.zeros((src.count, height, width), dtype=src.dtypes[0])
        for band_idx in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, band_idx),
                destination=data[band_idx - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear,
            )
        return data, transform, src_crs


def tile_image(
    data: np.ndarray,
    transform: rasterio.transform.Affine,
    tile_size: int,
    overlap: int,
) -> list[tuple[int, int, int, Window, tuple[float, float, float, float]]]:
    """픽셀 데이터를 타일로 분할. 각 타일의 bbox 포함."""
    _, height, width = data.shape
    step = tile_size - overlap
    tiles = []
    idx = 0
    for row_start in range(0, height, step):
        for col_start in range(0, width, step):
            row_end = min(row_start + tile_size, height)
            col_end = min(col_start + tile_size, width)
            # 가장자리 타일이 tile_size보다 작으면 padding
            window = Window(
                col_off=col_start,
                row_off=row_start,
                width=col_end - col_start,
                height=row_end - row_start,
            )
            minx, miny = transform * (col_start, row_end)
            maxx, maxy = transform * (col_end, row_start)
            tiles.append(
                (idx, row_start // step, col_start // step, window, (minx, miny, maxx, maxy))
            )
            idx += 1
    return tiles


def save_tile(
    data: np.ndarray,
    window: Window,
    tile_size: int,
    output_path: Path,
) -> None:
    """타일 데이터를 PNG로 저장. 가장자리 padding 처리."""
    tile = data[:, window.row_off : window.row_off + window.height, window.col_off : window.col_off + window.width]
    # RGB 정사영상 가정 (3밴드). 밴드 순서가 RGB인지 BGR인지 원본 확인 필요.
    if tile.shape[0] >= 3:
        tile = tile[:3]
    else:
        tile = np.repeat(tile[:1], 3, axis=0)

    # Padding
    if tile.shape[1] < tile_size or tile.shape[2] < tile_size:
        padded = np.zeros((3, tile_size, tile_size), dtype=tile.dtype)
        padded[:, : tile.shape[1], : tile.shape[2]] = tile
        tile = padded

    # uint8로 정규화
    if tile.dtype != np.uint8:
        tile = np.clip(tile / max(tile.max(), 1) * 255, 0, 255).astype(np.uint8)

    img = Image.fromarray(tile.transpose(1, 2, 0), "RGB")
    img.save(output_path, "PNG")


def save_mask_tile(
    mask: np.ndarray,
    window: Window,
    tile_size: int,
    output_path: Path,
    ignore_value: int = 255,
) -> tuple[float, float]:
    """mask 타일을 단일 밴드 PNG (uint8) 로 저장.

    Args:
        mask: (H, W) uint8 — 학습용 mask (0~N-1 클래스 + ignore_value).
        window: 자를 윈도우.
        tile_size: 정사각 타일 크기.
        output_path: 출력 PNG 경로.
        ignore_value: padding 채울 값 (기본 255 = mmsegmentation ignore_index).

    Returns:
        ``(ignore_ratio, labeled_ratio)`` — 타일 내 ignore 픽셀 비율 + 라벨링 비율.
    """
    tile = mask[
        window.row_off : window.row_off + window.height,
        window.col_off : window.col_off + window.width,
    ]
    # padding — ignore_value 로 채움 (외곽 = 학습 제외)
    if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
        padded = np.full((tile_size, tile_size), ignore_value, dtype=np.uint8)
        padded[: tile.shape[0], : tile.shape[1]] = tile
        tile = padded

    # 통계 (skip 판정용)
    total = tile.size
    ignore_count = int((tile == ignore_value).sum())
    ignore_ratio = ignore_count / total
    labeled_ratio = 1.0 - ignore_ratio

    img = Image.fromarray(tile, "L")
    img.save(output_path, "PNG")
    return ignore_ratio, labeled_ratio


def load_mask_geotiff(mask_path: Path) -> np.ndarray:
    """mask GeoTIFF 를 (H, W) uint8 numpy array 로 로드 (단일 밴드)."""
    with rasterio.open(mask_path) as src:
        if src.count != 1:
            raise ValueError(
                f"mask GeoTIFF must be single-band, got {src.count} bands: {mask_path}"
            )
        return src.read(1)


def discover_mask_for(stem: str, mask_input: Path) -> Path | None:
    """영상 stem 매칭되는 mask GeoTIFF 자동 발견.

    Args:
        stem: 영상 GeoTIFF stem (예: ``"2D정사영상_경주시_5cm_230816_1"``).
        mask_input: mask 파일 또는 디렉토리.

    Returns:
        매칭 mask 경로. 단일 파일 인풋이면 그대로, 디렉토리면 ``{stem}.tif`` 우선 검색.
    """
    if mask_input.is_file():
        return mask_input
    candidates = [
        mask_input / f"{stem}.tif",
        mask_input / f"{stem}.tiff",
        mask_input / f"{stem}_mask.tif",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def process_file(
    src_path: Path,
    output_dir: Path,
    tile_size: int,
    overlap: int,
    target_crs: str,
    mask_path: Path | None = None,
    skip_empty_mask_ratio: float = 0.0,
    ignore_value: int = 255,
) -> list[TileMeta]:
    """단일 GeoTIFF를 타일로 분할 + 저장. mask_path 지정 시 영상 + mask 동시.

    Args:
        mask_path: 학습용 mask GeoTIFF (단일 밴드 uint8). None 이면 영상만.
            mask 가 영상과 같은 transform/CRS/size 라고 가정.
        skip_empty_mask_ratio: mask 의 ignore 비율이 이 값 이상이면 타일 skip.
            0.0 이면 비활성. 0.8 권장 (라벨 < 20% 인 타일 학습에서 제외).

    Returns:
        저장된 타일 meta 리스트 (skip 된 타일 제외).
    """
    log.info("Processing %s%s", src_path, f" + mask {mask_path.name}" if mask_path else "")
    data, transform, src_crs = reproject_to_target(src_path, target_crs)
    with rasterio.open(src_path) as src:
        gsd_cm = estimate_gsd_cm(src)

    mask_arr: np.ndarray | None = None
    if mask_path is not None:
        mask_arr = load_mask_geotiff(mask_path)
        # 영상과 mask shape 일치 확인 (reproject 안 한 원본 size 기준)
        with rasterio.open(src_path) as src:
            if (mask_arr.shape[0], mask_arr.shape[1]) != (src.height, src.width):
                raise ValueError(
                    f"mask shape {mask_arr.shape} != image shape "
                    f"({src.height}, {src.width}) — same transform required"
                )

    tiles = tile_image(data, transform, tile_size, overlap)
    meta_list: list[TileMeta] = []
    stem = src_path.stem

    # 출력 구조: mask 있으면 images/{stem}/ + masks/{stem}/, 없으면 {stem}/ (기존 호환)
    if mask_arr is not None:
        img_dir = output_dir / "images" / stem
        mask_dir = output_dir / "masks" / stem
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        meta_dir = output_dir / stem
        meta_dir.mkdir(parents=True, exist_ok=True)
    else:
        img_dir = output_dir / stem
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir = None
        meta_dir = img_dir

    skipped = 0
    for idx, row, col, window, bbox in tiles:
        tile_name = f"{stem}_r{row:04d}_c{col:04d}.png"
        # mask 있으면 ignore 비율 먼저 검사 → skip 판정
        if mask_arr is not None and skip_empty_mask_ratio > 0.0:
            mask_tile = mask_arr[
                window.row_off : window.row_off + window.height,
                window.col_off : window.col_off + window.width,
            ]
            # padding 영역도 ignore 로 가정해서 정확히 계산
            total_full = tile_size * tile_size
            ignore_in_actual = int((mask_tile == ignore_value).sum())
            padding = total_full - mask_tile.size
            ignore_ratio_full = (ignore_in_actual + padding) / total_full
            if ignore_ratio_full >= skip_empty_mask_ratio:
                skipped += 1
                continue

        tile_path = img_dir / tile_name
        save_tile(data, window, tile_size, tile_path)

        if mask_arr is not None and mask_dir is not None:
            mask_tile_path = mask_dir / tile_name
            save_mask_tile(mask_arr, window, tile_size, mask_tile_path, ignore_value=ignore_value)

        meta_list.append(
            TileMeta(
                source_file=str(src_path.name),
                tile_index=idx,
                row=row,
                col=col,
                tile_size=tile_size,
                source_crs=src_crs,
                target_crs=target_crs,
                bbox_minx=bbox[0],
                bbox_miny=bbox[1],
                bbox_maxx=bbox[2],
                bbox_maxy=bbox[3],
                gsd_cm=gsd_cm,
            )
        )

    meta_path = meta_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(m) for m in meta_list], f, ensure_ascii=False, indent=2)

    if mask_arr is not None:
        log.info(
            "  → %d tiles saved (skipped %d, threshold ignore≥%.0f%%) to images/%s + masks/%s",
            len(meta_list),
            skipped,
            skip_empty_mask_ratio * 100,
            stem,
            stem,
        )
    else:
        log.info("  → %d tiles saved to %s", len(meta_list), img_dir)
    return meta_list


def main() -> None:
    parser = argparse.ArgumentParser(description="정사영상 (+ 옵션 mask) 타일링")
    parser.add_argument("--input", type=Path, required=True, help="GeoTIFF 또는 디렉토리")
    parser.add_argument("--output", type=Path, required=True, help="출력 디렉토리")
    parser.add_argument("--tile-size", type=int, default=2048)
    parser.add_argument("--overlap", type=int, default=512)
    parser.add_argument("--target-crs", default="EPSG:5186")
    parser.add_argument(
        "--mask-input",
        type=Path,
        default=None,
        help="(옵션) 학습용 mask GeoTIFF 파일 또는 폴더. 영상 stem 매칭 자동.",
    )
    parser.add_argument(
        "--skip-empty-mask-ratio",
        type=float,
        default=0.0,
        help="mask 의 ignore(255) 비율이 이 값 이상이면 타일 skip. 0.0=비활성.",
    )
    parser.add_argument("--ignore-value", type=int, default=255)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    if args.input.is_file():
        files = [args.input]
    else:
        files = sorted(args.input.rglob("*.tif")) + sorted(args.input.rglob("*.tiff"))

    if not files:
        log.warning("No .tif files under %s", args.input)
        return

    total_tiles = 0
    for f in files:
        mask_path = None
        if args.mask_input is not None:
            mask_path = discover_mask_for(f.stem, args.mask_input)
            if mask_path is None:
                log.warning(
                    "[%s] mask not found in %s — skipping mask for this file",
                    f.stem,
                    args.mask_input,
                )
        meta = process_file(
            f,
            args.output,
            args.tile_size,
            args.overlap,
            args.target_crs,
            mask_path=mask_path,
            skip_empty_mask_ratio=args.skip_empty_mask_ratio,
            ignore_value=args.ignore_value,
        )
        total_tiles += len(meta)

    log.info("Done. %d files processed, %d tiles total.", len(files), total_tiles)


if __name__ == "__main__":
    main()
