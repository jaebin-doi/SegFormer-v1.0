"""Region-based train/val/test split.

cvat_to_dataset.py 의 flat 출력(images/, masks/)을 학습 config 가 기대하는
{train,val,test} 디렉토리 구조로 분할한다.

타일 단위 셔플 금지 — 동일 GeoTIFF(source_file)에서 나온 타일이 train/val/test 에
흩뿌려지면 spatial leakage 발생. 따라서 GeoTIFF 단위로 셔플 후 분배한다.

split 정책 (`docs/training/대표님_보고서_2026-04-28.md` §4.1 / TODO §2.2):
- n=1  : 1/0/0  — sanity only (--allow-sanity 필수)
- n=2  : 1/1/0  — sanity only (--allow-sanity 필수)
- n=3  : 1/1/1
- n=4  : 2/1/1
- n=5  : 3/1/1
- n=6  : 4/1/1   (현 case: farmland-5cm-v1.0)
- n=7  : 5/1/1
- n>=8 : 비율 적용 + max(1, ...) 보장

사용:
    python src/preprocessing/split_dataset.py \\
        --images-root data/dataset/farmland-v0.1-flat/images \\
        --masks-root  data/dataset/farmland-v0.1-flat/masks \\
        --metas       data/tiles/sample01/meta.json \\
                      data/tiles/sample02/meta.json \\
                      data/tiles/sample03/meta.json \\
        --output      data/dataset/farmland-5cm-v1.0 \\
        --train-ratio 0.70 --val-ratio 0.15 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)


@dataclass
class SplitResult:
    """split 결과 — manifest 저장 + 호출자 검증용."""

    train_sources: list[str] = field(default_factory=list)
    val_sources: list[str] = field(default_factory=list)
    test_sources: list[str] = field(default_factory=list)
    train_tiles: list[str] = field(default_factory=list)
    val_tiles: list[str] = field(default_factory=list)
    test_tiles: list[str] = field(default_factory=list)
    skipped_tiles: list[str] = field(default_factory=list)
    seed: int = 42

    def manifest(self) -> dict:
        return {
            "seed": self.seed,
            "splits": {
                "train": {
                    "source_files": self.train_sources,
                    "tile_count": len(self.train_tiles),
                },
                "val": {
                    "source_files": self.val_sources,
                    "tile_count": len(self.val_tiles),
                },
                "test": {
                    "source_files": self.test_sources,
                    "tile_count": len(self.test_tiles),
                },
            },
            "skipped": {
                "count": len(self.skipped_tiles),
                "tiles": self.skipped_tiles,
            },
        }


def tile_stem(meta_entry: dict) -> str:
    """tiling.py 의 파일명 규칙 재구성.

    `tiling.py:162` 와 동일해야 한다 (한쪽 변경 시 양쪽 동기 갱신).
    형식: `{source_stem}_r{row:04d}_c{col:04d}`
    """
    src_stem = Path(meta_entry["source_file"]).stem
    return f"{src_stem}_r{meta_entry['row']:04d}_c{meta_entry['col']:04d}"


def load_metas(meta_paths: Iterable[Path]) -> dict[str, list[str]]:
    """다중 meta.json 을 읽어 source_file 단위로 타일 stem 그룹핑.

    Args:
        meta_paths: tiling.py 가 GeoTIFF 별로 생성한 meta.json 들의 경로 목록.

    Returns:
        {source_file (str): [tile_stem1, tile_stem2, ...]}
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for meta_path in meta_paths:
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json not found: {meta_path}")
        entries = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            raise ValueError(f"{meta_path} 의 최상위는 list 여야 함 (TileMeta 배열)")
        for entry in entries:
            if "source_file" not in entry:
                raise ValueError(f"{meta_path} 의 entry 에 source_file 누락: {entry}")
            groups[entry["source_file"]].append(tile_stem(entry))
    return dict(groups)


def assign_split_counts(
    n: int, train_ratio: float, val_ratio: float, allow_sanity: bool
) -> tuple[int, int, int]:
    """GeoTIFF 개수 별 train/val/test 개수 결정.

    n=1, 2 는 sanity only. allow_sanity=False 면 ValueError.
    n>=3 은 train/val/test 모두 최소 1개 보장.
    """
    if n < 1:
        raise ValueError("source_file 0개 — split 불가")

    if n == 1:
        if not allow_sanity:
            raise ValueError(
                "GeoTIFF 1개 → 정식 학습 불가. --allow-sanity 명시 시 train 단독 모드"
            )
        return 1, 0, 0

    if n == 2:
        if not allow_sanity:
            raise ValueError(
                "GeoTIFF 2개 → 정식 KPI 평가 불가 (test 비어있음). --allow-sanity 명시 시 1/1/0"
            )
        return 1, 1, 0

    # n >= 3 : train/val/test 모두 non-empty 보장
    test_n = max(1, round(n * (1.0 - train_ratio - val_ratio)))
    val_n = max(1, round(n * val_ratio))
    train_n = n - val_n - test_n
    if train_n < 1:
        raise ValueError(
            f"비율 조정 필요: n={n} train_ratio={train_ratio} val_ratio={val_ratio} "
            f"→ train_n={train_n}"
        )
    return train_n, val_n, test_n


def split_by_region(
    images_root: Path,
    masks_root: Path,
    meta_paths: list[Path],
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    allow_sanity: bool = False,
    force: bool = False,
) -> SplitResult:
    """GeoTIFF 단위로 train/val/test 분할 + 파일 복사 + manifest 저장.

    Returns:
        SplitResult — split 별 source_files / tile stems / skipped 목록.
    """
    # output 디렉토리 정책
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(
                f"output_dir 이미 존재: {output_dir}. 덮어쓰려면 --force 명시"
            )
        log.warning("output_dir overwriting (--force): %s", output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 다중 meta 통합
    groups = load_metas(meta_paths)
    sources = sorted(groups.keys())
    n = len(sources)
    log.info("Loaded %d source_file group(s) from %d meta.json", n, len(meta_paths))

    # 2. split 개수 결정
    train_n, val_n, test_n = assign_split_counts(n, train_ratio, val_ratio, allow_sanity)
    log.info("Split counts: train=%d val=%d test=%d", train_n, val_n, test_n)

    # 3. GeoTIFF 단위 셔플 + 분배
    rng = random.Random(seed)
    shuffled = sources[:]
    rng.shuffle(shuffled)
    train_src = shuffled[:train_n]
    val_src = shuffled[train_n : train_n + val_n]
    test_src = shuffled[train_n + val_n : train_n + val_n + test_n]

    result = SplitResult(
        train_sources=sorted(train_src),
        val_sources=sorted(val_src),
        test_sources=sorted(test_src),
        seed=seed,
    )

    # 4. 디렉토리 생성 + 파일 복사
    splits_data: list[tuple[str, list[str], list[str]]] = [
        ("train", train_src, result.train_tiles),
        ("val", val_src, result.val_tiles),
        ("test", test_src, result.test_tiles),
    ]
    for split_name, srcs, tile_acc in splits_data:
        (output_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (output_dir / "masks" / split_name).mkdir(parents=True, exist_ok=True)
        for src in srcs:
            for stem in groups[src]:
                img_src = images_root / f"{stem}.png"
                mask_src = masks_root / f"{stem}.png"
                if not img_src.exists() or not mask_src.exists():
                    log.warning(
                        "Tile missing — skipped: stem=%s img_exists=%s mask_exists=%s",
                        stem, img_src.exists(), mask_src.exists()
                    )
                    result.skipped_tiles.append(stem)
                    continue
                shutil.copy2(img_src, output_dir / "images" / split_name / f"{stem}.png")
                shutil.copy2(mask_src, output_dir / "masks" / split_name / f"{stem}.png")
                tile_acc.append(stem)

    # 5. manifest 저장
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(result.manifest(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info(
        "Manifest saved: %s (skipped=%d)", manifest_path, len(result.skipped_tiles)
    )
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="cvat_to_dataset flat 출력 → region-based train/val/test split"
    )
    parser.add_argument("--images-root", type=Path, required=True, help="flat images 디렉토리")
    parser.add_argument("--masks-root", type=Path, required=True, help="flat masks 디렉토리")
    parser.add_argument(
        "--metas",
        type=Path,
        nargs="+",
        required=True,
        help="tiling.py 가 GeoTIFF 별로 만든 meta.json 들 (다중 입력)",
    )
    parser.add_argument("--output", type=Path, required=True, help="결과 데이터셋 루트")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-sanity",
        action="store_true",
        help="GeoTIFF 1~2 개일 때만 허용 (정식 KPI 평가 불가)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="output 디렉토리가 이미 존재해도 덮어쓰기",
    )
    args = parser.parse_args()

    result = split_by_region(
        images_root=args.images_root,
        masks_root=args.masks_root,
        meta_paths=args.metas,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        allow_sanity=args.allow_sanity,
        force=args.force,
    )

    print(f"train: {len(result.train_tiles)} tiles ({len(result.train_sources)} GeoTIFFs)")
    print(f"val:   {len(result.val_tiles)} tiles ({len(result.val_sources)} GeoTIFFs)")
    print(f"test:  {len(result.test_tiles)} tiles ({len(result.test_sources)} GeoTIFFs)")
    print(f"skipped: {len(result.skipped_tiles)} tiles")


if __name__ == "__main__":
    main()
