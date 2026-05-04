"""농지 분류 SegFormer 학습 스크립트.

- 프레임워크: open-mmlab/mmsegmentation (Apache-2.0)
- 백본: MiT-B2 (timm, Apache-2.0)
- MLflow 자동 로깅 (params, metrics, artifacts)
- model_registry.json에 상태 반영 (training → evaluating)
- MODEL_CONFIGS.md v2 준수: class_weight를 학습 마스크에서 자동 계산 (inverse frequency)

사용:
    python src/training/train_segformer_farmland.py \\
        --dataset farmland-v1.0 \\
        --config src/training/configs/segformer_farmland.py
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
from filelock import FileLock
from PIL import Image

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_REGISTRY_PATH = REPO_ROOT / "model_registry.json"
MODELS_DIR = REPO_ROOT / "models"

# 하한 — 빈 클래스에도 최소 가중치를 주어 학습 붕괴 방지
MIN_CLASS_WEIGHT = 0.1


def compute_class_weights(masks_dir: Path, num_classes: int) -> list[float]:
    """학습셋 마스크를 전수 스캔해서 inverse frequency class weight를 계산.

    MODEL_CONFIGS.md v2 공식:
        weight = 1.0 / (class_pixel_counts / class_pixel_counts.sum())
        weight /= weight.sum()

    빈 클래스(0 픽셀)는 MIN_CLASS_WEIGHT로 대체 후 재정규화.

    Args:
        masks_dir: 마스크 PNG 디렉토리 (0~num_classes-1 정수값).
        num_classes: 총 클래스 수.

    Returns:
        길이 num_classes의 정규화된 weight 리스트.
    """
    if not masks_dir.exists():
        raise FileNotFoundError(f"masks directory not found: {masks_dir}")

    counts = np.zeros(num_classes, dtype=np.int64)
    mask_files = sorted(masks_dir.glob("*.png"))
    if not mask_files:
        raise FileNotFoundError(f"no mask PNGs under {masks_dir}")

    for mask_path in mask_files:
        arr = np.array(Image.open(mask_path), dtype=np.int64)
        # 0 ~ num_classes-1 범위만 카운트 (255 등 ignore 값 제외)
        valid = arr[(arr >= 0) & (arr < num_classes)]
        if valid.size == 0:
            continue
        bc = np.bincount(valid, minlength=num_classes)
        counts += bc[:num_classes]

    total = counts.sum()
    if total == 0:
        log.warning("All masks empty — returning uniform weights")
        return [1.0 / num_classes] * num_classes

    # inverse frequency
    freq = counts / total
    # 빈 클래스 처리 — 매우 큰 수 대신 MIN_CLASS_WEIGHT 경로로
    weights = np.zeros(num_classes, dtype=np.float64)
    nonzero = freq > 0
    weights[nonzero] = 1.0 / freq[nonzero]
    weights[~nonzero] = MIN_CLASS_WEIGHT

    # 정규화 (sum=1 기준)
    weights = weights / weights.sum()
    # MIN_CLASS_WEIGHT가 정규화 과정에서 0에 가까워지면 하한 재적용
    weights = np.maximum(weights, MIN_CLASS_WEIGHT / num_classes)
    weights = weights / weights.sum()

    result = weights.tolist()
    log.info(
        "Class weights (inverse frequency): %s",
        [f"{w:.4f}" for w in result],
    )
    log.info("Class pixel counts: %s", counts.tolist())
    return result


def update_registry(status: str, version: str | None = None, run_id: str | None = None) -> None:
    """model_registry.json의 farmland 상태 업데이트 (filelock)."""
    lock = FileLock(str(MODEL_REGISTRY_PATH) + ".lock", timeout=10)
    with lock:
        with MODEL_REGISTRY_PATH.open(encoding="utf-8") as f:
            data = json.load(f)
        data["segformer_farmland"]["status"] = status
        if version:
            data["segformer_farmland"].setdefault("versions", {})[version] = {
                "trained_at": datetime.utcnow().isoformat(),
                "mlflow_run_id": run_id,
                "status": status,
            }
        with MODEL_REGISTRY_PATH.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def train(config_path: Path, work_dir: Path, dataset_version: str) -> str:
    """MMSegmentation Runner로 학습 실행.

    class_weight를 학습 마스크에서 자동 계산(inverse frequency)하여
    cfg.model.decode_head.loss_decode[0]["class_weight"]에 주입한다.

    Returns:
        MLflow run_id
    """
    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile(str(config_path))
    cfg.work_dir = str(work_dir)

    # 데이터셋 경로 override
    dataset_root = REPO_ROOT / "data" / "dataset" / dataset_version
    cfg.train_dataloader.dataset.data_root = str(dataset_root)
    cfg.val_dataloader.dataset.data_root = str(dataset_root)
    cfg.test_dataloader.dataset.data_root = str(dataset_root)

    # class_weight 자동 계산 (inverse frequency)
    num_classes = cfg.model.decode_head.num_classes
    train_masks_dir = dataset_root / "masks" / "train"
    class_weights = compute_class_weights(train_masks_dir, num_classes)
    # MMSeg의 loss_decode[0] (CrossEntropyLoss)에 주입
    cfg.model.decode_head.loss_decode[0]["class_weight"] = class_weights

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("farmland_segformer")

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "config": str(config_path),
                "dataset_version": dataset_version,
                "backbone": "MiT-B2",
                "framework": "mmsegmentation",
                "license": "Apache-2.0",
                "lr": cfg.optim_wrapper.optimizer.lr,
                "batch_size": cfg.train_dataloader.batch_size,
                "max_iters": cfg.train_cfg.max_iters,
                "crop_size": str(cfg.crop_size if hasattr(cfg, "crop_size") else "(512, 512)"),
                "loss_ratio_ce_dice": "0.7:0.3",
            }
        )
        # 클래스별 가중치를 MLflow 메트릭으로 기록 (추후 MLflow UI에서 비교)
        for i, w in enumerate(class_weights):
            mlflow.log_metric(f"class_weight_{i}", float(w))

        runner = Runner.from_cfg(cfg)
        runner.train()

        # 최종 체크포인트 아티팩트 업로드
        final_ckpt = Path(work_dir) / "iter_final.pth"
        if not final_ckpt.exists():
            # 가장 최근 checkpoint 찾기
            ckpts = sorted(Path(work_dir).glob("iter_*.pth"))
            if ckpts:
                final_ckpt = ckpts[-1]
        if final_ckpt.exists():
            mlflow.log_artifact(str(final_ckpt), artifact_path="checkpoints")

        return run.info.run_id


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "src/training/configs/segformer_farmland.py",
    )
    parser.add_argument("--dataset", required=True, help="dataset version (예: farmland-v1.0)")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=REPO_ROOT / "work_dirs" / "farmland",
    )
    parser.add_argument("--version", default=None, help="모델 버전 (예: v1.1)")
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    version = args.version or f"v{datetime.now():%Y%m%d_%H%M}"
    update_registry("training")

    try:
        run_id = train(args.config, args.work_dir, args.dataset)
    except Exception:
        log.exception("Training failed")
        update_registry("not_started")
        raise

    # 최종 체크포인트를 models/로 복사
    ckpts = sorted(args.work_dir.glob("iter_*.pth"))
    if ckpts:
        dest = MODELS_DIR / f"farmland_{version}.pth"
        shutil.copy2(ckpts[-1], dest)
        log.info("Saved checkpoint to %s", dest)

    update_registry("evaluating", version=version, run_id=run_id)
    log.info("Done. MLflow run_id=%s version=%s", run_id, version)


if __name__ == "__main__":
    main()
