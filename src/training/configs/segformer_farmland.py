"""MMSegmentation SegFormer 농지 분류 config (v2 — 4-class 팜맵 자동 라벨).

백본: MiT-B2 (timm pretrained, Apache-2.0)
프레임워크: open-mmlab/mmsegmentation (Apache-2.0)
클래스: 4개 (논/밭/과수/시설) — 팜맵 INTPR_CD 01/02/03/04 직접 매핑
입력: 2048×2048 RGB 타일 → 학습 시 512×512로 RandomCrop

학습 정책 (v2):
- 학습 crop 512×512, 원본 scale 2048×2048
- loss: Focal Loss (γ=2) + class_weight (반비례) — 클래스 불균형 (논 84% / 과수 2%) 대응
- augmentation: RandomResize(0.5~2.0x) + RandomCrop + H/V flip + 약한 PhotoMetricDistortion
- max_iter 80k (데이터 양 보수적 시작), val_interval 4k
- ignore_index=255 (mmsegmentation 기본) — 외곽 + 농지 외 픽셀 자동 제외

⛔ NVlabs/SegFormer 코드 또는 NVlabs 배포 가중치 사용 금지 (NVIDIA NC)
"""
from __future__ import annotations

# 4개 클래스 — 팜맵 INTPR_CD 01/02/03/04 매핑
NUM_CLASSES = 4
CLASS_NAMES = [
    "rice_paddy",   # 0  (INTPR_CD 01, 논)
    "dry_field",    # 1  (INTPR_CD 02, 밭)
    "orchard",      # 2  (INTPR_CD 03, 과수)
    "greenhouse",   # 3  (INTPR_CD 04, 시설)
]

# Class weight (반비례 정규화) — 옵션 D mask 픽셀 통계 기반:
#   논(0):    1.77억 → 1.00x
#   밭(1):    0.34억 → 5.21x
#   과수(2):  0.04억 → 23.0x  (★ 표본 부족, 큰 가중치)
#   시설(3):  0.16억 → 11.2x
# (Focal Loss 와 함께 사용 — 둘 다 minority class 가중치 효과)
CLASS_WEIGHTS = [1.0, 5.21, 23.0, 11.2]

# 입력 해상도 (원본 타일 — 팜맵 자동 라벨 흐름 = 2048×2048)
IMG_SCALE = (2048, 2048)
# 학습 crop 해상도 (RandomCrop 후)
CROP_SIZE = (512, 512)

# 데이터셋 경로 (런타임에 덮어씀)
DATA_ROOT = "data/dataset/farmland"

# 정규화 (ImageNet)
IMG_NORM_CFG = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

# ── 모델 ──
model = dict(
    type="EncoderDecoder",
    data_preprocessor=dict(
        type="SegDataPreProcessor",
        mean=IMG_NORM_CFG["mean"],
        std=IMG_NORM_CFG["std"],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=CROP_SIZE,
    ),
    backbone=dict(
        type="MixVisionTransformer",
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 4, 6, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        # ⚠️ init_cfg는 timm MiT-B2 가중치 URL (Apache-2.0)로 지정
        # NVlabs 가중치는 사용 금지
        init_cfg=dict(
            type="Pretrained",
            checkpoint="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-segformer/mit_b2.pth",
        ),
    ),
    decode_head=dict(
        type="SegformerHead",
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=NUM_CLASSES,
        norm_cfg=dict(type="SyncBN", requires_grad=True),
        align_corners=False,
        # v2 정책: Focal Loss (γ=2) + class_weight 반비례 — 클래스 불균형 대응.
        # 단순 CE + Dice 는 minority class (과수 2% / 시설 7%) 학습 실패 위험.
        loss_decode=[
            dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=0.7,
                class_weight=CLASS_WEIGHTS,
            ),
            dict(type="DiceLoss", loss_weight=0.3),
        ],
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)

# ── 데이터 파이프라인 (MODEL_CONFIGS.md v2) ──
# src/preprocessing/transforms.py의 get_segformer_train_pipeline과 동일한 구조를
# MMSeg config 문법으로 인라인. 외부 import 의존성 없이 mmengine이 읽을 수 있도록 함.
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", reduce_zero_label=False),
    dict(type="RandomResize", scale=IMG_SCALE, ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type="RandomCrop", crop_size=CROP_SIZE, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(type="RandomFlip", prob=0.5, direction="vertical"),
    # 약한 색상 왜곡 — 위성/항공 텍스처 깨짐 방지 위해 brightness/contrast 만 조정.
    # 농지 도메인은 계절별 색조 변화가 의미를 가지므로 강한 hue 왜곡 금지.
    dict(
        type="PhotoMetricDistortion",
        brightness_delta=16,
        contrast_range=(0.9, 1.1),
        saturation_range=(0.9, 1.1),
        hue_delta=5,
    ),
    dict(type="PackSegInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=IMG_SCALE, keep_ratio=True),
    dict(type="LoadAnnotations", reduce_zero_label=False),
    dict(type="PackSegInputs"),
]

# ── Dataset ──
dataset_type = "CustomDataset"

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    # RepeatDataset — 데이터 양 적은 v1.0 에서 epoch 당 더 많은 iteration 보장.
    # 또한 RandomCrop + RandomResize 가 매번 다른 sample 을 만들어 과수/시설 minority
    # class 가 더 자주 학습됨.
    dataset=dict(
        type="RepeatDataset",
        times=4,
        dataset=dict(
            type=dataset_type,
            data_root=DATA_ROOT,
            data_prefix=dict(img_path="images/train", seg_map_path="masks/train"),
            img_suffix=".png",
            seg_map_suffix=".png",
            pipeline=train_pipeline,
        ),
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=DATA_ROOT,
        data_prefix=dict(img_path="images/val", seg_map_path="masks/val"),
        img_suffix=".png",
        seg_map_suffix=".png",
        pipeline=test_pipeline,
    ),
)
# test_dataloader / test_evaluator는 val과 별도 정의.
# val_dataloader 를 그대로 alias 하면 region-based split 에서 test split 이 사실상
# val split 으로 평가되어 KPI 신뢰도가 깨진다. data_prefix 만 images/test, masks/test
# 로 분리하고 evaluator 는 동일 metric 으로 새 객체 생성.
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=DATA_ROOT,
        data_prefix=dict(img_path="images/test", seg_map_path="masks/test"),
        img_suffix=".png",
        seg_map_suffix=".png",
        pipeline=test_pipeline,
    ),
)

val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU", "mFscore"])
test_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU", "mFscore"])

# ── Optimizer ──
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            "pos_block": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
            "head": dict(lr_mult=10.0),
        }
    ),
)

# ── Scheduler ──
# v2: max_iter 160k → 80k 단축 (데이터 양 보수적 시작, 성능 saturate 시 조기 종료)
MAX_ITERS = 80000

param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type="PolyLR",
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=MAX_ITERS,
        by_epoch=False,
    ),
]

# ── Training ──
train_cfg = dict(type="IterBasedTrainLoop", max_iters=MAX_ITERS, val_interval=4000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# Early stopping은 custom hook으로 처리 (patience 20 epochs)
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=4000,
        max_keep_ckpts=3,
        save_best="mIoU",
        rule="greater",
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="SegLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)
log_processor = dict(by_epoch=False)
log_level = "INFO"
load_from = None
resume = False
