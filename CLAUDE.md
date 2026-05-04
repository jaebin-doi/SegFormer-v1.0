# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

4-class farmland semantic segmentation with SegFormer (MiT-B2 backbone, mmsegmentation). Korean agricultural orthophotos → mask GeoTIFFs auto-labeled from Farm Map SHP → tiled PNG dataset → trained model.

Classes: `0=rice_paddy(논)`, `1=dry_field(밭)`, `2=orchard(과수)`, `3=greenhouse(시설)`, `255=ignore`. Mapping is fixed by Farm Map `INTPR_CD` 01–04.

Documentation, code comments, log messages, and many directory names are Korean. Preserve Korean strings verbatim when editing.

## License constraints (do not violate)

- Allowed: Apache-2.0, BSD, MIT only. AGPL/GPL/NC are forbidden — see `requirements.txt` header.
- **NVlabs SegFormer code and NVlabs pretrained weights are forbidden** (NVIDIA non-commercial). The MiT-B2 backbone must be loaded from the timm release URL hardcoded in `src/training/configs/segformer_farmland.py` (`init_cfg.checkpoint`). Do not swap it for an NVlabs URL.

## Pipeline (end-to-end, 3 stages)

```
[A] Auto-label    src/labeling/cadastral_rasterize.py     (Farm Map SHP → mask GeoTIFF, original size, EPSG:5186, uint8)
[B] Build dataset src/preprocessing/{tiling,split_dataset}.py  (image+mask co-tiling, region split)
[C] Train         mim train mmsegmentation src/training/configs/segformer_farmland.py
```

The three stages are wired by shell scripts under `scripts/`. Always run via the scripts when possible — they set `PYTHONIOENCODING=utf-8` (mandatory for Korean paths on Windows) and inject the right `--cfg-options` into `mim train`.

### Common commands

```bash
# Auto-label all 4 options; OPTION=D produces the AI-training mask GeoTIFFs
bash scripts/labeling/run_four_options.sh
OPTION=D bash scripts/labeling/run_four_options.sh

# Build training dataset (tiling + region split). Requires option D output.
bash scripts/training/build_dataset.sh

# Train (RTX 5080, ~80k iter, bf16 AMP)
bash scripts/training/train_segformer_farmland.sh

# Evaluate best checkpoint against KPI gates
bash scripts/training/eval_segformer_farmland.sh
```

All scripts default `PYTHON=C:/Users/admin/Miniconda3/python.exe` and accept overrides via env vars (`PYTHON`, `DATASET_DIR`, `WORK_DIR`, `CONFIG`, `AMP`, `GEOTIFF_DIR`, `MASK_DIR`, `TILE_SIZE`, `OVERLAP`, `SKIP_RATIO`, `TRAIN_RATIO`, `VAL_RATIO`, `SEED`, `CHECKPOINT`).

## Architecture notes that span multiple files

**Option D mask is the contract between labeling and training.** `cadastral_rasterize.rasterize_to_mask` writes single-band uint8 GeoTIFFs with `transform/CRS/size identical to the source orthophoto`. Everything downstream (`tiling.py --mask-input`, the training config, the KPI evaluator) assumes that 1:1 alignment. If you change rasterization, also update the tiling assumption that mask + image share `transform/CRS/size`.

**Tiling co-processes image and mask.** `src/preprocessing/tiling.py` with `--mask-input` writes paired tiles into `output/images/{stem}/` and `output/masks/{stem}/`, plus a per-source `meta.json`. `--skip-empty-mask-ratio 0.8` drops tiles whose ignore (255) ratio exceeds the threshold — the training config relies on this to keep `cat_max_ratio=0.75` from collapsing.

**Split is region-based, never tile-based.** `src/preprocessing/split_dataset.py` enforces region (source GeoTIFF) granularity to prevent spatial leakage. Splits for small `n` are hardcoded (n=3 → 1/1/1, n=6 → 4/1/1, etc.); n≤2 requires `--allow-sanity`. `build_dataset.sh` collects `meta.json` from every tiled source and passes them via `--metas`.

**Class imbalance is handled in three places, all keyed to the same statistic** (论 84% / 밭 16% / 과수 2% / 시설 7.5%):
1. `CLASS_WEIGHTS = [1.0, 5.21, 23.0, 11.2]` injected into FocalLoss in `segformer_farmland.py`
2. `RepeatDataset times=4` around the train dataset (more crops per epoch → more chances to sample minority)
3. `RandomCrop(cat_max_ratio=0.75)` to reject crops dominated by a single class

If you tune one, keep the three consistent. The KPI gates (`docs/training/v1.0_학습_설계_v2.md` §4) are also calibrated to this distribution.

**Train script injects `data_root` dynamically.** `scripts/training/train_segformer_farmland.sh` overrides `data_root` and the per-dataloader `data_root` via `--cfg-options` rather than editing the config. The config's `DATA_ROOT = "data/dataset/farmland"` is just a default — don't rely on it being current.

## Things to know before changing code

- Several script-level integrations reference files that are not in the repo and must be created or stubbed before the corresponding script will run end-to-end:
  - `scripts/labeling/install_deps.sh` (called from `run_four_options.sh`)
  - `src/labeling/cadastral_bootstrap.py` providing `get_valid_region` (imported by `cadastral_rasterize.py`)
  - `src/evaluation/eval_segformer_farmland.py` (called by `eval_segformer_farmland.sh`)
  - Source data dirs `data/labels/farmmap/` and `img/1. 농촌 학습(전, 답, 과 구분)/경주/`
- The `public/` directory is a static HTML dashboard (Tailwind via CDN, Lucide icons, plain JS). No build step. Pages `m1_model_training.html` … `m5_model_performance.html` are independent; shared styling is in `public/css/mis-style.css` and shared JS in `public/js/main.js`.
- `requirements.txt` declares the FastAPI backend, MLflow, DVC, ONNX runtime, etc., but no backend service code exists yet in this version — only the labeling and training Python modules.
- Defaults assume RTX 5080 + bf16. If on different hardware, override `AMP=0` and consider lowering `batch_size` / `num_workers` in the config (see the KPI playbook in `docs/training/v1.0_학습_설계_v2.md` §4.3).

## Authoritative docs

- `docs/training/v1.0_학습_설계_v2.md` — training policy, KPI gates, fallback playbook (read this before changing the config or the dataset build).
- `docs/training/v1.0_연속지적도_자동라벨링_전환_보고서.md` — labeling policy decision log; the "Option D" choice is the current canon.
- `docs/labeling/cadastral_labeling_guide.md` — Farm Map SHP schema, CRS, encoding details.
