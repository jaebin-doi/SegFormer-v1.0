# 농지 자동 라벨링 가이드 — 팜맵(Farm Map) SHP → 4 옵션 출력

**버전**: v8 (CVAT/VWorld 폐기, 팜맵 SHP 단독 메인 흐름, 2026-04-30)
**대상 환경**: 노트북 (Windows + Miniconda3 Python 3.13)
**범위**: 자동 라벨링만 — 사람 검수 ❌, CVAT ❌

---

## 1. 한눈에 보기

```
┌─────────────────────────────────────────┐
│ 입력 데이터                              │
│  · 정사영상 GeoTIFF — img/.../경주/*.tif │
│  · 팜맵 SHP — data/labels/farmmap/      │
│              FARM_1.shp ~ FARM_9.shp     │
└──────────────────┬──────────────────────┘
                   │
                   │ scripts/labeling/run_four_options.sh
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 1. install_deps.sh — rasterio/shapely/pyproj/geopandas  │
│ 2. cadastral_preview.py / cadastral_rasterize.py        │
│      pyogrio bbox filter (cp949)                        │
│      INTPR_CD 매핑 (01=논 / 02=밭 / 03=과수 / 04=시설)  │
│      EPSG:5179 → EPSG:5186 reproject                    │
│      GeoTIFF alpha 영역 자동 제외 (valid_region clip)   │
│ 3. 4 옵션 출력 (아래 §5)                                 │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 사용 데이터 / 방식 (★ 정확 명세)

### 2.1 사용 데이터셋

| 데이터 | 출처 | 형식 | 좌표계 | 인코딩 |
|---|---|---|---|---|
| **농림축산식품부 팜맵 (Farm Map)** | 공공데이터포털(`data.go.kr`) — "팜맵" 검색 또는 농림축산식품 데이터포털 | ESRI Shapefile (.shp/.dbf/.shx/.prj 4종 세트, ZIP) | **EPSG:5179** (Korea 2000 통합원점) | DBF **cp949** |

**전국 단위 분할**: FARM_1 ~ FARM_9 (9 SHP, 합계 약 6.6 GB).
경주 영역 = `FARM_2` + `FARM_3` 자동 매칭.

**핵심 컬럼**:
| 컬럼 | 의미 | 우리 사용 |
|---|---|---|
| `INTPR_CD` | 판독 분류 코드 (2자리) — 01/02/03/04 | ✅ class 매핑 |
| `INTPR_NM` | 판독 분류 한글명 — 논/밭/과수/시설 | ✅ 텍스트 라벨 (옵션 C) |
| `PNU_LNM_CD` | 필지 고유번호 (PNU, 19자리) | ✅ 라벨 |
| `LGL_EMD_NM` | 법정읍면동 한글명 | ✅ 라벨 |
| `LNM` | 지번 | ✅ 라벨 |
| `AREA` | 면적 (㎡) | 통계 |
| `FMAP_INNB` / `VDPT_YR` / `CHG_CFCD/NM` / `INVD_CFCD/NM` / `ITPINP_DE` / `RNHST_CD/NM` | 메타 | (보존만, 학습 영향 없음) |

### 2.2 INTPR_CD 클래스 매핑

| INTPR_CD | INTPR_NM | 우리 class_code | 우리 class_name |
|---|---|---|---|
| 01 | 논 | 0 | rice_paddy |
| 02 | 밭 | 1 | dry_field |
| 03 | 과수 | 2 | orchard |
| 04 | 시설 | 3 | greenhouse |
| 그 외 / 미정의 | — | — | 자동 skip (whitelist 정책) |

### 2.3 처리 방식 (코드 흐름)

```
1) discover_farmmap_shps(data/labels/farmmap/)
     → 9 SHP 자동 발견 정렬 리스트

2) for each GeoTIFF:
     a) rasterio.open(tif).bounds (EPSG:5186)
     b) bbox EPSG:5186 → EPSG:5179 reproject (pyproj.Transformer)
     c) for each FARM_*.shp:
          pyogrio.read_dataframe(shp, bbox=bbox_5179, encoding="cp949")
          ↑ SHP 인덱스로 spatial filter (매우 빠름, FARM_2/3 외 SHP 는 0건)
     d) for each row:
          - INTPR_CD ∈ {01,02,03,04} whitelist 필터
          - polygon EPSG:5179 → EPSG:5186 reproject (shapely.ops.transform)
          - GeoTIFF alpha 채널 valid_region 과 intersection clip
            (rasterio.features.shapes 로 alpha mask → MultiPolygon)
          - 빈 polygon 자동 제외
     e) 옵션별 출력 (다음 §5)
```

**API 호출 ❌** — 모든 처리 오프라인 (다운로드 SHP 직접 읽기). VWorld API 호출 없음.

---

## 3. 빠른 실행

```bash
# 4 옵션 모두
bash scripts/labeling/run_four_options.sh

# 옵션 단독
OPTION=A bash scripts/labeling/run_four_options.sh   # 팜맵 논밭만 PNG
OPTION=B bash scripts/labeling/run_four_options.sh   # 팜맵 4-class PNG
OPTION=C bash scripts/labeling/run_four_options.sh   # 4-class + 텍스트 PNG
OPTION=D bash scripts/labeling/run_four_options.sh   # AI 학습용 mask GeoTIFF
```

---

## 4. 사전 준비

### 4.1 팜맵 SHP 다운로드

1. 공공데이터포털(`data.go.kr`) 접속 → 검색 "팜맵"
2. 회원가입 + 활용신청
3. 시군구별 또는 전국 9 분할 ZIP 다운로드
4. 압축 해제 위치: `data/labels/farmmap/`
   - 결과: `FARM_1.shp` `FARM_1.dbf` `FARM_1.shx` `FARM_1.prj` ... `FARM_9.shp` ...

### 4.2 의존성 설치

```bash
bash scripts/labeling/install_deps.sh
# rasterio / shapely / pyproj / geopandas / fiona / pyogrio / requests
```

---

## 5. 4 옵션 — 작업 시 선택

| 옵션 | 폴더 | 출력 | 클래스 | 텍스트 | 용도 |
|---|---|---|---|---|---|
| **A** | `data/labels/팜맵_논밭만/` | PNG 6장 | 논/밭 (2-class) | ❌ | 검수 (단순) |
| **B** | `data/labels/팜맵_논밭과수시설/` | PNG 6장 | 논/밭/과수/시설 (4-class) | ❌ | 검수 (전체) |
| **C** | `data/labels/팜맵_논밭과수시설_지적정보/` | PNG 6장 | 4-class | ✅ "주소 지번 \| 농지분류" | 정밀 검수 |
| **D** | `data/labels/팜맵_논밭과수시설_AI학습용/` | **mask GeoTIFF 6장** | 4-class | ❌ | **AI 학습** |

### 5.1 색상 팔레트 (대표님 지정)

- 🟢 **논** rice_paddy = `#22C55E` (green-500)
- 🟧 **밭** dry_field = `#F97316` (orange-500)
- 🟥 **과수** orchard = `#EF4444` (red-500)
- 🟦 **시설** greenhouse = `#3B82F6` (blue-500)

### 5.2 옵션 D — AI 학습용 mask GeoTIFF 사양

| 속성 | 값 |
|---|---|
| 픽셀 값 | 0=논 / 1=밭 / 2=과수 / 3=시설 / 255=ignore |
| dtype | uint8, 단일 밴드 |
| size | **원본 정사영상과 동일** (다운샘플 ❌) |
| CRS | EPSG:5186 (원본과 동일) |
| transform | 원본과 동일 (1:1 픽셀 매핑) |
| 압축 | LZW |
| nodata | 255 |
| 출력 6장 합계 | 6.6 MB (원본 RGB 합계 약 5GB 대비 압축률 매우 높음) |

mmsegmentation `LoadAnnotations` 가 그대로 읽음 (ignore_index=255 기본).

---

## 6. 코드 카탈로그

| 파일 | 역할 |
|---|---|
| `src/labeling/farmmap_loader.py` | 팜맵 SHP 다중 입력 → polygon 메모리. cp949/EPSG:5179/INTPR_CD 처리 |
| `src/labeling/cadastral_preview.py` | PNG (옵션 A/B/C) — 다운샘플 + polygon overlay + (옵션) 텍스트 라벨 |
| `src/labeling/cadastral_rasterize.py` | mask GeoTIFF (옵션 D) — 원본 1:1, rasterio.features.rasterize |
| `src/labeling/cadastral_bootstrap.py` | (레거시 — VWorld API 흐름, 메인 라벨링에는 사용 안 함. `get_valid_region()` 만 alpha clip 용으로 재사용) |
| `tests/labeling/test_farmmap_loader.py` | 6 PASSED |
| `tests/labeling/test_cadastral_bootstrap.py` | 26 PASSED |
| `tests/labeling/test_cadastral_rasterize.py` | 4 PASSED |
| `scripts/labeling/install_deps.sh` | pip install 지리 deps |
| `scripts/labeling/run_four_options.sh` | A/B/C/D 통합 실행 (OPTION 환경변수) |

---

## 7. 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `pyogrio.errors.DataSourceError` | 팜맵 SHP 경로 오류 | `data/labels/farmmap/` 안에 FARM_*.shp 4종 파일 세트 있는지 확인 |
| 한글 깨짐 | DBF 인코딩 잘못 | 코드에 `encoding='cp949'` 명시됨. 다른 SHP 라면 인코딩 확인 |
| polygon 안 그려짐 | bbox 안 겹침 (다른 시군구 SHP) | 9 SHP 모두 자동 검색해도 겹침 영역 있어야 결과 발생. 경주 = FARM_2/3 |
| 마스크 GeoTIFF 매우 큼 | LZW 압축 안 됨 | 코드에 `compress='lzw'` 명시. 비활성된 경우만 발생 |

---

## 8. 변경 이력

| 날짜 | 버전 | 변경 |
|---|---|---|
| 2026-04-30 | v1 | VWorld 연속지적도 → GeoPackage |
| 2026-04-30 | v2 | alpha clip + PNG 색상 부드러운 톤 |
| 2026-04-30 | v3 | render_overlay 패턴 채택 (fill α=46, outline α=255) |
| 2026-04-30 | v4 | GeoPackage 단계 제거 + GeoTIFF 출력 추가 |
| 2026-04-30 | v5 | 팜맵 통합 + VWorld/팜맵 비교 모드 |
| 2026-04-30 | v6 | 팜맵 4-class + 지적정보 텍스트 라벨 (3 모드) |
| 2026-04-30 | v7 | 4 옵션 분기 + AI 학습용 mask GeoTIFF (옵션 D) |
| 2026-04-30 | **v8** | **CVAT 폐기 + VWorld API 메인 흐름 폐기** — 팜맵 SHP 단독. 사용 데이터/방식 명세 추가. 사람 검수 ❌. |
