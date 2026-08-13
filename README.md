# AIChallenge26 Multimodal Agentic Video Retrieval System

Hệ thống truy xuất video đa phương thức theo hai pha:

- **Offline:** video → keyframe → metadata đa phương thức → visual/text index.
- **Online:** truy vấn → tìm kiếm theo từng modality → hợp nhất, rerank và trả về video/frame/timestamp.

`backend/` và `src/` là pipeline lõi của repository. `competition/` là workspace độc lập để thử nghiệm các format dữ liệu, chiến lược retrieval và submission của từng cuộc thi; thư mục này **không phải kiến trúc chính của hệ thống** và không được dùng làm nơi lưu artifact mặc định của pipeline lõi.

## Trạng thái hiện tại

| Thành phần | Trạng thái | Ghi chú |
| --- | --- | --- |
| Keyframe extraction | Đã triển khai | TransNetV2, FFmpeg, shot-aware sampling, pHash; có tùy chọn dense coverage và CLIP dedup |
| Visual indexing | Đã triển khai | SigLIP2 image embedding, FAISS `IndexFlatIP`, frame map và encoder manifest |
| Multimodal ingestion | Đã triển khai | Caption, OCR, object detection và ASR |
| Metadata indexing | Đã triển khai | Temporal neighbors, segment aggregation và BM25 text index |
| Retrieval | Đã triển khai | Visual, caption, OCR, ASR, object, hybrid, temporal và QA evidence |
| API | Một phần | Có router/wrapper cho search và retrieval; chưa có `FastAPI()` app tổng để chạy server |
| Evaluation | Một phần | Có metric/evaluator cho keyframe và retrieval; benchmark/ablation/leaderboard tổng còn là khung |
| Agent layer | Chưa triển khai | Các file trong `backend/app/services/agent/` hiện là placeholder |
| Frontend và database | Chưa triển khai | Mới có cấu trúc thư mục/placeholder |
| Competition adapters | Tách biệt | Runner, validation, experiment và submission nằm riêng trong `competition/` |

## Kiến trúc pipeline hiện tại

```text
                                OFFLINE PIPELINE

data/raw/video/*.mp4
        │
        ▼
TransNetV2 shot detection + FFmpeg frame extraction + dedup
        │
        ├── data/keyframes/<video_id>/*.jpg
        └── data/metadata/keyframes_<video_id>.jsonl
                    │
          ┌─────────┴───────────────────────────────┐
          │                                         │
          ▼                                         ▼
 SigLIP2 image encoder                 Multimodal ingestion
          │                            ├── BLIP caption
          │                            ├── EasyOCR vi/en
          │                            ├── YOLO11n objects
          │                            └── faster-whisper ASR
          │                                         │
          ▼                                         ▼
 normalized embeddings                 captions/ocr/objects/asr JSONL
          │                                         │
          ▼                                         ▼
 FAISS + frame map + manifest          segment aggregation + neighbors
          │                                         │
          │                                         ▼
          │                                  BM25 text index
          └──────────────────────┬──────────────────┘
                                 │
                                 ▼
                                ONLINE

query → mode dispatch → visual/text candidates → merge → weighted rerank
                                      │
                                      ├── temporal ordered-event matching
                                      ├── QA evidence grouping
                                      └── result + same-shot neighbors
```

Hai nhánh visual và metadata có thể được build độc lập sau khi có keyframe. Hybrid retrieval chỉ thực sự đa phương thức khi đã có text index; nếu text index chưa tồn tại, mode `hybrid` tự hạ cấp về visual-only.

## Cấu trúc repository

```text
backend/
  app/api/                    # Python wrappers và FastAPI routers
  app/models/                 # Retrieval/metadata data contracts
  app/services/
    indexing/                 # Keyframe, embedding, FAISS, validation
    ingestion/                # Caption, OCR, object, ASR
    metadata/                 # Frame map và metadata lookup
    retrieval/                # Visual/text/hybrid/temporal/QA retrieval
    evaluation/               # Metric và evaluator đã triển khai một phần
    agent/                    # Placeholder, chưa có agent runtime
  tests/                      # Unit/regression tests của pipeline lõi

src/indexing/                 # Neighbor index và segment-level aggregation
configs/retrieval.yaml        # Cấu hình retrieval lõi
data/                         # Dữ liệu và artifact mặc định; phần lớn bị Git ignore
docs/                         # Schema, contract và tài liệu kỹ thuật chi tiết
frontend/                     # Frontend scaffold, chưa có ứng dụng chạy được
experiments/                  # Khu vực thí nghiệm chung
notebooks/                    # Notebook launcher; không chứa thuật toán lõi
reports/                      # Báo cáo benchmark/evaluation
competition/                  # Testbed riêng cho các cuộc thi và submission
```

## Cài đặt

Chạy từ root repository bằng PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Pipeline cần binary `ffmpeg` và `ffprobe` trong `PATH`; package `ffmpeg-python` không tự cài hai binary này.

```powershell
ffmpeg -version
ffprobe -version
```

Các model được tải lazy khi chạy lần đầu và cache dưới `data/model_cache/`. GPU CUDA được khuyến nghị cho SigLIP2, BLIP, OCR, YOLO và ASR; tất cả CLI chính đều có chế độ `--device auto` hoặc CPU fallback phù hợp.

## Chuẩn bị dữ liệu

Đặt video nguồn tại:

```text
data/raw/video/
  L27_V001.mp4
  L27_V002.mp4
  ...
```

Artifact lõi luôn ghi vào các thư mục con của `data/`:

```text
data/keyframes/       # JPEG keyframe
data/metadata/        # JSON/JSONL metadata, frame map, manifest, report
data/embeddings/      # NumPy visual embeddings
data/indexes/         # FAISS và BM25 index
data/model_cache/     # Model weights/cache
```

Video, model và artifact lớn đã được Git ignore; không commit chúng vào repository.

## Chạy offline pipeline

### 1. Trích xuất keyframe

```powershell
.\.venv\Scripts\python.exe -B backend\app\services\indexing\extract_keyframes.py `
  --video-dir data\raw\video `
  --video-glob *.mp4 `
  --output-dir data\keyframes `
  --shot-device auto
```

Mặc định dùng strategy `legacy`:

- shot dưới 4 giây: lấy midpoint;
- shot từ 4 đến 8 giây: lấy frame tại 1/3 và 2/3;
- shot dài hơn 8 giây: lấy một frame mỗi 4 giây;
- loại frame gần trùng bằng pHash trong cùng cửa sổ thời gian.

Output cho mỗi video:

```text
data/keyframes/<video_id>/*.jpg
data/metadata/keyframes_<video_id>.jsonl
data/metadata/keyframes_<video_id>_extract_report.json
```

Extractor còn hỗ trợ `--strategy dense_coverage` để sinh candidate dày và bảo đảm temporal coverage. Đây là strategy nâng cao, không phải mặc định. Dùng `--enable-clip-dedup` nếu muốn bổ sung OpenCLIP near-duplicate filtering.

### 2. Sinh SigLIP2 image embedding

```powershell
Get-ChildItem data\metadata\keyframes_*.jsonl | ForEach-Object {
  .\.venv\Scripts\python.exe -B backend\app\services\indexing\build_siglip2_index.py `
    --metadata-path $_.FullName `
    --batch-size auto `
    --num-workers 4 `
    --device auto
}
```

Encoder mặc định là `google/siglip2-so400m-patch16-384`. Mỗi vector được chuẩn hóa L2 và đi kèm metadata chứa model/revision/dimension để khóa embedding contract.

```text
data/embeddings/siglip2_so400m_patch16_384_<video_id>.npy
data/metadata/siglip2_so400m_patch16_384_embeddings_<video_id>.jsonl
data/metadata/siglip2_so400m_patch16_384_skipped_<video_id>.jsonl
data/metadata/siglip2_so400m_patch16_384_benchmark_<video_id>.json
```

### 3. Build visual FAISS index

```powershell
.\.venv\Scripts\python.exe -B backend\app\services\indexing\build_faiss_index.py
```

Lệnh mặc định gom toàn bộ embedding SigLIP2 trong `data/embeddings/` và sinh:

```text
data/indexes/siglip2_so400m_patch16_384_flat_ip.faiss
data/metadata/siglip2_so400m_patch16_384_faiss_metadata.jsonl
data/metadata/siglip2_so400m_patch16_384_frame_map.json
data/metadata/siglip2_so400m_patch16_384_faiss_manifest.json
data/metadata/siglip2_so400m_patch16_384_index_report.json
```

FAISS mặc định dùng `IndexFlatIP` trên vector đã normalize, vì vậy inner product tương đương cosine similarity. Index, frame map và manifest phải được build cùng một lượt và cùng encoder contract.

### 4. Sinh metadata đa phương thức

Các CLI nhận cả một file `keyframes_<video_id>.jsonl` hoặc thư mục chứa toàn bộ metadata.

```powershell
.\.venv\Scripts\python.exe -B backend\app\services\ingestion\run_caption.py `
  --metadata-path data\metadata --device auto --batch-size 4 --segment-caption

.\.venv\Scripts\python.exe -B backend\app\services\ingestion\run_ocr.py `
  --metadata-path data\metadata --device auto --batch-size 4 --conf-threshold 0.3

.\.venv\Scripts\python.exe -B backend\app\services\ingestion\run_object_detection.py `
  --metadata-path data\metadata --device auto --batch-size 8

.\.venv\Scripts\python.exe -B backend\app\services\ingestion\run_asr.py `
  --video-path data\raw\video `
  --metadata-path data\metadata `
  --video-glob *.mp4 `
  --device auto --backend auto --model-size small
```

Model mặc định:

| Modality | Model/backend | Output |
| --- | --- | --- |
| Caption | `Salesforce/blip-image-captioning-base` | `captions_<video_id>.jsonl` |
| OCR | EasyOCR `vi` + `en` | `ocr_<video_id>.jsonl` |
| Object | YOLO11n (`yolo11n.pt`) | `objects_<video_id>.jsonl` |
| ASR | faster-whisper `small`, fallback openai-whisper | `asr_<video_id>.jsonl`, `asr_segments_<video_id>.jsonl` |

Mỗi pipeline ghi thêm `<artifact>_<video_id>_report.json`. Mặc định các record đã có được bỏ qua để resume an toàn; dùng `--overwrite` khi chủ động muốn build lại modality tương ứng. Metadata keyframe gốc và visual index không bị các bước này sửa trực tiếp.

### 5. Build temporal neighbor index

```powershell
.\.venv\Scripts\python.exe -m src.indexing.build_neighbor_index `
  --input data\metadata `
  --output data\metadata\neighbors_all.jsonl `
  --window-seconds 5
```

Neighbor chỉ được tạo trong cùng `video_id`, sắp xếp theo timestamp và không chứa center frame.

### 6. Aggregate segment-level metadata

```powershell
.\.venv\Scripts\python.exe -m src.indexing.build_segment_metadata `
  --input data\metadata `
  --captions data\metadata `
  --ocr data\metadata `
  --asr data\metadata `
  --objects data\metadata `
  --output data\metadata\segments_all.jsonl `
  --strategy auto
```

`auto` ưu tiên boundary từ `segment_id`/`shot_id`; dữ liệu cũ không có boundary có thể dùng `--strategy fixed --fixed-duration-seconds 10`. Segment record tổng hợp caption, OCR, ASR, object và giữ `source_ids`/`source_intervals` để truy ngược evidence.

### 7. Build BM25 text index

```powershell
.\.venv\Scripts\python.exe -B backend\app\services\indexing\build_text_index.py `
  --metadata data\metadata `
  --output data\indexes\retrieval_text_index.json
```

Folder mode ưu tiên `segments_all.jsonl`, sau đó `segments_*.jsonl`, rồi mới fallback về các artifact modality riêng lẻ.

## Chạy retrieval

Entry point ổn định hiện tại là Python wrapper:

```powershell
.\.venv\Scripts\python.exe -c "from backend.app.api.search import search; import json; print(json.dumps(search('a person cooking', top_k=5, mode='hybrid'), ensure_ascii=False, indent=2))"
```

Các mode được hỗ trợ:

| Mode | Nguồn | Hành vi |
| --- | --- | --- |
| `visual` | SigLIP2 + FAISS | Text-to-keyframe semantic search |
| `caption` | BM25 caption | Tìm theo mô tả ảnh |
| `ocr` | BM25 OCR | Tìm chữ xuất hiện trong frame |
| `asr` | BM25 transcript | Tìm nội dung lời nói |
| `object` | BM25 object labels | Tìm vật thể |
| `hybrid` | Visual + các text modality có sẵn | Merge candidate, weighted rerank và same-shot dedup |
| `temporal` | Nhiều hybrid subquery | Ghép event theo thứ tự trong cùng video |
| `qa_evidence` | Hybrid retrieval | Gom evidence phục vụ câu hỏi |

Mỗi result chuẩn gồm `video_id`, `frame_id`, `timestamp`, `frame_index`, `shot_id`, `segment_id`, `score`, đường dẫn keyframe, metadata đa phương thức, `modality_scores` và các frame lân cận cùng shot khi có.

Trọng số hybrid, pool size, giới hạn `top_k`, same-shot dedup, text-index path và temporal gap nằm trong `configs/retrieval.yaml`. Các đường dẫn/runtime visual có thể override qua biến môi trường; danh sách đầy đủ nằm trong [Retrieval API contract](docs/retrieval_api_contract.md).

## API hiện tại

Repository có hai router:

- `backend.app.api.search.router`: `POST /search`, dispatch bằng trường `mode`.
- `backend.app.api.retrieval.router`: các endpoint `/retrieval/visual`, `/hybrid`, `/caption`, `/ocr`, `/asr`, `/object`, `/temporal` và `/qa-evidence`.

Hiện chưa có file application tạo `FastAPI()` và `include_router(...)`, nên chưa có lệnh `uvicorn ...` chính thức. Không nên mô tả backend như một service deploy hoàn chỉnh cho đến khi app tổng, health check và deployment wiring được bổ sung.

## Vai trò của `competition/`

`competition/` là **competition sandbox** nằm bên cạnh hệ thống lõi:

- adapter input/output theo format TKIS/VKIS hoặc cuộc thi khác;
- runner end-to-end, resume và lineage theo từng run;
- keyframe/retrieval ablation, dense rescue, ensemble;
- validation và sinh `submission.csv`;
- artifact riêng dưới `competition/` hoặc `competition/runs/<run_id>/`.

Code tại đây có thể tái sử dụng service từ `backend/` và `src/`, nhưng pipeline lõi không được import ngược từ `competition/`. Dữ liệu chuẩn của hệ thống vẫn nằm trong `data/`, cấu hình retrieval chuẩn vẫn là `configs/retrieval.yaml`, và API runtime không phụ thuộc competition runner.

Hướng dẫn chi tiết để chạy test cuộc thi nằm trong [competition/README.md](competition/README.md). Không đưa command submission/leaderboard vào luồng cài đặt chính của repository.

## Kiểm tra

Chạy regression tests của pipeline lõi:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s backend\tests -p "test_*.py"
```

Kiểm tra riêng competition sandbox khi thay đổi code trong `competition/`:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s competition\tests -p "test_*.py"
```

Một số smoke test model thật cần model cache, video và phần cứng phù hợp; unit test còn lại dùng artifact synthetic/fake để kiểm tra contract và tính deterministic.

## Quy tắc artifact quan trọng

- Sau khi extract lại keyframe, phải encode lại embedding và build lại FAISS.
- Không trộn embedding từ model/revision/dimension khác nhau trong cùng index.
- Retrieval đọc encoder contract từ manifest và từ chối query vector sai dimension.
- Không sửa trực tiếp keyframe metadata để nhét kết quả caption/OCR/ASR/object; mỗi modality có artifact riêng và được aggregate ở bước segment.
- `frame_map`, FAISS index và manifest là một bộ artifact bất khả phân.
- Thay đổi path/config ở runtime cần xóa cache engine trong process bằng `clear_retrieval_caches()` trước khi search lại.
- Pipeline lõi ghi vào `data/`; competition runner ghi vào workspace của `competition/`. Không dùng chung output root giữa hai luồng.

## Tài liệu liên quan

- [Kiến trúc tổng quan](docs/architecture.md)
- [Metadata schema](docs/metadata_schema.md)
- [Keyframe extraction](docs/keyframe_extraction.md)
- [Retrieval API contract](docs/retrieval_api_contract.md)
- [Evaluation protocol](docs/eval_protocol.md)
- [Service boundaries](docs/service_boundaries.md)
- [Competition sandbox](competition/README.md)

Khi tài liệu thiết kế cũ khác với implementation, README này và code trong `backend/`, `src/` là nguồn tham chiếu cho pipeline đang chạy; các module placeholder không được xem là tính năng đã hoàn thành.
