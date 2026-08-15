# Multimodal Agentic Video Retrieval System

Repository này xây dựng hệ thống truy hồi video đa phương thức tổng quát, tách thành hai
luồng:

- **Offline pipeline**: video → keyframe → metadata đa phương thức → embedding/index.
- **Online pipeline**: query → query understanding → retrieval đa nguồn → fusion/rerank →
  candidate có timestamp và evidence.

Đây là tài liệu cho pipeline tổng thể của repository. Quy trình đóng gói dữ liệu và tạo
submission cho HCMC AI Challenge nằm riêng tại
[competition/README.md](competition/README.md).

## Kiến trúc tổng thể

```text
<<<<<<< HEAD
video -> dense keyframes -> SigLIP2 embeddings -> FAISS
                         -> Qwen3.5-9B captions
                         -> PP-OCRv5 vi/en
                         -> YOLOE open-vocabulary evidence
                         -> segment metadata -> BM25 text index
query -> visual/text/temporal retrieval -> hybrid rerank -> evidence
      -> BGE-M3 dense text -> BGE cross-encoder -> grounded QA (Qwen3.5)
```

- Visual: SigLIP2, giữ nguyên embedding/index contract hiện có.
- Caption: `Qwen/Qwen3.5-9B` revision `c202236`, output JSON có cấu trúc và
  caption fallback.
- OCR: `PP-OCRv5_server_det` + `latin_PP-OCRv5_mobile_rec`, hỗ trợ tiếng Việt
  và tiếng Anh.
- Object evidence: `yoloe-26l-seg.pt`, vocabulary cấu hình được. Kết quả chỉ là
  bằng chứng mềm; không dùng làm hard filter loại candidate.
- Text retrieval: caption, OCR và objects.
- Dense text: `BAAI/bge-m3` (1024 chiều, FAISS IP); BM25 vẫn được giữ để bắt
  exact keyword/OCR.
- QA rerank: `BAAI/bge-reranker-v2-m3`; grounded answer dùng
  `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a`, mặc định
  tắt và chỉ load khi chạy QA.
=======
Raw videos
  -> shot/keyframe extraction
  -> SigLIP2 visual embeddings
  -> Qwen captions + PP-OCRv5 + YOLOE objects
  -> frame/segment metadata
  -> FAISS visual index + BM25 text index + neighbor index

User query
  -> agent/query planning
  -> visual | caption | OCR | object | hybrid | temporal retrieval
  -> candidate merge + rank fusion + reranking
  -> video_id, frame_id, timestamp, score, evidence
```

Nguyên tắc chính:

- FAISS chỉ nhận embedding cùng encoder contract và dimension.
- Caption, OCR và object metadata là evidence mềm; object không phải hard filter.
- Agent chỉ lập kế hoạch/gọi retrieval, không sửa metadata hoặc index.
- Audio hiện không được xử lý.
- Temporal retrieval là engine độc lập; query expansion không tự sinh temporal event.

## Cấu trúc repository

```text
backend/app/services/
  ingestion/       caption, OCR, object extraction
  indexing/        keyframe, SigLIP2, FAISS, text/neighbor index
  retrieval/       visual, lexical, hybrid, temporal, fusion, reranking
  agent/           query planning, decomposition, safe query expansion
  evaluation/      metrics, benchmark, ablation, report

backend/app/api/   các FastAPI router và Python wrapper
src/indexing/      canonical neighbor/segment metadata builders
configs/           retrieval runtime configuration
data/              raw data và artifact mặc định
docs/              architecture, schema và API contract
competition/       pipeline thi đấu tách biệt
```
>>>>>>> origin/main

## Cài đặt

Yêu cầu Python, FFmpeg trên `PATH`, và driver CUDA phù hợp nếu chạy GPU.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

<<<<<<< HEAD
Lệnh trên cài đầy đủ runtime CPU, bao gồm PaddlePaddle cho PP-OCRv5. Với NVIDIA
GPU, không cài `requirements.txt`; cài dependency chung trước rồi chọn đúng một
PaddlePaddle profile theo driver/runtime:

```powershell
python -m pip install -r requirements-core.txt
python -m pip install -r requirements-gpu-cu118.txt
# hoặc
python -m pip install -r requirements-core.txt
python -m pip install -r requirements-gpu-cu126.txt
```

Không cài đồng thời `paddlepaddle` và `paddlepaddle-gpu`. Máy NVIDIA 50-series
trên Windows có thể cần wheel chuyên biệt theo Python/driver từ hướng dẫn chính
thức của PaddlePaddle thay vì hai profile trên. PyTorch/CUDA vẫn phải tương thích
với driver của máy. Quantization caption 4/8-bit cần CUDA và `bitsandbytes`
tương thích.

Kiểm tra các runtime model sau khi cài:

```powershell
python -c "import torch, transformers, accelerate, paddle, paddleocr, ultralytics; print({'torch': torch.__version__, 'cuda': torch.cuda.is_available(), 'transformers': transformers.__version__, 'accelerate': accelerate.__version__, 'paddle': paddle.__version__, 'paddle_cuda': paddle.is_compiled_with_cuda(), 'paddleocr': paddleocr.__version__, 'ultralytics': ultralytics.__version__})"
```
=======
PaddlePaddle phải được cài riêng theo môi trường. Repo đã được kiểm thử với
PaddlePaddle `3.2.2`; không cài đồng thời wheel CPU và GPU. Ví dụ CPU:

```powershell
python -m pip install paddlepaddle==3.2.2
```

Với CUDA, cài `paddlepaddle-gpu==3.2.2` từ index chính thức tương ứng kiến trúc GPU.
Blackwell `sm_120` (RTX 5090/5080/5070) dùng index `cu129` hoặc mới hơn; không dùng
wheel `cu118`, vì wheel đó không chứa mã cho `sm_120`. Luôn chạy smoke test Torch và
Paddle trước khi tải model hoặc bắt đầu Phase 3.
Kiểm tra FFmpeg:
>>>>>>> origin/main

```powershell
ffmpeg -version
```

Model được lazy-load và cache dưới `data\model_cache`. Lần chạy đầu cần mạng nếu
checkpoint chưa có.

## Data layout mặc định

```text
data/
  raw/             video nguồn
  keyframes/       ảnh keyframe theo video
  metadata/        frame metadata, caption, OCR, objects, segment, manifests
  embeddings/      SigLIP2 .npy
  indexes/         FAISS và lexical text index
  model_cache/     Hugging Face, PaddleOCR, YOLO model cache
```

Các stage không ghi đè video hoặc keyframe nguồn. Artifact metadata có provenance về
model, revision và tham số để kiểm tra resume/rebuild.

## Chạy offline pipeline

Các lệnh dưới đây minh họa pipeline tổng quát cho `VIDEO_001`. Với nhiều video, chạy
stage theo từng file/folder rồi build index chung ở cuối.

### 1. Trích xuất keyframe từ video

```powershell
.\.venv\Scripts\python.exe backend\app\services\indexing\extract_keyframes.py `
  --video-path data\raw\VIDEO_001.mp4 `
  --output-dir data\keyframes `
  --strategy dense_coverage `
  --candidate-interval-sec 0.5 `
  --max-gap-seconds 2.0
```

Output chính:

```text
data/keyframes/VIDEO_001/...
data/metadata/keyframes_VIDEO_001.jsonl
data/metadata/keyframes_VIDEO_001_extract_report.json
```

Nếu đã có sẵn một thư mục keyframe, có thể chuẩn hóa metadata và tạo embedding trong
một lệnh:

```powershell
.\.venv\Scripts\python.exe -m backend.app.services.indexing.run_keyframe_siglip2_pipeline `
  --keyframe-dir data\keyframes\VIDEO_001 `
  --video-id VIDEO_001 `
  --video-path data\raw\VIDEO_001.mp4 `
  --output-root data `
  --device cuda
```

### 2. Tạo SigLIP2 embedding

Bỏ qua bước này nếu đã dùng `run_keyframe_siglip2_pipeline` ở trên.

```powershell
.\.venv\Scripts\python.exe -m backend.app.services.indexing.build_siglip2_index `
  --metadata-path data\metadata\keyframes_VIDEO_001.jsonl `
  --device cuda `
  --batch-size auto
```

Lặp lại cho mỗi file `keyframes_<video>.jsonl`. Embedding và embedding metadata phải giữ
đúng thứ tự/identity; validator và FAISS builder sẽ fail closed nếu contract lệch.

### 3. Sinh metadata đa phương thức

Caption (mặc định `Qwen/Qwen3.5-4B`):

```powershell
.\.venv\Scripts\python.exe backend\app\services\ingestion\run_caption.py `
  --metadata-path data\metadata `
  --output-dir data\metadata `
  --device cuda `
  --batch-size 2 `
  --dtype auto `
  --quantization 4bit `
  --segment-caption
```

OCR tiếng Việt/Anh:

```powershell
.\.venv\Scripts\python.exe backend\app\services\ingestion\run_ocr.py `
  --metadata-path data\metadata `
  --output-dir data\metadata `
  --device cuda `
  --batch-size 4 `
  --conf-threshold 0.3
```

Object evidence:

```powershell
.\.venv\Scripts\python.exe backend\app\services\ingestion\run_object_detection.py `
  --metadata-path data\metadata `
  --output-dir data\metadata `
  --device cuda `
  --batch-size 8 `
  --conf-threshold 0.25
```

Ba CLI nhận một JSONL hoặc cả thư mục, hỗ trợ resume theo provenance và chỉ load model
khi có work cần xử lý. Dùng `--overwrite` khi chủ động tái tạo đúng modality.

### 4. Tạo neighbor và segment metadata

Neighbor index:

```powershell
.\.venv\Scripts\python.exe -m backend.app.services.indexing.neighbor_index `
  --input data\metadata `
  --output data\metadata\neighbors_all.jsonl `
  --window-seconds 5
```

Segment metadata kết hợp caption/OCR/objects:

<<<<<<< HEAD
## Colab end-to-end và contract submission

Notebook low-memory [`notebooks/E2E.ipynb`](notebooks/E2E.ipynb) và notebook
full precision [`notebooks/E2E_FULL_PRECISION.ipynb`](notebooks/E2E_FULL_PRECISION.ipynb)
đều chạy đủ 10 stage, build BGE-M3 index, dùng BGE reranker cho TKIS rồi tạo
`results/submission.csv`. Output public **không đổi**: đúng 100 query trong
`questions.csv` (50 TKIS + 50 VKIS), mỗi query 100 answer theo đúng thứ tự và
header của `sample_submission.csv`.

Profile low-memory dành cho T4/L4 và quantize Qwen caption 4-bit; caption có thể
khác nhẹ. Profile full precision giữ cấu hình cũ nhưng cần GPU tối thiểu khoảng
24 GB, khuyến nghị A100 40 GB. Dùng `RUN_ID` riêng cho từng profile khi benchmark.

Public dataset hiện không có QA. Vì vậy notebook chạy KIS/AVS/QA smoke riêng
sau submission và ghi `results/task_smoke.json`; QA không được thêm vào
`submission.csv`. Trong notebook, hai BGE mode và Qwen QA mode được đặt
`required`: model lỗi thì cell fail thay vì âm thầm fallback sang pipeline cũ.

## Test riêng KIS, AVS và QA

Trước khi chạy, trỏ service vào artifacts của một `run_root` đã hoàn tất. Ví dụ
PowerShell (đổi `$runRoot` theo máy):

```powershell
$runRoot = "E:\runs\new-model-001"
$env:RETRIEVAL_INDEX_PATH = "$runRoot\indexes\siglip2_so400m_patch16_384_flat_ip.faiss"
$env:RETRIEVAL_FRAME_MAP_PATH = "$runRoot\metadata\siglip2_so400m_patch16_384_frame_map.json"
$env:RETRIEVAL_MANIFEST_PATH = "$runRoot\metadata\siglip2_so400m_patch16_384_faiss_manifest.json"
$env:RETRIEVAL_TEXT_INDEX_PATH = "$runRoot\indexes\retrieval_text_index.json"
$env:QA_BGE_DENSE_ENABLED = "true"
$env:QA_BGE_RERANKER_ENABLED = "true"
$env:QA_BGE_INDEX_ROOT = "$runRoot\indexes\bge_m3"
$env:QA_BGE_DEVICE = "cuda"
$env:QA_BGE_MODEL_CACHE_DIR = "data\model_cache\bge_m3"
$env:QA_ANSWER_MODEL_CACHE_DIR = "data\model_cache\qa_answer"
```

KIS và AVS dừng ở evidence, không load Qwen:

```powershell
python -m backend.app.services.retrieval.run_task_smoke --task kis --top-k 5
python -m backend.app.services.retrieval.run_task_smoke --task avs --top-k 5
```

QA end-to-end có grounded answer:

```powershell
$env:QA_ANSWER_MODE = "required"
$env:QA_ANSWER_DEVICE = "cuda"
$env:QA_ANSWER_QUANTIZATION = "4bit"
python -m backend.app.services.retrieval.run_task_smoke `
  --task qa --top-k 5 `
  --qa-query "Người phụ nữ áo đỏ đang cầm gì?" `
  --output "$runRoot\results\qa_smoke.json"
```

Chạy cả ba task:

```powershell
python -m backend.app.services.retrieval.run_task_smoke `
  --task all --top-k 5 `
  --output "$runRoot\results\task_smoke.json"
```

`QA_ANSWER_MODE=off|optional|required` chỉ điều khiển answerer. Parser, router,
evidence bundle, BGE dense và BGE reranker có feature flag riêng. Smoke chỉ xác
nhận checkpoint/artifact/contract chạy được; muốn so chất lượng phải dùng dev
labels, và locked test chỉ được mở một lần theo policy trong
`competition/evaluation/`.

BGE-M3 chỉ index metadata của keyframe semantic/canonical đã được chọn; dense
candidate frames không thuộc source contract này. Với QA non-temporal, answerer
nhận tối đa Top-3 evidence. Với temporal, chỉ chain `strict` đầy đủ (tối đa 5
event) mới được gọi Qwen; `relaxed_gap` và `sparse_compat` chỉ trả chain để audit
và abstain.

## Schema metadata chính

Caption record chứa `caption`, `structured_caption`, `caption_parse_status` và
provenance. `structured_caption` có các khóa:

```json
{
  "scene": "",
  "people": [{"type": "", "attributes": []}],
  "objects": [],
  "actions": [],
  "relationships": [],
  "colors": [],
  "visible_text": [],
  "caption": ""
}
=======
```powershell
.\.venv\Scripts\python.exe -m backend.app.services.indexing.extract_segments `
  --input data\metadata `
  --captions data\metadata `
  --ocr data\metadata `
  --objects data\metadata `
  --output data\metadata\segments_all.jsonl `
  --strategy auto
>>>>>>> origin/main
```

### 5. Build visual FAISS index

Sau khi mọi video đã có SigLIP2 embedding:

```powershell
.\.venv\Scripts\python.exe -m backend.app.services.indexing.build_faiss_index
```

Builder mặc định đọc `data\embeddings\siglip2_so400m_patch16_384_*.npy` và ghi:

```text
data/indexes/siglip2_so400m_patch16_384_flat_ip.faiss
data/metadata/siglip2_so400m_patch16_384_frame_map.json
data/metadata/siglip2_so400m_patch16_384_faiss_manifest.json
data/metadata/siglip2_so400m_patch16_384_index_report.json
```

### 6. Build lexical text index

```powershell
.\.venv\Scripts\python.exe -m backend.app.services.indexing.build_text_index `
  --metadata data\metadata `
  --output data\indexes\retrieval_text_index.json
```

Text index chỉ chứa caption, OCR và object labels/counts. Nếu `segments_all.jsonl` tồn
tại, builder ưu tiên segment metadata; nếu chưa có, nó đọc các artifact modality riêng.

## Chạy online retrieval

Runtime mặc định đọc path/model contract từ FAISS manifest và
`configs\retrieval.yaml`. Có thể gọi trực tiếp Python wrapper:

```powershell
.\.venv\Scripts\python.exe -c "from backend.app.api.search import search; import json; print(json.dumps(search('a person holding a red bag', top_k=10, mode='hybrid'), ensure_ascii=False, indent=2))"
```

Các mode hiện có:

- `visual`: encode text bằng SigLIP2 và tìm trong FAISS.
- `caption`, `ocr`, `object`: lexical retrieval theo một modality.
- `hybrid`: hợp nhất visual và các text modality hiện có rồi rerank.
- `temporal`: ordered subqueries trong cùng video.
- `qa`: trả evidence phục vụ question answering.

Hybrid tự giảm về visual-only nếu text index chưa tồn tại. Mode text cụ thể sẽ báo
`FileNotFoundError` nếu dependency chưa được build.

Các router tại `backend\app\api\search.py` và `backend\app\api\retrieval.py` có thể được
mount vào FastAPI. Repository hiện chưa cung cấp top-level ASGI app chính thức, vì vậy
không nên chạy một lệnh `uvicorn backend.app.main:app` chưa tồn tại.

### Runtime overrides

Các biến môi trường thường dùng:

```text
RETRIEVAL_CONFIG_PATH
RETRIEVAL_INDEX_PATH
RETRIEVAL_FRAME_MAP_PATH
RETRIEVAL_MANIFEST_PATH
RETRIEVAL_TEXT_INDEX_PATH
RETRIEVAL_MODEL_CACHE_DIR
RETRIEVAL_DEVICE
RETRIEVAL_DEFAULT_TOP_K
RETRIEVAL_MAX_TOP_K
RETRIEVAL_MIN_SCORE
```

## Agent và query expansion

Agent layer chứa query planning, decomposition, tool execution và query expansion an
toàn. Query expansion:

- luôn giữ Original Query;
- sinh tối đa hai paraphrase hợp lệ bằng local Qwen provider;
- bảo vệ OCR literal, số lượng, màu, mã, proper name, phủ định và quan hệ;
- route OCR/object theo structured decomposition;
- cap đóng góp expansion bằng weighted reciprocal-rank fusion;
- dùng Original Query ở dense/metadata/VLM reranker.

Hiện query expansion được tích hợp vào advanced retrieval path và competition runner;
Python wrapper `backend.app.api.search.search` vẫn là stable visual/text/hybrid API thông
thường. Cấu hình expansion nằm trong `configs\retrieval.yaml`.

## Tài liệu liên quan

- [docs/architecture.md](docs/architecture.md): kiến trúc offline/online và service layer.
- [docs/metadata_schema.md](docs/metadata_schema.md): metadata contract.
- [docs/retrieval_api_contract.md](docs/retrieval_api_contract.md): retrieval runtime/API.
- [docs/service_boundaries.md](docs/service_boundaries.md): ranh giới module.
- [backend/README.md](backend/README.md): tóm tắt backend CLI.
- [competition/README.md](competition/README.md): pipeline thi đấu end-to-end.

## Kiểm thử

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s backend\tests -v
.\.venv\Scripts\python.exe -m unittest discover -s competition\tests -v
```

Backend test dùng fixture/fake model nên không tải checkpoint thật. Smoke test SigLIP2
thật chỉ chạy khi bật biến môi trường tương ứng và cache đã có.
