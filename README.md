# HCMC AI Challenge 2026 — Multimodal Video Retrieval

Hệ thống truy hồi video theo keyframe, giữ kiến trúc SigLIP2 + FAISS và bổ sung
metadata thị giác phục vụ retrieval. Pipeline hiện tại không đọc hoặc xử lý âm
thanh.

## Kiến trúc hiện tại

```text
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
  `Qwen/Qwen3.5-9B@c202236`, mặc định tắt và chỉ load khi chạy QA.

## Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

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

Model được lazy-load và cache dưới `data/model_cache/`. Lần chạy đầu cần mạng để
tải checkpoint nếu cache chưa có.

## Chạy từng metadata pipeline

Caption:

```powershell
.\.venv\Scripts\python.exe backend\app\services\ingestion\run_caption.py `
  --metadata-path data\metadata\keyframes_L01_V001.jsonl `
  --device cuda --batch-size 2 --dtype auto --segment-caption
```

OCR:

```powershell
.\.venv\Scripts\python.exe backend\app\services\ingestion\run_ocr.py `
  --metadata-path data\metadata\keyframes_L01_V001.jsonl `
  --device cuda --batch-size 4 --conf-threshold 0.3
```

YOLOE:

```powershell
.\.venv\Scripts\python.exe backend\app\services\ingestion\run_object_detection.py `
  --metadata-path data\metadata\keyframes_L01_V001.jsonl `
  --device cuda --batch-size 8 `
  --vocabulary person face clothing bag phone computer screen book bottle cup food table chair vehicle car motorcycle bicycle bus sign animal
```

Mỗi pipeline ghi JSONL riêng và report provenance gồm model, revision, package
version, device, tham số và thời gian chạy. Resume chỉ tái sử dụng record có
cùng model/revision; thay checkpoint sẽ tạo lại đúng artifact modality đó, không
xóa dữ liệu lịch sử khác.

## Chạy competition pipeline

```powershell
.\.venv\Scripts\python.exe -m competition.pipeline keyframes `
  --public-root data\public `
  --output-root competition\artifacts `
  --device cuda --resume

.\.venv\Scripts\python.exe -m competition.pipeline index `
  --public-root data\public --output-root competition\artifacts

.\.venv\Scripts\python.exe -m competition.pipeline segments `
  --public-root data\public --output-root competition\artifacts

.\.venv\Scripts\python.exe -m competition.pipeline text-index `
  --public-root data\public --output-root competition\artifacts
```

Xem đầy đủ workflow và tùy chọn model trong
[`competition/README.md`](competition/README.md).

## Colab end-to-end và contract submission

Notebook [`notebooks/colab_retrieval_v2_launcher.ipynb`](notebooks/colab_retrieval_v2_launcher.ipynb)
chạy đủ 10 stage, build BGE-M3 index, dùng BGE reranker cho TKIS rồi tạo
`results/submission.csv`. Output public **không đổi**: đúng 100 query trong
`questions.csv` (50 TKIS + 50 VKIS), mỗi query 100 answer theo đúng thứ tự và
header của `sample_submission.csv`.

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
```

OCR record chứa `ocr_text`, `ocr_text_normalized`, `ocr_text_unaccented`,
`raw_ocr_text`, `text_regions[]` với polygon/confidence/language và
`image_size`.

Object record chứa `objects[]` với `class_id`, `class_name`, `confidence`,
`bbox_xyxy`; đồng thời có `object_counts`, vocabulary, prompt mode và
`evidence_only: true`.

## VRAM thực tế nên dự trù

Các số sau là ước lượng vận hành, phụ thuộc độ phân giải ảnh và độ dài output:

- RTX 5090 32 GB: Qwen BF16 batch 1–2; nếu thiếu VRAM dùng batch 1 hoặc 4-bit.
- A100 40 GB: Qwen BF16 batch 2–4.
- A100 80 GB: Qwen BF16 batch 4–8 sau khi benchmark workload thật.
- PP-OCRv5 và YOLOE nhẹ hơn đáng kể; tăng batch riêng sau khi caption model đã
  được giải phóng khỏi GPU.

Không chạy đồng thời các model nặng trong orchestration mặc định. Pipeline dùng
chung một backend theo corpus rồi giải phóng model giữa các modality.

## Kiểm thử

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s backend\tests -v
.\.venv\Scripts\python.exe -m unittest discover -s competition\tests -v
```

Các test ingestion dùng fake backend nên không tải checkpoint thật. Smoke test
GPU/model thật cần được chạy riêng trên máy có đủ VRAM và cache/model access.
