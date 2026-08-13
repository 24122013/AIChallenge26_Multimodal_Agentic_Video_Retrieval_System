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
```

- Visual: SigLIP2, giữ nguyên embedding/index contract hiện có.
- Caption: `Qwen/Qwen3.5-9B` revision `c202236`, output JSON có cấu trúc và
  caption fallback.
- OCR: `PP-OCRv5_server_det` + `latin_PP-OCRv5_mobile_rec`, hỗ trợ tiếng Việt
  và tiếng Anh.
- Object evidence: `yoloe-26l-seg.pt`, vocabulary cấu hình được. Kết quả chỉ là
  bằng chứng mềm; không dùng làm hard filter loại candidate.
- Text retrieval: caption, OCR và objects.

## Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PaddlePaddle phải được cài riêng theo nền tảng. Ví dụ CPU:

```powershell
pip install paddlepaddle
```

Với CUDA, chọn đúng wheel `paddlepaddle-gpu` theo phiên bản CUDA từ hướng dẫn
PaddlePaddle; không cài đồng thời wheel CPU và GPU. Quantization caption 4/8-bit
cần CUDA và `bitsandbytes` tương thích.

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
