# Backend

Backend cung cấp ingestion metadata, indexing và retrieval API.

## Ingestion

- `run_caption.py`: Qwen3-VL-8B-Instruct image-to-text, JSON retrieval schema, lazy loading,
  batch, dtype và quantization.
- `run_ocr.py`: PP-OCRv5 detection + Latin recognition cho tiếng Việt/Anh,
  polygon/confidence và Unicode normalization.
- `run_object_detection.py`: YOLOE open-vocabulary, configurable vocabulary và
  evidence-only output.

Các pipeline không ghi đè keyframe source. Mỗi modality có JSONL/report riêng,
cache dưới `data/model_cache/` và resume theo model/revision.

## Indexing và retrieval

- SigLIP2 + FAISS giữ nguyên visual contract.
- Text index v3 gồm caption, OCR và object labels/counts.
- API hỗ trợ visual, caption, OCR, object, hybrid và temporal modes.
- Object evidence là soft signal trong reranking, không phải hard filter.

## Chạy CLI

```powershell
.\.venv\Scripts\python.exe backend\app\services\ingestion\run_caption.py --help
.\.venv\Scripts\python.exe backend\app\services\ingestion\run_ocr.py --help
.\.venv\Scripts\python.exe backend\app\services\ingestion\run_object_detection.py --help
```

Caption mặc định dùng `Qwen/Qwen3-VL-8B-Instruct` tại revision bất biến
`b5bc35aa2d1dc2db88ca1482375afc801511bffb`, cache dưới
`data/model_cache/caption`. Hãy profile VRAM trên máy đích theo dtype,
quantization và batch size; không tái sử dụng ước lượng của model 4B.

Xuất CSV thực hành (chỉ KIS và QA; TRAKE chưa triển khai):

```powershell
python -m backend.app.services.submission.export_query --task kis `
  --query "người mặc áo đỏ cầm điện thoại" --top-k 100 `
  --output data/submissions/kis_result.csv
```

## Test

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s backend\tests -v
```
