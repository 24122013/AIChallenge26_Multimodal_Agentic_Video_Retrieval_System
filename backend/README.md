# Backend

Backend cung cấp ingestion metadata, indexing và retrieval API.

## Ingestion

- `run_caption.py`: Qwen3.5-4B multimodal, JSON retrieval schema, lazy loading,
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

## Test

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s backend\tests -v
```
