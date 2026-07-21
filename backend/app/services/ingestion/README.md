# Ingestion / Metadata (Team P3)

Sinh và gộp metadata cho keyframe: **caption, OCR, ASR, objects** → **unified metadata** / **frame_map enrich**.
Tuân thủ `docs/metadata_schema.md` (v1.1) và `docs/service_boundaries.md` (không chứa search / rerank / agent).

## Thành phần

| File | Vai trò |
|---|---|
| `../../models/metadata.py` | Dataclass schema: Video, Segment, Keyframe, Caption, OCR, ASR, ObjectAnnotation, EmbeddingMetadata, UnifiedMetadataRecord |
| `scheme_validator.py` | Kiểm tra record đúng schema (required fields, kiểu, prefix ID) |
| `caption_pipeline.py` | Caption keyframe — backend `stub` \| `qwen` (Qwen2.5-VL) |
| `ocr_pipeline.py` | OCR keyframe — backend `stub` \| `easyocr` |
| `object_pipeline.py` | Object detection — backend `stub` \| `yolo` (Ultralytics) |
| `asr_pipeline.py` | ASR theo segment — backend `stub` \| `whisper` (faster-whisper) |
| `metadata_builder.py` | Join mọi sidecar theo `frame_id`/`segment_id` → unified metadata + enrich `frame_map.json` |
| `run_*.py` | CLI wrapper |

M��i pipeline có **stub backend** chạy được ngay (không cần GPU/model) để đảm bảo end-to-end,
và **backend thật** nạp model qua *lazy import* (chỉ import khi khởi tạo) để máy GPU cắm vào.

## Chạy (stub — kiểm tra luồng)

```bash
export PYTHONPATH=.
VID=L01_V001

python -B backend/app/services/ingestion/run_caption.py \
  --metadata-path data/metadata/keyframes_$VID.jsonl \
  --output-path   data/metadata/captions_$VID.jsonl --backend stub

python -B backend/app/services/ingestion/run_ocr.py \
  --metadata-path data/metadata/keyframes_$VID.jsonl \
  --output-path   data/metadata/ocr_$VID.jsonl --backend stub

python -B backend/app/services/ingestion/run_object_detection.py \
  --metadata-path data/metadata/keyframes_$VID.jsonl \
  --output-path   data/metadata/objects_$VID.jsonl --backend stub

python -B backend/app/services/ingestion/run_asr.py \
  --video-path data/raw/video/$VID.mp4 \
  --output-path data/metadata/asr_$VID.jsonl --backend stub

python -B backend/app/services/ingestion/metadata_builder.py \
  --keyframe-metadata-path data/metadata/keyframes_$VID.jsonl \
  --output-path            data/metadata/unified_$VID.jsonl \
  --caption-path data/metadata/captions_$VID.jsonl \
  --ocr-path     data/metadata/ocr_$VID.jsonl \
  --objects-path data/metadata/objects_$VID.jsonl \
  --asr-path     data/metadata/asr_$VID.jsonl \
  --enrich-frame-map data/metadata/openclip_vit_b16_frame_map.json \
  --enriched-frame-map-out data/metadata/openclip_vit_b16_frame_map.json
```

## Chạy model thật (máy GPU)

```bash
# cần cài thêm: transformers, easyocr, ultralytics, faster-whisper
python -B backend/app/services/ingestion/run_caption.py ... --backend qwen  --model-id Qwen/Qwen2.5-VL-3B-Instruct
python -B backend/app/services/ingestion/run_ocr.py     ... --backend easyocr --languages en vi
python -B backend/app/services/ingestion/run_object_detection.py ... --backend yolo --model-id yolov8n.pt
python -B backend/app/services/ingestion/run_asr.py     ... --backend whisper --model-id large-v3
```

## Output

- `data/metadata/captions_<vid>.jsonl` → `{frame_id, caption, caption_model}`
- `data/metadata/ocr_<vid>.jsonl` → `{frame_id, ocr_text, ocr_confidence, ocr_model}`
- `data/metadata/objects_<vid>.jsonl` → `{frame_id, objects:[{label,confidence,bbox?}], object_model}`
- `data/metadata/asr_<vid>.jsonl` → `{segment_id, video_id, transcript, language, start_time, end_time, ...}`
- `data/metadata/unified_<vid>.jsonl` → `UnifiedMetadataRecord` (retrieval/UI đọc từ đây)
- `frame_map.json` được bổ sung `caption / ocr_text / objects / transcript` (giữ nguyên key & field cũ)

## Test

```bash
python -B -m unittest backend.tests.test_metadata_pipeline
```
