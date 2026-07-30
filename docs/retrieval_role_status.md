# Retrieval Role Status

Updated: 2026-07-30

## Current conclusion

Retrieval Phase 1-3 is code-complete on the current working tree:

- SigLIP2 visual search against FAISS and frame map.
- Optional minimum visual score filtering.
- Caption, OCR, ASR and object lexical search.
- Hybrid candidate merge, duplicate removal and weighted reranking.
- Ordered temporal query decomposition and same-video matching.
- Python wrappers and FastAPI router endpoints.

This does not mean production search is data-ready. The current workspace has:

- 16 raw MP4 videos.
- One keyframe metadata file, `keyframes_L27_V009.jsonl`, with 226 records.
- No `segments_*.jsonl`.
- No caption, OCR, ASR or object JSONL artifacts.
- No production FAISS index under `data/indexes`.

The FAISS smoke index under `data/smoke` contains only two cloud/sunset frames.
It is suitable for wiring tests only and cannot validate a cooking query.

## Retrieval verification

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s backend\tests -p "test_*.py"
```

Result: 39 tests passed, 1 environment-dependent test skipped.

With the two-frame smoke index and `RETRIEVAL_MIN_SCORE=0.10`, both `visual` and
`hybrid` return an empty result for `a person is cooking`. This is preferable to
returning an unrelated cloud frame. The production threshold still needs
calibration on labelled queries.

## Required handoffs

### Extraction / Indexing role

Required:

1. Extract keyframes for the remaining videos.
2. Encode all keyframes with the same SigLIP2 manifest contract.
3. Build and validate the production FAISS index, frame map and manifest.

Expected artifacts:

```text
data/indexes/siglip2_so400m_patch16_384_flat_ip.faiss
data/metadata/siglip2_so400m_patch16_384_frame_map.json
data/metadata/siglip2_so400m_patch16_384_faiss_manifest.json
```

Current blocker: installed PyTorch is CPU-only. Encoding 226 frames on CPU did
not finish within a 15-minute validation window. This needs a CUDA-enabled
PyTorch environment or a machine/job that can finish the production encode.

### Metadata role

Required:

1. Generate caption, OCR, object and ASR JSONL artifacts.
2. Build `segments_all.jsonl` or per-video `segments_*.jsonl`.
3. Hand the artifacts to Retrieval, preserving `video_id`, frame/segment IDs,
   timestamps and keyframe paths.

Retrieval can then build its text index:

```powershell
.\.venv\Scripts\python.exe -B backend\app\services\indexing\build_text_index.py `
  --metadata data\metadata `
  --output data\indexes\retrieval_text_index.json
```

The command currently stops with:

```text
No segment or multimodal JSONL artifacts found in data\metadata
```

### Backend role

Required:

1. Install the API runtime dependencies from `requirements.txt`.
2. Provide the project-level `FastAPI()` app.
3. Include `backend.app.api.search.router` or
   `backend.app.api.retrieval.router`.
4. Add deployment-level error logging and health checks.

Current blocker: FastAPI is not installed in the active `.venv`, so router
registration cannot be exercised live here. Plain Python retrieval wrappers are
verified.

### Evaluation role

Required:

1. Provide labelled text queries and relevant video/time ranges.
2. Measure Recall@K, MRR or nDCG per modality and for hybrid search.
3. Calibrate `RETRIEVAL_MIN_SCORE`, modality weights and temporal max gap.

The value `0.10` is a smoke-test threshold, not a production recommendation.

### Frontend role

Required after Backend exposes the API:

- Render video, timestamp, score and same-shot neighbors.
- Show an explicit no-confident-result state for an empty result list.
- Allow visual, hybrid and temporal modes if required by the product flow.

## Production run order

```text
raw videos
  -> keyframes
  -> SigLIP2 image embeddings
  -> production FAISS + frame map + manifest
  -> caption/OCR/object/ASR
  -> segment metadata
  -> Retrieval text index
  -> visual/hybrid/temporal query
  -> evaluation and threshold calibration
  -> Backend API and UI
```
