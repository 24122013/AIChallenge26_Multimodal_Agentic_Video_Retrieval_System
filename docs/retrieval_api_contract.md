# Retrieval API Contract v0.1

Owner: P4 Retrieval

Implemented scope:

- Phase 1: SigLIP2 visual text-to-keyframe retrieval.
- Phase 2: lexical caption, OCR and object retrieval.
- Phase 3: hybrid candidate merge/rerank and ordered temporal retrieval.

Phase 2 and multimodal Phase 3 require a generated Retrieval text index. Hybrid
mode degrades to visual-only when that optional index is not present.

## Visual Search

Input:

```json
{
  "query": "a man cooking in a kitchen",
  "top_k": 20
}
```

Output:

```json
{
  "query": "a man cooking in a kitchen",
  "top_k": 20,
  "latency_ms": 123.4,
  "results": [
    {
      "video_id": "L01_V001",
      "frame_id": "FRAME_L01_V001_000001",
      "segment_id": "SEG_L01_V001_000001",
      "shot_id": "SHOT_L01_V001_000001",
      "timestamp": 1.25,
      "timestamp_source": "matched_frame",
      "timestamp_confidence": 0.9,
      "faiss_index": 0,
      "frame_index": 37,
      "score": 0.92,
      "keyframe_path": "data/keyframes/L01_V001/000001.jpg",
      "thumbnail_path": "data/keyframes/L01_V001/000001.jpg",
      "caption": "",
      "ocr_text": "",
      "objects": [],
      "modality_scores": {
        "visual": 0.92
      }
    }
  ]
}
```

## Runtime Inputs

### Current implemented runtime

The current `search_visual.py` implementation expects:

- FAISS index:
  `data/indexes/siglip2_so400m_patch16_384_flat_ip.faiss`
- Frame map:
  `data/metadata/siglip2_so400m_patch16_384_frame_map.json`
- Manifest:
  `data/metadata/siglip2_so400m_patch16_384_faiss_manifest.json`
- Model cache: `data/model_cache/siglip2`

Retrieval reads the model name, revision, vector dimension and normalization
contract from the manifest. It encodes text with
`model.get_text_features(**inputs)` and rejects a query/index dimension
mismatch before FAISS search.

These defaults can be overridden with environment variables:

- `RETRIEVAL_INDEX_PATH`
- `RETRIEVAL_FRAME_MAP_PATH`
- `RETRIEVAL_MANIFEST_PATH`
- `RETRIEVAL_DEVICE`
- `RETRIEVAL_MODEL_CACHE_DIR`
- `RETRIEVAL_NO_AUTOCAST`
- `RETRIEVAL_DEFAULT_TOP_K`
- `RETRIEVAL_MAX_TOP_K`
- `RETRIEVAL_MIN_SCORE`
- `RETRIEVAL_TEXT_INDEX_PATH`
- `RETRIEVAL_CONFIG_PATH`

`RETRIEVAL_MIN_SCORE` is optional. When set, visual candidates below the
threshold are discarded instead of returning a low-confidence match merely to
fill `top_k`.

## Multimodal and Hybrid Search

Build the lexical text index after Metadata has generated caption, OCR,
object or `segments_*.jsonl` artifacts:

```powershell
.\.venv\Scripts\python.exe -B backend\app\services\indexing\build_text_index.py `
  --metadata data\metadata `
  --output data\indexes\retrieval_text_index.json
```

Supported modes in `backend.app.api.search.search`:

- `visual`: SigLIP2 query embedding against FAISS.
- `caption`, `ocr`, `object`: one lexical metadata modality.
- `hybrid`: visual and available text candidates, deduplicated and reranked.
- `temporal`: ordered subqueries in the same video with a configurable time gap.

The mode-specific score is preserved in `modality_scores`. Text-only modes fail
with a dependency-specific `FileNotFoundError` if the text index has not been
built. The HTTP router converts this condition to HTTP 503.

### Encoder contract

The new offline indexing artifacts are:

- FAISS index:
  `data/indexes/siglip2_so400m_patch16_384_flat_ip.faiss`
- Frame map:
  `data/metadata/siglip2_so400m_patch16_384_frame_map.json`
- Manifest:
  `data/metadata/siglip2_so400m_patch16_384_faiss_manifest.json`

The manifest `schema_version` is `1.2`. Its `encoder` object is the authoritative
query-encoder contract:

```json
{
  "model_family": "siglip2",
  "model_name": "google/siglip2-so400m-patch16-384",
  "model_revision": "<resolved-or-requested-revision>",
  "processor_name": "google/siglip2-so400m-patch16-384",
  "vector_dim": 1152,
  "input_resolution": 384,
  "normalized": true,
  "similarity": "cosine",
  "output_dtype": "float32"
}
```

`vector_dim` above is illustrative. Retrieval must trust the value detected and
stored in the actual manifest.

The current retrieval runtime validates this contract, loads the exact model
name and revision, uses the SigLIP2 processor with max-length padding, calls
`get_text_features()`, L2-normalizes to float32 and verifies
`encoder.vector_dim`.

## Definition of Done

- Empty query is rejected.
- `top_k` is clamped to `1..RETRIEVAL_MAX_TOP_K`.
- Invalid FAISS ids are skipped.
- Missing metadata records are skipped.
- Results always include `score`, `video_id`, `frame_id`, `timestamp`, and keyframe paths.
- Low-score filtering is deterministic when `RETRIEVAL_MIN_SCORE` is configured.
- Hybrid mode remains usable as visual-only when multimodal metadata is pending.
