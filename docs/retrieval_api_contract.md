# Retrieval API Contract v0.1

Owner: P4 Retrieval

Phase 1 scope is visual text-to-keyframe retrieval only.

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
      "objects": []
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
