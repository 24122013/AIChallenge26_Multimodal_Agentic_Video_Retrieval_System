# Retrieval API Contract v0.1

Owner: P4 Retrieval

Implemented scope:

- Phase 1: SigLIP2 visual text-to-keyframe retrieval.
- Phase 2: lexical caption, OCR and object retrieval.
- Phase 3: hybrid candidate merge/rerank and ordered temporal retrieval.
- TRAKE: ordered event parsing, event-wise hybrid retrieval, coverage-first video
  gating, K-best original-frame alignment and ranked sequence hypotheses.

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
- `kis_visual`: canonical Visual KIS. It returns `task="kis"`; selected-keyframe
  coarse retrieval is visual-only, then full-dense rescue, per-clip KIS-profile
  CSES and deterministic visual reranking run when the dense bundle is available.
- `caption`, `ocr`, `object`: one lexical metadata modality.
- `hybrid`: visual and available text candidates, deduplicated and reranked.
- `kis_temporal`: Temporal KIS profile. It returns ordinary KIS `candidates` with
  canonical `task="kis"`; when the full dense artifact is available it runs
  dense global rescue, per-clip temporal-profile CSES and deterministic rerank.
- `temporal`: the existing ordered evidence flow used by QA; output is normalized
  candidates plus temporal evidence/matches.
- `trake`: a separate sequence-first task returning one same-video original
  `frame_index` per ordered event, with at most 100 hypotheses.

The mode-specific score is preserved in `modality_scores`. Text-only modes fail
with a dependency-specific `FileNotFoundError` if the text index has not been
built. The HTTP router converts this condition to HTTP 503.

## Unified Online and TRAKE Contract

`search_online()` is the canonical Python entrypoint. TRAKE also exposes a
sequence-specific wrapper:

```python
from backend.app.services.retrieval.retrieval_manager import search_online, search_trake

response = search_online(query="...", task="trake", top_k=100)
visual_kis = search_online(query="a red bus", task="kis_visual", top_k=20)
temporal_kis = search_online(query="first appearance on the cyclo", task="kis_temporal", top_k=100)
core_response = search_trake(query="...", top_k=100)
```

The online wrapper preserves the core sequence schema and adds
`requested_task="trake"`.

TRAKE must be requested explicitly; the `auto` planner does not infer it. The
following FastAPI router contracts are implemented but require an application
to mount them:

- `POST /retrieval/trake` with `{"query":"...","top_k":100}`;
- `POST /retrieval/kis-temporal` with `{"query":"...","top_k":100}`;
- `POST /retrieval/online` with
  `{"query":"...","task":"trake","top_k":100}`;
- `POST /search` with `{"query":"...","mode":"trake","top_k":100}`.

Temporal KIS is distinct from both legacy routes: `task="temporal"` remains the
QA/evidence contract with `temporal_matches`, while `task="trake"` returns ordered
same-video sequence `hypotheses`. A missing optional dense artifact may fall back
to sparse KIS, but the trace must then report both coarse-to-dense and CSES as not
executed.

The direct `visual` diagnostic remains selected-keyframe-only. It must not be
used as proof that dense rescue or CSES executed. Visual KIS fallback uses the
same selected engine but explicitly reports `coarse_to_dense.executed=false` and
`cses.executed=false` without synthetic CSES candidate fields.

The dedicated request model accepts `top_k` in `1..100` and defaults to 100.
Python service calls clamp to `1..min(trake.max_answers,100)`. HTTP retrieval
responses use `{"success":true,"data":<payload>,"message":null}`. The payload
below abbreviates nested retrieval results and the repeated compatibility alias:

```json
{
  "schema_version": "1.0",
  "query": "a runner.\nE1: first touches the bar\nE2: reaches peak height",
  "requested_task": "trake",
  "task": "trake",
  "top_k": 100,
  "event_plan": {
    "original_query": "a runner.\nE1: first touches the bar\nE2: reaches peak height",
    "context": "a runner",
    "events": [
      {
        "index": 0,
        "name": "event_1",
        "original_text": "first touches the bar",
        "retrieval_query": "a runner. first touches the bar",
        "boundary_type": "first_contact",
        "protected_terms": ["first", "touches"]
      },
      {
        "index": 1,
        "name": "event_2",
        "original_text": "reaches peak height",
        "retrieval_query": "a runner. reaches peak height",
        "boundary_type": "peak",
        "protected_terms": ["peak"]
      }
    ],
    "parser_source": "deterministic_list",
    "confidence": 1.0,
    "warnings": []
  },
  "hypotheses": [
    {
      "rank": 1,
      "video_id": "L10_V010",
      "frame_ids": [101, 203],
      "score": 0.82,
      "score_breakdown": {
        "event_scores": [0.9, 0.8],
        "coverage": 1.0,
        "gap_penalty": 0.05,
        "duplicate_location_penalty": 0.0
      },
      "path_id": "TRP-...",
      "events": [
        {
          "event_index": 0,
          "normalized_score": 0.9,
          "rank": 1,
          "retrieval_query": "a runner. first touches the bar",
          "warnings": [],
          "result": {
            "video_id": "L10_V010",
            "frame_id": "INTERNAL_KF_001",
            "frame_index": 101
          }
        },
        {
          "event_index": 1,
          "normalized_score": 0.8,
          "rank": 1,
          "retrieval_query": "a runner. reaches peak height",
          "warnings": [],
          "result": {
            "video_id": "L10_V010",
            "frame_id": "INTERNAL_KF_002",
            "frame_index": 203
          }
        }
      ],
      "lineage": [
        {
          "event_index": 0,
          "video_id": "L10_V010",
          "original_frame_index": 101,
          "internal_frame_id": "INTERNAL_KF_001",
          "source": "canonical_metadata"
        },
        {
          "event_index": 1,
          "video_id": "L10_V010",
          "original_frame_index": 203,
          "internal_frame_id": "INTERNAL_KF_002",
          "source": "canonical_metadata"
        }
      ],
      "warnings": ["local_refinement_scorer_unavailable"]
    }
  ],
  "candidates": [{"rank": 1, "video_id": "L10_V010", "frame_ids": [101, 203]}],
  "trace": {
    "alignment": {"ordering_field": "original_frame_index", "hard_max_gap": null},
    "refinement": {"scorer_available": false, "fallback_is_canonical_frame_index": true}
  },
  "latency_ms": 12.3
}
```

`hypotheses` is the ranked unit. The `candidates` field is a compatibility alias
containing the same complete objects, never flattened events. Every valid
hypothesis has one video, exactly N non-negative ordered original frame indexes,
and per-event lineage. Internal `RetrievalResult.frame_id`, timestamps, filenames
and FAISS row IDs are not submission frame IDs.

Run the same contract from the CLI:

```powershell
python -m backend.app.pipelines.online_pipeline `
  --task trake --top-k 100 --query "context... E1: ... E2: ..."
```

The parser is deterministic and treats query text as untrusted data. Canonical
queries contain optional free-form context followed by `E1:`, `E2:`... event
markers. Source order is retained even when labels are duplicated or skipped;
legacy numbered/bulleted inputs remain compatible. Boundary-critical terms are
preserved and mapped conservatively to `first_contact`, `first_leave`,
`first_transition`, `peak`, `state`, or `unknown`.

### TRAKE runtime and local-refinement limitation

`configs/retrieval.yaml:trake` validates event retrieval width, video scoring
weights, rank/percentile normalization, beam/DP alignment, soft gap penalty,
local window, diversity, max answers and evaluation cutoffs. There is no hard
180-second gap filter in the TRAKE aligner.

Local decode is bounded around coarse original frames and resolves only
`<RETRIEVAL_TRAKE_VIDEO_ROOT>/<video_id>.mp4` after path-traversal checks. The
cached production pipeline currently injects no `LocalFrameScorer`, so it emits
canonical coarse frames with `local_refinement_scorer_unavailable`. Decoder and
scorer protocols support controlled deployments/tests, but no pose/contact model
or VLM verifier is active.

### Submission export

`POST /search/export` accepts `task="trake"` and returns raw UTF-8 CSV with a
stable `trake_result.csv` filename. The equivalent CLI is:

```powershell
python -m backend.app.services.submission.export_query `
  --task trake --top-k 100 --query "context... E1: ... E2: ..." `
  --output data/submissions/trake_result.csv
```

Until an official sample submission is added, the isolated header assumption is
`video_id,frame_id_1,...,frame_id_N`. Each `frame_id_j` is original zero-based
`frame_index`. Export validates lineage fail-closed, writes no score/timestamp/
path/trace, and exact-deduplicates the whole `(video_id, tuple(frame_ids))`
sequence.

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
- `task="temporal"` retains the QA/evidence contract; `task="trake"` returns
  complete sequences and never flattens them into ordinary candidates.
- TRAKE outputs at most 100 hypotheses, each with exactly N same-video original
  frame indexes and explicit lineage; missing lineage is rejected rather than
  inferred from another identity field.
- TRAKE submission export accepts only 1..100 rows, deduplicates whole sequences
  and emits original `frame_index` values under the provisional `frame_id_N`
  columns.
