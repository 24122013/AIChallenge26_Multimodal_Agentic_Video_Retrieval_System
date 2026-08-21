# TRAKE audit handoff — 2026-08-21

## Outcome

The canonical lion-dance statement parses as exactly four events in source order, with boundaries `first_transition`, `first_contact`, `first_transition`, `first_transition`. The exact original query and each event's original text are retained. On the repository's real corpus, which contains only `L27_V001`, the production route now returns `status=insufficient_support`, `hypotheses=[]`, and warnings `no_video_supports_all_events` plus `insufficient_event_support`.

No ASR, Qwen, VLM/cross-encoder reranker, answer model, or KIS CSES was added to TRAKE. `RETRIEVAL_TRAKE_BGE_RERANKER_ENABLED=false` remains unchanged.

## Root causes and fixes

1. Boundary parsing used the first weak `first/đầu tiên` match and therefore hid a later strong `bắt đầu`. The parser now collects all cues/actions and applies an explicit priority.
2. Event text had only one retrieval representation. `TemporalEvent` now additively carries `event_context`, `target_text`, `retrieval_query`, `refinement_query`, normalized text, warnings, semantic confidence, source label and parser trace.
3. `original_query` was stripped/whitespace-collapsed through the online route. TRAKE now preserves the exact input; normalization is used only for deterministic parsing/comparison.
4. Empty list markers were dropped. They now retain an indexed event slot, emit explicit warnings and skip retrieval, causing a fail-closed empty sequence.
5. Plan confidence represented list structure only. It is now derived from separate structural and semantic confidence; per-event trace records matched/rejected cues and fallback reasons.
6. Alignment allowed an equal-frame fallback and ranking accepted non-decreasing frames. Both now require strictly increasing original zero-based frame indexes.
7. Relative RRF/rank-normalized scores always produced a winner for an unrelated corpus. Video gating now requires complete coverage plus configurable absolute pre-RRF SigLIP/BGE support floors.
8. Refinement selected frame-wise semantic maxima and repeatedly decoded/encoded overlapping windows. It now combines semantic state with embedding-delta motion, confirms crossings, fails back on weak/flat motion, and caches decoded frames, text embeddings and image embeddings per request.
9. Canonical retrieval and BGE encoded events sequentially. Context + events are batch encoded where the engine exposes the batch surface; model instances remain process-cached.
10. Cached/local model mode could still permit Hugging Face probes. SigLIP2 and BGE-M3 now set offline flags before importing/loading Transformers when `local_files_only` is active.
11. The UI returned `null` while searching and had no cancellation/timeout. It now retains a visible progress panel with stage, elapsed time, Cancel, AbortController, a configurable TRAKE timeout, HTTP errors, and explicit empty/insufficient-support display.

## Contract before and after

Before, an event exposed `index`, `name`, `original_text`, `retrieval_query`, `boundary_type`, and `protected_terms`; plan `confidence=1.0` could coexist with unknown event semantics. After, all previous fields remain and the additive fields above distinguish coarse retrieval from target refinement. The plan exposes `structural_confidence` and `semantic_confidence`; the public response adds `status`, `warnings`, preflight readiness, support policy, model loads, batch counts and per-stage/cache counters.

Duplicate/skipped labels remain presentation metadata and preserve source order. CRLF, inline `E1:text`, numbered lists, multi-sentence events, internal connectives and prompt-injection-like event data are covered by regression tests.

## Lion-dance parse

The complete serialized parser output is in `reports/trake_lion_parse.json`.

| Event | Context/target result | Boundary |
|---|---|---|
| E1 | context: lion spins on column 4 then lands; target: `Lân bắt đầu xoay vòng trên cột số 4` | `first_transition` |
| E2 | target retains `4 chân`, `hoàn toàn`, `chạm đất`, `đầu tiên` | `first_contact` |
| E3 | target: `2 người biểu diễn lân bắt đầu cúi chào ban giám khảo` | `first_transition` |
| E4 | context: `Lân tiến lại chào một con rồng`; target: `Con rồng bắt đầu cử động đầu` | `first_transition` |

Parser result: source `deterministic_list`, indexes `[0,1,2,3]`, structural confidence `1.0`, semantic confidence `0.92`, overall confidence `0.964`.

## Runtime flow

`SearchBoard/useSearch` → `/api/search` → `OnlinePipeline` exact TRAKE route → deterministic parser → batched canonical hybrid + batched BGE-M3 dense → RRF → absolute-support video gate → strict K-best alignment → bounded cached SigLIP2/motion refinement → strict lineage/ranking validation → sequence-first response → `TrakeDisplay`.

Incomplete paths are never flattened or returned. Timeout is HTTP 504 with a sanitized TRAKE timeout payload. Insufficient support is a normal sequence response with no hypotheses and explicit warnings.

## Production benchmark

Same query, same `L27_V001` corpus, cold process, CUDA, production BGE dense on, refinement configured on:

| Measurement | Before final support fix | Final |
|---|---:|---:|
| Core total | 50,668.7 ms | 19,025.5 ms |
| Event retrieval | 18,418.7 ms | 19,024.2 ms (includes 10,733.2 ms model cold start) |
| Refinement | 32,232.0 ms | 0.012 ms |
| Returned hypotheses | 5 false positives | 0 |
| BGE model forwards | 4 | 1 batched |
| Canonical context/event batch | 1 batch for 5 queries | 1 batch for 5 queries |
| Requested refinement-frame uses | 4,840 | 0 |
| Unique frames decoded/encoded with cache | 740 / 740 | 0 / 0 |

Final wall time was 19,224.3 ms. Cold-load counters: SigLIP2 one load (7,146.0 ms), BGE-M3 one load (3,587.2 ms). The local refiner shares the same SigLIP2 encoder and does not load another checkpoint. The user's earlier 3-event, refinement-off observation of about 130 seconds is retained as an external baseline, but it is not presented as a same-query measurement.

The original no-cache workload for the measured ten paths would decode/encode 4,840 frame uses (10 paths × 4 events × 121 frames). The request cache reduced this to 740 unique frames before support gating made the final out-of-corpus workload zero.

## Validation

- Full backend: `447 passed, 1 skipped, 319 subtests passed` in 9.16 s. The single warning is a Starlette/httpx deprecation from a test dependency.
- Required TRAKE files are included in the full run and pass.
- Frontend node tests: 8/8 passed.
- Frontend TypeScript/Vite production build: passed.
- Frontend ESLint: passed.
- Production out-of-corpus check: `L27_V001`, four events, zero gated videos, zero paths, zero refinement frames, `insufficient_support`.

## Files changed

- Configuration/runtime: `.env`, `configs/retrieval.yaml`, retrieval config/manager, SigLIP2 visual search, BGE dense, hybrid search.
- TRAKE core: `models.py`, `query_parser.py`, `event_retrieval.py`, `candidate_video.py`, `temporal_alignment.py`, `temporal_refinement.py`, `ranking.py`, `pipeline.py`, package exports.
- API/orchestration: both search/retrieval API wrappers and `online_pipeline.py`.
- Frontend: `useSearch.ts`, `App.tsx`, `SearchBoard.tsx`, `TrakeDisplay.tsx`, shared types.
- Tests/reports: TRAKE parser/pipeline tests, TRAKE frontend state tests, this report and the serialized parse fixture.

## Remaining limits

The repository does not contain the lion-dance video, so retrieval correctness for those four moments cannot be measured here; only parser correctness and out-of-corpus rejection are established. Absolute SigLIP/BGE support floors (`0.15`/`0.55`) are configurable and should be calibrated on a representative held-out TRAKE corpus before a competition release. Motion refinement uses deterministic embedding delta, not pose/contact geometry, so “all four feet fully contact” still depends on the complete semantic query and confirmation frames; weak evidence correctly falls back instead of claiming a precise boundary.
