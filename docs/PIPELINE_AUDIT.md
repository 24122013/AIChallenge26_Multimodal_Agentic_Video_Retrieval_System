# Canonical pipeline audit

## Scope

The supported public task routes are Known Item Search (KIS), AVS, temporal
evidence, TRAKE (Temporal Retrieval and Alignment of Key Events), and grounded
Question Answering (QA). The canonical runtime is under `backend/`, with shared
indexing helpers under `src/`.

## Offline flow

1. Read original videos from `data/raw/video/`.
2. Extract keyframes and preserve each original zero-based `frame_index` in
   canonical metadata.
3. Generate Qwen3-VL captions, OCR and object evidence.
4. Build visual/text embeddings and indexes under `data/embeddings/` and
   `data/indexes/`.
5. Keep model/cache/report artifacts under their respective `data/` directories.

The sparse frame selected by this flow is a **technical keyframe** used for
indexing. A TRAKE **semantic keyframe** is the frame satisfying one event
criterion (for example first contact, fully leaving, or a local peak). Local
refinement may inspect a bounded window around a technical keyframe, but it does
not create or query a corpus-wide dense-frame index.

## Online flow

- KIS uses the existing planned hybrid retrieval stack and keeps its ranking.
- QA uses the complete grounded QA pipeline: shared retrieval, evidence bundle,
  answer generation/validation and citations. It fails closed on abstention,
  insufficient evidence, invalid citations or required-model failure.
- `temporal` remains the existing QA/evidence route and returns normalized
  candidates plus temporal matches. It is not an alias for TRAKE.
- `trake` uses its own deterministic parser and pipeline under
  `backend/app/services/trake/`: independent hybrid retrieval per ordered event,
  event-local score normalization/diversity, video coverage gating, K-best
  original-frame alignment without a hard maximum gap, optional bounded local
  refinement, then exact dedupe/near-sequence NMS and ranked top-100 output.
- `search_trake()` and `search_online(..., task="trake")` use a corpus-generation
  cached `TrakePipeline`. CLI `online_pipeline --task trake`,
  `POST /retrieval/trake`, `POST /retrieval/online`, and search mode `trake`
  expose the same sequence-first contract. `auto` does not infer TRAKE.
- CSV export lives in `backend/app/services/submission/`. It never exports
  score, timestamp, internal IDs, paths or traces.

## Identity contract

`RetrievalResult.frame_id` may be an internal candidate/keyframe identity.
Submission `frame_id` is different: it is the non-negative integer in
`RetrievalResult.frame_index`, copied from the original-video frame map during
indexing. Export intentionally skips candidates without this mapping rather than
guessing from filenames, timestamps or FAISS rows.

A TRAKE hypothesis has one `video_id`, exactly N `frame_ids`, and N ordered
lineage entries. Each lineage entry must repeat its event index, video and
`original_frame_index`; event retrieval/ranking reject missing original-frame
identity and incomplete/unordered sequences. Submission revalidates the emitted
lineage against `frame_ids` fail-closed and
deduplicates `(video_id, tuple(frame_ids))`, not individual frames. The current
provisional CSV header is `video_id,frame_id_1,...,frame_id_N`.

## TRAKE response and refinement audit

The public response contains `schema_version`, original `query`, `task=trake`,
`top_k`, `event_plan`, ranked `hypotheses`, trace and latency. A compatibility
`candidates` alias may be present, but each item is still a complete sequence.
Trace records parser fallback, per-stage candidate counts, video coverage,
alignment settings, refinement availability/warnings and latency.

`refinement_enabled` is a control-plane flag, not evidence that a semantic model
is loaded. The cached production constructor currently injects no
`LocalFrameScorer`, so refinement returns the canonical coarse frame with
`local_refinement_scorer_unavailable`. The scorer/decoder interfaces and
first-transition/first-leave/peak selection are testable by injection. No
pose/contact model or VLM verifier is wired, and no geometric boundary accuracy
claim is made.

## Current risks

- No official `data/sample_submission.csv` is present. Until one is supplied,
  export uses isolated provisional headers and normalizes `video_id` to the
  filename stem.
- The repository exposes routers but still has no canonical FastAPI application
  factory; route contracts are unit-testable and ready to mount.
- Full model/dataset E2E behavior still requires target-machine profiling and
  offline checkpoint preparation.
- TRAKE sparse retrieval/alignment is implemented and unit-tested, but local
  semantic scorer quality and top-100 R@k require a ground-truth full-corpus
  benchmark; coarse fallback should not be described as verified refinement.
