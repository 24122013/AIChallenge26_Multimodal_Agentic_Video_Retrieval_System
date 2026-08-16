# Canonical pipeline audit

## Scope

The supported AIC 2026 practice tasks are textual Known Item Search (KIS) and
grounded Question Answering (QA). TRAKE is planned but not implemented. The
canonical runtime is under `backend/`, with shared indexing helpers under `src/`.

## Offline flow

1. Read original videos from `data/raw/video/`.
2. Extract keyframes and preserve each original zero-based `frame_index` in
   canonical metadata.
3. Generate Qwen3-VL captions, OCR and object evidence.
4. Build visual/text embeddings and indexes under `data/embeddings/` and
   `data/indexes/`.
5. Keep model/cache/report artifacts under their respective `data/` directories.

## Online flow

- KIS uses the existing planned hybrid retrieval stack and keeps its ranking.
- QA uses the complete grounded QA pipeline: shared retrieval, evidence bundle,
  answer generation/validation and citations. It fails closed on abstention,
  insufficient evidence, invalid citations or required-model failure.
- CSV export lives in `backend/app/services/submission/`. It never exports
  score, timestamp, internal IDs, paths or traces.

## Identity contract

`RetrievalResult.frame_id` may be an internal candidate/keyframe identity.
Submission `frame_id` is different: it is the non-negative integer in
`RetrievalResult.frame_index`, copied from the original-video frame map during
indexing. Export intentionally skips candidates without this mapping rather than
guessing from filenames, timestamps or FAISS rows.

## Current risks

- No official `data/sample_submission.csv` is present. Until one is supplied,
  export uses long-form headers and normalizes `video_id` to the filename stem.
- The repository exposes routers but still has no canonical FastAPI application
  factory; route contracts are unit-testable and ready to mount.
- Full model/dataset E2E behavior still requires target-machine profiling and
  offline checkpoint preparation.
