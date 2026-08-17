# Evaluation Protocol v0

Owner: P1/P4

Phase 1 retrieval evaluation is intentionally small. The goal is to prove that
the baseline runs end-to-end before optimizing ranking quality.

## Metrics

- `Recall@K`: whether a known relevant video/frame appears in top K.
- `MRR`: reciprocal rank of the first relevant result.
- `Latency`: wall-clock query time in milliseconds.

## Baseline Query Set

Use 20-50 natural-language queries against the current subset. Each query should
store:

```json
{
  "query": "a man cooking in a kitchen",
  "relevant_video_id": "L01_V001",
  "relevant_frame_id": "FRAME_L01_V001_000123"
}
```

`relevant_frame_id` can be empty when only video-level ground truth is available.

## Phase 1 Pass Criteria

- Visual search returns top-k keyframes for every query.
- Average latency is reported.
- At least one Recall@K number is reported, even if the ground truth is rough.
- Failure cases are written down for the next iteration.

## TRAKE evaluation

TRAKE evaluates ranked **sequence hypotheses**, not independent retrieved
keyframes. A prediction and its ground truth use original zero-based video frame
indexes:

```json
{
  "prediction": {
    "video_id": "L10_V010",
    "frame_ids": [101, 156, 203, 251]
  },
  "ground_truth": {
    "video_id": "L10_V010",
    "intervals": [[95, 105], [145, 155], [195, 205], [245, 255]]
  }
}
```

`frame_ids[j]` must correspond to event j. It is the original-video
`frame_index`, not internal `RetrievalResult.frame_id`, a selected-keyframe
ordinal, timestamp, filename or FAISS row. Each `[s_j,e_j]` interval is inclusive.

For N events:

```text
wrong video_id:
    R-Score = 0

correct video_id:
    R-Score = (1/N) * sum_j I(s_j <= frame_ids[j] <= e_j)

R@k = max R-Score among the first k ranked hypotheses
k in {1, 5, 20, 50, 100}

Final Score = mean(R@1, R@5, R@20, R@50, R@100)
```

The example above scores `0.75`: frames 101, 203 and 251 hit their intervals;
frame 156 misses `[145,155]`. A sequence with identical frames but the wrong
video scores `0`.

Pure functions are in
`backend/app/services/evaluation/trake_metrics.py` and are re-exported by
`backend.app.services.evaluation.metrics`:

```python
from backend.app.services.evaluation.trake_metrics import (
    best_r_at_k,
    trake_final_score,
    trake_metrics_report,
    trake_r_score,
)

r_score = trake_r_score(prediction, ground_truth)
r_at_20 = best_r_at_k(ranked_predictions, ground_truth, 20)
final = trake_final_score(ranked_predictions, ground_truth)
report = trake_metrics_report(ranked_predictions, ground_truth)
```

The evaluator fails closed on:

- prediction/ground-truth event-count mismatch;
- empty, negative or reversed intervals;
- non-integer or negative submitted frames;
- duplicate whole hypotheses `(video_id, tuple(frame_ids))`;
- more than 100 hypotheses;
- invalid or duplicate cutoffs.

Empty ranked output is valid and scores zero. Two hypotheses may reuse one or
more event frames when their complete sequences differ.

### TRAKE diagnostic report

Alongside `r_at_1`, `r_at_5`, `r_at_20`, `r_at_50`, `r_at_100` and
`final_score`, the one-query report contains:

- `video_at_1`, `video_at_5`, `video_at_20`: whether the correct video occurs by
  that rank, independent of frame hits;
- `per_event_hit_rate`: the hit vector for the earliest best-scoring hypothesis
  within the largest evaluated cutoff;
- `matched_event_ratio`: mean of that vector, equal to that hypothesis's
  R-Score when its video is correct;
- `hypothesis_count` and `event_count`.

The report intentionally does not combine successful events from different
hypotheses into a synthetic perfect sequence. Dataset-level reporting should
average the per-query official metrics over annotated queries and separately
aggregate Video@k and event-position hit rates. When no ground truth is supplied,
leave these metrics absent/null rather than treating retrieval confidence as a
hit.

### TRAKE benchmark checklist

For each benchmark run, persist:

1. query text and exact ordered event annotation;
2. correct video and inclusive original-frame intervals;
3. full ranked output up to 100 hypotheses;
4. `event_plan`, lineage-safe `frame_ids`, warnings and stage trace;
5. R@1/5/20/50/100, Final Score, Video@1/5/20 and per-event hits;
6. total and stage latency plus the retrieval config/corpus generation.

Compare at least event-wise retrieval, chronological alignment, coverage gating,
K-best alignment, sequence diversity and any injected local scorer as separate
ablations. Do not report the default coarse fallback as verified semantic
refinement: production currently has no injected `LocalFrameScorer`. The decoder
only resolves `<video_root>/<video_id>.mp4`; no pose/contact model or VLM verifier
is active. Full-corpus semantic-boundary accuracy therefore remains an empirical
benchmark requirement.
