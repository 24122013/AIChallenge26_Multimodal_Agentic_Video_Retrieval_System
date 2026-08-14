# Phase 0 QA evaluation contract

`qa_example_dataset.jsonl` is a bilingual, synthetic **workflow fixture**. It
covers the eight semantic answer slots: object, color, OCR, action, count,
location, yes/no, and identity. It also contains answerable and unanswerable
outcomes. Answerability is deliberately not an answer type: for example, a
question asking for an unseen name remains `identity` with `answerable=false`.
It exists to validate schemas, routing tests, and metric code. It is not a
production dataset and its numbers must not be reported as retrieval or QA
quality.

Every label contains `question`, `task_mode`, `answer_type`,
`known_constraints`, `gold_evidence`, `gold_answer`, and `answerable`, plus a
stable `query_id`. A prediction may provide:

```json
{
  "query_id": "qa-dev-vi-object",
  "query_plan": {
    "task_mode": "qa",
    "answer_type": "object",
    "known_constraints": {
      "subject": ["người phụ nữ"],
      "attributes": ["áo đỏ"],
      "actions": ["cầm"]
    },
    "constraint_roles": {
      "subject": {"người phụ nữ": "context"},
      "attributes": {"áo đỏ": "context"},
      "actions": {"cầm": "context"}
    },
    "answer_event_index": null
  },
  "evidence": [{
    "evidence_id": "E001",
    "video_id": "example-v01",
    "frame_id": "120"
  }],
  "answer": {
    "status": "answered",
    "answer": "điện thoại",
    "evidence_ids": ["E001"]
  },
  "latency_ms": 42.5,
  "peak_vram_mb": 512.0
}
```

`evaluate_qa_predictions` reports Evidence Hit@1/5/10, MRR, nDCG@10,
answer EM/token-F1, abstention accuracy, task-mode accuracy, answer-type macro
F1, constraint precision/recall/F1, Temporal Chain Hit@5, integrity rates, and
latency/VRAM distributions when those resource fields are present. It also
emits dataset/query-set SHA-256 values so a real report can be bound to its
frozen manifest.

`known_constraints` only accepts `subject`, `objects`, `attributes`, `actions`,
`locations`, and `ocr_terms`. Parser output assigns every phrase a `context` or
`hypothesis` role. A yes/no hypothesis such as `green`, `OPEN`, or `mũ bảo hiểm`
must not be treated as positive retrieval evidence.

Gold evidence must use stable lineage: `video_id` plus `shot_id`, `frame_id`, or
a `start_time`/`end_time` window. A gold frame set may use `frame_ids`. A
predicted point `timestamp` matches an inclusive gold time window only when the
video ID also matches. IDs such as `E001` are local response citation handles
only. An abstention is valid only when the response explicitly contains
`status="insufficient_evidence"` and `answer=null`; errors, disabled responses,
missing fields, and contradictory status/answer pairs count as invalid failures.

For a temporal label, add one or more acceptable ordered chains. Each event
must use the same stable lineage contract and every chain is limited to five
events:

```json
{
  "gold_temporal_chains": [[
    {"video_id": "V001", "shot_id": "S001"},
    {"video_id": "V001", "start_time": 12.0, "end_time": 14.0}
  ]]
}
```

Only a prediction in the top five with `match_mode="strict"`, the same event
count, and a lineage match at every ordered position scores a temporal hit.
`relaxed_gap` and `sparse_compat` remain inspection-only.

The reported `unsupported_answer` is intentionally a conservative
**gold-lineage proxy**: an answered response is flagged when it answers an
unanswerable label, cites an invalid local ID, or none of its cited evidence
matches gold lineage. It does not prove semantic entailment. A publishable
unsupported-claim rate still needs adjudication or a separately pinned judge.

Use development IDs freely. For the example locked-test workflow, call
`evaluate_locked_test_once` with a receipt path outside source labels. A
successful run creates the receipt atomically; later attempts fail instead of
silently reusing the split. The checked-in locked split is visible and therefore
only demonstrates process. A real locked test must use hidden, adjudicated
labels and a protected result store.

## Real gates and 9B versus 4B benchmark

`qa_real_split_manifest.template.json` is a fail-closed template, not the
requested 80+80 dataset. Fill the exact query IDs, 64-character SHA-256 values,
40 Vietnamese/40 English counts, and adjudication fields only after the real
labels exist. `evaluate_real_dev_gates()` and
`evaluate_answerer_only_gates()` in `qa_quality_gates.py` consume the evaluator
report directly and reject missing, non-finite, out-of-range, placeholder, or
manifest-mismatched inputs. For answerer-only evaluation, call
`evaluate_answerer_only_predictions()` first; it refuses any prediction whose
evidence lineage differs from gold. The stricter answerer gate will not relabel
an ordinary retrieval report as `evidence_source=gold`.

The model comparison consumes two cache-miss reports made with the same frozen
adjudicated 80-query real-dev dataset, evidence, prompt, decoding, GPU/software
stack, and 4-bit mode. Each manifest must record 40 Vietnamese/40 English,
the frozen quotas/query hash, and `gpu_memory_gb >= 16`; every report row must
prove `model_invoked=true` and `cache_hit=false`:

```powershell
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe -m competition.evaluation.qa_model_benchmark `
  --baseline-report reports\qa\qwen35-9b.json `
  --candidate-report reports\qa\qwen35-4b.json `
  --baseline-manifest reports\qa\qwen35-9b.manifest.json `
  --candidate-manifest reports\qa\qwen35-4b.manifest.json `
  --output reports\qa\qwen35-4b-eligibility.json
```

Exit code `0` means 4B met every promotion gate; exit code `2` means it did
not. The comparison never changes the runtime default from pinned Qwen3.5-9B.
No real Qwen/BGE result is checked into this repository, so unit-test success
must not be presented as model quality or 4B eligibility.
