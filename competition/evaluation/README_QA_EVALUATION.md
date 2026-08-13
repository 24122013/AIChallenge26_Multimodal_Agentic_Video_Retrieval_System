# Phase 0 QA evaluation contract

`qa_example_dataset.jsonl` is a bilingual, synthetic **workflow fixture**. It
covers object, color, OCR, action, count, location, yes/no, identity, and
unanswerable questions. It exists to validate schemas, routing tests, and metric
code. It is not a production dataset and its numbers must not be reported as
retrieval or QA quality.

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
    }
  },
  "evidence": [{"evidence_id": "example-v01-f120"}],
  "answer": {"status": "answered", "answer": "điện thoại"},
  "latency_ms": 42.5,
  "peak_vram_mb": 512.0
}
```

`evaluate_qa_predictions` reports Evidence Hit@1/5/10, MRR, nDCG@10,
answer EM/token-F1, abstention accuracy, task-mode accuracy, answer-type macro
F1, constraint precision/recall/F1, and latency/VRAM distributions when those
resource fields are present.

Use development IDs freely. For the example locked-test workflow, call
`evaluate_locked_test_once` with a receipt path outside source labels. A
successful run creates the receipt atomically; later attempts fail instead of
silently reusing the split. The checked-in locked split is visible and therefore
only demonstrates process. A real locked test must use hidden, adjudicated
labels and a protected result store.
