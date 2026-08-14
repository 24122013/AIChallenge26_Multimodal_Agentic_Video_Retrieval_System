"""Evaluation contracts and metrics for competition experiments."""

from competition.evaluation.qa_evaluation import (
    QA_ANSWER_TYPES,
    LockedTestReuseError,
    evaluate_answerer_only_predictions,
    evaluate_qa_predictions,
    evaluate_locked_test_once,
    load_qa_dataset,
    load_qa_predictions,
    load_split_manifest,
)
from competition.evaluation.qa_model_benchmark import evaluate_4b_eligibility
from competition.evaluation.qa_quality_gates import (
    evaluate_answerer_only_gates,
    evaluate_real_dev_gates,
    validate_locked_split_manifest,
)

__all__ = [
    "QA_ANSWER_TYPES",
    "LockedTestReuseError",
    "evaluate_answerer_only_predictions",
    "evaluate_qa_predictions",
    "evaluate_locked_test_once",
    "load_qa_dataset",
    "load_qa_predictions",
    "load_split_manifest",
    "evaluate_4b_eligibility",
    "evaluate_answerer_only_gates",
    "evaluate_real_dev_gates",
    "validate_locked_split_manifest",
]
