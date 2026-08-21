from __future__ import annotations

import csv
import io
import unittest
from unittest import mock

from backend.app.services.submission.csv_export import (
    SubmissionExportError,
    export_query_csv,
    serialize_trake_csv,
)
from backend.app.services.submission.schemas import ExportRequest, SubmissionTask
from backend.app.models.retrieval import RetrievalResult
from backend.app.services.trake.models import EventCandidate, TrakeHypothesis


def _rows(text: str) -> list[list[str]]:
    return list(csv.reader(io.StringIO(text, newline="")))


def _hypothesis(
    video_id: str,
    frame_ids: list[int],
    *,
    internal_prefix: str = "KF",
) -> dict[str, object]:
    return {
        "video_id": video_id,
        "frame_ids": frame_ids,
        "score": 0.99,
        "lineage": [
            {
                "event_index": index,
                "video_id": video_id,
                "original_frame_index": frame_index,
                "internal_frame_id": f"{internal_prefix}{index}",
                "source": "retrieval_result.frame_index",
            }
            for index, frame_index in enumerate(frame_ids)
        ],
    }


class TrakeSubmissionTests(unittest.TestCase):
    def test_request_supports_trake_and_keeps_submission_bounds(self) -> None:
        request = ExportRequest.parse("  ordered   events ", " TrAkE ", 100)
        self.assertEqual(request.query, "ordered events")
        self.assertIs(request.task, SubmissionTask.TRAKE)
        for value in (0, 101, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ExportRequest.parse("events", "trake", value)  # type: ignore[arg-type]

    def test_serializes_n_plus_one_columns_from_original_frame_lineage(self) -> None:
        response = {
            "event_plan": {"events": [{}, {}, {}, {}]},
            "hypotheses": [
                _hypothesis("folder/L10_V010.mp4", [101, 150, 203, 251]),
            ],
        }
        parsed = _rows(serialize_trake_csv(response))
        self.assertEqual(
            parsed,
            [
                ["L10_V010", "101", "150", "203", "251"],
            ],
        )

    def test_deduplicates_whole_sequence_not_individual_frames(self) -> None:
        first = _hypothesis("V1", [10, 20, 30])
        parsed = _rows(
            serialize_trake_csv(
                [
                    first,
                    dict(first),
                    _hypothesis("V1", [10, 21, 30]),
                    _hypothesis("V2", [10, 20, 30]),
                ]
            )
        )
        self.assertEqual(
            parsed,
            [
                ["V1", "10", "20", "30"],
                ["V1", "10", "21", "30"],
                ["V2", "10", "20", "30"],
            ],
        )

    def test_missing_or_inconsistent_lineage_is_omitted_fail_closed(self) -> None:
        missing = {"video_id": "V1", "frame_ids": [10, 20]}
        wrong_frame = _hypothesis("V1", [11, 20])
        wrong_frame["lineage"][0]["original_frame_index"] = 10  # type: ignore[index]
        wrong_video = _hypothesis("V1", [12, 20])
        wrong_video["lineage"][1]["video_id"] = "V2"  # type: ignore[index]
        missing_source = _hypothesis("V1", [13, 20])
        del missing_source["lineage"][0]["source"]  # type: ignore[index]
        negative = _hypothesis("V1", [14, 20])
        negative["lineage"][0]["original_frame_index"] = -1  # type: ignore[index]
        valid = _hypothesis("V1", [15, 20])

        parsed = _rows(
            serialize_trake_csv(
                {
                    "event_plan": {"events": [{}, {}]},
                    "hypotheses": [
                        missing,
                        wrong_frame,
                        wrong_video,
                        missing_source,
                        negative,
                        valid,
                    ],
                }
            )
        )
        self.assertEqual(parsed, [["V1", "15", "20"]])

    def test_internal_frame_id_and_timestamp_are_never_used_as_fallback(self) -> None:
        response = {
            "event_plan": {"events": [{}, {}]},
            "hypotheses": [
                {
                    "video_id": "V1",
                    "frame_ids": ["KF0001", 20],
                    "events": [
                        {"result": {"frame_id": "123", "timestamp": 1.5}},
                        {"result": {"frame_id": "456", "timestamp": 2.5}},
                    ],
                }
            ],
        }
        parsed = _rows(serialize_trake_csv(response))
        self.assertEqual(
            parsed,
            [],
        )

    def test_model_to_dict_derives_lineage_only_from_retrieval_frame_index(self) -> None:
        candidates = tuple(
            EventCandidate(
                event_index=index,
                result=RetrievalResult(
                    video_id="V1",
                    frame_id=f"INTERNAL_{index}",
                    timestamp=1000.0 + index,
                    score=0.9,
                    faiss_index=9000 + index,
                    frame_index=frame_index,
                ),
                normalized_score=1.0,
            )
            for index, frame_index in enumerate((17, 29))
        )
        hypothesis = TrakeHypothesis(
            video_id="V1",
            frame_ids=(17, 29),
            coarse_candidates=candidates,
        )
        self.assertEqual(
            _rows(serialize_trake_csv([hypothesis]))[0],
            ["V1", "17", "29"],
        )

        missing = TrakeHypothesis(
            video_id="V1",
            frame_ids=(17, 29),
            coarse_candidates=(
                candidates[0],
                EventCandidate(
                    event_index=1,
                    result=RetrievalResult(
                        video_id="V1",
                        frame_id="29",
                        timestamp=29.0,
                        score=0.8,
                        faiss_index=29,
                        frame_index=None,
                    ),
                    normalized_score=0.8,
                ),
            ),
        )
        self.assertEqual(
            _rows(serialize_trake_csv([missing])),
            [],
        )

    def test_limit_is_applied_after_invalid_and_duplicate_rows(self) -> None:
        hypotheses = [_hypothesis("V", [index, index + 200]) for index in range(105)]
        hypotheses.insert(0, {"video_id": "V", "frame_ids": [0, 200]})
        parsed = _rows(serialize_trake_csv(hypotheses, top_k=100))
        self.assertEqual(len(parsed), 100)
        self.assertEqual(parsed[0], ["V", "0", "200"])
        self.assertEqual(parsed[-1], ["V", "99", "299"])

    def test_export_uses_trake_online_route_and_stable_filename(self) -> None:
        online_search = mock.Mock(
            return_value={
                "event_plan": {"events": [{}, {}]},
                "hypotheses": [_hypothesis("V9", [12, 31])],
            }
        )
        exported = export_query_csv(
            "first event then second event",
            "trake",
            7,
            online_search=online_search,
        )
        self.assertEqual(exported.filename, "trake_result.csv")
        self.assertEqual(exported.row_count, 1)
        self.assertEqual(_rows(exported.content)[0], ["V9", "12", "31"])
        self.assertEqual(
            online_search.call_args.kwargs,
            {"query": "first event then second event", "task": "trake", "top_k": 7},
        )

    def test_missing_hypothesis_contract_is_rejected(self) -> None:
        with self.assertRaisesRegex(SubmissionExportError, "ranked hypotheses"):
            serialize_trake_csv({"event_plan": {"events": [{}]}})
        with self.assertRaisesRegex(SubmissionExportError, "event count"):
            serialize_trake_csv([])


if __name__ == "__main__":
    unittest.main()
