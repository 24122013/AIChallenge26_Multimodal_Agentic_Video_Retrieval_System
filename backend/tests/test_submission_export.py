from __future__ import annotations

import csv
import io
import unittest
from unittest import mock

from backend.app.services.submission.csv_export import (
    SubmissionExportError,
    export_query_csv,
    serialize_kis_csv,
    serialize_qa_csv,
)
from backend.app.services.submission.schemas import ExportRequest, ExportedCsv, SubmissionTask


def rows(text: str) -> list[list[str]]:
    return list(csv.reader(io.StringIO(text, newline="")))


class SubmissionCsvTests(unittest.TestCase):
    def test_kis_schema_rank_dedupe_and_limit_100(self) -> None:
        candidates = [
            {"video_id": "V001.mp4", "frame_id": f"KF{i}", "frame_index": i}
            for i in range(105)
        ]
        candidates.insert(1, {"video_id": "V001", "frame_id": "other", "frame_index": 0})
        parsed = rows(serialize_kis_csv(candidates, top_k=100))
        self.assertEqual(parsed[0], ["video_id", "frame_id"])
        self.assertEqual(len(parsed), 101)
        self.assertEqual(parsed[1:4], [["V001", "0"], ["V001", "1"], ["V001", "2"]])

    def test_keyframe_ordinal_is_never_used_as_original_frame_index(self) -> None:
        parsed = rows(
            serialize_kis_csv(
                [
                    {"video_id": "V1", "frame_id": "000001"},
                    {"video_id": "V1", "frame_id": "KF2", "metadata": {"frame_index": 420}},
                ]
            )
        )
        self.assertEqual(parsed, [["video_id", "frame_id"], ["V1", "420"]])

    def test_qa_unicode_citation_order_and_safe_quoting(self) -> None:
        answer = 'Cô ấy nói "xin chào", rồi đi.\nDòng thứ hai.'
        response = {
            "answer": {"status": "answered", "answer": answer, "evidence_ids": ["E2", "E1"]},
            "evidence": [
                {"evidence_id": "E1", "video_id": "a.mp4", "frame_index": 10},
                {"evidence_id": "E2", "video_id": "b.mp4", "frame_index": 20},
                {"evidence_id": "E3", "video_id": "c.mp4", "frame_index": 30},
            ],
        }
        text = serialize_qa_csv(response, top_k=3)
        parsed = rows(text)
        self.assertEqual(parsed[0], ["video_id", "frame_id", "answer"])
        self.assertEqual([row[:2] for row in parsed[1:]], [["b", "20"], ["a", "10"], ["c", "30"]])
        self.assertTrue(all(row[2] == answer for row in parsed[1:]))

    def test_top_k_bounds_and_trake_are_supported(self) -> None:
        for value in (0, 101):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ExportRequest.parse("query", "kis", value)
        self.assertIs(
            ExportRequest.parse("query", "trake", 10).task,
            SubmissionTask.TRAKE,
        )

    def test_qa_abstain_and_invalid_citation_fail(self) -> None:
        with self.assertRaises(SubmissionExportError):
            serialize_qa_csv(
                {"answer": {"status": "insufficient_evidence", "answer": None}, "evidence": []}
            )
        with self.assertRaises(SubmissionExportError):
            serialize_qa_csv(
                {
                    "answer": {"status": "answered", "answer": "Có", "evidence_ids": ["missing"]},
                    "evidence": [{"evidence_id": "E1", "video_id": "V", "frame_index": 1}],
                }
            )

    def test_export_uses_canonical_online_service(self) -> None:
        kis = export_query_csv(
            "query",
            "kis",
            5,
            online_search=lambda **_: {
                "candidates": [{"video_id": "folder/V9.mp4", "frame_index": 12}]
            },
        )
        self.assertEqual(kis.filename, "kis_result.csv")
        self.assertNotIn("..", kis.filename)
        self.assertEqual(rows(kis.content)[1], ["V9", "12"])

        online_search = mock.Mock(
            return_value={
                "answer": {"status": "answered", "answer": "đáp án", "evidence_ids": ["E1"]},
                "evidence": [{"evidence_id": "E1", "video_id": "V2", "frame_index": 7}],
            }
        )
        qa = export_query_csv(
            "câu hỏi",
            "qa",
            5,
            online_search=online_search,
        )
        self.assertEqual(rows(qa.content)[1], ["V2", "7", "đáp án"])
        self.assertEqual(online_search.call_args.kwargs["task"], "qa")


class SubmissionApiTests(unittest.TestCase):
    def test_api_media_type_disposition_and_filename(self) -> None:
        try:
            from fastapi import FastAPI
            from fastapi.testclient import TestClient
            from backend.app.api import search as search_api
        except ImportError as exc:  # pragma: no cover
            self.skipTest(str(exc))
        app = FastAPI()
        app.include_router(search_api.router)
        exported = ExportedCsv(
            "video_id,frame_id\r\nV1,3\r\n",
            "kis_result.csv",
            1,
            SubmissionTask.KIS,
        )
        with mock.patch.object(search_api, "export_query_csv", return_value=exported):
            response = TestClient(app).post(
                "/search/export", json={"query": "../unsafe", "task": "kis", "top_k": 1}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/csv; charset=utf-8")
        self.assertEqual(
            response.headers["content-disposition"], 'attachment; filename="kis_result.csv"'
        )
        self.assertNotIn("..", response.headers["content-disposition"])

    def test_api_accepts_trake_and_rejects_out_of_range_top_k(self) -> None:
        try:
            from fastapi import FastAPI
            from fastapi.testclient import TestClient
            import backend.app.api.search as search_api
        except ImportError as exc:  # pragma: no cover
            self.skipTest(str(exc))
        app = FastAPI()
        app.include_router(search_api.router)
        client = TestClient(app)
        exported = ExportedCsv(
            "video_id,frame_id_1,frame_id_2\r\nV1,3,7\r\n",
            "trake_result.csv",
            1,
            SubmissionTask.TRAKE,
        )
        with mock.patch.object(search_api, "export_query_csv", return_value=exported):
            trake = client.post(
                "/search/export", json={"query": "events", "task": "trake", "top_k": 10}
            )
        self.assertEqual(trake.status_code, 200)
        self.assertEqual(
            trake.headers["content-disposition"],
            'attachment; filename="trake_result.csv"',
        )
        for top_k in (0, 101):
            with self.subTest(top_k=top_k):
                response = client.post(
                    "/search/export", json={"query": "frame", "task": "kis", "top_k": top_k}
                )
                self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
