from __future__ import annotations

import unittest
from unittest import mock

from backend.app.api import retrieval as retrieval_api
from backend.app.api import search as search_api


@unittest.skipIf(
    retrieval_api.router is None or search_api.router is None,
    "FastAPI is not installed",
)
class OnlineApiContractTest(unittest.TestCase):
    def test_retrieval_online_forwards_explicit_context_and_debug(self) -> None:
        body = retrieval_api.OnlineSearchBody(
            query="a red car",
            task="kis",
            top_k=7,
            expanded_queries=["red vehicle"],
            include_context=True,
            debug=False,
        )
        expected = {"task": "kis", "candidates": []}

        with mock.patch.object(
            retrieval_api,
            "search_online",
            return_value=expected,
        ) as search_online:
            response = retrieval_api.online_search_endpoint(body)

        self.assertEqual(response["data"], expected)
        search_online.assert_called_once_with(
            "a red car",
            task="kis",
            top_k=7,
            expanded_queries=["red vehicle"],
            include_context=True,
            debug=False,
        )

    def test_retrieval_online_omits_unspecified_overrides(self) -> None:
        body = retrieval_api.OnlineSearchBody(query="a red car")

        with mock.patch.object(
            retrieval_api,
            "search_online",
            return_value={},
        ) as search_online:
            retrieval_api.online_search_endpoint(body)

        self.assertIsNone(body.include_context)
        self.assertIsNone(body.debug)
        search_online.assert_called_once_with(
            "a red car",
            task="auto",
            top_k=20,
            expanded_queries=[],
        )

    def test_unified_search_forwards_explicit_false_values(self) -> None:
        body = search_api.SearchBody(
            query="a blue bicycle",
            mode="avs",
            top_k=9,
            include_context=False,
            debug=False,
        )
        expected = {"task": "avs", "candidates": []}

        with mock.patch.object(
            search_api,
            "search_online",
            return_value=expected,
        ) as search_online:
            response = search_api.search_endpoint(body)

        self.assertEqual(response["data"], expected)
        search_online.assert_called_once_with(
            query="a blue bicycle",
            task="avs",
            top_k=9,
            include_context=False,
            debug=False,
        )

    def test_unified_search_omits_unspecified_overrides(self) -> None:
        with mock.patch.object(
            search_api,
            "search_online",
            return_value={},
        ) as search_online:
            search_api.search("a blue bicycle", 9, "avs")

        search_online.assert_called_once_with(
            query="a blue bicycle",
            task="avs",
            top_k=9,
        )

    def test_unified_qa_forwards_requested_frame_count(self) -> None:
        expected = {"task": "qa", "evidence": [], "answer": {"status": "disabled"}}
        with mock.patch.object(
            search_api,
            "search_online",
            return_value=expected,
        ) as search_online:
            response = search_api.search("What is written on the sign?", 100, "qa")

        self.assertEqual(response, expected)
        search_online.assert_called_once_with(
            query="What is written on the sign?",
            task="qa",
            top_k=100,
            expanded_queries=[],
        )

    def test_unified_search_accepts_kis_temporal_as_a_kis_profile(self) -> None:
        expected = {"task": "kis", "candidates": []}
        with mock.patch.object(
            search_api,
            "search_online",
            return_value=expected,
        ) as search_online:
            response = search_api.search(
                "Khoảnh khắc đầu tiên người dẫn xuất hiện trên xích lô",
                100,
                "kis_temporal",
            )

        self.assertEqual(response, expected)
        search_online.assert_called_once_with(
            query="Khoảnh khắc đầu tiên người dẫn xuất hiện trên xích lô",
            task="kis_temporal",
            top_k=100,
        )

    def test_unified_search_routes_kis_visual_to_online_pipeline(self) -> None:
        expected = {"task": "kis", "candidates": []}
        with mock.patch.object(
            search_api,
            "search_online",
            return_value=expected,
        ) as search_online:
            response = search_api.search("a red bus", 20, "kis_visual")

        self.assertEqual(response, expected)
        search_online.assert_called_once_with(
            query="a red bus",
            task="kis_visual",
            top_k=20,
        )

    def test_visual_diagnostic_route_remains_direct(self) -> None:
        visual_response = mock.Mock()
        visual_response.to_dict.return_value = {"task": "visual", "results": []}
        with (
            mock.patch.object(
                search_api,
                "search_visual",
                return_value=visual_response,
            ) as search_visual,
            mock.patch.object(search_api, "search_online") as search_online,
        ):
            response = search_api.search("a red bus", 20, "visual")

        self.assertEqual(response["task"], "visual")
        search_visual.assert_called_once_with(query="a red bus", top_k=20)
        search_online.assert_not_called()


if __name__ == "__main__":
    unittest.main()
