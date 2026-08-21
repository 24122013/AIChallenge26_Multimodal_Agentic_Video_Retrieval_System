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


if __name__ == "__main__":
    unittest.main()
