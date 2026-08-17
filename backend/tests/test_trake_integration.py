from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from backend.app.api import retrieval as retrieval_api
from backend.app.api import search as search_api
from backend.app.pipelines.online_pipeline import (
    SUPPORTED_ONLINE_TASKS,
    build_parser,
)
from backend.app.services.retrieval import retrieval_manager
from backend.app.services.retrieval.retrieval_config import TrakeConfig
from backend.app.services.trake import RequiredTrakePipelineError


class TrakePublicIntegrationTest(unittest.TestCase):
    def test_cli_and_api_models_accept_trake_with_public_limit(self) -> None:
        self.assertIn("trake", SUPPORTED_ONLINE_TASKS)
        args = build_parser().parse_args(
            ["--task", "trake", "--query", "enter then sit", "--top-k", "100"]
        )
        self.assertEqual(args.task, "trake")
        self.assertEqual(args.top_k, 100)

        body = retrieval_api.TrakeSearchBody(query="enter then sit", top_k=100)
        self.assertEqual(body.top_k, 100)
        self.assertIn(
            "/retrieval/trake",
            {route.path for route in retrieval_api.router.routes},
        )

    def test_api_aliases_delegate_to_sequence_route(self) -> None:
        response = {"task": "trake", "hypotheses": []}
        with mock.patch.object(
            retrieval_api,
            "search_trake",
            return_value=response,
        ) as direct:
            self.assertIs(retrieval_api.trake_search("query", 50), response)
        direct.assert_called_once_with(query="query", top_k=50)

        with mock.patch.object(
            search_api,
            "search_online",
            return_value=response,
        ) as online:
            self.assertIs(
                search_api._dispatch_search("query", 150, "trake"),
                response,
            )
        online.assert_called_once_with(query="query", task="trake", top_k=100)

    def test_manager_search_trake_uses_cached_pipeline_and_caps_top_k(self) -> None:
        pipeline = mock.Mock()
        pipeline.search.return_value = {"task": "trake", "hypotheses": []}
        runtime = SimpleNamespace(trake=TrakeConfig())
        with (
            mock.patch.object(
                retrieval_manager,
                "get_runtime_config",
                return_value=runtime,
            ),
            mock.patch.object(
                retrieval_manager,
                "get_trake_pipeline",
                return_value=pipeline,
            ),
        ):
            response = retrieval_manager.search_trake("query", top_k=500)

        self.assertEqual(response["task"], "trake")
        pipeline.search.assert_called_once_with(query="query", top_k=100)

    def test_required_trake_dependency_maps_to_sanitized_503(self) -> None:
        error = RequiredTrakePipelineError(
            "Required TRAKE BGE is unavailable",
            failure_code="required_bge_unavailable",
        )
        body = retrieval_api.TrakeSearchBody(query="event", top_k=5)
        with mock.patch.object(retrieval_api, "search_trake", side_effect=error):
            response = retrieval_api.trake_search_endpoint(body)

        self.assertEqual(response.status_code, 503)
        self.assertNotIn("event", response.body.decode("utf-8"))
        self.assertIn("required_bge_unavailable", response.body.decode("utf-8"))

        search_body = search_api.SearchBody(
            query="private query",
            mode="trake",
            top_k=5,
        )
        with mock.patch.object(search_api, "_dispatch_search", side_effect=error):
            alias_response = search_api.search_endpoint(search_body)

        self.assertEqual(alias_response.status_code, 503)
        self.assertNotIn("private query", alias_response.body.decode("utf-8"))
        self.assertIn(
            "required_bge_unavailable",
            alias_response.body.decode("utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
