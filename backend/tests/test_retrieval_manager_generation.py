from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from backend.app.services.retrieval import retrieval_manager


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RetrievalManagerGenerationCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.metadata = self.root / "metadata"
        self.indexes = self.root / "indexes"
        self.metadata.mkdir(parents=True)
        self.indexes.mkdir(parents=True)
        self.paths = {
            "visual_index": self.indexes / "visual.faiss",
            "visual_frame_map": self.metadata / "frame_map.json",
            "visual_manifest": self.metadata / "visual_manifest.json",
            "text_index": self.indexes / "text.json",
        }
        self.corpus_manifest = self.metadata / "offline_corpus_manifest.json"
        self.environment = {
            "RETRIEVAL_CORPUS_MANIFEST_PATH": str(self.corpus_manifest),
            "RETRIEVAL_INDEX_PATH": str(self.paths["visual_index"]),
            "RETRIEVAL_FRAME_MAP_PATH": str(self.paths["visual_frame_map"]),
            "RETRIEVAL_MANIFEST_PATH": str(self.paths["visual_manifest"]),
            "RETRIEVAL_TEXT_INDEX_PATH": str(self.paths["text_index"]),
            "QA_ANSWER_MODE": "off",
        }
        retrieval_manager.clear_retrieval_caches()

    def tearDown(self) -> None:
        retrieval_manager.clear_retrieval_caches()
        self._temporary.cleanup()

    def _publish(self, revision: str) -> str:
        artifacts: dict[str, dict[str, str]] = {}
        hashes: dict[str, str] = {}
        for role, path in self.paths.items():
            path.write_text(f"{role}:{revision}\n", encoding="utf-8")
            digest = _sha256(path)
            hashes[role] = digest
            artifacts[role] = {
                "path": path.relative_to(self.root).as_posix(),
                "sha256": digest,
            }
        generation = retrieval_manager._generation(hashes)
        manifest = {
            "schema_version": "1.0",
            "status": "passed",
            "revision": revision,
            "bge_enabled": False,
            "artifacts": artifacts,
            "bundle_generation": generation,
        }
        self.corpus_manifest.write_text(
            json.dumps(manifest, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return generation

    def _publish_with_context(
        self,
        revision: str,
        *,
        neighbor_payload: str,
    ) -> tuple[str, Path]:
        self._publish(revision)
        neighbor_path = self.metadata / "neighbors_all.jsonl"
        neighbor_path.write_text(neighbor_payload, encoding="utf-8")
        manifest = json.loads(self.corpus_manifest.read_text(encoding="utf-8"))
        digest = _sha256(neighbor_path)
        manifest["artifacts"]["neighbor_metadata"] = {
            "path": neighbor_path.relative_to(self.root).as_posix(),
            "sha256": digest,
        }
        hashes = {
            role: str(item["sha256"])
            for role, item in manifest["artifacts"].items()
        }
        generation = retrieval_manager._generation(hashes)
        manifest["bundle_generation"] = generation
        self.corpus_manifest.write_text(
            json.dumps(manifest, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return generation, neighbor_path

    def test_visual_factory_refreshes_after_committed_generation_changes(self) -> None:
        generation_1 = self._publish("g1")
        constructed: list[SimpleNamespace] = []

        def build(_config: object) -> SimpleNamespace:
            engine = SimpleNamespace()
            constructed.append(engine)
            return engine

        with (
            mock.patch.dict(os.environ, self.environment, clear=False),
            mock.patch.object(
                retrieval_manager,
                "VisualSearchEngine",
                side_effect=build,
            ) as factory,
        ):
            first = retrieval_manager.get_visual_search_engine()
            self.assertIs(first, retrieval_manager.get_visual_search_engine())
            generation_2 = self._publish("g2")
            second = retrieval_manager.get_visual_search_engine()
            self.assertIs(second, retrieval_manager.get_visual_search_engine())

        self.assertIsNot(first, second)
        self.assertEqual(first.corpus_generation, generation_1)
        self.assertEqual(second.corpus_generation, generation_2)
        self.assertEqual(factory.call_count, 2)

    def test_publishing_sentinel_blocks_stale_cached_engine(self) -> None:
        self._publish("g1")
        with (
            mock.patch.dict(os.environ, self.environment, clear=False),
            mock.patch.object(
                retrieval_manager,
                "VisualSearchEngine",
                side_effect=lambda _config: SimpleNamespace(),
            ) as factory,
        ):
            first = retrieval_manager.get_visual_search_engine()
            self.corpus_manifest.write_text(
                json.dumps({"schema_version": "1.0", "status": "publishing"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not fully published"):
                retrieval_manager.get_visual_search_engine()
            self.assertEqual(factory.call_count, 1)

            generation_2 = self._publish("g2")
            second = retrieval_manager.get_visual_search_engine()

        self.assertIsNot(first, second)
        self.assertEqual(second.corpus_generation, generation_2)

    def test_generation_flip_during_load_is_not_cached_or_returned(self) -> None:
        self._publish("g1")
        calls = 0
        generation_2: str | None = None

        def build(_config: object) -> SimpleNamespace:
            nonlocal calls, generation_2
            calls += 1
            if calls == 1:
                generation_2 = self._publish("g2")
            return SimpleNamespace()

        with (
            mock.patch.dict(os.environ, self.environment, clear=False),
            mock.patch.object(
                retrieval_manager,
                "VisualSearchEngine",
                side_effect=build,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "corpus changed"):
                retrieval_manager.get_visual_search_engine()
            recovered = retrieval_manager.get_visual_search_engine()

        self.assertEqual(calls, 2)
        self.assertEqual(recovered.corpus_generation, generation_2)

    def test_qa_pipeline_rebuilds_and_rejects_wrong_nested_generation(self) -> None:
        generation_1 = self._publish("g1")
        evidence_1 = SimpleNamespace(corpus_generation=generation_1)
        with (
            mock.patch.dict(os.environ, self.environment, clear=False),
            mock.patch.object(
                retrieval_manager,
                "get_qa_evidence_search_engine",
                return_value=evidence_1,
            ),
        ):
            first = retrieval_manager.get_qa_search_pipeline()

        generation_2 = self._publish("g2")
        evidence_2 = SimpleNamespace(corpus_generation=generation_2)
        with (
            mock.patch.dict(os.environ, self.environment, clear=False),
            mock.patch.object(
                retrieval_manager,
                "get_qa_evidence_search_engine",
                return_value=evidence_2,
            ),
        ):
            second = retrieval_manager.get_qa_search_pipeline()

        self.assertIsNot(first, second)
        self.assertIs(second.evidence_engine, evidence_2)
        self.assertEqual(second.corpus_generation, generation_2)

        retrieval_manager.get_qa_search_pipeline.cache_clear()
        wrong = SimpleNamespace(corpus_generation=generation_1)
        with (
            mock.patch.dict(os.environ, self.environment, clear=False),
            mock.patch.object(
                retrieval_manager,
                "get_qa_evidence_search_engine",
                return_value=wrong,
            ),
            self.assertRaisesRegex(ValueError, "another corpus generation"),
        ):
            retrieval_manager.get_qa_search_pipeline()

    def test_missing_manifest_keeps_legacy_single_entry_cache(self) -> None:
        with (
            mock.patch.dict(os.environ, self.environment, clear=False),
            mock.patch.object(
                retrieval_manager,
                "VisualSearchEngine",
                side_effect=lambda _config: SimpleNamespace(),
            ) as factory,
        ):
            first = retrieval_manager.get_visual_search_engine()
            second = retrieval_manager.get_visual_search_engine()

        self.assertIs(first, second)
        self.assertIsNone(first.corpus_generation)
        factory.assert_called_once()

    def test_legacy_committed_corpus_reports_dense_bundle_as_missing(self) -> None:
        self._publish("legacy-without-dense")

        with (
            mock.patch.dict(os.environ, self.environment, clear=False),
            self.assertRaisesRegex(FileNotFoundError, "dense-candidate bundle"),
        ):
            retrieval_manager.get_dense_candidate_index()

    def test_online_dense_loader_rejects_cross_generation_mix(self) -> None:
        corpus_key = retrieval_manager._CorpusCacheKey(
            manifest_path=str(self.corpus_manifest),
            bundle_generation="expected-generation",
            manifest_contract_sha256="expected-contract",
        )
        wrong = SimpleNamespace(corpus_generation="other-generation")

        with (
            mock.patch.object(
                retrieval_manager,
                "get_dense_candidate_index",
                return_value=wrong,
            ),
            self.assertRaisesRegex(ValueError, "another corpus generation"),
        ):
            retrieval_manager._get_online_dense_index_for_generation(corpus_key)

    def test_missing_optional_context_does_not_block_lazy_qa_or_trake(self) -> None:
        generation = self._publish("without-context")
        missing_neighbors = self.metadata / "missing-neighbors.jsonl"
        missing_segments = self.metadata / "missing-segments.jsonl"
        missing_frame_map = self.metadata / "missing-frame-map.json"
        environment = {
            **self.environment,
            "ONLINE_NEIGHBOR_CONTEXT_ENABLED": "true",
            "ONLINE_SEGMENT_CONTEXT_ENABLED": "true",
            "ONLINE_NEIGHBOR_PATH": str(missing_neighbors),
            "ONLINE_SEGMENT_PATH": str(missing_segments),
        }
        runtime = SimpleNamespace(
            query_expansion=SimpleNamespace(enabled=False),
            hybrid=SimpleNamespace(max_top_k=200),
        )
        qa = SimpleNamespace(
            corpus_generation=generation,
            search=mock.Mock(return_value={"task": "qa"}),
        )
        trake = SimpleNamespace(
            corpus_generation=generation,
            search=mock.Mock(return_value={"task": "trake"}),
        )

        with (
            mock.patch.dict(os.environ, environment, clear=False),
            mock.patch.object(
                retrieval_manager,
                "load_visual_search_config",
                return_value=SimpleNamespace(frame_map_path=missing_frame_map),
            ),
            mock.patch.object(
                retrieval_manager,
                "get_runtime_config",
                return_value=runtime,
            ),
            mock.patch.object(
                retrieval_manager,
                "get_hybrid_search_engine",
                return_value=object(),
            ),
            mock.patch.object(
                retrieval_manager,
                "get_qa_search_pipeline",
                return_value=qa,
            ),
            mock.patch.object(
                retrieval_manager,
                "get_trake_pipeline",
                return_value=trake,
            ),
        ):
            corpus_key = retrieval_manager._current_corpus_cache_key()
            pipeline = retrieval_manager.get_online_pipeline.__wrapped__(corpus_key)
            context_summary = pipeline.context_index.summary()
            self.assertEqual(context_summary["neighbor_record_count"], 0)
            self.assertEqual(context_summary["segment_record_count"], 0)
            self.assertEqual(
                pipeline.qa_pipeline.search("question", top_k=1),
                {"task": "qa"},
            )
            self.assertEqual(
                pipeline.trake_pipeline.search("event", top_k=1),
                {"task": "trake"},
            )

        qa.search.assert_called_once()
        trake.search.assert_called_once()

    def test_committed_but_corrupt_context_fails_closed(self) -> None:
        _generation, neighbor_path = self._publish_with_context(
            "corrupt-context",
            neighbor_payload="{not-json}\n",
        )
        missing_frame_map = self.metadata / "missing-frame-map.json"
        environment = {
            **self.environment,
            "ONLINE_NEIGHBOR_CONTEXT_ENABLED": "true",
            "ONLINE_SEGMENT_CONTEXT_ENABLED": "false",
            "ONLINE_NEIGHBOR_PATH": str(neighbor_path),
        }

        with (
            mock.patch.dict(os.environ, environment, clear=False),
            mock.patch.object(
                retrieval_manager,
                "load_visual_search_config",
                return_value=SimpleNamespace(frame_map_path=missing_frame_map),
            ),
            self.assertRaisesRegex(ValueError, "Invalid JSON"),
        ):
            retrieval_manager.get_online_context_index()

    def test_context_cache_refreshes_after_committed_generation_changes(self) -> None:
        generation_1, neighbor_path = self._publish_with_context(
            "context-g1",
            neighbor_payload=(
                json.dumps(
                    {
                        "video_id": "V1",
                        "frame_id": "F1",
                        "neighbors_before": [],
                        "neighbors_after": [],
                    }
                )
                + "\n"
            ),
        )
        missing_frame_map = self.metadata / "missing-frame-map.json"
        environment = {
            **self.environment,
            "ONLINE_NEIGHBOR_CONTEXT_ENABLED": "true",
            "ONLINE_SEGMENT_CONTEXT_ENABLED": "false",
            "ONLINE_NEIGHBOR_PATH": str(neighbor_path),
        }

        with (
            mock.patch.dict(os.environ, environment, clear=False),
            mock.patch.object(
                retrieval_manager,
                "load_visual_search_config",
                return_value=SimpleNamespace(frame_map_path=missing_frame_map),
            ),
        ):
            first = retrieval_manager.get_online_context_index()
            generation_2, _ = self._publish_with_context(
                "context-g2",
                neighbor_payload=(
                    json.dumps(
                        {
                            "video_id": "V1",
                            "frame_id": "F2",
                            "neighbors_before": [],
                            "neighbors_after": [],
                        }
                    )
                    + "\n"
                ),
            )
            second = retrieval_manager.get_online_context_index()

        self.assertNotEqual(generation_1, generation_2)
        self.assertIsNot(first, second)
        self.assertEqual(
            first.lookup(video_id="V1", frame_id="F1", timestamp=0.0).sources,
            ("neighbors_all",),
        )
        self.assertEqual(
            second.lookup(video_id="V1", frame_id="F1", timestamp=0.0).sources,
            (),
        )
        self.assertEqual(
            second.lookup(video_id="V1", frame_id="F2", timestamp=0.0).sources,
            ("neighbors_all",),
        )

    def test_trake_factory_is_cached_per_committed_corpus_generation(self) -> None:
        generation_1 = self._publish("g1")
        current_generation = generation_1
        constructed: list[SimpleNamespace] = []

        class FakeTrakePipeline:
            def __init__(
                self,
                *,
                retrieval_engine,
                dense_event_engine=None,
                event_reranker=None,
                bge_contract=None,
                local_scorer=None,
                config,
            ) -> None:
                self.retrieval_engine = retrieval_engine
                self.dense_event_engine = dense_event_engine
                self.event_reranker = event_reranker
                self.bge_contract = bge_contract
                self.local_scorer = local_scorer
                self.config = config
                constructed.append(self)

        fake_module = types.ModuleType("backend.app.services.trake.pipeline")
        fake_module.TRAKE_SCHEMA_VERSION = "1.0"
        fake_module.TrakePipeline = FakeTrakePipeline

        def retrieval_engine() -> SimpleNamespace:
            return SimpleNamespace(
                corpus_generation=current_generation,
                visual_engine=SimpleNamespace(
                    encoder=SimpleNamespace(
                        encode=mock.Mock(),
                        encode_images=mock.Mock(),
                    )
                ),
            )

        with (
            mock.patch.dict(os.environ, self.environment, clear=False),
            mock.patch.dict(
                sys.modules,
                {"backend.app.services.trake.pipeline": fake_module},
            ),
            mock.patch.object(
                retrieval_manager,
                "get_hybrid_search_engine",
                side_effect=retrieval_engine,
            ),
        ):
            first = retrieval_manager.get_trake_pipeline()
            self.assertIs(first, retrieval_manager.get_trake_pipeline())
            current_generation = self._publish("g2")
            second = retrieval_manager.get_trake_pipeline()
            self.assertIs(second, retrieval_manager.get_trake_pipeline())

        self.assertIsNot(first, second)
        self.assertEqual(first.corpus_generation, generation_1)
        self.assertEqual(second.corpus_generation, current_generation)
        self.assertIsNotNone(first.local_scorer)
        self.assertIsNotNone(second.local_scorer)
        self.assertEqual(len(constructed), 2)


if __name__ == "__main__":
    unittest.main()
