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
                config,
            ) -> None:
                self.retrieval_engine = retrieval_engine
                self.dense_event_engine = dense_event_engine
                self.event_reranker = event_reranker
                self.bge_contract = bge_contract
                self.config = config
                constructed.append(self)

        fake_module = types.ModuleType("backend.app.services.trake.pipeline")
        fake_module.TrakePipeline = FakeTrakePipeline

        def retrieval_engine() -> SimpleNamespace:
            return SimpleNamespace(corpus_generation=current_generation)

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
        self.assertEqual(len(constructed), 2)


if __name__ == "__main__":
    unittest.main()
