from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "colab_retrieval_v2_launcher.ipynb"


class ColabRetrievalNotebookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = NOTEBOOK_PATH.read_text(encoding="utf-8")
        cls.notebook = json.loads(cls.raw)

    def test_notebook_json_and_code_cells_are_valid(self) -> None:
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertEqual(self.notebook["metadata"]["accelerator"], "GPU")
        for index, cell in enumerate(self.notebook["cells"]):
            if cell["cell_type"] == "code":
                ast.parse(
                    "".join(cell.get("source", [])),
                    filename=f"{NOTEBOOK_PATH}:cell-{index}",
                )

    def test_colab_memory_dependency_and_source_guards_are_present(self) -> None:
        for required in (
            "GIT_BRANCH = 'feat/query-expansion'",
            "CAPTION_BATCH_SIZE = 1",
            "CAPTION_QUANTIZATION = '4bit'",
            "PADDLE_PACKAGE = 'paddlepaddle-gpu==3.3.0'",
            "paddle.device.is_compiled_with_cuda()",
            "GPU_MEMORY_MIB < 14000",
            "source_size_bytes",
            "REPO_ROOT / 'competition' / 'run_retrieval_v2.py'",
        ):
            self.assertIn(required, self.raw)
        self.assertNotIn("Caption Qwen 9B", self.raw)
        self.assertNotIn("GPU_MEMORY_MIB < 30000", self.raw)
        self.assertNotIn("codex/retrieval-leaderboard-v2", self.raw)

    def test_notebook_requires_real_query_expansion_trace(self) -> None:
        for required in (
            "REQUIRE_QUERY_EXPANSION = True",
            "query_traces.jsonl",
            "provider_failures",
            "tkis_with_accepted_paraphrase",
            "rerank_canonical_query",
            "vkis_provider_calls",
            "expansion_limit_failures",
            "expected_expansion_budget",
            "fusion_cap_failures",
            "metadata_evidence_failures",
            "'--caption-batch-size'",
            "'--caption-quantization'",
        ):
            self.assertIn(required, self.raw)


if __name__ == "__main__":
    unittest.main()
