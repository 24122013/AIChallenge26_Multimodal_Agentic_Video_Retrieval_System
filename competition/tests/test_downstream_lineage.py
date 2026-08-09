from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from competition.downstream_lineage import (
    validate_stage_manifest,
    write_stage_manifest,
)


class DownstreamLineageTest(unittest.TestCase):
    def test_round_trip_and_rejects_changed_sources_or_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "segments.jsonl"
            output = root / "text_index.json"
            manifest = root / "text_index_manifest.json"
            source.write_text('{"segment_id":"S0"}\n', encoding="utf-8")
            output.write_text('{"version":2}', encoding="utf-8")
            canonical = [{"video_id": "V0", "phase3_selection_run_id": "RUN0"}]

            write_stage_manifest(
                manifest,
                stage="text-index",
                canonical_sources=canonical,
                input_paths={"segments": source},
                output_paths={"text_index": output},
                config={"text_index_version": 2},
            )
            validated = validate_stage_manifest(
                manifest,
                stage="text-index",
                canonical_sources=canonical,
                input_paths={"segments": source},
                output_paths={"text_index": output},
            )
            self.assertEqual(validated["status"], "passed")

            output.write_text('{"version":2,"changed":true}', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "outputs changed"):
                validate_stage_manifest(
                    manifest,
                    stage="text-index",
                    canonical_sources=canonical,
                    input_paths={"segments": source},
                    output_paths={"text_index": output},
                )

            output.write_text('{"version":2}', encoding="utf-8")
            source.write_text('{"segment_id":"S1"}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "inputs changed"):
                validate_stage_manifest(
                    manifest,
                    stage="text-index",
                    canonical_sources=canonical,
                    input_paths={"segments": source},
                    output_paths={"text_index": output},
                )

            source.write_text('{"segment_id":"S0"}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "canonical keyframes"):
                validate_stage_manifest(
                    manifest,
                    stage="text-index",
                    canonical_sources=[
                        {"video_id": "V0", "phase3_selection_run_id": "RUN1"}
                    ],
                    input_paths={"segments": source},
                    output_paths={"text_index": output},
                )

    def test_missing_manifest_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "segments.jsonl"
            output = root / "text_index.json"
            source.write_text("{}\n", encoding="utf-8")
            output.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "manifest is missing"):
                validate_stage_manifest(
                    root / "missing.json",
                    stage="text-index",
                    canonical_sources=[{"video_id": "V0"}],
                    input_paths={"segments": source},
                    output_paths={"text_index": output},
                )


if __name__ == "__main__":
    unittest.main()
