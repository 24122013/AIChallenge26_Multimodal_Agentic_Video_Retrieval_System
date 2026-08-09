from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from competition.downstream_lineage import validate_stage_manifest
from competition.pipeline import (
    CorpusVideo,
    _require_current_text_index_lineage,
    build_parser,
    competition_index_paths,
    segments_command,
    text_index_command,
)


class Phase4PipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.public_root = self.root / "public"
        self.output_root = self.root / "output"
        self.paths = competition_index_paths(self.output_root)
        self.video = CorpusVideo(
            filename="V0.mp4",
            relative_path=Path("videos/V0.mp4"),
            fps=25.0,
            frame_count=250,
        )
        self.canonical = [
            {
                "video_id": "V0",
                "keyframe_strategy": "multimodal_coverage",
                "phase3_selection_run_id": "RUN0",
            }
        ]

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _args(self, command: str):
        return build_parser().parse_args(
            [
                command,
                "--public-root",
                str(self.public_root),
                "--output-root",
                str(self.output_root),
            ]
        )

    def test_segments_command_commits_manifest_after_output(self) -> None:
        modality = self.output_root / "metadata" / "captions_V0.jsonl"
        modality.parent.mkdir(parents=True, exist_ok=True)
        modality.write_text("{}\n", encoding="utf-8")
        multimodal = {"captions:V0": modality}

        def fake_build(_metadata_dir: Path, output_path: Path, **_kwargs):
            output_path.write_text(
                json.dumps(
                    {
                        "video_id": "V0",
                        "segment_id": "S0",
                        "start_keyframe": "F0",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return {"record_count": 1}

        with (
            mock.patch("competition.pipeline.load_corpus", return_value=[self.video]),
            mock.patch(
                "competition.pipeline._require_multimodal_artifacts",
                return_value=multimodal,
            ),
            mock.patch(
                "competition.pipeline._require_current_canonical_publish",
                return_value=self.canonical,
            ),
            mock.patch(
                "competition.pipeline.build_segment_metadata",
                side_effect=fake_build,
            ),
        ):
            segments_command(self._args("segments"))

        validate_stage_manifest(
            self.paths["segments_manifest"],
            stage="segments",
            canonical_sources=self.canonical,
            input_paths=multimodal,
            output_paths={"segments": self.paths["segments"]},
        )

    def test_text_index_manifest_is_required_and_rejects_tampering(self) -> None:
        self.paths["segments"].parent.mkdir(parents=True, exist_ok=True)
        self.paths["segments"].write_text(
            json.dumps(
                {
                    "video_id": "V0",
                    "segment_id": "S0",
                    "start_keyframe": "F0",
                    "captions_aggregated": "a bicycle",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        with (
            mock.patch("competition.pipeline.load_corpus", return_value=[self.video]),
            mock.patch(
                "competition.pipeline._require_current_segments_lineage",
                return_value=(self.canonical, {}),
            ),
        ):
            text_index_command(self._args("text-index"))

        with mock.patch(
            "competition.pipeline._require_current_segments_lineage",
            return_value=(self.canonical, {}),
        ):
            _require_current_text_index_lineage(
                corpus=[self.video],
                public_root=self.public_root,
                output_root=self.output_root,
            )
            self.paths["text_index"].write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "outputs changed"):
                _require_current_text_index_lineage(
                    corpus=[self.video],
                    public_root=self.public_root,
                    output_root=self.output_root,
                )


if __name__ == "__main__":
    unittest.main()
